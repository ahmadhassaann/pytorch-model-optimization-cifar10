"""Shared utilities for Assignment 03 (Edge Computing & Intelligent Systems).

Contains: paths, data loading, the CIFAR ResNet-18 (quantization friendly and
with configurable internal widths for structured pruning), training/evaluation
helpers, quantization helpers and model loading for all five model variants.
"""
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import (DeQuantStub, QuantStub, convert,
                                   get_default_qat_qconfig, get_default_qconfig,
                                   prepare, prepare_qat)
from torch.ao.quantization.fuse_modules import fuse_modules, fuse_modules_qat
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

MODEL_FILES = {
    "fp32": "resnet18_fp32.pth",
    "ptq": "resnet18_ptq_int8.pth",
    "qat": "resnet18_qat_int8.pth",
    "unstructured": "resnet18_unstructured.pth",
    "structured": "resnet18_structured.pth",
}
MODEL_KINDS = list(MODEL_FILES)


def model_path(kind):
    return os.path.join(MODELS_DIR, MODEL_FILES[kind])


def torchscript_path(pth_path):
    """TorchScript twin of an INT8 .pth checkpoint (e.g. x_int8.pth -> x_int8_ts.pt)."""
    return pth_path[:-4] + "_ts.pt"


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def torch_load(path):
    """Load our own checkpoints (INT8 state dicts contain non-tensor objects)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # very old torch without weights_only
        return torch.load(path, map_location="cpu")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _transforms():
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    return train_tf, test_tf


def get_test_transform():
    return _transforms()[1]


def get_test_set(quick=False):
    ds = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                      transform=get_test_transform())
    return Subset(ds, range(1000)) if quick else ds


def get_loaders(batch_size=128, workers=2, quick=False):
    """Return (train_loader, test_loader). quick=True uses small subsets (smoke test)."""
    train_tf, test_tf = _transforms()
    train_set = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    if quick:
        train_set = Subset(train_set, range(2048))
        test_set = Subset(test_set, range(1000))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, test_loader


def get_calib_loader(num_samples=2000, batch_size=128, workers=2, quick=False, seed=42):
    """Random subset of the TRAIN set (no augmentation) used for PTQ calibration."""
    _, test_tf = _transforms()
    ds = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=test_tf)
    if quick:
        num_samples = min(num_samples, 256)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=num_samples, replace=False).tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=workers)


# --------------------------------------------------------------------------- #
# Model: CIFAR ResNet-18 (eager-mode quantization friendly)
# --------------------------------------------------------------------------- #
DEFAULT_MID = [64, 64, 128, 128, 256, 256, 512, 512]  # conv1 width inside each BasicBlock
_BLOCK_CFG = [(64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1),
              (128, 256, 2), (256, 256, 1), (256, 512, 2), (512, 512, 1)]  # (in, out, stride)
BLOCK_PREFIXES = [f"layer{l}.{b}" for l in range(1, 5) for b in range(2)]


class BasicBlock(nn.Module):
    """conv-bn-relu-conv-bn (+shortcut) -> relu.

    `mid_ch` (output channels of conv1 / input channels of conv2) can differ from
    `out_ch`; this is what structured pruning shrinks, leaving the residual stream
    untouched so the skip-connection shapes always stay valid.
    """

    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                                          nn.BatchNorm2d(out_ch))
        else:
            self.shortcut = nn.Identity()
        self.skip_add = FloatFunctional()  # quantization-safe residual add (+ReLU)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.skip_add.add_relu(out, self.shortcut(x))


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10, mid_channels=None):
        super().__init__()
        mid = list(mid_channels) if mid_channels is not None else list(DEFAULT_MID)
        assert len(mid) == 8, "mid_channels needs one entry per BasicBlock (8)"
        self.mid_channels = mid
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        blocks = [BasicBlock(i, m, o, s) for (i, o, s), m in zip(_BLOCK_CFG, mid)]
        self.layer1 = nn.Sequential(*blocks[0:2])
        self.layer2 = nn.Sequential(*blocks[2:4])
        self.layer3 = nn.Sequential(*blocks[4:6])
        self.layer4 = nn.Sequential(*blocks[6:8])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        return self.dequant(x)

    def blocks(self):
        return [self.get_submodule(p) for p in BLOCK_PREFIXES]

    def fuse(self, qat=False):
        """Fuse Conv+BN(+ReLU). Use qat=False in eval mode (PTQ), qat=True in train mode (QAT)."""
        groups = [["conv1", "bn1", "relu"]]
        for p, blk in zip(BLOCK_PREFIXES, self.blocks()):
            groups.append([f"{p}.conv1", f"{p}.bn1", f"{p}.relu"])
            groups.append([f"{p}.conv2", f"{p}.bn2"])
            if isinstance(blk.shortcut, nn.Sequential):
                groups.append([f"{p}.shortcut.0", f"{p}.shortcut.1"])
        (fuse_modules_qat if qat else fuse_modules)(self, groups, inplace=True)
        return self


# --------------------------------------------------------------------------- #
# Training / evaluation helpers
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, device, scheduler=None, desc="train"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """Top-1 accuracy (%)."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def count_params(model):
    """(total, non-zero) parameter counts of a float model."""
    total = nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += int(torch.count_nonzero(p).item())
    return total, nonzero


def weight_sparsity(model):
    """Sparsity (%) over Conv2d/Linear weights."""
    total = zeros = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total += m.weight.numel()
            zeros += int((m.weight == 0).sum().item())
    return 100.0 * zeros / max(total, 1)


@torch.no_grad()
def count_macs(model, input_size=(1, 3, 32, 32)):
    """Multiply-accumulate count of Conv2d + Linear layers (float models only)."""
    macs = [0]

    def conv_hook(m, inp, out):
        macs[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def fc_hook(m, inp, out):
        macs[0] += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(fc_hook))
    try:
        model.eval()
        model(torch.zeros(*input_size))
    except Exception:
        macs[0] = None
    finally:
        for h in hooks:
            h.remove()
    return macs[0]


# --------------------------------------------------------------------------- #
# Quantization helpers
# --------------------------------------------------------------------------- #
def set_backend():
    """Select the quantized engine: fbgemm (x86) if available, else qnnpack (ARM)."""
    engines = torch.backends.quantized.supported_engines
    backend = "fbgemm" if "fbgemm" in engines else "qnnpack"
    torch.backends.quantized.engine = backend
    return backend


def ptq_qconfig(backend):
    return get_default_qconfig(backend)


def qat_qconfig(backend):
    return get_default_qat_qconfig(backend)


def build_int8_skeleton(qat):
    """Architecture-only INT8 model (used to load a state dict when TorchScript is unavailable)."""
    backend = set_backend()
    m = ResNet18CIFAR()
    if qat:
        m.train()
        m.fuse(qat=True)
        m.qconfig = qat_qconfig(backend)
        prepare_qat(m, inplace=True)
    else:
        m.eval()
        m.fuse(qat=False)
        m.qconfig = ptq_qconfig(backend)
        prepare(m, inplace=True)
    m.eval()
    return convert(m, inplace=False)


def save_torchscript(model, path, example=None):
    """Save an executable TorchScript copy (loadable without this source code)."""
    model.eval()
    example = example if example is not None else torch.randn(1, 3, 32, 32)
    try:
        try:
            ts = torch.jit.trace(model, example)
        except Exception:
            ts = torch.jit.script(model)
        ts.save(path)
        return True
    except Exception as e:  # not fatal: the .pth state dict is still saved
        print(f"[warn] could not save TorchScript model ({e})")
        return False


def load_model(kind):
    """Load any of: fp32 | ptq | qat | unstructured | structured. Returns an eval-mode CPU model."""
    path = model_path(kind)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run the pipeline first (see README)")
    if kind in ("ptq", "qat"):
        set_backend()
        ts_path = torchscript_path(path)
        if os.path.exists(ts_path):
            try:
                m = torch.jit.load(ts_path, map_location="cpu")
                return m.eval()
            except Exception as e:
                print(f"[warn] TorchScript load failed ({e}); rebuilding from state dict")
        m = build_int8_skeleton(qat=(kind == "qat"))
        m.load_state_dict(torch_load(path))
        return m.eval()
    ckpt = torch_load(path)
    if kind == "structured":
        m = ResNet18CIFAR(mid_channels=ckpt["mid_channels"])
        m.load_state_dict(ckpt["state_dict"])
    else:
        m = ResNet18CIFAR()
        m.load_state_dict(ckpt)
    return m.eval()
