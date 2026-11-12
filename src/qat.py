"""Quantization-Aware Training (QAT) starting from the pretrained FP32 ResNet-18.

Steps: load FP32 -> fuse (QAT mode) -> qconfig -> insert fake-quant modules
-> fine-tune -> convert to INT8 -> evaluate -> save (INT8 + fake-quant checkpoint).
"""
import argparse
import copy
import math
import os

import torch
from torch.ao.nn.intrinsic.qat import freeze_bn_stats
from torch.ao.quantization import convert, disable_observer, prepare_qat

from common import (ResNet18CIFAR, ensure_dir, evaluate, file_size_mb, get_device, get_loaders,
                    model_path, qat_qconfig, save_torchscript, set_backend, set_seed,
                    torch_load, torchscript_path, train_one_epoch)

FAKEQUANT_FILE = "resnet18_qat_fakequant.pth"  # pre-convert model, used for the ONNX (QDQ) export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=model_path("fp32"))
    p.add_argument("--out", default=model_path("qat"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    backend = set_backend()
    device = get_device(args.device)
    print(f"Backend: {backend} | training device: {device}")
    train_loader, test_loader = get_loaders(args.batch_size, args.workers, args.quick)

    # 1. initialise from the pretrained FP32 model
    model = ResNet18CIFAR()
    model.load_state_dict(torch_load(args.ckpt))
    fp32_acc = evaluate(model, test_loader)
    print(f"FP32 accuracy: {fp32_acc:.2f}%")

    # 2. fuse + insert fake-quantization modules
    model.train()
    model.fuse(qat=True)
    model.qconfig = qat_qconfig(backend)
    prepare_qat(model, inplace=True)
    model.to(device)

    # 3. fine-tune with fake quantization
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs * len(train_loader)))
    obs_freeze = math.ceil(0.6 * args.epochs)   # stop updating observers (ranges)
    bn_freeze = math.ceil(0.8 * args.epochs)    # freeze BN statistics
    for ep in range(args.epochs):
        if ep >= obs_freeze:
            model.apply(disable_observer)
        if ep >= bn_freeze:
            model.apply(freeze_bn_stats)
        loss, tr_acc = train_one_epoch(model, train_loader, opt, device, sched, f"QAT {ep + 1}")
        acc = evaluate(model, test_loader, device)
        print(f"[{ep + 1:02d}/{args.epochs}] loss {loss:.4f} | train {tr_acc:.2f}% | fake-quant test {acc:.2f}%")

    # 4. convert to a real INT8 model (CPU)
    model.cpu().eval()
    ensure_dir(os.path.dirname(args.out))
    fq_path = os.path.join(os.path.dirname(args.out), FAKEQUANT_FILE)
    torch.save(model.state_dict(), fq_path)      # keep the fake-quant weights for ONNX export
    int8 = convert(copy.deepcopy(model), inplace=False).eval()

    acc = evaluate(int8, test_loader)
    print(f"QAT INT8 accuracy: {acc:.2f}% (FP32 {fp32_acc:.2f}%)")

    # 5. export
    torch.save(int8.state_dict(), args.out)
    ts_ok = save_torchscript(int8, torchscript_path(args.out))
    print(f"Saved QAT INT8 model -> {args.out} ({file_size_mb(args.out):.2f} MB)")
    if ts_ok:
        print(f"Saved TorchScript    -> {torchscript_path(args.out)}")
    print(f"Saved fake-quant model -> {fq_path}")


if __name__ == "__main__":
    main()
