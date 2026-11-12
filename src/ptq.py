"""Post-Training Static Quantization (PTQ) of the FP32 ResNet-18 -> INT8.

Steps: load FP32 -> fuse Conv+BN+ReLU -> set backend/qconfig -> insert observers
-> calibrate on a subset of the TRAIN set -> convert to INT8 -> evaluate -> save.
"""
import argparse
import os

import torch
from torch.ao.quantization import convert, prepare

from common import (ResNet18CIFAR, ensure_dir, evaluate, file_size_mb, get_calib_loader,
                    get_loaders, model_path, ptq_qconfig, save_torchscript, set_backend,
                    set_seed, torch_load, torchscript_path)


@torch.no_grad()
def calibrate(model, loader):
    """Run representative data through the observed model so observers collect ranges."""
    model.eval()
    for x, _ in loader:
        model(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=model_path("fp32"))
    p.add_argument("--out", default=model_path("ptq"))
    p.add_argument("--calib-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    backend = set_backend()                      # fbgemm (x86) or qnnpack (ARM)
    print(f"Quantization backend: {backend}")
    _, test_loader = get_loaders(args.batch_size, args.workers, args.quick)
    calib_loader = get_calib_loader(args.calib_samples, args.batch_size, args.workers,
                                    args.quick, args.seed)

    model = ResNet18CIFAR()
    model.load_state_dict(torch_load(args.ckpt))
    model.eval()
    fp32_acc = evaluate(model, test_loader)
    print(f"FP32 accuracy: {fp32_acc:.2f}%")

    model.fuse(qat=False)                        # 1. fuse Conv+BN(+ReLU)
    model.qconfig = ptq_qconfig(backend)         # 2. configure backend qconfig
    prepare(model, inplace=True)                 # 3. insert observers
    calibrate(model, calib_loader)               # 4. calibration
    int8 = convert(model, inplace=False)         # 5. FP32 -> INT8
    int8.eval()

    acc = evaluate(int8, test_loader)
    print(f"PTQ INT8 accuracy: {acc:.2f}% (drop {fp32_acc - acc:+.2f} pts)")

    ensure_dir(os.path.dirname(args.out))
    torch.save(int8.state_dict(), args.out)
    ts_ok = save_torchscript(int8, torchscript_path(args.out))
    print(f"Saved PTQ INT8 model -> {args.out} ({file_size_mb(args.out):.2f} MB)")
    if ts_ok:
        print(f"Saved TorchScript    -> {torchscript_path(args.out)}")


if __name__ == "__main__":
    main()
