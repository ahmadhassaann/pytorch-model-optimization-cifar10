"""Structured channel pruning of ResNet-18 with architecture reconstruction.

For every BasicBlock the output channels of `conv1` are ranked by the L1 norm of
their filters; the weakest are removed. Because conv1's output feeds bn1 and conv2,
we physically slice:
    conv1.weight[out]  ->  bn1.{weight,bias,running_mean,running_var}[out]  ->  conv2.weight[:, in]
and build a NEW, smaller network (no masked zero channels remain). The residual
stream (block outputs) is left intact so skip-connection shapes stay valid.
"""
import argparse
import csv
import os

import torch

from common import (BLOCK_PREFIXES, RESULTS_DIR, ResNet18CIFAR, count_macs, count_params,
                    ensure_dir, evaluate, file_size_mb, get_device, get_loaders, model_path,
                    set_seed, torch_load)
from finetune import finetune


def select_channels(model, ratio, min_channels=8):
    """Indices of conv1 output channels to KEEP in each block (L1-norm importance)."""
    keep = []
    for blk, mid in zip(model.blocks(), model.mid_channels):
        score = blk.conv1.weight.detach().abs().sum(dim=(1, 2, 3))
        n_keep = max(min_channels, int(round(mid * (1.0 - ratio))))
        idx = torch.argsort(score, descending=True)[:n_keep].sort().values
        keep.append(idx)
    return keep


def rebuild_pruned(model, keep):
    """Create a physically smaller ResNet18CIFAR and copy the surviving weights into it."""
    new_mid = [len(i) for i in keep]
    new = ResNet18CIFAR(mid_channels=new_mid)
    old_sd = model.state_dict()
    new_sd = new.state_dict()
    index = dict(zip(BLOCK_PREFIXES, keep))
    bn_params = ("weight", "bias", "running_mean", "running_var")

    for key, tensor in old_sd.items():
        t = tensor
        for prefix, idx in index.items():
            if key == f"{prefix}.conv1.weight":
                t = tensor[idx]                      # drop output filters
            elif key == f"{prefix}.conv2.weight":
                t = tensor[:, idx]                   # drop matching input channels
            elif any(key == f"{prefix}.bn1.{n}" for n in bn_params):
                t = tensor[idx]                      # drop matching BN channels
        assert new_sd[key].shape == t.shape, f"shape mismatch at {key}"
        new_sd[key] = t.clone()
    new.load_state_dict(new_sd)
    return new, new_mid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=model_path("fp32"))
    p.add_argument("--out", default=model_path("structured"))
    p.add_argument("--ratio", type=float, default=0.5, help="fraction of conv1 channels removed per block")
    p.add_argument("--min-channels", type=int, default=8)
    p.add_argument("--ft-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    train_loader, test_loader = get_loaders(args.batch_size, args.workers, args.quick)

    model = ResNet18CIFAR()
    model.load_state_dict(torch_load(args.ckpt))
    model.eval()
    base_acc = evaluate(model, test_loader)
    base_params, _ = count_params(model)
    base_macs = count_macs(model)
    print(f"Original: acc {base_acc:.2f}% | params {base_params:,} | MACs {base_macs:,}")

    keep = select_channels(model, args.ratio, args.min_channels)
    pruned, new_mid = rebuild_pruned(model, keep)
    pruned.eval()

    # sanity: the reconstructed network must run with valid tensor dimensions
    with torch.no_grad():
        assert pruned(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    pr_params, _ = count_params(pruned)
    pr_macs = count_macs(pruned)
    acc_pruned = evaluate(pruned, test_loader)
    print(f"Pruned widths (conv1 per block): {new_mid}")
    print(f"After pruning (no fine-tune): acc {acc_pruned:.2f}% | params {pr_params:,} "
          f"({100 * pr_params / base_params:.1f}%) | MACs {pr_macs:,} ({100 * pr_macs / base_macs:.1f}%)")

    print("Fine-tuning ...")
    acc = finetune(pruned, train_loader, test_loader, args.ft_epochs, args.lr, device, tag="ft")
    pruned.cpu().eval()
    acc = evaluate(pruned, test_loader)
    print(f"After fine-tuning: acc {acc:.2f}%")

    ensure_dir(os.path.dirname(args.out))
    torch.save({"mid_channels": new_mid, "state_dict": pruned.state_dict()}, args.out)
    ensure_dir(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, "structured_pruning_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "acc", "params", "macs"])
        w.writerow(["original", f"{base_acc:.2f}", base_params, base_macs])
        w.writerow(["pruned_no_finetune", f"{acc_pruned:.2f}", pr_params, pr_macs])
        w.writerow(["pruned_finetuned", f"{acc:.2f}", pr_params, pr_macs])
    print(f"Saved structured-pruned model -> {args.out} ({file_size_mb(args.out):.2f} MB)")


if __name__ == "__main__":
    main()
