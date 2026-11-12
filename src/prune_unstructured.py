"""Unstructured (magnitude / L1) iterative pruning with masking, fine-tuning and
permanent mask removal.

Global magnitude pruning over every Conv2d and Linear weight. Each of N iterations
prunes the same fraction of the *remaining* weights so that the target sparsity is
reached geometrically; a short fine-tune follows each iteration.
"""
import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from common import (RESULTS_DIR, ResNet18CIFAR, count_params, ensure_dir, evaluate, file_size_mb,
                    get_device, get_loaders, model_path, set_seed, torch_load, weight_sparsity)
from finetune import finetune


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=model_path("fp32"))
    p.add_argument("--out", default=model_path("unstructured"))
    p.add_argument("--sparsity", type=float, default=0.8, help="final global sparsity (0-1)")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--ft-epochs-per-iter", type=int, default=1)
    p.add_argument("--final-ft-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
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
    base_acc = evaluate(model.to(device), test_loader, device)
    print(f"Baseline FP32 accuracy: {base_acc:.2f}%")

    targets = [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    per_iter = 1.0 - (1.0 - args.sparsity) ** (1.0 / args.iterations)

    ensure_dir(RESULTS_DIR)
    log_path = os.path.join(RESULTS_DIR, "unstructured_pruning_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "sparsity_pct", "acc_after_pruning", "acc_after_finetune"])
        for it in range(1, args.iterations + 1):
            # prune a fraction of the *remaining* weights (mask is accumulated per layer)
            prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=per_iter)
            sp = weight_sparsity(model)
            acc_pruned = evaluate(model, test_loader, device)
            print(f"Iteration {it}/{args.iterations}: sparsity {sp:.1f}% | acc after pruning {acc_pruned:.2f}%")
            acc_ft = finetune(model, train_loader, test_loader, args.ft_epochs_per_iter,
                              args.lr, device, tag=f"iter{it}")
            writer.writerow([it, f"{sp:.2f}", f"{acc_pruned:.2f}", f"{acc_ft:.2f}"])

    # final (longer) fine-tune while the masks are still applied
    print("Final fine-tuning ...")
    acc = finetune(model, train_loader, test_loader, args.final_ft_epochs, args.lr, device, tag="final")

    # make pruning permanent: weight = weight_orig * mask, then drop reparametrisation
    for module, name in targets:
        prune.remove(module, name)
    model.cpu().eval()
    final_sp = weight_sparsity(model)
    total, nonzero = count_params(model)
    acc = evaluate(model, test_loader)
    print(f"Permanent pruning done: sparsity {final_sp:.1f}% | non-zero params {nonzero:,}/{total:,} "
          f"| accuracy {acc:.2f}%")

    ensure_dir(os.path.dirname(args.out))
    torch.save(model.state_dict(), args.out)
    print(f"Saved unstructured-pruned model -> {args.out} ({file_size_mb(args.out):.2f} MB, dense storage)")


if __name__ == "__main__":
    main()
