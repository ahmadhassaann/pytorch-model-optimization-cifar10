"""Train the baseline FP32 ResNet-18 on CIFAR-10 and save its checkpoint."""
import argparse
import csv
import os
import time

import torch

from common import (RESULTS_DIR, ResNet18CIFAR, ensure_dir, evaluate, get_device,
                    get_loaders, model_path, set_seed, train_one_epoch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--out", default=model_path("fp32"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--quick", action="store_true", help="tiny subset smoke test")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")
    train_loader, test_loader = get_loaders(args.batch_size, args.workers, args.quick)

    model = ResNet18CIFAR().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.15)

    ensure_dir(RESULTS_DIR)
    log_path = os.path.join(RESULTS_DIR, "baseline_train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_acc", "seconds"])
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            loss, tr_acc = train_one_epoch(model, train_loader, opt, device, sched, f"epoch {ep}")
            acc = evaluate(model, test_loader, device)
            sec = time.time() - t0
            print(f"[{ep:02d}/{args.epochs}] loss {loss:.4f} | train {tr_acc:.2f}% | test {acc:.2f}% | {sec:.0f}s")
            writer.writerow([ep, f"{loss:.4f}", f"{tr_acc:.2f}", f"{acc:.2f}", f"{sec:.0f}"])

    ensure_dir(os.path.dirname(args.out))
    torch.save(model.cpu().state_dict(), args.out)
    print(f"Saved FP32 checkpoint -> {args.out}\nFinal test accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
