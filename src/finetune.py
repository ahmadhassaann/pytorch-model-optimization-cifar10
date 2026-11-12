"""Fine-tuning helper used by both pruning pipelines (also usable as a CLI for the
structured-pruned model)."""
import argparse

import torch

from common import (ResNet18CIFAR, evaluate, get_device, get_loaders, model_path, set_seed,
                    torch_load, train_one_epoch)


def finetune(model, train_loader, test_loader, epochs, lr, device, weight_decay=5e-4, tag="finetune"):
    """SGD + cosine fine-tuning. Returns the final test accuracy (%)."""
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs * len(train_loader)))
    acc = evaluate(model, test_loader, device)
    for ep in range(1, epochs + 1):
        loss, tr_acc = train_one_epoch(model, train_loader, opt, device, sched, f"{tag} {ep}")
        acc = evaluate(model, test_loader, device)
        print(f"  [{tag} {ep}/{epochs}] loss {loss:.4f} | train {tr_acc:.2f}% | test {acc:.2f}%")
    return acc


def main():
    p = argparse.ArgumentParser(description="Further fine-tune the structured-pruned checkpoint.")
    p.add_argument("--ckpt", default=model_path("structured"))
    p.add_argument("--out", default=model_path("structured"))
    p.add_argument("--epochs", type=int, default=5)
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
    ckpt = torch_load(args.ckpt)
    model = ResNet18CIFAR(mid_channels=ckpt["mid_channels"])
    model.load_state_dict(ckpt["state_dict"])
    acc = finetune(model, train_loader, test_loader, args.epochs, args.lr, device)
    torch.save({"mid_channels": ckpt["mid_channels"], "state_dict": model.cpu().state_dict()}, args.out)
    print(f"Saved -> {args.out} (test acc {acc:.2f}%)")


if __name__ == "__main__":
    main()
