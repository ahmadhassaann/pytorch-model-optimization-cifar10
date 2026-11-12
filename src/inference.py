"""Inference pipeline for all five models (fp32, ptq, qat, unstructured, structured).

Examples:
    python src/inference.py --model fp32 --image cat.png
    python src/inference.py --model all --index 7          # CIFAR-10 test image #7
"""
import argparse
import time

import torch
import torch.nn.functional as F
from PIL import Image

from common import CLASSES, MODEL_KINDS, get_test_set, get_test_transform, load_model


def load_input(args):
    """Return (tensor[1,3,32,32], true_label_or_None)."""
    if args.image:
        img = Image.open(args.image).convert("RGB").resize((32, 32), Image.BILINEAR)
        return get_test_transform()(img).unsqueeze(0), None
    x, y = get_test_set()[args.index]
    return x.unsqueeze(0), y


@torch.no_grad()
def predict(model, x, topk=3):
    t0 = time.perf_counter()
    logits = model(x)
    ms = (time.perf_counter() - t0) * 1000
    probs = F.softmax(logits, dim=1)[0]
    conf, idx = probs.topk(topk)
    return [(CLASSES[i], c.item()) for i, c in zip(idx.tolist(), conf)], ms


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="fp32", choices=MODEL_KINDS + ["all"])
    p.add_argument("--image", help="path to an image file (resized to 32x32)")
    p.add_argument("--index", type=int, default=0, help="CIFAR-10 test index (if no --image)")
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    x, label = load_input(args)
    if label is not None:
        print(f"Input: CIFAR-10 test image #{args.index} (true class: {CLASSES[label]})")
    else:
        print(f"Input: {args.image}")

    kinds = MODEL_KINDS if args.model == "all" else [args.model]
    for kind in kinds:
        try:
            model = load_model(kind)
        except FileNotFoundError as e:
            print(f"\n[{kind}] skipped: {e}")
            continue
        model(x)  # warm-up
        top, ms = predict(model, x, args.topk)
        print(f"\n[{kind}]  latency {ms:.2f} ms")
        for name, conf in top:
            print(f"   {name:<12s} {100 * conf:6.2f}%")


if __name__ == "__main__":
    main()
