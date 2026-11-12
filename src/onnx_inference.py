"""CPU inference + benchmark of the exported ONNX models with ONNX Runtime.

    python src/onnx_inference.py                      # benchmark every models/*.onnx
    python src/onnx_inference.py --image cat.png      # classify one image with every ONNX model
Writes results/onnx_benchmark.csv.
"""
import argparse
import glob
import os
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

from common import (CLASSES, MODELS_DIR, RESULTS_DIR, ensure_dir, file_size_mb, get_test_set,
                    get_test_transform)


def make_session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def classify_image(sess, image_path):
    img = Image.open(image_path).convert("RGB").resize((32, 32), Image.BILINEAR)
    x = get_test_transform()(img).unsqueeze(0).numpy()
    probs = softmax(sess.run(None, {"input": x})[0])[0]
    top = probs.argsort()[::-1][:3]
    return [(CLASSES[i], float(probs[i])) for i in top]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image")
    p.add_argument("--num-samples", type=int, default=2000, help="test images used for accuracy")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--latency-samples", type=int, default=200)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(MODELS_DIR, "*.onnx")))
    if not paths:
        raise SystemExit("No .onnx files in models/ - run src/export_onnx.py first.")

    if args.image:
        for path in paths:
            top = classify_image(make_session(path, args.threads), args.image)
            print(f"{os.path.basename(path)}: " + ", ".join(f"{n} {100 * c:.1f}%" for n, c in top))
        return

    n = 500 if args.quick else args.num_samples
    dataset = get_test_set()
    xs = np.stack([dataset[i][0].numpy() for i in range(n)])
    ys = np.array([dataset[i][1] for i in range(n)])
    n_lat = 20 if args.quick else args.latency_samples

    rows = []
    for path in paths:
        name = os.path.basename(path)
        print(f"Benchmarking {name} ...")
        sess = make_session(path, args.threads)
        preds = []
        t0 = time.perf_counter()
        for i in range(0, n, args.batch_size):
            preds.append(sess.run(None, {"input": xs[i:i + args.batch_size]})[0].argmax(1))
        elapsed = time.perf_counter() - t0
        preds = np.concatenate(preds)
        times = []
        for i in range(10 + n_lat):
            t1 = time.perf_counter()
            sess.run(None, {"input": xs[i % n][None]})
            if i >= 10:
                times.append((time.perf_counter() - t1) * 1000)
        rows.append({"onnx_model": name, "accuracy_pct": round(100.0 * float((preds == ys).mean()), 2),
                     "latency_ms_bs1_mean": round(float(np.mean(times)), 3),
                     "latency_ms_bs1_p95": round(float(np.percentile(times, 95)), 3),
                     "throughput_img_per_s": round(n / elapsed, 1),
                     "file_size_mb": round(file_size_mb(path), 3), "test_samples": n,
                     "provider": "CPUExecutionProvider", "threads": args.threads})
        print(f"   acc {rows[-1]['accuracy_pct']:.2f}% | {rows[-1]['latency_ms_bs1_mean']:.2f} ms/img")

    ensure_dir(RESULTS_DIR)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "onnx_benchmark.csv"), index=False)
    print("\n" + df.to_string(index=False))
    print("\nSaved results/onnx_benchmark.csv")


if __name__ == "__main__":
    main()
