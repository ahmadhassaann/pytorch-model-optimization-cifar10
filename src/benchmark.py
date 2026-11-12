"""Automated benchmark of all five models on the CIFAR-10 test set (CPU).

Outputs (in results/):
    benchmark_results.csv  - one row per model: accuracy, latency, throughput, size, sparsity, ...
    predictions.csv        - per-sample true label + prediction/confidence of every model
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (CLASSES, MODEL_KINDS, RESULTS_DIR, count_macs, count_params, ensure_dir,
                    file_size_mb, get_test_set, load_model, model_path, set_seed, weight_sparsity)


@torch.no_grad()
def run_dataset(model, loader):
    """Predict the whole test set. Returns labels, preds, confidences, elapsed seconds."""
    labels, preds, confs = [], [], []
    t0 = time.perf_counter()
    for x, y in loader:
        probs = F.softmax(model(x), dim=1)
        conf, pred = probs.max(1)
        labels.append(y)
        preds.append(pred)
        confs.append(conf)
    elapsed = time.perf_counter() - t0
    return (torch.cat(labels).numpy(), torch.cat(preds).numpy(),
            torch.cat(confs).numpy(), elapsed)


@torch.no_grad()
def single_image_latency(model, dataset, n=200, warmup=10, threads=1):
    """Batch-size-1 latency (ms): mean and p95 over n test images, using `threads` CPU threads."""
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    times = []
    for i in range(warmup + n):
        x = dataset[i % len(dataset)][0].unsqueeze(0)
        t0 = time.perf_counter()
        model(x)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(prev)
    return float(np.mean(times)), float(np.percentile(times, 95))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=MODEL_KINDS, choices=MODEL_KINDS)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latency-samples", type=int, default=200)
    p.add_argument("--latency-threads", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    dataset = get_test_set(args.quick)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    n_lat = 20 if args.quick else args.latency_samples

    rows, pred_cols = [], {}
    labels = None
    for kind in args.models:
        path = model_path(kind)
        if not os.path.exists(path):
            print(f"[skip] {kind}: {path} not found")
            continue
        print(f"Benchmarking {kind} ...")
        model = load_model(kind)
        labels, preds, confs, elapsed = run_dataset(model, loader)
        lat_mean, lat_p95 = single_image_latency(model, dataset, n_lat, threads=args.latency_threads)

        is_int8 = kind in ("ptq", "qat")
        if is_int8:
            total = nonzero = macs = sparsity = None
        else:
            total, nonzero = count_params(model)
            macs = count_macs(model)
            sparsity = round(weight_sparsity(model), 2)
        rows.append({
            "model": kind,
            "checkpoint": os.path.basename(path),
            "precision": "INT8" if is_int8 else "FP32",
            "accuracy_pct": round(100.0 * float((preds == labels).mean()), 2),
            "latency_ms_bs1_mean": round(lat_mean, 3),
            "latency_ms_bs1_p95": round(lat_p95, 3),
            "throughput_img_per_s": round(len(labels) / elapsed, 1),
            "file_size_mb": round(file_size_mb(path), 3),
            "params_total": total,
            "params_nonzero": nonzero,
            "weight_sparsity_pct": sparsity,
            "macs": macs,
            "test_samples": len(labels),
        })
        pred_cols[f"{kind}_pred"] = [CLASSES[i] for i in preds]
        pred_cols[f"{kind}_conf"] = np.round(confs, 4)
        print(f"   acc {rows[-1]['accuracy_pct']:.2f}% | {lat_mean:.2f} ms/img (bs=1) | "
              f"{rows[-1]['throughput_img_per_s']} img/s | {rows[-1]['file_size_mb']} MB")

    if not rows:
        raise SystemExit("No model checkpoints found - run the pipeline first (see README).")

    ensure_dir(RESULTS_DIR)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "benchmark_results.csv"), index=False)
    preds_df = pd.DataFrame({"index": np.arange(len(labels)),
                             "true_label": [CLASSES[i] for i in labels], **pred_cols})
    preds_df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)
    print("\n" + summary[["model", "accuracy_pct", "latency_ms_bs1_mean", "throughput_img_per_s",
                          "file_size_mb", "weight_sparsity_pct"]].to_string(index=False))
    print(f"\nSaved results/benchmark_results.csv and results/predictions.csv")


if __name__ == "__main__":
    main()
