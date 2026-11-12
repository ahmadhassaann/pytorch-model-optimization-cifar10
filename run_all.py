"""One-command, reproducible pipeline (all steps, in order).

    python run_all.py            # full run (GPU recommended)
    python run_all.py --quick    # tiny smoke test (~minutes on CPU) to verify the setup
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--baseline-epochs", type=int, default=30)
    p.add_argument("--qat-epochs", type=int, default=5)
    p.add_argument("--final-ft-epochs", type=int, default=5)
    p.add_argument("--structured-ft-epochs", type=int, default=10)
    args = p.parse_args()

    q = ["--quick"] if args.quick else []
    if args.quick:
        args.baseline_epochs = args.qat_epochs = 1
        args.final_ft_epochs = args.structured_ft_epochs = 1

    steps = [
        ["train_baseline.py", "--epochs", str(args.baseline_epochs)] + q,
        ["ptq.py"] + q,
        ["qat.py", "--epochs", str(args.qat_epochs)] + q,
        ["prune_unstructured.py", "--final-ft-epochs", str(args.final_ft_epochs)] + q,
        ["prune_structured.py", "--ft-epochs", str(args.structured_ft_epochs)] + q,
        ["benchmark.py"] + q,
        ["export_onnx.py"] + q,
        ["onnx_inference.py"] + q,
        ["inference.py", "--model", "all", "--index", "0"],
    ]
    for step in steps:
        cmd = [sys.executable, os.path.join(ROOT, "src", step[0])] + step[1:]
        print("\n" + "=" * 78 + f"\n>>> {' '.join(step)}\n" + "=" * 78, flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    print("\nPipeline finished. See models/ and results/.")


if __name__ == "__main__":
    main()
