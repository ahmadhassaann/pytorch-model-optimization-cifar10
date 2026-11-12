# Technical Report - Assignment 03: Model Compression for Edge Deployment

**Name:** ______  **Roll No:** ______  **Section:** ______

> Fill in the tables with the numbers from `results/*.csv` and paste screenshots of the terminal output.

## 1. Setup
- Dataset: CIFAR-10 (50k train / 10k test), normalization with per-channel mean/std, crop + flip augmentation.
- Model: ResNet-18 adapted to 32x32 inputs (3x3 stem, no max-pool), ~11.2M parameters.
- Hardware / software: CPU/GPU ______, PyTorch ______, ONNX Runtime ______, backend `fbgemm`/`qnnpack`.

## 2. Baseline FP32
Training recipe (SGD, nesterov, OneCycle, 30 epochs, wd 5e-4). Final test accuracy: ____ %.
Screenshot: `train_baseline.py` output / `results/baseline_train_log.csv`.

## 3. Post-Training Quantization (Q1)
Layer fusion (Conv+BN+ReLU), qconfig, calibration on 2000 training images, `convert`. Accuracy: ____ %, size ____ MB.

## 4. Quantization-Aware Training (Q1)
Fake-quant insertion, 5 epochs of fine-tuning (observers frozen after 60%, BN stats after 80%). Accuracy: ____ %, size ____ MB.

## 5. Unstructured Pruning (Q2)
Global L1 pruning, 5 iterations to 80% sparsity, fine-tuning, `prune.remove`.

| Iteration | Sparsity % | Acc after pruning | Acc after fine-tune |
|---|---|---|---|
| 1 | | | |

## 6. Structured Pruning (Q2)
L1-norm channel ranking of block `conv1` filters, slicing conv1/bn1/conv2, rebuilt network.
Widths: ______. Parameters ____ -> ____, MACs ____ -> ____, accuracy ____ %.

## 7. Benchmark (from `results/benchmark_results.csv`)

| Model | Accuracy % | Latency ms (bs=1) | Throughput img/s | Size MB | Sparsity % | MACs |
|---|---|---|---|---|---|---|
| fp32 | | | | | | |
| ptq | | | | | | |
| qat | | | | | | |
| unstructured | | | | | | |
| structured | | | | | | |

## 8. ONNX deployment (from `results/onnx_benchmark.csv`)
Exported models, ONNX Runtime CPU results, and notes on INT8 (QDQ) export.

## 9. Discussion
Accuracy vs. size vs. latency trade-offs; why unstructured sparsity does not speed up dense CPU inference; PTQ vs QAT.

## 10. Screenshots
(Paste screenshots of each script run and of the CSV outputs.)
