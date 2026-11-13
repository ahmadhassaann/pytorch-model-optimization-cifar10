# CIFAR-10 ResNet-18 Model Optimization Report

## Executive summary

This project evaluates four model-optimization approaches for CIFAR-10 image
classification: post-training static quantization (PTQ), quantization-aware
training (QAT), iterative unstructured pruning, and structured channel pruning.
All methods start from a CIFAR-10-specific ResNet-18 FP32 baseline.

On the PyTorch CPU benchmark over all 10,000 test images, the baseline reached
**93.68%** top-1 accuracy, with a mean batch-size-1 latency of **19.984 ms** and
a checkpoint size of **42.701 MB**. The principal results were:

- **PTQ INT8** retained accuracy (**93.69%**) while reducing mean latency to
  **9.560 ms** and checkpoint size to **10.784 MB**.
- **QAT INT8** achieved the highest accuracy (**94.15%**) among the five
  PyTorch-benchmarked models, with **9.824 ms** mean latency and a **10.784 MB**
  checkpoint.
- **Unstructured pruning** reached **80.00% weight sparsity** and **94.21%**
  accuracy, but its dense checkpoint, parameter count, MAC count, and measured
  latency remained essentially unchanged from the baseline.
- **Structured pruning** recovered baseline-level accuracy after fine-tuning
  (**93.65%**) while removing about half the parameters and MACs. Its mean
  latency was **12.339 ms**, and its checkpoint size was **21.734 MB**.

The ONNX Runtime benchmark gives a separate view of exported deployment
artifacts. It used 2,000 test images rather than the full 10,000, so its
accuracy figures should not be compared as if they were measured on the same
sample set as the PyTorch benchmark. The ONNX structured model had the lowest
reported mean latency (**7.311 ms**) in that benchmark. The exported QAT ONNX
model was **42.787 MB** and had a mean latency of **20.484 ms**, so the QAT
deployment result depends strongly on the export/runtime path.

## 1. Objective and scope

The objective is to compare model compression and acceleration approaches
against a common image-classification task and baseline. The project produces
five variants:

1. FP32 ResNet-18 baseline.
2. PTQ INT8 model.
3. QAT INT8 model.
4. Iteratively unstructured-pruned FP32 model.
5. Structurally channel-pruned FP32 model.

The repository includes training, conversion, evaluation, checkpoint export,
ONNX export, and benchmark scripts. This report summarizes the recorded
artifacts under `results/` and the implementation in `src/`. It does not claim
measurements beyond those files.

## 2. Dataset and model

### Dataset and preprocessing

The task uses CIFAR-10, which contains 32-by-32 RGB images in ten classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
Training images are augmented with a random crop after four-pixel padding and
random horizontal flips. Both training and test images are converted to
tensors and normalized using channel means `(0.4914, 0.4822, 0.4465)` and
standard deviations `(0.2470, 0.2435, 0.2616)`.

The main PyTorch benchmark uses the complete 10,000-image CIFAR-10 test set.
The ONNX benchmark uses its default 2,000-image test subset. The PTQ
calibration loader samples 2,000 images from the training set by default;
ONNX PTQ export uses a separate calibration setting of 512 samples by default,
read in batches of 32 (up to eight batches).

### Architecture

The model is a CIFAR-oriented ResNet-18 implemented in `src/common.py`. It
uses a 3-by-3 convolutional stem with stride 1 and does not use the ImageNet
7-by-7 stem or initial max-pooling layer. It has eight residual BasicBlocks,
adaptive average pooling, and a ten-class linear output. Residual additions
use a quantization-compatible `FloatFunctional` operation.

The baseline contains **11,173,962** parameters and performs
**555,422,720 MACs** for a single 32-by-32 input according to the project's
Conv2d/Linear hook-based counter.

## 3. Optimization methods

### FP32 baseline

The baseline is initialized from scratch and trained using SGD with momentum
0.9, Nesterov momentum, weight decay `5e-4`, and a OneCycle learning-rate
schedule. The recorded training log contains 30 epochs. Its test accuracy
increased from 60.61% after epoch 1 to 93.68% after epoch 30. Training
accuracy at the final epoch was 99.40%, indicating a noticeable train/test
gap at the end of training.

### Post-training static quantization

PTQ starts from the saved FP32 checkpoint. The implementation fuses
convolution, batch-normalization, and (where applicable) ReLU modules, selects
the available `fbgemm` or `qnnpack` quantized backend, inserts observers,
calibrates activation ranges on training images, and converts the model to
INT8. The eager-mode PyTorch INT8 checkpoint and TorchScript copy are saved.

For ONNX, the implementation quantizes the FP32 ONNX graph with ONNX Runtime
static quantization in QDQ format. This is a separate export path from the
eager-mode PyTorch checkpoint.

### Quantization-aware training

QAT initializes from the FP32 checkpoint, fuses modules in QAT mode, inserts
fake-quantization modules, and fine-tunes using SGD. The default pipeline uses
five QAT epochs. Observers are disabled late in training and batch-normalization
statistics are frozen in the final portion. The model is then converted to
eager-mode INT8 and saved along with a fake-quant checkpoint and TorchScript
copy.

The ONNX QAT export is produced from the saved fake-quant model using
exportable fake-quant modules, rather than by exporting the eager INT8 operators
directly. This difference is relevant when interpreting the ONNX file size and
runtime measurements.

### Iterative unstructured pruning

Unstructured pruning applies global L1-magnitude pruning to Conv2d and Linear
weights over five iterations, targeting 80% total sparsity. Each iteration
prunes a fraction of the remaining eligible weights and is followed by
fine-tuning. The default pipeline performs a final five-epoch fine-tune, then
removes the pruning reparameterization so masked zeros are stored directly in
dense tensors.

The recorded schedule progressed from 27.52% sparsity after iteration 1 to
80.00% after iteration 5. Accuracy after the intermediate pruning and
fine-tuning steps is recorded in `results/unstructured_pruning_log.csv`.

### Structured channel pruning

Structured pruning ranks each BasicBlock's `conv1` output channels using the
L1 norm of its filters. At the default 0.5 removal ratio, the least important
channels are removed, the associated batch-normalization values and `conv2`
input channels are sliced, and a smaller model is physically reconstructed.
The residual stream widths are preserved to keep skip-connection shapes
valid. The reconstructed model is fine-tuned for ten epochs by default.

This changes layer dimensions rather than leaving zero-valued channels in
dense tensors. As a result, the resulting model has fewer parameters and MACs.

## 4. Evaluation methodology

### PyTorch benchmark

`src/benchmark.py` evaluates every available model on the CIFAR-10 test set.
It reports top-1 accuracy, checkpoint file size, mean and p95 batch-size-1
latency, throughput, parameter counts, weight sparsity, and MACs where the
model type supports those calculations. The recorded run has 10,000 test
samples. Batch-size-1 latency uses one CPU thread, ten warm-up inferences, and
200 timed samples by default.

INT8 parameter, sparsity, and MAC fields are blank in the PyTorch CSV because
the benchmark intentionally does not calculate those values for its eager
quantized model objects. These blanks are unavailable measurements, not zero
parameters or zero computation.

### ONNX Runtime benchmark

`src/onnx_inference.py` evaluates exported ONNX files using ONNX Runtime's
`CPUExecutionProvider`, one thread, and the default 2,000-image test subset.
Mean and p95 single-image latency use ten warm-ups and 200 measured samples.
Throughput is measured while processing the selected test subset in batches.

The PyTorch and ONNX measurements use different runtimes and test-set sizes.
Latency and throughput should therefore be interpreted within each benchmark
table, rather than treated as a perfectly controlled head-to-head runtime
comparison.

## 5. PyTorch benchmark results

| Model | Precision | Test accuracy | Mean latency (ms/img) | P95 latency (ms) | Throughput (img/s) | Checkpoint (MB) | Parameters | Nonzero parameters | Weight sparsity | MACs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 baseline | FP32 | 93.68% | 19.984 | 23.430 | 69.8 | 42.701 | 11,173,962 | 11,173,962 | 0.00% | 555,422,720 |
| PTQ | INT8 | 93.69% | 9.560 | 11.658 | 103.6 | 10.784 | — | — | — | — |
| QAT | INT8 | 94.15% | 9.824 | 14.544 | 103.8 | 10.784 | — | — | — | — |
| Unstructured pruning | FP32 | 94.21% | 20.676 | 23.251 | 69.9 | 42.702 | 11,173,962 | 2,242,480 | 80.00% | 555,422,720 |
| Structured pruning | FP32 | 93.65% | 12.339 | 14.017 | 113.9 | 21.734 | 5,679,306 | 5,679,306 | 0.00% | 281,744,384 |

All rows in this table use 10,000 test samples. Values are transcribed from
`results/benchmark_results.csv`.

### Baseline training

| Metric | Recorded value |
|---|---:|
| Epochs | 30 |
| Final train accuracy | 99.40% |
| Final test accuracy | 93.68% |
| Final train loss | 0.0225 |

Test accuracy peaked at 93.68% in the final logged epoch. The log shows
epoch-to-epoch variation before the final convergence, with a 90.56% result
at epoch 23 and a subsequent increase through epoch 30.

### Pruning progression

| Iteration | Sparsity | Accuracy after pruning | Accuracy after fine-tuning |
|---:|---:|---:|---:|
| 1 | 27.52% | 93.64% | 93.77% |
| 2 | 47.47% | 93.63% | 93.87% |
| 3 | 61.93% | 93.34% | 93.92% |
| 4 | 72.41% | 91.64% | 93.95% |
| 5 | 80.00% | 92.48% | 93.79% |

The structured-pruning log reports the baseline at 93.68% accuracy with
11,173,962 parameters and 555,422,720 MACs. Immediately after pruning, accuracy
dropped to 40.75%; after fine-tuning, it recovered to 93.65%. The reconstructed
model has 5,679,306 parameters and 281,744,384 MACs both before and after
fine-tuning.

## 6. ONNX Runtime benchmark results

| Exported model | Test accuracy | Mean latency (ms/img) | P95 latency (ms) | Throughput (img/s) | ONNX file (MB) |
|---|---:|---:|---:|---:|---:|
| FP32 baseline | 93.90% | 12.645 | 17.449 | 81.9 | 42.616 |
| PTQ INT8 QDQ | 92.85% | 9.433 | 11.363 | 139.4 | 10.766 |
| QAT fake-quant/QDQ | 93.95% | 20.484 | 24.700 | 56.6 | 42.787 |
| Structured pruning | 93.20% | 7.311 | 10.629 | 155.7 | 21.659 |
| Unstructured pruning | 94.25% | 12.751 | 17.982 | 80.8 | 42.616 |

These are 2,000-sample results from `results/onnx_benchmark.csv`, using
`CPUExecutionProvider` and one thread. In particular, the QAT ONNX result
should not be conflated with the 10.784 MB eager-mode PyTorch INT8 checkpoint:
the ONNX file is 42.787 MB and has a substantially different measured latency.

## 7. Analysis

### Quantization

Against the PyTorch FP32 baseline, PTQ reduced mean latency from 19.984 ms to
9.560 ms (about **52.2% lower**) and checkpoint size from 42.701 MB to
10.784 MB (about **74.7% smaller**), while its measured accuracy was 0.01
percentage points higher. QAT reduced mean latency by about **50.8%**, with
accuracy 0.47 percentage points above baseline and the same recorded checkpoint
size as PTQ.

The PyTorch benchmark therefore shows that both eager-mode INT8 variants offer
a favorable accuracy/latency/size tradeoff in this run. The separate ONNX
results are less uniform: PTQ QDQ is smaller and faster than the FP32 ONNX
baseline, while the QAT ONNX graph is close to FP32 size and slower than the
other reported ONNX models. The runtime/export representation matters; an INT8
label by itself does not establish that an exported graph will be smaller or
faster on a particular inference provider.

### Unstructured versus structured pruning

Unstructured pruning retains the full dense model shape. It leaves the total
parameter count and MAC count unchanged in the benchmark, and its checkpoint
is approximately the same size as the baseline. Its 80% weight sparsity is a
property of weight values, but the measured CPU latency did not improve
(20.676 ms versus 19.984 ms for baseline). This implementation does not use a
sparse storage format or sparse execution kernel, so sparsity alone does not
translate to a dense-kernel speedup.

Structured pruning physically reduces the widths of internal block layers.
Relative to the baseline, the recorded parameter count is about **49.2% lower**,
the MAC count about **49.3% lower**, and the checkpoint about **49.1% smaller**.
Mean CPU latency is about **38.3% lower**. Fine-tuning restores the accuracy
from 40.75% immediately after pruning to 93.65%, only 0.03 percentage points
below the baseline. This makes structured pruning the clearest option in this
run when the goal is to reduce dense-model computation and storage while
retaining near-baseline accuracy.

### Interpreting accuracy and benchmark differences

The observed small accuracy differences among variants are measurements for
the saved artifacts and evaluation subsets in this run; they are not evidence
that a compression method generally improves accuracy. The test-set scores in
the ONNX table use 2,000 images, whereas PyTorch scores use 10,000. Differences
between the tables can result from both the runtime/export path and the
different sample set.

Latency is hardware- and software-dependent. The recorded CSV files do not
include processor model, operating system, Python/PyTorch/ONNX Runtime versions,
or thermal/power conditions. Consequently, these absolute timings should be
treated as local measurements, not portable performance guarantees.

## 8. Conclusions and recommendations

1. **For a compact eager-mode CPU checkpoint:** PTQ is a strong default in
   these measurements. It approximately halves batch-size-1 latency, reduces
   checkpoint size by roughly three quarters, and retains baseline accuracy.
2. **For best measured PyTorch accuracy among these artifacts:** QAT reached
   94.15% and retained the compact INT8 checkpoint size, with a latency close
   to PTQ.
3. **For reducing dense model computation:** structured channel pruning gives
   the strongest measured reduction in parameters, MACs, checkpoint size, and
   PyTorch latency while recovering near-baseline accuracy after fine-tuning.
4. **For a sparse-aware deployment target:** unstructured pruning may be useful
   where the deployment stack supports sparse weight storage or sparse
   kernels. The current dense CPU benchmark does not demonstrate a speed or
   file-size advantage from its 80% sparsity.
5. **For ONNX deployment:** benchmark the actual target provider and exported
   graph. In this run PTQ QDQ offered an accuracy/size/latency improvement over
   the FP32 ONNX baseline, the structured graph was fastest, and QAT ONNX was
   neither the smallest nor fastest export.

These conclusions apply to the saved models and benchmark conditions recorded
in this repository. A deployment decision should be validated on the target
hardware with the same test data, thread settings, runtime version, and
latency protocol.

## 9. Reproducibility and artifacts

The scripts accept a `--seed` argument (default 42), and `run_all.py` runs the
pipeline in sequence. For a full run, the default pipeline configures 30
baseline epochs, 5 QAT epochs, 5 final unstructured-pruning fine-tuning epochs,
and 10 structured-pruning fine-tuning epochs. A `--quick` mode is available
for smoke testing on smaller subsets; its results should not be mixed with the
full-run metrics in this report.

Key artifacts:

- `results/baseline_train_log.csv` — epoch-level baseline training metrics.
- `results/benchmark_results.csv` — PyTorch accuracy, latency, throughput,
  checkpoint size, sparsity, parameters, and MAC counts.
- `results/onnx_benchmark.csv` — ONNX Runtime CPU benchmark.
- `results/unstructured_pruning_log.csv` — pruning iteration progression.
- `results/structured_pruning_log.csv` — structured-pruning accuracy,
  parameter, and MAC comparison.
- `results/predictions.csv` — per-sample labels, predictions, and confidence
  values from the PyTorch benchmark.
- `models/` — saved checkpoints, TorchScript files, and ONNX exports.

The recorded results do not identify the exact machine or dependency versions.
For a reproducible performance study, record those details along with each
benchmark run and rerun all variants under the same runtime and hardware
conditions.

## 10. References within this project

- `README.md` — setup, pipeline commands, deliverables, and method overview.
- `src/common.py` — dataset transforms, CIFAR ResNet-18, metrics, and
  quantization helpers.
- `src/train_baseline.py`, `src/ptq.py`, `src/qat.py` — baseline and
  quantization procedures.
- `src/prune_unstructured.py`, `src/prune_structured.py`, `src/finetune.py` —
  pruning and fine-tuning procedures.
- `src/benchmark.py`, `src/export_onnx.py`, `src/onnx_inference.py` — PyTorch
  benchmark, ONNX export, and ONNX Runtime evaluation.
