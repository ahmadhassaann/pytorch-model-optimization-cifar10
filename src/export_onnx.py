"""Export the models to ONNX for CPU deployment with ONNX Runtime.

  resnet18_fp32.onnx          FP32 baseline
  resnet18_unstructured.onnx  unstructured-pruned (dense tensors containing zeros)
  resnet18_structured.onnx    structured-pruned (physically smaller network)
  resnet18_ptq_int8.onnx      INT8 (QDQ) - ONNX Runtime static quantization of the FP32 ONNX
                              (PyTorch eager INT8 ops cannot be exported to ONNX directly)
  resnet18_qat_int8.onnx      QAT model exported with fake-quant nodes (QuantizeLinear /
                              DequantizeLinear), which ONNX Runtime executes as INT8 (QDQ)

Each export is attempted independently; failures are reported, not fatal.
"""
import argparse
import os

import numpy as np
import onnx
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from torch.ao.quantization import (MovingAverageMinMaxObserver,
                                   MovingAveragePerChannelMinMaxObserver, QConfig, disable_observer,
                                   prepare_qat)
from torch.ao.quantization.fake_quantize import FakeQuantize

from common import (MODELS_DIR, ResNet18CIFAR, file_size_mb, get_calib_loader, load_model,
                     set_backend, set_seed, torch_load)
from qat import FAKEQUANT_FILE

OPSET = 13


def export_onnx(model, path):
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=OPSET,
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)   # torch >= 2.5
    except TypeError:
        torch.onnx.export(model, dummy, path, **kwargs)                 # older torch
    onnx.checker.check_model(onnx.load(path))


class CifarCalibReader(CalibrationDataReader):
    def __init__(self, loader, max_batches=8):
        self.batches = [x.numpy() for _, (x, _) in zip(range(max_batches), loader)]
        self.it = iter(self.batches)

    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {"input": b}


def export_qat_fakequant(path):
    """Rebuild the QAT model with plain (exportable) FakeQuantize modules and export it.

    The default fbgemm QAT qconfig uses a fused fake-quant op that ONNX cannot represent,
    so a functionally identical qconfig built from standard FakeQuantize is used here.
    State-dict keys are identical, so the trained fake-quant parameters load directly.
    """
    backend = set_backend()
    qconfig = QConfig(
        activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0,
                                          quant_max=255, dtype=torch.quint8,
                                          qscheme=torch.per_tensor_affine, reduce_range=True),
        weight=FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-128,
                                      quant_max=127, dtype=torch.qint8,
                                      qscheme=torch.per_channel_symmetric, ch_axis=0))
    m = ResNet18CIFAR()
    m.train()
    m.fuse(qat=True)
    m.qconfig = qconfig
    prepare_qat(m, inplace=True)
    m.load_state_dict(torch_load(os.path.join(MODELS_DIR, FAKEQUANT_FILE)))
    m.apply(disable_observer)
    export_onnx(m.eval(), path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib-samples", type=int, default=512)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    set_seed(args.seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    def out(name):
        return os.path.join(MODELS_DIR, name)

    # float models
    for kind, name in (("fp32", "resnet18_fp32.onnx"), ("unstructured", "resnet18_unstructured.onnx"),
                       ("structured", "resnet18_structured.onnx")):
        try:
            export_onnx(load_model(kind), out(name))
            print(f"[ok]   {name} ({file_size_mb(out(name)):.2f} MB)")
        except Exception as e:
            print(f"[fail] {name}: {e}")

    # INT8 via ONNX Runtime static quantization (QDQ) of the FP32 ONNX
    try:
        loader = get_calib_loader(args.calib_samples, 32, args.workers, args.quick, args.seed)
        quantize_static(out("resnet18_fp32.onnx"), out("resnet18_ptq_int8.onnx"),
                        CifarCalibReader(loader), quant_format=QuantFormat.QDQ, per_channel=True,
                        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
        print(f"[ok]   resnet18_ptq_int8.onnx ({file_size_mb(out('resnet18_ptq_int8.onnx')):.2f} MB)")
    except Exception as e:
        print(f"[fail] resnet18_ptq_int8.onnx: {e}")

    # QAT (fake-quant / QDQ)
    try:
        export_qat_fakequant(out("resnet18_qat_int8.onnx"))
        print(f"[ok]   resnet18_qat_int8.onnx ({file_size_mb(out('resnet18_qat_int8.onnx')):.2f} MB)")
    except Exception as e:
        print(f"[fail] resnet18_qat_int8.onnx: {e}")


if __name__ == "__main__":
    main()
