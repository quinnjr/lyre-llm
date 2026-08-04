# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

import os

import torch

from lyre.config import load_config
from lyre.errors import LyreError
from lyre.reporting import warn
from lyre.scripts.train import load_checkpoint


def _int8_path(out):
    base, ext = os.path.splitext(out)
    return f"{base}.int8{ext or '.onnx'}"


def _discard_partial(path):
    """Remove a half-written artefact so a failed export leaves no valid-looking file."""
    try:
        os.remove(path)
    except OSError:
        pass


def _quantize_onnx(src, dst):
    """Dynamic int8 quantization of an existing ONNX graph.

    Returns the backend name used, or raises :class:`LyreError` when no
    quantization backend is installed. Nothing is installed on the fly.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise LyreError(
            "int8 export needs onnxruntime with the quantization extra "
            "(pip install lyre[export]); ONNX graph quantization is unavailable: %s" % exc
        ) from exc
    try:
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
    except Exception as exc:
        # Unsupported ops, a failed model check, or an ORT graph-transformer
        # error are all ordinary "this graph will not quantize" outcomes, and
        # quantize_dynamic may already have written a partial dst.
        _discard_partial(dst)
        raise LyreError(
            f"onnxruntime could not quantize {src}: {type(exc).__name__}: {exc}"
        ) from exc
    return "onnxruntime.quantization.quantize_dynamic"


def _quantize_torch(model, example, dst):
    """Fallback: quantize the eager model, then re-export it to ONNX.

    torch's eager dynamic quantization only covers Linear/RNN modules, so a
    purely convolutional encoder comes back untouched. Detect that instead of
    writing a fp32 file under an ``.int8`` name.
    """
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError as exc:
        raise LyreError(
            f"no int8 quantization backend available (pip install lyre[export]): {exc}"
        ) from exc
    quantizable = {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}
    if not any(type(m) in quantizable for m in model.modules()):
        raise LyreError(
            "torch dynamic quantization only covers Linear/LSTM/GRU modules and this "
            "model has none, so it would emit an fp32 graph named int8. Install "
            "onnxruntime to quantize the ONNX graph instead."
        )
    qmodel = quantize_dynamic(model, quantizable, dtype=torch.qint8)
    _export_onnx(qmodel, example, dst)
    return "torch.ao.quantization.quantize_dynamic"


def _export_onnx(model, example, out):
    try:
        torch.onnx.export(
            model,
            example,
            out,
            input_names=["logmel"],
            output_names=["pitch_logits", "onset_logits"],
            opset_version=17,
            dynamic_axes={"logmel": {2: "time"}},
        )
    except ImportError as exc:
        _discard_partial(out)
        raise LyreError(
            "ONNX export needs the torch ONNX exporter dependencies "
            f"(pip install lyre[export]): {exc}"
        ) from exc
    except Exception as exc:
        # Anything the exporter raises (unsupported op, tracing failure) is a
        # reportable export failure, not a bug -- and it may have left a partial
        # file that would otherwise be announced as a successful export.
        _discard_partial(out)
        raise LyreError(
            f"ONNX export of {out} failed: {type(exc).__name__}: {exc}"
        ) from exc


def main(checkpoint, out, config_path, quantize=False):
    """Export the transcription model to ONNX, optionally also as int8.

    ``quantize`` additionally writes ``<out>.int8.onnx`` (spec: "ONNX +
    optional int8 quantization"). It prefers onnxruntime's graph-level dynamic
    quantizer and falls back to torch's eager dynamic quantization; when
    neither is importable it raises :class:`~lyre.errors.LyreError` rather than
    installing anything.
    """
    config = load_config(config_path)
    failures = []
    notes = []
    feats = config["features"]
    from lyre.transcriber.model import MultiPitchNet

    model = MultiPitchNet(
        n_mels=feats["n_mels"], n_notes=config["model"]["n_notes"], channels=config["model"]["channels"]
    )
    state = load_checkpoint(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    example = torch.randn(1, 1, feats["window_frames"], feats["n_mels"])
    _export_onnx(model, example, out)
    print(f"exported -> {out}")

    result = {"onnx": out, "int8": None, "backend": None, "failures": failures, "notes": notes}
    if not quantize:
        return result

    int8_out = _int8_path(out)
    try:
        backend = _quantize_onnx(out, int8_out)
    except LyreError as exc:
        warn(f"{exc}; falling back to torch dynamic quantization", notes)
        try:
            backend = _quantize_torch(model, example, int8_out)
        except LyreError as fallback_exc:
            # --int8 was asked for and no backend delivered it: that is a failed
            # run. But the fp32 graph above is real and usable, so it is still
            # returned and named -- a degraded export, not nothing at all.
            _discard_partial(int8_out)
            warn(
                f"no int8 artefact was written; every quantization backend failed. "
                f"Last error: {fallback_exc}",
                failures,
            )
            return result
    print(f"quantized (int8, {backend}) -> {int8_out}")
    result["int8"] = int8_out
    result["backend"] = backend
    return result
