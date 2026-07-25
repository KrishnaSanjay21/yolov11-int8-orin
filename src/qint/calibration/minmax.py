"""Min-max calibration scales — host-testable.

Symmetric INT8 quantization maps a real value ``x`` to ``round(x / scale)`` clamped
to [-127, 127]. The min-max calibrator sets ``scale = absmax / 127``.

Per-tensor vs per-channel
-------------------------
TensorRT applies *per-tensor* scales to activations (a single scale for the whole
tensor) and, by default, *per-channel* scales to convolution weights (one scale per
output channel). The spec asks for a per-tensor vs per-channel comparison "where TRT
permits" — the place it permits is weights. ``compute_weight_scales`` computes both so
the two can be compared numerically off-device; ``05_build_int8.sh`` documents how to
force per-tensor weights in TRT for the A/B.
"""
from __future__ import annotations

import numpy as np

INT8_MAX = 127.0


def minmax_scale(absmax, int8_max: float = INT8_MAX) -> np.ndarray:
    """scale = absmax / 127 (symmetric). Accepts scalar or array (per-channel)."""
    absmax = np.asarray(absmax, dtype=np.float64)
    scale = absmax / int8_max
    # guard against zero-range tensors (dead channels) -> tiny non-zero scale
    scale = np.where(scale > 0, scale, np.finfo(np.float64).tiny)
    return scale


def quantize(x: np.ndarray, scale, int8_max: float = INT8_MAX) -> np.ndarray:
    """Fake-quantize: round-to-nearest, clamp to [-127,127]. Returns INT8-valued floats."""
    scale = np.asarray(scale, dtype=np.float64)
    q = np.rint(np.asarray(x, dtype=np.float64) / scale)
    return np.clip(q, -int8_max, int8_max)


def dequantize(q: np.ndarray, scale) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) * np.asarray(scale, dtype=np.float64)


def fake_quant(x: np.ndarray, scale, int8_max: float = INT8_MAX) -> np.ndarray:
    """Round-trip x -> INT8 -> dequant, for error analysis."""
    return dequantize(quantize(x, scale, int8_max), scale)


def compute_weight_scales(weights: np.ndarray, per_channel: bool = True, out_axis: int = 0):
    """Compute symmetric scales for a conv weight tensor.

    weights : (out_ch, in_ch, kh, kw) typically.
    per_channel=True  -> one scale per ``out_axis`` channel (TRT default for weights).
    per_channel=False -> single per-tensor scale.

    Returns an array broadcastable against ``weights``.
    """
    w = np.asarray(weights, dtype=np.float64)
    if per_channel:
        axes = tuple(i for i in range(w.ndim) if i != out_axis)
        absmax = np.max(np.abs(w), axis=axes)  # (out_ch,)
        scale = minmax_scale(absmax)
        shape = [1] * w.ndim
        shape[out_axis] = -1
        return scale.reshape(shape)
    else:
        return minmax_scale(np.max(np.abs(w)))
