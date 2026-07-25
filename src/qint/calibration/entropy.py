"""Entropy (KL-divergence) calibration — host-testable.

This is the algorithm behind TensorRT's ``IInt8EntropyCalibrator2``: given a fine
histogram of |activation| (built by :class:`~qint.calibration.stats.HistogramCollector`),
search for the clipping threshold ``T`` such that quantizing the clipped distribution
into 128 levels minimizes the KL divergence between the original distribution ``P`` and
its requantized approximation ``Q``. The scale is then ``T / 127``.

Implementation follows the well-known reference (NVIDIA "8-bit Inference with TensorRT",
mirrored in the MXNet/PyTorch quantization toolkits). We keep it in numpy so the exact
threshold selection is unit-testable off-device; the on-device calibrator
(``trt_calibrator.py``) delegates to TensorRT's own C++ implementation of the same idea,
and we cross-check that the cached scales are in the same ballpark.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _smooth(dist: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Add mass to empty bins so KL is finite; keep support of the other dist."""
    dist = dist.astype(np.float64).copy()
    is_zero = dist == 0
    is_nonzero = ~is_zero
    n_zero = int(is_zero.sum())
    n_nonzero = int(is_nonzero.sum())
    if n_nonzero == 0:
        return np.full_like(dist, 1.0 / dist.size)
    dist[is_zero] = eps
    dist[is_nonzero] -= eps * n_zero / n_nonzero
    return dist


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q). Inputs are (possibly unnormalized) non-negative arrays; both are
    normalized to probability distributions internally."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    mask = p > 0
    # q>0 wherever p>0 is guaranteed by _smooth in the caller.
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


@dataclass
class EntropyResult:
    threshold_bin: int          # number of histogram bins kept below the clip
    threshold_value: float      # clip threshold in activation units
    scale: float                # threshold_value / 127
    num_bins: int
    target_bins: int
    kl: np.ndarray              # KL per candidate threshold (for plots/tests)


def entropy_calibrate(
    histogram: np.ndarray,
    upper: float,
    target_bins: int = 128,
) -> EntropyResult:
    """Find the KL-optimal INT8 clipping threshold for one tensor.

    Parameters
    ----------
    histogram : np.ndarray
        Fine histogram of |activation| over [0, upper] (from HistogramCollector).
    upper : float
        Upper edge of the histogram in activation units.
    target_bins : int
        Number of quantized levels on the positive side (128 for symmetric INT8).
    """
    hist = np.asarray(histogram, dtype=np.float64)
    n = hist.size
    if n <= target_bins:
        raise ValueError(f"histogram bins ({n}) must exceed target_bins ({target_bins})")
    bin_width = upper / n if upper > 0 else 1.0

    kls = np.full(n - target_bins + 1, np.inf)
    for t, threshold in enumerate(range(target_bins, n + 1)):
        sliced = hist[:threshold].copy()
        # reference P: clip — dump everything at/after threshold into the last kept bin
        p = sliced.copy()
        p[-1] += hist[threshold:].sum()
        nonzero = (p != 0).astype(np.float64)

        # merge the `threshold` bins into `target_bins` quantized bins
        merged = threshold // target_bins
        quant = np.zeros(target_bins, dtype=np.float64)
        for j in range(target_bins):
            start = j * merged
            stop = start + merged if j < target_bins - 1 else threshold
            quant[j] = sliced[start:stop].sum()

        # expand quantized bins back to `threshold` bins, spreading over occupied slots
        q = np.zeros(threshold, dtype=np.float64)
        for j in range(target_bins):
            start = j * merged
            stop = start + merged if j < target_bins - 1 else threshold
            occ = nonzero[start:stop].sum()
            if occ > 0:
                q[start:stop] = np.where(nonzero[start:stop] > 0, quant[j] / occ, 0.0)

        if p.sum() <= 0 or q.sum() <= 0:
            continue
        kls[t] = kl_divergence(_smooth(p), _smooth(q))

    best = int(np.argmin(kls))
    threshold_bin = best + target_bins
    threshold_value = (threshold_bin + 0.5) * bin_width
    scale = threshold_value / 127.0
    return EntropyResult(
        threshold_bin=threshold_bin,
        threshold_value=float(threshold_value),
        scale=float(scale),
        num_bins=n,
        target_bins=target_bins,
        kl=kls,
    )
