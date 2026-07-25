"""Activation statistics collectors — the host-side mirror of what a TRT INT8
calibrator accumulates per tensor across the calibration set.

``MinMaxCollector``  -> feeds the min-max calibrator.
``HistogramCollector`` -> feeds the entropy (KL) calibrator.

Both accumulate incrementally so they can process the calibration set in batches
(32 / 128 / 512 images) without holding all activations in memory — matching how
the on-device calibrator streams batches.
"""
from __future__ import annotations

import numpy as np


class MinMaxCollector:
    """Running symmetric absolute-max (and signed min/max) per tensor."""

    def __init__(self) -> None:
        self._min = np.inf
        self._max = -np.inf
        self._count = 0

    def update(self, activations: np.ndarray) -> "MinMaxCollector":
        a = np.asarray(activations, dtype=np.float64)
        if a.size == 0:
            return self
        self._min = min(self._min, float(a.min()))
        self._max = max(self._max, float(a.max()))
        self._count += a.size
        return self

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    @property
    def absmax(self) -> float:
        if self._count == 0:
            return 0.0
        return max(abs(self._min), abs(self._max))


class HistogramCollector:
    """Accumulate a histogram of |activation| over [0, running_absmax].

    TensorRT's entropy calibrator builds a fine histogram (typically 2048 bins) of
    absolute activation magnitudes, then searches for the clipping threshold that
    minimizes KL divergence between the full distribution and its 128-bin (INT8)
    requantization.

    Because activation ranges are unknown up front and grow across batches, we
    rebin lazily: when a batch exceeds the current upper bound we widen the range
    and rescale existing counts into the new bins (count-preserving).
    """

    def __init__(self, num_bins: int = 2048) -> None:
        if num_bins < 128:
            raise ValueError("num_bins must be >= 128 (INT8 target is 128 bins)")
        self.num_bins = num_bins
        self._hist = np.zeros(num_bins, dtype=np.float64)
        self._upper = 0.0  # current histogram upper edge (in |activation|)

    def _grow_to(self, new_upper: float) -> None:
        if new_upper <= self._upper or self._hist.sum() == 0:
            self._upper = max(self._upper, new_upper)
            return
        # Re-project existing counts onto the widened range, preserving total count.
        old_edges = np.linspace(0, self._upper, self.num_bins + 1)
        old_centers = 0.5 * (old_edges[:-1] + old_edges[1:])
        new_hist = np.zeros(self.num_bins, dtype=np.float64)
        new_bin = np.clip(
            (old_centers / new_upper * self.num_bins).astype(int), 0, self.num_bins - 1
        )
        np.add.at(new_hist, new_bin, self._hist)
        self._hist = new_hist
        self._upper = new_upper

    def update(self, activations: np.ndarray) -> "HistogramCollector":
        a = np.abs(np.asarray(activations, dtype=np.float64)).ravel()
        if a.size == 0:
            return self
        amax = float(a.max())
        if amax > self._upper:
            self._grow_to(amax if amax > 0 else 1e-12)
        if self._upper <= 0:
            self._upper = 1e-12
        idx = np.clip((a / self._upper * self.num_bins).astype(int), 0, self.num_bins - 1)
        np.add.at(self._hist, idx, 1.0)
        return self

    @property
    def histogram(self) -> np.ndarray:
        return self._hist.copy()

    @property
    def upper(self) -> float:
        return self._upper

    def bin_edges(self) -> np.ndarray:
        return np.linspace(0, self._upper, self.num_bins + 1)
