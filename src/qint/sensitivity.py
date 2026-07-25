"""Layer-level quantization sensitivity — host-testable.

We quantify how much INT8 quantization distorts each layer's output using SQNR
(signal-to-quantization-noise ratio):

    SQNR_dB = 10 * log10( ||x||^2 / ||x - fake_quant(x)||^2 )

Low SQNR => the layer's output is badly represented in INT8 => a good candidate to
keep in FP16. We rank layers by SQNR (ascending) and take the top-K most sensitive.

The device script ``09_layer_sensitivity.py`` captures per-layer activations from the
real network via polygraphy/TRT and feeds them here; this module is the pure-numpy
scoring/ranking core and is fully unit-tested on host.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .calibration.minmax import fake_quant, minmax_scale


@dataclass
class LayerSensitivity:
    name: str
    sqnr_db: float
    mse: float
    energy: float          # ||x||^2, used to weight global contribution
    scale: float

    @property
    def error_energy(self) -> float:
        return self.mse * self.energy if not np.isinf(self.mse) else self.energy


def sqnr_db(reference: np.ndarray, quantized: np.ndarray) -> float:
    """SQNR in dB between a reference tensor and its (dequantized) quantization."""
    ref = np.asarray(reference, dtype=np.float64).ravel()
    q = np.asarray(quantized, dtype=np.float64).ravel()
    signal = float(np.sum(ref ** 2))
    noise = float(np.sum((ref - q) ** 2))
    if noise <= 0:
        return float("inf")
    if signal <= 0:
        return float("-inf")
    return 10.0 * np.log10(signal / noise)


def score_layer(name: str, activation: np.ndarray, scale: Optional[float] = None) -> LayerSensitivity:
    """Score one layer by fake-quantizing its activation with a min-max scale."""
    a = np.asarray(activation, dtype=np.float64)
    if scale is None:
        scale = float(minmax_scale(np.max(np.abs(a))))
    q = fake_quant(a, scale)
    mse = float(np.mean((a - q) ** 2))
    return LayerSensitivity(
        name=name,
        sqnr_db=sqnr_db(a, q),
        mse=mse,
        energy=float(np.sum(a ** 2)),
        scale=float(scale),
    )


def rank_layers(layers: Sequence[LayerSensitivity]) -> List[LayerSensitivity]:
    """Most sensitive first: ascending SQNR (worst representation at the top)."""
    return sorted(layers, key=lambda l: (l.sqnr_db, -l.error_energy))


def top_k_sensitive(layers: Sequence[LayerSensitivity], k: int = 5) -> List[LayerSensitivity]:
    return rank_layers(layers)[:k]


def format_sensitivity_table(layers: Sequence[LayerSensitivity], k: Optional[int] = None) -> str:
    ranked = rank_layers(layers)
    if k is not None:
        ranked = ranked[:k]
    lines = [
        "| rank | layer | SQNR (dB) | MSE | output energy | err·energy |",
        "|---|---|---|---|---|---|",
    ]
    for i, l in enumerate(ranked, 1):
        sqnr = "inf" if np.isinf(l.sqnr_db) else f"{l.sqnr_db:7.2f}"
        lines.append(
            f"| {i} | {l.name} | {sqnr} | {l.mse:.3e} | {l.energy:.3e} | {l.error_energy:.3e} |"
        )
    return "\n".join(lines)
