"""Mixed-precision planning — host-testable.

Turns a layer-sensitivity ranking into a concrete precision map (which layers to pin
to FP16 while the rest stay INT8) and provides the accuracy-per-millisecond bookkeeping
that DECISIONS.md must report.

The device build script reads a :class:`PrecisionPlan` (serialized to JSON) and applies
it to the TensorRT network via ``layer.precision = trt.float16`` +
``builder_config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from .sensitivity import LayerSensitivity, rank_layers


@dataclass
class PrecisionPlan:
    """A per-layer precision assignment for a mixed-precision engine."""

    fp16_layers: List[str] = field(default_factory=list)
    default_precision: str = "int8"
    note: str = ""

    def precision_of(self, layer_name: str) -> str:
        return "fp16" if layer_name in set(self.fp16_layers) else self.default_precision

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "PrecisionPlan":
        d = json.loads(text)
        return PrecisionPlan(
            fp16_layers=list(d.get("fp16_layers", [])),
            default_precision=d.get("default_precision", "int8"),
            note=d.get("note", ""),
        )

    def to_trt_directives(self) -> List[dict]:
        """Serializable directives the on-device builder applies to the TRT network."""
        return [{"layer": name, "precision": "fp16", "set_output_type": True}
                for name in self.fp16_layers]


def build_fallback_plan(
    layers: Sequence[LayerSensitivity],
    k: int = 5,
    note: str = "",
) -> PrecisionPlan:
    """Keep the top-``k`` most sensitive layers (lowest SQNR) in FP16."""
    ranked = rank_layers(layers)[:k]
    return PrecisionPlan(
        fp16_layers=[l.name for l in ranked],
        default_precision="int8",
        note=note or f"top-{k} SQNR-sensitive layers pinned to FP16",
    )


@dataclass
class AccuracyPerMs:
    """Accuracy bought per unit of extra latency by a mixed-precision config."""

    config: str
    d_map5095: float           # absolute mAP@50-95 gained vs the pure-INT8 baseline
    d_latency_ms: float        # extra latency vs the pure-INT8 baseline (mean, ms)

    @property
    def map_per_ms(self) -> float:
        """mAP points (in %) recovered per millisecond of added latency.

        Positive = the FP16 fallback recovered accuracy. If latency did not increase
        (or decreased), returns +inf when accuracy improved, else the raw delta.
        """
        if self.d_latency_ms <= 0:
            return float("inf") if self.d_map5095 > 0 else 0.0
        return (self.d_map5095 * 100.0) / self.d_latency_ms


def accuracy_per_ms(
    int8_map5095: float,
    int8_latency_ms: float,
    configs: Dict[str, Dict[str, float]],
) -> List[AccuracyPerMs]:
    """Compute accuracy-per-ms for each mixed-precision config vs the INT8 baseline.

    ``configs`` maps a config name -> {"map5095": float, "latency_ms": float}.
    """
    out: List[AccuracyPerMs] = []
    for name, m in configs.items():
        out.append(
            AccuracyPerMs(
                config=name,
                d_map5095=m["map5095"] - int8_map5095,
                d_latency_ms=m["latency_ms"] - int8_latency_ms,
            )
        )
    return out
