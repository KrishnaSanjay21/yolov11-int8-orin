"""Result serialization + BENCHMARKS.md rendering — pure python, host-testable.

Device scripts dump results as JSON (``results/raw/*.json``); the committed
``scripts/fill_benchmarks.py`` reads them and renders the tables. Keeping the render
logic here (not inline in the device script) means it is unit-tested and the device run
is just "produce numbers", not "format markdown".
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .accuracy.metrics import AccuracyResult


# ---- AccuracyResult <-> dict ------------------------------------------------
def accuracy_to_dict(res: AccuracyResult) -> dict:
    return {
        "precision": res.precision,
        "num_classes": res.num_classes,
        "method": res.method,
        "iou_thresholds": list(res.iou_thresholds),
        # JSON keys must be strings
        "per_class_ap": {
            str(c): {str(t): (None if math.isnan(v) else v) for t, v in d.items()}
            for c, d in res.per_class_ap.items()
        },
    }


def accuracy_from_dict(d: dict) -> AccuracyResult:
    per_class = {
        int(c): {float(t): (float("nan") if v is None else float(v)) for t, v in dd.items()}
        for c, dd in d["per_class_ap"].items()
    }
    return AccuracyResult(
        precision=d["precision"],
        num_classes=int(d["num_classes"]),
        iou_thresholds=tuple(float(t) for t in d["iou_thresholds"]),
        per_class_ap=per_class,
        method=d.get("method", "101"),
    )


# ---- Benchmark rows ---------------------------------------------------------
@dataclass
class BenchmarkRow:
    """One row of BENCHMARKS.md — one precision/config.

    Fields default to None so a partially-measured row still renders (with "—"),
    making it obvious what still needs a device run rather than faking a value.
    """

    config: str
    latency_mean_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_fps: Optional[float] = None
    power_w: Optional[float] = None
    mem_mb: Optional[float] = None
    map50: Optional[float] = None
    map5095: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "BenchmarkRow":
        return BenchmarkRow(**{k: d.get(k) for k in BenchmarkRow.__dataclass_fields__})


def _fmt(v, spec="{:.2f}") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return spec.format(v)


def render_benchmarks_table(rows: List[BenchmarkRow]) -> str:
    header = (
        "| config | mAP@50 | mAP@50-95 | latency mean (ms) | latency p99 (ms) "
        "| throughput (FPS) | power (W) | mem (MB) |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r.config} | {_fmt(r.map50, '{:.3f}')} | {_fmt(r.map5095, '{:.3f}')} "
            f"| {_fmt(r.latency_mean_ms)} | {_fmt(r.latency_p99_ms)} "
            f"| {_fmt(r.throughput_fps, '{:.1f}')} | {_fmt(r.power_w)} | {_fmt(r.mem_mb, '{:.0f}')} |"
        )
    return "\n".join(lines)


def build_rows(
    bench_by_config: Dict[str, dict],
    acc_by_config: Dict[str, dict],
    order: Optional[List[str]] = None,
) -> List[BenchmarkRow]:
    """Merge per-config benchmark dicts and accuracy dicts into BenchmarkRows.

    A config present in only one of the two maps still yields a row (with the
    missing half left as None), so a partial device run renders honestly.
    """
    configs = order or sorted(set(bench_by_config) | set(acc_by_config))
    rows: List[BenchmarkRow] = []
    for cfg in configs:
        b = bench_by_config.get(cfg, {})
        row = BenchmarkRow.from_dict({**b, "config": cfg})
        a = acc_by_config.get(cfg)
        if a is not None:
            res = accuracy_from_dict(a)
            row.map50 = None if math.isnan(res.map50) else res.map50
            row.map5095 = None if math.isnan(res.map5095) else res.map5095
        rows.append(row)
    return rows


def save_json(path: str, obj) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)
