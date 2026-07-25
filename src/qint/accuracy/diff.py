"""Per-class accuracy diffing across precision configs.

Given a baseline :class:`AccuracyResult` (typically FP32) and one or more candidate
results (FP16, INT8, ...), compute per-class absolute AP deltas and flag any class
whose AP drops by more than a threshold (default 2% absolute, per the spec).

Aggregate-only numbers are explicitly NOT the product here — the per-class table is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .metrics import AccuracyResult

DEFAULT_FLAG_THRESHOLD = 0.02  # 2% absolute AP drop


@dataclass
class ClassDelta:
    class_id: int
    class_name: str
    base_ap50: float
    cand_ap50: float
    base_ap5095: float
    cand_ap5095: float

    @property
    def d_ap50(self) -> float:
        return self.cand_ap50 - self.base_ap50

    @property
    def d_ap5095(self) -> float:
        return self.cand_ap5095 - self.base_ap5095

    def flagged(self, threshold: float = DEFAULT_FLAG_THRESHOLD) -> bool:
        """True if EITHER metric dropped by more than ``threshold`` (absolute)."""
        drop50 = -self.d_ap50
        drop5095 = -self.d_ap5095
        return (drop50 > threshold) or (drop5095 > threshold)


@dataclass
class PrecisionDiff:
    baseline_precision: str
    candidate_precision: str
    deltas: List[ClassDelta]
    threshold: float
    base_map50: float
    cand_map50: float
    base_map5095: float
    cand_map5095: float

    @property
    def flagged_classes(self) -> List[ClassDelta]:
        return [d for d in self.deltas if d.flagged(self.threshold)]

    @property
    def d_map50(self) -> float:
        return self.cand_map50 - self.base_map50

    @property
    def d_map5095(self) -> float:
        return self.cand_map5095 - self.base_map5095


def diff_precisions(
    baseline: AccuracyResult,
    candidate: AccuracyResult,
    class_names: Optional[Sequence[str]] = None,
    threshold: float = DEFAULT_FLAG_THRESHOLD,
) -> PrecisionDiff:
    """Diff a candidate precision against a baseline, per class."""
    classes = sorted(set(baseline.classes_with_gt) | set(candidate.classes_with_gt))
    deltas: List[ClassDelta] = []
    for c in classes:
        name = class_names[c] if class_names is not None and c < len(class_names) else str(c)
        deltas.append(
            ClassDelta(
                class_id=c,
                class_name=name,
                base_ap50=_safe(baseline, c, "ap50"),
                cand_ap50=_safe(candidate, c, "ap50"),
                base_ap5095=_safe(baseline, c, "ap5095"),
                cand_ap5095=_safe(candidate, c, "ap5095"),
            )
        )
    return PrecisionDiff(
        baseline_precision=baseline.precision,
        candidate_precision=candidate.precision,
        deltas=deltas,
        threshold=threshold,
        base_map50=baseline.map50,
        cand_map50=candidate.map50,
        base_map5095=baseline.map5095,
        cand_map5095=candidate.map5095,
    )


def _safe(res: AccuracyResult, c: int, which: str) -> float:
    try:
        v = res.ap50(c) if which == "ap50" else res.ap5095(c)
    except KeyError:
        return float("nan")
    return v


def format_delta_table(diff: PrecisionDiff) -> str:
    """Render a Markdown per-class delta table with >threshold flags."""
    pct = lambda v: "  n/a " if (v is None or np.isnan(v)) else f"{100*v:6.2f}"
    dpct = lambda v: "  n/a " if (v is None or np.isnan(v)) else f"{100*v:+6.2f}"
    thr = int(round(diff.threshold * 100))
    lines = [
        f"### Per-class AP: {diff.baseline_precision} → {diff.candidate_precision}",
        "",
        f"Flag = AP@50 or AP@50-95 dropped > {thr}% absolute.",
        "",
        "| class | AP@50 base | AP@50 cand | ΔAP@50 | AP@50-95 base | AP@50-95 cand | ΔAP@50-95 | flag |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in diff.deltas:
        flag = "🚩" if d.flagged(diff.threshold) else ""
        lines.append(
            f"| {d.class_name} | {pct(d.base_ap50)} | {pct(d.cand_ap50)} | {dpct(d.d_ap50)} "
            f"| {pct(d.base_ap5095)} | {pct(d.cand_ap5095)} | {dpct(d.d_ap5095)} | {flag} |"
        )
    lines += [
        "|—|—|—|—|—|—|—|—|",
        f"| **mAP** | {pct(diff.base_map50)} | {pct(diff.cand_map50)} | {dpct(diff.d_map50)} "
        f"| {pct(diff.base_map5095)} | {pct(diff.cand_map5095)} | {dpct(diff.d_map5095)} | |",
        "",
        f"**Flagged classes ({len(diff.flagged_classes)}):** "
        + (", ".join(d.class_name for d in diff.flagged_classes) or "none"),
    ]
    return "\n".join(lines)
