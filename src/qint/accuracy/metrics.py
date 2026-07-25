"""COCO-style detection accuracy metrics — pure numpy, host-testable.

Implements per-class Average Precision at a single IoU threshold, aggregated into:
  * mAP@50      — AP at IoU=0.50, averaged over classes
  * mAP@50-95   — AP averaged over IoU in {0.50, 0.55, ..., 0.95}, then over classes

Matching follows the standard COCO greedy protocol:
  * predictions sorted by descending score,
  * each prediction matched to the highest-IoU *unmatched* GT of the same class
    that meets the IoU threshold,
  * matched GT becomes unavailable (one-to-one),
  * unmatched predictions are false positives; unmatched GT are false negatives.

AP integration defaults to COCO's 101-point interpolation. VOC-style all-point
interpolation is available via ``method="all"`` for cross-checking.

Only classes that have at least one ground-truth box count toward mAP (COCO behavior).
Per-class AP is still reported for every such class so degradations are visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..engine.interface import Detection

# COCO's 10 IoU thresholds: 0.50 : 0.95 : 0.05
COCO_IOU_THRESHOLDS: Tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. a:(M,4), b:(N,4) in xyxy -> (M,N)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])  # (M,N,2)
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def compute_ap(recall: np.ndarray, precision: np.ndarray, method: str = "101") -> float:
    """Area under the precision-recall curve.

    method="101": COCO 101-point interpolation (recall sampled at 0.00..1.00).
    method="all": VOC2010+ all-point interpolation (exact area under envelope).
    """
    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    # monotone-decreasing precision envelope
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mrec = np.concatenate([[0.0], recall, [1.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    if method == "101":
        recall_points = np.linspace(0.0, 1.0, 101)
        # for each sampled recall, take max precision at recall >= point
        ap = 0.0
        for rp in recall_points:
            idx = np.searchsorted(mrec, rp, side="left")
            p = mpre[idx] if idx < len(mpre) else 0.0
            ap += p
        return float(ap / len(recall_points))
    elif method == "all":
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    else:
        raise ValueError(f"unknown AP method: {method!r}")


def _ap_single_class_iou(
    preds: List[Detection],
    gts: List[Tuple[int, tuple]],  # (image_id, box)
    iou_thr: float,
    method: str,
) -> float:
    """AP for one class at one IoU threshold.

    ``preds`` : all predictions of this class (any image).
    ``gts``   : all GT of this class as (image_id, box).
    """
    n_gt = len(gts)
    if n_gt == 0:
        return float("nan")  # class with no GT is excluded from mAP by caller
    if len(preds) == 0:
        return 0.0

    # group GT boxes per image, track matched state
    gt_boxes_by_img: Dict[int, np.ndarray] = {}
    gt_matched_by_img: Dict[int, np.ndarray] = {}
    for img_id, box in gts:
        gt_boxes_by_img.setdefault(img_id, []).append(box)
    for img_id in list(gt_boxes_by_img):
        gt_boxes_by_img[img_id] = np.asarray(gt_boxes_by_img[img_id], dtype=np.float64)
        gt_matched_by_img[img_id] = np.zeros(len(gt_boxes_by_img[img_id]), dtype=bool)

    order = np.argsort([-p.score for p in preds], kind="stable")
    tp = np.zeros(len(preds), dtype=np.float64)
    fp = np.zeros(len(preds), dtype=np.float64)

    for rank, pi in enumerate(order):
        pred = preds[pi]
        gboxes = gt_boxes_by_img.get(pred.image_id)
        if gboxes is None or len(gboxes) == 0:
            fp[rank] = 1.0
            continue
        ious = box_iou(np.asarray([pred.box], dtype=np.float64), gboxes)[0]
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr and not gt_matched_by_img[pred.image_id][best]:
            tp[rank] = 1.0
            gt_matched_by_img[pred.image_id][best] = True
        else:
            fp[rank] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    return compute_ap(recall, precision, method=method)


@dataclass
class AccuracyResult:
    """Structured accuracy result for a single precision config."""

    precision: str
    num_classes: int
    iou_thresholds: Tuple[float, ...]
    # per_class_ap[class_id][iou_thr] = AP (nan for classes with no GT)
    per_class_ap: Dict[int, Dict[float, float]] = field(default_factory=dict)
    method: str = "101"

    # --- convenience accessors --------------------------------------------
    def ap50(self, class_id: int) -> float:
        return self.per_class_ap[class_id][0.50]

    def ap5095(self, class_id: int) -> float:
        vals = [v for v in self.per_class_ap[class_id].values() if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def classes_with_gt(self) -> List[int]:
        return sorted(
            c for c, d in self.per_class_ap.items()
            if any(not np.isnan(v) for v in d.values())
        )

    @property
    def map50(self) -> float:
        vals = [self.ap50(c) for c in self.classes_with_gt]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def map5095(self) -> float:
        vals = [self.ap5095(c) for c in self.classes_with_gt]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")


def evaluate(
    detections: Sequence[Detection],
    ground_truth: Sequence,  # objects with .image_id, .class_id, .box  (e.g. GTBox)
    num_classes: int,
    precision: str = "",
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
    method: str = "101",
) -> AccuracyResult:
    """Evaluate a flat list of detections against ground truth.

    Returns an :class:`AccuracyResult` with per-class AP at each IoU threshold.
    """
    iou_thresholds = tuple(round(float(t), 4) for t in iou_thresholds)
    preds_by_class: Dict[int, List[Detection]] = {c: [] for c in range(num_classes)}
    gts_by_class: Dict[int, List[Tuple[int, tuple]]] = {c: [] for c in range(num_classes)}
    for d in detections:
        preds_by_class.setdefault(d.class_id, []).append(d)
    for g in ground_truth:
        gts_by_class.setdefault(g.class_id, []).append((g.image_id, g.box))

    per_class_ap: Dict[int, Dict[float, float]] = {}
    for c in range(num_classes):
        per_class_ap[c] = {}
        for thr in iou_thresholds:
            per_class_ap[c][thr] = _ap_single_class_iou(
                preds_by_class.get(c, []), gts_by_class.get(c, []), thr, method
            )
    return AccuracyResult(
        precision=precision,
        num_classes=num_classes,
        iou_thresholds=iou_thresholds,
        per_class_ap=per_class_ap,
        method=method,
    )
