"""YOLO post-processing — pure numpy, host-testable.

Decoding raw model output into :class:`Detection` objects (confidence threshold,
xywh->xyxy, class-wise NMS, un-letterbox to original coords) is precision-agnostic, so
it lives here and is unit-tested on host. The device runner just feeds it raw arrays.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .interface import Detection
from ..calibration.dataloader import compute_letterbox_params


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """(cx,cy,w,h) -> (x1,y1,x2,y2)."""
    b = np.asarray(boxes, dtype=np.float64)
    out = np.empty_like(b)
    out[..., 0] = b[..., 0] - b[..., 2] / 2
    out[..., 1] = b[..., 1] - b[..., 3] / 2
    out[..., 2] = b[..., 0] + b[..., 2] / 2
    out[..., 3] = b[..., 1] + b[..., 3] / 2
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.7) -> List[int]:
    """Single-class greedy NMS. Returns kept indices (by descending score)."""
    from ..accuracy.metrics import box_iou

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    order = np.argsort(-scores, kind="stable")
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return keep


def scale_boxes_to_original(
    boxes_xyxy: np.ndarray, src_hw: Tuple[int, int], imgsz: int = 640
) -> np.ndarray:
    """Undo letterboxing: map boxes in the letterboxed square back to original pixels."""
    scale, _, (pad_top, pad_left) = compute_letterbox_params(src_hw, imgsz)
    b = np.asarray(boxes_xyxy, dtype=np.float64).copy()
    b[:, [0, 2]] -= pad_left
    b[:, [1, 3]] -= pad_top
    b /= scale
    h, w = src_hw
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, h)
    return b


def decode_predictions(
    raw: np.ndarray,
    image_id: int,
    src_hw: Tuple[int, int],
    num_classes: int,
    conf_thr: float = 0.25,
    iou_thr: float = 0.7,
    imgsz: int = 640,
    max_det: int = 300,
) -> List[Detection]:
    """Decode one image's raw YOLOv11 head output into Detections.

    ``raw`` is expected as (4 + num_classes, num_anchors) — the transposed YOLOv11
    output for a single image (box in xywh @ imgsz scale, then per-class scores).
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[0] == 4 + num_classes:
        pred = raw.T  # (anchors, 4+nc)
    elif raw.shape[1] == 4 + num_classes:
        pred = raw
    else:
        raise ValueError(f"cannot interpret raw shape {raw.shape} with nc={num_classes}")

    boxes = xywh2xyxy(pred[:, :4])
    cls_scores = pred[:, 4:]
    class_ids = np.argmax(cls_scores, axis=1)
    confs = cls_scores[np.arange(len(cls_scores)), class_ids]

    keep_mask = confs >= conf_thr
    boxes, confs, class_ids = boxes[keep_mask], confs[keep_mask], class_ids[keep_mask]
    if len(boxes) == 0:
        return []

    boxes = scale_boxes_to_original(boxes, src_hw, imgsz)

    dets: List[Detection] = []
    for c in np.unique(class_ids):
        idx = np.where(class_ids == c)[0]
        keep = nms(boxes[idx], confs[idx], iou_thr)
        for k in keep:
            j = idx[k]
            dets.append(Detection(image_id=image_id, class_id=int(c),
                                  score=float(confs[j]), box=tuple(boxes[j])))
    dets.sort(key=lambda d: -d.score)
    return dets[:max_det]
