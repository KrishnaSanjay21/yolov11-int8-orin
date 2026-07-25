"""Minimal COCO-format ground-truth parsing — pure python, host-testable.

We only need image list + boxes + class names for evaluation, so we parse the parts
of a COCO ``instances`` dict we use rather than depend on pycocotools (which is awkward
on Jetson). ``load_coco_gt`` takes an already-loaded dict so it is trivially testable.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .engine.stub import GTBox


def load_coco_gt(coco: dict) -> Tuple[List[GTBox], List[str], Dict[int, Tuple[str, Tuple[int, int]]]]:
    """Parse a COCO instances dict.

    Returns
    -------
    gts : list[GTBox]
        Ground-truth boxes in xyxy, with ``image_id`` remapped to a dense 0..N-1 index
        and ``class_id`` remapped to a dense 0..C-1 index (contiguous category order).
    class_names : list[str]
        Dense class-id -> category name.
    images : dict[int, (file_name, (h, w))]
        Dense image index -> (file name, (height, width)) for the runner/letterbox.
    """
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_dense = {c["id"]: i for i, c in enumerate(cats)}
    class_names = [c["name"] for c in cats]

    imgs = sorted(coco["images"], key=lambda im: im["id"])
    img_id_to_dense = {im["id"]: i for i, im in enumerate(imgs)}
    images = {
        img_id_to_dense[im["id"]]: (im["file_name"], (int(im["height"]), int(im["width"])))
        for im in imgs
    }

    gts: List[GTBox] = []
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]  # COCO bbox is xywh (top-left)
        gts.append(
            GTBox(
                image_id=img_id_to_dense[ann["image_id"]],
                class_id=cat_id_to_dense[ann["category_id"]],
                box=(float(x), float(y), float(x + w), float(y + h)),
            )
        )
    return gts, class_names, images
