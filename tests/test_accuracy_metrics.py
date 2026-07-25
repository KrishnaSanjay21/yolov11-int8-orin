import numpy as np
import pytest

from qint.accuracy.metrics import (
    box_iou, compute_ap, evaluate, COCO_IOU_THRESHOLDS,
)
from qint.engine.stub import GTBox
from qint.engine.interface import Detection


def test_box_iou_identity_and_disjoint():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)
    b = np.array([[100, 100, 110, 110]], dtype=float)
    assert box_iou(a, b)[0, 0] == pytest.approx(0.0)


def test_box_iou_half_overlap():
    a = np.array([[0, 0, 2, 2]], dtype=float)      # area 4
    b = np.array([[1, 0, 3, 2]], dtype=float)      # area 4, overlap area 2
    # inter=2, union=6 -> 1/3
    assert box_iou(a, b)[0, 0] == pytest.approx(1.0 / 3.0)


def test_box_iou_empty():
    assert box_iou(np.zeros((0, 4)), np.ones((3, 4))).shape == (0, 3)


def test_compute_ap_perfect_is_one():
    # a single TP at recall 1, precision 1 -> AP 1.0
    recall = np.array([1.0])
    precision = np.array([1.0])
    assert compute_ap(recall, precision, method="101") == pytest.approx(1.0)
    assert compute_ap(recall, precision, method="all") == pytest.approx(1.0)


def test_compute_ap_half_precision():
    # PR curve flat at precision 0.5 out to recall 1 -> AP 0.5
    recall = np.array([0.5, 1.0])
    precision = np.array([0.5, 0.5])
    assert compute_ap(recall, precision, method="all") == pytest.approx(0.5, abs=1e-9)
    assert compute_ap(recall, precision, method="101") == pytest.approx(0.5, abs=0.01)


def _make_gt(n_per_class=3, n_classes=2):
    gts = []
    for img in range(n_per_class):
        for c in range(n_classes):
            x = 10 + 20 * img + 100 * c
            gts.append(GTBox(image_id=img, class_id=c, box=(x, x, x + 10, x + 10)))
    return gts


def test_evaluate_perfect_predictions_gives_ap1():
    gts = _make_gt()
    # perfect detections: exact boxes, high score
    dets = [Detection(g.image_id, g.class_id, 0.99, g.box) for g in gts]
    res = evaluate(dets, gts, num_classes=2, precision="fp32")
    assert res.map50 == pytest.approx(1.0)
    assert res.map5095 == pytest.approx(1.0)
    for c in [0, 1]:
        assert res.ap50(c) == pytest.approx(1.0)


def test_evaluate_no_predictions_gives_zero():
    gts = _make_gt()
    res = evaluate([], gts, num_classes=2)
    assert res.map50 == pytest.approx(0.0)
    for c in [0, 1]:
        assert res.ap50(c) == pytest.approx(0.0)


def test_class_without_gt_is_nan_and_excluded():
    gts = _make_gt(n_classes=1)  # only class 0 has GT
    dets = [Detection(g.image_id, g.class_id, 0.9, g.box) for g in gts]
    res = evaluate(dets, gts, num_classes=2)
    assert np.isnan(res.ap50(1))          # class 1 has no GT
    assert res.classes_with_gt == [0]     # excluded from mAP
    assert res.map50 == pytest.approx(1.0)


def test_localization_error_hurts_ap5095_more_than_ap50():
    gts = _make_gt(n_classes=1)
    # shift boxes by 1px on a 10px box -> IoU=81/119≈0.68: a TP at 0.5/0.65 but not 0.7+
    shifted = []
    for g in gts:
        x1, y1, x2, y2 = g.box
        shifted.append(Detection(g.image_id, g.class_id, 0.9, (x1 + 1, y1 + 1, x2 + 1, y2 + 1)))
    res = evaluate(shifted, gts, num_classes=1)
    assert res.ap50(0) > res.ap5095(0)
