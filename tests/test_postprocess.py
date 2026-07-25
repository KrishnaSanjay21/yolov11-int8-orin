import numpy as np
import pytest

from qint.engine.postprocess import (
    xywh2xyxy, nms, scale_boxes_to_original, decode_predictions,
)


def test_xywh2xyxy():
    b = np.array([[10, 10, 4, 6]])  # cx,cy,w,h
    out = xywh2xyxy(b)
    assert list(out[0]) == [8, 7, 12, 13]


def test_nms_suppresses_overlap():
    boxes = np.array([
        [0, 0, 10, 10],
        [1, 1, 11, 11],   # heavy overlap w/ #0
        [100, 100, 110, 110],
    ], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_thr=0.5)
    assert 0 in keep and 2 in keep
    assert 1 not in keep


def test_nms_keeps_all_when_disjoint():
    boxes = np.array([[0, 0, 5, 5], [50, 50, 55, 55]], dtype=float)
    keep = nms(boxes, np.array([0.5, 0.4]), iou_thr=0.5)
    assert sorted(keep) == [0, 1]


def test_scale_boxes_inverts_letterbox():
    # 1280x640 (h,w) -> scale 0.5, pad_left=160. A box at original (200,100,300,200):
    src_hw = (1280, 640)
    orig = np.array([[200.0, 100.0, 300.0, 200.0]])
    # forward: *0.5 then + pad_left(160) on x
    lb = orig.copy() * 0.5
    lb[:, [0, 2]] += 160
    back = scale_boxes_to_original(lb, src_hw, imgsz=640)
    assert back == pytest.approx(orig, abs=1e-6)


def test_decode_predictions_recovers_a_box():
    nc = 3
    num_anchors = 5
    raw = np.zeros((4 + nc, num_anchors), dtype=np.float32)
    # anchor 0: a box centered at (320,320) size 40x40, class 1 score 0.9
    raw[:4, 0] = [320, 320, 40, 40]
    raw[4 + 1, 0] = 0.9
    dets = decode_predictions(raw, image_id=7, src_hw=(640, 640), num_classes=nc,
                              conf_thr=0.25, imgsz=640)
    assert len(dets) == 1
    d = dets[0]
    assert d.class_id == 1
    assert d.image_id == 7
    assert d.score == pytest.approx(0.9, abs=1e-6)
    assert d.box[0] == pytest.approx(300, abs=1e-4)
    assert d.box[2] == pytest.approx(340, abs=1e-4)


def test_decode_filters_low_conf():
    nc = 2
    raw = np.zeros((4 + nc, 3), dtype=np.float32)
    raw[:4, 0] = [100, 100, 20, 20]
    raw[4, 0] = 0.1  # below threshold
    assert decode_predictions(raw, 0, (640, 640), nc, conf_thr=0.25) == []
