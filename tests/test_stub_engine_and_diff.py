import numpy as np
import pytest

from qint.engine.stub import StubEngine, GTBox, DegradationProfile
from qint.accuracy.metrics import evaluate
from qint.accuracy.diff import diff_precisions, format_delta_table


def _grid_gt(n_images=8, n_classes=3):
    gts = []
    for img in range(n_images):
        for c in range(n_classes):
            x = 10 + 30 * c
            y = 10 + 25 * img
            gts.append(GTBox(image_id=img, class_id=c, box=(x, y, x + 20, y + 20)))
    return gts


def test_stub_perfect_reproduces_gt():
    gts = _grid_gt()
    eng = StubEngine(gts, num_classes=3, precision="fp32")
    ids = sorted({g.image_id for g in gts})
    dets = eng.infer_dataset(np.zeros((len(ids), 3, 8, 8), np.float32), ids)
    res = evaluate(dets, gts, num_classes=3, precision="fp32")
    assert res.map50 == pytest.approx(1.0)
    assert res.map5095 == pytest.approx(1.0)


def test_stub_is_deterministic():
    gts = _grid_gt()
    prof = DegradationProfile(drop_prob={0: 0.5}, loc_jitter={1: 0.3})
    ids = sorted({g.image_id for g in gts})
    e1 = StubEngine(gts, 3, "int8", profile=prof, seed=7)
    e2 = StubEngine(gts, 3, "int8", profile=prof, seed=7)
    d1 = e1.infer_dataset(None, ids)
    d2 = e2.infer_dataset(None, ids)
    assert len(d1) == len(d2)
    for a, b in zip(d1, d2):
        assert a == b


def test_diff_flags_degraded_class_only():
    gts = _grid_gt(n_images=12, n_classes=3)
    ids = sorted({g.image_id for g in gts})
    imgs = np.zeros((len(ids), 3, 8, 8), np.float32)

    fp32 = StubEngine(gts, 3, "fp32")
    # INT8: class 2 degraded hard (drops + jitter), classes 0/1 fine
    prof = DegradationProfile(
        drop_prob={2: 0.6},
        loc_jitter={2: 0.5},
        score_bias={2: -0.4},
    )
    int8 = StubEngine(gts, 3, "int8", profile=prof, seed=3)

    r_fp32 = evaluate(fp32.infer_dataset(imgs, ids), gts, 3, "fp32")
    r_int8 = evaluate(int8.infer_dataset(imgs, ids), gts, 3, "int8")

    diff = diff_precisions(r_fp32, r_int8, class_names=["a", "b", "c"], threshold=0.02)
    flagged = {d.class_id for d in diff.flagged_classes}
    assert 2 in flagged
    assert 0 not in flagged and 1 not in flagged
    # table renders and mentions the flagged class
    table = format_delta_table(diff)
    assert "🚩" in table
    assert "class" in table


def test_diff_no_flags_when_identical():
    gts = _grid_gt()
    ids = sorted({g.image_id for g in gts})
    eng = StubEngine(gts, 3, "fp32")
    r = evaluate(eng.infer_dataset(None, ids), gts, 3, "fp32")
    diff = diff_precisions(r, r, threshold=0.02)
    assert diff.flagged_classes == []
    assert diff.d_map50 == pytest.approx(0.0)
