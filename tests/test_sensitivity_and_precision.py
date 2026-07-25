import numpy as np
import pytest

from qint.sensitivity import (
    sqnr_db, score_layer, rank_layers, top_k_sensitive, LayerSensitivity,
)
from qint.precision import (
    build_fallback_plan, PrecisionPlan, accuracy_per_ms, AccuracyPerMs,
)


def test_sqnr_infinite_when_lossless():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isinf(sqnr_db(x, x))


def test_sqnr_lower_for_bigger_noise():
    x = np.linspace(-1, 1, 100)
    small = sqnr_db(x, x + 0.001)
    big = sqnr_db(x, x + 0.1)
    assert small > big  # more noise -> lower SQNR


def test_score_layer_wider_range_is_more_sensitive():
    rng = np.random.default_rng(0)
    narrow = rng.normal(0, 1, 5000)
    wide = np.concatenate([rng.normal(0, 1, 5000), [1000.0]])  # outlier widens scale
    s_narrow = score_layer("narrow", narrow)
    s_wide = score_layer("wide", wide)
    # the outlier inflates the min-max scale -> worse SQNR for the bulk
    assert s_wide.sqnr_db < s_narrow.sqnr_db


def test_ranking_and_top_k():
    layers = [
        LayerSensitivity("a", sqnr_db=40, mse=1e-4, energy=1.0, scale=0.1),
        LayerSensitivity("b", sqnr_db=10, mse=1e-1, energy=1.0, scale=0.1),
        LayerSensitivity("c", sqnr_db=25, mse=1e-2, energy=1.0, scale=0.1),
    ]
    ranked = rank_layers(layers)
    assert [l.name for l in ranked] == ["b", "c", "a"]  # ascending SQNR
    assert [l.name for l in top_k_sensitive(layers, 2)] == ["b", "c"]


def test_build_fallback_plan_picks_worst_layers():
    layers = [
        LayerSensitivity(f"layer{i}", sqnr_db=float(i), mse=0.0, energy=1.0, scale=0.1)
        for i in range(10)
    ]
    plan = build_fallback_plan(layers, k=3)
    assert plan.fp16_layers == ["layer0", "layer1", "layer2"]
    assert plan.default_precision == "int8"
    assert plan.precision_of("layer0") == "fp16"
    assert plan.precision_of("layer9") == "int8"


def test_precision_plan_json_roundtrip():
    plan = PrecisionPlan(fp16_layers=["x", "y"], note="test")
    plan2 = PrecisionPlan.from_json(plan.to_json())
    assert plan2.fp16_layers == ["x", "y"]
    assert plan2.note == "test"
    directives = plan.to_trt_directives()
    assert directives[0] == {"layer": "x", "precision": "fp16", "set_output_type": True}


def test_accuracy_per_ms_computation():
    configs = {
        "int8+fp16_top5": {"map5095": 0.512, "latency_ms": 8.2},
        "int8+fp16_top1": {"map5095": 0.505, "latency_ms": 7.6},
    }
    out = {a.config: a for a in accuracy_per_ms(0.498, 7.4, configs)}
    top5 = out["int8+fp16_top5"]
    assert top5.d_map5095 == pytest.approx(0.014)
    assert top5.d_latency_ms == pytest.approx(0.8)
    # 1.4 mAP points over 0.8 ms
    assert top5.map_per_ms == pytest.approx(1.4 / 0.8, rel=1e-6)


def test_accuracy_per_ms_free_gain_is_inf():
    a = AccuracyPerMs("free", d_map5095=0.01, d_latency_ms=0.0)
    assert np.isinf(a.map_per_ms)
