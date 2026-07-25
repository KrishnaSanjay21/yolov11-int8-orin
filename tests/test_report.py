import math
import numpy as np

from qint.report import (
    accuracy_to_dict, accuracy_from_dict, BenchmarkRow, render_benchmarks_table,
)
from qint.accuracy.metrics import evaluate
from qint.engine.stub import StubEngine, GTBox


def _res():
    gts = [GTBox(0, 0, (10, 10, 20, 20)), GTBox(0, 1, (30, 30, 40, 40))]
    eng = StubEngine(gts, num_classes=3, precision="fp32")  # class 2 has no GT -> nan
    dets = eng.infer_dataset(None, [0])
    return evaluate(dets, gts, num_classes=3, precision="fp32")


def test_accuracy_roundtrip_preserves_nan_and_values():
    res = _res()
    d = accuracy_to_dict(res)
    res2 = accuracy_from_dict(d)
    assert res2.precision == "fp32"
    assert res2.map50 == res.map50
    # class 2 had no GT -> nan must survive the JSON round-trip (None <-> nan)
    assert math.isnan(res2.ap50(2))
    assert res2.classes_with_gt == res.classes_with_gt


def test_benchmark_row_partial_renders_with_dash():
    rows = [
        BenchmarkRow("fp32", latency_mean_ms=12.3, map50=0.55),
        BenchmarkRow("int8"),  # nothing measured yet
    ]
    table = render_benchmarks_table(rows)
    assert "fp32" in table and "int8" in table
    assert "—" in table  # unmeasured cells show em dash, not a fake 0
    assert "12.30" in table


def test_benchmark_row_dict_roundtrip():
    r = BenchmarkRow("int8", latency_mean_ms=8.1, power_w=9.4, map5095=0.42)
    r2 = BenchmarkRow.from_dict(r.to_dict())
    assert r2 == r


def test_build_rows_merges_accuracy_and_bench():
    from qint.report import build_rows, accuracy_to_dict
    res = _res()  # precision "fp32"
    acc = {"fp32": accuracy_to_dict(res)}
    bench = {
        "fp32": {"latency_mean_ms": 12.0, "throughput_fps": 80.0},
        "int8": {"latency_mean_ms": 6.0},  # no accuracy measured yet
    }
    rows = build_rows(bench, acc, order=["fp32", "int8"])
    assert [r.config for r in rows] == ["fp32", "int8"]
    assert rows[0].latency_mean_ms == 12.0
    assert rows[0].map50 == res.map50           # merged from accuracy
    assert rows[1].map50 is None                # honest: not measured
    assert rows[1].latency_mean_ms == 6.0
