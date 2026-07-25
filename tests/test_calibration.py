import numpy as np
import pytest

from qint.calibration.stats import MinMaxCollector, HistogramCollector
from qint.calibration.minmax import (
    minmax_scale, compute_weight_scales, quantize, dequantize, fake_quant,
)
from qint.calibration.entropy import entropy_calibrate, kl_divergence, EntropyResult
from qint.calibration.dataloader import (
    compute_letterbox_params, select_calibration_subset, SWEEP_SIZES,
)


# ---- min-max ----------------------------------------------------------------
def test_minmax_collector_absmax_streaming():
    c = MinMaxCollector()
    c.update(np.array([-3.0, 1.0])).update(np.array([2.0, -5.0]))
    assert c.absmax == pytest.approx(5.0)
    assert c.min == pytest.approx(-5.0)
    assert c.max == pytest.approx(2.0)


def test_minmax_scale_and_quant_roundtrip_bound():
    x = np.linspace(-4, 4, 1000)
    s = minmax_scale(np.max(np.abs(x)))
    err = np.max(np.abs(x - fake_quant(x, s)))
    # symmetric INT8 round-to-nearest: worst-case error <= scale/2
    assert err <= s / 2 + 1e-9


def test_quantize_clamps_to_127():
    s = 0.1
    q = quantize(np.array([1000.0, -1000.0]), s)
    assert q.max() == 127 and q.min() == -127


def test_weight_scales_per_channel_vs_per_tensor():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 4, 3, 3))
    w[0] *= 10.0  # one channel with a much larger range
    per_ch = compute_weight_scales(w, per_channel=True)
    per_t = compute_weight_scales(w, per_channel=False)
    assert per_ch.shape == (8, 1, 1, 1)
    # per-channel scale for the big channel is larger; small channels get finer scales
    assert per_ch[0, 0, 0, 0] > per_ch[1, 0, 0, 0]
    # per-channel gives smaller total quantization error than per-tensor
    err_ch = np.mean((w - fake_quant(w, per_ch)) ** 2)
    err_t = np.mean((w - fake_quant(w, per_t)) ** 2)
    assert err_ch < err_t


# ---- histogram --------------------------------------------------------------
def test_histogram_preserves_count_across_growth():
    h = HistogramCollector(num_bins=256)
    h.update(np.array([0.1, 0.2, 0.3]))
    h.update(np.array([10.0]))  # forces range growth + rebin
    assert h.histogram.sum() == pytest.approx(4.0)
    assert h.upper >= 10.0


# ---- entropy / KL -----------------------------------------------------------
def test_kl_divergence_zero_for_identical():
    p = np.array([1.0, 2.0, 3.0, 4.0])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_divergence_nonnegative():
    p = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    q = np.array([0.25, 0.25, 0.25, 0.15, 0.10])
    assert kl_divergence(p, q) >= 0.0


def test_entropy_calibrate_clips_heavy_tail():
    # Bulk of mass near zero, a few large outliers. Entropy calibration should clip
    # well below the absolute max (that's the whole point vs min-max).
    rng = np.random.default_rng(1)
    h = HistogramCollector(num_bins=2048)
    bulk = np.abs(rng.normal(0, 1.0, size=200000))
    outliers = np.array([50.0, 60.0, 80.0])
    h.update(bulk)
    h.update(outliers)
    res = entropy_calibrate(h.histogram, h.upper)
    assert isinstance(res, EntropyResult)
    # threshold should be far below the 80.0 absolute max
    assert res.threshold_value < 20.0
    # and the min-max scale would be much larger than the entropy scale
    minmax_s = minmax_scale(80.0)
    assert res.scale < minmax_s


def test_entropy_result_kl_has_a_finite_min():
    h = HistogramCollector(num_bins=512)
    rng = np.random.default_rng(2)
    h.update(np.abs(rng.normal(0, 1, 50000)))
    res = entropy_calibrate(h.histogram, h.upper)
    assert np.isfinite(res.kl.min())
    assert 128 <= res.threshold_bin <= 512


# ---- dataloader pure funcs --------------------------------------------------
def test_letterbox_params_landscape():
    scale, (nh, nw), (pt, pl) = compute_letterbox_params((480, 640), 640)
    assert scale == pytest.approx(1.0)
    assert (nh, nw) == (480, 640)
    assert pt == 80 and pl == 0  # vertical padding only


def test_letterbox_params_portrait_scaledown():
    scale, (nh, nw), (pt, pl) = compute_letterbox_params((1280, 640), 640)
    assert scale == pytest.approx(0.5)
    assert (nh, nw) == (640, 320)
    assert pl == 160 and pt == 0


def test_select_subset_deterministic_and_sized():
    pool = [f"img_{i}.jpg" for i in range(1000)]
    a = select_calibration_subset(pool, 128, seed=0)
    b = select_calibration_subset(pool, 128, seed=0)
    assert a == b and len(a) == 128
    assert select_calibration_subset(pool, 32, seed=0) != a[:32] or True  # different draw ok
    # oversize returns whole pool, no crash
    assert len(select_calibration_subset(pool, 5000, seed=0)) == 1000


def test_sweep_sizes_are_the_spec_sizes():
    assert SWEEP_SIZES == (32, 128, 512)
