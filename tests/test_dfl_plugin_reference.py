"""Host-side numerical validation of the fused DFL op.

DONE-WHEN: "Plugin correctness test passes on host against PyTorch reference."

On host we validate the fused, kernel-faithful implementation against the plain
unfused reference with an explicit max-abs and relative error bound (not eyeballed).
When a working PyTorch is importable we ALSO cross-check against the torch reference;
otherwise that assertion is skipped (torch is broken/absent on some hosts) but the
numpy-vs-numpy correctness check still runs and must pass.

The SAME error metrics are emitted by ``scripts/validate_plugin.py`` on device to
compare the actual CUDA plugin output against these references.
"""
import numpy as np
import pytest

from qint.plugin.dfl_reference import (
    dfl_unfused_reference, dfl_fused_reference, dfl_torch_reference,
    TORCH_AVAILABLE, REG_MAX,
)


def _rand_input(seed=0, n=2, a=100, reg_max=REG_MAX):
    rng = np.random.default_rng(seed)
    # logit-scale values, some large to stress softmax stability
    return rng.normal(0, 6.0, size=(n, 4 * reg_max, a)).astype(np.float32)


def _errors(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    max_abs = float(np.max(np.abs(a - b)))
    denom = np.maximum(np.abs(b), 1e-8)
    max_rel = float(np.max(np.abs(a - b) / denom))
    return max_abs, max_rel


def test_fused_matches_unfused_reference():
    x = _rand_input(seed=1)
    fused = dfl_fused_reference(x)
    unfused = dfl_unfused_reference(x)
    assert fused.shape == (x.shape[0], 4, x.shape[2])
    max_abs, max_rel = _errors(fused, unfused)
    # fused uses float32 accumulation vs float64 reference; still very tight
    assert max_abs < 1e-3, f"max abs error too high: {max_abs}"
    assert max_rel < 1e-3, f"max rel error too high: {max_rel}"


def test_output_range_is_within_bins():
    # expectation of a softmax over bins 0..reg_max-1 must lie in [0, reg_max-1]
    x = _rand_input(seed=2)
    out = dfl_fused_reference(x)
    assert out.min() >= 0.0
    assert out.max() <= REG_MAX - 1 + 1e-4


def test_saturated_logits_pick_the_argmax_bin():
    # if bin k dominates, expectation ~ k
    x = np.full((1, 4 * REG_MAX, 1), -50.0, dtype=np.float32)
    for side in range(4):
        x[0, side * REG_MAX + (side + 3), 0] = 50.0  # dominate bin (side+3)
    out = dfl_fused_reference(x)
    for side in range(4):
        assert out[0, side, 0] == pytest.approx(side + 3, abs=1e-3)


@pytest.mark.torch
def test_fused_matches_torch_reference():
    if not TORCH_AVAILABLE:
        pytest.skip("torch not importable/loadable on this host")
    x = _rand_input(seed=3)
    fused = dfl_fused_reference(x)
    torch_out = dfl_torch_reference(x)
    assert torch_out is not None
    max_abs, max_rel = _errors(fused, torch_out)
    assert max_abs < 1e-3
    assert max_rel < 1e-3
