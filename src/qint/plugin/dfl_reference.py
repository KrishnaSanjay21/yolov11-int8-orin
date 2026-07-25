"""Reference implementations of the fused DFL op — the plugin's ground truth.

The custom TensorRT plugin (``src/qint/plugin/dfl_plugin/``) fuses YOLOv11's
Distribution-Focal-Loss decode into a single kernel:

    input  x : (N, 4 * reg_max, A)   raw logits for 4 box sides over ``reg_max`` bins
    step 1   : reshape -> (N, 4, reg_max, A)
    step 2   : softmax over the reg_max axis
    step 3   : expectation E[j] = sum_j softmax_j * j,  j = 0..reg_max-1
    output   : (N, 4, A)             expected distance per side

Three references are provided so plugin correctness can be checked with a real
max-abs / relative error, not by eye:

  * :func:`dfl_unfused_reference` — the plain, obviously-correct composition
    (explicit softmax, then a weighted sum). This is *the reference op*.
  * :func:`dfl_fused_reference` — a single-pass, numerically-stable implementation
    that mirrors exactly what the CUDA kernel does (max-subtract, one accumulation
    loop). The host test asserts it matches the unfused reference; on device the CUDA
    kernel output is asserted against THIS.
  * :func:`dfl_torch_reference` — optional PyTorch equivalent used as an independent
    third opinion when torch is importable. Loads lazily and tolerates a broken torch
    install (returns None) so the host suite stays green without torch.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

REG_MAX = 16  # YOLOv11 uses 16 DFL bins


def dfl_unfused_reference(x: np.ndarray, reg_max: int = REG_MAX) -> np.ndarray:
    """The plain reference: explicit softmax over bins, then weighted sum.

    x : (N, 4*reg_max, A) -> returns (N, 4, A).
    """
    x = np.asarray(x, dtype=np.float64)
    n, c, a = x.shape
    if c != 4 * reg_max:
        raise ValueError(f"expected channel dim {4*reg_max}, got {c}")
    xr = x.reshape(n, 4, reg_max, a)
    m = xr.max(axis=2, keepdims=True)
    e = np.exp(xr - m)
    soft = e / e.sum(axis=2, keepdims=True)
    j = np.arange(reg_max, dtype=np.float64).reshape(1, 1, reg_max, 1)
    return (soft * j).sum(axis=2)


def dfl_fused_reference(x: np.ndarray, reg_max: int = REG_MAX) -> np.ndarray:
    """Single-pass, kernel-faithful implementation (what the .cu kernel computes).

    For each (n, side, a): find max over bins, accumulate sum(exp) and sum(exp*j) in
    one pass, output = weighted_sum / exp_sum. Uses float32 accumulation to match the
    CUDA kernel's precision as closely as is reasonable on host.
    """
    x = np.asarray(x, dtype=np.float32)
    n, c, a = x.shape
    if c != 4 * reg_max:
        raise ValueError(f"expected channel dim {4*reg_max}, got {c}")
    out = np.zeros((n, 4, a), dtype=np.float32)
    for ni in range(n):
        for side in range(4):
            base = side * reg_max
            block = x[ni, base:base + reg_max, :]           # (reg_max, A)
            m = block.max(axis=0)                            # (A,)
            e = np.exp(block - m[None, :]).astype(np.float32)  # (reg_max, A)
            esum = e.sum(axis=0)                             # (A,)
            j = np.arange(reg_max, dtype=np.float32)[:, None]
            wsum = (e * j).sum(axis=0)                       # (A,)
            out[ni, side, :] = wsum / esum
    return out


def dfl_torch_reference(x: np.ndarray, reg_max: int = REG_MAX) -> Optional[np.ndarray]:
    """PyTorch reference; returns None if torch cannot be imported/loaded on host.

    Mirrors ultralytics' DFL: softmax over the bin axis then a fixed 1x1 conv whose
    weights are ``arange(reg_max)`` — algebraically identical to the expectation.
    """
    import faulthandler
    # A broken torch install can fail its native DLL load with a fatal-looking
    # (but catchable) error; temporarily silence faulthandler's native dump so a
    # broken optional dependency doesn't spew a scary traceback into the test log.
    _fh_was_enabled = faulthandler.is_enabled()
    faulthandler.disable()
    try:
        import torch  # noqa: F401
        import torch.nn.functional as F
    except Exception:
        # ImportError, or (as on this dev host) an OSError from a broken DLL load.
        return None
    finally:
        if _fh_was_enabled:
            faulthandler.enable()
    t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    n, c, a = t.shape
    tr = t.view(n, 4, reg_max, a)
    soft = F.softmax(tr, dim=2)
    j = torch.arange(reg_max, dtype=torch.float32).view(1, 1, reg_max, 1)
    out = (soft * j).sum(dim=2)
    return out.detach().cpu().numpy()


TORCH_AVAILABLE = dfl_torch_reference(np.zeros((1, 4 * REG_MAX, 1), dtype=np.float32)) is not None
