"""TensorRT INT8 calibrators (entropy + minmax) with a committable cache.  # RUN ON DEVICE

Wraps a :class:`CalibrationDataLoader` as a TensorRT ``IInt8Calibrator``. Two variants,
selected by name so ``04_calibrate_int8.py`` can build BOTH and we can compare:

  * ``EntropyCalibrator``  -> ``trt.IInt8EntropyCalibrator2``  (KL / histogram based)
  * ``MinMaxCalibrator``   -> ``trt.IInt8MinMaxCalibrator``    (absolute-max based)

The calibration cache is written to (and, if present, read from) ``cache_file``. The
cache is COMMITTED to the repo (spec requirement) so an INT8 engine can be rebuilt
byte-for-byte without re-streaming images. Cache filenames encode calibrator + set size,
e.g. ``calib_cache/entropy_512.cache``.

Requires TensorRT + pycuda; imported lazily so the host suite never touches this file.
"""
from __future__ import annotations

import os
from typing import Optional

# NB: no top-level tensorrt/pycuda import — see module docstring.


def _make_base(trt):
    """Return the two calibrator base classes bound to the imported trt module."""

    class _MixinCache:
        """Shared batching + cache logic; combined with a trt base class below."""

        def _init_cache(self, loader, cache_file, input_name):
            import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
            import pycuda.driver as cuda

            self._loader = loader
            self._it = iter(loader)
            self._cache_file = cache_file
            self._input_name = input_name
            self._cuda = cuda
            self._device_input = None
            self._batch_shape = None

        def get_batch_size(self):
            return self._loader.cfg.batch

        def get_batch(self, names):
            try:
                arr = next(self._it)
            except StopIteration:
                return None
            arr = arr.astype("float32").ravel()
            if self._device_input is None:
                self._device_input = self._cuda.mem_alloc(arr.nbytes)
            self._cuda.memcpy_htod(self._device_input, arr)
            return [int(self._device_input)]

        def read_calibration_cache(self):
            if self._cache_file and os.path.exists(self._cache_file):
                with open(self._cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            if self._cache_file:
                os.makedirs(os.path.dirname(self._cache_file) or ".", exist_ok=True)
                with open(self._cache_file, "wb") as f:
                    f.write(cache)

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2, _MixinCache):
        def __init__(self, loader, cache_file, input_name="images"):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self._init_cache(loader, cache_file, input_name)

    class MinMaxCalibrator(trt.IInt8MinMaxCalibrator, _MixinCache):
        def __init__(self, loader, cache_file, input_name="images"):
            trt.IInt8MinMaxCalibrator.__init__(self)
            self._init_cache(loader, cache_file, input_name)

    return EntropyCalibrator, MinMaxCalibrator


def make_calibrator(kind: str, loader, cache_file: str, input_name: str = "images"):
    """Factory. ``kind`` in {"entropy", "minmax"}. RUN ON DEVICE."""
    import tensorrt as trt  # RUN ON DEVICE

    Entropy, MinMax = _make_base(trt)
    kind = kind.lower()
    if kind == "entropy":
        return Entropy(loader, cache_file, input_name)
    if kind == "minmax":
        return MinMax(loader, cache_file, input_name)
    raise ValueError(f"unknown calibrator kind: {kind!r} (expected 'entropy'|'minmax')")


def parse_cache_scales(cache_file: str) -> dict:
    """Parse a TensorRT calibration cache into {tensor_name: scale_float}.

    Cache format is text: a header line then ``<tensor>: <hex_float32>`` per tensor,
    where the hex encodes the FP32 scale. Handy for the host-side sanity check that
    committed caches are non-degenerate. Pure text parsing -> safe to call anywhere.
    """
    import struct

    scales: dict = {}
    if not os.path.exists(cache_file):
        return scales
    with open(cache_file, "r", errors="ignore") as f:
        for line in f:
            if ":" not in line:
                continue
            name, _, hexval = line.strip().partition(":")
            hexval = hexval.strip()
            try:
                raw = int(hexval, 16)
                scale = struct.unpack("<f", struct.pack("<I", raw & 0xFFFFFFFF))[0]
                scales[name.strip()] = float(scale)
            except (ValueError, struct.error):
                continue
    return scales
