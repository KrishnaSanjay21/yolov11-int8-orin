#!/usr/bin/env bash
# Full INT8 build matrix.  # RUN ON DEVICE
#   - calibrator sweep:   entropy vs minmax
#   - calib-size sweep:   32 / 128 / 512
#   - weight granularity: per-channel (default) vs per-tensor
# Produces the engines + committed calibration caches consumed by 07_run_accuracy.py.
set -euo pipefail

for kind in entropy minmax; do
  for size in 32 128 512; do
    echo "=== INT8 build: kind=$kind size=$size (per-channel weights) ==="
    python3 scripts/04_calibrate_int8.py --kind "$kind" --size "$size"
  done
done

# Per-tensor vs per-channel A/B — do it at the best calib size only to bound cost.
# NOTE on "where TRT permits": TensorRT quantizes *activations* per-tensor always; the
# per-tensor/per-channel choice it exposes is for *conv weights*. --per-tensor-weights
# forces the weaker (per-tensor) weight scheme so the accuracy delta is attributable.
echo "=== INT8 build: entropy 512, per-TENSOR weights (A/B against per-channel) ==="
python3 scripts/04_calibrate_int8.py --kind entropy --size 512 --per-tensor-weights

echo "All INT8 engines + calibration caches built. Caches in calib_cache/ (commit them)."
