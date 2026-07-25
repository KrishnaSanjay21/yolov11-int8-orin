#!/usr/bin/env bash
# Record + lock the Jetson environment for reproducible numbers.  # RUN ON DEVICE
# Source this at the top of a benchmarking session:  source scripts/00_env.sh
set -euo pipefail

echo "# RUN ON DEVICE — capturing environment"

# Pin the perf mode. MAXN (id 0) + jetson_clocks gives repeatable latency/throughput.
# Record which nvpmodel you used in BENCHMARKS.md — power numbers are meaningless without it.
sudo nvpmodel -m 0 || echo "WARN: nvpmodel failed (need sudo / correct model id)"
sudo jetson_clocks || echo "WARN: jetson_clocks failed"

mkdir -p results/raw
{
  echo "captured_utc: $(date -u +%FT%TZ)"
  echo "nvpmodel: $(sudo nvpmodel -q 2>/dev/null | tr '\n' ' ')"
  echo "l4t: $(cat /etc/nv_tegra_release 2>/dev/null | head -1)"
  echo "jetpack_trt: $(dpkg -l 2>/dev/null | grep -E 'nvinfer' | awk '{print $2"="$3}' | tr '\n' ' ')"
  echo "cuda: $(nvcc --version 2>/dev/null | grep release || echo n/a)"
  echo "python: $(python3 --version 2>&1)"
  echo "torch: $(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || echo n/a)"
  echo "trt_py: $(python3 -c 'import tensorrt as t;print(t.__version__)' 2>/dev/null || echo n/a)"
} | tee results/raw/environment.txt

echo "Environment recorded to results/raw/environment.txt — copy into BENCHMARKS.md."
