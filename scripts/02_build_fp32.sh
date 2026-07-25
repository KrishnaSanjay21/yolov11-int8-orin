#!/usr/bin/env bash
# Build FP32 baseline engine.  # RUN ON DEVICE
set -euo pipefail
mkdir -p weights
# trtexec is the reference path; the python builder (qint.engine.build) produces an
# identical engine and is what the mixed-precision/INT8 scripts use.
trtexec \
  --onnx=weights/yolo11s.onnx \
  --saveEngine=weights/yolo11s_fp32.engine \
  --memPoolSize=workspace:4096 \
  --verbose 2>&1 | tee results/raw/build_fp32.log
echo "Built weights/yolo11s_fp32.engine"
