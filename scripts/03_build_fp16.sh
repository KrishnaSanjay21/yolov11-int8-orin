#!/usr/bin/env bash
# Build FP16 engine.  # RUN ON DEVICE
set -euo pipefail
mkdir -p weights
trtexec \
  --onnx=weights/yolo11s.onnx \
  --saveEngine=weights/yolo11s_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096 \
  --verbose 2>&1 | tee results/raw/build_fp16.log
echo "Built weights/yolo11s_fp16.engine"
