#!/usr/bin/env bash
# Export YOLOv11s -> ONNX.  # RUN ON DEVICE (needs ultralytics + torch)
set -euo pipefail
mkdir -p weights

# Full graph (with DFL decode) — used for FP32/FP16/INT8 baselines.
python3 -m qint.surgery.onnx_export --weights yolo11s.pt --imgsz 640 --opset 17 \
  --out weights/yolo11s.onnx

echo "Exported weights/yolo11s.onnx"
echo "NOTE: for the DFL-plugin A/B, export a no-DFL variant and graft the DFL node"
echo "      (see DECISIONS.md 'Plugin' section)."
