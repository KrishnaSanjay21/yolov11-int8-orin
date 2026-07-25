# Intentionally does NOT import onnx_export at package import time:
# onnx_export requires torch + ultralytics (device-only). Import it explicitly
# (`from qint.surgery.onnx_export import export_yolov11s`) inside device scripts.
