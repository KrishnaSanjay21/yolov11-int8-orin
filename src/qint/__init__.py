"""qint — INT8 PTQ toolkit for YOLOv11s on Jetson Orin NX.

Package layout by *where the code runs*:

  Host-testable (numpy only, CPU, tested in ``tests/``):
    qint.engine.interface   — InferenceEngine ABC + Detection dataclass
    qint.engine.stub        — StubEngine: deterministic fake engine for host tests
    qint.accuracy.metrics   — per-class AP, mAP@50, mAP@50-95 (COCO-style)
    qint.accuracy.diff      — FP32/FP16/INT8 per-class delta table + >2% flagging
    qint.calibration.stats  — activation histogram / min-max collection
    qint.calibration.entropy— KL-divergence (entropy) calibrator scale search
    qint.calibration.minmax — min-max calibrator scale
    qint.sensitivity        — per-layer quant-error (SQNR) ranking, top-5
    qint.precision          — precision-map planner (selective FP16 fallback)
    qint.plugin.dfl_reference — numpy reference for the fused DFL op

  Device-only (imported lazily; require TensorRT/CUDA/torch — RUN ON DEVICE):
    qint.engine.trt_runner       — real TensorRT engine runner
    qint.calibration.trt_calibrator — IInt8*Calibrator wrappers
    qint.surgery.onnx_export     — YOLOv11s -> ONNX

Nothing in the device-only modules is imported at package import time, so
``import qint`` works on a bare host.
"""

__all__ = ["engine", "accuracy", "calibration", "sensitivity", "precision", "plugin"]
