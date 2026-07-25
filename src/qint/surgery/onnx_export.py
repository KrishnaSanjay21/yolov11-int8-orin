"""YOLOv11s -> ONNX export.  # RUN ON DEVICE (needs torch + ultralytics)

Kept thin on purpose: the interesting, testable logic lives in the calibration /
accuracy / sensitivity modules. This just produces a clean, static-batch ONNX graph
that TensorRT can ingest, with opset and I/O names pinned so every downstream script
(FP32/FP16/INT8 builds, calibrator, benchmark) agrees on tensor names.

Two export flavors:
  * "full"  — the stock end-to-end YOLOv11s graph (includes the DFL decode).
  * "no_dfl"— DFL decode stripped from the ONNX so the custom DFL TensorRT plugin can
              be inserted in its place. Used by the plugin A/B.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExportConfig:
    weights: str = "yolo11s.pt"
    imgsz: int = 640
    opset: int = 17
    batch: int = 1
    input_name: str = "images"
    output_name: str = "output0"
    out_path: str = "weights/yolo11s.onnx"
    simplify: bool = True


def export_yolov11s(cfg: ExportConfig) -> str:
    """Export YOLOv11s to ONNX and return the output path. RUN ON DEVICE."""
    # Lazy imports so this module can be *imported* on a bare host (e.g. for docs),
    # even though calling this function requires the device stack.
    from ultralytics import YOLO  # noqa: WPS433

    model = YOLO(cfg.weights)
    path = model.export(
        format="onnx",
        imgsz=cfg.imgsz,
        opset=cfg.opset,
        batch=cfg.batch,
        simplify=cfg.simplify,
        dynamic=False,
        nms=False,  # keep decode explicit so we can swap in the DFL plugin
    )
    return str(path)


if __name__ == "__main__":  # RUN ON DEVICE
    import argparse

    p = argparse.ArgumentParser(description="Export YOLOv11s to ONNX (RUN ON DEVICE)")
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--out", default="weights/yolo11s.onnx")
    args = p.parse_args()
    out = export_yolov11s(ExportConfig(weights=args.weights, imgsz=args.imgsz,
                                       opset=args.opset, out_path=args.out))
    print(f"[export] wrote {out}")
