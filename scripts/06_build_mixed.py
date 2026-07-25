#!/usr/bin/env python3
"""Build an INT8 engine with selective FP16 fallback on sensitive layers.  # RUN ON DEVICE

Reads a PrecisionPlan (from 09_layer_sensitivity.py) and builds INT8+FP16-mixed engines,
including the ablation ladder (top-1, top-3, top-5 FP16) so DECISIONS.md can report the
accuracy-per-millisecond each fallback buys.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="weights/yolo11s.onnx")
    ap.add_argument("--plan", default="results/plan_fp16_top5.json")
    ap.add_argument("--kind", default="entropy")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--images", default="data/calib")
    ap.add_argument("--ladder", default="1,3,5", help="FP16 top-k values to build")
    args = ap.parse_args()

    from qint.precision import PrecisionPlan
    from qint.calibration.dataloader import CalibrationDataLoader, LoaderConfig
    from qint.calibration.trt_calibrator import make_calibrator
    from qint.engine.build import BuildConfig, build_engine

    with open(args.plan) as f:
        full_plan = PrecisionPlan.from_json(f.read())

    for k in (int(x) for x in args.ladder.split(",")):
        sub = PrecisionPlan(
            fp16_layers=full_plan.fp16_layers[:k],
            default_precision="int8",
            note=f"top-{k} FP16 fallback",
        )
        cache = f"calib_cache/{args.kind}_{args.size}.cache"  # reuse committed cache
        loader = CalibrationDataLoader(LoaderConfig(image_dir=args.images, size=args.size))
        calibrator = make_calibrator(args.kind, loader, cache_file=cache)
        engine = f"weights/yolo11s_int8_{args.kind}_{args.size}_fp16top{k}.engine"
        build_engine(BuildConfig(
            onnx_path=args.onnx, engine_path=engine, precision="int8",
            calibrator=calibrator, precision_plan=sub,
        ))
        print(f"[mixed] built {engine}  (FP16 layers: {sub.fp16_layers})")


if __name__ == "__main__":
    main()
