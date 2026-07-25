#!/usr/bin/env python3
"""Build an INT8 engine with a chosen calibrator + calibration-set size.  # RUN ON DEVICE

Writes/reuses a COMMITTED calibration cache named calib_cache/<kind>_<size>.cache so
INT8 engines are reproducible without re-streaming images.

Examples
--------
  python3 scripts/04_calibrate_int8.py --kind entropy --size 512
  python3 scripts/04_calibrate_int8.py --kind minmax  --size 128 --per-tensor-weights
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="weights/yolo11s.onnx")
    ap.add_argument("--kind", choices=["entropy", "minmax"], default="entropy")
    ap.add_argument("--size", type=int, default=512, help="calib set size (32/128/512)")
    ap.add_argument("--images", default="data/calib")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--per-tensor-weights", action="store_true",
                    help="force per-tensor weight quant (default: per-channel)")
    ap.add_argument("--engine", default=None)
    args = ap.parse_args()

    from qint.calibration.dataloader import CalibrationDataLoader, LoaderConfig
    from qint.calibration.trt_calibrator import make_calibrator, parse_cache_scales
    from qint.engine.build import BuildConfig, build_engine

    os.makedirs("calib_cache", exist_ok=True)
    cache = f"calib_cache/{args.kind}_{args.size}.cache"
    wtag = "pertensor" if args.per_tensor_weights else "perchannel"
    engine = args.engine or f"weights/yolo11s_int8_{args.kind}_{args.size}_{wtag}.engine"

    loader = CalibrationDataLoader(LoaderConfig(
        image_dir=args.images, batch=args.batch, size=args.size))
    print(f"[calib] {args.kind} calibrator, {len(loader)} images, cache={cache}")

    calibrator = make_calibrator(args.kind, loader, cache_file=cache)
    build_engine(BuildConfig(
        onnx_path=args.onnx, engine_path=engine, precision="int8",
        calibrator=calibrator, per_channel_weights=not args.per_tensor_weights,
    ))
    print(f"[calib] built {engine}")

    scales = parse_cache_scales(cache)
    if scales:
        vals = list(scales.values())
        print(f"[calib] cache has {len(scales)} tensor scales; "
              f"min={min(vals):.3e} max={max(vals):.3e}")
        assert all(s > 0 for s in vals), "degenerate (zero) scale in cache!"


if __name__ == "__main__":
    main()
