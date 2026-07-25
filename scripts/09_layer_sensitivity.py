"""Per-layer INT8 sensitivity + FP16-fallback plan.  # RUN ON DEVICE

Captures per-layer activations from the FP32 network over a small probe set (via
polygraphy's layerwise output marking), scores each layer's INT8 SQNR with the HOST-
TESTED qint.sensitivity code, ranks them, and writes the top-5 into a PrecisionPlan
(results/plan_fp16_top5.json) consumed by 06_build_mixed.py.

polygraphy ships with TensorRT; if unavailable, fall back to onnxruntime layer probing.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="weights/yolo11s.onnx")
    ap.add_argument("--images", default="data/calib")
    ap.add_argument("--num-probe", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    import numpy as np
    from qint.sensitivity import score_layer, top_k_sensitive, format_sensitivity_table
    from qint.precision import build_fallback_plan
    from qint.calibration.dataloader import compute_letterbox_params

    # ---- gather per-layer activations (FP32) via polygraphy ----------------
    from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
    from polygraphy.backend.onnx import modify_outputs, onnx_from_path
    import onnx as _onnx  # noqa: F401
    import cv2

    model = onnx_from_path(args.onnx)
    model = modify_outputs(model, outputs="mark all")  # expose every intermediate tensor
    runner = OnnxrtRunner(SessionFromOnnx(model.SerializeToString()))

    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))[: args.num_probe]
    if not paths:
        raise SystemExit(f"no probe images in {args.images}")

    # accumulate per-layer activations across the probe set
    acc = {}
    with runner:
        for pth in paths:
            img = cv2.imread(pth)
            h, w = img.shape[:2]
            _, (nh, nw), (pt, pl) = compute_letterbox_params((h, w), args.imgsz)
            canvas = np.full((args.imgsz, args.imgsz, 3), 114, np.uint8)
            canvas[pt:pt + nh, pl:pl + nw] = cv2.resize(img, (nw, nh))
            x = (canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)) / 255.0
            outs = runner.infer({runner.get_input_metadata().keys().__iter__().__next__(): x})
            for name, val in outs.items():
                acc.setdefault(name, []).append(np.asarray(val).ravel())

    layers = []
    for name, chunks in acc.items():
        activation = np.concatenate(chunks)
        if activation.size == 0:
            continue
        layers.append(score_layer(name, activation))

    print(format_sensitivity_table(layers, k=args.top_k))
    print()
    print("Top-5 most quantization-sensitive layers:")
    for i, l in enumerate(top_k_sensitive(layers, args.top_k), 1):
        sqnr = "inf" if np.isinf(l.sqnr_db) else f"{l.sqnr_db:.2f} dB"
        print(f"  {i}. {l.name}  SQNR={sqnr}")

    plan = build_fallback_plan(layers, k=args.top_k,
                               note=f"top-{args.top_k} SQNR-sensitive layers (probe n={len(paths)})")
    os.makedirs("results", exist_ok=True)
    with open(f"results/plan_fp16_top{args.top_k}.json", "w") as f:
        f.write(plan.to_json())
    print(f"\nWrote results/plan_fp16_top{args.top_k}.json")


if __name__ == "__main__":
    main()
