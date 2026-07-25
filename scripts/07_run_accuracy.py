#!/usr/bin/env python3
"""Evaluate per-class accuracy for one engine.  # RUN ON DEVICE

Runs a TensorRT engine over the val set, computes per-class AP@50 / AP@50-95, and dumps
an AccuracyResult JSON to results/raw/accuracy_<config>.json. Run once per engine; then
scripts/fill_benchmarks.py assembles the per-class delta tables (FP32 vs FP16 vs INT8).

The heavy lifting (mAP math, diffing) is the HOST-TESTED qint.accuracy code — this
script only wires a real engine to it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--config", required=True, help="label, e.g. int8_entropy_512")
    ap.add_argument("--ann", required=True, help="COCO instances json for the val set")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001, help="low conf for mAP (COCO-style)")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0, help="cap #images (0=all)")
    args = ap.parse_args()

    import numpy as np
    from qint.coco import load_coco_gt
    from qint.calibration.dataloader import compute_letterbox_params
    from qint.engine.trt_runner import TRTRunner
    from qint.accuracy.metrics import evaluate
    from qint.report import accuracy_to_dict, save_json
    import cv2

    with open(args.ann) as f:
        gts, class_names, images = load_coco_gt(json.load(f))
    ids = sorted(images.keys())
    if args.limit:
        ids = ids[: args.limit]
        keep = set(ids)
        gts = [g for g in gts if g.image_id in keep]

    runner = TRTRunner(args.engine, num_classes=len(class_names), precision=args.config,
                       imgsz=args.imgsz, conf_thr=args.conf, iou_thr=args.iou)

    all_dets = []
    for i in ids:
        fname, (h, w) = images[i]
        img = cv2.imread(os.path.join(args.img_dir, fname))
        scale, (nh, nw), (pt, pl) = compute_letterbox_params((h, w), args.imgsz)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((args.imgsz, args.imgsz, 3), 114, np.uint8)
        canvas[pt:pt + nh, pl:pl + nw] = resized
        chw = (canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)) / 255.0
        runner.src_hw = (h, w)  # per-image original size for un-letterboxing
        all_dets.extend(runner.infer_batch(chw[None], [i])[0])

    res = evaluate(all_dets, gts, num_classes=len(class_names),
                   precision=args.config)
    out = accuracy_to_dict(res)
    out["class_names"] = class_names
    save_json(f"results/raw/accuracy_{args.config}.json", out)
    print(f"[accuracy] {args.config}: mAP@50={res.map50:.4f} mAP@50-95={res.map5095:.4f}")
    print(f"[accuracy] wrote results/raw/accuracy_{args.config}.json")


if __name__ == "__main__":
    main()
