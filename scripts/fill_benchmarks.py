#!/usr/bin/env python3
"""Assemble BENCHMARKS.md from device result JSONs.

Committed, host-runnable (pure python + qint). Reads:
  results/raw/bench_<config>.json      (from 08_benchmark.py)
  results/raw/accuracy_<config>.json   (from 07_run_accuracy.py)
and rewrites the region between the FILLED markers in BENCHMARKS.md with:
  * the one-row-per-config summary table, and
  * per-class AP delta tables (FP32 baseline vs each candidate) with >2% flags.

Usage (after device runs):  python3 scripts/fill_benchmarks.py
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

START = "<!-- FILLED:START -->"
END = "<!-- FILLED:END -->"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="BENCHMARKS.md")
    ap.add_argument("--baseline", default="fp32")
    args = ap.parse_args()

    from qint.report import build_rows, render_benchmarks_table, accuracy_from_dict
    from qint.accuracy.diff import diff_precisions, format_delta_table

    bench = {}
    acc = {}
    for p in glob.glob(os.path.join(args.raw, "bench_*.json")):
        cfg = os.path.basename(p)[len("bench_"):-len(".json")]
        with open(p, encoding="utf-8") as f:
            bench[cfg] = json.load(f)
    for p in glob.glob(os.path.join(args.raw, "accuracy_*.json")):
        cfg = os.path.basename(p)[len("accuracy_"):-len(".json")]
        with open(p, encoding="utf-8") as f:
            acc[cfg] = json.load(f)

    if not bench and not acc:
        print("No result JSONs found under", args.raw, "- run 07/08 on device first.")
        return

    # stable, human order: fp32, fp16, then int8 variants sorted
    def rank(cfg):
        return (0 if cfg == "fp32" else 1 if cfg == "fp16" else 2, cfg)
    order = sorted(set(bench) | set(acc), key=rank)

    rows = build_rows(bench, acc, order=order)
    parts = ["### Summary (one row per precision config)", "",
             render_benchmarks_table(rows), ""]

    # per-class delta tables vs baseline
    if args.baseline in acc:
        base = accuracy_from_dict(acc[args.baseline])
        names = acc[args.baseline].get("class_names")
        parts.append("### Per-class AP deltas vs " + args.baseline)
        parts.append("")
        for cfg in order:
            if cfg == args.baseline or cfg not in acc:
                continue
            cand = accuracy_from_dict(acc[cfg])
            d = diff_precisions(base, cand, class_names=names, threshold=0.02)
            parts.append(format_delta_table(d))
            parts.append("")
    else:
        parts.append(f"> baseline `{args.baseline}` accuracy JSON not found; "
                     "per-class deltas skipped.")

    filled = "\n".join(parts)

    text = (open(args.out, encoding="utf-8").read()
            if os.path.exists(args.out) else _skeleton())
    if START in text and END in text:
        pre = text.split(START)[0]
        post = text.split(END)[1]
        text = pre + START + "\n" + filled + "\n" + END + post
    else:
        text = text.rstrip() + "\n\n" + START + "\n" + filled + "\n" + END + "\n"
    # utf-8 explicitly: tables use → and 🚩; Windows' default cp1252 can't encode them.
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Filled {args.out} ({len(rows)} configs).")


def _skeleton():
    return f"# Benchmarks\n\n{START}\n{END}\n"


if __name__ == "__main__":
    main()
