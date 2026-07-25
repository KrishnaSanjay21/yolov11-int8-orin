# DECISIONS

Engineering decisions for INT8 PTQ of YOLOv11s on Jetson Orin NX.

Numbers marked **`‹fill from results›`** come from committed device runs, not guesses.
Each decision states its **rule** up front so the choice is reproducible: run the sweep,
read the table, apply the rule. `scripts/fill_benchmarks.py` regenerates the tables.

---

## 1. Entropy vs min-max calibrator

**Rule:** pick the calibrator with the higher **mAP@50-95** at the chosen calibration
size, tie-broken by fewer classes flagged (>2% AP drop). Latency is identical between
the two (the calibrator only sets activation scales; the engine graph is unchanged), so
this is a pure accuracy decision.

**A-priori reasoning.**
- **min-max** sets `scale = absmax/127`. It is safe (never clips) but a single outlier
  activation stretches the range so the bulk of values use only a few of the 255 INT8
  levels — coarse quantization where it matters. Detection heads and post-SiLU
  activations in YOLO have heavy positive tails, so min-max tends to lose small-object /
  low-contrast recall.
- **entropy (IInt8EntropyCalibrator2)** searches for a clip threshold minimizing KL
  divergence between the FP32 distribution and its INT8 requantization. It deliberately
  clips the tail to spend resolution on the bulk. This usually wins for detection, at
  some risk on layers whose tail *is* the signal (large-object logits).

**Expectation (to confirm/refute on device):** entropy ≥ min-max on aggregate
mAP@50-95, with min-max occasionally better on a large-object-dominated class. The
host-side `test_entropy_calibrate_clips_heavy_tail` demonstrates the mechanism
numerically (entropy scale < min-max scale on a heavy-tailed tensor).

**Decision:** `‹fill from results›` (compare `results/raw/accuracy_int8_entropy_512.json`
vs `accuracy_int8_minmax_512.json`).

| calibrator @512 | mAP@50 | mAP@50-95 | #classes flagged >2% |
|---|---|---|---|
| entropy | ‹fill› | ‹fill› | ‹fill› |
| minmax  | ‹fill› | ‹fill› | ‹fill› |

---

## 2. Calibration set size (32 / 128 / 512)

**Rule:** choose the smallest size past which mAP@50-95 improves by < 0.2 absolute
points (diminishing returns). Larger sets cost only calibration wall-time, not
inference — but a too-small set gives unstable, unrepresentative scales.

**A-priori reasoning.** Scales are set from activation statistics; 32 images often
under-represent rare classes and give noisy tails, inflating per-class variance. 128 is
usually enough for stable aggregate mAP; 512 mostly tightens the worst per-class
outliers. The images must be drawn from the **train/val distribution** and cover the
class mix — `select_calibration_subset` draws a fixed seeded permutation so the set is
reproducible and the cache is committable.

**Decision:** `‹fill from results›`

| size | mAP@50-95 (entropy) | Δ vs prev | worst per-class drop |
|---|---|---|---|
| 32  | ‹fill› | — | ‹fill› |
| 128 | ‹fill› | ‹fill› | ‹fill› |
| 512 | ‹fill› | ‹fill› | ‹fill› |

---

## 3. Per-tensor vs per-channel weight scales

**What TRT permits.** TensorRT quantizes **activations per-tensor** always (a single
scale per activation tensor). The granularity choice it exposes is for **convolution
weights**: per-channel (one scale per output channel, the default) vs per-tensor.
`compute_weight_scales(..., per_channel=...)` computes both host-side; the host test
`test_weight_scales_per_channel_vs_per_tensor` shows per-channel yields strictly lower
weight MSE when channels have unequal ranges (they always do in trained convs).

`04_calibrate_int8.py --per-tensor-weights` forces the weaker scheme for the A/B.

**Rule:** keep per-channel (default) unless per-tensor is within 0.2 mAP@50-95 *and*
measurably faster — it generally is not, since per-channel weight scales are free at
inference on Orin's INT8 path.

**Decision:** `‹fill from results›`

| weight granularity (entropy@512) | mAP@50-95 | latency mean (ms) |
|---|---|---|
| per-channel | ‹fill› | ‹fill› |
| per-tensor  | ‹fill› | ‹fill› |

---

## 4. Custom plugin: fused DFL decode

**Chosen op.** YOLOv11's Distribution-Focal-Loss decode: softmax over `reg_max=16` bins
per box side, then expectation `E[j] = Σ softmax_j · j`. Fusing softmax + expectation
into one kernel avoids materializing the softmax tensor and a separate reduction.

**Why keep it FP32.** Softmax is quantization-sensitive (a global normalization over a
small axis); running it in INT8 corrupts the argmax-adjacent bins that dominate the
expectation. The plugin advertises FP32-only I/O, so the decode stays exact while the
convolutional backbone runs INT8 — a targeted, principled precision boundary.

**Validation (real error, not eyeballed).** `scripts/validate_plugin.py` compares the
CUDA kernel output against the numpy reference (and PyTorch when available) with
**max-abs** and **max-rel** error against `--tol` (default 1e-3). Host fused-vs-unfused
already passes at **max_abs ≈ 2.6e-6, max_rel ≈ 3.6e-7** (float32 accumulation vs float64
reference). Device CUDA-vs-reference: `‹fill from validate_plugin.py --plugin ...›`.

---

## 5. Layer-level sensitivity → selective FP16 fallback

**Metric.** Per-layer SQNR (dB) of INT8 fake-quant vs FP32 activation
(`qint.sensitivity`). Lowest SQNR = worst-represented = first to pin to FP16. Ties break
on error·energy so a low-SQNR but negligible-magnitude layer doesn't win a slot.

**Rule.** Walk the top-1 → top-3 → top-5 FP16 ladder (`06_build_mixed.py`). Adopt the
config that clears all >2% class flags at the **best accuracy-per-millisecond**
(`qint.precision.accuracy_per_ms`). Stop adding FP16 layers once flags are cleared —
each FP16 layer costs latency.

**Top-5 sensitive layers (from `09_layer_sensitivity.py`):** `‹fill from results›`

**Accuracy-per-millisecond bought:** `‹fill from results›`

| config | mAP@50-95 | Δ vs pure INT8 | Δ latency (ms) | mAP points / ms |
|---|---|---|---|---|
| INT8 (baseline) | ‹fill› | 0 | 0 | — |
| INT8 + FP16 top-1 | ‹fill› | ‹fill› | ‹fill› | ‹fill› |
| INT8 + FP16 top-3 | ‹fill› | ‹fill› | ‹fill› | ‹fill› |
| INT8 + FP16 top-5 | ‹fill› | ‹fill› | ‹fill› | ‹fill› |

**Final recommended deployment config:** `‹fill after applying the rules above›`
