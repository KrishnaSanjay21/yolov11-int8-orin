# Which classes degraded under INT8 — and why

This is the required written answer. It has two parts:

1. **Pre-registered hypotheses** — what I expect to degrade and the mechanism, written
   *before* reading device numbers so the analysis can't be retrofitted.
2. **Findings** — filled from `results/raw/accuracy_*.json` after the device run, then
   each hypothesis marked **confirmed / refuted / partial** against the flagged classes.

The flagged set (any class losing >2% absolute AP@50 or AP@50-95, FP32→INT8) comes from
`qint.accuracy.diff`; see `BENCHMARKS.md` for the full per-class table.

---

## Part 1 — Pre-registered hypotheses (mechanism-first)

INT8 PTQ degrades some classes more than others. The mechanisms, and the class
signatures each predicts:

- **H1 — Small objects lose the most.** Per-tensor activation quantization gives coarse
  spatial-feature resolution; small objects occupy few pixels and low activation
  magnitudes, so their features land in the bottom INT8 levels and get flattened.
  *Predicted casualties:* classes dominated by small instances (COCO: `traffic light`,
  `sports ball`, `remote`, `mouse`, `cell phone`, small `bird`). Signature: **AP@50-95
  drops more than AP@50** (localization precision hit hardest).

- **H2 — Rare classes are under-calibrated.** Calibration statistics are dominated by
  frequent classes; rare classes contribute little to the histograms, so their
  activation ranges are represented by borrowed scales. *Predicted casualties:* the
  long-tail classes with few val instances. Signature: **high per-class variance vs
  calibration set size** — improves noticeably from 32→512.

- **H3 — Visually similar / confusable pairs get worse.** INT8 rounding erodes the
  small logit margins that separate near-classes, increasing cross-class false
  positives. *Predicted casualties:* `truck`↔`bus`↔`car`, `cat`↔`dog`. Signature: AP
  drop paired with rising off-diagonal confusion.

- **H4 — Heavy-tailed-activation classes suffer under min-max, not entropy.** Classes
  whose detection relies on a few high-magnitude activations are hurt when min-max
  spends dynamic range on outliers. Signature: **min-max flags them, entropy does not.**

- **H5 — Large-object classes may prefer min-max.** Where the tail *is* the signal,
  entropy's clipping can shave a large-object class. Signature: **entropy flags a
  big-object class (`train`, `bus`) that min-max keeps.** (This is the counter-case that
  makes the entropy-vs-minmax choice non-trivial.)

**Mitigation already built in:** the layers whose INT8 SQNR is worst (`09_layer_sensitivity.py`)
are exactly those feeding the sensitive decode/head path; pinning the top-5 to FP16
(`06_build_mixed.py`) should recover H1/H3 classes first. The DFL decode is kept FP32 in
the plugin for the same reason — softmax over bins is where small localization errors
compound.

---

## Part 2 — Findings (fill after device run)

**Flagged classes (FP32 → INT8 entropy@512):** `‹fill from BENCHMARKS.md›`

| class | ΔAP@50 | ΔAP@50-95 | typical instance size | hypothesis it matches |
|---|---|---|---|---|
| ‹fill› | ‹fill› | ‹fill› | ‹fill› | H? |

**Hypothesis scorecard:**

- H1 (small objects): ‹confirmed / refuted / partial› — evidence: ‹fill›
- H2 (rare classes, size-sensitivity): ‹…› — evidence: 32 vs 512 per-class deltas: ‹fill›
- H3 (confusable pairs): ‹…› — evidence: ‹fill›
- H4 (heavy-tail, min-max only): ‹…› — evidence: classes flagged by minmax but not entropy: ‹fill›
- H5 (large-object prefers min-max): ‹…› — evidence: ‹fill›

**Did selective FP16 fallback recover the flagged classes?** `‹fill›` — compare
`accuracy_int8_entropy_512` vs `accuracy_int8_entropy_512_fp16top5` per-class, and report
the accuracy-per-ms from DECISIONS.md §5.

**One-paragraph verdict:** `‹fill: which classes degraded, the dominant mechanism, and
whether the deployed config's mitigations closed the gap within the latency budget.›`
