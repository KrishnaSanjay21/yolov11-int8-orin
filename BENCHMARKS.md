# Benchmarks — YOLOv11s @ 640, Jetson Orin NX

All numbers below the FILLED marker are produced on device by committed scripts and
assembled by `python scripts/fill_benchmarks.py`. **Do not hand-edit inside the
markers** — re-run the script. Empty cells render as `—` (measured value missing), never
as a fake 0.

## How these are produced

| metric | source | script |
|---|---|---|
| per-class mAP@50 / mAP@50-95, Δ, >2% flags | COCO-style eval over val set | `07_run_accuracy.py` → `fill_benchmarks.py` |
| latency mean / p99 (ms) | 1000 timed iters, 100 warmup, locked clocks | `08_benchmark.py` |
| throughput (FPS) | iters / wall-clock | `08_benchmark.py` |
| power (W) | INA3221 rails summed during the run | `08_benchmark.py` |
| memory (MB) | used-RAM delta around load+run | `08_benchmark.py` |

## Environment (paste from `results/raw/environment.txt`)

```
device:            Jetson Orin NX ‹8GB|16GB›
nvpmodel / clocks: ‹e.g. MAXN, jetson_clocks ON›     # RUN ON DEVICE: source scripts/00_env.sh
JetPack / L4T:     ‹fill›
TensorRT:          ‹fill›
CUDA / cuDNN:      ‹fill›
torch / ultralytics: ‹fill›
val set:           ‹e.g. COCO val2017 (5000 imgs) | custom N imgs, C classes›
input:             640x640 letterbox, batch=1
```

## Configs benchmarked

`fp32`, `fp16`, `int8_entropy_{32,128,512}`, `int8_minmax_{32,128,512}`,
`int8_entropy_512_pertensor`, `int8_entropy_512_fp16top{1,3,5}`.

<!-- FILLED:START -->
_Run `scripts/fill_benchmarks.py` on device (after 07 + 08) to populate the summary
table and per-class delta tables here._

### Summary (one row per precision config)

| config | mAP@50 | mAP@50-95 | latency mean (ms) | latency p99 (ms) | throughput (FPS) | power (W) | mem (MB) |
|---|---|---|---|---|---|---|---|
| fp32 | — | — | — | — | — | — | — |
| fp16 | — | — | — | — | — | — | — |
| int8_entropy_512 | — | — | — | — | — | — | — |
| int8_minmax_512 | — | — | — | — | — | — | — |
| int8_entropy_512_fp16top5 | — | — | — | — | — | — | — |

### Per-class AP deltas vs fp32

_Per-class table with 🚩 on any class losing >2% absolute AP@50 or AP@50-95 is
generated here, one block per candidate precision._
<!-- FILLED:END -->
