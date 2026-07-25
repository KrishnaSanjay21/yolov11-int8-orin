# YOLOv11s INT8 PTQ for Jetson Orin NX

Post-training quantization of YOLOv11s (FP32 → FP16 → INT8 TensorRT engines) with
**honest per-class accuracy accounting**, a custom fused TensorRT plugin, calibrator /
calibration-size sweeps, and layer-level sensitivity-driven selective FP16 fallback.

The project is split by **where code runs**:

| | Host (CPU, any OS) | Device (Jetson Orin NX) |
|---|---|---|
| Deps | `numpy`, `pytest` (+ optional `torch`) | TensorRT, CUDA, torch, ultralytics, pycuda, cv2 |
| What | model surgery logic, calibration math, mAP/diff tooling, layer sensitivity, precision planning, plugin reference, **all unit tests** | engine builds, INT8 calibration, CUDA plugin, accuracy runs, benchmarks |
| Marked | — | every device-only file/line says `# RUN ON DEVICE` |

Nothing device-specific runs on the host, and no device numbers are fabricated: every
latency/power/accuracy figure is produced by a committed script **you** run on the Orin
and lands in `results/raw/*.json` → rendered into `BENCHMARKS.md`.

## Live demo (Streamlit — no Orin needed)

An interactive frontend (`streamlit_app.py`) exposes the **host-testable core** so you
can explore the toolkit in a browser without any device:

- **Calibration explorer** — entropy (KL) vs min-max on any distribution, live histogram + KL curve
- **Per-class accuracy** — FP32 vs INT8 mAP per class with >2% flagging (stub engine, sliders for per-class degradation)
- **Layer sensitivity** — SQNR ranking → generated FP16-fallback `PrecisionPlan`
- **DFL plugin validation** — fused vs reference op with real max-abs / max-rel error
- **Benchmarks** — upload a device result JSON to render the report table (device numbers only)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Deploy to Streamlit Community Cloud in two clicks — see **[DEPLOY.md](DEPLOY.md)**.

## Layout

```
src/qint/
  engine/        interface (ABC) + StubEngine (host) + TRTRunner (device) + build + postprocess
  accuracy/      COCO-style per-class AP (metrics) + FP32/FP16/INT8 diffing (diff)
  calibration/   stats, minmax, entropy(KL)  [host]  + trt_calibrator, dataloader  [device]
  sensitivity.py per-layer SQNR ranking, top-5
  precision.py   selective-FP16 precision plan + accuracy-per-ms
  plugin/        dfl_reference.py [host]  + dfl_plugin/ (C++/CUDA TensorRT plugin) [device]
  report.py      result JSON <-> BENCHMARKS.md rendering
  coco.py        COCO GT parsing
scripts/         00..09 build/calibrate/accuracy/benchmark + validate_plugin + fill_benchmarks
tests/           host unit tests (numpy only) — the whole accuracy/calibration/plugin core
calib_cache/     COMMITTED calibration caches (entropy_512.cache, ...)
results/         templates + device-produced raw JSON
DECISIONS.md     entropy vs minmax, calib size, FP16 layer choices, accuracy-per-ms
BENCHMARKS.md    per-class mAP delta + latency/throughput/power/mem, one row per config
ACCURACY_NOTES.md the written "which classes degraded and why" analysis
```

## Host: run the tests (no device needed)

```bash
pip install -r requirements-host.txt
pytest                       # 49+ tests: mAP math, calibrators, sensitivity, plugin reference
python scripts/validate_plugin.py     # numeric fused-vs-reference DFL check (host)
```

The plugin's PyTorch cross-check runs automatically **if** a working `torch` is present;
otherwise it is skipped and the numpy fused-vs-unfused check still validates correctness.

## Device: full pipeline (RUN ON DEVICE)

```bash
source scripts/00_env.sh                     # lock clocks, record env
bash   scripts/01_export_onnx.sh             # yolo11s.pt -> ONNX
bash   scripts/02_build_fp32.sh              # FP32 baseline engine
bash   scripts/03_build_fp16.sh              # FP16 engine
bash   scripts/05_build_int8.sh              # INT8 matrix: {entropy,minmax} x {32,128,512} + weight A/B
bash   scripts/build_plugin.sh               # build libdfl_plugin.so
python scripts/validate_plugin.py --plugin src/qint/plugin/dfl_plugin/build/libdfl_plugin.so
python scripts/09_layer_sensitivity.py       # top-5 sensitive layers -> FP16 plan
python scripts/06_build_mixed.py             # INT8 + FP16-fallback ladder (top-1/3/5)

# measure (per engine):
python scripts/07_run_accuracy.py --engine weights/<eng>.engine --config <cfg> \
       --ann data/val.json --img-dir data/val
python scripts/08_benchmark.py   --engine weights/<eng>.engine --config <cfg>

python scripts/fill_benchmarks.py            # assemble BENCHMARKS.md from results/raw/*.json
```

## Done-when → where it lives

- **FP32/FP16/INT8 engines, committed build scripts** → `scripts/02,03,04,05` + `qint.engine.build`.
- **Entropy & minmax calibrators, committed cache, 32/128/512 sweep** → `qint.calibration.*`, `scripts/04,05`, `calib_cache/`.
- **Per-tensor vs per-channel** → `qint.calibration.minmax.compute_weight_scales` + `--per-tensor-weights` (see DECISIONS.md).
- **Custom fused TRT plugin + numeric validation** → `src/qint/plugin/dfl_plugin/` + `scripts/validate_plugin.py` + `tests/test_dfl_plugin_reference.py`.
- **Per-class mAP@50 / mAP@50-95 FP32/FP16/INT8, >2% flag** → `qint.accuracy.*`, `scripts/07`, rendered in `BENCHMARKS.md`.
- **Top-5 sensitive layers + selective FP16** → `qint.sensitivity`, `qint.precision`, `scripts/09,06`.
- **BENCHMARKS.md template filled by committed script** → `BENCHMARKS.md` + `scripts/fill_benchmarks.py`.
- **Written "which classes degraded and why"** → `ACCURACY_NOTES.md`.
```
