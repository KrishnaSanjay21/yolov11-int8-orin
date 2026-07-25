# Calibration caches (COMMITTED)

TensorRT INT8 calibration caches are committed here so INT8 engines rebuild
reproducibly **without** re-streaming calibration images. `.gitignore` deliberately
does NOT ignore this directory.

Filenames encode `<calibrator>_<setsize>[.<weightscheme>]`:

```
entropy_32.cache     entropy_128.cache     entropy_512.cache
minmax_32.cache      minmax_128.cache      minmax_512.cache
```

Produced by `scripts/04_calibrate_int8.py` (via `scripts/05_build_int8.sh`). A cache is a
text file of `tensor_name: <hex float32 scale>` lines; `qint.calibration.trt_calibrator.
parse_cache_scales` reads them for the sanity check (no zero/degenerate scales).

> These are device-produced. Commit them after the first `05_build_int8.sh` run so
> teammates get byte-identical INT8 engines. They are NOT present until that run.
