#!/usr/bin/env python3
"""Latency / throughput / power / memory for one engine.  # RUN ON DEVICE

Emits results/raw/bench_<config>.json consumed by fill_benchmarks.py. Power is read
from the Jetson INA3221 sysfs rails (tegra-based); if the rail path differs on your
JetPack, set --power-rail. Memory is the delta in used RAM around engine load+run.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def read_power_w(rail_glob):
    """Sum instantaneous power (mW->W) across matching INA3221 rails. Best-effort."""
    total_mw = 0.0
    found = False
    for p in glob.glob(rail_glob):
        try:
            with open(p) as f:
                total_mw += float(f.read().strip())
                found = True
        except OSError:
            continue
    return (total_mw / 1000.0) if found else None


def used_mem_mb():
    try:
        with open("/proc/meminfo") as f:
            info = {k.strip(): v for k, v in (l.split(":") for l in f)}
        total = float(info["MemTotal"].split()[0])
        avail = float(info["MemAvailable"].split()[0])
        return (total - avail) / 1024.0
    except (OSError, KeyError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--power-rail",
                    default="/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/power*_input")
    args = ap.parse_args()

    import numpy as np
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt
    from qint.report import BenchmarkRow, save_json

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    mem_before = used_mem_mb()
    with open(args.engine, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    in_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(engine.num_io_tensors - 1)
    x = np.ascontiguousarray(np.random.rand(1, 3, args.imgsz, args.imgsz).astype(np.float32))
    ctx.set_input_shape(in_name, x.shape)
    out_shape = tuple(ctx.get_tensor_shape(out_name))
    y = np.empty(out_shape, np.float32)
    d_in = cuda.mem_alloc(x.nbytes)
    d_out = cuda.mem_alloc(y.nbytes)
    ctx.set_tensor_address(in_name, int(d_in))
    ctx.set_tensor_address(out_name, int(d_out))
    stream = cuda.Stream()

    def once():
        cuda.memcpy_htod_async(d_in, x, stream)
        ctx.execute_async_v3(stream.handle)
        stream.synchronize()

    for _ in range(args.warmup):
        once()

    lat_ms = []
    powers = []
    t_all0 = time.perf_counter()
    for i in range(args.iters):
        t0 = time.perf_counter()
        once()
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
        if i % 50 == 0:
            p = read_power_w(args.power_rail)
            if p is not None:
                powers.append(p)
    wall = time.perf_counter() - t_all0
    mem_after = used_mem_mb()

    lat = np.array(lat_ms)
    row = BenchmarkRow(
        config=args.config,
        latency_mean_ms=float(lat.mean()),
        latency_p99_ms=float(np.percentile(lat, 99)),
        throughput_fps=float(args.iters / wall),
        power_w=(float(np.mean(powers)) if powers else None),
        mem_mb=(float(mem_after - mem_before) if (mem_after and mem_before) else None),
    )
    save_json(f"results/raw/bench_{args.config}.json", row.to_dict())
    print(f"[bench] {args.config}: mean={row.latency_mean_ms:.3f}ms "
          f"p99={row.latency_p99_ms:.3f}ms fps={row.throughput_fps:.1f} "
          f"power={row.power_w} mem={row.mem_mb}")


if __name__ == "__main__":
    main()
