#!/usr/bin/env python3
"""Validate the fused DFL plugin numerically against the reference op.

Two modes, one script:

  * HOST (no --plugin): runs the numpy fused-vs-unfused reference check (and torch, if
    available) and prints max-abs / max-rel error. This is the same check as
    tests/test_dfl_plugin_reference.py, runnable standalone.  # runs anywhere

  * DEVICE (--plugin path/to/libdfl_plugin.so): builds a tiny TensorRT network whose
    single node is the DFL plugin, runs random inputs through it on the GPU, and
    compares the CUDA output against the numpy reference with the SAME error metrics.
    # RUN ON DEVICE

Exit code is non-zero if any error exceeds --tol, so it can gate CI / a device smoke test.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def errors(a, b):
    import numpy as np
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    max_abs = float(np.max(np.abs(a - b)))
    max_rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-8)))
    return max_abs, max_rel


def run_host(reg_max, tol):
    import numpy as np
    from qint.plugin.dfl_reference import (
        dfl_fused_reference, dfl_unfused_reference, dfl_torch_reference, TORCH_AVAILABLE,
    )
    rng = np.random.default_rng(0)
    x = rng.normal(0, 6.0, size=(2, 4 * reg_max, 100)).astype(np.float32)
    fused = dfl_fused_reference(x, reg_max)
    unfused = dfl_unfused_reference(x, reg_max)
    ma, mr = errors(fused, unfused)
    print(f"[host] fused vs unfused numpy:  max_abs={ma:.3e}  max_rel={mr:.3e}")
    ok = ma < tol and mr < tol
    if TORCH_AVAILABLE:
        t = dfl_torch_reference(x, reg_max)
        ma2, mr2 = errors(fused, t)
        print(f"[host] fused vs torch:         max_abs={ma2:.3e}  max_rel={mr2:.3e}")
        ok = ok and ma2 < tol and mr2 < tol
    else:
        print("[host] torch not available -> torch cross-check skipped")
    return ok


def run_device(plugin_so, reg_max, tol):
    import ctypes
    import numpy as np
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt
    from qint.plugin.dfl_reference import dfl_unfused_reference

    ctypes.CDLL(plugin_so, mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")

    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("DFL", "1", "")
    if creator is None:
        print("ERROR: DFL plugin creator not found after loading", plugin_so)
        return False
    fields = trt.PluginFieldCollection([
        trt.PluginField("reg_max", np.array([reg_max], np.int32), trt.PluginFieldType.INT32)
    ])
    plugin = creator.create_plugin("dfl", fields)

    A = 100
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    inp = network.add_input("x", trt.float32, (1, 4 * reg_max, A))
    layer = network.add_plugin_v2([inp], plugin)
    layer.get_output(0).name = "y"
    network.mark_output(layer.get_output(0))

    config = builder.create_builder_config()
    engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine)
    ctx = engine.create_execution_context()

    rng = np.random.default_rng(1)
    x = np.ascontiguousarray(rng.normal(0, 6.0, (1, 4 * reg_max, A)).astype(np.float32))
    y = np.empty((1, 4, A), np.float32)
    d_in = cuda.mem_alloc(x.nbytes)
    d_out = cuda.mem_alloc(y.nbytes)
    cuda.memcpy_htod(d_in, x)
    ctx.set_tensor_address("x", int(d_in))
    ctx.set_tensor_address("y", int(d_out))
    stream = cuda.Stream()
    ctx.execute_async_v3(stream.handle)
    stream.synchronize()
    cuda.memcpy_dtoh(y, d_out)

    ref = dfl_unfused_reference(x, reg_max)
    ma, mr = errors(y, ref)
    print(f"[device] CUDA plugin vs numpy reference:  max_abs={ma:.3e}  max_rel={mr:.3e}")
    return ma < tol and mr < tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default=None, help="path to libdfl_plugin.so (DEVICE mode)")
    ap.add_argument("--reg-max", type=int, default=16)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    ok = run_host(args.reg_max, args.tol)
    if args.plugin:
        ok = run_device(args.plugin, args.reg_max, args.tol) and ok

    print("RESULT:", "PASS" if ok else "FAIL", f"(tol={args.tol})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
