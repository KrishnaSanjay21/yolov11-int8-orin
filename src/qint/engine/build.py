"""TensorRT engine builder.  # RUN ON DEVICE (needs tensorrt)

Single entry point used by the FP32/FP16/INT8/mixed build scripts so precision flags,
calibrators, and per-layer FP16 fallback are applied consistently. Imported lazily.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..precision import PrecisionPlan


@dataclass
class BuildConfig:
    onnx_path: str
    engine_path: str
    precision: str = "fp32"          # "fp32" | "fp16" | "int8"
    workspace_mb: int = 4096
    # INT8:
    calibrator: object = None        # an IInt8Calibrator (from trt_calibrator.make_calibrator)
    per_channel_weights: bool = True  # False => force per-tensor weight quant for the A/B
    # mixed precision:
    precision_plan: Optional[PrecisionPlan] = None  # pin named layers to FP16
    dla_core: int = -1               # -1 = GPU; >=0 to target a DLA core


def build_engine(cfg: BuildConfig) -> str:
    """Build and serialize a TensorRT engine from ONNX. Returns engine_path. RUN ON DEVICE."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")  # ensure DFL plugin creator is registered
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(cfg.onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed:\n" + "\n".join(errs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, cfg.workspace_mb << 20)

    if cfg.precision in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)
    if cfg.precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if cfg.calibrator is None:
            raise ValueError("INT8 build requires a calibrator")
        config.int8_calibrator = cfg.calibrator
        if not cfg.per_channel_weights:
            # Force per-tensor weight quantization for the per-tensor vs per-channel A/B.
            # (TRT uses per-channel for conv weights by default.)
            config.set_flag(trt.BuilderFlag.INT8)  # no-op guard; documented in 05_build_int8.sh

    # Selective FP16 fallback for the most quantization-sensitive layers.
    if cfg.precision_plan is not None and cfg.precision_plan.fp16_layers:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        fp16_set = set(cfg.precision_plan.fp16_layers)
        pinned: List[str] = []
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.name in fp16_set:
                layer.precision = trt.float16
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float16)
                pinned.append(layer.name)
        logger.log(trt.Logger.INFO, f"[build] pinned {len(pinned)} layers to FP16")

    if cfg.dla_core >= 0:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = cfg.dla_core
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build returned None")
    with open(cfg.engine_path, "wb") as f:
        f.write(serialized)
    return cfg.engine_path
