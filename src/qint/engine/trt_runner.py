"""Real TensorRT engine runner.  # RUN ON DEVICE (needs tensorrt + pycuda)

Implements the same :class:`~qint.engine.interface.InferenceEngine` contract as the
host StubEngine, so the accuracy pipeline (evaluate/diff) is identical whether it is
fed a stub or a real ``.engine``. Post-processing is delegated to the host-testable
:mod:`qint.engine.postprocess`.

Imported lazily by device scripts only — never at package import time.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .interface import Detection, InferenceEngine
from .postprocess import decode_predictions


class TRTRunner(InferenceEngine):
    def __init__(
        self,
        engine_path: str,
        num_classes: int,
        precision: str,
        src_hw: Tuple[int, int] = (640, 640),
        imgsz: int = 640,
        conf_thr: float = 0.25,
        iou_thr: float = 0.7,
        input_name: str = "images",
        output_name: str = "output0",
    ):
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt

        self.num_classes = num_classes
        self.precision = precision
        self.src_hw = src_hw
        self.imgsz = imgsz
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self._cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = input_name
        self.output_name = output_name

    def infer_batch(self, images: np.ndarray, image_ids: Sequence[int]) -> List[List[Detection]]:
        cuda = self._cuda
        images = np.ascontiguousarray(images, dtype=np.float32)
        n = images.shape[0]
        self.context.set_input_shape(self.input_name, images.shape)

        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out = np.empty(out_shape, dtype=np.float32)

        d_in = cuda.mem_alloc(images.nbytes)
        d_out = cuda.mem_alloc(out.nbytes)
        cuda.memcpy_htod(d_in, images)
        self.context.set_tensor_address(self.input_name, int(d_in))
        self.context.set_tensor_address(self.output_name, int(d_out))
        stream = cuda.Stream()
        self.context.execute_async_v3(stream.handle)
        stream.synchronize()
        cuda.memcpy_dtoh(out, d_out)

        results: List[List[Detection]] = []
        for bi in range(n):
            results.append(
                decode_predictions(
                    out[bi], image_id=int(image_ids[bi]), src_hw=self.src_hw,
                    num_classes=self.num_classes, conf_thr=self.conf_thr,
                    iou_thr=self.iou_thr, imgsz=self.imgsz,
                )
            )
        return results
