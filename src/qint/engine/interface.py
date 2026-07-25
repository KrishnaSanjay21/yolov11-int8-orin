"""Inference-engine abstraction.

The accuracy / diff tooling talks to *this* interface, never to TensorRT directly.
That is what lets the whole accuracy pipeline be tested on a host CPU with ``StubEngine``
standing in for a real ``.engine`` file.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detection in absolute pixel xyxy coordinates.

    Attributes
    ----------
    image_id : int
        Index of the image this detection belongs to.
    class_id : int
        Predicted class index.
    score : float
        Confidence in [0, 1].
    box : tuple[float, float, float, float]
        (x1, y1, x2, y2), x2>=x1, y2>=y1.
    """

    image_id: int
    class_id: int
    score: float
    box: tuple

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class InferenceEngine(ABC):
    """Minimal contract every engine (stub or real TRT) must satisfy.

    ``infer_batch`` takes a preprocessed NCHW float array and returns, per image,
    a list of ``Detection`` in the ORIGINAL image coordinate space. Coordinate
    un-letterboxing is the engine's responsibility so the accuracy layer is
    engine-agnostic.
    """

    #: number of classes the model predicts
    num_classes: int
    #: human label ("fp32" | "fp16" | "int8" | "int8-entropy-512" | ...)
    precision: str

    @abstractmethod
    def infer_batch(self, images: np.ndarray, image_ids: Sequence[int]) -> List[List[Detection]]:
        """Run inference on a batch.

        Parameters
        ----------
        images : np.ndarray
            (N, C, H, W) float32, preprocessed (letterboxed, normalized).
        image_ids : Sequence[int]
            Length-N ids used to stamp the returned Detections.
        """
        raise NotImplementedError

    def infer_dataset(self, images: np.ndarray, image_ids: Sequence[int]) -> List[Detection]:
        """Convenience: flatten per-image results across a whole (small) dataset."""
        out: List[Detection] = []
        for per_image in self.infer_batch(images, image_ids):
            out.extend(per_image)
        return out
