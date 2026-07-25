"""StubEngine — a deterministic, dependency-free fake engine for HOST tests.

Why this exists
---------------
The spec requires the accuracy-diff tooling to be testable on CPU "with a stub engine".
A real TensorRT ``.engine`` cannot run on the host, so we need something that:

  * implements :class:`~qint.engine.interface.InferenceEngine`,
  * produces detections we can reason about analytically, and
  * can *simulate quantization degradation* per class so the diff/flagging logic
    is exercised end-to-end (FP32 "perfect", INT8 "class 3 got worse", etc.).

The stub derives its predictions from a supplied ground-truth set, then applies a
deterministic, seeded perturbation controlled by a :class:`DegradationProfile`. No
randomness leaks across runs: the RNG is seeded from ``(seed, image_id, gt_index)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .interface import Detection, InferenceEngine


@dataclass
class GTBox:
    """A ground-truth box used to synthesize predictions."""

    image_id: int
    class_id: int
    box: tuple  # (x1, y1, x2, y2)


@dataclass
class DegradationProfile:
    """How badly a given precision config mangles the (otherwise perfect) predictions.

    All knobs are deterministic given the seed.

    Parameters
    ----------
    drop_prob : dict[int, float]
        Per-class probability a true object is missed (recall loss). Realized
        deterministically via a hash so a given (image, gt) either drops or not.
    score_bias : dict[int, float]
        Per-class additive shift to confidence (negative => lower scores => worse AP
        ranking). Clamped to [0, 1].
    loc_jitter : dict[int, float]
        Per-class localization noise as a fraction of box size, pushing IoU down
        (hurts mAP@50-95 more than mAP@50).
    false_pos_per_image : dict[int, float]
        Per-class expected number of spurious low-confidence boxes per image.
    """

    drop_prob: Dict[int, float] = field(default_factory=dict)
    score_bias: Dict[int, float] = field(default_factory=dict)
    loc_jitter: Dict[int, float] = field(default_factory=dict)
    false_pos_per_image: Dict[int, float] = field(default_factory=dict)

    @staticmethod
    def perfect() -> "DegradationProfile":
        return DegradationProfile()


def _rng(seed: int, *ids: int) -> np.random.Generator:
    """Deterministic per-element RNG. Same inputs -> same stream, every run/OS."""
    mix = np.uint64(seed & 0xFFFFFFFF)
    for i in ids:
        mix = (mix * np.uint64(1000003) + np.uint64(int(i) & 0xFFFFFFFF)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return np.random.default_rng(int(mix))


class StubEngine(InferenceEngine):
    """Synthesize detections from ground truth with a per-precision degradation profile."""

    def __init__(
        self,
        ground_truth: Sequence[GTBox],
        num_classes: int,
        precision: str = "stub",
        base_score: float = 0.9,
        profile: DegradationProfile | None = None,
        seed: int = 0,
    ) -> None:
        self.num_classes = num_classes
        self.precision = precision
        self.base_score = base_score
        self.profile = profile or DegradationProfile.perfect()
        self.seed = seed
        # index GT by image for O(1) lookup
        self._gt_by_image: Dict[int, List[GTBox]] = {}
        for gt in ground_truth:
            self._gt_by_image.setdefault(gt.image_id, []).append(gt)

    # -- InferenceEngine ---------------------------------------------------
    def infer_batch(self, images: np.ndarray, image_ids: Sequence[int]) -> List[List[Detection]]:
        # StubEngine ignores pixel content; it works off image_ids + stored GT.
        # (images is accepted only to honor the interface / shape checks.)
        if images is not None and len(images) != len(image_ids):
            raise ValueError("images batch size must match image_ids length")
        return [self._predict_image(int(i)) for i in image_ids]

    # -- internals ---------------------------------------------------------
    def _predict_image(self, image_id: int) -> List[Detection]:
        dets: List[Detection] = []
        p = self.profile
        for gi, gt in enumerate(self._gt_by_image.get(image_id, [])):
            r = _rng(self.seed, image_id, gi, gt.class_id)
            # (1) recall: maybe drop this true object
            if r.random() < p.drop_prob.get(gt.class_id, 0.0):
                continue
            # (2) localization jitter
            box = self._jitter(gt.box, p.loc_jitter.get(gt.class_id, 0.0), r)
            # (3) score with per-class bias
            score = float(np.clip(self.base_score + p.score_bias.get(gt.class_id, 0.0), 0.0, 1.0))
            dets.append(Detection(image_id=image_id, class_id=gt.class_id, score=score, box=box))
            # (4) false positives near this object
            fp_lambda = p.false_pos_per_image.get(gt.class_id, 0.0)
            n_fp = int(r.poisson(fp_lambda)) if fp_lambda > 0 else 0
            for k in range(n_fp):
                r2 = _rng(self.seed, image_id, gi, gt.class_id, 900 + k)
                fp_box = self._jitter(gt.box, 0.6, r2)  # badly localized
                dets.append(
                    Detection(image_id=image_id, class_id=gt.class_id,
                              score=float(0.1 + 0.2 * r2.random()), box=fp_box)
                )
        return dets

    @staticmethod
    def _jitter(box, frac: float, r: np.random.Generator) -> tuple:
        if frac <= 0:
            return tuple(float(v) for v in box)
        x1, y1, x2, y2 = box
        w, h = (x2 - x1), (y2 - y1)
        dx = (r.random() - 0.5) * 2 * frac * w
        dy = (r.random() - 0.5) * 2 * frac * h
        # scale jitter too
        sw = 1.0 + (r.random() - 0.5) * frac
        sh = 1.0 + (r.random() - 0.5) * frac
        cx, cy = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
        nw, nh = w * sw, h * sh
        return (cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2)
