"""Calibration image loading + the calibration-set-size sweep.

Two pieces are pure-python / numpy and HOST-TESTABLE:
  * :func:`compute_letterbox_params` — the letterbox scale/pad arithmetic.
  * :func:`select_calibration_subset` — deterministic subset selection for the
    32 / 128 / 512 image sweep (stable across runs & machines via a seeded shuffle).

The actual pixel decode/resize (:class:`CalibrationDataLoader`) needs OpenCV and runs
on device; cv2 is imported lazily so this module imports fine on a bare host.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np

# The spec's calibration-set-size sweep.
SWEEP_SIZES: Tuple[int, ...] = (32, 128, 512)


def compute_letterbox_params(
    src_hw: Tuple[int, int], target: int = 640
) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    """Compute (scale, (new_h, new_w), (pad_top, pad_left)) for square letterboxing.

    Matches ultralytics' default: scale to fit, pad to a centered square. Pure math,
    no image needed — so it is unit-tested on host.
    """
    h, w = src_hw
    if h <= 0 or w <= 0:
        raise ValueError("source dims must be positive")
    scale = min(target / h, target / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    pad_top = (target - new_h) // 2
    pad_left = (target - new_w) // 2
    return scale, (new_h, new_w), (pad_top, pad_left)


def select_calibration_subset(
    all_paths: Sequence[str], size: int, seed: int = 0
) -> List[str]:
    """Deterministically pick ``size`` calibration images.

    The sweep sizes are nested-consistent under a fixed seed only in the sense that
    each size draws from the same seeded permutation, so results are reproducible and
    committable. If ``size`` exceeds the pool, returns the whole pool (documented, not
    silently padded).
    """
    paths = list(all_paths)
    if size >= len(paths):
        return paths
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(paths))
    return [paths[i] for i in perm[:size]]


@dataclass
class LoaderConfig:
    image_dir: str
    imgsz: int = 640
    batch: int = 8
    size: int = 512          # one of SWEEP_SIZES
    seed: int = 0
    mean: float = 0.0        # YOLO uses 0..1 scaling, no mean subtraction
    scale: float = 1.0 / 255.0


class CalibrationDataLoader:
    """Streams preprocessed NCHW float32 batches for the TRT calibrator. RUN ON DEVICE."""

    def __init__(self, cfg: LoaderConfig):
        self.cfg = cfg
        import glob
        import os

        pool = sorted(
            glob.glob(os.path.join(cfg.image_dir, "*.jpg"))
            + glob.glob(os.path.join(cfg.image_dir, "*.png"))
        )
        if not pool:
            raise FileNotFoundError(f"no images in {cfg.image_dir}")
        self.paths = select_calibration_subset(pool, cfg.size, cfg.seed)

    def _load_one(self, path: str) -> np.ndarray:
        import cv2  # RUN ON DEVICE

        img = cv2.imread(path)  # BGR HWC
        if img is None:
            raise IOError(f"failed to read {path}")
        h, w = img.shape[:2]
        scale, (nh, nw), (pt, pl) = compute_letterbox_params((h, w), self.cfg.imgsz)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.cfg.imgsz, self.cfg.imgsz, 3), 114, dtype=np.uint8)
        canvas[pt:pt + nh, pl:pl + nw] = resized
        rgb = canvas[:, :, ::-1]  # BGR->RGB
        chw = rgb.transpose(2, 0, 1).astype(np.float32)
        return (chw - self.cfg.mean) * self.cfg.scale

    def __iter__(self) -> Iterator[np.ndarray]:
        batch: List[np.ndarray] = []
        for p in self.paths:
            batch.append(self._load_one(p))
            if len(batch) == self.cfg.batch:
                yield np.stack(batch, 0)
                batch = []
        if batch:
            yield np.stack(batch, 0)

    def __len__(self) -> int:
        return len(self.paths)
