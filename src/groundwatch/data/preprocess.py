"""Deterministic pre-processing, run identically at training and inference.

compute-and-data-pipeline.md splits pre-processing into two buckets so they cannot be
mixed up (a classic source of train/deploy skew):

* deterministic (this module): undistort / rectify, flat-field, colour-normalise.
  Versioned: changing any parameter means retraining, so the version string is written
  into every checkpoint and experiment record.
* train-time augmentation (augment.py): stochastic, never runs at inference.

ASSUMPTION: the calibration-dependent steps are stubs that pass the image through. They
need the rig's lens calibration, the measured illumination field and the in-frame colour
target, none of which exist yet. Whether frames arrive here already undistorted (done on
the rig) or raw is also undecided; the stubs make the slot explicit either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PREPROCESS_VERSION = "0.1.0-passthrough"


@dataclass(frozen=True)
class PreprocessConfig:
    flat_field: str | None = None  # path to a calibrated illumination field (HxW float), later
    colour_target: bool = False  # use the in-frame grey/colour target, later
    undistort: str | None = None  # path to lens calibration, later


def to_float(image: np.ndarray) -> np.ndarray:
    """uint8 / uint16 -> float32 in [0, 1]. 10-12-bit raw is expected to arrive as uint16."""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    return image.astype(np.float32)


def preprocess(image: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    img = to_float(image)
    if cfg.undistort:
        raise NotImplementedError("lens undistortion needs rig calibration (not available yet)")
    if cfg.flat_field:
        raise NotImplementedError("flat-field correction needs the measured illumination field")
    if cfg.colour_target:
        raise NotImplementedError("colour normalisation needs the in-frame target detector")
    return img
