"""Train-time-only augmentation (never runs at inference).

Photometric jitter for the drive-by lighting channel, plus geometric flips applied to
image and label together. Deliberately light: heavy, irreversible enhancement (CLAHE,
sharpening, tone-mapping) is avoided per compute-and-data-pipeline.md.

ASSUMPTION: horizontal flips are safe (left and right walls look alike). Vertical flips
are off by default, because the back (roof) and the floor are different classes and the
frame has a strong top/bottom structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    hflip: bool = True
    vflip: bool = False
    brightness: float = 0.25  # multiplicative, +/- fraction
    contrast: float = 0.25
    noise_std: float = 0.01
    enabled: bool = True


def augment(
    image: np.ndarray, label: np.ndarray | None, rng: np.random.Generator, cfg: AugmentConfig
) -> tuple[np.ndarray, np.ndarray | None]:
    """``image`` float32 HxWxC in [0, 1]; ``label`` HxW (or None for unlabelled/SSL)."""
    if not cfg.enabled:
        return image, label
    if cfg.hflip and rng.random() < 0.5:
        image = image[:, ::-1]
        label = label[:, ::-1] if label is not None else None
    if cfg.vflip and rng.random() < 0.5:
        image = image[::-1]
        label = label[::-1] if label is not None else None
    if cfg.brightness:
        image = image * (1.0 + rng.uniform(-cfg.brightness, cfg.brightness))
    if cfg.contrast:
        mean = image.mean()
        image = (image - mean) * (1.0 + rng.uniform(-cfg.contrast, cfg.contrast)) + mean
    if cfg.noise_std:
        image = image + rng.normal(0.0, cfg.noise_std, image.shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    # TODO: motion blur and cutout (named in the pipeline doc) once a blur kernel model of
    # the rig's drive speed / exposure is known.
    return np.ascontiguousarray(image), (np.ascontiguousarray(label) if label is not None else None)
