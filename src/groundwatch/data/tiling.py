"""Tiling for 24 MP frames (docs/compute-and-data-pipeline.md, "Tiling, explained from scratch").

Three ways of cutting a frame, one per job:

* ``tile_grid``           systematic, overlapping, full coverage. Used at inference and for
                          evaluation, then recombined with ``Stitcher``.
* ``random_crop``         uniform random crops. Used for SSL pretraining.
* ``LabelAwareSampler``   crops centred on labelled pixels, with rare classes oversampled.
                          Used for supervised head training, so a mostly-sound drive does
                          not drown rockfall and torn mesh in background tiles.

Every tile is described by a ``Tile`` (x, y offset in the full frame, plus size), so
predictions can be placed back by offset; ``tile_manifest`` gives the per-tile
bookkeeping table the doc recommends. Tiles are always cut on the fly from the full
frame and never written to disk.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    x: int  # left, in full-frame pixels
    y: int  # top
    w: int
    h: int

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y, self.y + self.h), slice(self.x, self.x + self.w)


@dataclass(frozen=True)
class TilingConfig:
    tile_size: int = 1024
    overlap: int = 256

    @property
    def stride(self) -> int:
        return self.tile_size - self.overlap

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0 <= self.overlap < self.tile_size:
            raise ValueError("overlap must be in [0, tile_size)")


def _axis_starts(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets along one axis. The last tile is pinned to the far edge, so the
    grid covers the frame exactly with no padding (its overlap with the previous tile
    is just larger). If the frame is smaller than a tile, one tile at 0 (caller pads)."""
    if length <= tile:
        return [0]
    n = math.ceil((length - tile) / stride + 1)
    starts = [min(i * stride, length - tile) for i in range(n)]
    return sorted(set(starts))


def num_tiles(width: int, height: int, cfg: TilingConfig) -> int:
    """Doc formula: ceil((W-T)/S + 1) * ceil((H-T)/S + 1). 6000x4000 @ 1024/256 -> 40."""
    return len(_axis_starts(width, cfg.tile_size, cfg.stride)) * len(
        _axis_starts(height, cfg.tile_size, cfg.stride)
    )


def tile_grid(width: int, height: int, cfg: TilingConfig) -> list[Tile]:
    """Row-major systematic grid covering the whole frame."""
    tw, th = min(cfg.tile_size, width), min(cfg.tile_size, height)
    return [
        Tile(x, y, tw, th)
        for y in _axis_starts(height, cfg.tile_size, cfg.stride)
        for x in _axis_starts(width, cfg.tile_size, cfg.stride)
    ]


def tile_manifest(frame_id: str, width: int, height: int, cfg: TilingConfig) -> list[dict]:
    return [{"frame_id": frame_id, "tile_index": i, **asdict(t)} for i, t in enumerate(tile_grid(width, height, cfg))]


def crop(array: np.ndarray, tile: Tile, pad_value: int | float = 0) -> np.ndarray:
    """Cut a tile from an HxW or HxWxC array, padding if the frame is smaller than the tile."""
    out = array[tile.slices]
    ph, pw = tile.h - out.shape[0], tile.w - out.shape[1]
    if ph or pw:
        pad = [(0, ph), (0, pw)] + [(0, 0)] * (array.ndim - 2)
        out = np.pad(out, pad, constant_values=pad_value)
    return out


def _clamp_tile(cx: int, cy: int, size: int, width: int, height: int) -> Tile:
    tw, th = min(size, width), min(size, height)
    x = int(np.clip(cx - tw // 2, 0, width - tw))
    y = int(np.clip(cy - th // 2, 0, height - th))
    return Tile(x, y, tw, th)


def random_crop(width: int, height: int, size: int, rng: np.random.Generator) -> Tile:
    tw, th = min(size, width), min(size, height)
    return Tile(int(rng.integers(0, width - tw + 1)), int(rng.integers(0, height - th + 1)), tw, th)


class LabelAwareSampler:
    """Pick crop centres on labelled pixels, choosing the class first.

    A class is drawn with probability proportional to ``class_freq ** -power`` over the
    classes present in this frame (power=0: uniform over present classes; power=1: fully
    inverse-frequency). Then a random pixel of that class is the crop centre, jittered so
    objects do not always sit dead-centre. With probability ``uniform_prob`` a plain random
    crop is returned instead, so background context is still seen.

    ``class_freq`` is the training-set pixel frequency per class; pass None for uniform.
    """

    def __init__(
        self,
        size: int,
        num_classes: int,
        ignore_index: int = 255,
        class_freq: np.ndarray | None = None,
        power: float = 0.5,
        uniform_prob: float = 0.2,
        jitter: float = 0.25,
    ) -> None:
        self.size = size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        freq = np.ones(num_classes) if class_freq is None else np.asarray(class_freq, dtype=float)
        if freq.shape != (num_classes,):
            raise ValueError("class_freq must have one entry per class")
        self.weights = np.where(freq > 0, np.maximum(freq, 1e-12) ** -power, 0.0)
        self.uniform_prob = uniform_prob
        self.jitter = jitter

    def sample(self, label: np.ndarray, rng: np.random.Generator) -> Tile:
        h, w = label.shape
        if rng.random() < self.uniform_prob:
            return random_crop(w, h, self.size, rng)
        present = np.zeros(self.num_classes, dtype=bool)
        vals = np.unique(label)
        vals = vals[vals != self.ignore_index]
        present[vals[vals < self.num_classes]] = True
        weights = self.weights * present
        if weights.sum() == 0:
            return random_crop(w, h, self.size, rng)
        cls = int(rng.choice(self.num_classes, p=weights / weights.sum()))
        ys, xs = np.nonzero(label == cls)
        k = int(rng.integers(len(ys)))
        j = int(self.jitter * self.size)
        cx = int(xs[k]) + (int(rng.integers(-j, j + 1)) if j else 0)
        cy = int(ys[k]) + (int(rng.integers(-j, j + 1)) if j else 0)
        return _clamp_tile(cx, cy, self.size, w, h)


def class_pixel_frequency(labels, num_classes: int, ignore_index: int = 255) -> np.ndarray:
    """Pixel frequency per class over an iterable of encoded label maps (ignore excluded)."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for lab in labels:
        v = lab[lab != ignore_index]
        counts += np.bincount(v.ravel(), minlength=num_classes)[:num_classes]
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def is_void_tile(
    image: np.ndarray,
    dark_threshold: float = 0.06,
    max_dark_fraction: float = 0.9,
    min_std: float = 0.01,
) -> bool:
    """True if a tile is mostly the unlit/void band (compute-and-data-pipeline.md).

    ``image`` is float in [0, 1] (HxW or HxWxC). A tile is void if more than
    ``max_dark_fraction`` of pixels are below ``dark_threshold`` luminance, or it is
    nearly flat (std below ``min_std``).

    ASSUMPTION: thresholds are placeholders until tuned on rig imagery after flat-field
    correction. They should be set so skipped tiles are ones an annotator would mark ignore.
    """
    lum = image.mean(axis=-1) if image.ndim == 3 else image
    return bool((lum < dark_threshold).mean() > max_dark_fraction or lum.std() < min_std)


def blend_window(h: int, w: int, kind: str = "linear", floor: float = 0.05) -> np.ndarray:
    """Per-pixel weight for stitching: high in the tile centre, low at the edges, so the
    seam is decided by whichever tile saw the pixel with more context."""
    if kind == "uniform":
        return np.ones((h, w), dtype=np.float32)
    if kind != "linear":
        raise ValueError(f"unknown window {kind!r}")

    def ramp(n: int) -> np.ndarray:
        r = np.minimum(np.arange(n) + 1, np.arange(n)[::-1] + 1).astype(np.float32)
        return r / r.max()

    win = np.outer(ramp(h), ramp(w))
    return np.maximum(win, floor).astype(np.float32)


class Stitcher:
    """Recombine per-tile class probabilities into one full-frame map.

    Overlap regions average the probabilities of all covering tiles (weighted by
    ``blend_window``), then argmax, which avoids the seams a per-tile argmax-and-paste
    leaves. This covers the semantic ("stuff") classes. Instance merging across tile
    boundaries (plates, bolts; Phase 2) is deliberately not done here; the docs point
    to SAHI for that.
    """

    def __init__(self, width: int, height: int, num_classes: int, window: str = "linear") -> None:
        self.width, self.height, self.num_classes = width, height, num_classes
        self.window = window
        self._acc = np.zeros((num_classes, height, width), dtype=np.float32)
        self._wsum = np.zeros((height, width), dtype=np.float32)

    def add(self, tile: Tile, probs: np.ndarray) -> None:
        """``probs``: C x tile.h x tile.w (softmax output for this tile; padding cropped off)."""
        c, h, w = probs.shape
        if c != self.num_classes:
            raise ValueError(f"expected {self.num_classes} channels, got {c}")
        h, w = min(h, self.height - tile.y), min(w, self.width - tile.x)
        win = blend_window(tile.h, tile.w, self.window)[:h, :w]
        ys, xs = slice(tile.y, tile.y + h), slice(tile.x, tile.x + w)
        self._acc[:, ys, xs] += probs[:, :h, :w] * win
        self._wsum[ys, xs] += win

    def probabilities(self) -> np.ndarray:
        if (self._wsum == 0).any():
            raise RuntimeError("some pixels were not covered by any tile")
        return self._acc / self._wsum

    def labels(self) -> np.ndarray:
        return self.probabilities().argmax(axis=0).astype(np.uint8)
