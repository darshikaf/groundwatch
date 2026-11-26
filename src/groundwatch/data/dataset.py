"""Frame / tile datasets over a manifest. numpy-only; ``as_torch_dataset`` wraps for PyTorch.

Four modes, matching the jobs in compute-and-data-pipeline.md:

  train_tiles   label-aware random crops from full-res labelled frames (head training)
  eval_tiles    systematic overlapping grid over full-res frames (val / test / inference)
  downscaled    whole frame resized so the long side is ``downscale_long_side`` (Phase-1
                surface seg may skip tiling entirely, per the pipeline doc)
  ssl_crops     uniform random crops, no labels (self-supervised pretraining)

Labels are drawn on reduced-resolution frames (~1500-2000 px, labelling rule book §7)
while images are full-res. ASSUMPTION: a label raster whose size differs from its image
is upsampled to image size with nearest-neighbour before tiling, provided the aspect
ratios agree within 1%. That keeps tile offsets in full-frame pixels everywhere.

Each item is a dict:
    image     float32, C x H x W, [0, 1]
    label     uint8 H x W of training indices (ignore = taxonomy.ignore_index), or None
    frame_id, drive_id
    tile      (x, y, w, h) in full-frame pixels (the whole frame for ``downscaled``)
    void      bool, eval_tiles only: tile is mostly unlit band and can be skipped
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .augment import AugmentConfig, augment
from .manifest import FrameRecord
from .preprocess import PreprocessConfig, preprocess
from .taxonomy import Taxonomy
from .tiling import (
    LabelAwareSampler,
    Tile,
    TilingConfig,
    crop,
    is_void_tile,
    random_crop,
    tile_grid,
)

MODES = ("train_tiles", "eval_tiles", "downscaled", "ssl_crops")

Image.MAX_IMAGE_PIXELS = 200_000_000  # 24 MP stereo mosaics are fine; keep PIL's bomb check above that


@dataclass
class DatasetConfig:
    mode: str = "train_tiles"
    tiling: TilingConfig = field(default_factory=TilingConfig)
    crops_per_frame: int = 8  # train_tiles / ssl_crops: items per frame per epoch
    downscale_long_side: int = 1536
    class_freq_power: float = 0.5
    uniform_crop_prob: float = 0.2
    max_crop_ignore_fraction: float = 0.8  # resample train crops that are mostly ignore
    max_resample: int = 5
    cache_frames: int = 4  # decoded full-res frames kept in memory (per worker)
    seed: int = 0


def _load_rgb(path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode not in ("RGB", "I;16", "L"):
            im = im.convert("RGB")
        arr = np.asarray(im)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return arr


def _load_label(path, size: tuple[int, int]) -> np.ndarray:
    """Load a source-id raster and bring it to ``size`` = (w, h) with nearest-neighbour."""
    with Image.open(path) as im:
        if im.mode not in ("L", "P"):
            raise ValueError(f"{path}: label must be single-channel 8-bit, got mode {im.mode}")
        if im.mode == "P":
            im = Image.fromarray(np.asarray(im))  # palette indices are the ids
        if im.size != size:
            lw, lh = im.size
            if abs(lw / lh - size[0] / size[1]) > 0.01 * (size[0] / size[1]):
                raise ValueError(f"{path}: label aspect {im.size} does not match image {size}")
            im = im.resize(size, Image.Resampling.NEAREST)
        return np.asarray(im)


class FrameDataset:
    def __init__(
        self,
        records: list[FrameRecord] | tuple[FrameRecord, ...],
        taxonomy: Taxonomy | None,
        cfg: DatasetConfig = DatasetConfig(),
        preprocess_cfg: PreprocessConfig = PreprocessConfig(),
        augment_cfg: AugmentConfig | None = None,
        class_freq: np.ndarray | None = None,
    ) -> None:
        if cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        needs_labels = cfg.mode in ("train_tiles", "downscaled")
        if needs_labels and taxonomy is None:
            raise ValueError(f"mode {cfg.mode} needs a taxonomy")
        if cfg.mode == "train_tiles":
            missing = [r.frame_id for r in records if not r.is_labelled]
            if missing:
                raise ValueError(f"train_tiles needs labelled frames; unlabelled: {missing[:5]}")
        self.records = list(records)
        self.taxonomy = taxonomy
        self.cfg = cfg
        self.preprocess_cfg = preprocess_cfg
        self.augment_cfg = augment_cfg
        self.epoch = 0
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()
        self._sampler = (
            LabelAwareSampler(
                cfg.tiling.tile_size,
                taxonomy.num_classes,
                taxonomy.ignore_index,
                class_freq=class_freq,
                power=cfg.class_freq_power,
                uniform_prob=cfg.uniform_crop_prob,
            )
            if cfg.mode == "train_tiles"
            else None
        )
        self._index: list[tuple[int, Tile | None]] = self._build_index()

    # -- indexing -------------------------------------------------------------------

    def _build_index(self) -> list[tuple[int, Tile | None]]:
        if self.cfg.mode == "eval_tiles":
            index = []
            for i, r in enumerate(self.records):
                if r.width is None or r.height is None:
                    raise ValueError("eval_tiles needs width/height; call manifest.fill_sizes first")
                index += [(i, t) for t in tile_grid(r.width, r.height, self.cfg.tiling)]
            return index
        if self.cfg.mode == "downscaled":
            return [(i, None) for i in range(len(self.records))]
        return [(i, None) for i in range(len(self.records)) for _ in range(self.cfg.crops_per_frame)]

    def set_epoch(self, epoch: int) -> None:
        """Random crops are a function of (seed, epoch, item), so runs are reproducible
        and each epoch sees fresh crops. Call once per epoch from the training loop."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self._index)

    # -- loading --------------------------------------------------------------------

    def _frame(self, i: int) -> tuple[np.ndarray, np.ndarray | None]:
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        r = self.records[i]
        img = preprocess(_load_rgb(r.image), self.preprocess_cfg)
        lab = None
        if r.label is not None and self.taxonomy is not None and self.cfg.mode != "ssl_crops":
            lab = self.taxonomy.encode(_load_label(r.label, (img.shape[1], img.shape[0])))
        self._cache[i] = (img, lab)
        while len(self._cache) > self.cfg.cache_frames:
            self._cache.popitem(last=False)
        return img, lab

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng([self.cfg.seed, self.epoch, idx])

    def __getitem__(self, idx: int) -> dict:
        fi, tile = self._index[idx]
        r = self.records[fi]
        img, lab = self._frame(fi)
        h, w = img.shape[:2]
        rng = self._rng(idx)
        void = None
        ignore = self.taxonomy.ignore_index if self.taxonomy else 255

        if self.cfg.mode == "downscaled":
            img, lab = self._downscale(img, lab)
            tile = Tile(0, 0, w, h)
        elif self.cfg.mode == "eval_tiles":
            void = is_void_tile(crop(img, tile))
            img = crop(img, tile)
            lab = crop(lab, tile, pad_value=ignore) if lab is not None else None
        elif self.cfg.mode == "ssl_crops":
            tile = random_crop(w, h, self.cfg.tiling.tile_size, rng)
            img, lab = crop(img, tile), None
        else:  # train_tiles
            for _ in range(self.cfg.max_resample):
                tile = self._sampler.sample(lab, rng)
                if (crop(lab, tile, ignore) == ignore).mean() <= self.cfg.max_crop_ignore_fraction:
                    break
            img, lab = crop(img, tile), crop(lab, tile, pad_value=ignore)

        # Pad to full tile size when the frame is smaller than a tile (rare; tests, phone stills).
        if self.cfg.mode in ("train_tiles", "eval_tiles", "ssl_crops"):
            t = self.cfg.tiling.tile_size
            full = Tile(0, 0, t, t)
            img = crop(img, full)
            lab = crop(lab, full, pad_value=ignore) if lab is not None else None

        if self.augment_cfg is not None and self.cfg.mode in ("train_tiles", "ssl_crops"):
            img, lab = augment(img, lab, rng, self.augment_cfg)

        item = {
            "image": np.ascontiguousarray(img.transpose(2, 0, 1)),
            "label": lab,
            "frame_id": r.frame_id,
            "drive_id": r.drive_id,
            "tile": (tile.x, tile.y, tile.w, tile.h),
        }
        if void is not None:
            item["void"] = void
        return item

    def _downscale(self, img: np.ndarray, lab: np.ndarray | None):
        h, w = img.shape[:2]
        s = self.cfg.downscale_long_side / max(h, w)
        size = (max(1, round(w * s)), max(1, round(h * s)))
        # Resize each channel in float ("F") mode so 10-12-bit dynamic range survives.
        out_img = np.stack(
            [
                np.asarray(Image.fromarray(np.ascontiguousarray(img[..., c], dtype=np.float32)).resize(size, Image.Resampling.BOX))
                for c in range(img.shape[2])
            ],
            axis=-1,
        ).astype(np.float32)
        out_lab = None
        if lab is not None:
            out_lab = np.asarray(Image.fromarray(lab).resize(size, Image.Resampling.NEAREST))
        return out_img, out_lab


def as_torch_dataset(ds: FrameDataset):
    """Wrap for torch.utils.data.DataLoader. Requires the ``train`` extra (torch)."""
    import torch
    from torch.utils.data import Dataset

    class _TorchFrameDataset(Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, idx):
            item = ds[idx]
            out = dict(item)
            out["image"] = torch.from_numpy(item["image"])
            if item["label"] is not None:
                out["label"] = torch.from_numpy(item["label"].astype(np.int64))
            return out

        def set_epoch(self, epoch: int) -> None:
            ds.set_epoch(epoch)

    return _TorchFrameDataset()
