"""Class taxonomy, loaded from config rather than hard-coded.

Several class decisions are still open with the geotech (see the SME checklist in
docs/labelling-rule-book.md): whether water / efflorescence is a class, whether open
drive ahead is ``other`` or ``ignore``, where brick bulkheads go, and so on. To keep the
training code independent of those decisions, label rasters store **source ids** (the
ids the labelling export writes), and a taxonomy YAML maps each source id to a training
index or to ignore. Changing a decision means editing the YAML, not the code, and in most
cases not relabelling either (merges and "send to ignore" are pure remaps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

IGNORE_INDEX = 255  # Cityscapes convention, per the Phase-1 labelling approach.


@dataclass(frozen=True)
class ClassDef:
    name: str
    train_index: int
    source_ids: tuple[int, ...]


@dataclass
class Taxonomy:
    name: str
    classes: list[ClassDef]
    ignore_index: int = IGNORE_INDEX
    ignore_source_ids: tuple[int, ...] = ()
    # Source ids with no mapping go to ignore (safe default) unless strict is set.
    strict: bool = True
    _lut: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        indices = sorted(c.train_index for c in self.classes)
        if indices != list(range(len(self.classes))):
            raise ValueError(
                f"train_index values must be 0..{len(self.classes) - 1} with no gaps, got {indices}"
            )
        if self.ignore_index < len(self.classes):
            raise ValueError("ignore_index collides with a class index")
        lut = np.full(256, -1, dtype=np.int32)
        for c in self.classes:
            for sid in c.source_ids:
                if lut[sid] != -1:
                    raise ValueError(f"source id {sid} mapped twice")
                lut[sid] = c.train_index
        for sid in self.ignore_source_ids:
            if lut[sid] != -1:
                raise ValueError(f"source id {sid} mapped to a class and to ignore")
            lut[sid] = self.ignore_index
        self._lut = lut

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_names(self) -> list[str]:
        return [c.name for c in sorted(self.classes, key=lambda c: c.train_index)]

    def encode(self, source_label: np.ndarray) -> np.ndarray:
        """Map a raster of source ids (uint8) to training indices, ignore = ``ignore_index``."""
        if source_label.dtype != np.uint8:
            if source_label.min() < 0 or source_label.max() > 255:
                raise ValueError("source label ids must fit in uint8")
            source_label = source_label.astype(np.uint8)
        out = self._lut[source_label]
        unmapped = out == -1
        if unmapped.any():
            if self.strict:
                bad = np.unique(source_label[unmapped]).tolist()
                raise ValueError(f"unmapped source ids in label: {bad}")
            out = np.where(unmapped, self.ignore_index, out)
        return out.astype(np.uint8)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Taxonomy":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        classes = [
            ClassDef(
                name=c["name"],
                train_index=int(c["train_index"]),
                source_ids=tuple(int(s) for s in c["source_ids"]),
            )
            for c in cfg["classes"]
        ]
        return cls(
            name=cfg.get("name", Path(path).stem),
            classes=classes,
            ignore_index=int(cfg.get("ignore_index", IGNORE_INDEX)),
            ignore_source_ids=tuple(int(s) for s in cfg.get("ignore_source_ids", [])),
            strict=bool(cfg.get("strict", True)),
        )
