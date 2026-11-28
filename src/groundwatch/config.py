"""Experiment config: one YAML per experiment under configs/experiments/."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .data.augment import AugmentConfig
from .data.dataset import DatasetConfig
from .data.tiling import TilingConfig


@dataclass
class ExperimentConfig:
    name: str
    manifest: Path
    taxonomy: Path
    train: DatasetConfig = field(default_factory=DatasetConfig)
    eval: DatasetConfig = field(default_factory=lambda: DatasetConfig(mode="eval_tiles"))
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    use_val_drive: bool = True
    min_frames_per_drive: int = 1
    model: dict = field(default_factory=dict)  # opaque until the model code lands


def _dataset_cfg(d: dict | None, default_mode: str) -> DatasetConfig:
    d = dict(d or {})
    tiling = TilingConfig(**d.pop("tiling", {}))
    d.setdefault("mode", default_mode)
    return DatasetConfig(tiling=tiling, **d)


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Relative paths inside the YAML resolve against the repo root (the YAML is expected
    at configs/experiments/<name>.yaml); absolute paths are used as-is."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    root = path.resolve().parent.parent.parent

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else root / q

    split = cfg.get("split", {})
    return ExperimentConfig(
        name=cfg["name"],
        manifest=resolve(cfg["data"]["manifest"]),
        taxonomy=resolve(cfg["data"]["taxonomy"]),
        train=_dataset_cfg(cfg.get("train_data"), "train_tiles"),
        eval=_dataset_cfg(cfg.get("eval_data"), "eval_tiles"),
        augment=AugmentConfig(**cfg.get("augment", {})),
        use_val_drive=bool(split.get("use_val_drive", True)),
        min_frames_per_drive=int(split.get("min_frames_per_drive", 1)),
        model=cfg.get("model", {}),
    )
