"""Training entry point.

Today this only plans: it loads the experiment, builds the leave-one-drive-out folds,
and reports frames, tiles and class balance per fold, so the data side can be checked
before any model exists. ``--train`` is where the per-fold training loop goes.

    groundwatch-plan configs/experiments/phase1_surface_seg.yaml [--stats]
"""

from __future__ import annotations

import argparse
import sys

from .config import load_experiment
from .data.dataset import FrameDataset
from .data.manifest import drives, fill_sizes, load_manifest
from .data.splits import leave_one_drive_out
from .data.taxonomy import Taxonomy
from .data.tiling import class_pixel_frequency, num_tiles


def plan(config_path: str, stats: bool = False) -> list:
    exp = load_experiment(config_path)
    taxonomy = Taxonomy.from_yaml(exp.taxonomy)
    records = fill_sizes(load_manifest(exp.manifest))
    labelled = [r for r in records if r.is_labelled]
    print(f"experiment: {exp.name}")
    print(f"taxonomy:   {taxonomy.name} ({taxonomy.num_classes} classes: {', '.join(taxonomy.class_names)})")
    print(f"frames:     {len(records)} total, {len(labelled)} labelled, drives: {', '.join(drives(records))}")

    folds = leave_one_drive_out(
        records, use_val_drive=exp.use_val_drive, min_frames_per_drive=exp.min_frames_per_drive
    )
    for fold in folds:
        print(fold.summary())
        if exp.eval.mode == "eval_tiles":
            n = sum(num_tiles(r.width, r.height, exp.eval.tiling) for r in fold.test)
            print(f"  test tiles: {n} ({exp.eval.tiling.tile_size}px, overlap {exp.eval.tiling.overlap})")
        if stats:
            ds = FrameDataset(fold.train, taxonomy, exp.train)
            freq = class_pixel_frequency(
                (ds._frame(i)[1] for i in range(len(ds.records))), taxonomy.num_classes, taxonomy.ignore_index
            )
            print("  train class freq: " + ", ".join(f"{n}={f:.3f}" for n, f in zip(taxonomy.class_names, freq)))
    return folds


def train(config_path: str) -> None:
    # Per fold: class_pixel_frequency on fold.train -> FrameDataset(train_tiles, class_freq=...)
    # -> as_torch_dataset -> DataLoader; eval with FrameDataset(eval_tiles) + Stitcher per
    # frame, mIoU on the held-out drive; report mean and per-drive spread across folds.
    raise NotImplementedError("model and training loop are not part of the skeleton yet")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config")
    p.add_argument("--stats", action="store_true", help="read labels and report class balance per fold")
    p.add_argument("--train", action="store_true", help="run training (not implemented yet)")
    args = p.parse_args(argv)
    if args.train:
        train(args.config)
    else:
        plan(args.config, stats=args.stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
