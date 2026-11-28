"""Synthetic frames + labels + manifest, for tests and smoke runs. Not real imagery."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image


def make_synthetic_dataset(
    out_dir: str | Path,
    n_drives: int = 3,
    frames_per_drive: int = 3,
    size: tuple[int, int] = (1500, 1000),  # (w, h)
    label_scale: float = 0.5,  # labels drawn at reduced resolution, like the real ones
    source_ids: tuple[int, ...] = (1, 2, 3, 4, 5),
    unlabelled_per_drive: int = 1,
    seed: int = 0,
) -> Path:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, h = size
    lw, lh = round(w * label_scale), round(h * label_scale)
    rows = []
    for d in range(n_drives):
        for f in range(frames_per_drive + unlabelled_per_drive):
            fid = f"d{d}_f{f:03d}"
            img = (rng.random((h, w, 3)) * 200).astype(np.uint8)
            img[: h // 6] = 0  # an unlit band along the top, like the black falloff
            Image.fromarray(img).save(out / "images" / f"{fid}.png")
            row = {"frame_id": fid, "drive_id": f"drive_{d}", "image": f"images/{fid}.png", "label": "", "source": "synthetic"}
            if f < frames_per_drive:
                lab = np.zeros((lh, lw), dtype=np.uint8)  # 0 = unlabelled -> ignore
                bands = np.array_split(np.arange(lw), len(source_ids))
                for sid, cols in zip(source_ids, bands):
                    lab[lh // 6 :, cols] = sid
                lab[: lh // 6] = 255  # void band drawn as ignore
                Image.fromarray(lab).save(out / "labels" / f"{fid}.png")
                row["label"] = f"labels/{fid}.png"
            rows.append(row)
    manifest = out / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["frame_id", "drive_id", "image", "label", "source"])
        wr.writeheader()
        wr.writerows(rows)
    return manifest
