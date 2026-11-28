"""Frame manifest: the one table that says which frames exist and which drive each came from.

Full-resolution frames on disk stay the source of truth (compute-and-data-pipeline.md:
tiles are cut on the fly, never materialised). The manifest is a CSV with one row per
frame. Required columns:

    frame_id   unique id for the frame
    drive_id   the drive (tunnel) the frame was captured in; the split grouping key
    image      path to the pre-processed frame, relative to the manifest's directory

Optional columns:

    label      path to a single-channel label raster of *source* ids (see taxonomy.py)
    pass_id    survey pass / capture session (repeat passes of one drive share drive_id)
    source     capture channel, e.g. dslr, phone, video, rig
    width, height  frame size in pixels; filled from the image if missing

ASSUMPTION: a "drive" is one physical tunnel section. Repeat passes over the same drive
are grouped under the same drive_id so a held-out drive is never seen in training via an
earlier or later pass. If the team wants pass-level grouping instead, split on pass_id.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

REQUIRED_COLUMNS = ("frame_id", "drive_id", "image")


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    drive_id: str
    image: Path
    label: Path | None = None
    pass_id: str | None = None
    source: str | None = None
    width: int | None = None
    height: int | None = None

    @property
    def is_labelled(self) -> bool:
        return self.label is not None


def _opt(row: dict, key: str) -> str | None:
    v = row.get(key)
    if v is None:
        return None
    v = v.strip()
    return v or None


def load_manifest(path: str | Path, root: str | Path | None = None) -> list[FrameRecord]:
    """Read a manifest CSV. Relative paths resolve against ``root`` (default: the CSV's dir)."""
    path = Path(path)
    base = Path(root) if root is not None else path.parent
    records: list[FrameRecord] = []
    seen: set[str] = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        for lineno, row in enumerate(reader, start=2):
            fid = (row["frame_id"] or "").strip()
            drive = (row["drive_id"] or "").strip()
            if not fid or not drive:
                raise ValueError(f"{path}:{lineno}: frame_id and drive_id are required")
            if fid in seen:
                raise ValueError(f"{path}:{lineno}: duplicate frame_id {fid!r}")
            seen.add(fid)
            label = _opt(row, "label")
            w, h = _opt(row, "width"), _opt(row, "height")
            records.append(
                FrameRecord(
                    frame_id=fid,
                    drive_id=drive,
                    image=base / row["image"].strip(),
                    label=(base / label) if label else None,
                    pass_id=_opt(row, "pass_id"),
                    source=_opt(row, "source"),
                    width=int(w) if w else None,
                    height=int(h) if h else None,
                )
            )
    return records


def fill_sizes(records: list[FrameRecord]) -> list[FrameRecord]:
    """Fill missing width/height by reading image headers (cheap, no pixel decode)."""
    from PIL import Image

    out = []
    for r in records:
        if r.width is None or r.height is None:
            with Image.open(r.image) as im:
                r = replace(r, width=im.width, height=im.height)
        out.append(r)
    return out


def drives(records: list[FrameRecord]) -> list[str]:
    return sorted({r.drive_id for r in records})
