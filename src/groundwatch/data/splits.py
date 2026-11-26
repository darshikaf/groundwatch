"""Leave-one-drive-out cross-validation.

Frames from the same drive are strongly correlated (same rock, same lighting, same
support install, and consecutive video frames are near-duplicates). The MVP showed that
random frame splits inflate scores, so every split here is grouped by drive: each fold
holds out one whole drive as test, and no frame from that drive appears in train or val.

Validation (for early stopping / model selection) is a second held-out drive, chosen by
rotation so each drive serves as val once. With fewer than three drives there is no
room for a separate val drive and val is left empty; the caller then has to select on
a fixed epoch budget rather than peeking at test.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .manifest import FrameRecord, drives


@dataclass(frozen=True)
class Fold:
    index: int
    test_drive: str
    val_drive: str | None
    train: tuple[FrameRecord, ...]
    val: tuple[FrameRecord, ...]
    test: tuple[FrameRecord, ...]

    def summary(self) -> str:
        return (
            f"fold {self.index}: test={self.test_drive} ({len(self.test)} frames), "
            f"val={self.val_drive or '-'} ({len(self.val)}), train={len(self.train)}"
        )


def leave_one_drive_out(
    records: list[FrameRecord],
    *,
    use_val_drive: bool = True,
    labelled_only: bool = True,
    min_frames_per_drive: int = 1,
) -> list[Fold]:
    """Build one fold per drive.

    ``labelled_only`` drops unlabelled frames (they feed SSL, not the supervised head).
    Drives with fewer than ``min_frames_per_drive`` frames are never used as a test or val
    drive (too few frames for a stable score) but still contribute training frames.
    """
    pool = [r for r in records if r.is_labelled] if labelled_only else list(records)
    if not pool:
        raise ValueError("no frames to split" + (" (no labelled frames)" if labelled_only else ""))
    counts = Counter(r.drive_id for r in pool)
    all_drives = drives(pool)
    eligible = [d for d in all_drives if counts[d] >= min_frames_per_drive]
    if len(all_drives) < 2:
        raise ValueError(
            f"leave-one-drive-out needs at least 2 drives, got {all_drives}. "
            "Check drive_id in the manifest."
        )
    if not eligible:
        raise ValueError("no drive has enough frames to be held out")

    folds = []
    for i, test_drive in enumerate(eligible):
        val_drive = None
        if use_val_drive and len(all_drives) >= 3:
            # Rotate through the other eligible drives so val is spread across folds.
            others = [d for d in eligible if d != test_drive]
            if others:
                val_drive = others[i % len(others)]
        train = tuple(r for r in pool if r.drive_id not in (test_drive, val_drive))
        if not train:
            raise ValueError(f"fold {i} has no training frames")
        folds.append(
            Fold(
                index=i,
                test_drive=test_drive,
                val_drive=val_drive,
                train=train,
                val=tuple(r for r in pool if r.drive_id == val_drive),
                test=tuple(r for r in pool if r.drive_id == test_drive),
            )
        )
    for f in folds:
        assert_no_drive_leakage(f)
    return folds


def assert_no_drive_leakage(fold: Fold) -> None:
    """Fail loudly if any drive appears in more than one of train / val / test."""
    tr = {r.drive_id for r in fold.train}
    va = {r.drive_id for r in fold.val}
    te = {r.drive_id for r in fold.test}
    overlap = (tr & va) | (tr & te) | (va & te)
    if overlap:
        raise AssertionError(f"fold {fold.index}: drive leakage across splits: {sorted(overlap)}")
