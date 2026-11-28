from pathlib import Path

import pytest

from groundwatch.data.manifest import FrameRecord, load_manifest
from groundwatch.data.splits import Fold, assert_no_drive_leakage, leave_one_drive_out


def recs(spec):
    out = []
    for drive, n in spec.items():
        for i in range(n):
            out.append(FrameRecord(f"{drive}_{i}", drive, Path("x.png"), Path("y.png")))
    return out


def test_one_fold_per_drive_no_leakage():
    folds = leave_one_drive_out(recs({"a": 4, "b": 3, "c": 5, "d": 2}))
    assert [f.test_drive for f in folds] == ["a", "b", "c", "d"]
    for f in folds:
        assert {r.drive_id for r in f.test} == {f.test_drive}
        assert f.val_drive and f.val_drive != f.test_drive
        assert f.test_drive not in {r.drive_id for r in f.train}
        assert f.val_drive not in {r.drive_id for r in f.train}
        assert len(f.train) + len(f.val) + len(f.test) == 14


def test_val_drive_rotates():
    folds = leave_one_drive_out(recs({"a": 1, "b": 1, "c": 1, "d": 1}))
    assert len({f.val_drive for f in folds}) > 1


def test_two_drives_has_no_val():
    folds = leave_one_drive_out(recs({"a": 3, "b": 3}))
    assert all(f.val_drive is None and not f.val for f in folds)


def test_single_drive_refused():
    with pytest.raises(ValueError, match="at least 2 drives"):
        leave_one_drive_out(recs({"a": 10}))


def test_small_drives_train_only():
    folds = leave_one_drive_out(recs({"a": 10, "b": 10, "tiny": 1}), min_frames_per_drive=5)
    assert [f.test_drive for f in folds] == ["a", "b"]
    assert all(f.val_drive != "tiny" for f in folds)
    assert all(any(r.drive_id == "tiny" for r in f.train) for f in folds)


def test_unlabelled_frames_excluded_by_default():
    rs = recs({"a": 2, "b": 2}) + [FrameRecord("u", "a", Path("x.png"))]
    folds = leave_one_drive_out(rs)
    assert all("u" not in {r.frame_id for r in f.train + f.test} for f in folds)


def test_leakage_check_catches_overlap():
    r = recs({"a": 1})[0]
    with pytest.raises(AssertionError):
        assert_no_drive_leakage(Fold(0, "a", None, (r,), (), (r,)))


def test_manifest_loading(synthetic):
    rs = load_manifest(synthetic)
    assert len(rs) == 12
    assert sum(r.is_labelled for r in rs) == 9
    assert rs[0].image.is_absolute() or rs[0].image.exists()


def test_manifest_rejects_duplicates(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("frame_id,drive_id,image\nf1,a,x.png\nf1,b,y.png\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(p)


def test_manifest_requires_drive(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("frame_id,image\nf1,x.png\n")
    with pytest.raises(ValueError, match="drive_id"):
        load_manifest(p)
