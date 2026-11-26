import numpy as np
import pytest

from groundwatch.data.augment import AugmentConfig
from groundwatch.data.dataset import DatasetConfig, FrameDataset
from groundwatch.data.manifest import fill_sizes, load_manifest
from groundwatch.data.splits import leave_one_drive_out
from groundwatch.data.tiling import Stitcher, Tile, TilingConfig, num_tiles


def labelled(synthetic):
    return [r for r in fill_sizes(load_manifest(synthetic)) if r.is_labelled]


def test_train_tiles_shapes_and_labels(synthetic, taxonomy):
    ds = FrameDataset(labelled(synthetic), taxonomy, DatasetConfig(tiling=TilingConfig(512, 128), crops_per_frame=4),
                      augment_cfg=AugmentConfig())
    assert len(ds) == 9 * 4
    item = ds[0]
    assert item["image"].shape == (3, 512, 512) and item["image"].dtype == np.float32
    assert item["label"].shape == (512, 512)
    vals = set(np.unique(item["label"]).tolist())
    assert vals <= set(range(taxonomy.num_classes)) | {255}


def test_train_crops_reproducible_and_change_per_epoch(synthetic, taxonomy):
    cfg = DatasetConfig(tiling=TilingConfig(256, 64), crops_per_frame=2, uniform_crop_prob=1.0)
    ds = FrameDataset(labelled(synthetic), taxonomy, cfg)
    a, b = ds[3]["tile"], ds[3]["tile"]
    ds.set_epoch(1)
    c = ds[3]["tile"]
    assert a == b and a != c


def test_eval_tiles_cover_every_frame(synthetic, taxonomy):
    rs = labelled(synthetic)
    tcfg = TilingConfig(512, 128)
    ds = FrameDataset(rs, taxonomy, DatasetConfig(mode="eval_tiles", tiling=tcfg))
    assert len(ds) == sum(num_tiles(r.width, r.height, tcfg) for r in rs)
    assert all(isinstance(ds[i]["void"], bool) for i in range(len(ds)))


def test_eval_tiles_stitch_back_to_label(synthetic, taxonomy):
    """Feed the ground truth as one-hot 'predictions' through tile -> stitch: must round-trip."""
    rs = labelled(synthetic)[:1]
    tcfg = TilingConfig(512, 128)
    ds = FrameDataset(rs, taxonomy, DatasetConfig(mode="eval_tiles", tiling=tcfg))
    r = rs[0]
    st = Stitcher(r.width, r.height, taxonomy.num_classes)
    full = np.full((r.height, r.width), 255, dtype=np.uint8)
    for i in range(len(ds)):
        it = ds[i]
        x, y, w, h = it["tile"]
        lab = it["label"][:h, :w]
        full[y : y + h, x : x + w] = lab
        probs = np.eye(taxonomy.num_classes, dtype=np.float32)[np.where(lab == 255, 0, lab)].transpose(2, 0, 1)
        st.add(Tile(x, y, w, h), probs)
    pred = st.labels()
    m = full != 255
    assert (pred[m] == full[m]).all()


def test_label_upsampled_to_image_size(synthetic, taxonomy):
    ds = FrameDataset(labelled(synthetic), taxonomy, DatasetConfig(mode="eval_tiles"))
    img, lab = ds._frame(0)
    assert lab.shape == img.shape[:2]


def test_downscaled_mode(synthetic, taxonomy):
    ds = FrameDataset(labelled(synthetic), taxonomy, DatasetConfig(mode="downscaled", downscale_long_side=600))
    it = ds[0]
    assert it["image"].shape == (3, 400, 600) and it["label"].shape == (400, 600)


def test_ssl_crops_include_unlabelled(synthetic):
    rs = fill_sizes(load_manifest(synthetic))
    ds = FrameDataset(rs, None, DatasetConfig(mode="ssl_crops", tiling=TilingConfig(256, 0), crops_per_frame=2))
    assert len(ds) == 24 and ds[0]["label"] is None


def test_train_rejects_unlabelled(synthetic, taxonomy):
    with pytest.raises(ValueError, match="labelled"):
        FrameDataset(fill_sizes(load_manifest(synthetic)), taxonomy, DatasetConfig())


def test_folds_feed_datasets(synthetic, taxonomy):
    folds = leave_one_drive_out(load_manifest(synthetic))
    assert len(folds) == 3
    for f in folds:
        tr = FrameDataset(f.train, taxonomy, DatasetConfig(tiling=TilingConfig(256, 64), crops_per_frame=1))
        assert {tr[i]["drive_id"] for i in range(len(tr))}.isdisjoint({f.test_drive, f.val_drive})
