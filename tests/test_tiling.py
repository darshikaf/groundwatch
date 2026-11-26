import numpy as np
import pytest

from groundwatch.data.tiling import (
    LabelAwareSampler,
    Stitcher,
    TilingConfig,
    blend_window,
    crop,
    is_void_tile,
    num_tiles,
    tile_grid,
)


def test_doc_example_24mp_gives_40_tiles():
    cfg = TilingConfig(tile_size=1024, overlap=256)
    assert cfg.stride == 768
    assert num_tiles(6000, 4000, cfg) == 40
    assert len(tile_grid(6000, 4000, cfg)) == 40


@pytest.mark.parametrize("w,h,t,o", [(6000, 4000, 1024, 256), (1500, 1000, 512, 128), (1025, 1024, 1024, 0), (300, 200, 512, 64)])
def test_grid_covers_frame_exactly_and_stays_inside(w, h, t, o):
    cfg = TilingConfig(t, o)
    cover = np.zeros((h, w), dtype=int)
    for tile in tile_grid(w, h, cfg):
        assert tile.x >= 0 and tile.y >= 0 and tile.x + tile.w <= w and tile.y + tile.h <= h
        cover[tile.slices] += 1
    assert cover.min() >= 1


def test_neighbouring_tiles_overlap_at_least_configured_amount():
    tiles = tile_grid(6000, 4000, TilingConfig(1024, 256))
    xs = sorted({t.x for t in tiles})
    assert all(b - a <= 768 for a, b in zip(xs, xs[1:]))
    assert xs[-1] + 1024 == 6000


def test_crop_pads_small_frames():
    a = np.ones((100, 80), dtype=np.uint8)
    from groundwatch.data.tiling import Tile

    out = crop(a, Tile(0, 0, 128, 128), pad_value=255)
    assert out.shape == (128, 128)
    assert (out[100:] == 255).all() and (out[:, 80:] == 255).all()


def test_stitcher_reconstructs_constant_and_has_no_seams():
    w, h, c = 1500, 1000, 3
    truth = (np.arange(w)[None, :] * c // w).repeat(h, axis=0).astype(np.uint8)
    onehot = np.eye(c, dtype=np.float32)[truth].transpose(2, 0, 1)
    st = Stitcher(w, h, c)
    for t in tile_grid(w, h, TilingConfig(512, 128)):
        st.add(t, onehot[:, t.slices[0], t.slices[1]])
    assert (st.labels() == truth).all()
    np.testing.assert_allclose(st.probabilities().sum(axis=0), 1.0, rtol=1e-5)


def test_stitcher_averages_overlap():
    from groundwatch.data.tiling import Tile

    st = Stitcher(4, 1, 2, window="uniform")
    st.add(Tile(0, 0, 3, 1), np.stack([np.ones((1, 3)), np.zeros((1, 3))]))
    st.add(Tile(1, 0, 3, 1), np.stack([np.zeros((1, 3)), np.ones((1, 3))]))
    np.testing.assert_allclose(st.probabilities()[0, 0], [1, 0.5, 0.5, 0])


def test_stitcher_rejects_uncovered_pixels():
    from groundwatch.data.tiling import Tile

    st = Stitcher(10, 10, 2)
    st.add(Tile(0, 0, 5, 5), np.ones((2, 5, 5)))
    with pytest.raises(RuntimeError):
        st.labels()


def test_blend_window_peaks_in_centre():
    win = blend_window(64, 64)
    assert win[32, 32] == win.max() and win[0, 0] < win[32, 32]


def test_label_aware_sampler_oversamples_rare_class():
    lab = np.zeros((2000, 2000), dtype=np.uint8)
    lab[1900:, 1900:] = 1  # tiny rare class in a corner
    freq = np.array([0.9975, 0.0025])
    s = LabelAwareSampler(256, 2, class_freq=freq, power=1.0, uniform_prob=0.0, jitter=0.0)
    rng = np.random.default_rng(0)
    hits = sum((crop(lab, s.sample(lab, rng)) == 1).any() for _ in range(200))
    assert hits > 150  # inverse-frequency weighting picks the rare class nearly always


def test_label_aware_sampler_falls_back_when_all_ignore():
    lab = np.full((300, 300), 255, dtype=np.uint8)
    t = LabelAwareSampler(128, 3, uniform_prob=0.0).sample(lab, np.random.default_rng(0))
    assert t.w == 128 and t.h == 128


def test_void_tile_detection():
    assert is_void_tile(np.zeros((64, 64, 3), dtype=np.float32))
    rng = np.random.default_rng(0)
    assert not is_void_tile(rng.random((64, 64, 3)).astype(np.float32))
