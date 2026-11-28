import shutil

from groundwatch.config import load_experiment
from groundwatch.train import plan

from .conftest import REPO


def test_experiment_config_loads():
    exp = load_experiment(REPO / "configs/experiments/phase1_surface_seg.yaml")
    assert exp.train.mode == "train_tiles" and exp.eval.mode == "eval_tiles"
    assert exp.train.tiling.tile_size == 1024 and exp.train.tiling.overlap == 256
    assert exp.taxonomy.exists()


def test_plan_runs_end_to_end(tmp_path, synthetic, capsys):
    # Lay out a throwaway repo root so the config's relative paths resolve.
    (tmp_path / "configs/experiments").mkdir(parents=True)
    shutil.copytree(REPO / "configs/taxonomy", tmp_path / "configs/taxonomy")
    cfg = (REPO / "configs/experiments/phase1_surface_seg.yaml").read_text()
    cfg = cfg.replace("min_frames_per_drive: 5", "min_frames_per_drive: 1")
    (tmp_path / "configs/experiments/e.yaml").write_text(cfg)
    folds = plan(str(tmp_path / "configs/experiments/e.yaml"), stats=True)
    assert len(folds) == 3
    out = capsys.readouterr().out
    assert "fold 0" in out and "train class freq" in out
