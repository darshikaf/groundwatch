from pathlib import Path

import pytest

from groundwatch.data.synthetic import make_synthetic_dataset
from groundwatch.data.taxonomy import Taxonomy

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def taxonomy():
    return Taxonomy.from_yaml(REPO / "configs/taxonomy/phase1_surface.yaml")


@pytest.fixture
def synthetic(tmp_path):
    return make_synthetic_dataset(tmp_path / "data", n_drives=3, frames_per_drive=3)
