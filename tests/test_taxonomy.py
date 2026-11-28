import numpy as np
import pytest

from groundwatch.data.taxonomy import ClassDef, Taxonomy


def test_phase1_taxonomy_loads(taxonomy):
    assert taxonomy.num_classes == 8
    assert taxonomy.class_names[0] == "mesh"
    assert taxonomy.ignore_index == 255


def test_encode_maps_and_ignores(taxonomy):
    src = np.array([[1, 2, 0], [255, 22, 21]], dtype=np.uint8)
    out = taxonomy.encode(src)
    assert out.tolist() == [[0, 1, 255], [255, 7, 255]]


def test_strict_rejects_unknown_ids(taxonomy):
    with pytest.raises(ValueError, match="unmapped"):
        taxonomy.encode(np.array([[99]], dtype=np.uint8))


def test_non_strict_sends_unknown_to_ignore():
    t = Taxonomy("t", [ClassDef("a", 0, (1,))], strict=False)
    assert t.encode(np.array([[1, 9]], dtype=np.uint8)).tolist() == [[0, 255]]


def test_index_gaps_rejected():
    with pytest.raises(ValueError):
        Taxonomy("t", [ClassDef("a", 0, (1,)), ClassDef("b", 2, (2,))])


def test_double_mapping_rejected():
    with pytest.raises(ValueError):
        Taxonomy("t", [ClassDef("a", 0, (1,)), ClassDef("b", 1, (1,))])
