import numpy as np
import pytest
from hashing_config import (
    get_selected_hashes,
    HASH_CATEGORIES,
    FEATURE_HASHES,
    _neighbor_count,
)


def _naive_neighbor_count(img):
    """Reference O(n^2) 8-neighbour count with zero padding at the borders."""
    out = np.zeros(img.shape, dtype=int)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            count = 0
            for i in range(max(0, y - 1), min(img.shape[0], y + 2)):
                for j in range(max(0, x - 1), min(img.shape[1], x + 2)):
                    if (i != y or j != x) and img[i, j]:
                        count += 1
            out[y, x] = count
    return out


def test_neighbor_count_matches_naive_reference():
    rng = np.random.default_rng(0)
    for _ in range(50):
        h, w = rng.integers(1, 12, size=2)
        img = (rng.random((h, w)) < 0.4).astype(np.uint8) * 255
        assert np.array_equal(_neighbor_count(img), _naive_neighbor_count(img))


def test_neighbor_count_treats_borders_as_background():
    # A single foreground pixel has no neighbours regardless of position.
    img = np.zeros((3, 3), dtype=np.uint8)
    img[0, 0] = 255
    assert _neighbor_count(img)[0, 0] == 0
    # A full 3x3 block: the centre pixel sees all 8 neighbours.
    full = np.full((3, 3), 255, dtype=np.uint8)
    assert _neighbor_count(full)[1, 1] == 8


def test_basic_preset():
    assert get_selected_hashes("basic") == HASH_CATEGORIES["basic"]


def test_all_preset():
    assert get_selected_hashes("all") == HASH_CATEGORIES["all"]


def test_none_preset():
    assert get_selected_hashes("none") == set()


def test_basic_minus_dhash():
    result = get_selected_hashes("basic,-dhash")
    expected = HASH_CATEGORIES["basic"] - {"dhash"}
    assert result == expected


def test_all_minus_feature():
    result = get_selected_hashes("all,-feature")
    expected = HASH_CATEGORIES["all"] - FEATURE_HASHES
    assert result == expected
