import pytest
from hashing_config import get_selected_hashes, HASH_CATEGORIES, FEATURE_HASHES


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
