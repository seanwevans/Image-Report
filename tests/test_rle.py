import numpy as np

from utils import rle_encode, rle_decode


def test_encode_basic():
    assert rle_encode([0, 0, 0, 5, 5, 2]) == "0:3 5:2 2:1"


def test_encode_empty():
    assert rle_encode([]) == ""


def test_decode_empty():
    assert rle_decode("") == []
    assert rle_decode("   ") == []


def test_round_trip_simple():
    values = [0, 0, 10, 10, 10, 3, 0, 0, 0, 7]
    assert rle_decode(rle_encode(values)) == values


def test_round_trip_random():
    rng = np.random.default_rng(0)
    for _ in range(20):
        # Small alphabet so runs actually occur, like a projection profile.
        values = rng.integers(0, 4, size=int(rng.integers(0, 200))).tolist()
        assert rle_decode(rle_encode(values)) == values


def test_projection_profile_compresses():
    # A profile with wide zero margins encodes to far fewer tokens than entries.
    profile = [0] * 500 + [12, 12, 8] + [0] * 500
    encoded = rle_encode(profile)
    assert rle_decode(encoded) == profile
    assert len(encoded.split()) < len(profile)
