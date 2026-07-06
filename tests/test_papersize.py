import math

from papersize import (
    STANDARD_SIZES_MM,
    guess_paper_size,
    paper_size_confidence,
)


def test_confidence_is_one_for_perfect_match():
    assert paper_size_confidence("A4", 0.0) == 1.0


def test_confidence_decreases_with_difference():
    a4 = STANDARD_SIZES_MM["A4"]
    # A difference of the paper's full (w + h) drives confidence to 0.
    assert paper_size_confidence("A4", a4.width_mm + a4.height_mm) == 0.0
    mid = paper_size_confidence("A4", (a4.width_mm + a4.height_mm) / 2.0)
    assert math.isclose(mid, 0.5, abs_tol=1e-9)


def test_confidence_unknown_type_is_zero():
    assert paper_size_confidence("NotAPaper", 0.0) == 0.0


def test_confidence_is_scale_free_across_dpi():
    # An exact A4 page at any DPI should report full confidence, because the
    # score is normalized by the paper's mm dimensions rather than by pixels.
    for dpi in (150, 300, 600):
        width = round(210 / 25.4 * dpi)
        height = round(297 / 25.4 * dpi)
        guesses = guess_paper_size(width, height, dpi)
        best = next(iter(guesses))
        assert best == "A4"
        assert paper_size_confidence(best, guesses[best]) > 0.98
