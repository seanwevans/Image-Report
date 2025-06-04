# papersize.py - Paper size definitions and guessing logic

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from utils import logger


@dataclass
class PaperSizeStandard:
    """Represents a standard paper size in millimeters."""

    width_mm: float
    height_mm: float

    @property
    def area(self) -> float:
        """Calculate area in square millimeters."""
        return self.width_mm * self.height_mm

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width / height)."""
        return self.width_mm / self.height_mm if self.height_mm > 0 else 0

    def scale(self, factor: float) -> "PaperSizeStandard":
        """Return a new Size scaled by factor."""
        return PaperSizeStandard(self.width_mm * factor, self.height_mm * factor)

    def __str__(self) -> str:
        return f"({self.width_mm:.1f} mm x {self.height_mm:.1f} mm)"


# Standard Paper Sizes (in millimeters)
STANDARD_SIZES_MM: Dict[str, PaperSizeStandard] = {
    # ISO A Series
    "A0": PaperSizeStandard(841, 1189),
    "A1": PaperSizeStandard(594, 841),
    "A2": PaperSizeStandard(420, 594),
    "A3": PaperSizeStandard(297, 420),
    "A4": PaperSizeStandard(210, 297),
    "A5": PaperSizeStandard(148, 210),
    "A6": PaperSizeStandard(105, 148),
    # ISO B Series
    "B0": PaperSizeStandard(1000, 1414),
    "B1": PaperSizeStandard(707, 1000),
    "B2": PaperSizeStandard(500, 707),
    "B3": PaperSizeStandard(353, 500),
    "B4": PaperSizeStandard(250, 353),
    "B5": PaperSizeStandard(176, 250),
    # U.S. Sizes (converted to millimeters)
    "Letter": PaperSizeStandard(8.5 * 25.4, 11 * 25.4),  # 215.9 x 279.4
    "Legal": PaperSizeStandard(8.5 * 25.4, 14 * 25.4),  # 215.9 x 355.6
    "Tabloid": PaperSizeStandard(11 * 25.4, 17 * 25.4),  # 279.4 x 431.8
    "Ledger": PaperSizeStandard(17 * 25.4, 11 * 25.4),  # 431.8 x 279.4
    # Japanese JIS B Sizes
    "JIS B0": PaperSizeStandard(1030, 1456),
    "JIS B1": PaperSizeStandard(728, 1030),
    "JIS B2": PaperSizeStandard(515, 728),
    "JIS B3": PaperSizeStandard(364, 515),
    "JIS B4": PaperSizeStandard(257, 364),
    "JIS B5": PaperSizeStandard(182, 257),
}

MM_PER_INCH = 25.4


def guess_paper_size(pixel_width: int, pixel_height: int, dpi: int) -> Dict[str, float]:
    """
    Guesses the standard paper size based on pixel dimensions and DPI.

    Args:
        pixel_width: Image width in pixels.
        pixel_height: Image height in pixels.
        dpi: Assumed Dots Per Inch for the image.

    Returns:
        A dictionary mapping paper size names (str) to a difference score (float).
        Lower scores indicate a better match. Scores account for orientation.
    """
    if dpi <= 0:
        logger.warning(
            "Invalid DPI value (%d) for paper size guess. Cannot proceed.", dpi
        )
        return {}

    image_width_mm = (pixel_width / dpi) * MM_PER_INCH
    image_height_mm = (pixel_height / dpi) * MM_PER_INCH
    logger.debug(
        "Image dimensions: %.1f mm x %.1f mm (at %d DPI)",
        image_width_mm,
        image_height_mm,
        dpi,
    )

    differences: Dict[str, float] = {}

    for name, standard_size in STANDARD_SIZES_MM.items():
        diff_portrait = abs(image_width_mm - standard_size.width_mm) + abs(
            image_height_mm - standard_size.height_mm
        )

        diff_landscape = abs(image_width_mm - standard_size.height_mm) + abs(
            image_height_mm - standard_size.width_mm
        )

        differences[name] = min(diff_portrait, diff_landscape)
        logger.debug(
            " -> %s: Portrait diff=%.2f, Landscape diff=%.2f. Min diff=%.2f",
            name,
            diff_portrait,
            diff_landscape,
            differences[name],
        )

    sorted_guesses = sorted(differences.items(), key=lambda item: item[1])

    logger.debug(
        "Top paper size guesses (based on %d DPI): %s",
        dpi,
        [(name, f"{score:.2f}") for name, score in sorted_guesses[:3]],
    )

    return dict(sorted_guesses)


def guess_dpi(
    paper_type: str, pixel_width: int, pixel_height: int
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimates the DPI based on a known paper type and pixel dimensions.

    Args:
        paper_type: The assumed standard paper type name (e.g., "A4", "Letter").
        pixel_width: Image width in pixels.
        pixel_height: Image height in pixels.

    Returns:
        A tuple containing the estimated DPI based on width and height (float, float).
        Returns (None, None) if paper type is unknown or dimensions are invalid.
    """
    if paper_type not in STANDARD_SIZES_MM:
        logger.warning("Unknown paper type '%s' for DPI guessing.", paper_type)
        return None, None

    standard_size = STANDARD_SIZES_MM[paper_type]

    paper_width_in_p = standard_size.width_mm / MM_PER_INCH
    paper_height_in_p = standard_size.height_mm / MM_PER_INCH

    paper_width_in_l = standard_size.height_mm / MM_PER_INCH
    paper_height_in_l = standard_size.width_mm / MM_PER_INCH

    dpi_w_p, dpi_h_p = None, None
    if paper_width_in_p > 0:
        dpi_w_p = pixel_width / paper_width_in_p
    if paper_height_in_p > 0:
        dpi_h_p = pixel_height / paper_height_in_p

    dpi_w_l, dpi_h_l = None, None
    if paper_width_in_l > 0:
        dpi_w_l = pixel_width / paper_width_in_l
    if paper_height_in_l > 0:
        dpi_h_l = pixel_height / paper_height_in_l

    diff_p = (
        abs(dpi_w_p - dpi_h_p)
        if dpi_w_p is not None and dpi_h_p is not None
        else float("inf")
    )
    diff_l = (
        abs(dpi_w_l - dpi_h_l)
        if dpi_w_l is not None and dpi_h_l is not None
        else float("inf")
    )

    if diff_p <= diff_l and dpi_w_p is not None:
        logger.debug(
            "Guessed DPI based on %s (Portrait): width=%.1f, height=%.1f",
            paper_type,
            dpi_w_p,
            dpi_h_p,
        )
        return dpi_w_p, dpi_h_p
    elif diff_l < diff_p and dpi_w_l is not None:
        logger.debug(
            "Guessed DPI based on %s (Landscape): width=%.1f, height=%.1f",
            paper_type,
            dpi_w_l,
            dpi_h_l,
        )
        return dpi_w_l, dpi_h_l
    elif dpi_w_p is not None:
        logger.debug(
            "Guessed DPI based on %s (Fallback Portrait): width=%.1f, height=%.1f",
            paper_type,
            dpi_w_p,
            dpi_h_p,
        )
        return dpi_w_p, dpi_h_p
    elif dpi_w_l is not None:
        logger.debug(
            "Guessed DPI based on %s (Fallback Landscape): width=%.1f, height=%.1f",
            paper_type,
            dpi_w_l,
            dpi_h_l,
        )
        return dpi_w_l, dpi_h_l
    else:
        logger.warning("Could not calculate valid DPI for %s.", paper_type)
        return None, None
