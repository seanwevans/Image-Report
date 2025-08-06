# analysis.py - Image analysis functions (bounding boxes, spectral analysis)

import cv2
import numpy as np
from typing import List, Dict, Optional

import logging
from utils import non_max_suppression

logger = logging.getLogger(__name__)


def find_bounding_boxes(
    image: np.ndarray, nms_overlap_thresh: float = 0.3
) -> np.ndarray:
    """
    Finds bounding boxes for relevant regions in the image using MSER
    and applies Non-Maximum Suppression.

    Args:
        image: Input image (BGR format).
        nms_overlap_thresh: IoU threshold for suppressing overlapping boxes.
            Must be between 0.0 and 1.0.

    Returns:
        Numpy array of bounding boxes [[x1, y1, x2, y2], ...], sorted by top-left corner.
        Returns empty array if no boxes found or on error.
    """
    logger.debug("Starting bounding box detection...")
    try:
        if image is None or image.size == 0:
            logger.warning("Cannot find bounding boxes: Input image is empty.")
            return np.array([], dtype=int).reshape(0, 4)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        if not regions:
            logger.warning("No MSER regions detected.")
            return np.array([], dtype=int).reshape(0, 4)

        boxes = []
        for points in regions:
            hull = cv2.convexHull(points.reshape(-1, 1, 2))
            x, y, w, h = cv2.boundingRect(hull)
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            boxes.append([x1, y1, x2, y2])

        if not boxes:
            logger.warning("No bounding boxes derived from MSER regions.")
            return np.array([], dtype=int).reshape(0, 4)

        boxes_np = np.array(boxes)
        logger.debug("Detected %d raw boxes before NMS.", len(boxes_np))

        suppressed_boxes = non_max_suppression(boxes_np, nms_overlap_thresh)
        logger.debug(
            "Found %d boxes after NMS (threshold=%.2f).",
            len(suppressed_boxes),
            nms_overlap_thresh,
        )

        if len(suppressed_boxes) == 0:
            return np.array([], dtype=int).reshape(0, 4)

        sorted_indices = np.lexsort((suppressed_boxes[:, 0], suppressed_boxes[:, 1]))
        sorted_boxes = suppressed_boxes[sorted_indices]

        return sorted_boxes

    except cv2.error as e:
        logger.error(
            "OpenCV error during bounding box detection: %s", e, exc_info=False
        )
        return np.array([], dtype=int).reshape(0, 4)
    except Exception as e:
        logger.error(
            "Unexpected error during bounding box detection: %s", e, exc_info=True
        )
        return np.array([], dtype=int).reshape(0, 4)


def calculate_spectral_analysis(image: np.ndarray) -> Optional[Dict[str, List[int]]]:
    """
    Calculates horizontal and vertical projection profiles (pixel counts).

    Args:
        image: Input image (BGR format).

    Returns:
        Dictionary {"horizontal": [...], "vertical": [...]} containing pixel counts
        per row and column, respectively. Returns None on error.
    """
    logger.debug("Starting spectral analysis...")
    try:
        if image is None or image.size == 0:
            logger.warning("Cannot perform spectral analysis: Input image is empty.")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        height, width = thresholded.shape
        horizontal_counts = np.count_nonzero(thresholded, axis=1).tolist()
        vertical_counts = np.count_nonzero(thresholded, axis=0).tolist()
        logger.debug(
            "Spectral analysis complete: %d rows, %d columns.", height, width
        )

        return {"horizontal": horizontal_counts, "vertical": vertical_counts}

    except cv2.error as e:
        logger.error("OpenCV error during spectral analysis: %s", e, exc_info=False)
        return None
    except Exception as e:
        logger.error("Unexpected error during spectral analysis: %s", e, exc_info=True)
        return None
