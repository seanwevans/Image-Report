import numpy as np
import cv2
from analysis import find_bounding_boxes, calculate_spectral_analysis


def test_find_bounding_boxes_simple_rectangle():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 30), (70, 80), (255, 255, 255), -1)
    boxes = find_bounding_boxes(img, nms_overlap_thresh=0.1)

    assert boxes.shape[1] == 4
    assert boxes.shape[0] > 0


def test_find_bounding_boxes_empty():
    boxes = find_bounding_boxes(np.array([]))
    assert boxes.size == 0


def test_calculate_spectral_analysis_simple():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[2:5, 3:7] = 255
    result = calculate_spectral_analysis(img)

    expected_horizontal = [10, 10, 6, 6, 6, 10, 10, 10, 10, 10]
    expected_vertical = [10, 10, 10, 7, 7, 7, 7, 10, 10, 10]

    assert result is not None
    assert result["horizontal"] == expected_horizontal
    assert result["vertical"] == expected_vertical


def test_calculate_spectral_analysis_empty():
    assert calculate_spectral_analysis(None) is None
    assert calculate_spectral_analysis(np.array([])) is None
