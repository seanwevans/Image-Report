# utils.py - Utility functions for argument parsing, logging, and XML creation

import argparse
import logging
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Union

from lxml import etree
import numpy as np
import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PROG = "ir"
VERSION = "1.3.0"

DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_DPI = 600
SUPPORTED_IMAGE_FORMATS = [
    "avif",
    "bmp",
    "exr",
    "gif",
    "hdr",
    "jpeg",
    "jpg",
    "png",
    "webp",
    "pbm",
    "pfm",
    "pgm",
    "pic",
    "pnm",
    "ppm",
    "pxm",
    "ras",
    "sr",
    "tif",
    "tiff",
]


def setup_logging(
    log_level_stream: int = logging.INFO,
    log_level_file: int = logging.DEBUG,
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """Initializes logging handlers and format.

    Existing handlers are removed so the function can be safely invoked
    multiple times without duplicating log output.
    """

    # Remove existing handlers to avoid duplicate logs when called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    log_format = logging.Formatter("%(asctime)-23s [%(levelname)8s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    stream_handler.setLevel(log_level_stream)
    logger.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="UTF-8", mode="a")
        file_handler.setFormatter(log_format)
        file_handler.setLevel(log_level_file)
        logger.addHandler(file_handler)


def parse_args(args: List[str]) -> argparse.Namespace:
    """Parses command-line arguments."""

    from hashing_config import HASH_CATEGORIES

    argp = argparse.ArgumentParser(
        prog=PROG,
        description="Generate an XML report with metadata and features from an image or directory of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    argp.add_argument(
        "input_path",
        type=Path,
        help="Path to the input image file or directory containing images.",
    )
    argp.add_argument(
        "output_path",
        type=Path,
        help="Path to the output XML file (if input is file) or directory (if input is directory).",
    )

    proc_group = argp.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--hashes",
        type=str,
        default="all",
        help="Specify which hashes to compute. Examples: "
        "'all', 'none', 'basic', 'perceptual', 'feature', 'sift,dhash', 'basic,-dhash'. "
        f"Categories: {', '.join(HASH_CATEGORIES.keys())}. Prefix with '-' to exclude.",
    )
    proc_group.add_argument(
        "--nms-threshold",
        type=float,
        default=DEFAULT_NMS_THRESHOLD,
        help="Overlap threshold (0.0-1.0) for Non-Maximum Suppression of bounding boxes.",
    )
    proc_group.add_argument(
        "--default-dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Assumed DPI for paper size guessing if resolution info is missing.",
    )
    proc_group.add_argument(
        "--continue-on-error",
        action="store_true",
        help="In batch mode, continue processing other images if one fails.",
    )
    proc_group.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes for batch processing.",
    )

    log_group = argp.add_argument_group("Logging Options")
    log_group.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to optional log file. If not specified, logging only goes to console."
        " If input is a directory, consider setting this explicitly."
        " Default name in single file mode is <input_name>.ir.log",
    )
    log_group.add_argument(
        "--log-level-file",
        type=str.upper,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the file.",
    )
    log_group.add_argument(
        "--log-level-console",
        type=str.upper,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the console.",
    )

    log_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress console output (sets console level to WARNING).",
    )

    help_group = argp.add_argument_group("Help and Version")
    help_group.add_argument(
        "--version",
        action="version",
        version=f"{PROG} v{VERSION}",
        help="Show program's version number and exit.",
    )
    help_group.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit."
    )

    parsed_args = argp.parse_args(args)

    if not parsed_args.input_path.is_dir() and parsed_args.log_file is None:
        parsed_args.log_file = parsed_args.input_path.with_suffix(".ir.log")

    if parsed_args.quiet:
        parsed_args.log_level_console = "WARNING"

    parsed_args.log_level_file_int = getattr(
        logging, parsed_args.log_level_file, logging.DEBUG
    )
    parsed_args.log_level_console_int = getattr(
        logging, parsed_args.log_level_console, logging.INFO
    )

    return parsed_args


def create_element(
    parent: etree._Element,
    tag: str,
    text: Optional[str] = None,
    attrib: Optional[Dict[str, str]] = None,
) -> etree._Element:
    """Helper function to create and append an XML element."""

    if attrib is None:
        attrib = {}

    clean_attrib = {k: str(v) for k, v in attrib.items() if v is not None}
    elem = etree.SubElement(parent, tag, attrib=clean_attrib)
    if text is not None:
        elem.text = str(text)

    return elem


def non_max_suppression(boxes: np.ndarray, overlap_thresh: float) -> np.ndarray:
    """
    Condenses overlapping bounding boxes based on overlap threshold (IoU).
    Input boxes: numpy array of shape (N, 4) with format [x1, y1, x2, y2].
    Returns: numpy array of shape (M, 4) with suppressed boxes.
    """

    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)

    if len(boxes) == 0:
        return np.array([], dtype=int).reshape(0, 4)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype(float)

    picked_indices = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        picked_indices.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        intersection_area = w * h
        union_area = area[i] + area[idxs[:last]] - intersection_area
        overlap = intersection_area / union_area

        idxs_to_delete = np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        idxs = np.delete(idxs, idxs_to_delete)

    return boxes[picked_indices].astype(int)
