# utils.py - Utility functions for argument parsing, logging, and XML creation

import argparse
import logging
from pathlib import Path
import platform
import sys
from typing import List, Dict, Optional, Union, Any
from lxml import etree
import numpy as np
import cv2

# --- Constants ---
PROG = "ir"  # Program name
VERSION = "1.3.0"  # Updated version

# Supported Image Formats (Consider making this configurable or detecting dynamically if needed)
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

DEFAULT_NMS_THRESHOLD = 0.3  # Default Non-Maximum Suppression overlap
DEFAULT_DPI = 600  # Default DPI assumption for paper size guessing

# --- Logging Setup ---
logger = logging.getLogger(__name__)  # Use module-specific logger


def setup_logging(
    log_level_stream: int = logging.INFO,
    log_level_file: int = logging.DEBUG,
    log_file: Optional[Union[str, Path]] = None,
    append_handlers: bool = False,
) -> None:
    """Initializes logging handlers and format."""
    logger.setLevel(
        logging.DEBUG
    )  # Set logger level to lowest to allow handlers to control output

    if not append_handlers:
        # Remove existing handlers *added by this logger* to avoid duplication on re-init
        # Be cautious if other libraries might configure the root logger.
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    log_format = logging.Formatter(
        "%(asctime)-23s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stream Handler (Console)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(
            sys.stdout
        )  # Use stdout for info/debug, stderr for warnings/errors?
        stream_handler.setFormatter(log_format)
        stream_handler.setLevel(log_level_stream)
        logger.addHandler(stream_handler)
        logger.debug(
            "Added StreamHandler with level %s", logging.getLevelName(log_level_stream)
        )

    # File Handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if not any(
            isinstance(h, logging.FileHandler)
            and h.baseFilename == str(log_path.resolve())
            for h in logger.handlers
        ):
            file_handler = logging.FileHandler(
                log_path, encoding="UTF-8", mode="a"
            )  # Append mode
            file_handler.setFormatter(log_format)
            file_handler.setLevel(log_level_file)
            logger.addHandler(file_handler)
            logger.debug(
                "Added FileHandler for %s with level %s",
                log_path.resolve(),
                logging.getLevelName(log_level_file),
            )

    logger.info("Logging initialized for %s v%s", PROG, VERSION)
    logger.debug("Python %s on %s", sys.version.split()[0], platform.system())


# --- Argument Parsing ---
def parse_args(args: List[str]) -> argparse.Namespace:
    """Parses command-line arguments."""
    from hashing_config import (
        HASH_CATEGORIES,
    )  # Import here to avoid circular dependency if config needs utils

    argp = argparse.ArgumentParser(
        prog=PROG,
        description="Generate an XML report with metadata and features from an image or directory of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,  # Add custom help/version later
    )

    # --- Input/Output ---
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

    # --- Processing Options ---
    proc_group = argp.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--hashes",
        type=str,
        default="basic",
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
        default=None,  # ProcessPoolExecutor default (number of processors)
        help="Maximum number of worker processes for batch processing.",
    )

    # --- Logging Options ---
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
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the file.",
    )
    log_group.add_argument(
        "--log-level-console",
        type=str,
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

    # --- Help/Version ---
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

    # --- Post-processing Args ---
    # Set default log file name for single image input if not provided
    if not parsed_args.input_path.is_dir() and parsed_args.log_file is None:
        parsed_args.log_file = parsed_args.input_path.with_suffix(".ir.log")

    # Handle quiet flag
    if parsed_args.quiet:
        parsed_args.log_level_console = "WARNING"

    # Convert log level strings to logging constants
    parsed_args.log_level_file_int = getattr(
        logging, parsed_args.log_level_file.upper(), logging.DEBUG
    )
    parsed_args.log_level_console_int = getattr(
        logging, parsed_args.log_level_console.upper(), logging.INFO
    )

    return parsed_args


# --- XML Creation Helper ---
def create_element(
    parent: etree._Element,
    tag: str,
    text: Optional[str] = None,
    attrib: Optional[Dict[str, str]] = None,
) -> etree._Element:
    """Helper function to create and append an XML element."""
    if attrib is None:
        attrib = {}

    # Clean attributes: ensure values are strings
    clean_attrib = {k: str(v) for k, v in attrib.items() if v is not None}

    elem = etree.SubElement(parent, tag, attrib=clean_attrib)
    if text is not None:
        elem.text = str(text)  # Ensure text is string

    return elem


# --- Non-Maximum Suppression (from v1.2.0, slightly cleaned) ---
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

    # Ensure float type for calculations
    if boxes.dtype.kind == "i":
        boxes = boxes.astype(float)

    picked_indices = []

    # Grab coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute area
    area = (x2 - x1 + 1) * (y2 - y1 + 1)  # Add 1 for pixel inclusivity

    # Sort by bottom-right y-coordinate (or score if available)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        # Grab the last index (highest y2) and add it to picked list
        last = len(idxs) - 1
        i = idxs[last]
        picked_indices.append(i)

        # Find intersection coordinates with remaining boxes
        # np.maximum/minimum handles broadcasting correctly
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute width and height of intersection
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute Intersection over Union (IoU)
        intersection_area = w * h
        union_area = area[i] + area[idxs[:last]] - intersection_area
        overlap = (
            intersection_area / union_area
        )  # Using IoU is generally more robust than simple overlap ratio

        # Delete indices of boxes with overlap greater than threshold
        # Also delete the last index itself
        idxs_to_delete = np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        idxs = np.delete(idxs, idxs_to_delete)

    # Return only the picked boxes in integer format
    return boxes[picked_indices].astype(int)
