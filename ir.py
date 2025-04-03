#!/usr/bin/env python3

""" ir.py  -  Generate report from image """

from concurrent.futures import ProcessPoolExecutor

import argparse
import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import platform
import sys
from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    Union,
    Any,
    Callable,
    TypeVar,
    Set,
    Iterable,
    cast,
)

import aiofiles
import numpy as np
import cv2
from lxml import etree
from PIL import Image
import imagehash
from skimage.feature import local_binary_pattern
from skimage.feature import hog as skimage_hog
from skimage.morphology import skeletonize, thin
from skimage.measure import label, regionprops


PROG = Path(__file__).stem
VERSION = "1.2.0"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# utils.py
supported_image_formats = [
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


def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse command-line arguments"""

    argp = argparse.ArgumentParser(
        prog=PROG,
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    argp.add_argument(
        "input_path",
        type=Path,
        help="Path to input file",
    )

    argp.add_argument(
        "out_path",
        type=Path,
        help="Path to output",
    )

    argp.add_argument(
        "--version",
        action="version",
        version=f"{PROG} v{VERSION}",
    )

    argp.add_argument(
        "--help",
        action="help",
        help="show this help message and exit",
    )

    return argp.parse_args(args)


def init(
    params: Optional[argparse.Namespace] = None,
    log_file: Optional[Union[str, Path]] = None,
    stream_level: int = logging.DEBUG,
    file_level: int = logging.DEBUG,
) -> None:
    """initialize logs"""

    for handler in logger.handlers:
        logger.removeHandler(handler)

    log_format = logging.Formatter(
        "%(asctime)-23s %(module)s.%(funcName)s %(levelname)-8s %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    stream_handler.setLevel(stream_level)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="UTF-8")
        file_handler.setFormatter(log_format)
        file_handler.setLevel(file_level)
        logger.addHandler(file_handler)

    logger.info("🍍 %s (v%s)", Path(__file__).resolve(), VERSION)

    if params:
        logger.debug("Parameters:")
        for param, value in vars(params).items():
            logger.debug("  %-16s%s", param, value)

    logger.debug("Logs:")
    for log_handle in logger.handlers:
        logger.debug("  %s", log_handle)


def non_max_suppression(boxes: np.ndarray, overlap_thresh: float) -> np.ndarray:
    """
    Condense extraneous bounding boxes based on overlap_thresh
    """
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )

    return boxes[pick].astype("int")


def boxes_from_image(img: np.ndarray, overlap_thresh: float = 0) -> np.ndarray:
    """Get bounding boxes for all contiguous pixel areas"""

    boxes = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    for hull in hulls:
        mins = np.amin(hull, axis=0)
        maxs = np.amax(hull, axis=0)

        x0 = mins[0][0]
        y0 = mins[0][1]
        x1 = maxs[0][0]
        y1 = maxs[0][1]
        boxes.append([x0, y0, x1, y1])

    boxes = non_max_suppression(boxes, overlap_thresh)

    return boxes[np.lexsort(boxes.T[::-1])]


def spectral_analysis(img: np.ndarray) -> Dict[str, List[int]]:
    """Use image data to segment page into lines and columns"""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    counts = {}
    counts["horiz"] = []
    counts["vert"] = []

    for i in range(thr.shape[1]):
        black_px = thr.shape[0] - cv2.countNonZero(thr[:, i])
        counts["vert"].append(black_px)

    for i in range(thr.shape[0]):
        black_px = thr.shape[1] - cv2.countNonZero(thr[i, :])
        counts["horiz"].append(black_px)

    return counts


def feature_to_hash(descriptors: np.ndarray) -> str:
    """Quantize descriptors and convert to a hexadecimal digest"""
    descriptors = descriptors.flatten()
    descriptors = (descriptors / 255.0 * 15).astype(int)
    hash_value = "".join([format(v, "x") for v in descriptors[:32]])

    return hash_value


def create_element(
    parent: etree.Element,
    tag: str,
    text: Optional[str] = None,
    attrib: Optional[Dict[str, str]] = None,
) -> etree.SubElement:
    """Helper function to create an XML element"""

    if attrib is None:
        attrib = {}

    elem = etree.SubElement(parent, tag, attrib)
    if text is not None:
        elem.text = text

    return elem


def create_hash_element(
    parent: etree.Element, hash_type: str, image: np.ndarray
) -> None:
    """Function to create hash elements"""

    if hash_type in cv2_hash_functions:
        hash_function = hash_type + "Hash_create"
        hasher = getattr(cv2.img_hash, hash_function)()
        hash_sub_elem = create_element(parent, hash_type)
        hash_sub_elem.text = hasher.compute(image).tobytes().hex()

    else:
        hash_sub_elem = create_element(parent, hash_type.__name__)
        hash_sub_elem.text = hash_type(image)


# stomach.py
def dhash(image: Image.Image) -> str:
    return str(imagehash.dhash(image))


def wavelet_hash(image: Image.Image) -> str:
    return str(imagehash.whash(image))


def histogram_hash(image: np.ndarray) -> str:
    histogram = cv2.calcHist(
        [image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    histogram = cv2.normalize(histogram, histogram).flatten()
    return "".join([format(int(v * 255), "02x") for v in histogram])


def gabor_hash(image: np.ndarray) -> str:
    gabor_filter = cv2.getGaborKernel(
        (5, 5), 1.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F
    )
    gabor_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_filter)
    hash_value = "".join([format(int(v), "02x") for v in gabor_image.flatten()[:32]])
    return hash_value


def sift_hash(image: np.ndarray) -> str:
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image, None)
    return feature_to_hash(descriptors)


def surf_hash(image: np.ndarray) -> str:
    surf = cv2.xfeatures2d.SURF_create()
    _, descriptors = surf.detectAndCompute(image, None)
    return feature_to_hash(descriptors)


def orb_hash(image: np.ndarray) -> str:
    orb = cv2.ORB_create()
    _, descriptors = orb.detectAndCompute(image, None)
    return feature_to_hash(descriptors)


def tamura_hash(image: np.ndarray) -> str:
    coarseness = np.mean(cv2.absdiff(image[:-1, :], image[1:, :])) + np.mean(
        cv2.absdiff(image[:, :-1], image[:, 1:])
    )
    contrast = np.std(image)
    hash_value = format(int(coarseness), "04x") + format(int(contrast), "04x")
    return hash_value


def fuzzy_hash(image: np.ndarray) -> str:
    quantized_image = (image // 32).astype(np.uint8)
    hash_value = "".join([format(v, "02x") for v in quantized_image.flatten()[:32]])
    return hash_value


def geometric_hash(image: np.ndarray) -> str:
    edges = cv2.Canny(image, 100, 200)
    hash_value = "".join([format(v, "02x") for v in edges.flatten()[:32]])
    return hash_value


def fourier_hash(image: np.ndarray) -> str:
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted))
    magnitude_spectrum = (magnitude_spectrum / magnitude_spectrum.max() * 255).astype(
        np.uint8
    )
    hash_value = "".join([format(v, "02x") for v in magnitude_spectrum.flatten()[:32]])
    return hash_value


def skeleton_hash(image: np.ndarray) -> str:
    """
    Create a hash based on the skeleton of a character.
    This is useful for detecting structurally similar characters.
    """
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    binary = binary.astype(bool)

    skeleton = skeletonize(binary)
    skeleton_img = skeleton.astype(np.uint8) * 255

    def neighbors(pos, img):
        y, x = pos
        n_count = 0
        for i in range(max(0, y - 1), min(img.shape[0], y + 2)):
            for j in range(max(0, x - 1), min(img.shape[1], x + 2)):
                if (i != y or j != x) and img[i, j]:
                    n_count += 1
        return n_count

    endpoints = []
    branchpoints = []
    for y in range(skeleton_img.shape[0]):
        for x in range(skeleton_img.shape[1]):
            if skeleton_img[y, x]:
                n = neighbors((y, x), skeleton_img)
                if n == 1:
                    endpoints.append((y, x))
                elif n > 2:
                    branchpoints.append((y, x))

    endpoint_count = len(endpoints)
    branch_count = len(branchpoints)
    y_indices, x_indices = np.where(skeleton_img)
    if len(y_indices) > 0 and len(x_indices) > 0:
        cy = np.mean(y_indices)
        cx = np.mean(x_indices)
    else:
        cy, cx = 0, 0

    top_left = sum(1 for y, x in endpoints if y < cy and x < cx)
    top_right = sum(1 for y, x in endpoints if y < cy and x >= cx)
    bottom_left = sum(1 for y, x in endpoints if y >= cy and x < cx)
    bottom_right = sum(1 for y, x in endpoints if y >= cy and x >= cx)

    branch_tl = sum(1 for y, x in branchpoints if y < cy and x < cx)
    branch_tr = sum(1 for y, x in branchpoints if y < cy and x >= cx)
    branch_bl = sum(1 for y, x in branchpoints if y >= cy and x < cx)
    branch_br = sum(1 for y, x in branchpoints if y >= cy and x >= cx)

    pixel_count = np.sum(skeleton_img) // 255
    values = [
        endpoint_count,
        branch_count,
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        branch_tl,
        branch_tr,
        branch_bl,
        branch_br,
        pixel_count,
    ]

    hash_value = "".join(format(min(v, 15), "x") for v in values)
    return hash_value


def contour_hash(image: np.ndarray) -> str:
    """
    Create a hash based on character contours.
    Useful for comparing boundary shapes.
    """

    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "0" * 16

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    circularity = 0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)

    M = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(M).flatten()
    normalized_moments = []
    for moment in hu_moments:
        if moment != 0:
            normalized_moments.append(
                min(15, int(-np.sign(moment) * np.log10(abs(moment)) * 1.5))
            )
        else:
            normalized_moments.append(0)

    aspect_ratio = 0
    if M["m00"] != 0:
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0

    normalized_moments = [min(15, max(0, m)) for m in normalized_moments]
    hash_components = normalized_moments + [
        int(circularity * 15),
        int(aspect_ratio * 15),
    ]

    hash_value = "".join(format(int(v), "x") for v in hash_components[:16])
    return hash_value


def zoning_hash(image: np.ndarray) -> str:
    """
    Create a hash based on zoning the character.
    Divides the image into a grid and calculates pixel density in each zone.
    """

    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    zones_y, zones_x = 4, 4
    height, width = binary.shape
    zone_height = height // zones_y
    zone_width = width // zones_x
    densities = []

    for y in range(zones_y):
        for x in range(zones_x):
            y_start = y * zone_height
            y_end = (y + 1) * zone_height if y < zones_y - 1 else height
            x_start = x * zone_width
            x_end = (x + 1) * zone_width if x < zones_x - 1 else width
            zone = binary[y_start:y_end, x_start:x_end]
            zone_size = (y_end - y_start) * (x_end - x_start)
            if zone_size > 0:
                density = np.sum(zone) / (zone_size * 255)
                densities.append(min(15, int(density * 16)))
            else:
                densities.append(0)

    hash_value = "".join(format(d, "x") for d in densities)
    return hash_value


def stroke_direction_hash(image: np.ndarray) -> str:
    """
    Create a hash based on stroke directions.
    Detects dominant stroke directions in the character.
    """
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(gradient_x, gradient_y)
    direction = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)

    threshold = np.mean(magnitude) * 1.5
    mask = magnitude > threshold

    bins = 8
    bin_size = 360 // bins
    direction_bins = np.zeros(bins, dtype=int)

    for i in range(direction.shape[0]):
        for j in range(direction.shape[1]):
            if mask[i, j]:
                bin_idx = int(direction[i, j] // bin_size) % bins
                direction_bins[bin_idx] += 1

    total = np.sum(direction_bins)
    if total > 0:
        normalized_bins = (direction_bins / total * 15).astype(int)
    else:
        normalized_bins = np.zeros(bins, dtype=int)

    horizontal_sum = direction_bins[0] + direction_bins[4]  # 0° and 180°
    vertical_sum = direction_bins[2] + direction_bins[6]  # 90° and 270°

    if vertical_sum > 0:
        hv_ratio = min(15, int((horizontal_sum / vertical_sum) * 4))
    else:
        hv_ratio = 15

    diagonal1_sum = direction_bins[1] + direction_bins[5]  # 45° and 225°
    diagonal2_sum = direction_bins[3] + direction_bins[7]  # 135° and 315°

    if diagonal2_sum > 0:
        diag_ratio = min(15, int((diagonal1_sum / diagonal2_sum) * 4))
    else:
        diag_ratio = 15

    bin_hash = "".join(format(b, "x") for b in normalized_bins)
    ratio_hash = format(hv_ratio, "x") + format(diag_ratio, "x")

    return bin_hash + ratio_hash


def junction_hash(image: np.ndarray) -> str:
    """
    Create a hash based on junction analysis.
    Detects and categorizes junction points in the character.
    """
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    binary = binary.astype(bool)
    thinned = thin(binary)
    skeleton_img = thinned.astype(np.uint8) * 255

    def count_neighbors(img, y, x):
        n_count = 0
        neighbors_pos = []
        for i in range(max(0, y - 1), min(img.shape[0], y + 2)):
            for j in range(max(0, x - 1), min(img.shape[1], x + 2)):
                if (i != y or j != x) and img[i, j]:
                    n_count += 1
                    neighbors_pos.append((i, j))
        return n_count, neighbors_pos

    endpoints = []  # 1 neighbor
    continuation = []  # 2 neighbors
    t_junctions = []  # 3 neighbors
    x_junctions = []  # 4+ neighbors

    for y in range(skeleton_img.shape[0]):
        for x in range(skeleton_img.shape[1]):
            if skeleton_img[y, x]:
                n_count, _ = count_neighbors(skeleton_img, y, x)
                if n_count == 1:
                    endpoints.append((y, x))
                elif n_count == 2:
                    continuation.append((y, x))
                elif n_count == 3:
                    t_junctions.append((y, x))
                elif n_count >= 4:
                    x_junctions.append((y, x))

    height, width = skeleton_img.shape

    quadrants = [
        [(0, 0), (height // 2, width // 2)],  # Top-left
        [(0, width // 2), (height // 2, width)],  # Top-right
        [(height // 2, 0), (height, width // 2)],  # Bottom-left
        [(height // 2, width // 2), (height, width)],  # Bottom-right
    ]

    endpoint_dist = [0, 0, 0, 0]
    t_junction_dist = [0, 0, 0, 0]
    x_junction_dist = [0, 0, 0, 0]

    for i, ((y1, x1), (y2, x2)) in enumerate(quadrants):
        for y, x in endpoints:
            if y1 <= y < y2 and x1 <= x < x2:
                endpoint_dist[i] += 1

        for y, x in t_junctions:
            if y1 <= y < y2 and x1 <= x < x2:
                t_junction_dist[i] += 1

        for y, x in x_junctions:
            if y1 <= y < y2 and x1 <= x < x2:
                x_junction_dist[i] += 1

    total_endpoints = len(endpoints)
    total_t_junctions = len(t_junctions)
    total_x_junctions = len(x_junctions)
    total_continuation = len(continuation)

    features = [
        min(15, total_endpoints),
        min(15, total_t_junctions),
        min(15, total_x_junctions),
        min(15, total_continuation // 16),
    ] + [min(15, v) for v in endpoint_dist + t_junction_dist + x_junction_dist]

    hash_value = "".join(format(f, "x") for f in features)
    return hash_value


cv2_hash_functions = {
    "Average": cv2.img_hash.AverageHash_create(),
    "BlockMean": cv2.img_hash.BlockMeanHash_create(),
    "ColorMoment": cv2.img_hash.ColorMomentHash_create(),
    "MarrHildreth": cv2.img_hash.MarrHildrethHash_create(),
    "P": cv2.img_hash.PHash_create(),
    "RadialVariance": cv2.img_hash.RadialVarianceHash_create(),
}


hash_functions = {
    "dhash": dhash,
    "wavelet": wavelet_hash,
    "histogram": histogram_hash,
    "gabor": gabor_hash,
    "sift": sift_hash,
    "surf": surf_hash,
    "orb": orb_hash,
    "tamura": tamura_hash,
    "fuzzy": fuzzy_hash,
    "geometric": geometric_hash,
    "fourier": fourier_hash,
    "skeleton": skeleton_hash,
    "contour": contour_hash,
    "zoning": zoning_hash,
    "stroke_direction": stroke_direction_hash,
    "junction": junction_hash,
}


# papersize.py
@dataclass
class Size:
    width: float
    height: float

    def area(self) -> float:
        """Calculate area of the size"""
        return self.width * self.height

    def aspect_ratio(self) -> float:
        """Calculate aspect ratio"""
        return self.width / self.height

    def scale(self, factor: float) -> "Size":
        """Return a new Size scaled by factor"""
        return Size(self.width * factor, self.height * factor)


standard_sizes = {
    # ISO A Series
    "A0": Size(841, 1189),
    "A1": Size(594, 841),
    "A2": Size(420, 594),
    "A3": Size(297, 420),
    "A4": Size(210, 297),
    "A5": Size(148, 210),
    "A6": Size(105, 148),
    # ISO B Series
    "B0": Size(1000, 1414),
    "B1": Size(707, 1000),
    "B2": Size(500, 707),
    "B3": Size(353, 500),
    "B4": Size(250, 353),
    "B5": Size(176, 250),
    # U.S. Sizes (converted to millimeters)
    "letter": Size(8.5 * 25.4, 11 * 25.4),
    "legal": Size(8.5 * 25.4, 14 * 25.4),
    "labloid": Size(11 * 25.4, 17 * 25.4),
    "ledger": Size(17 * 25.4, 11 * 25.4),
    # Japanese JIS B Sizes
    "JIS B0": Size(1030, 1456),
    "JIS B1": Size(728, 1030),
    "JIS B2": Size(515, 728),
    "JIS B3": Size(364, 515),
    "JIS B4": Size(257, 364),
    "JIS B5": Size(182, 257),
}


def guess_paper_size(img: np.ndarray, dpi: int = 600) -> Dict[str, float]:
    image_width_mm = (img.shape[1] / dpi) * 25.4
    image_height_mm = (img.shape[0] / dpi) * 25.4

    differences = {}
    for size in standard_sizes:
        if size not in differences:
            differences[size] = -1
        differences[size] = abs(image_width_mm - standard_sizes[size].width) + abs(
            image_height_mm - standard_sizes[size].height
        )

    return differences


def guess_dpi(
    paper_type: str, pixel_width: float, pixel_height: float
) -> Tuple[float, float]:
    if paper_type not in standard_sizes:
        raise ValueError(f"Unknown paper type: {paper_type}")

    paper_size = standard_sizes[paper_type]

    paper_width_in_inches = paper_size.width / 25.4
    paper_height_in_inches = paper_size.height / 25.4

    dpi_width = pixel_width / paper_width_in_inches
    dpi_height = pixel_height / paper_height_in_inches

    return dpi_width, dpi_height


# ir.py
def gen_xml(input_path: Path, img: np.ndarray) -> etree.Element:
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pimg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # <report>
    root = etree.Element("report")

    #   <Metadata>
    info_elem = etree.SubElement(root, "metadata")

    #     <hash>
    hash_elem = etree.SubElement(info_elem, "hash")
    for func in cv2_hash_functions:
        try:
            create_hash_element(hash_elem, func, gimg)
        except:
            pass

    for func in hash_functions.values():
        if func.__name__ in ["dhash", "wavelet_hash"]:
            to_pass = pimg
        elif func.__name__ == "histogram_hash":
            to_pass = img
        else:
            to_pass = gimg
        try:
            create_hash_element(hash_elem, func, to_pass)
        except:
            pass

    #     <dimension>
    dim_elem = create_element(info_elem, "dimensions")
    create_element(dim_elem, "width", text=str(img.shape[1]), attrib={"unit": "pixel"})
    create_element(dim_elem, "height", text=str(img.shape[0]), attrib={"unit": "pixel"})
    logger.info("👀 %d pixels in image", img.shape[0] * img.shape[1])

    #     <paper-size>
    #       <match rank="1" type="exact" score="1.0000">A4</match>
    #       <match rank="2" type="approximate" score="0.3456">Letter</match>
    #       <match rank="3" type="approximate" score="0.2212">Legal</match>
    #     </paper-size>
    guesses = guess_paper_size(img)

    ps_elem = etree.SubElement(dim_elem, "paper-size")
    candidates = sorted(guesses.items(), key=lambda x: x[1])

    for n in range(3):
        paper_size, distance = candidates[n]
        score = 1 / (1 + distance)
        create_element(
            ps_elem,
            "match",
            text=paper_size,
            attrib={
                "type": "exact" if score == 1 else "approximate",
                "confidence": f"{score:4.3f}",
            },
        )

    w_text = str(round(img.shape[1] / standard_sizes[candidates[0][0]].width * 25.4))
    h_text = str(round(img.shape[0] / standard_sizes[candidates[0][0]].height * 25.4))

    res_elem = create_element(dim_elem, "resolution", attrib={"type": "approximate"})
    create_element(
        res_elem, "resolution", text=w_text, attrib={"measure": "width", "unit": "dpi"}
    )
    create_element(
        res_elem, "resolution", text=h_text, attrib={"measure": "height", "unit": "dpi"}
    )

    #     <path>
    path_elem = create_element(info_elem, "path", text=str(input_path))
    #   </Metadata>

    # Contiguous pixel area bounding boxes
    cpa_bbs = boxes_from_image(img)
    box_elem = etree.SubElement(
        root, "bboxes", boxes=str(len(cpa_bbs)), format="XYTL-XYBR"
    )

    for n, bbox in enumerate(cpa_bbs):
        bbox_elem = etree.SubElement(box_elem, "bbox", n=str(n))
        bbox_elem.text = str(bbox)

    # Spectral counts
    counts = spectral_analysis(img)
    count_elem = etree.SubElement(root, "counts")

    dir_map = {"horiz": "rows", "vert": "columns"}
    for orientation, way in dir_map.items():
        d_elem = etree.SubElement(count_elem, way)
        for i, count in enumerate(counts[orientation], start=1):
            if count == 0:
                continue
            elem = etree.SubElement(d_elem, way[:-1], n=str(i))
            elem.text = str(count)

    return root


def process_image_sync(image_path: Path, out_path: Path):
    """Process a single image using blocking"""
    try:
        img = cv2.imread(str(image_path))
    except Exception as exc:
        logger.critical(exc, exc_info=True)
        raise exc

    if img is None:
        logger.critical("Failed to read image '%s'", image_path.resolve())
        raise IOError(f"Failed to read image '{image_path.resolve()}'")

    try:
        etree.ElementTree(gen_xml(image_path, img)).write(
            str(out_path), encoding="UTF-8", xml_declaration=True, pretty_print=True
        )
    except Exception as exc:
        logger.critical(exc, exc_info=True)
        raise exc


async def process_image_async(image_path: Path, out_path: Path) -> None:
    """Process a single image in a non-blocking manner"""
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        await loop.run_in_executor(pool, process_image_sync, image_path, out_path)

    logger.info("💥 %s", out_path.resolve())


async def process_batch(image_paths: List[Path], out_dir: Path) -> None:
    """Process a batch of images concurrently"""
    with ProcessPoolExecutor() as pool:
        loop = asyncio.get_running_loop()
        tasks = []

        for image_path in image_paths:
            out_path = out_dir / f"{image_path.stem}.xml"
            tasks.append(
                loop.run_in_executor(pool, process_image_sync, image_path, out_path)
            )

        completed_tasks = await asyncio.gather(*tasks)
        for image_path in image_paths:
            out_path = out_dir / f"{image_path.stem}.xml"
            logger.info("💥 %s", out_path.resolve())

    logger.info("✅ All images processed to %s", out_dir)


def main(args: List[str]) -> None:
    """script entry-point"""

    params = parse_args(args)

    init(params, log_file=params.input_path.with_suffix(".ir"))

    if params.input_path.is_dir():
        params.out_path.mkdir(parents=True, exist_ok=True)
        image_paths = [
            img
            for fmt in supported_image_formats
            for img in params.input_path.glob(f"*.{fmt}")
        ]
        asyncio.run(process_batch(image_paths, params.out_path))
    else:
        process_image_sync(params.input_path, params.out_path)

    logger.info("💥 %s", params.out_path.resolve())


if __name__ == "__main__":
    main(sys.argv[1:])
