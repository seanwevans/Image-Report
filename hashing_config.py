# hashing_config.py - Configuration for image hashing functions

import cv2
import imagehash
import sys
from typing import Set, Dict, Callable, Union, List, Optional
import numpy as np
from PIL import Image

from skimage.morphology import skeletonize
from skimage.morphology import thin

from utils import logger

ImageType = Union[np.ndarray, Image.Image]
HashFunctionType = Callable[[ImageType], str]


def feature_to_hash(descriptors: Optional[np.ndarray]) -> str:
    """Quantize descriptors and convert to a hexadecimal digest"""

    if descriptors is None or descriptors.size == 0:
        return "0" * 32

    descriptors = descriptors.flatten()
    descriptors = descriptors[np.isfinite(descriptors)]
    if descriptors.size == 0:
        return "0" * 32

    max_val = np.max(np.abs(descriptors))
    if max_val > 0:
        norm_descriptors = descriptors / max_val
    else:
        norm_descriptors = descriptors
    quantized = (norm_descriptors * 15).astype(int)
    hash_value = "".join([format(v & 0xF, "x") for v in quantized[:32]])

    return hash_value.ljust(32, "0")


def dhash(image: Image.Image) -> str:
    return str(imagehash.dhash(image))


def wavelet_hash(image: Image.Image) -> str:
    return str(imagehash.whash(image))


def histogram_hash(image: np.ndarray) -> str:
    try:
        histogram = cv2.calcHist(
            [image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        histogram = cv2.normalize(histogram, histogram).flatten()
        return "".join([format(int(v * 255), "02x") for v in histogram])
    except cv2.error:
        return "histogram_error"


def gabor_hash(image: np.ndarray) -> str:
    try:
        gabor_filter = cv2.getGaborKernel(
            (5, 5), 1.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F
        )
        gabor_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_filter)
        hash_value = "".join(
            [format(int(v), "02x") for v in gabor_image.flatten()[:32]]
        )
        return hash_value.ljust(32, "0")
    except cv2.error:
        return "gabor_error"


def sift_hash(image: np.ndarray) -> str:
    try:
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute(image, None)
        return feature_to_hash(descriptors)
    except cv2.error:
        return "sift_unavailable_or_error"


def surf_hash(image: np.ndarray) -> str:
    try:
        surf = cv2.xfeatures2d.SURF_create()
        _, descriptors = surf.detectAndCompute(image, None)
        return feature_to_hash(descriptors)
    except AttributeError:
        return "surf_unavailable"
    except cv2.error:
        return "surf_error"


def orb_hash(image: np.ndarray) -> str:
    try:
        orb = cv2.ORB_create()
        _, descriptors = orb.detectAndCompute(image, None)
        return feature_to_hash(descriptors)
    except cv2.error:
        return "orb_error"


def tamura_hash(image: np.ndarray) -> str:
    try:
        coarseness = np.mean(cv2.absdiff(image[:-1, :], image[1:, :])) + np.mean(
            cv2.absdiff(image[:, :-1], image[:, 1:])
        )
        contrast = np.std(image)
        coarseness = coarseness if np.isfinite(coarseness) else 0
        contrast = contrast if np.isfinite(contrast) else 0
        hash_value = format(min(int(coarseness), 0xFFFF), "04x") + format(
            min(int(contrast), 0xFFFF), "04x"
        )

        return hash_value.ljust(8, "0")
    except Exception:
        return "tamura_error"


def fuzzy_hash(image: np.ndarray) -> str:
    try:
        quantized_image = (image // 32).astype(np.uint8)
        hash_value = "".join([format(v, "02x") for v in quantized_image.flatten()[:32]])
        return hash_value.ljust(32, "0")
    except Exception:
        return "fuzzy_error"


def geometric_hash(image: np.ndarray) -> str:
    try:
        edges = cv2.Canny(image, 100, 200)
        hash_value = "".join([format(v, "02x") for v in edges.flatten()[:32]])
        return hash_value.ljust(32, "0")
    except cv2.error:
        return "geometric_error"


def fourier_hash(image: np.ndarray) -> str:
    try:
        if image.size == 0:
            return "fourier_empty_input"
        f_transform = np.fft.fft2(image)
        f_transform_shifted = np.fft.fftshift(f_transform)

        magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1e-10)
        max_mag = np.max(magnitude_spectrum)

        if max_mag <= 0:
            magnitude_spectrum_norm = np.zeros_like(magnitude_spectrum)
        else:
            magnitude_spectrum_norm = magnitude_spectrum / max_mag * 255

        magnitude_spectrum_uint8 = magnitude_spectrum_norm.astype(np.uint8)
        hash_value = "".join(
            [format(v, "02x") for v in magnitude_spectrum_uint8.flatten()[:32]]
        )
        return hash_value.ljust(32, "0")
    except Exception as e:
        return f"fourier_error"


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


CV2_HASH_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
try:
    CV2_HASH_FUNCTIONS["Average"] = cv2.img_hash.AverageHash_create().compute
    CV2_HASH_FUNCTIONS["BlockMean"] = cv2.img_hash.BlockMeanHash_create().compute
    CV2_HASH_FUNCTIONS["ColorMoment"] = cv2.img_hash.ColorMomentHash_create().compute
    CV2_HASH_FUNCTIONS["MarrHildreth"] = cv2.img_hash.MarrHildrethHash_create().compute
    CV2_HASH_FUNCTIONS["PHash"] = cv2.img_hash.PHash_create().compute
    CV2_HASH_FUNCTIONS[
        "RadialVariance"
    ] = cv2.img_hash.RadialVarianceHash_create().compute
except AttributeError:
    logger.warning(
        "Warning: Some or all cv2.img_hash functions not found. Check OpenCV installation."
    )

IMAGEHASH_FUNCTIONS: Dict[str, HashFunctionType] = {
    "dhash": dhash,
    "wavelet": wavelet_hash,
}

CUSTOM_HASH_FUNCTIONS_GRAY: Dict[str, HashFunctionType] = {
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

CUSTOM_HASH_FUNCTIONS_COLOR: Dict[str, HashFunctionType] = {
    "histogram": histogram_hash,
    "gabor": gabor_hash,
}

CV2_HASH_NAMES: Set[str] = set(CV2_HASH_FUNCTIONS.keys())
IMAGEHASH_NAMES: Set[str] = set(IMAGEHASH_FUNCTIONS.keys())
CUSTOM_GRAY_NAMES: Set[str] = set(CUSTOM_HASH_FUNCTIONS_GRAY.keys())
CUSTOM_COLOR_NAMES: Set[str] = set(CUSTOM_HASH_FUNCTIONS_COLOR.keys())

PERCEPTUAL_HASHES: Set[str] = {"Average", "BlockMean", "PHash", "dhash", "wavelet"}
FEATURE_HASHES: Set[str] = {"sift", "surf", "orb"}
TEXTURE_HASHES: Set[str] = {"gabor", "tamura"}
STRUCTURAL_HASHES: Set[str] = {
    "MarrHildreth",
    "RadialVariance",
    "skeleton",
    "contour",
    "zoning",
    "stroke_direction",
    "junction",
    "geometric",
}
COLOR_HASHES: Set[str] = {"ColorMoment", "histogram"}
OTHER_HASHES: Set[str] = {"fuzzy", "fourier"}

ALL_HASHES: Set[str] = (
    CV2_HASH_NAMES | IMAGEHASH_NAMES | CUSTOM_GRAY_NAMES | CUSTOM_COLOR_NAMES
)

# Mapping of lowercase hash names to their canonical forms
HASH_NAME_LOOKUP: Dict[str, str] = {name.lower(): name for name in ALL_HASHES}

HASH_CATEGORIES: Dict[str, Set[str]] = {
    "all": ALL_HASHES,
    "none": set(),
    "cv2": CV2_HASH_NAMES,
    "imagehash": IMAGEHASH_NAMES,
    "custom_gray": CUSTOM_GRAY_NAMES,
    "custom_color": CUSTOM_COLOR_NAMES,
    "perceptual": PERCEPTUAL_HASHES,
    "feature": FEATURE_HASHES,
    "texture": TEXTURE_HASHES,
    "structural": STRUCTURAL_HASHES,
    "color": COLOR_HASHES,
    "other": OTHER_HASHES,
    "basic": PERCEPTUAL_HASHES | {"histogram", "zoning"},
}


def get_selected_hashes(specifier: str) -> Set[str]:
    """
    Parses the hash specifier string (e.g., "all", "basic,-sift", "dhash,phash")
    and returns the set of selected hash names.
    """
    if not specifier:
        return HASH_CATEGORIES["basic"]

    selected_hashes = set()
    parts = specifier.split(",")
    clean_parts = [p.strip() for p in parts]

    for part in clean_parts:
        if not part:
            continue

        if part.startswith("-"):
            # Exclusion
            exclude_part = part[1:]
            key = exclude_part.lower()
            if key in HASH_CATEGORIES:
                selected_hashes -= HASH_CATEGORIES[key]
            elif key in HASH_NAME_LOOKUP:
                selected_hashes.discard(HASH_NAME_LOOKUP[key])
        else:
            # Inclusion
            key = part.lower()
            if key in HASH_CATEGORIES:
                selected_hashes.update(HASH_CATEGORIES[key])
            elif key in HASH_NAME_LOOKUP:
                selected_hashes.add(HASH_NAME_LOOKUP[key])
            else:
                logger.warning(
                    "Warning: Unknown hash or category '%s' ignored.", part
                )

    if all(p.startswith("-") for p in clean_parts if p) and not any(
        p for p in clean_parts if not p.startswith("-")
    ):
        initial_set = ALL_HASHES.copy()
        for part in clean_parts:
            if not part:
                continue
            exclude_part = part[1:]
            key = exclude_part.lower()
            if key in HASH_CATEGORIES:
                initial_set -= HASH_CATEGORIES[key]
            elif key in HASH_NAME_LOOKUP:
                initial_set.discard(HASH_NAME_LOOKUP[key])
        return initial_set

    if "none" in [p.lower() for p in clean_parts] and len(clean_parts) == 1:
        return set()

    return selected_hashes
