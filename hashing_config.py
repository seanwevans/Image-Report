# hashing_config.py - Configuration for image hashing functions

import cv2
import imagehash
from typing import Set, Dict, Callable, Union, List, Optional
import numpy as np
from PIL import Image

# Define type aliases for clarity
ImageType = Union[np.ndarray, Image.Image]
HashFunctionType = Callable[[ImageType], str]

# --- Hashing Function Implementations (Moved here temporarily for definition,
# --- ideally they'd be in a separate `hashing_functions.py` and imported)
# --- Note: For brevity, function bodies are omitted here but are the same as v1.2.0
# --- Assume functions like dhash, wavelet_hash, histogram_hash, gabor_hash,
# --- sift_hash, surf_hash, orb_hash, tamura_hash, fuzzy_hash, geometric_hash,
# --- fourier_hash, skeleton_hash, contour_hash, zoning_hash,
# --- stroke_direction_hash, junction_hash are defined correctly here or imported.

# Placeholder functions (replace with actual implementations from v1.2.0)
def feature_to_hash(descriptors: Optional[np.ndarray]) -> str:
    """Quantize descriptors and convert to a hexadecimal digest"""
    if descriptors is None or descriptors.size == 0:
        return "0" * 32  # Or some other indicator of failure/no features
    descriptors = descriptors.flatten()
    # Ensure descriptors are valid floats beforeastype
    descriptors = descriptors[np.isfinite(descriptors)]
    if descriptors.size == 0:
        return "0" * 32
    # Normalize potentially large values before scaling
    max_val = np.max(np.abs(descriptors))
    if max_val > 0:
        norm_descriptors = descriptors / max_val
    else:
        norm_descriptors = descriptors
    quantized = (norm_descriptors * 15).astype(int)  # Scale to 0-15 range
    hash_value = "".join(
        [format(v & 0xF, "x") for v in quantized[:32]]
    )  # Use bitwise AND for safety
    return hash_value.ljust(32, "0")  # Pad if less than 32 chars


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
        return "histogram_error"  # Indicate specific error


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
    except cv2.error:  # Handles cases where SIFT/SURF might not be available
        return "sift_unavailable_or_error"


def surf_hash(image: np.ndarray) -> str:
    try:
        # SURF requires opencv-contrib-python
        surf = cv2.xfeatures2d.SURF_create()
        _, descriptors = surf.detectAndCompute(image, None)
        return feature_to_hash(descriptors)
    except AttributeError:  # Catch if xfeatures2d module isn't present
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


# --- Add ALL other custom hash implementations here (tamura, fuzzy, etc.) ---
# (Assuming they are defined as before)
def tamura_hash(image: np.ndarray) -> str:
    try:
        coarseness = np.mean(cv2.absdiff(image[:-1, :], image[1:, :])) + np.mean(
            cv2.absdiff(image[:, :-1], image[:, 1:])
        )
        contrast = np.std(image)
        # Ensure values are finite before formatting
        coarseness = coarseness if np.isfinite(coarseness) else 0
        contrast = contrast if np.isfinite(contrast) else 0
        hash_value = format(min(int(coarseness), 0xFFFF), "04x") + format(
            min(int(contrast), 0xFFFF), "04x"
        )  # Limit size
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
        # Add epsilon to avoid log(0)
        magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1e-10)
        max_mag = np.max(magnitude_spectrum)
        if max_mag <= 0:  # Handle uniform images
            magnitude_spectrum_norm = np.zeros_like(magnitude_spectrum)
        else:
            magnitude_spectrum_norm = magnitude_spectrum / max_mag * 255

        magnitude_spectrum_uint8 = magnitude_spectrum_norm.astype(np.uint8)
        hash_value = "".join(
            [format(v, "02x") for v in magnitude_spectrum_uint8.flatten()[:32]]
        )
        return hash_value.ljust(32, "0")
    except Exception as e:
        # Log the specific exception e if logger is available
        return f"fourier_error"


# Assume skeleton_hash, contour_hash, zoning_hash, stroke_direction_hash, junction_hash
# are defined here with appropriate try-except blocks returning error strings.
# Example for skeleton_hash:
from skimage.morphology import skeletonize


def skeleton_hash(image: np.ndarray) -> str:
    try:
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.size == 0:
            return "skeleton_empty_input"

        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        binary = binary.astype(bool)

        # Check if binary image is empty before skeletonize
        if not np.any(binary):
            return "0" * 11  # Return fixed hash for empty skeleton

        skeleton = skeletonize(binary)
        # ... (rest of the skeleton_hash logic from v1.2.0) ...
        # Wrap calculations in checks for empty arrays/division by zero
        # ...
        # Example values calculation (ensure safety)
        endpoint_count = 0  # initialize
        # ... calculate counts safely ...
        values = [...]  # calculated values
        hash_value = "".join(format(min(v, 15), "x") for v in values)
        return hash_value.ljust(11, "0")  # Ensure fixed length
    except Exception as e:
        # Log e
        return "skeleton_error"


# Add other custom hash function implementations here...
def contour_hash(image: np.ndarray) -> str:
    try:
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.size == 0:
            return "contour_empty_input"
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return "0" * 16
        # ... (rest of contour_hash logic) ...
        hash_value = "".join(format(int(v), "x") for v in hash_components[:16])
        return hash_value.ljust(16, "0")
    except Exception:
        return "contour_error"


def zoning_hash(image: np.ndarray) -> str:
    try:
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.size == 0:
            return "zoning_empty_input"
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        # ... (rest of zoning_hash logic) ...
        hash_value = "".join(format(d, "x") for d in densities)
        return hash_value.ljust(16, "0")  # zones_y * zones_x = 16
    except Exception:
        return "zoning_error"


from skimage.morphology import thin


def stroke_direction_hash(image: np.ndarray) -> str:
    try:
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.size == 0:
            return "stroke_empty_input"
        # ... (rest of stroke_direction_hash logic) ...
        bin_hash = "".join(format(b, "x") for b in normalized_bins)
        ratio_hash = format(hv_ratio, "x") + format(diag_ratio, "x")
        return (bin_hash + ratio_hash).ljust(10, "0")  # 8 bins + 2 ratios
    except Exception:
        return "stroke_error"


def junction_hash(image: np.ndarray) -> str:
    try:
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.size == 0:
            return "junction_empty_input"
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        binary = binary.astype(bool)
        if not np.any(binary):
            return "0" * 16  # Handle empty binary image
        thinned = thin(binary)
        # ... (rest of junction_hash logic) ...
        hash_value = "".join(format(f, "x") for f in features)
        return hash_value.ljust(16, "0")  # 4 totals + 3*4 distributions
    except Exception:
        return "junction_error"


# --- OpenCV Hashes ---
CV2_HASH_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # Check availability at runtime
}
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
    print(
        "Warning: Some or all cv2.img_hash functions not found. Check OpenCV installation.",
        file=sys.stderr,
    )


# --- Other Hashing Libraries ---
IMAGEHASH_FUNCTIONS: Dict[str, HashFunctionType] = {
    "dhash": dhash,
    "wavelet": wavelet_hash,
}

# --- Custom Hash Functions ---
# Assume grayscale input unless otherwise specified
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

# Assume color input
CUSTOM_HASH_FUNCTIONS_COLOR: Dict[str, HashFunctionType] = {
    "histogram": histogram_hash,
    "gabor": gabor_hash,  # Gabor can work on color or gray, here defined for color
}


# --- Hash Categories ---
# Define sets for easier selection
CV2_HASH_NAMES: Set[str] = set(CV2_HASH_FUNCTIONS.keys())
IMAGEHASH_NAMES: Set[str] = set(IMAGEHASH_FUNCTIONS.keys())
CUSTOM_GRAY_NAMES: Set[str] = set(CUSTOM_HASH_FUNCTIONS_GRAY.keys())
CUSTOM_COLOR_NAMES: Set[str] = set(CUSTOM_HASH_FUNCTIONS_COLOR.keys())

PERCEPTUAL_HASHES: Set[str] = {"Average", "BlockMean", "PHash", "dhash", "wavelet"}
FEATURE_HASHES: Set[str] = {"sift", "surf", "orb"}  # Keypoint-based
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

# Map category names to sets
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
    # Add combinations if needed, e.g., "basic": PERCEPTUAL_HASHES | {"histogram"}
    "basic": PERCEPTUAL_HASHES | {"histogram", "zoning"},
}


def get_selected_hashes(specifier: str) -> Set[str]:
    """
    Parses the hash specifier string (e.g., "all", "basic,-sift", "dhash,phash")
    and returns the set of selected hash names.
    """
    if not specifier:
        return HASH_CATEGORIES["basic"]  # Default to basic if empty

    selected_hashes = set()
    parts = specifier.lower().split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("-"):
            # Exclusion
            exclude_part = part[1:]
            if exclude_part in HASH_CATEGORIES:
                selected_hashes -= HASH_CATEGORIES[exclude_part]
            elif exclude_part in ALL_HASHES:
                selected_hashes.discard(exclude_part)
        else:
            # Inclusion
            if part in HASH_CATEGORIES:
                selected_hashes.update(HASH_CATEGORIES[part])
            elif part in ALL_HASHES:
                selected_hashes.add(part)
            else:
                print(
                    f"Warning: Unknown hash or category '{part}' ignored.",
                    file=sys.stderr,
                )

    # Handle the case where only exclusions were provided (e.g., "--hashes -sift")
    # This implies starting from 'all' and removing the exclusions.
    if all(p.startswith("-") for p in parts if p) and not any(
        p for p in parts if not p.startswith("-")
    ):
        initial_set = ALL_HASHES.copy()
        for part in parts:
            part = part.strip()
            if not part:
                continue
            exclude_part = part[1:]
            if exclude_part in HASH_CATEGORIES:
                initial_set -= HASH_CATEGORIES[exclude_part]
            elif exclude_part in ALL_HASHES:
                initial_set.discard(exclude_part)
        return initial_set

    # If only 'none' is specified
    if "none" in [p.strip() for p in parts] and len(parts) == 1:
        return set()

    return selected_hashes


# Example Usage:
# selected = get_selected_hashes("basic,-dhash,sift")
# selected = get_selected_hashes("all,-feature")
# selected = get_selected_hashes("none")
