#!/usr/bin/env python3

""" ir.py  -  Generate report from image """

import argparse
from collections import namedtuple
import logging
from pathlib import Path
import platform
import sys

import numpy as np
import cv2
from lxml import etree
from PIL import Image
import imagehash
from skimage.feature import local_binary_pattern
from skimage.feature import hog as skimage_hog


PROG = Path(__file__).stem
VERSION = "1.0.0"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def parse_args(args):
    """Parse command-line arguments"""

    argp = argparse.ArgumentParser(
        prog=PROG,
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    argp.add_argument(
        "image_path",
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
    params=None, log_file=None, stream_level=logging.DEBUG, file_level=logging.DEBUG
):
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


def non_max_suppression(boxes, overlap_thresh):
    """
    Optimized version of the non max suppression algorithm.
    Condenses extraneous boxes based on overlap_thresh
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


def boxes_from_image(img, overlap_thresh=0):
    """Use MSER to get bounding boxes for all contiguous pixel areas in image_path"""

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


def spectral_analysis(img):
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


def feature_to_hash(descriptors):
    # Quantize the descriptors and convert to hexadecimal
    descriptors = descriptors.flatten()
    descriptors = (descriptors / 255.0 * 15).astype(int)  # Quantize to 4-bit values
    hash_value = "".join(
        [format(v, "x") for v in descriptors[:32]]
    )  # Taking the first 32 values
    return hash_value


def dhash(image):
    # image = Image.fromarray(image)
    return str(imagehash.dhash(image))


def wavelet_hash(image):
    # image = Image.fromarray(image)
    return str(imagehash.whash(image))


def histogram_hash(image):
    histogram = cv2.calcHist(
        [image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    histogram = cv2.normalize(histogram, histogram).flatten()
    return "".join([format(int(v * 255), "02x") for v in histogram])


def gabor_hash(image):
    gabor_filter = cv2.getGaborKernel(
        (5, 5), 1.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F
    )
    gabor_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_filter)
    hash_value = "".join(
        [format(int(v), "02x") for v in gabor_image.flatten()[:32]]
    )  # Taking the first 32 values
    return hash_value


def sift_hash(image):
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image, None)
    return feature_to_hash(descriptors)


def surf_hash(image):
    surf = cv2.xfeatures2d.SURF_create()
    _, descriptors = surf.detectAndCompute(image, None)
    return feature_to_hash(descriptors)


def orb_hash(image):
    orb = cv2.ORB_create()
    _, descriptors = orb.detectAndCompute(image, None)
    return feature_to_hash(descriptors)


def tamura_hash(image):
    coarseness = np.mean(cv2.absdiff(image[:-1, :], image[1:, :])) + np.mean(
        cv2.absdiff(image[:, :-1], image[:, 1:])
    )
    contrast = np.std(image)
    hash_value = format(int(coarseness), "04x") + format(int(contrast), "04x")
    return hash_value


def fuzzy_hash(image):
    quantized_image = (image // 32).astype(np.uint8)
    hash_value = "".join(
        [format(v, "02x") for v in quantized_image.flatten()[:32]]
    )  # Taking the first 32 values
    return hash_value


def geometric_hash(image):
    edges = cv2.Canny(image, 100, 200)
    hash_value = "".join(
        [format(v, "02x") for v in edges.flatten()[:32]]
    )  # Taking the first 32 values
    return hash_value


def fourier_hash(image):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted))
    magnitude_spectrum = (magnitude_spectrum / magnitude_spectrum.max() * 255).astype(
        np.uint8
    )
    hash_value = "".join(
        [format(v, "02x") for v in magnitude_spectrum.flatten()[:32]]
    )  # Taking the first 32 values
    return hash_value


def create_element(parent, tag, text=None, attrib=None):
    """Helper function to create an XML element"""

    if attrib is None:
        attrib = {}

    elem = etree.SubElement(parent, tag, attrib)
    if text is not None:
        elem.text = text

    return elem


def create_hash_element(parent, hash_type, image):
    """Function to create hash elements"""

    if hash_type in cv2_hash_functions:
        hash_function = hash_type + "Hash_create"
        hasher = getattr(cv2.img_hash, hash_function)()
        hash_sub_elem = create_element(parent, hash_type)
        hash_sub_elem.text = hasher.compute(image).tobytes().hex()

    else:
        hash_sub_elem = create_element(parent, hash_type.__name__)
        hash_sub_elem.text = hash_type(image)


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
}

Size = namedtuple("Size", ["width", "height"])

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
    "Letter": Size(8.5 * 25.4, 11 * 25.4),
    "Legal": Size(8.5 * 25.4, 14 * 25.4),
    "Tabloid": Size(11 * 25.4, 17 * 25.4),
    "Ledger": Size(17 * 25.4, 11 * 25.4),
    # Japanese JIS B Sizes
    "JIS B0": Size(1030, 1456),
    "JIS B1": Size(728, 1030),
    "JIS B2": Size(515, 728),
    "JIS B3": Size(364, 515),
    "JIS B4": Size(257, 364),
    "JIS B5": Size(182, 257),
}


def guess_paper_size(img, dpi=600, top_n=3):
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


def guess_dpi(paper_type, pixel_width, pixel_height):
    if paper_type not in standard_sizes:
        raise ValueError(f"Unknown paper type: {paper_type}")

    paper_size = standard_sizes[paper_type]

    paper_width_in_inches = paper_size.width / 25.4
    paper_height_in_inches = paper_size.height / 25.4

    dpi_width = pixel_width / paper_width_in_inches
    dpi_height = pixel_height / paper_height_in_inches

    return dpi_width, dpi_height


def main(args):
    """script entry-point"""

    params = parse_args(args)
    init(params, log_file=params.image_path.with_suffix(".ir"))

    # in
    try:
        img = cv2.imread(str(params.image_path))
    except Exception as exc:
        logger.critical(exc, exc_info=True)
        raise exc
    if img is None:
        logger.critical("Failed to read image '%s'", params.image_path.resolve())
        raise IOError(f"Failed to read image '{params.image_path.resolve()}'")

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pimg = Image.open(params.image_path)

    # <report>
    root = etree.Element("report")

    #   <Metadata>
    info_elem = etree.SubElement(root, "metadata")

    #     <hash>
    hash_elem = etree.SubElement(info_elem, "hash")
    for func in cv2_hash_functions:
        create_hash_element(hash_elem, func, gimg)

    create_hash_element(hash_elem, histogram_hash, img)
    create_hash_element(hash_elem, gabor_hash, gimg)
    create_hash_element(hash_elem, tamura_hash, gimg)
    create_hash_element(hash_elem, fuzzy_hash, gimg)
    create_hash_element(hash_elem, geometric_hash, gimg)
    create_hash_element(hash_elem, fourier_hash, gimg)
    create_hash_element(hash_elem, dhash, pimg)
    create_hash_element(hash_elem, wavelet_hash, pimg)

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
    path_elem = create_element(info_elem, "path", text=str(params.image_path.resolve()))
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

    # out
    etree.ElementTree(root).write(
        str(params.out_path), encoding="UTF-8", xml_declaration=True, pretty_print=True
    )
    logger.info("💥 %s", params.out_path.resolve())


if __name__ == "__main__":
    main(sys.argv[1:])
