# Image Report Generator (ir.py)

## Overview
**ir.py** is a Python script designed to analyze images and generate structured XML reports. This tool extracts metadata, calculates various image hashes, and identifies bounding boxes of contiguous pixel areas. It is useful for image processing, document analysis, and feature extraction.

## Features
- **Metadata Extraction:** Collects image dimensions and path information.
- **Hash Generation:** Supports multiple hashing methods including average, wavelet, histogram, and Gabor hashes.
- **Bounding Box Detection:** Utilizes MSER (Maximally Stable Extremal Regions) to detect regions of interest.
- **Spectral Analysis:** Analyzes pixel distribution to segment images into rows and columns.
- **Paper Size Estimation:** Estimates the paper size of the image by comparing it to standard dimensions.

## Dependencies
- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Pillow (`PIL`)
- lxml (`etree`)
- scikit-image (`skimage`)
- imagehash

Ensure all dependencies are installed by running:
```bash
pip install opencv-python-headless numpy pillow lxml scikit-image imagehash
```

## Usage
### Command Line Execution
```bash
python3 ir.py <image_path> <output_path>
```
### Arguments
- `image_path` : Path to the input image file.
- `out_path` : Path to save the generated XML report.

### Optional Flags
- `--help` : Display help message.
- `--version` : Show script version.

## Example
```bash
python ir.py ./sample_image.jpg ./output_report.xml
```

## Output
The script generates an XML report containing:
- **Metadata**: Hash values, image dimensions, and file path.
- **Bounding Boxes**: Detected regions in the image.
- **Spectral Analysis**: Row and column segmentation based on pixel density.

## Functions Overview
- **parse_args()** : Parses command-line arguments.
- **init()** : Initializes logging configuration.
- **non_max_suppression()** : Reduces overlapping bounding boxes.
- **boxes_from_image()** : Detects bounding boxes in the image.
- **spectral_analysis()** : Analyzes pixel distribution.
- **hash functions** : Multiple hash algorithms for feature extraction.
- **create_element()** : Helper function to generate XML elements.
- **main()** : Entry point to process the image and generate the report.

## Logging
Logs are saved alongside the image with the `.ir` extension, providing details of the process and any encountered issues.

## License
This project is licensed under the MIT License.

## Version
- **1.0.0**

