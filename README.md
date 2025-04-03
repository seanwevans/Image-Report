# IR (Image Reporter)

A versatile image analysis tool for generating detailed XML reports from images.

## Description

IR (Image Reporter) is a Python-based command-line utility for extracting and analyzing features from images. It generates comprehensive XML reports containing:

- Metadata about the image (dimensions, estimated paper size)
- Various image hash values (perceptual, feature-based, structural, etc.)
- Bounding boxes for detected regions of interest
- Spectral analysis (horizontal and vertical projection profiles)

The tool is designed to be efficient and flexible, with support for processing individual images or batches of images in parallel.

## Features

- **Comprehensive Metadata**: Automatically detects image dimensions and estimates paper sizes
- **Multiple Hashing Algorithms**: Supports 25+ image hashing algorithms including:
  - Perceptual hashes (Average, PHash, DHash, Wavelet)
  - Feature-based hashes (SIFT, SURF, ORB)
  - Structural hashes (Marr-Hildreth, Contour, Skeleton)
  - Texture hashes (Gabor, Tamura)
  - Color hashes (ColorMoment, Histogram)
- **Region Detection**: Identifies regions of interest using MSER algorithm with non-maximum suppression
- **Spectral Analysis**: Calculates horizontal and vertical projection profiles
- **Parallel Processing**: Efficiently processes batches of images using multiprocessing
- **Flexible Configuration**: Extensive command-line options for customization

## Installation

### Prerequisites

- Python 3.7+
- Required packages (install via pip):
  ```
  opencv-contrib-python
  numpy
  lxml
  Pillow
  imagehash
  scikit-image
  ```

### Installation Steps

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ir.git
   cd ir
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make the script executable (Linux/Mac):
   ```
   chmod +x ir
   ```

## Usage

### Basic Usage

Process a single image:
```
./ir input.jpg output.xml
```

Process all images in a directory:
```
./ir input_directory/ output_directory/
```

### Command Line Options

```
usage: ir input_path output_path [options]

Generate an XML report with metadata and features from an image or directory of images.

Processing Options:
  --hashes HASHES       Specify which hashes to compute. Examples: 'all', 'none',
                        'basic', 'perceptual', 'feature', 'sift,dhash',
                        'basic,-dhash'. Categories: all, none, cv2, imagehash,
                        custom_gray, custom_color, perceptual, feature, texture,
                        structural, color, other, basic. Prefix with '-' to
                        exclude. (default: basic)
  --nms-threshold NMS_THRESHOLD
                        Overlap threshold (0.0-1.0) for Non-Maximum Suppression of
                        bounding boxes. (default: 0.3)
  --default-dpi DEFAULT_DPI
                        Assumed DPI for paper size guessing if resolution info is
                        missing. (default: 600)
  --continue-on-error   In batch mode, continue processing other images if one
                        fails. (default: False)
  --max-workers MAX_WORKERS
                        Maximum number of worker processes for batch processing.
                        (default: None)

Logging Options:
  --log-file LOG_FILE   Path to optional log file. If not specified, logging only
                        goes to console. If input is a directory, consider setting
                        this explicitly. Default name in single file mode is
                        <input_name>.ir.log (default: None)
  --log-level-file {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level for the file. (default: DEBUG)
  --log-level-console {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level for the console. (default: INFO)
  -q, --quiet           Suppress console output (sets console level to WARNING).
                        (default: False)

Help and Version:
  --version             Show program's version number and exit.
  -h, --help            Show this help message and exit.
```

### Examples

Calculate only perceptual hashes:
```
./ir input.jpg output.xml --hashes perceptual
```

Process all images in a directory with custom settings:
```
./ir input_dir/ output_dir/ --hashes all,-feature --nms-threshold 0.4 --default-dpi 300 --continue-on-error
```

Verbose output with debug logging:
```
./ir input.jpg output.xml --log-level-console DEBUG
```

## Hash Selection

IR supports multiple categories of image hashes that can be selected using the `--hashes` parameter:

- `basic` - Default set (perceptual hashes plus histogram, zoning)
- `all` - All available hash algorithms
- `perceptual` - Perceptual hashing algorithms (Average, BlockMean, PHash, dhash, wavelet)
- `feature` - Feature-based algorithms (SIFT, SURF, ORB)
- `texture` - Texture-based algorithms (Gabor, Tamura)
- `structural` - Structural algorithms (MarrHildreth, RadialVariance, skeleton, contour, etc.)
- `color` - Color-based algorithms (ColorMoment, histogram)
- `none` - No hashes

Combined selectors are also supported:
```
--hashes perceptual,histogram,zoning  # Only these specific hashes
--hashes all,-feature                 # All hashes except feature-based ones
--hashes basic,-dhash,sift            # Basic set minus dhash, plus SIFT
```

## Output Format

The tool generates XML reports with the following structure:

```xml
<image_report version="1.3.0" source_file="example.jpg">
  <metadata>
    <!-- Source file info, dimensions, estimated paper size -->
  </metadata>
  <hashes count="25">
    <!-- Various hash values -->
  </hashes>
  <analysis>
    <bounding_boxes count="42">
      <!-- Detected regions with coordinates -->
    </bounding_boxes>
    <spectral_analysis>
      <!-- Horizontal and vertical projection profiles -->
    </spectral_analysis>
  </analysis>
</image_report>
```

## Project Structure

- `ir` - Main executable script
- `utils.py` - Utility functions for logging, argument parsing, and XML creation
- `analysis.py` - Image analysis functions (bounding boxes, spectral analysis)
- `hashing_config.py` - Configuration for image hashing functions
- `papersize.py` - Paper size definitions and guessing logic
- `requirements.txt` - Package dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.