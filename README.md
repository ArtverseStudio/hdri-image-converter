# HDR Image Converter

This project converts a standard 8-bit image into a 32-bit HDR image using exposure bracketing, luminance-based light source masking, and advanced tone mapping.

## Features
- Generates multiple exposure brackets from a single input image
- Detects and boosts light sources using a feathered luminance mask
- Merges exposures into a true HDR image
- Produces tone-mapped previews (Reinhard, Drago, Mantiuk)
- Automatic white balance and color correction
- Saves debug images for mask visualization

## Folder Structure
```
project_root/
│
├── input/              # Place input images here
│   └── input.jpg      # Example input image
├── output/            # Generated files appear here
│   ├── output.hdr     # Output HDR image
│   ├── exposure_bracket_*.jpg  # Exposure brackets
│   └── preview_*.jpg  # Tone-mapped previews
├── src/               # Source code
│   ├── cli.py         # Command line interface
│   ├── config.py      # Configuration parameters
│   ├── exposure.py    # Exposure bracketing
│   ├── fileio.py      # File I/O operations
│   ├── hdr_converter.py  # Main HDR pipeline
│   ├── mask.py        # Light source masking
│   └── tonemap.py     # Tone mapping operators
└── README.md          # Project documentation
```

## Usage
1. Place your 8-bit input images in the `input/` folder.
2. Run the script using one of these methods:
   ```bash
   # Process a single image
   python -m src.cli single input.jpg output.hdr

   # Process all images in a folder
   python -m src.cli batch input_folder output_folder
   ```
3. Check the `output/` folder for:
   - HDR image (.hdr and .exr formats)
   - Exposure brackets
   - Tone-mapped previews
   - Light mask visualization

## Requirements
- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

## Customization
All parameters can be adjusted in `src/config.py` with recommended value ranges in the comments:

- **Light Source Detection**
  - LUMINANCE_PERCENTILE: Threshold for detecting light sources (90-99)
  - MASK_INTENSITY: Light source boost strength (1.0-5.0)
  - FEATHER_RADIUS: Edge smoothness (0-15)

- **Exposure Control**
  - EXPOSURE_VALUES: Range -6 to +6 EV, recommended 1-2 EV steps
  - BLEND_RATIO: Light preservation strength (0.0-1.0)
  - GAMMA: Standard is 2.2, range 1.8-2.4

- **Tone Mapping** (per operator)
  - Reinhard: gamma (1.0-2.4), intensity/adaptation (0.0-1.0)
  - Drago: gamma (1.0-3.0), bias (0.7-1.0)
  - Mantiuk: gamma (1.0-3.0), contrast (0.5-2.0), saturation (0.0-2.0)

Refer to the comments in each source file for detailed explanations of parameters and their effects.

## Adjusting HDRI Brightness

To make your HDRI output darker or brighter, edit the `NORMALIZATION_KEY` parameter in your configuration file (`src/config.py`):

```
NORMALIZATION_KEY = 0.20  # Lower = darker, Higher = brighter (typical range: 0.05–0.25)
```

- **Lower values** (e.g., 0.10) make the HDRI darker.
- **Higher values** (e.g., 0.25) make the HDRI brighter.

After changing this value, re-run the pipeline to see the effect.

## License
MIT License
