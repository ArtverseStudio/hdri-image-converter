# HDR Image Converter

---

A Python pipeline for converting standard 8-bit images into 32-bit HDR images using synthetic exposure stacking, luminance-based masking, and Reinhard tone mapping. Designed for Blender, 3D, and VFX workflows.

---

## Features
- **Single-image HDR expansion** or synthetic exposure stacking
- **Luminance-based masking** for highlight recovery
- **Reinhard tone mapping** for natural previews
- **Automatic white balance and color correction**
- **Batch processing** for folders of images
- **Configurable pipeline** via `src/config.py`
- **Clean output structure**

---

## Quickstart
1. **Install requirements:**
   ```bash
   pip install opencv-python numpy imageio
   ```
2. **Place your 8-bit input images in the `input/` folder.**
3. **Run the pipeline:**
   ```bash
   # Single image
   python -m src.cli single input/your_image.jpg output/your_output.hdr

   # Batch mode (all images in input/)
   python -m src.cli batch input output
   ```
4. **Check the `output/` folder** for HDR images and tone-mapped previews.

---

## Folder Structure
```
project_root/
├── input/              # Place input images here
├── output/             # All results saved here
├── src/                # Source code
│   ├── cli.py
│   ├── config.py
│   ├── hdr_converter.py
│   ├── tonemap.py
│   └── ...
├── README.md
├── LICENSE
└── .gitignore
```

---

## Configuration
All parameters can be adjusted in `src/config.py`:
- **LUMINANCE_PERCENTILE**: Mask threshold for highlight recovery (90-99)
- **FEATHER_RADIUS**: Mask edge softness (0-15)
- **BLEND_RATIO**: Highlight blending (0.0-1.0)
- **EXPOSURE_START/END/STEP**: Synthetic exposure range
- **NORMALIZATION_KEY**: Scene brightness (lower = darker HDRI)
- **PREVIEW_SCALE**: Only affects preview brightness
- **TONEMAP_PARAMS**: Reinhard tone mapping parameters

---

## Adjusting HDRI Brightness
To make your HDRI output darker or brighter, edit the `NORMALIZATION_KEY` parameter in `src/config.py`:
```python
NORMALIZATION_KEY = 0.20  # Lower = darker, Higher = brighter (typical range: 0.05–0.25)
```
After changing this value, re-run the pipeline to see the effect.

---

## Contributing
Pull requests and issues are welcome! Please open an issue for bugs or feature requests.

---

## License
MIT License
