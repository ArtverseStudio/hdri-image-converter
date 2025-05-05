"""
Exposure Stack Module
---------------------
Handles the creation of exposure-bracketed images from a single input image.
Performs luminance-based light source masking and blending for underexposed brackets.

Functions:
    create_exposure_stack: Generate exposure stack and save bracket images.
"""

from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
from src.config import get_config
from src.mask import create_luminance_mask, save_mask_debug
from src.fileio import save_image
from src.logger import setup_logger
from src.color import linear_to_srgb

logger = setup_logger("exposure")
config = get_config()

def create_exposure_stack(
    img: np.ndarray,
    ev_values: List[float],
    luminance_percentile: Optional[float] = None,
    feather_radius: Optional[int] = None,
    show_mask: Optional[bool] = None,
    blend_ratio: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Generate a stack of exposure-bracketed images from a single input image.
    All processing is done in linear RGB. The light mask is used as a multiplicative boost to preserve highlights.
    Args:
        img: Input image (uint8, HxWx3, sRGB)
        ev_values: List of exposure values (stops)
        luminance_percentile: Light detection threshold
        feather_radius: Mask edge smoothing
        show_mask: Enable/disable debug visualization
        blend_ratio: Light source boost strength
        output_dir: Optional directory to save debug images
    Returns:
        images: List of exposure-bracketed images (uint8, sRGB)
        times: Corresponding exposure times (float32)
    """
    # Use config values if not specified
    luminance_percentile = luminance_percentile if luminance_percentile is not None else config.LUMINANCE_PERCENTILE
    feather_radius = feather_radius if feather_radius is not None else config.FEATHER_RADIUS
    show_mask = show_mask if show_mask is not None else config.SHOW_MASK
    blend_ratio = blend_ratio if blend_ratio is not None else config.BLEND_RATIO
    
    # Convert output_dir to Path if provided
    output_dir_path = Path(output_dir) if output_dir else None
    
    if output_dir_path and not output_dir_path.exists():
        logger.warning(f"Output directory does not exist: {output_dir_path}")
        try:
            output_dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir_path}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            output_dir_path = None

    images = []
    times = []
    GAMMA = config.GAMMA
    # Convert input to float32 sRGB [0,1], then to linear RGB
    img_srgb = img.astype(np.float32) / 255.0
    img_linear = np.clip(img_srgb, 0, 1) ** GAMMA
    
    # Create sharper light mask from linear luminance
    img_gray_linear = cv2.cvtColor(img_linear, cv2.COLOR_BGR2GRAY)
    light_mask = create_luminance_mask((img_gray_linear * 255).astype(np.uint8), luminance_percentile, feather_radius)
    light_mask_3c = np.repeat(light_mask[:, :, np.newaxis], 3, axis=2).astype(np.float32)

    if show_mask and output_dir_path:
        mask_path = output_dir_path / 'light_mask_debug.jpg'
        save_mask_debug((img_srgb * 255).astype(np.uint8), light_mask, filename=str(mask_path))
        logger.info(f"Saved light mask debug image: {mask_path}")

    for ev in ev_values:
        exposure_time = 2.0 ** ev
        times.append(exposure_time)
        
        # Simulate exposure in linear RGB
        exposed = img_linear * exposure_time
        
        # For underexposed stops, gently blend in the light mask to preserve highlights
        if ev < 0:
            blend = blend_ratio
            exposed = (1 - light_mask_3c * blend) * exposed + (light_mask_3c * blend) * img_linear
        
        np.clip(exposed, 0, 1, out=exposed)
        
        # Use standard gamma for display
        exposed_srgb = exposed ** (1.0 / GAMMA)
        exposed_8bit = (exposed_srgb * 255).astype(np.uint8)
        
        images.append(exposed_8bit)
        if output_dir_path:
            bracket_path = output_dir_path / f"exposure_bracket_{ev:+.2f}ev.jpg"
            try:
                save_image(str(bracket_path), exposed_8bit)
                logger.info(f"Saved exposure bracket: {bracket_path}")
            except Exception as e:
                logger.error(f"Failed to save exposure bracket: {e}")
    
    return images, np.array(times, dtype=np.float32)
