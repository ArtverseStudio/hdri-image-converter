"""
Tone Mapping Module
-------------------
Provides access to OpenCV's Reinhard tone mapping algorithm and utilities for applying it.

Functions:
    safe_tonemap: Safely apply Reinhard tone mapping to HDR images.
    get_tonemap_operators: Factory for Reinhard tone mapping operator.
    clamp_image: Safely clamp image values.
"""

import cv2
import numpy as np
import logging
from typing import Dict

from src.config import Config # Import Config for parameter access

# Setup logger
logger = logging.getLogger(__name__)

def get_tonemap_operators(config: Config) -> Dict[str, cv2.Tonemap]:
    """Return a dictionary with only the Reinhard tone mapping operator."""
    params = config.TONEMAP_PARAMS.get('Reinhard', {})
    return {
        'Reinhard': cv2.createTonemapReinhard(
            gamma=params.get('gamma', 1.0),
            intensity=params.get('intensity', 0.0),
            light_adapt=params.get('light_adapt', 0.0),
            color_adapt=params.get('color_adapt', 0.0)
        )
    }

def safe_tonemap(hdr_image: np.ndarray, operator: cv2.Tonemap) -> np.ndarray:
    """
    Apply the Reinhard tone mapping operator to an HDR image with basic error handling.
    Input HDR image should be float32, BGR format, and non-negative.

    Args:
        hdr_image: Input HDR image (float32, BGR).
        operator: Initialized OpenCV Reinhard tone mapping operator.

    Returns:
        Tone-mapped LDR image (float32, BGR, range typically [0, 1]).
        Returns a black image of the same size on failure.
    """
    if not isinstance(hdr_image, np.ndarray) or hdr_image.dtype != np.float32:
        logger.error("Input image for tone mapping must be float32 NumPy array.")
        return np.zeros_like(hdr_image, dtype=np.float32) if isinstance(hdr_image, np.ndarray) else np.zeros((100,100,3), dtype=np.float32)
    if hdr_image.min() < 0:
         logger.warning(f"Tone mapping input has negative values (min={hdr_image.min():.4f}). Clamping to 0.")
         hdr_image = np.maximum(hdr_image, 0)
    try:
        ldr_image = operator.process(hdr_image.copy())
        ldr_image_clipped = np.clip(ldr_image, 0, 1)
        return ldr_image_clipped.astype(np.float32)
    except cv2.error as cv_err:
         logger.error(f"OpenCV error during tone mapping process: {cv_err}", exc_info=False)
         return np.zeros_like(hdr_image, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error applying tone mapping operator {type(operator).__name__}: {e}", exc_info=True)
        return np.zeros_like(hdr_image, dtype=np.float32)

def clamp_image(img: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    Clamp image pixel values to a specified range [min_val, max_val].
    """
    clamped_img = img.copy()
    if min_val is not None:
        clamped_img = np.maximum(clamped_img, min_val)
    if max_val is not None:
        clamped_img = np.minimum(clamped_img, max_val)
    return clamped_img
