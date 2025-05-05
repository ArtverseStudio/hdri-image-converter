"""
Image Masking Module
--------------------
Functions for creating masks based on image properties like luminance,
and applying effects using masks (e.g., feathering, saving debug visuals).
"""

import cv2
import numpy as np
from typing import Union, Any, Optional, Tuple, Literal
from pathlib import Path
from src.logger import setup_logger
from src.fileio import save_image, ensure_dir # Use fileio for saving

logger = setup_logger("mask")

def create_luminance_mask(img_gray_uint8: np.ndarray, percentile: int = 95, feather_radius: int = 5) -> np.ndarray:
    """
    Create a binary mask identifying the brightest regions based on a percentile threshold,
    optionally applying Gaussian blur for soft edges (feathering).

    Args:
        img_gray_uint8: Input 8-bit grayscale image (0-255).
        percentile: Luminance percentile (0-100) to use as the threshold. 
                    Higher values select brighter areas.
        feather_radius: Radius for Gaussian blur to soften mask edges. 
                        Must be a positive odd integer. If 0 or even, no feathering is applied.

    Returns:
        A float32 mask array with values in the range [0, 1].
    """
    if img_gray_uint8.ndim != 2:
        raise ValueError("Input image must be grayscale (2 dimensions)")
    if img_gray_uint8.dtype != np.uint8:
        raise ValueError("Input image must be 8-bit unsigned integer (uint8)")
    if not (0 < percentile <= 100):
        raise ValueError("Percentile must be between 0 (exclusive) and 100 (inclusive)")

    try:
        # Determine the threshold value based on the percentile
        threshold_value = np.percentile(img_gray_uint8, percentile)
        logger.debug(f"Luminance mask threshold ({percentile}th percentile): {threshold_value:.2f}")

        # Create binary mask: 1 where intensity >= threshold, 0 otherwise
        # Note: Using >= includes the threshold value itself
        mask_binary = (img_gray_uint8 >= threshold_value).astype(np.uint8) * 255
        
        # Feather the mask edges using Gaussian blur if radius is valid
        mask_feathered = mask_binary # Start with binary mask
        if feather_radius > 0 and feather_radius % 2 != 0: # Radius must be positive and odd
            logger.debug(f"Applying Gaussian blur feathering with radius: {feather_radius}")
            mask_feathered = cv2.GaussianBlur(mask_binary, (feather_radius, feather_radius), 0)
        elif feather_radius > 0:
            logger.warning(f"Feather radius ({feather_radius}) must be odd. Skipping feathering.")
        else:
             logger.debug("Feather radius is 0 or less. Skipping feathering.")

        # Normalize the final mask to float32 range [0, 1]
        mask_float = mask_feathered.astype(np.float32) / 255.0
        return mask_float
        
    except Exception as e:
        logger.error(f"Error creating luminance mask: {e}", exc_info=True)
        # Return a default empty mask (all zeros) on error?
        # Or re-raise? For now, re-raise to indicate failure.
        raise

def save_mask_debug(image: np.ndarray, mask: np.ndarray, filename: Union[str, Path] = 'mask_debug.jpg') -> bool:
    """
    Save a visualization of a mask applied to an image for debugging.
    Overlays the mask (typically float [0,1]) onto the image using a distinct color (e.g., red).

    Args:
        image: The original image (BGR uint8 or float [0,1]) to overlay the mask onto.
        mask: The mask to visualize (grayscale, float [0,1] recommended).
        filename: Path where the debug image should be saved.

    Returns:
        True if the debug image was saved successfully, False otherwise.
    """
    try:
        # Ensure the output directory exists
        output_path = Path(filename)
        if not ensure_dir(output_path.parent):
             logger.error(f"Cannot save mask debug image, failed to ensure output directory: {output_path.parent}")
             return False
             
        # Prepare image: Convert to BGR uint8 if needed
        if image.dtype != np.uint8:
             if np.max(image) <= 1.0 and np.min(image) >= 0.0:
                 image_uint8 = (image * 255).astype(np.uint8)
             else:
                 # Attempt normalization if range seems off (e.g., linear HDR)
                 logger.warning("Input image for mask debug is not uint8 or float [0,1]. Attempting normalization.")
                 norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                 image_uint8 = norm_img.astype(np.uint8)
                 if image_uint8.ndim == 2: # Convert grayscale to BGR
                      image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
        else:
            image_uint8 = image
            
        if image_uint8.ndim == 2: # Convert grayscale to BGR if needed
            image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
        elif image_uint8.shape[2] != 3:
             logger.error(f"Cannot visualize mask on image with unsupported channels: {image_uint8.shape}")
             return False

        # Prepare mask: Ensure it's single-channel float [0,1]
        if mask.ndim == 3:
            logger.warning("Input mask has 3 channels, converting to grayscale for visualization.")
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
        
        if mask_gray.dtype != np.float32:
            if np.max(mask_gray) > 1.0 and mask_gray.dtype == np.uint8:
                 mask_float = mask_gray.astype(np.float32) / 255.0
            else:
                 mask_float = mask_gray.astype(np.float32) # Assume it's [0,1] if not uint8
        else:
             mask_float = mask_gray

        mask_float = np.clip(mask_float, 0, 1) # Ensure [0,1] range

        # Create colored overlay: Red where mask is high
        overlay = np.zeros_like(image_uint8, dtype=np.float32)
        # Red color in BGR is (0, 0, 255)
        overlay[..., 2] = mask_float * 255 # Apply mask intensity to Red channel

        # Blend the overlay with the original image
        # alpha controls the transparency of the overlay
        alpha = 0.5 
        blended = cv2.addWeighted(image_uint8.astype(np.float32), 1.0 - alpha, overlay, alpha, 0)
        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Save the blended image
        success = save_image(output_path, blended_uint8)
        if success:
            logger.info(f"Saved mask debug image: {output_path}")
        return success
        
    except Exception as e:
        logger.error(f"Error saving mask debug image to {filename}: {e}", exc_info=True)
        return False
