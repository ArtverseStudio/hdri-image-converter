"""
Color Space Conversion and White Balance Module
-------------------------------------------------
Provides functions for converting between color spaces (sRGB <-> linear RGB)
and applying white balance corrections.
"""

import numpy as np
import cv2 # Used only if HAS_XPHOTO for specific WB
import logging
from typing import Tuple, Optional, Dict

# Configure logger
logger = logging.getLogger(__name__)

# Check for optional OpenCV modules needed for some WB methods
try:
    import cv2.xphoto
    HAS_XPHOTO = True
except ImportError:
    HAS_XPHOTO = False

# Constants for sRGB conversion
SRGB_GAMMA_THRESHOLD = 0.04045
SRGB_ALPHA = 0.055

def srgb_to_linear(img_srgb: np.ndarray) -> np.ndarray:
    """
    Convert an image from sRGB color space to linear RGB.
    Assumes input is float type in [0, 1] range.
    Uses the standard sRGB Electro-Optical Transfer Function (EOTF) inverse.
    
    Args:
        img_srgb: Image in sRGB color space (float, range [0, 1]).
        
    Returns:
        Image in linear RGB color space (float, range [0, 1+]).
    """
    if not np.issubdtype(img_srgb.dtype, np.floating):
        raise ValueError("Input image must be float type for srgb_to_linear")
    
    # Create masks for linear and gamma parts of the sRGB curve
    linear_mask = img_srgb <= SRGB_GAMMA_THRESHOLD
    gamma_mask = ~linear_mask # Inverse of linear_mask
    
    # Apply inverse transformations
    img_linear = np.zeros_like(img_srgb)
    img_linear[linear_mask] = img_srgb[linear_mask] / 12.92
    img_linear[gamma_mask] = np.power((img_srgb[gamma_mask] + SRGB_ALPHA) / (1 + SRGB_ALPHA), 2.4)
    
    return img_linear

def linear_to_srgb(img_linear: np.ndarray) -> np.ndarray:
    """
    Convert an image from linear RGB color space to sRGB.
    Assumes input is float type, typically in [0, 1+] range.
    Clips negative values before conversion. Values > 1 are handled by the formula.
    Uses the standard sRGB Opto-Electronic Transfer Function (OETF).
    
    Args:
        img_linear: Image in linear RGB color space (float, range [0, 1+]).
        
    Returns:
        Image in sRGB color space (float, range [0, 1]).
    """
    if not np.issubdtype(img_linear.dtype, np.floating):
        raise ValueError("Input image must be float type for linear_to_srgb")
        
    # Clip negative values to zero before applying OETF
    img_linear_nonneg = np.maximum(img_linear, 0)

    # Create masks for linear and gamma parts of the OETF
    # Note: The threshold for the *inverse* EOTF corresponds to 0.0031308 in linear space
    linear_threshold_linear = 0.0031308 
    linear_mask = img_linear_nonneg <= linear_threshold_linear
    gamma_mask = ~linear_mask

    # Apply OETF transformations
    img_srgb = np.zeros_like(img_linear_nonneg)
    img_srgb[linear_mask] = img_linear_nonneg[linear_mask] * 12.92
    img_srgb[gamma_mask] = (1 + SRGB_ALPHA) * np.power(img_linear_nonneg[gamma_mask], 1.0/2.4) - SRGB_ALPHA
    
    # Final result should be clipped to [0, 1] as per sRGB standard
    return np.clip(img_srgb, 0, 1)

def calculate_white_balance(image: np.ndarray, method: str = 'gray_world') -> np.ndarray:
    """
    Calculate white balance scaling factors based on the chosen method.
    
    Args:
        image: Input image (linear RGB float assumed, but works on sRGB too).
        method: White balance algorithm ('gray_world', 'simple' via OpenCV if available).
                Other methods like 'learning_based' could be added here.
                
    Returns:
        Numpy array of scaling factors [scale_R, scale_G, scale_B].
        Returns [1., 1., 1.] if method is unknown or fails.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel image (H, W, C)")
        
    logger.debug(f"Calculating white balance factors using method: {method}")
    h, w, c = image.shape
    image_float = image.astype(np.float32) # Ensure float for calculations
    
    # Default scale factors (no change)
    wb_scales = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    try:
        if method == 'gray_world':
            # Assumes average color of the scene is gray
            avg_rgb = np.mean(image_float, axis=(0, 1))
            # Scale factors are mean_gray / mean_channel
            mean_gray = np.mean(avg_rgb)
            # Avoid division by zero for pure black images/channels
            wb_scales = mean_gray / (avg_rgb + 1e-9)
            # Normalize scales so Green channel scale is 1.0 (common practice)
            wb_scales /= (wb_scales[1] + 1e-9)
            logger.debug(f"Gray World WB scales (R, G, B): {wb_scales}")
            
        elif method == 'simple':
            # Requires OpenCV xphoto module
            if HAS_XPHOTO and hasattr(cv2.xphoto, 'createSimpleWB'):
                logger.debug("Using OpenCV SimpleWB (internal calculation, returning [1,1,1])")
                # SimpleWB applies correction directly, doesn't return scales easily.
                # We return default scales and rely on apply_white_balance to call SimpleWB.
                pass # Keep default wb_scales = [1, 1, 1]
            else:
                logger.warning("OpenCV xphoto module or SimpleWB not available for 'simple' WB method. Using default [1,1,1].")
                
        # Placeholder for other potential methods
        # elif method == 'learning_based':
        #     # Implement learning-based white balance if needed
        #     pass
            
        else:
            logger.warning(f"Unknown white balance method: '{method}'. Using default [1,1,1].")

        # Ensure scales are positive
        wb_scales = np.maximum(wb_scales, 1e-9)
        return wb_scales.astype(np.float32)

    except Exception as e:
        logger.error(f"Error calculating white balance factors with method '{method}': {e}", exc_info=True)
        # Return default scales on error
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)

def apply_white_balance(image: np.ndarray, wb_scales: Optional[np.ndarray] = None, method: str = 'gray_world') -> np.ndarray:
    """
    Apply white balance correction to an image.
    Can either apply pre-calculated scaling factors or use an OpenCV method directly.

    Args:
        image: Input image (linear RGB float recommended for accuracy).
        wb_scales: Optional pre-calculated scaling factors [scale_R, scale_G, scale_B]. 
                   If None, factors are calculated using the specified `method`.
        method: White balance algorithm to use if `wb_scales` is None, or if using
                an OpenCV method like 'simple' that applies correction directly.

    Returns:
        White-balanced image (same type and range characteristics as input).
    """
    # ... (function body as refactored previously) ...
