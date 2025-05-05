"""
File Input/Output Module
------------------------
Handles loading and saving images, managing directories, and listing files.
"""

from pathlib import Path
import os
import cv2
import numpy as np
import logging
from typing import List, Optional, Union

# Setup logger for fileio module
logger = logging.getLogger(__name__)

# Supported image extensions (case-insensitive)
SUPPORTED_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'
}

def ensure_dir(path: Union[str, Path]) -> bool:
    """
    Ensure that a directory exists, creating it if necessary.
    Handles potential errors during directory creation.
    
    Args:
        path: Path to the directory.
        
    Returns:
        True if the directory exists or was created successfully, False otherwise.
    """
    try:
        dir_path = Path(path)
        if not dir_path.exists():
            logger.info(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
        elif not dir_path.is_dir():
            logger.error(f"Path exists but is not a directory: {dir_path}")
            return False
        return True
    except PermissionError:
        logger.error(f"Permission denied creating directory: {path}")
        return False
    except Exception as e:
        logger.error(f"Error ensuring directory exists {path}: {e}", exc_info=True)
        return False

def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from the specified path using OpenCV.
    Performs basic validation on the path and loaded image.
    
    Args:
        path: Path to the image file.
        
    Returns:
        Loaded image as a NumPy array (BGR format).
        
    Raises:
        FileNotFoundError: If the path does not exist or is not a file.
        ValueError: If the file cannot be loaded as an image by OpenCV.
        Exception: For other unexpected errors.
    """
    try:
        image_path = Path(path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found or path is not a file: {image_path}")

        # Load image using OpenCV (reads as BGR by default)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        
        if img is None:
            # Check if the file extension is potentially unsupported but exists
            if image_path.exists(): 
                 raise ValueError(f"Failed to load image (OpenCV returned None). Unsupported format or corrupt file?: {image_path}")
            else:
                 # This case should be caught by is_file() check, but added for robustness
                 raise FileNotFoundError(f"Image file disappeared after check?: {image_path}")

        logger.debug(f"Image loaded successfully: {image_path} (shape: {img.shape})")
        return img
    
    except FileNotFoundError as fnf_err:
        logger.error(str(fnf_err))
        raise
    except ValueError as val_err:
        logger.error(str(val_err))
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading image {path}: {e}", exc_info=True)
        raise

def save_image(path: Union[str, Path], image: np.ndarray, quality: Optional[int] = 95) -> bool:
    """
    Save an image to the specified path using OpenCV.
    Creates the output directory if it doesn't exist.
    Allows specifying JPEG quality.
    
    Args:
        path: Output path for the image file.
        image: Image data as a NumPy array.
        quality: JPEG quality setting (0-100), applicable only for .jpg/.jpeg.
                 Higher means better quality and larger file size. Default: 95.
        
    Returns:
        True if saving was successful, False otherwise.
    """
    try:
        output_path = Path(path)
        # Ensure the parent directory exists
        if not ensure_dir(output_path.parent):
            logger.error(f"Cannot save image, failed to ensure output directory exists: {output_path.parent}")
            return False
        
        # Prepare OpenCV parameters based on file extension
        params = []
        ext = output_path.suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            logger.debug(f"Saving JPEG with quality={quality}")
        elif ext == '.png':
             # Example: Add PNG compression params if needed
             # params = [cv2.IMWRITE_PNG_COMPRESSION, 9] # 0-9, higher is more compression
             logger.debug("Saving PNG (default compression)")
             pass 
        elif ext in ['.tif', '.tiff']:
             logger.debug("Saving TIFF (default compression)")
             pass
        # Add params for other formats if necessary (e.g., WEBP)

        # Save the image
        success = cv2.imwrite(str(output_path), image, params)
        
        if success:
            logger.debug(f"Image saved successfully: {output_path}")
            return True
        else:
            logger.error(f"Failed to save image (OpenCV returned False): {output_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error saving image {path}: {e}", exc_info=True)
        return False

def list_image_files(directory: Union[str, Path]) -> List[Path]:
    """
    List all supported image files in a given directory (non-recursive).
    
    Args:
        directory: Path to the directory to scan.
        
    Returns:
        A list of Path objects for found image files.
        Returns an empty list if the directory doesn't exist or contains no images.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.warning(f"Directory not found or is not a directory: {dir_path}")
        return []
    
    image_files = []
    try:
        for item in dir_path.iterdir():
            # Check if it's a file and has a supported extension
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                image_files.append(item)
    except Exception as e:
        logger.error(f"Error listing files in directory {directory}: {e}", exc_info=True)
        return [] # Return empty list on error
        
    logger.debug(f"Found {len(image_files)} supported image file(s) in {dir_path}")
    return image_files
