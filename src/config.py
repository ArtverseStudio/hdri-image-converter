"""
Config Module
-------------
Centralizes all configuration for the HDR pipeline.
Supports file-based config override and validation.

Usage:
    from src.config import get_config
    config = get_config()
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union, Set
from pathlib import Path
import os
import numpy as np
import json
import logging
from src.logger import setup_logger

# Configure logger
logger = setup_logger("config")

def _validate_config(cfg: 'Config') -> List[str]:
    """
    Validate configuration values are within acceptable ranges.
    
    Args:
        cfg: Configuration object to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Validate values
    if not (0 < cfg.LUMINANCE_PERCENTILE <= 100):
        errors.append("LUMINANCE_PERCENTILE must be in range (0, 100]")
    
    if not (cfg.FEATHER_RADIUS >= 0):
        errors.append("FEATHER_RADIUS must be >= 0")
    
    if not (0 < cfg.BLEND_RATIO <= 1):
        errors.append("BLEND_RATIO must be in range (0, 1]")
    
    if not (1.0 <= cfg.GAMMA <= 3.0):
        errors.append("GAMMA must be in range [1.0, 3.0]")
    
    if hasattr(cfg, 'HDR_MAX_VALUE') and not (1.0 <= cfg.HDR_MAX_VALUE <= 10.0):
        errors.append("HDR_MAX_VALUE must be in range [1.0, 10.0]")
    
    if hasattr(cfg, 'PREVIEW_SCALE') and not (0.0 < cfg.PREVIEW_SCALE <= 5.0):
        errors.append("PREVIEW_SCALE must be in range (0, 5.0]")
    
    # Check for directories
    try:
        data_dir = cfg.data_dir_path
        if not data_dir.exists():
            logger.warning(f"DATA_DIR does not exist: {data_dir}")
    except Exception as e:
        errors.append(f"Error resolving DATA_DIR: {e}")

    try:
        output_dir = cfg.output_dir_path
        if not output_dir.exists():
            logger.warning(f"OUTPUT_DIR does not exist: {output_dir}")
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created OUTPUT_DIR: {output_dir}")
            except Exception as e:
                errors.append(f"Failed to create OUTPUT_DIR: {e}")
    except Exception as e:
        errors.append(f"Error resolving OUTPUT_DIR: {e}")

    # Return all validation errors
    return errors

@dataclass(frozen=True)
class Config:
    """
    Configuration parameters for the HDR image processing pipeline.
    
    Attributes:
        # --- Masking (for highlight control in exposure stack) ---
        LUMINANCE_PERCENTILE: Percentile used to identify bright image areas for the mask (0-100).
        FEATHER_RADIUS: Gaussian blur radius for softening mask edges (>=0, should be odd).
        BLEND_RATIO: Base factor controlling how much the original image is blended into the EV=-1 stop using the luminance mask (0-1). Blend increases for lower EVs.

        # --- Exposure Stack Simulation ---
        CUSTOM_EXPOSURE_VALUES: Optional list of specific EV stops to simulate. Overrides range if set.
        EXPOSURE_START: Starting EV stop for simulated exposure range.
        EXPOSURE_END: Ending EV stop for simulated exposure range.
        EXPOSURE_STEP: Step size (in EV stops) for simulated exposure range.
        EXPOSURE_VALUES: Computed list of EV stops based on range or custom list.
        GAMMA: Assumed display/camera gamma exponent used for converting linear to simulated sRGB exposure stops (e.g., 2.2).
        SAVE_EXPOSURE_STOPS: Whether to save the individual simulated sRGB exposure stop images to the output directory.
        ADAPTIVE_EXPOSURE: (Currently unused) Intended flag for content-based exposure adjustment.

        # --- HDR Merging (using OpenCV Debevec) ---
        CALIBRATION_SAMPLES: Number of pixel samples used by OpenCV for camera response function calibration.
        CALIBRATION_LAMBDA: Smoothness parameter for the camera response function estimation.
        CACHE_RESPONSE_CURVE: (Currently unused) Intended flag to cache the estimated response curve.

        # --- HDR Processing & Normalization ---
        HDRI_MODE: 'stack' to generate HDR via exposure stack merging, 'single' for single-image pseudo-HDR expansion.
        HDR_MAX_VALUE: Maximum plausible value for single-image expansion or post-merge adjustments (>=1.0).
        NORMALIZATION_KEY: Target value (scene key) for the scene's log-average luminance after normalization (0-1). Lower values result in a darker, less clipped image.

        # --- Preview Generation ---
        PREVIEW_SCALE: Additional scaling factor applied *only* to the HDR image before generating tone-mapped previews.
        PREVIEW_TONEMAPPER: Which tone mapping operator(s) to use for the combined preview: 'Reinhard', 'Drago', 'Mantiuk', 'all', or specific name. Limits computation if not 'all'.
        TONEMAP_PARAMS: Dictionary of parameters for specific OpenCV tone mapping operators.
        
        # --- File I/O ---
        DATA_DIR: Default base directory for input images.
        OUTPUT_DIR: Default base directory for output results.

        # --- (Currently Unused/Placeholder) ---
        ADAPTIVE_MASK: Intended flag for adaptive thresholding in mask generation.
        WHITE_BALANCE_METHOD: Method for white balance correction.
    """
    
    # --- Masking (for highlight control in exposure stack) ---
    LUMINANCE_PERCENTILE: float = 98
    FEATHER_RADIUS: int = 5
    BLEND_RATIO: float = 0.4

    # --- Exposure Stack Simulation ---
    CUSTOM_EXPOSURE_VALUES: list = field(default_factory=list) # Use `list` instead of `lambda: []`
    EXPOSURE_START: float = -2.0
    EXPOSURE_END: float = 2.0
    EXPOSURE_STEP: float = 2.0
    
    @property
    def EXPOSURE_VALUES(self) -> List[float]:
        """Compute list of exposure values (EV stops) based on config."""
        if self.CUSTOM_EXPOSURE_VALUES:
            # Ensure custom values are floats
            return [float(ev) for ev in self.CUSTOM_EXPOSURE_VALUES]
        # Generate range, ensuring endpoint is included
        return [float(f'{ev:.2f}') for ev in np.arange(
            self.EXPOSURE_START, 
            self.EXPOSURE_END + self.EXPOSURE_STEP / 2, # Add half step to include endpoint reliably
            self.EXPOSURE_STEP
        )]
    
    GAMMA: float = 2.2                # Assumed display/camera gamma for linear -> sRGB conversion
    SAVE_EXPOSURE_STOPS: bool = True   # Save individual simulated sRGB stops (Default: True for debugging)
    ADAPTIVE_EXPOSURE: bool = False    # (Unused currently)

    # --- HDR Merging (using OpenCV Debevec) ---
    CALIBRATION_SAMPLES: int = 100    # Pixel samples for CRF calibration
    CALIBRATION_LAMBDA: float = 2.0   # CRF smoothness parameter
    CACHE_RESPONSE_CURVE: bool = False # (Unused currently)

    # --- HDR Processing & Normalization ---
    HDRI_MODE: str = "single"
    HDR_MAX_VALUE: float = 6.0        # Max value for single-image expansion/adjustments (>=1.0)
    NORMALIZATION_KEY: float = 0.15   # Slightly lower for less bright HDR

    # --- Preview Generation ---
    PREVIEW_SCALE: float = 0.8        # No extra scaling for previews
    PREVIEW_TONEMAPPER: str = "Reinhard"
    TONEMAP_PARAMS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'Reinhard': {
            'gamma': 1.0, 'intensity': 0.0, 'light_adapt': 0.0, 'color_adapt': 0.0
        }
    })
    
    # --- File I/O ---
    # Default locations relative to project root (assuming config.py is in src/)
    DATA_DIR: str = str(Path(__file__).resolve().parent.parent / "input")
    OUTPUT_DIR: str = str(Path(__file__).resolve().parent.parent / "output")

    # --- (Currently Unused/Placeholder) ---
    ADAPTIVE_MASK: bool = True
    WHITE_BALANCE_METHOD: str = "gray_world"
    
    @property
    def data_dir_path(self) -> Path:
        """Return the data directory as a Path object."""
        return Path(self.DATA_DIR).resolve()
    
    @property
    def output_dir_path(self) -> Path:
        """Return the output directory as a Path object."""
        return Path(self.OUTPUT_DIR).resolve()

def _load_config_from_file(path: Union[str, Path, None]) -> Optional[Dict[str, Any]]:
    """
    Load configuration from a JSON file.
    
    Args:
        path: Path to JSON config file
        
    Returns:
        Dictionary of config values or None if loading failed
    """
    if not path:
        return None
    
    try:
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return None
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return data
    
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in config file: {path}")
        return None
    except Exception as e:
        logger.error(f"Error loading config from {path}: {str(e)}")
        return None

# Global config instance
_config: Optional[Config] = None

def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Get the global configuration object, optionally loading from a file.
    
    Args:
        config_path: Optional path to JSON config file
        
    Returns:
        Configuration object with all parameters
    """
    global _config
    
    # If config_path provided, always reload
    if config_path is not None:
        # Load from file
        config_data = _load_config_from_file(config_path)
        
        if config_data:
            # Filter out unknown fields
            base = asdict(Config())
            filtered_data = {k: v for k, v in config_data.items() if k in base}
            
            # Allow HDRI_MODE from config file
            if 'HDRI_MODE' in config_data:
                filtered_data['HDRI_MODE'] = config_data['HDRI_MODE']
            
            # Create new config
            try:
                new_config = Config(**filtered_data)
                
                # Validate config
                errors = _validate_config(new_config)
                if errors:
                    for err in errors:
                        logger.error(f"Config validation error: {err}")
                    logger.warning("Falling back to default configuration")
                    _config = Config()
                else:
                    _config = new_config
                    logger.info("Configuration successfully loaded and validated")
            except Exception as e:
                logger.error(f"Error creating config from file data: {e}")
                logger.warning("Falling back to default configuration")
                _config = Config()
        else:
            # Use default config
            _config = Config()
            
        # Validate even default config
        errors = _validate_config(_config)
        if errors:
            for err in errors:
                logger.error(f"Default config validation error: {err}")
    
    # If no _config exists, create default
    if _config is None:
        _config = Config()
        errors = _validate_config(_config)
        if errors:
            for err in errors:
                logger.error(f"Default config validation error: {err}")
    
    return _config

def save_config(config: Config, output_path: Union[str, Path]) -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration object to save
        output_path: Path to save the config to
        
    Returns:
        True if saving succeeded, False otherwise
    """
    try:
        # Convert config to dict
        config_dict = asdict(config)
        
        # Remove computed property
        if 'EXPOSURE_VALUES' in config_dict:
            del config_dict['EXPOSURE_VALUES']
        
        # Ensure directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON file
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
            
        logger.info(f"Configuration saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        return False
