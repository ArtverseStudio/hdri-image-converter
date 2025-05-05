"""
HDR Image Converter - Main Orchestrator
--------------------------------------
High-level pipeline for creating HDR images from single LDR inputs
using synthetic exposure stacking or single-image expansion.

Usage:
    python -m src.cli single input.jpg output.hdr
    python -m src.cli batch input_folder output_folder
"""

import sys
import numpy as np
import cv2
import warnings
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
import argparse
import time
import imageio # For EXR fallback

# Internal imports
from src.config import get_config, Config # Import Config type hint
from src.fileio import load_image, save_image, ensure_dir
from src.mask import create_luminance_mask, save_mask_debug
from src.tonemap import safe_tonemap, get_tonemap_operators # Keep for previews
from src.logger import setup_logger
from src.color import srgb_to_linear, linear_to_srgb, calculate_white_balance, apply_white_balance

# Check for OpenCV optional modules
try:
    import cv2.xphoto
    HAS_XPHOTO = True
except ImportError:
    HAS_XPHOTO = False

# --- Constants ---
PREVIEW_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
PREVIEW_LABEL_SCALE = 0.9 # Slightly smaller
PREVIEW_LABEL_THICKNESS = 2
PREVIEW_LABEL_COLOR = (255, 255, 255)
PREVIEW_MARGIN = 15
PREVIEW_LABEL_Y_POS = 30
PREVIEW_TOP_MARGIN = 40

# --- Logger ---
logger = setup_logger("hdr_converter")

# --- Helper Functions ---

def get_output_dir(output_path: Path) -> Path:
    """Determine output directory, ensuring 'output' subdirectory exists only at the top level relative to output_path's parent or CWD."""
    # Always use the top-level 'output' directory relative to the project root or CWD
    project_root = Path.cwd()
    output_dir = project_root / 'output'
    ensure_dir(output_dir)
    return output_dir

# --- Core Processing Steps ---

def load_and_preprocess(input_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load LDR (sRGB) image, convert to linear RGB float32."""
    preprocess_start = time.perf_counter()
    logger.info(f"Loading and preprocessing: {input_path}")
    try:
        img_srgb_uint8 = load_image(input_path)
        img_srgb_float = img_srgb_uint8.astype(np.float32) / 255.0
        img_linear_float = srgb_to_linear(img_srgb_float)
        logger.debug(f"Load & Preprocess took {time.perf_counter() - preprocess_start:.3f}s")
        return img_srgb_uint8, img_linear_float
    except FileNotFoundError:
        logger.error(f"Input image file not found: {input_path}")
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing {input_path}: {e}", exc_info=True)
        raise

# --- Single-image Pseudo-HDR expansion ---
def expand_single_image(img_linear: np.ndarray, config: Config) -> np.ndarray:
    """Expand single linear LDR image to pseudo-HDR based on percentile luminance."""
    expand_start = time.perf_counter()
    logger.info("Expanding single image to pseudo-HDR (linear)...")
    max_hdr_value = getattr(config, 'HDR_MAX_VALUE', 6.0)
    try:
        L = 0.2126 * img_linear[..., 2] + 0.7152 * img_linear[..., 1] + 0.0722 * img_linear[..., 0]
        L = np.maximum(L, 1e-9) # Avoid issues with pure black images
        L_white = np.percentile(L, 99)
        logger.debug(f"Single image expansion: Estimated white point L_white={L_white:.4f}, Target max={max_hdr_value}")
        scale_factor = max_hdr_value / (L_white + 1e-9)
        img_expanded = np.clip(img_linear * scale_factor, 0, max_hdr_value) # Clip to target max
        logger.debug(f"Single image expansion took {time.perf_counter() - expand_start:.3f}s")
        return img_expanded.astype(np.float32)
    except Exception as e:
        logger.error(f"Error during single image expansion: {e}", exc_info=True)
        raise

# --- Exposure Stack Generation ---
def generate_exposure_stack(img_linear_float: np.ndarray, config: Config, output_dir: Path, output_base_name: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Generate synthetic exposure stack (sRGB uint8) with highlight blending."""
    stack_start = time.perf_counter()
    logger.info("Generating synthetic exposure stack (sRGB uint8)...")
    try:
        images_srgb = []
        exposure_times = []

        # --- Luminance Mask Creation (from original image, always) ---
        mask_start = time.perf_counter()
        # Use the original linear image for mask generation
        luminance = 0.2126 * img_linear_float[..., 2] + 0.7152 * img_linear_float[..., 1] + 0.0722 * img_linear_float[..., 0]
        luminance_clipped = np.clip(luminance, 0, 1) # Clip before uint8 conversion

        luminance_percentile = getattr(config, 'LUMINANCE_PERCENTILE', 95)
        feather_radius = getattr(config, 'FEATHER_RADIUS', 31) | 1
        logger.debug(f"Creating luminance mask (percentile={luminance_percentile}, feather={feather_radius})...")
        # create_luminance_mask expects 8-bit input
        light_mask = create_luminance_mask((luminance_clipped * 255).astype(np.uint8), luminance_percentile, feather_radius)
        light_mask_3c = np.repeat(light_mask[..., np.newaxis], 3, axis=2).astype(np.float32) # Float mask [0,1] for blending
        logger.debug(f"Mask creation took {time.perf_counter() - mask_start:.3f}s")

        save_stops = getattr(config, 'SAVE_EXPOSURE_STOPS', False) # Default False now
        if save_stops:
            mask_path = output_dir / f'{output_base_name}_luminance_mask.jpg'
            try:
                save_mask_debug(linear_to_srgb(img_linear_float), light_mask, filename=mask_path) # Save viz using sRGB base
            except Exception as e:
                 logger.warning(f"Could not save luminance mask debug image: {e}")

        # --- Exposure Simulation Loop ---
        sim_loop_start = time.perf_counter()
        base_blend_ratio = getattr(config, 'BLEND_RATIO', 0.6) # Get base ratio from config
        gamma = getattr(config, 'GAMMA', 2.2)
        exposure_values = config.EXPOSURE_VALUES # Use property
        logger.debug(f"Simulating EV stops: {exposure_values} (Gamma={gamma}, BlendRatio={base_blend_ratio:.2f}, SaveStops={save_stops})")
        
        # --- No Fading Blend - Use Constant Base Ratio ---
        # We removed the fading logic. Blending will use the base_blend_ratio for all negative stops.
        # This allows the simulated stops to contribute more brightness to highlights.
        # logger.debug("Using constant blend ratio for negative EVs.")
        
        for ev in exposure_values:
            exposure_time = 2.0 ** ev
            exposed_linear = img_linear_float * exposure_time # Apply EV scale

            # --- Mask Blending for Highlight Preservation ---
            # Blend original linear image into highlights for underexposed stops.
            # Uses a constant blend ratio defined in config.
            if ev < 0:
                current_blend_ratio = base_blend_ratio # Use the base ratio directly
                logger.debug(f"  EV={ev:<+5.1f}: Applying blend ratio = {current_blend_ratio:.3f}")
                
                # Apply blending
                exposed_linear = (1.0 - light_mask_3c * current_blend_ratio) * exposed_linear + \
                                 (light_mask_3c * current_blend_ratio) * img_linear_float

            # Clip linear result to [0, 1] before gamma correction
            exposed_linear_clipped = np.clip(exposed_linear, 0, 1)

            # Approximate gamma correction (linear to sRGB)
            exposed_srgb_approx = np.power(exposed_linear_clipped, 1.0 / gamma)

            # Convert to 8-bit for saving / OpenCV merging
            img_ev_srgb_uint8 = (exposed_srgb_approx * 255).astype(np.uint8)

            images_srgb.append(img_ev_srgb_uint8)
            exposure_times.append(exposure_time)

            if save_stops:
                ev_str = f"{ev:+.1f}".replace('.', 'p').replace('-', 'm')
                exposure_path = output_dir / f"{output_base_name}_exposure_EV{ev_str}.jpg"
                try:
                    # Consider saving in a lossless format like PNG if performance allows? JPG is fast but lossy.
                    save_image(exposure_path, img_ev_srgb_uint8)
                except Exception as e:
                    logger.warning(f"Could not save exposure stop {exposure_path}: {e}")
        
        logger.debug(f"Exposure simulation loop took {time.perf_counter() - sim_loop_start:.3f}s")
        logger.info(f"Exposure stack generation finished ({len(images_srgb)} images). Took {time.perf_counter() - stack_start:.3f}s")
        return images_srgb, np.array(exposure_times, dtype=np.float32)

    except Exception as e:
        logger.error(f"Error generating exposure stack: {e}", exc_info=True)
        raise

def merge_exposures(images_srgb: List[np.ndarray], exposure_times: np.ndarray, config: Config) -> np.ndarray:
    """Merge exposure stack (sRGB uint8) to linear HDR float32 using Debevec."""
    merge_start = time.perf_counter()
    logger.info(f"Merging {len(images_srgb)} exposures to linear HDR...")
    if len(images_srgb) != len(exposure_times) or len(images_srgb) < 2:
        raise ValueError("Need >= 2 images and matching exposure times to merge.")
    try:
        # TODO: Add config option to use different merge algorithms (Mertens, Robertson?)
        merge_debevec = cv2.createMergeDebevec()
        
        # Note: Explicit CRF calibration might improve results but adds time.
        # calibrate_debevec = cv2.createCalibrateDebevec(samples=config.CALIBRATION_SAMPLES, lambda_=config.CALIBRATION_LAMBDA)
        # crf = calibrate_debevec.process(images_srgb, exposure_times.copy())
        # merged_hdr = merge_debevec.process(images_srgb, exposure_times.copy(), crf)
        
        merged_hdr = merge_debevec.process(images_srgb, exposure_times.copy())

        # Clamp negative values which can sometimes occur
        merged_hdr = np.maximum(merged_hdr, 0)
        logger.debug(f"Exposure merging took {time.perf_counter() - merge_start:.3f}s")
        return merged_hdr.astype(np.float32)
    except cv2.error as e:
        logger.error(f"OpenCV error during merging: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error merging exposures: {e}", exc_info=True)
        raise

def scale_hdr_by_key(hdr_linear: np.ndarray, config: Config) -> np.ndarray:
    """Scale linear HDR based on log-average luminance (scene key). Output remains linear HDR (values can be > 1)."""
    norm_start = time.perf_counter()
    logger.info("Scaling HDR image based on log-average luminance (scene key)...")
    if hdr_linear.min() < 0:
         logger.warning(f"Input HDR has negative values (min={hdr_linear.min():.4f}). Clamping to 0 before scaling.")
         hdr_linear = np.maximum(hdr_linear, 0)
    try:
        logger.info(f"HDR stats before scaling: min={hdr_linear.min():.4f}, max={hdr_linear.max():.4f}, mean={hdr_linear.mean():.4f}")
        luminance = 0.2126 * hdr_linear[..., 2] + 0.7152 * hdr_linear[..., 1] + 0.0722 * hdr_linear[..., 0]
        luminance_log = np.log(np.maximum(luminance, 1e-9)) # Avoid log(0)
        log_average_luminance = np.exp(np.mean(luminance_log))

        key_value = getattr(config, 'NORMALIZATION_KEY', 0.03) # Target key
        logger.info(f"Log-average luminance (scene key input): {log_average_luminance:.4f}")
        logger.info(f"Target normalization key value: {key_value:.4f}")

        scale_factor = key_value / (log_average_luminance + 1e-9) # Avoid division by zero
        logger.debug(f"HDR scaling factor: {scale_factor:.4f}")
        hdr_scaled = hdr_linear * scale_factor

        # DO NOT CLAMP here - output should remain linear HDR
        logger.info(f"HDR stats after scaling: min={hdr_scaled.min():.4f}, max={hdr_scaled.max():.4f}, mean={hdr_scaled.mean():.4f}")
        # Optional histogram of scaled values (log scale might be useful)
        # hist, bins = np.histogram(np.log10(np.maximum(hdr_scaled, 1e-9)).flatten(), bins=50)

        logger.debug(f"HDR scaling took {time.perf_counter() - norm_start:.3f}s")
        return hdr_scaled.astype(np.float32)
    except Exception as e:
        logger.error(f"Error in HDR scaling: {e}", exc_info=True)
        raise

def apply_white_balance_correction(hdr_linear: np.ndarray, config: Config) -> np.ndarray:
    """Apply white balance correction to the linear HDR image."""
    wb_start = time.perf_counter()
    logger.info("Applying color correction (white balance)...")
    method = getattr(config, 'WHITE_BALANCE_METHOD', 'gray_world')
    logger.debug(f"Using white balance method: {method}")
    hdr_float = hdr_linear.astype(np.float32) # Ensure float32 input
    try:
        corrected_hdr = hdr_float
        if method == 'simple' and HAS_XPHOTO and hasattr(cv2.xphoto, 'createSimpleWB'):
            wb = cv2.xphoto.createSimpleWB()
            corrected_hdr = wb.balanceWhite(hdr_float)
        elif method == 'gray_world':
            wb_factors = calculate_white_balance(hdr_float, method='gray_world')
            corrected_hdr = apply_white_balance(hdr_float, wb_factors)
        else:
             logger.warning(f"White balance method '{method}' not recognized or unavailable. Skipping.")

        corrected_hdr = np.maximum(corrected_hdr, 0) # Ensure non-negative
        logger.debug(f"White balance correction took {time.perf_counter() - wb_start:.3f}s")
        return corrected_hdr.astype(np.float32)
    except Exception as e:
        logger.error(f"Error applying white balance: {e}", exc_info=True)
        return hdr_float # Return original on error

def save_hdr_output(hdr_linear: np.ndarray, output_dir: Path, output_base_name: str) -> Optional[Path]:
    """Save the final linear HDR image to .hdr and optionally .exr formats."""
    save_start = time.perf_counter()
    logger.info("Saving final HDR output file(s)...")
    hdr_path, exr_path = None, None
    hdr_to_save = hdr_linear.astype(np.float32)

    # Save .hdr (Radiance RGBE) - Primary Output
    try:
        hdr_path_target = output_dir / f"{output_base_name}.hdr"
        cv2.imwrite(str(hdr_path_target), hdr_to_save)
        hdr_path = hdr_path_target
        logger.info(f"Saved Radiance HDR: {hdr_path}")
    except Exception as e:
        logger.error(f"Failed to save .hdr file ({hdr_path_target}): {e}", exc_info=True)
        hdr_path = None # Ensure path is None if failed

    # Save .exr (OpenEXR) - Optional Fallback/Alternative
    exr_path_target = output_dir / f"{output_base_name}.exr"
    try:
        cv2.imwrite(str(exr_path_target), hdr_to_save)
        exr_path = exr_path_target
        logger.info(f"Saved OpenEXR (via OpenCV): {exr_path}")
    except cv2.error as cv_exr_err:
        logger.warning(f"OpenCV EXR save failed ({cv_exr_err}). Trying imageio...")
        try:
            imageio.imwrite(str(exr_path_target), hdr_to_save, format='EXR-FI')
            exr_path = exr_path_target
            logger.info(f"Saved OpenEXR (via imageio): {exr_path}")
        except ImportError:
             logger.warning("imageio or EXR plugin not found, cannot save .exr.")
        except Exception as e2:
            logger.warning(f"imageio EXR save failed ({exr_path_target}): {e2}")
    except Exception as e:
         logger.error(f"Unexpected error saving .exr file ({exr_path_target}): {e}", exc_info=True)

    logger.debug(f"Saving HDR outputs took {time.perf_counter() - save_start:.3f}s")
    return hdr_path # Return path to the primary (.hdr) output

def generate_previews(hdr_scaled_linear: np.ndarray, img_srgb_uint8: np.ndarray, output_dir: Path, output_base_name: str, config: Config, tone_mapper_override: Optional[str]) -> None:
    """Generate and save Original and Tone Mapped LDR previews."""
    preview_start = time.perf_counter()
    logger.info("Generating LDR preview images...")
    preview_paths = {'original': None, 'dr': None, 'combined': None}
    try:
        # --- Original sRGB Preview ---
        # Removed saving of output_preview_original.jpg as per user request
        # try:
        #     path = output_dir / f"{output_base_name}_preview_original.jpg"
        #     save_image(path, img_srgb_uint8)
        #     preview_paths['original'] = path
        #     logger.debug(f"Saved original preview: {path}")
        # except Exception as e: logger.error(f"Failed to save original preview: {e}")

        # --- Dynamic Range Visualization (REMOVED) ---
        # try:
        #     path = output_dir / f"{output_base_name}_preview_dynamic_range.jpg"
        #     luminance = 0.2126 * hdr_scaled_linear[..., 2] + 0.7152 * hdr_scaled_linear[..., 1] + 0.0722 * hdr_scaled_linear[..., 0]
        #     log_luminance = np.log10(np.maximum(luminance, 1e-9))
        #     range_min, range_max = np.min(log_luminance), np.max(log_luminance)
        #     dyn_range_norm = np.clip((log_luminance - range_min) / (range_max - range_min + 1e-9), 0, 1)
        #     dyn_range_color = cv2.applyColorMap((dyn_range_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        #     save_image(path, dyn_range_color)
        #     preview_paths['dr'] = path
        #     logger.debug(f"Saved dynamic range preview (Log10 Lum range [{range_min:.2f}, {range_max:.2f}]): {path}")
        # except Exception as e: logger.error(f"Failed to create/save dynamic range preview: {e}")
        logger.info("Skipping dynamic range preview generation.") # Log skipping

        # --- Tone Mapped LDR Previews ---
        logger.info("Generating tone mapped LDR previews...")
        tm_start = time.perf_counter()
        preview_mapper_setting = tone_mapper_override if tone_mapper_override else getattr(config, 'PREVIEW_TONEMAPPER', 'Reinhard')
        preview_scale = getattr(config, 'PREVIEW_SCALE', 1.0) # Extra scaling only for previews
        logger.info(f"Tone mapper(s) for preview: '{preview_mapper_setting}', Preview scale: {preview_scale}")

        # Apply extra scaling factor specifically for tone mapping previews
        hdr_for_tm_preview = hdr_scaled_linear * preview_scale

        available_tonemaps = get_tonemap_operators(config) # Get operators with config params
        tonemaps_to_run = {}
        if preview_mapper_setting and preview_mapper_setting.lower() == 'all':
            tonemaps_to_run = available_tonemaps
        elif preview_mapper_setting and preview_mapper_setting in available_tonemaps:
            tonemaps_to_run = {preview_mapper_setting: available_tonemaps[preview_mapper_setting]}
        elif preview_mapper_setting:
             logger.warning(f"Preview tone mapper '{preview_mapper_setting}' not recognized. Available: {list(available_tonemaps.keys())}")

        ldr_results = {}
        if tonemaps_to_run:
            logger.debug(f"Applying tone mapper(s): {list(tonemaps_to_run.keys())}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for name, op in tonemaps_to_run.items():
                    try:
                        # Input to tonemap is scaled linear HDR
                        ldr_linear = safe_tonemap(hdr_for_tm_preview.copy(), op)
                        # Convert result to sRGB uint8 for preview display
                        ldr_srgb = linear_to_srgb(ldr_linear)
                        ldr_results[name] = np.clip(ldr_srgb * 255, 0, 255).astype(np.uint8)
                    except Exception as e:
                        logger.error(f"Tone mapping preview with {name} failed: {e}")
            logger.debug(f"Tone mapping operators took {time.perf_counter() - tm_start:.3f}s")
        else:
            logger.warning("No tone mapping operators selected for combined preview.")

        # --- Combined Preview Image ---
        if ldr_results:
            try:
                comb_start = time.perf_counter()
                h, w = next(iter(ldr_results.values())).shape[:2]
                orig_resized = cv2.resize(img_srgb_uint8, (w, h), interpolation=cv2.INTER_AREA)
                num_images = len(ldr_results) + 1
                combined_w = w * num_images + PREVIEW_MARGIN * (num_images - 1)
                combined_h = h + PREVIEW_TOP_MARGIN
                combined_img = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

                # Add Original
                combined_img[PREVIEW_TOP_MARGIN:, 0:w] = orig_resized
                cv2.putText(combined_img, "Original (sRGB)", (int(w*0.1), PREVIEW_LABEL_Y_POS), PREVIEW_LABEL_FONT, PREVIEW_LABEL_SCALE, PREVIEW_LABEL_COLOR, PREVIEW_LABEL_THICKNESS, cv2.LINE_AA)

                # Add Tone Mapped Results
                current_x = w + PREVIEW_MARGIN
                preferred_order = ['Reinhard']
                ordered_names = [n for n in preferred_order if n in ldr_results] + [n for n in ldr_results if n not in preferred_order]
                for name in ordered_names:
                    ldr_img = ldr_results[name]
                    combined_img[PREVIEW_TOP_MARGIN:, current_x:current_x + w] = ldr_img
                    label = f"{name} (TM)"
                    cv2.putText(combined_img, label, (current_x + int(w*0.1), PREVIEW_LABEL_Y_POS), PREVIEW_LABEL_FONT, PREVIEW_LABEL_SCALE, PREVIEW_LABEL_COLOR, PREVIEW_LABEL_THICKNESS, cv2.LINE_AA)
                    current_x += w + PREVIEW_MARGIN

                path = output_dir / f"{output_base_name}_preview_combined_LDR.jpg"
                save_image(path, combined_img)
                preview_paths['combined'] = path
                logger.info(f"Saved combined LDR preview: {path}")
                logger.debug(f"Combined preview creation took {time.perf_counter() - comb_start:.3f}s")
            except Exception as e: logger.error(f"Failed to create/save combined preview: {e}")

        logger.info(f"Preview generation finished. Took {time.perf_counter() - preview_start:.3f}s")
        # Log paths that were successfully created
        # if preview_paths['original']: logger.info(f"  Preview (Original): {preview_paths['original']}")
        # if preview_paths['dr']: logger.info(f"  Preview (Dyn Range): {preview_paths['dr']}") # Commented out log
        if preview_paths['combined']: logger.info(f"  Preview (Combined LDR): {preview_paths['combined']}")

    except Exception as e:
        logger.error(f"Error generating previews: {e}", exc_info=True)

# --- Main HDR Creation Workflow ---

def create_hdr_pipeline(input_path: Union[str, Path], output_path: Union[str, Path], config: Config, tone_mapper_override: Optional[str] = None) -> Optional[Path]:
    """Main pipeline orchestrator for HDR image creation."""
    overall_start_time = time.perf_counter()
    logger.info("=" * 15 + " Starting HDR Conversion Pipeline " + "=" * 15)
    final_hdr_path: Optional[Path] = None
    try:
        input_path = Path(input_path).resolve()
        output_path = Path(output_path) # Base path for naming outputs
        output_dir = get_output_dir(output_path)
        output_base_name = output_path.stem
        logger.info(f"Input image: {input_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Output base name: {output_base_name}")

        # --- Step 1: Load and Preprocess ---
        img_srgb_uint8, img_linear_float = load_and_preprocess(input_path)

        # --- Step 2 & 3: Generate/Expand to Linear HDR ---
        hdri_mode = getattr(config, 'HDRI_MODE', 'stack')
        hdr_linear_unscaled: Optional[np.ndarray] = None

        if hdri_mode == 'single':
            hdr_linear_unscaled = expand_single_image(img_linear_float, config)
        elif hdri_mode == 'stack':
            images_srgb, exposure_times = generate_exposure_stack(img_linear_float, config, output_dir, output_base_name)
            hdr_linear_unscaled = merge_exposures(images_srgb, exposure_times, config)
        else:
             raise ValueError(f"Invalid HDRI_MODE: {hdri_mode}. Must be 'stack' or 'single'.")

        if hdr_linear_unscaled is None:
            raise RuntimeError("HDR image generation/expansion failed.")

        # --- Step 4: Scale Linear HDR based on Scene Key ---
        hdr_scaled_linear = scale_hdr_by_key(hdr_linear_unscaled, config)

        # --- Step 5: White Balance Correction ---
        hdr_corrected_linear = apply_white_balance_correction(hdr_scaled_linear, config)

        # --- Step 6: Save Final Linear HDR Output ---
        final_hdr_path = save_hdr_output(hdr_corrected_linear, output_dir, output_base_name)

        # --- Step 7: Generate LDR Previews ---
        # Previews use the final corrected & scaled linear HDR
        generate_previews(hdr_corrected_linear, img_srgb_uint8, output_dir, output_base_name, config, tone_mapper_override)

        # --- Completion ---
        total_time = time.perf_counter() - overall_start_time
        logger.info("=" * 15 + " HDR Conversion Pipeline Finished " + "=" * 15)
        logger.info(f"Total processing time: {total_time:.2f} seconds.")
        if final_hdr_path:
            logger.info(f"Successfully generated HDR: {final_hdr_path}")
        else:
            logger.error("Failed to save the final .hdr output file!")
        logger.info("="*60)
        return final_hdr_path

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Pipeline failed: {e}", exc_info=False) # Log known errors concisely
        logger.info("="*60)
        return None
    except Exception as e:
        logger.error(f"Unhandled pipeline error: {e}", exc_info=True) # Log unexpected errors with traceback
        logger.info("="*60)
        return None

# --- Command Line Interface ---
def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Create a linear HDR image (.hdr) from a single LDR input image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_image", help="Path to the input LDR image (e.g., input.jpg).")
    parser.add_argument("output_hdr", help="Desired path for the output HDR file (e.g., output/result.hdr). Output files will be placed in an 'output' subdirectory relative to this path's parent.")
    parser.add_argument("--data_dir", default=None, help="Optional base directory for relative input image paths.")
    parser.add_argument("--tone_mapper", dest='tone_mapper_override', default=None,
                        choices=['Reinhard', 'Drago', 'Mantiuk', 'all'],
                       help="Tone mapping operator for combined LDR preview (default: uses config setting). 'all' generates all available.")
    parser.add_argument("--config", default=None, help="Path to custom JSON configuration file.")

    args = parser.parse_args()

    try:
        # Load configuration
        config = get_config(args.config) # Handles defaults and file loading

        # Resolve input path
        input_path = Path(args.input_image)
        if not input_path.is_absolute() and args.data_dir:
            input_path = Path(args.data_dir).resolve() / input_path
        input_path = input_path.resolve() # Ensure absolute path

        # Define output base path (directory determined within pipeline)
        output_path = Path(args.output_hdr)

        # Run pipeline
        result_path = create_hdr_pipeline(input_path, output_path, config, args.tone_mapper_override)

        if result_path:
            logger.info(f"Pipeline completed successfully. Output: {result_path}")
            sys.exit(0) # Success
        else:
            logger.error("Pipeline failed.")
            sys.exit(1) # Failure

    except Exception as e:
        # Catch-all for setup errors before pipeline starts
        logger.error(f"Critical error during setup or execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()