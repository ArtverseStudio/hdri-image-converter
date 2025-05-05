"""
Command-Line Interface (CLI) Module
------------------------------------
Handles command-line argument parsing and orchestrates the HDR conversion 
process for single images or batch processing using the main pipeline.
"""

import argparse
import sys
from pathlib import Path
import os
import time
from typing import Optional

# Ensure src directory is in path for relative imports
# This might be needed if running the script directly
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logger import setup_logger
from src.config import get_config
from src.hdr_converter import create_hdr_pipeline # Import the main pipeline function

# Setup logger for the CLI module
logger = setup_logger("cli")

def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for single or batch processing."""
    parser = argparse.ArgumentParser(
        description="Convert LDR image(s) to HDR format (.hdr).",
        formatter_class=argparse.RawDescriptionHelpFormatter, # Keep formatting
        epilog='''Examples:
  Single image:
    python -m src.cli single input/my_image.jpg output/my_hdr.hdr
  Batch processing:
    python -m src.cli batch input_folder output_folder --config custom_config.json
'''
    )

    subparsers = parser.add_subparsers(dest='mode', required=True, 
                                       help='Processing mode: \'single\' or \'batch\'')

    # --- Single Image Mode --- 
    parser_single = subparsers.add_parser('single', help='Process a single image.',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_single.add_argument("input_image", type=str, 
                             help="Path to the input LDR image.")
    parser_single.add_argument("output_hdr", type=str, 
                             help="Desired output path for the .hdr file (e.g., output/result.hdr).")
    parser_single.add_argument("--config", type=str, default=None, 
                             help="Path to a custom JSON configuration file.")
    parser_single.add_argument("--tone-mapper", dest='tone_mapper_override', default=None,
                             choices=['Reinhard', 'Drago', 'Mantiuk', 'all'],
                             help="Override tone mapper for combined LDR preview (default: use config).")

    # --- Batch Mode --- 
    parser_batch = subparsers.add_parser('batch', help='Process all images in a directory.',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_batch.add_argument("input_dir", type=str, 
                            help="Directory containing input LDR images.")
    parser_batch.add_argument("output_dir", type=str, 
                            help="Directory where HDR outputs will be saved.")
    parser_batch.add_argument("--config", type=str, default=None, 
                            help="Path to a custom JSON configuration file.")
    parser_batch.add_argument("--tone-mapper", dest='tone_mapper_override', default=None,
                            choices=['Reinhard', 'Drago', 'Mantiuk', 'all'],
                            help="Override tone mapper for combined LDR previews (default: use config).")
    parser_batch.add_argument("--ext", type=str, default="jpg", 
                            help="Input image file extension to process (e.g., jpg, png, tif).")
    # Potential future args for batch: --recursive, --num-workers

    return parser.parse_args()

def _resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    """Resolve a path string to an absolute Path object, using base_dir if relative."""
    path = Path(path_str)
    if not path.is_absolute() and base_dir:
        return (base_dir / path).resolve()
    return path.resolve()

def run_single_conversion(args: argparse.Namespace) -> None:
    """Execute HDR conversion for a single image based on parsed arguments."""
    logger.info("--- Running in Single Image Mode ---")
    try:
        # Load configuration
        config = get_config(args.config)

        # Resolve paths 
        # Input: Relative to config.DATA_DIR if not absolute, else relative to CWD
        input_path = Path(args.input_image)
        if not input_path.is_absolute():
             # Try resolving relative to DATA_DIR first
             data_dir = getattr(config, 'data_dir_path', Path.cwd() / 'input') # Use property or default
             potential_path = data_dir / input_path
             if potential_path.is_file():
                  input_path = potential_path.resolve()
                  logger.debug(f"Resolved relative input path using DATA_DIR: {input_path}")
             else:
                  # Fallback to resolving relative to CWD
                  input_path = (Path.cwd() / input_path).resolve()
                  logger.debug(f"Resolved relative input path using CWD: {input_path}")
        else:
             input_path = input_path.resolve() # Ensure absolute even if provided

        output_path = Path(args.output_hdr) # Output path determines base name/relative location

        if not input_path.is_file():
            logger.error(f"Input image not found or is not a file: {input_path}")
            return # Exit function gracefully

        logger.info(f"Processing single image: {input_path} -> {output_path} (base name)")
        
        # Call the main pipeline function
        start_time = time.perf_counter()
        result_path = create_hdr_pipeline(
            input_path=input_path, 
            output_path=output_path, 
            config=config, 
            tone_mapper_override=args.tone_mapper_override
        )
        elapsed = time.perf_counter() - start_time

        if result_path:
            logger.info(f"HDR conversion successful. Output: {result_path} (Completed in {elapsed:.2f}s)")
        else:
            logger.error(f"HDR conversion failed for {input_path}. (Failed in {elapsed:.2f}s)")

    except Exception as e:
        logger.error(f"An unexpected error occurred during single image processing: {e}", exc_info=True)

def run_batch_conversion(args: argparse.Namespace) -> None:
    """Execute HDR conversion for a batch of images based on parsed arguments."""
    logger.info("--- Running in Batch Processing Mode ---")
    total_start_time = time.perf_counter()
    try:
        # Load configuration
        config = get_config(args.config)

        # Resolve input and output directories
        input_dir = _resolve_path(args.input_dir, base_dir=Path.cwd())
        output_dir_base = _resolve_path(args.output_dir, base_dir=Path.cwd()) # Base for output structure

        if not input_dir.is_dir():
            logger.error(f"Input directory not found or is not a directory: {input_dir}")
            return
            
        # Ensure the base output directory exists (pipeline handles the 'output' subfolder)
        try:
            output_dir_base.mkdir(parents=True, exist_ok=True)
            logger.info(f"Input directory: {input_dir}")
            logger.info(f"Base output directory: {output_dir_base}")
        except OSError as e:
            logger.error(f"Could not create base output directory {output_dir_base}: {e}")
            return

        # Find input images
        extension = args.ext.lstrip('.') # Remove leading dot if present
        image_files = list(input_dir.glob(f"*.{extension}"))

        if not image_files:
            logger.warning(f"No images found with extension '.{extension}' in directory: {input_dir}")
            return

        logger.info(f"Found {len(image_files)} image(s) with extension '.{extension}' to process.")

        success_count = 0
        fail_count = 0
        for i, input_path in enumerate(image_files):
            logger.info(f"--- Processing image {i + 1}/{len(image_files)}: {input_path.name} ---")
            start_time = time.perf_counter()
            
            # Define output path based on input name and base output dir
            output_hdr_name = f"{input_path.stem}.hdr"
            # The pipeline will place results in an 'output' subfolder relative to output_dir_base
            output_path_base = output_dir_base / output_hdr_name 
            
            try:
                result_path = create_hdr_pipeline(
                    input_path=input_path,
                    output_path=output_path_base,
                    config=config,
                    tone_mapper_override=args.tone_mapper_override
                )
                elapsed = time.perf_counter() - start_time
                if result_path:
                    logger.info(f"Successfully processed {input_path.name}. Output: {result_path} (Took {elapsed:.2f}s)")
                    success_count += 1
                else:
                    logger.error(f"Failed to process {input_path.name}. (Failed in {elapsed:.2f}s)")
                    fail_count += 1
            except Exception as e:
                 # Catch errors specific to one image in the batch
                 elapsed = time.perf_counter() - start_time
                 logger.error(f"Error processing {input_path.name}: {e} (Failed in {elapsed:.2f}s)", exc_info=True)
                 fail_count += 1
            logger.info("-" * 60) # Separator between images

        total_elapsed = time.perf_counter() - total_start_time
        logger.info("--- Batch Processing Summary ---")
        logger.info(f"Total images processed: {len(image_files)}")
        logger.info(f"Successful conversions: {success_count}")
        logger.info(f"Failed conversions: {fail_count}")
        logger.info(f"Total batch time: {total_elapsed:.2f} seconds")

    except Exception as e:
        logger.error(f"An unexpected error occurred during batch processing setup: {e}", exc_info=True)

def main():
    """Main entry point for the CLI application."""
    args = _parse_arguments()

    if args.mode == 'single':
        run_single_conversion(args)
    elif args.mode == 'batch':
        run_batch_conversion(args)
    else:
        # Should not happen due to argparse 'required=True'
        logger.error(f"Invalid mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
