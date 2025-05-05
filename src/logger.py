"""
Logger Setup Module
-------------------
Provides a centralized function to configure and retrieve loggers.
Ensures consistent logging format and level across the application.
Uses a singleton pattern to avoid duplicate handlers.
"""

import logging
import sys

# --- Configuration ---
LOG_LEVEL = logging.INFO # Default log level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
LOG_FORMAT = '[%(asctime)s] %(levelname)-7s [%(name)s]: %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- Singleton Pattern for Logger Setup ---
_loggers = {}
_handler = None

def setup_logger(name: str, level: int = LOG_LEVEL) -> logging.Logger:
    """
    Get a logger instance, configuring the root handler only once.

    Args:
        name: Name of the logger (typically __name__ of the calling module).
        level: Logging level for this specific logger (defaults to global LOG_LEVEL).

    Returns:
        Configured logger instance.
    """
    global _handler
    
    # Configure the root handler only once
    if _handler is None:
        _handler = logging.StreamHandler(sys.stdout) # Log to standard output
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        _handler.setFormatter(formatter)
        
        # Get the root logger and add the handler
        root_logger = logging.getLogger() 
        # Set root logger level - Note: This affects ALL loggers unless they have a specific level set.
        # Setting it to the lowest possible level (DEBUG) allows handlers/loggers to filter upwards.
        root_logger.setLevel(logging.DEBUG) 
        
        # Avoid adding duplicate handlers if this function is somehow called concurrently initially
        if not root_logger.hasHandlers() or not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            root_logger.addHandler(_handler)
            # print(f"Root logger handler configured by logger: {name}") # Debug print
        # else:
            # print(f"Root logger handler already configured (attempt by {name})" ) # Debug print

    # Retrieve the specific logger
    logger = logging.getLogger(name)
    
    # Set the level for this specific logger
    # If not already set, apply the requested level.
    # If already set (e.g., by previous call), keep the existing level 
    # unless explicitly overridden? Current logic sets level each time.
    logger.setLevel(level)
    
    # Ensure logger doesn't propagate to root if it has its own handler (doesn't apply here as we use root handler)
    # logger.propagate = False 

    # Store logger reference (optional, mainly for tracking created loggers)
    if name not in _loggers:
        _loggers[name] = logger
        # print(f"Logger '{name}' created/retrieved with level {logging.getLevelName(level)}.") # Debug print

    return logger

# Example usage (typically in other modules):
# from src.logger import setup_logger
# logger = setup_logger(__name__)
# logger.info("This is an info message.")
# logger.debug("This is a debug message.")
