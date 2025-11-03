"""
AudioTranscribe: Logging Utilities
----------------------------------
Centralized logging configuration and utilities for consistent logging across the application.
"""

import logging
import logging.config
import logging.handlers
import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps
import time

# Try to import json logger
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False


class PerformanceLogger:
    """Context manager for logging performance metrics."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        if exc_type:
            self.logger.error(
                f"Operation {self.operation_name} failed after {elapsed_time:.3f}s: {exc_val}",
                exc_info=True
            )
        else:
            self.logger.info(f"Operation {self.operation_name} completed in {elapsed_time:.3f}s")


class StructuredLogger:
    """Logger wrapper for structured logging with context."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.context = {}
    
    def add_context(self, **kwargs):
        """Add context that will be included in all log messages."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear the context."""
        self.context = {}
    
    def _log_with_context(self, level: int, msg: str, **kwargs):
        """Log a message with context."""
        extra = {**self.context, **kwargs}
        self.logger.log(level, msg, extra={'structured': extra})
    
    def debug(self, msg: str, **kwargs):
        self._log_with_context(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        self._log_with_context(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        self._log_with_context(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        self._log_with_context(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, **kwargs)


class ErrorTracker:
    """Track and aggregate errors for reporting."""
    
    def __init__(self, max_errors: int = 100):
        self.errors = []
        self.max_errors = max_errors
        self.error_counts = {}
    
    def track_error(self, error_type: str, error_msg: str, details: Optional[Dict] = None):
        """Track an error occurrence."""
        timestamp = datetime.utcnow().isoformat()
        
        # Count error types
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Store error details
        error_entry = {
            'timestamp': timestamp,
            'type': error_type,
            'message': error_msg,
            'details': details or {},
            'traceback': traceback.format_exc() if sys.exc_info()[0] else None
        }
        
        self.errors.append(error_entry)
        
        # Keep only the most recent errors
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_types': self.error_counts,
            'recent_errors': self.errors[-10:],  # Last 10 errors
            'most_common': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }
    
    def clear(self):
        """Clear all tracked errors."""
        self.errors = []
        self.error_counts = {}


def log_exceptions(logger: Optional[logging.Logger] = None):
    """Decorator to log exceptions from functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}: {str(e)}",
                    exc_info=True,
                    extra={
                        'function': func.__name__,
                        'module': func.__module__,
                        'args': str(args)[:200],  # Truncate for safety
                        'kwargs': str(kwargs)[:200]
                    }
                )
                raise
        return wrapper
    return decorator


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    use_json: bool = False,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses default)
        log_dir: Directory for log files
        use_json: Whether to use JSON formatting
        console_output: Whether to log to console
        file_output: Whether to log to file
    
    Returns:
        Configured root logger
    """
    # Ensure log directory exists
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine log file path
    if not log_file and file_output:
        log_dir = log_dir or "./logs"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(log_dir, "application.log")
    
    # Create formatters
    formatters = {
        "standard": {
            "format": "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)-8s] %(name)s [%(filename)s:%(lineno)d] - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    }
    
    # Add JSON formatter if available and requested
    if use_json and JSON_LOGGER_AVAILABLE:
        formatters["json"] = {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    
    # Create handlers
    handlers = {}
    
    if console_output:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": log_level,
            "stream": "ext://sys.stdout"
        }
    
    if file_output and log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json" if (use_json and JSON_LOGGER_AVAILABLE) else "detailed",
            "level": "DEBUG",  # File gets everything
            "filename": log_file,
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 5
        }
    
    # Create logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {
            "": {  # Root logger
                "handlers": list(handlers.keys()),
                "level": "DEBUG",
                "propagate": True
            },
            "werkzeug": {
                "level": "INFO"
            },
            "flask.app": {
                "level": "INFO"
            },
            "urllib3": {
                "level": "WARNING"
            }
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Get root logger
    logger = logging.getLogger()
    logger.info(f"Logging configured: level={log_level}, file={log_file}, json={use_json and JSON_LOGGER_AVAILABLE}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(logger: Optional[logging.Logger] = None, log_args: bool = True, log_result: bool = False):
    """
    Decorator to log function calls with arguments and results.
    
    Args:
        logger: Logger to use (if None, uses function's module logger)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            # Log function call
            if log_args:
                logger.debug(
                    f"Calling {func.__name__}",
                    extra={
                        'function': func.__name__,
                        'args': str(args)[:500],
                        'kwargs': str(kwargs)[:500]
                    }
                )
            else:
                logger.debug(f"Calling {func.__name__}")
            
            # Execute function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # Log result
                if log_result:
                    logger.debug(
                        f"{func.__name__} completed in {elapsed:.3f}s",
                        extra={
                            'function': func.__name__,
                            'elapsed_time': elapsed,
                            'result': str(result)[:500]
                        }
                    )
                else:
                    logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
                
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {elapsed:.3f}s: {str(e)}",
                    exc_info=True,
                    extra={
                        'function': func.__name__,
                        'elapsed_time': elapsed,
                        'error': str(e)
                    }
                )
                raise
        
        return wrapper
    return decorator


# Global error tracker instance
error_tracker = ErrorTracker()


def log_error(error_type: str, error_msg: str, logger: Optional[logging.Logger] = None, **details):
    """
    Log an error and track it in the error tracker.
    
    Args:
        error_type: Type/category of error
        error_msg: Error message
        logger: Logger to use
        **details: Additional error details
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Log the error
    logger.error(f"[{error_type}] {error_msg}", extra=details)
    
    # Track the error
    error_tracker.track_error(error_type, error_msg, details)


def get_error_summary() -> Dict[str, Any]:
    """Get summary of tracked errors."""
    return error_tracker.get_summary()


def clear_error_tracker():
    """Clear the error tracker."""
    error_tracker.clear()