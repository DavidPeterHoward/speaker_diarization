#!/usr/bin/env python3
"""
AudioTranscribe: Automatic Multi-Speaker Transcription System
---------------------------------------------------------
A modular Flask-based application for transcribing audio recordings with speaker diarization.
"""

# Suppress non-critical warnings before importing dependencies
try:
    from .suppress_warnings import suppress_dependency_warnings
    suppress_dependency_warnings()
except ImportError:
    # Fallback if suppress_warnings doesn't exist
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

import argparse
import logging
import logging.config
import os
import sys
import asyncio

from .models import Config, init_db
from .transcription import (TRANSCRIPTION_BACKENDS, DIARIZATION_BACKENDS,
                         process_audio_file)
from .app import app
from .meta_recursive_integration import initialize_meta_recursive_system
from .speaker_api import register_speaker_api

# ------------------------ LOGGING SETUP ------------------------ #
def setup_logging():
    """Configure structured logging to console and file."""
    Config.ensure_directories()
    
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            },
            "json": {
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "standard",
                "level": "DEBUG",
                "filename": Config.LOG_FILE,
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": "DEBUG",
                "propagate": True,
            },
            "werkzeug": {
                "level": "INFO",
            },
            "flask.app": {
                "level": "INFO",
            },
        },
    }
    
    # Check if json logging is available
    try:
        from pythonjsonlogger import jsonlogger
        logging.config.dictConfig(LOGGING_CONFIG)
    except ImportError:
        # Fall back to standard logging if json logger is not available
        del LOGGING_CONFIG["formatters"]["json"]
        logging.config.dictConfig(LOGGING_CONFIG)

# Create logger
logger = logging.getLogger(__name__)

# ------------------------ COMMAND LINE INTERFACE ------------------------ #
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AudioTranscribe: Speaker-aware audio transcription")
    
    # Mode selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--server', action='store_true', help="Run as web server")
    group.add_argument('--file', help="Process a single audio file")
    
    # Processing options
    parser.add_argument('--model', default=Config.DEFAULT_MODEL_SIZE, 
                      choices=['tiny', 'base', 'small', 'medium', 'large'],
                      help="Whisper model size")
    parser.add_argument('--transcription', default=Config.DEFAULT_TRANSCRIPTION_BACKEND,
                      choices=['faster_whisper', 'whisper'],
                      help="Transcription backend")
    parser.add_argument('--diarization', default=Config.DEFAULT_DIARIZATION_BACKEND,
                      choices=['pyannote', 'resemblyzer'],
                      help="Diarization backend")
    
    # Server options
    parser.add_argument('--host', default=Config.HOST, help="Server host")
    parser.add_argument('--port', type=int, default=Config.PORT, help="Server port")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode")
    
    return parser.parse_args()

# ------------------------ MAIN ------------------------ #
def main():
    """Main application entry point."""
    # Set up logging
    setup_logging()
    logger.info("Starting AudioTranscribe")
    
    # Ensure directories exist
    Config.ensure_directories()
    
    # Initialize database
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)
    
    # Parse arguments
    args = parse_arguments()
    
    # Log available backends
    logger.info("Available transcription backends:")
    for name, info in TRANSCRIPTION_BACKENDS.items():
        logger.info(f"  - {name}: {'Available' if info['available'] else 'Not available'}")
    
    logger.info("Available diarization backends:")
    for name, info in DIARIZATION_BACKENDS.items():
        logger.info(f"  - {name}: {'Available' if info['available'] else 'Not available'}")
    
    # Check for enhanced mode
    enhanced_mode = os.environ.get('AUDIOTRANSCRIBE_ENHANCED_MODE', 'false').lower() == 'true'
    
    if enhanced_mode:
        logger.info("Enhanced mode enabled - registering enhanced API endpoints")
        try:
            from app.api_enhanced import enhanced_api
            app.register_blueprint(enhanced_api)
            logger.info("Enhanced API endpoints registered")
        except Exception as e:
            logger.warning(f"Enhanced API registration failed: {e}")
    
    # Run in appropriate mode
    if args.server:
        # Run as web server
        logger.info(f"Starting web server on {args.host}:{args.port}")

        if not any(info['available'] for _, info in TRANSCRIPTION_BACKENDS.items()):
            logger.error("No transcription backends available. Please install at least one of: faster_whisper, whisper")
            sys.exit(1)

        # Register speaker API
        register_speaker_api(app)
        logger.info("Speaker fine-tuning API registered")

        # Initialize meta-recursive system
        logger.info("Initializing meta-recursive self-improvement system...")
        try:
            # Create event loop for async initialization
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(initialize_meta_recursive_system())
                logger.info("Meta-recursive system initialized successfully")
            finally:
                loop.close()
        except Exception as e:
            logger.warning(f"Meta-recursive system initialization failed: {e}")
            logger.info("Continuing without meta-recursive features")

        app.run(host=args.host, port=args.port, debug=args.debug)
    elif args.file:
        # Process a single file
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        
        logger.info(f"Processing file: {args.file}")
        logger.info(f"Using model: {args.model}")
        logger.info(f"Using transcription backend: {args.transcription}")
        logger.info(f"Using diarization backend: {args.diarization}")
        
        result = process_audio_file(
            args.file, 
            model_size=args.model,
            transcription_backend=args.transcription,
            diarization_backend=args.diarization
        )
        
        if result.success:
            logger.info(f"Processing complete: {result.message}")
            if result.job and result.job.output_path:
                logger.info(f"Output saved to: {result.job.output_path}")
        else:
            logger.error(f"Processing failed: {result.message}")
            if result.error:
                logger.error(f"Error details: {str(result.error)}")
            sys.exit(1)
    else:
        # No mode specified, run server by default
        logger.info("No mode specified, starting web server")
        app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()