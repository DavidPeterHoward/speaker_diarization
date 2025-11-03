#!/usr/bin/env python3
"""
Standalone entry point to run the AudioTranscribe server
"""

import sys
import os
import argparse
import logging
import asyncio

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import modules directly
from app.models import Config, init_db
from app.transcription import (TRANSCRIPTION_BACKENDS, DIARIZATION_BACKENDS,
                              process_audio_file)
from app.app import app
from app.meta_recursive_integration import initialize_meta_recursive_system
from app.speaker_api import register_speaker_api

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
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": "INFO",
        },
        "loggers": {
            "app": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="AudioTranscribe: Speaker-aware audio transcription")
    parser.add_argument("--server", action="store_true", help="Run web server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--model", default="base", help="Default model size (tiny, base, small, medium, large)")
    parser.add_argument("--transcription", default="faster_whisper", help="Default transcription backend")
    parser.add_argument("--diarization", default="resemblyzer", help="Default diarization backend")
    parser.add_argument("file", nargs="?", help="Audio file to process")

    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

    if not any(info['available'] for _, info in TRANSCRIPTION_BACKENDS.items()):
        logger.error("No transcription backends available. Please install at least one of: faster_whisper, whisper")
        sys.exit(1)

    if args.server:
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

        logger.info(f"Starting web server on {args.host}:{args.port}")
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
