#!/usr/bin/env python3
"""
AudioTranscribe: Main Entry Point
---------------------------------
This is the main entry point for the AudioTranscribe application.
It delegates actual functionality to the modular components in the app/ directory.

MAJOR BUG FIX: This file previously contained 1792 lines of duplicated code.
Original moved to main_original_duplicate.py for reference.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add app directory to path for imports
app_dir = Path(__file__).parent / "app"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Import from modular components
try:
    from models import Config, init_db
    from app import app as flask_app
    from transcription import process_audio_file
    from logging_utils import setup_logging
    from error_handlers import register_error_handlers
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Please ensure all files are in the app/ directory and dependencies are installed.")
    sys.exit(1)

# Create logger
logger = logging.getLogger(__name__)

def run_server(host: Optional[str] = None, port: Optional[int] = None, debug: Optional[bool] = None):
    """Run the Flask web server."""
    host = host or Config.HOST
    port = port or Config.PORT
    debug = debug if debug is not None else Config.DEBUG
    
    logger.info(f"Starting AudioTranscribe server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Visit http://{host}:{port} in your browser")
    
    # Register error handlers
    register_error_handlers(flask_app)
    
    try:
        flask_app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

def process_file(file_path: str, model_size: Optional[str] = None, 
                transcription_backend: Optional[str] = None, 
                diarization_backend: Optional[str] = None) -> bool:
    """Process a single audio file."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    model_size = model_size or Config.DEFAULT_MODEL_SIZE
    transcription_backend = transcription_backend or Config.DEFAULT_TRANSCRIPTION_BACKEND
    diarization_backend = diarization_backend or Config.DEFAULT_DIARIZATION_BACKEND
    
    logger.info(f"Processing {file_path}")
    logger.info(f"Model size: {model_size}")
    logger.info(f"Transcription backend: {transcription_backend}")
    logger.info(f"Diarization backend: {diarization_backend}")
    
    try:
        result = process_audio_file(
            file_path,
            model_size=model_size,
            transcription_backend=transcription_backend,
            diarization_backend=diarization_backend
        )
        
        if result.success:
            logger.info(f"Processing successful: {result.message}")
            if result.job and result.job.output_path:
                logger.info(f"Output saved to: {result.job.output_path}")
            return True
        else:
            logger.error(f"Processing failed: {result.message}")
            if result.error:
                logger.error(f"Error details: {result.error}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
        return False

def run_tests():
    """Run test/example script."""
    logger.info("Running example usage script...")
    try:
        from example_usage import main as example_main
        example_main()
        return True
    except ImportError as e:
        logger.error(f"Could not import example_usage: {e}")
        return False
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        return False

def main() -> int:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='AudioTranscribe: Automatic Multi-Speaker Transcription System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Operation modes
    parser.add_argument('--server', action='store_true',
                       help='Run as web server (default)')
    parser.add_argument('--file', type=str,
                       help='Process a single audio file')
    parser.add_argument('--test', action='store_true',
                       help='Run test/example script')
    
    # Server options
    parser.add_argument('--host', type=str,
                       help=f'Server host (default: from config)')
    parser.add_argument('--port', type=int,
                       help=f'Server port (default: from config)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    # Processing options
    parser.add_argument('--model-size', type=str, 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help=f'Whisper model size (default: from config)')
    parser.add_argument('--transcription-backend', type=str,
                       choices=['whisper', 'faster_whisper'],
                       help=f'Transcription backend (default: from config)')
    parser.add_argument('--diarization-backend', type=str,
                       choices=['pyannote', 'resemblyzer'],
                       help=f'Diarization backend (default: from config)')
    
    # Logging options
    parser.add_argument('--log-level', type=str, 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Set logging level')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output')
    
    args = parser.parse_args()
    
    # Setup logging first
    try:
        setup_logging(
            log_level=args.log_level,
            console_output=not args.quiet,
            log_dir=Config.DATA_DIR
        )
        logger.info("AudioTranscribe starting...")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        return 1
    
    # Ensure directories exist
    try:
        Config.ensure_directories()
        logger.debug("Data directories ensured")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return 1
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return 1
    
    # Handle different operation modes
    try:
        if args.file:
            # Process a single file
            success = process_file(
                args.file,
                model_size=args.model_size,
                transcription_backend=args.transcription_backend,
                diarization_backend=args.diarization_backend
            )
            return 0 if success else 1
        elif args.test:
            # Run test/example script
            success = run_tests()
            return 0 if success else 1
        else:
            # Default: run as server
            run_server(
                host=args.host,
                port=args.port,
                debug=args.debug
            )
            return 0
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        return 0
    except Exception as e:
        logger.exception(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())