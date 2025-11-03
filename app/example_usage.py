#!/usr/bin/env python3
"""
Example usage of AudioTranscribe with mock backends.
This script demonstrates how to use the system when the real backends aren't available.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from models import Config
from transcription import process_audio_file

def main():
    """Main function to demonstrate usage with sample audio."""
    # Ensure data directories exist
    Config.ensure_directories()
    
    # Sample audio path - for testing, create a simple file if none exists
    sample_path = os.path.join(Config.UPLOAD_FOLDER, "sample.wav")
    
    if not os.path.exists(sample_path):
        logger.info("Creating a sample audio file for testing")
        try:
            import numpy as np
            import soundfile as sf
            
            # Create a simple sine wave
            sample_rate = 16000
            duration = 10  # seconds
            t = np.linspace(0, duration, duration * sample_rate)
            
            # Generate a sine wave that changes pitch
            frequencies = [440, 880, 440, 220, 440]
            segments = np.split(t, len(frequencies))
            signal = np.concatenate([np.sin(2 * np.pi * f * seg) for f, seg in zip(frequencies, segments)])
            
            # Add some noise
            signal = signal + 0.1 * np.random.randn(len(signal))
            
            # Normalize
            signal = signal / np.max(np.abs(signal))
            
            # Save to file
            sf.write(sample_path, signal, sample_rate)
            logger.info(f"Created sample audio file at {sample_path}")
        except ImportError:
            logger.warning("NumPy or SoundFile not available, creating empty file")
            with open(sample_path, 'wb') as f:
                # Create a minimal valid WAV file (44 bytes header + some data)
                # WAV header format:
                # - "RIFF" (4 bytes)
                # - File size minus 8 (4 bytes, little-endian)
                # - "WAVE" (4 bytes)
                # - "fmt " (4 bytes)
                # - Format chunk size (4 bytes, value: 16 for PCM)
                # - Format type (2 bytes, value: 1 for PCM)
                # - Channels (2 bytes, value: 1 for mono)
                # - Sample rate (4 bytes, value: 16000)
                # - Byte rate (4 bytes, value: 16000 * 1 * 2 = 32000)
                # - Block align (2 bytes, value: 1 * 2 = 2)
                # - Bits per sample (2 bytes, value: 16)
                # - "data" (4 bytes)
                # - Data size (4 bytes, little-endian)
                header = bytearray([
                    # "RIFF"
                    0x52, 0x49, 0x46, 0x46,
                    # File size - 8 (placeholder)
                    0x24, 0x00, 0x00, 0x00,
                    # "WAVE"
                    0x57, 0x41, 0x56, 0x45,
                    # "fmt "
                    0x66, 0x6D, 0x74, 0x20,
                    # Format chunk size
                    0x10, 0x00, 0x00, 0x00,
                    # Format type (PCM)
                    0x01, 0x00,
                    # Channels (mono)
                    0x01, 0x00,
                    # Sample rate (16000 Hz)
                    0x80, 0x3E, 0x00, 0x00,
                    # Byte rate (16000 * 1 * 2 = 32000)
                    0x00, 0x7D, 0x00, 0x00,
                    # Block align (1 * 2 = 2)
                    0x02, 0x00,
                    # Bits per sample (16)
                    0x10, 0x00,
                    # "data"
                    0x64, 0x61, 0x74, 0x61,
                    # Data size (placeholder)
                    0x00, 0x00, 0x00, 0x00
                ])
                
                # Add some sample data (one second of silence at 16000 Hz)
                data = bytearray([0x00, 0x00] * 16000)
                
                # Update file size and data size in header
                file_size = len(header) + len(data) - 8
                header[4:8] = file_size.to_bytes(4, byteorder='little')
                header[40:44] = len(data).to_bytes(4, byteorder='little')
                
                f.write(header + data)
            logger.info(f"Created empty WAV file at {sample_path}")
    
    # Process the audio file
    logger.info(f"Processing sample audio file: {sample_path}")
    
    # Try different combinations of backends
    backends = [
        ("faster_whisper", "pyannote"),
        ("whisper", "resemblyzer"),
        ("faster_whisper", "resemblyzer"),
        ("whisper", "pyannote")
    ]
    
    for transcription, diarization in backends:
        logger.info(f"Testing with {transcription} + {diarization}")
        result = process_audio_file(
            sample_path,
            model_size="base",
            transcription_backend=transcription,
            diarization_backend=diarization
        )
        
        if result.success:
            logger.info(f"Processing succeeded: {result.message}")
            if result.job and result.job.output_path:
                logger.info(f"Output saved to: {result.job.output_path}")
                logger.info(f"Segments: {len(result.job.segments)}")
                
                # Show the first few segments
                for i, segment in enumerate(result.job.segments[:3]):
                    logger.info(f"  Segment {i+1}: [{segment.start:.2f}-{segment.end:.2f}] {segment.speaker}: {segment.text}")
                
                if len(result.job.segments) > 3:
                    logger.info(f"  ... and {len(result.job.segments) - 3} more segments")
            else:
                logger.warning("No output path or job provided in result")
        else:
            logger.error(f"Processing failed: {result.message}")
            if result.error:
                logger.error(f"Error details: {str(result.error)}")
    
    logger.info("Testing complete")

if __name__ == "__main__":
    main()