"""
Suppress non-critical warnings from dependencies
"""
import warnings
import logging
import os

def suppress_dependency_warnings():
    """Suppress known non-critical warnings from dependencies."""
    
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='ctranslate2')
    warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
    warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
    warnings.filterwarnings('ignore', category=UserWarning, module='pyannote')
    
    # Set environment variables to reduce verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Only show errors from transformers
    
    # Configure logging to filter out specific warnings
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.setLevel(logging.ERROR)
    
    # Suppress specific PyTorch warnings
    try:
        import torch
        torch.set_warn_always(False)
    except:
        pass

def get_diagnostic_info():
    """Get diagnostic information about the environment."""
    info = {
        'python_version': None,
        'torch_available': False,
        'torch_version': None,
        'cuda_available': False,
        'ffmpeg_available': False,
        'audio_backends': {},
        'diarization_backends': {}
    }
    
    # Python version
    import sys
    info['python_version'] = sys.version
    
    # PyTorch
    try:
        import torch
        info['torch_available'] = True
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
    except ImportError:
        pass
    
    # FFmpeg
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            info['ffmpeg_available'] = True
            # Extract version from first line
            first_line = result.stdout.split('\n')[0]
            info['ffmpeg_version'] = first_line
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info['ffmpeg_available'] = False
    
    # Check audio backends
    try:
        from faster_whisper import WhisperModel
        info['audio_backends']['faster_whisper'] = True
    except ImportError:
        info['audio_backends']['faster_whisper'] = False
    
    try:
        import whisper
        info['audio_backends']['whisper'] = True
    except ImportError:
        info['audio_backends']['whisper'] = False
    
    # Check diarization backends
    try:
        from pyannote.audio import Pipeline
        info['diarization_backends']['pyannote'] = True
    except ImportError:
        info['diarization_backends']['pyannote'] = False
    
    try:
        from resemblyzer import VoiceEncoder
        info['diarization_backends']['resemblyzer'] = True
    except ImportError:
        info['diarization_backends']['resemblyzer'] = False
    
    return info

def print_diagnostic_info():
    """Print diagnostic information."""
    info = get_diagnostic_info()
    
    print("\n" + "="*60)
    print("SYSTEM DIAGNOSTIC INFORMATION")
    print("="*60)
    
    print(f"\nPython Version: {info['python_version'].split()[0]}")
    print(f"PyTorch Available: {info['torch_available']}")
    if info['torch_available']:
        print(f"PyTorch Version: {info['torch_version']}")
        print(f"CUDA Available: {info['cuda_available']}")
    
    print(f"\nFFmpeg Available: {info['ffmpeg_available']}")
    if info['ffmpeg_available']:
        print(f"FFmpeg Version: {info.get('ffmpeg_version', 'Unknown')}")
    
    print("\nTranscription Backends:")
    for backend, available in info['audio_backends'].items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {backend}: {status}")
    
    print("\nDiarization Backends:")
    for backend, available in info['diarization_backends'].items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {backend}: {status}")
    
    print("\n" + "="*60)
    print("WARNING ANALYSIS")
    print("="*60)
    
    warnings_analysis = []
    
    if not info['ffmpeg_available']:
        warnings_analysis.append({
            'severity': 'WARNING',
            'message': 'FFmpeg not found in PATH',
            'impact': 'May affect audio file format conversion',
            'solution': 'Install FFmpeg and add to PATH'
        })
    
    if not any(info['audio_backends'].values()):
        warnings_analysis.append({
            'severity': 'ERROR',
            'message': 'No transcription backends available',
            'impact': 'Cannot perform transcription',
            'solution': 'Install faster-whisper or openai-whisper'
        })
    
    if not any(info['diarization_backends'].values()):
        warnings_analysis.append({
            'severity': 'ERROR',
            'message': 'No diarization backends available',
            'impact': 'Cannot perform speaker diarization',
            'solution': 'Install pyannote.audio or resemblyzer'
        })
    
    if warnings_analysis:
        for warning in warnings_analysis:
            print(f"\n[{warning['severity']}] {warning['message']}")
            print(f"  Impact: {warning['impact']}")
            print(f"  Solution: {warning['solution']}")
    else:
        print("\n✓ All critical components are available")
    
    print("\n" + "="*60)
    
    return info

if __name__ == '__main__':
    print_diagnostic_info()

