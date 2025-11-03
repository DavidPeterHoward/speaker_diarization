#!/usr/bin/env python3
"""
Installation script for AudioTranscribe.
Helps set up the environment and install dependencies.
"""

import os
import sys
import subprocess
import argparse
import platform

def check_python_version():
    """Check if Python version is compatible."""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Python 3.8+ is required. You are using Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        return False
    print(f"✅ Python version {python_version.major}.{python_version.minor}.{python_version.micro} is compatible")
    return True

def create_virtual_env(venv_path='venv'):
    """Create a virtual environment."""
    if os.path.exists(venv_path):
        print(f"ℹ️ Virtual environment already exists at {venv_path}")
        return True
    
    try:
        subprocess.run([sys.executable, '-m', 'venv', venv_path], check=True)
        print(f"✅ Created virtual environment at {venv_path}")
        
        # Print activation instructions
        if platform.system() == 'Windows':
            print(f"   To activate: {venv_path}\\Scripts\\activate")
        else:
            print(f"   To activate: source {venv_path}/bin/activate")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def install_requirements(mode='basic', gpu=False, venv_path='venv'):
    """Install dependencies based on the selected mode."""
    # Determine pip path
    if platform.system() == 'Windows':
        pip_path = os.path.join(venv_path, 'Scripts', 'pip')
    else:
        pip_path = os.path.join(venv_path, 'bin', 'pip')
    
    # Make sure pip is up to date
    try:
        subprocess.run([pip_path, 'install', '--upgrade', 'pip'], check=True)
        print("✅ Upgraded pip to the latest version")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to upgrade pip: {e}")
        return False
    
    # Install core requirements
    try:
        subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
        print("✅ Installed core dependencies")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install core dependencies: {e}")
        return False
    
    if mode == 'full':
        # Install PyTorch
        if gpu:
            # Install PyTorch with CUDA
            try:
                subprocess.run([pip_path, 'install', 'torch', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'], check=True)
                print("✅ Installed PyTorch with CUDA support")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install PyTorch with CUDA: {e}")
                print("ℹ️ Trying to install CPU-only PyTorch")
                try:
                    subprocess.run([pip_path, 'install', 'torch', 'torchaudio'], check=True)
                    print("✅ Installed PyTorch (CPU only)")
                except subprocess.CalledProcessError as e2:
                    print(f"❌ Failed to install PyTorch: {e2}")
                    return False
        else:
            # Install CPU-only PyTorch
            try:
                subprocess.run([pip_path, 'install', 'torch', 'torchaudio'], check=True)
                print("✅ Installed PyTorch (CPU only)")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install PyTorch: {e}")
                return False
        
        # Install transcription backends
        try:
            subprocess.run([pip_path, 'install', 'openai-whisper'], check=True)
            print("✅ Installed OpenAI Whisper")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install OpenAI Whisper: {e}")
        
        try:
            subprocess.run([pip_path, 'install', 'faster-whisper'], check=True)
            print("✅ Installed Faster Whisper")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Faster Whisper: {e}")
        
        # Install diarization backends
        try:
            subprocess.run([pip_path, 'install', 'pyannote.audio'], check=True)
            print("✅ Installed Pyannote Audio")
            print("ℹ️ Note: You will need a HuggingFace token to use Pyannote. Set the HF_TOKEN environment variable.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Pyannote Audio: {e}")
        
        try:
            subprocess.run([pip_path, 'install', 'resemblyzer'], check=True)
            print("✅ Installed Resemblyzer")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Resemblyzer: {e}")
    
    return True

def setup_data_directories():
    """Create necessary data directories."""
    try:
        from models import Config
        Config.ensure_directories()
        print("✅ Created data directories")
        return True
    except Exception as e:
        print(f"❌ Failed to create data directories: {e}")
        
        # Try to create directories manually
        try:
            dirs = ['data', 'data/uploads', 'data/transcripts', 'data/audio_cache']
            for d in dirs:
                os.makedirs(d, exist_ok=True)
            print("✅ Created data directories manually")
            return True
        except Exception as e2:
            print(f"❌ Failed to create directories manually: {e2}")
            return False

def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description='Install AudioTranscribe')
    parser.add_argument('--mode', choices=['basic', 'full'], default='basic',
                      help='Installation mode: basic (mock backends only) or full (with ML backends)')
    parser.add_argument('--gpu', action='store_true', default=False,
                      help='Install with GPU support (CUDA)')
    parser.add_argument('--venv', default='venv',
                      help='Virtual environment path')
    args = parser.parse_args()
    
    print("=== AudioTranscribe Installation ===")
    print(f"Mode: {args.mode}")
    print(f"GPU support: {'Yes' if args.gpu else 'No'}")
    print(f"Virtual environment: {args.venv}")
    print("==================================")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    if not create_virtual_env(args.venv):
        return False
    
    # Install dependencies
    if not install_requirements(args.mode, args.gpu, args.venv):
        return False
    
    # Setup data directories
    if not setup_data_directories():
        return False
    
    print("\n=== Installation Complete ===")
    print("To start using AudioTranscribe:")
    if platform.system() == 'Windows':
        print(f"1. Activate the virtual environment: {args.venv}\\Scripts\\activate")
    else:
        print(f"1. Activate the virtual environment: source {args.venv}/bin/activate")
    print("2. Run the application: python main.py --server")
    print("3. Visit http://127.0.0.1:5000 in your browser")
    
    if args.mode == 'basic':
        print("\nNote: You've installed the basic version with mock backends.")
        print("For better transcription quality, consider installing the full version:")
        print(f"python {os.path.basename(__file__)} --mode full")
    
    if args.mode == 'full' and 'pyannote' in str(subprocess.run([os.path.join(args.venv, 'bin' if platform.system() != 'Windows' else 'Scripts', 'pip'), 'freeze'], capture_output=True, text=True).stdout):
        print("\nImportant: For Pyannote Audio to work, you need to:")
        print("1. Create a HuggingFace account at https://huggingface.co/")
        print("2. Visit https://huggingface.co/pyannote/speaker-diarization and accept the license")
        print("3. Create an access token in your HuggingFace account settings")
        print("4. Set the environment variable: export HF_TOKEN=your_token_here")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)