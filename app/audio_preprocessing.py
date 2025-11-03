"""
Advanced Audio Preprocessing Pipeline
=====================================
Comprehensive audio preprocessing for optimal transcription accuracy.
"""

import logging
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import wiener
from typing import Tuple, List, Dict, Optional
import tempfile
import os

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """Advanced audio preprocessing pipeline."""
    
    def __init__(self):
        self.sample_rate = 16000
        self.frame_length = 1024
        self.hop_length = 512
        
    def preprocess_audio(self, audio_path: str, config: Dict = None) -> str:
        """Comprehensive audio preprocessing pipeline."""
        if config is None:
            config = self.get_default_config()
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            logger.info(f"Loaded audio: {len(y)} samples at {sr}Hz")
            
            # Apply preprocessing steps
            y_processed = y.copy()
            
            # 1. Noise reduction
            if config.get('noise_reduction', True):
                y_processed = self.spectral_subtraction(y_processed, sr)
                logger.info("Applied spectral subtraction noise reduction")
            
            # 2. Voice Activity Detection and filtering
            if config.get('vad_filtering', True):
                y_processed = self.apply_vad_filtering(y_processed, sr)
                logger.info("Applied VAD filtering")
            
            # 3. Dynamic range compression
            if config.get('dynamic_range_compression', True):
                y_processed = self.dynamic_range_compression(y_processed)
                logger.info("Applied dynamic range compression")
            
            # 4. Spectral enhancement
            if config.get('spectral_enhancement', True):
                y_processed = self.spectral_enhancement(y_processed, sr)
                logger.info("Applied spectral enhancement")
            
            # 5. Normalization
            if config.get('normalization', True):
                y_processed = self.normalize_audio(y_processed)
                logger.info("Applied normalization")
            
            # Save processed audio
            output_path = self._save_processed_audio(y_processed, sr, audio_path)
            logger.info(f"Audio preprocessing complete: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return audio_path
    
    def spectral_subtraction(self, audio: np.ndarray, sr: int, noise_factor: float = 0.5) -> np.ndarray:
        """Reduce background noise using spectral subtraction."""
        try:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first 0.5 seconds
            noise_frames = int(0.5 * sr / 512)
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Apply spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            beta = 0.01  # Spectral floor factor
            
            # Calculate subtraction factor
            subtraction_factor = 1.0 - alpha * (noise_spectrum / (magnitude + 1e-10))
            subtraction_factor = np.maximum(subtraction_factor, beta)
            
            # Apply subtraction
            enhanced_magnitude = magnitude * subtraction_factor
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Spectral subtraction failed: {e}")
            return audio
    
    def apply_vad_filtering(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply Voice Activity Detection filtering."""
        try:
            # Calculate energy and spectral features
            frame_length = 1024
            hop_length = 512
            
            # Energy-based VAD
            energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            energy_threshold = np.mean(energy) + 0.5 * np.std(energy)
            
            # Spectral centroid for voice detection
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Combined VAD decision
            vad_decision = (energy > energy_threshold) & (spectral_centroid > 1000) & (zcr < 0.1)
            
            # Apply VAD filtering
            vad_audio = audio.copy()
            for i, is_speech in enumerate(vad_decision):
                start_sample = i * hop_length
                end_sample = min(start_sample + frame_length, len(audio))
                
                if not is_speech:
                    # Attenuate non-speech regions
                    vad_audio[start_sample:end_sample] *= 0.1
            
            return vad_audio
            
        except Exception as e:
            logger.warning(f"VAD filtering failed: {e}")
            return audio
    
    def dynamic_range_compression(self, audio: np.ndarray, ratio: float = 4.0, threshold: float = 0.1) -> np.ndarray:
        """Apply dynamic range compression."""
        try:
            # Convert to dB
            audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
            
            # Apply compression
            compressed_db = np.where(
                audio_db > threshold,
                threshold + (audio_db - threshold) / ratio,
                audio_db
            )
            
            # Convert back to linear
            compressed_audio = np.sign(audio) * (10 ** (compressed_db / 20))
            
            return compressed_audio
            
        except Exception as e:
            logger.warning(f"Dynamic range compression failed: {e}")
            return audio
    
    def spectral_enhancement(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply spectral enhancement for speech clarity."""
        try:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Spectral enhancement
            # Emphasize frequencies in speech range (300-3400 Hz)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            speech_mask = (freqs >= 300) & (freqs <= 3400)
            
            # Apply enhancement
            enhancement_factor = 1.2
            magnitude[speech_mask] *= enhancement_factor
            
            # Reconstruct signal
            enhanced_stft = magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Spectral enhancement failed: {e}")
            return audio
    
    def normalize_audio(self, audio: np.ndarray, target_lufs: float = -23.0) -> np.ndarray:
        """Normalize audio to target LUFS level."""
        try:
            # Calculate RMS
            rms = np.sqrt(np.mean(audio ** 2))
            
            # Target RMS for -23 LUFS (approximate)
            target_rms = 0.1
            
            # Apply normalization
            if rms > 0:
                normalized_audio = audio * (target_rms / rms)
            else:
                normalized_audio = audio
            
            # Prevent clipping
            max_val = np.max(np.abs(normalized_audio))
            if max_val > 0.95:
                normalized_audio = normalized_audio * (0.95 / max_val)
            
            return normalized_audio
            
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            return audio
    
    def get_default_config(self) -> Dict:
        """Get default preprocessing configuration."""
        return {
            'noise_reduction': True,
            'vad_filtering': True,
            'dynamic_range_compression': True,
            'spectral_enhancement': True,
            'normalization': True,
            'target_sr': 16000
        }
    
    def _save_processed_audio(self, audio: np.ndarray, sr: int, original_path: str) -> str:
        """Save processed audio to temporary file."""
        # Create output path
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_path = os.path.join(
            os.path.dirname(original_path),
            f"{base_name}_processed.wav"
        )
        
        # Save audio
        sf.write(output_path, audio, sr)
        return output_path

# Plugin registration
def advanced_audio_preprocessing(audio_path: str) -> str:
    """Advanced audio preprocessing plugin."""
    preprocessor = AudioPreprocessor()
    return preprocessor.preprocess_audio(audio_path)
