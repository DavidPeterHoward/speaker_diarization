"""
AudioTranscribe: Core Transcription Engine
-----------------------------------------
Handles audio processing, transcription, and speaker diarization.
Updated with mock fallbacks when real backends aren't available.
"""

import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf
from jinja2 import Template
import json
import librosa

# Get logger EARLY for CUDA checks below
logger = logging.getLogger(__name__)

# Import torch, but don't fail if not available
try:
    import torch
    TORCH_AVAILABLE = True
    
    # Check CUDA availability with safeguards
    CUDA_AVAILABLE = False
    CUDA_DEVICE_COUNT = 0
    if torch.cuda.is_available():
        try:
            # Test CUDA with a simple operation
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            CUDA_AVAILABLE = True
            CUDA_DEVICE_COUNT = torch.cuda.device_count()
            logger.info(f"CUDA available with {CUDA_DEVICE_COUNT} device(s)")
            if CUDA_DEVICE_COUNT > 0:
                logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except Exception as cuda_error:
            logger.warning(f"CUDA detected but not functional: {cuda_error}. Falling back to CPU.")
            CUDA_AVAILABLE = False
    else:
        logger.info("CUDA not available, using CPU")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    CUDA_DEVICE_COUNT = 0

from app.models import (
    Config, ProcessingResult, Segment, Speaker, TranscriptionJob,
    TranscriptionState, format_timestamp, get_all_speakers_with_embeddings,
    get_job, get_job_by_hash, hash_file, save_job, save_speaker
)

# logger already initialized above

# ------------------------ DYNAMIC IMPORTS WITH MOCKS ------------------------ #
# These are imported dynamically to allow for graceful fallbacks and cleaner error messages

TRANSCRIPTION_BACKENDS = {
    'faster_whisper': {'available': False, 'real_available': False, 'model': None, 'name': 'Faster Whisper'},
    'whisper': {'available': False, 'real_available': False, 'model': None, 'name': 'OpenAI Whisper'},
}

DIARIZATION_BACKENDS = {
    'pyannote': {'available': False, 'real_available': False, 'model': None, 'name': 'Pyannote'},
    'resemblyzer': {'available': False, 'real_available': False, 'model': None, 'name': 'Resemblyzer'},
}

# Try to import real backends first (unless forced to use mocks)
FORCE_MOCK = os.environ.get('AUDIOTRANSCRIBE_FORCE_MOCK', '0').lower() in ('1', 'true', 'yes')

try:
    if FORCE_MOCK:
        raise ImportError('Forced mock for faster_whisper')
    from faster_whisper import WhisperModel
    TRANSCRIPTION_BACKENDS['faster_whisper']['available'] = True
    TRANSCRIPTION_BACKENDS['faster_whisper']['real_available'] = True
    logger.info("Loaded real Faster Whisper backend")
except ImportError as e:
    logger.warning(f"Faster Whisper not available ({e}), will use mock implementation")
    # Import mock implementation
    try:
        from app.mock_backends import MockFasterWhisperModel as WhisperModel
        TRANSCRIPTION_BACKENDS['faster_whisper']['available'] = True
        logger.info("Loaded mock Faster Whisper backend")
    except ImportError as e2:
        logger.error(f"Failed to load mock Faster Whisper backend: {e2}")

try:
    if FORCE_MOCK:
        raise ImportError('Forced mock for whisper')
    import whisper
    TRANSCRIPTION_BACKENDS['whisper']['available'] = True
    TRANSCRIPTION_BACKENDS['whisper']['real_available'] = True
    logger.info("Loaded real OpenAI Whisper backend")
except ImportError as e:
    logger.warning(f"OpenAI Whisper not available ({e}), will use mock implementation")
    # Import mock implementation
    try:
        from app.mock_backends import MockWhisperModel as whisper
        # Mock the load_model method
        whisper.load_model = lambda model_size: whisper(model_size)
        TRANSCRIPTION_BACKENDS['whisper']['available'] = True
        logger.info("Loaded mock OpenAI Whisper backend")
    except ImportError as e2:
        logger.error(f"Failed to load mock OpenAI Whisper backend: {e2}")

try:
    if FORCE_MOCK:
        raise ImportError('Forced mock for pyannote')
    from pyannote.audio import Pipeline
    DIARIZATION_BACKENDS['pyannote']['available'] = True
    DIARIZATION_BACKENDS['pyannote']['real_available'] = True
    logger.info("Loaded real Pyannote backend")
except ImportError as e:
    logger.warning(f"Pyannote not available ({e}), will use mock implementation")
    # Import mock implementation
    try:
        from app.mock_backends import MockPyannotePipeline as Pipeline
        DIARIZATION_BACKENDS['pyannote']['available'] = True
        logger.info("Loaded mock Pyannote backend")
    except ImportError as e2:
        logger.error(f"Failed to load mock Pyannote backend: {e2}")

# Set up resemblyzer dependencies (real or mock)
resemblyzer_available = False
preprocess_wav = None
VoiceEncoder = None
librosa_load = None

try:
    if FORCE_MOCK:
        raise ImportError('Forced mock for resemblyzer')
    from resemblyzer import VoiceEncoder as RealVoiceEncoder, preprocess_wav as real_preprocess_wav
    import librosa
    resemblyzer_available = True
    VoiceEncoder = RealVoiceEncoder
    preprocess_wav = real_preprocess_wav
    librosa_load = librosa.load
    DIARIZATION_BACKENDS['resemblyzer']['available'] = True
    DIARIZATION_BACKENDS['resemblyzer']['real_available'] = True
    logger.info("Loaded real Resemblyzer backend")
except ImportError as e:
    logger.warning(f"Resemblyzer not available ({e}), will use mock implementation")
    # Import mock implementation
    try:
        from app.mock_backends import MockVoiceEncoder, mock_preprocess_wav, mock_librosa_load
        resemblyzer_available = True
        VoiceEncoder = MockVoiceEncoder
        preprocess_wav = mock_preprocess_wav
        librosa_load = mock_librosa_load
        DIARIZATION_BACKENDS['resemblyzer']['available'] = True
        logger.info("Loaded mock Resemblyzer backend")
    except ImportError as e2:
        logger.error(f"Failed to load mock Resemblyzer backend: {e2}")

# Log the backends status
logger.info("Transcription backends status:")
for name, info in TRANSCRIPTION_BACKENDS.items():
    real_status = "real" if info['real_available'] else "mock" if info['available'] else "unavailable"
    logger.info(f"  - {name}: {real_status}")

logger.info("Diarization backends status:")
for name, info in DIARIZATION_BACKENDS.items():
    real_status = "real" if info['real_available'] else "mock" if info['available'] else "unavailable"
    logger.info(f"  - {name}: {real_status}")

# ------------------------ PLUGIN SYSTEM ------------------------ #
class PluginRegistry:
    """Registry for audio processing pipeline plugins."""
    
    def __init__(self):
        self.audio_preprocessors = []
        self.transcription_plugins = []
        self.diarization_plugins = []
        self.post_processors = []
        self.output_formatters = []
    
    def register_audio_preprocessor(self, func):
        """Register an audio preprocessing function."""
        self.audio_preprocessors.append(func)
        return func
    
    def register_transcription_plugin(self, func):
        """Register a transcription plugin."""
        self.transcription_plugins.append(func)
        return func
    
    def register_diarization_plugin(self, func):
        """Register a speaker diarization plugin."""
        self.diarization_plugins.append(func)
        return func
    
    def register_post_processor(self, func):
        """Register a post-processing function."""
        self.post_processors.append(func)
        return func
    
    def register_output_formatter(self, func):
        """Register an output formatter."""
        self.output_formatters.append(func)
        return func

# Create global plugin registry
plugins = PluginRegistry()

# ------------------------ AUDIO UTILS ------------------------ #
def convert_audio_to_wav(input_path: str) -> str:
    """Convert audio file to WAV format using soundfile with proper error handling."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    
    try:
        # Read the audio file
        data, samplerate = sf.read(input_path)
        
        # Validate audio data
        if len(data) == 0:
            raise ValueError(f"Empty audio file: {input_path}")
        
        # Create a temporary file for the output
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=Config.AUDIO_CACHE_FOLDER) as temp:
            output_path = temp.name
        
        try:
            # Write to WAV format
            sf.write(output_path, data, samplerate)
            logger.info(f"Converted {input_path} to WAV format at {output_path}")
            return output_path
        except Exception as write_error:
            # Clean up temp file if write failed
            try:
                os.unlink(output_path)
            except OSError:
                pass
            raise write_error
            
    except (sf.SoundFileError, ValueError, OSError) as e:
        logger.error(f"Failed to convert audio to WAV: {e}")
        # Only return original path if it's a supported format
        if input_path.lower().endswith(('.wav', '.flac', '.ogg')):
            logger.warning(f"Using original file without conversion: {input_path}")
            return input_path
        else:
            raise ValueError(f"Unsupported audio format and conversion failed: {input_path}")

# ------------------------ TRANSCRIPTION ENGINE ------------------------ #
@plugins.register_transcription_plugin
def transcribe_with_faster_whisper(audio_path: str, model_size: str = Config.DEFAULT_MODEL_SIZE) -> List[Segment]:
    """Transcribe audio using Faster Whisper."""
    if not TRANSCRIPTION_BACKENDS['faster_whisper']['available']:
        raise ImportError("faster_whisper package is not installed")
    
    try:
        # Check if we should use mock backend
        if not TRANSCRIPTION_BACKENDS['faster_whisper']['real_available']:
            logger.info(f"Using mock Faster Whisper for {audio_path}")
            from mock_backends import MockFasterWhisperModel
            mock_model = MockFasterWhisperModel()
            segments, info = mock_model.transcribe(audio_path, beam_size=5)
        else:
            # Initialize the model if not already done
            if not TRANSCRIPTION_BACKENDS['faster_whisper']['model']:
                logger.info(f"Loading Faster Whisper model: {model_size}")
                
                # Determine device and compute type with CUDA support
                device = "cpu"
                compute_type = "int8"
                
                # Check if CUDA should be used (with environment override)
                use_cuda = os.environ.get('AUDIOTRANSCRIBE_USE_CUDA', 'auto').lower()
                
                if use_cuda == 'auto':
                    # Auto-detect: use CUDA if available and has sufficient memory
                    if CUDA_AVAILABLE and CUDA_DEVICE_COUNT > 0:
                        try:
                            # Check available GPU memory (need at least 2GB free for base model)
                            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                            if free_memory > 2 * 1024 * 1024 * 1024:  # 2GB
                                device = "cuda"
                                compute_type = "float16"  # Use float16 on GPU for better performance
                                logger.info(f"Using CUDA device with {free_memory / 1e9:.2f} GB free memory")
                            else:
                                logger.warning(f"Insufficient GPU memory ({free_memory / 1e9:.2f} GB free), using CPU")
                        except Exception as e:
                            logger.warning(f"Error checking GPU memory: {e}, falling back to CPU")
                elif use_cuda == 'true' or use_cuda == '1':
                    if CUDA_AVAILABLE:
                        device = "cuda"
                        compute_type = "float16"
                        logger.info("CUDA explicitly enabled via environment variable")
                    else:
                        logger.warning("CUDA requested but not available, falling back to CPU")
                else:
                    logger.info("Using CPU as specified")
                
                TRANSCRIPTION_BACKENDS['faster_whisper']['model'] = WhisperModel(
                    model_size, 
                    compute_type=compute_type, 
                    device=device
                )
                logger.info(f"Faster Whisper model loaded on {device} with compute_type={compute_type}")
            
            model = TRANSCRIPTION_BACKENDS['faster_whisper']['model']
            
            # Transcribe
            logger.info(f"Transcribing {audio_path} with Faster Whisper")
            segments, info = model.transcribe(audio_path, beam_size=5)
        
        # Convert to our segment format
        result = [
            Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                confidence=getattr(segment, 'confidence', 0.8)  # Use default confidence if not available
            ) for segment in segments
        ]
        
        logger.info(f"Transcription complete: {len(result)} segments, language: {info.language}")
        return result
    except Exception as e:
        logger.error(f"Faster Whisper transcription failed: {e}")
        # If transcription fails, return some dummy segments
        logger.warning("Returning fallback segments")
        return [
            Segment(start=0.0, end=2.0, text="Error during transcription.", confidence=0.5),
            Segment(start=2.0, end=4.0, text="Please check logs for details.", confidence=0.5),
        ]

@plugins.register_transcription_plugin
def transcribe_with_whisper(audio_path: str, model_size: str = Config.DEFAULT_MODEL_SIZE) -> List[Segment]:
    """Transcribe audio using OpenAI Whisper."""
    if not TRANSCRIPTION_BACKENDS['whisper']['available']:
        raise ImportError("whisper package is not installed")
    
    try:
        # Check if we should use mock backend
        if not TRANSCRIPTION_BACKENDS['whisper']['real_available']:
            logger.info(f"Using mock Whisper for {audio_path}")
            from mock_backends import MockWhisperModel
            mock_model = MockWhisperModel()
            result = mock_model.transcribe(audio_path)
        else:
            # Initialize the model if not already done
            if not TRANSCRIPTION_BACKENDS['whisper']['model']:
                logger.info(f"Loading Whisper model: {model_size}")
                TRANSCRIPTION_BACKENDS['whisper']['model'] = whisper.load_model(model_size)
            
            model = TRANSCRIPTION_BACKENDS['whisper']['model']
            
            # Transcribe
            logger.info(f"Transcribing {audio_path} with Whisper")
            result = model.transcribe(audio_path)
        
        # Convert to our segment format
        segments = []
        for segment in result["segments"]:
            segments.append(
                Segment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"].strip(),
                    confidence=segment["confidence"]
                )
            )
        
        logger.info(f"Transcription complete: {len(segments)} segments, language: {result['language']}")
        return segments
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        # If transcription fails, return some dummy segments
        logger.warning("Returning fallback segments")
        return [
            Segment(start=0.0, end=2.0, text="Error during transcription.", confidence=0.5),
            Segment(start=2.0, end=4.0, text="Please check logs for details.", confidence=0.5),
        ]

def transcribe_audio(audio_path: str, backend: str = Config.DEFAULT_TRANSCRIPTION_BACKEND, 
                    model_size: str = Config.DEFAULT_MODEL_SIZE) -> List[Segment]:
    """Transcribe audio using the selected backend."""
    if backend == 'faster_whisper' and TRANSCRIPTION_BACKENDS['faster_whisper']['available']:
        return transcribe_with_faster_whisper(audio_path, model_size)
    elif backend == 'whisper' and TRANSCRIPTION_BACKENDS['whisper']['available']:
        return transcribe_with_whisper(audio_path, model_size)
    else:
        # Find the first available backend
        for name, info in TRANSCRIPTION_BACKENDS.items():
            if info['available']:
                logger.warning(f"Requested backend {backend} not available, using {name} instead")
                if name == 'faster_whisper':
                    return transcribe_with_faster_whisper(audio_path, model_size)
                elif name == 'whisper':
                    return transcribe_with_whisper(audio_path, model_size)
        
        # If no backends available, return fallback segments
        logger.error("No transcription backend available")
        return [
            Segment(start=0.0, end=2.0, text="No transcription backend available.", confidence=0.5),
            Segment(start=2.0, end=4.0, text="Please install either whisper or faster_whisper.", confidence=0.5),
        ]

# ------------------------ SPEAKER DIARIZATION ------------------------ #
@plugins.register_diarization_plugin
def diarize_with_pyannote(audio_path: str) -> List[Tuple[float, float, str]]:
    """Perform speaker diarization using pyannote."""
    if not DIARIZATION_BACKENDS['pyannote']['available']:
        raise ImportError("pyannote.audio package is not installed")
    
    try:
        # Initialize the model if not already done
        if not DIARIZATION_BACKENDS['pyannote']['model']:
            logger.info("Loading pyannote speaker diarization model")
            # Get Hugging Face token from environment
            hf_token = os.environ.get("HF_TOKEN", "hf_dummy_token_for_mock")
            if not hf_token and DIARIZATION_BACKENDS['pyannote']['real_available']:
                logger.warning("HF_TOKEN environment variable not set. Pyannote may fail to download models.")
            
            DIARIZATION_BACKENDS['pyannote']['model'] = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                token=hf_token,
                use_auth_token=hf_token  # Fallback for older versions
            )
        
        pipeline = DIARIZATION_BACKENDS['pyannote']['model']
        
        # Process the audio
        logger.info(f"Running speaker diarization on {audio_path} with pyannote")
        diarization = pipeline(audio_path)
        
        # Extract speaker turns
        speaker_turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_turns.append((turn.start, turn.end, speaker))
        
        logger.info(f"Diarization complete: {len(speaker_turns)} speaker turns")
        return speaker_turns
    except Exception as e:
        logger.error(f"Pyannote diarization failed: {e}")
        # Return fallback speaker turns
        logger.warning("Returning fallback speaker turns")
        return [
            (0.0, 2.0, "SPEAKER_0"),
            (2.0, 4.0, "SPEAKER_1"),
            (4.0, 6.0, "SPEAKER_0"),
        ]

@plugins.register_diarization_plugin
def diarize_with_resemblyzer(audio_path: str) -> List[Tuple[float, float, str]]:
    """Perform speaker diarization using resemblyzer with optimized sliding window and clustering."""
    if not DIARIZATION_BACKENDS['resemblyzer']['available']:
        raise ImportError("resemblyzer package is not installed")
    
    try:
        # Load audio
        y, sr = librosa_load(audio_path, sr=16000, mono=True)
        audio_duration = len(y) / sr
        
        # Initialize the model if not already done
        if not DIARIZATION_BACKENDS['resemblyzer']['model']:
            logger.info("Loading resemblyzer voice encoder model")
            device = "cpu"  # Default to CPU
            
            # Check CUDA availability with safeguards
            use_cuda = os.environ.get('AUDIOTRANSCRIBE_USE_CUDA', 'auto').lower()
            if (use_cuda == 'auto' and CUDA_AVAILABLE) or (use_cuda in ('true', '1') and CUDA_AVAILABLE):
                try:
                    # Resemblyzer needs less memory, so we can be more lenient
                    device = "cuda"
                    logger.info("Using CUDA for Resemblyzer")
                except Exception:
                    logger.warning("CUDA initialization failed for Resemblyzer, using CPU")
                    device = "cpu"
            
            DIARIZATION_BACKENDS['resemblyzer']['model'] = VoiceEncoder(device=device)
        
        encoder = DIARIZATION_BACKENDS['resemblyzer']['model']
        
        # Optimized parameters: adaptive windowing based on audio length
        if audio_duration < 60:  # Short audio
            window_size = 2.0  # 2 seconds
            step_size = 0.5     # 0.5 second overlap
        elif audio_duration < 300:  # Medium audio
            window_size = 3.0  # 3 seconds
            step_size = 1.0     # 1 second overlap
        else:  # Long audio
            window_size = 4.0  # 4 seconds
            step_size = 2.0    # 2 second overlap
        
        max_clusters = min(8, max(2, int(audio_duration / 60)))  # Adaptive cluster count
        
        logger.info(f"Optimized windowing: window={window_size}s, step={step_size}s, max_clusters={max_clusters}")
        
        # Process in optimized sliding windows with batch processing
        logger.info(f"Running speaker diarization on {audio_path} with resemblyzer")
        
        # Collect all embeddings first for batch processing
        embeddings = []
        time_ranges = []
        
        window_samples = int(window_size * sr)
        step_samples = int(step_size * sr)
        
        for i in range(0, len(y) - window_samples, step_samples):
            start_time = i / sr
            end_time = (i + window_samples) / sr
            
            # Get audio segment
            segment = y[i:i + window_samples]
            
            # Preprocess and get embedding
            segment_preprocessed = preprocess_wav(segment, source_sr=sr)
            embedding = encoder.embed_utterance(segment_preprocessed)
            
            embeddings.append(embedding)
            time_ranges.append((start_time, end_time))
        
        # Clustering using optimized similarity computation
        speaker_turns = []
        if embeddings:
            # Use KMeans-like clustering for better speaker assignment
            from sklearn.cluster import KMeans
            try:
                embeddings_array = np.array(embeddings)
                n_clusters = min(max_clusters, len(embeddings))
                
                # Normalize embeddings for better clustering
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                embeddings_normalized = embeddings_array / (norms + 1e-8)
                
                # Cluster embeddings
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings_normalized)
                
                # Map clusters to speaker IDs
                speaker_ids = {label: f"SPEAKER_{i}" for i, label in enumerate(sorted(set(cluster_labels)))}
                
                # Create speaker turns
                for (start, end), label in zip(time_ranges, cluster_labels):
                    speaker_turns.append((start, end, speaker_ids[label]))
                    
            except ImportError:
                # Fallback to simple similarity-based clustering if sklearn not available
                logger.warning("sklearn not available, using simple similarity clustering")
                current_speaker = None
                prev_embedding = None
                speaker_counter = 0
                speaker_mapping = {}
                
                for (start, end), embedding in zip(time_ranges, embeddings):
                    if current_speaker is None:
                        current_speaker = f"SPEAKER_{speaker_counter}"
                        speaker_mapping[current_speaker] = embedding
                        speaker_counter += 1
                    else:
                        # Vectorized similarity computation
                        similarities = np.array([
                            np.dot(embedding, prev_emb) / (
                                np.linalg.norm(embedding) * np.linalg.norm(prev_emb) + 1e-8
                            )
                            for prev_emb in speaker_mapping.values()
                        ])
                        max_sim = np.max(similarities)
                        max_idx = np.argmax(similarities)
                        
                        if max_sim < 0.7:  # Threshold for new speaker
                            current_speaker = f"SPEAKER_{speaker_counter}"
                            speaker_mapping[current_speaker] = embedding
                            speaker_counter += 1
                        else:
                            # Use existing speaker
                            current_speaker = list(speaker_mapping.keys())[max_idx]
                    
                    speaker_turns.append((start, end, current_speaker))
                    prev_embedding = embedding
        
        logger.info(f"Resemblyzer diarization complete: {len(speaker_turns)} speaker turns")
        return speaker_turns
    except Exception as e:
        logger.error(f"Resemblyzer diarization failed: {e}")
        # Return fallback speaker turns
        logger.warning("Returning fallback speaker turns")
        return [
            (0.0, 2.0, "SPEAKER_A"),
            (2.0, 4.0, "SPEAKER_B"),
            (4.0, 6.0, "SPEAKER_A"),
        ]

def diarize_speakers(audio_path: str, backend: str = Config.DEFAULT_DIARIZATION_BACKEND) -> List[Tuple[float, float, str]]:
    """Diarize speakers using the selected backend."""
    if backend == 'pyannote' and DIARIZATION_BACKENDS['pyannote']['available']:
        return diarize_with_pyannote(audio_path)
    elif backend == 'resemblyzer' and DIARIZATION_BACKENDS['resemblyzer']['available']:
        return diarize_with_resemblyzer(audio_path)
    else:
        # Find the first available backend
        for name, info in DIARIZATION_BACKENDS.items():
            if info['available']:
                logger.warning(f"Requested backend {backend} not available, using {name} instead")
                if name == 'pyannote':
                    return diarize_with_pyannote(audio_path)
                elif name == 'resemblyzer':
                    return diarize_with_resemblyzer(audio_path)
        
        # If no backends available, return fallback speaker turns
        logger.error("No diarization backend available")
        return [
            (0.0, 2.0, "SPEAKER_Default_1"),
            (2.0, 4.0, "SPEAKER_Default_2"),
            (4.0, 6.0, "SPEAKER_Default_1"),
        ]
# ------------------------ SPEAKER IDENTIFICATION ------------------------ #
# Embedding cache for known speakers to avoid recomputation
_embedding_cache: Dict[str, np.ndarray] = {}

def _get_cached_speaker_embedding(speaker_id: str, embeddings_list: List[List[float]]) -> Optional[np.ndarray]:
    """Get cached embedding for a speaker or compute and cache it."""
    cache_key = f"{speaker_id}_{hash(str(embeddings_list))}"
    
    if cache_key in _embedding_cache:
        logger.debug(f"Using cached embedding for speaker {speaker_id}")
        return _embedding_cache[cache_key]
    
    if embeddings_list:
        avg_embedding = np.mean(np.array(embeddings_list), axis=0)
        _embedding_cache[cache_key] = avg_embedding
        return avg_embedding
    
    return None

def extract_speaker_embeddings(audio_path: str, speaker_turns: List[Tuple[float, float, str]]) -> Dict[str, List[List[float]]]:
    """Extract voice embeddings for each speaker segment with robust error handling."""
    if not DIARIZATION_BACKENDS['resemblyzer']['available']:
        logger.warning("Resemblyzer not available, skipping speaker embedding extraction")
        return {}
    
    if not speaker_turns:
        logger.warning("No speaker turns provided for embedding extraction")
        return {}
    
    try:
        # Load audio with error handling
        try:
            y, sr = librosa_load(audio_path, sr=16000, mono=True)
        except Exception as load_error:
            logger.error(f"Failed to load audio file {audio_path}: {load_error}")
            return {}
        
        # Validate audio data
        if len(y) == 0:
            logger.error(f"Empty audio file: {audio_path}")
            return {}
        
        # Initialize encoder if needed
        if not DIARIZATION_BACKENDS['resemblyzer']['model']:
            try:
                # Force CPU-only processing to avoid CUDA issues
                device = "cpu"
                DIARIZATION_BACKENDS['resemblyzer']['model'] = VoiceEncoder(device=device)
            except Exception as encoder_error:
                logger.error(f"Failed to initialize VoiceEncoder: {encoder_error}")
                return {}
        
        encoder = DIARIZATION_BACKENDS['resemblyzer']['model']
        
        # Process each speaker turn with bounds checking
        embeddings = {}
        for start, end, speaker in speaker_turns:
            try:
                # Validate time bounds
                if start < 0 or end <= start:
                    logger.warning(f"Invalid time segment: {start}-{end} for speaker {speaker}")
                    continue
                
                # Convert times to samples with bounds checking
                start_sample = max(0, int(start * sr))
                end_sample = min(len(y), int(end * sr))
                
                # Skip segments that are too short or invalid
                if end_sample - start_sample < sr:  # Less than 1 second
                    logger.debug(f"Skipping short segment ({end-start:.2f}s) for speaker {speaker}")
                    continue
                
                # Extract audio segment
                segment = y[start_sample:end_sample]
                
                # Preprocess and get embedding
                segment_preprocessed = preprocess_wav(segment, source_sr=sr)
                embedding = encoder.embed_utterance(segment_preprocessed)
                
                # Store embedding
                if speaker not in embeddings:
                    embeddings[speaker] = []
                
                embeddings[speaker].append(embedding.tolist())
                
            except Exception as segment_error:
                logger.warning(f"Failed to extract embedding for speaker {speaker} segment {start}-{end}: {segment_error}")
                continue
        
        logger.info(f"Extracted embeddings for {len(embeddings)} speakers")
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to extract speaker embeddings: {e}")
        return {}

def identify_speakers(audio_path: str, speaker_turns: List[Tuple[float, float, str]]) -> Dict[str, str]:
    """Identify speakers by matching embeddings with known speakers using cached embeddings."""
    # Extract embeddings for current speakers
    current_embeddings = extract_speaker_embeddings(audio_path, speaker_turns)
    if not current_embeddings:
        return {}
    
    # Get all known speakers with embeddings (cached)
    known_speakers = get_all_speakers_with_embeddings()
    if not known_speakers:
        logger.info("No known speakers in database for identification")
        return {}
    
    # Pre-compute and cache known speaker embeddings for efficient comparison
    known_speaker_embeddings = []
    known_speaker_names = []
    
    for speaker in known_speakers:
        if not speaker.embeddings:
            continue
        
        # Use cached embedding if available
        cached_emb = _get_cached_speaker_embedding(speaker.id, speaker.embeddings)
        if cached_emb is not None:
            known_speaker_embeddings.append(cached_emb)
            known_speaker_names.append(speaker.name)
        else:
            avg_embedding = np.mean(np.array(speaker.embeddings), axis=0)
            known_speaker_embeddings.append(avg_embedding)
            known_speaker_names.append(speaker.name)
    
    if not known_speaker_embeddings:
        return {}
    
    # Convert to numpy array for vectorized operations
    known_embeddings_array = np.array(known_speaker_embeddings)
    
    # Match current speakers to known speakers using vectorized similarity computation
    speaker_mapping = {}
    
    for current_id, embeddings in current_embeddings.items():
        if not embeddings:
            continue
        
        # Get or compute average embedding for current speaker
        avg_embedding = _get_cached_speaker_embedding(current_id, embeddings)
        if avg_embedding is None:
            avg_embedding = np.mean(np.array(embeddings), axis=0)
            _embedding_cache[f"{current_id}_{hash(str(embeddings))}"] = avg_embedding
        
        # Vectorized similarity computation (much faster than loop)
        # Normalize embeddings
        avg_embedding_norm = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        known_norms = np.linalg.norm(known_embeddings_array, axis=1, keepdims=True)
        known_normalized = known_embeddings_array / (known_norms + 1e-8)
        
        # Compute cosine similarities in batch
        similarities = np.dot(known_normalized, avg_embedding_norm)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity > Config.SPEAKER_SIMILARITY_THRESHOLD:
            speaker_mapping[current_id] = known_speaker_names[best_idx]
            logger.info(f"Matched speaker {current_id} to known speaker {known_speaker_names[best_idx]} with similarity {best_similarity:.3f}")
    
    return speaker_mapping

def align_segments_with_speakers(segments: List[Segment], 
                               speaker_turns: List[Tuple[float, float, str]], 
                               speaker_mapping: Dict[str, str] = None) -> List[Segment]:
    """Align transcription segments with identified speakers."""
    # If no speaker turns, return original segments
    if not speaker_turns:
        return segments
    
    # Default empty mapping if none provided
    if speaker_mapping is None:
        speaker_mapping = {}
    
    # For each segment, find the overlapping speaker turn
    for segment in segments:
        segment_midpoint = (segment.start + segment.end) / 2
        segment_speaker = None
        
        for start, end, speaker_id in speaker_turns:
            if segment_midpoint >= start and segment_midpoint <= end:
                # Map speaker ID to name if available
                speaker_name = speaker_mapping.get(speaker_id, f"Speaker {speaker_id.split('_')[-1]}")
                segment.speaker = speaker_name
                break
    
    return segments

def save_new_speakers(audio_path: str, speaker_turns: List[Tuple[float, float, str]], speaker_mapping: Dict[str, str]):
    """Save new speakers with their embeddings to the database."""
    # Extract speaker embeddings
    embeddings = extract_speaker_embeddings(audio_path, speaker_turns)
    if not embeddings:
        return
    
    # For each speaker, save to database if not already mapped
    for speaker_id, speaker_embeddings in embeddings.items():
        if speaker_id not in speaker_mapping and speaker_embeddings:
            # This is a new speaker
            display_name = f"Speaker {speaker_id.split('_')[-1]}"
            
            # Create and save the speaker
            speaker = Speaker(
                id=speaker_id,
                name=display_name,
                embeddings=speaker_embeddings
            )
            
            save_speaker(speaker)
            logger.info(f"Saved new speaker {display_name} to database")

# ------------------------ PROCESSING PIPELINE ------------------------ #
def process_audio_file(audio_path: str, model_size: str = Config.DEFAULT_MODEL_SIZE,
                      transcription_backend: str = Config.DEFAULT_TRANSCRIPTION_BACKEND,
                      diarization_backend: str = Config.DEFAULT_DIARIZATION_BACKEND) -> ProcessingResult:
    """Process an audio file through the complete pipeline."""
    job_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    try:
        # Create transcription job record
        job = TranscriptionJob(
            id=job_id,
            file_path=audio_path,
            file_hash=hash_file(audio_path),
            state=TranscriptionState.PROCESSING,
            timestamp=timestamp,
            model_size=model_size,
            transcription_backend=transcription_backend,
            diarization_backend=diarization_backend
        )
        save_job(job)
        
        # DISABLED: Check if we already processed this file
        # This functionality has been disabled to allow for retrying
        # existing_job = get_job_by_hash(job.file_hash)
        # if existing_job:
        #     logger.info(f"Found existing job for file hash {job.file_hash}, reusing results")
        #     # Copy segments from existing job
        #     job.segments = existing_job.segments
        #     job.state = TranscriptionState.COMPLETED
        #     job.output_path = generate_transcript_html(job)
        #     save_job(job)
        #     return ProcessingResult(
        #         success=True,
        #         message=f"Reused existing transcription for {os.path.basename(audio_path)}",
        #         job=job
        #     )
        
        # 1. Convert audio to WAV if needed
        working_audio_path = audio_path
        temp_files_to_cleanup = []
        
        if not working_audio_path.lower().endswith('.wav'):
            try:
                working_audio_path = convert_audio_to_wav(working_audio_path)
                if working_audio_path != audio_path:
                    temp_files_to_cleanup.append(working_audio_path)
            except Exception as conversion_error:
                job.state = TranscriptionState.FAILED
                job.error = f"Audio conversion failed: {conversion_error}"
                save_job(job)
                return ProcessingResult(
                    success=False,
                    message=f"Failed to convert audio file: {conversion_error}",
                    job=job,
                    error=conversion_error
                )
        
        # 2. Run audio preprocessing plugins with isolation
        for plugin in plugins.audio_preprocessors:
            try:
                result_path = plugin(working_audio_path)
                if result_path != working_audio_path:
                    # Clean up previous temp file if it was created by plugin
                    if working_audio_path in temp_files_to_cleanup:
                        try:
                            os.unlink(working_audio_path)
                            temp_files_to_cleanup.remove(working_audio_path)
                        except OSError:
                            pass
                    working_audio_path = result_path
                    temp_files_to_cleanup.append(working_audio_path)
            except Exception as e:
                logger.warning(f"Audio preprocessor {plugin.__name__} failed: {e}")
                # Continue with other plugins
        
        # 3. Transcribe audio
        segments = transcribe_audio(working_audio_path, transcription_backend, model_size)
        
        # 4. Diarize speakers
        speaker_turns = diarize_speakers(working_audio_path, diarization_backend)
        
        # 5. Identify speakers
        speaker_mapping = identify_speakers(working_audio_path, speaker_turns)
        
        # 6. Align segments with speakers
        segments = align_segments_with_speakers(segments, speaker_turns, speaker_mapping)
        
        # 7. Save new speakers for future identification
        save_new_speakers(working_audio_path, speaker_turns, speaker_mapping)
        
        # 8. Run post-processing plugins
        for plugin in plugins.post_processors:
            try:
                segments = plugin(segments)
            except Exception as e:
                logger.warning(f"Post-processor {plugin.__name__} failed: {e}")
        
        # 9. Update job with results
        job.segments = segments
        job.state = TranscriptionState.COMPLETED
        
        # 10. Generate output file
        try:
            job.output_path = generate_transcript_html(job)
        except Exception as html_error:
            logger.error(f"Failed to generate HTML output: {html_error}")
            # Continue without HTML output
        
        # 11. Save job to database
        save_job(job)
        
        # 12. Clean up temporary files
        for temp_file in temp_files_to_cleanup:
            try:
                os.unlink(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            except OSError as cleanup_error:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {cleanup_error}")
        
        return ProcessingResult(
            success=True, 
            message=f"Successfully processed {os.path.basename(audio_path)}",
            job=job
        )
    except Exception as e:
        logger.exception(f"Processing failed for {audio_path}: {e}")
        
        # Update job status to failed
        try:
            # Try to update the existing job object first
            if 'job' in locals() and job:
                job.state = TranscriptionState.FAILED
                job.error = str(e)
                save_job(job)
            else:
                # Fallback: retrieve from database
                job = get_job(job_id)
                if job:
                    job.state = TranscriptionState.FAILED
                    job.error = str(e)
                    save_job(job)
        except Exception as db_error:
            logger.error(f"Failed to update job status: {db_error}")
        
        # Clean up any temporary files
        if 'temp_files_to_cleanup' in locals():
            for temp_file in temp_files_to_cleanup:
                try:
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temporary file after error: {temp_file}")
                except OSError:
                    pass
        
        return ProcessingResult(
            success=False,
            message=f"Failed to process {os.path.basename(audio_path)}: {str(e)}",
            error=e
        )

# ------------------------ OUTPUT GENERATION ------------------------ #
def generate_transcript_html(job: TranscriptionJob) -> str:
    """Generate HTML transcript file."""
    try:
        # Create filename from job ID
        filename = f"transcript_{job.id}.html"
        output_path = os.path.join(Config.TRANSCRIPT_FOLDER, filename)
        
        # Format segments for display
        segments_data = []
        for segment in job.segments:
            segments_data.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'speaker': segment.speaker,
                'formatted_start': format_timestamp(segment.start),
                'formatted_end': format_timestamp(segment.end)
            })
        
        # Render HTML template
        html_template = Template("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Audio Transcript</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .header {
                    border-bottom: 1px solid #eee;
                    padding-bottom: 20px;
                    margin-bottom: 20px;
                }
                .segment {
                    margin-bottom: 15px;
                    padding: 10px;
                    border-radius: 5px;
                    background-color: #f9f9f9;
                }
                .segment:hover {
                    background-color: #f0f0f0;
                }
                .timestamp {
                    color: #888;
                    font-size: 0.8em;
                    cursor: pointer;
                }
                .speaker {
                    font-weight: bold;
                    color: #444;
                }
                .speaker-color-0 { color: #1f77b4; }
                .speaker-color-1 { color: #ff7f0e; }
                .speaker-color-2 { color: #2ca02c; }
                .speaker-color-3 { color: #d62728; }
                .speaker-color-4 { color: #9467bd; }
                .speaker-color-5 { color: #8c564b; }
                .speaker-color-6 { color: #e377c2; }
                .speaker-color-7 { color: #7f7f7f; }
                .meta {
                    font-size: 0.8em;
                    color: #888;
                    margin-top: 30px;
                    border-top: 1px solid #eee;
                    padding-top: 10px;
                }
                audio {
                    width: 100%;
                    margin: 20px 0;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Audio Transcript</h1>
                <p>Generated: {{ timestamp }}</p>
                <p>File hash: {{ file_hash }}</p>
            </div>
            
            <div id="transcript">
                {% for segment in segments %}
                <div class="segment" data-start="{{ segment.start }}" data-end="{{ segment.end }}">
                    <div class="timestamp" onclick="seekAudio({{ segment.start }})">
                        {{ segment.formatted_start }} - {{ segment.formatted_end }}
                    </div>
                    <div>
                        <span class="speaker speaker-color-{{ loop.index % 8 }}">{{ segment.speaker }}:</span>
                        {{ segment.text }}
                    </div>
                </div>
                {% endfor %}
            </div>
            
            <div class="meta">
                <p>Processed with AudioTranscribe</p>
                <p>Transcription model: {{ model_size }}</p>
                <p>Job ID: {{ job_id }}</p>
            </div>
            
            <script>
                function seekAudio(time) {
                    const audioElement = document.querySelector('audio');
                    if (audioElement) {
                        audioElement.currentTime = time;
                        audioElement.play();
                    }
                }
            </script>
        </body>
        </html>
        """)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_template.render(
                segments=segments_data,
                timestamp=job.timestamp,
                file_hash=job.file_hash,
                job_id=job.id,
                model_size=job.model_size
            ))
        
        logger.info(f"Transcript HTML generated: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate HTML transcript: {e}")
        raise

@plugins.register_output_formatter
def generate_json_output(job: TranscriptionJob) -> str:
    """Generate JSON output file."""
    try:
        # Create filename from job ID
        filename = f"transcript_{job.id}.json"
        output_path = os.path.join(Config.TRANSCRIPT_FOLDER, filename)
        
        # Prepare data
        output_data = {
            'job_id': job.id,
            'timestamp': job.timestamp,
            'file_hash': job.file_hash,
            'segments': [segment.to_dict() for segment in job.segments],
            'model_size': job.model_size,
            'transcription_backend': job.transcription_backend,
            'diarization_backend': job.diarization_backend
        }
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"JSON output generated: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate JSON output: {e}")
        raise

@plugins.register_output_formatter
def generate_srt_subtitle(job: TranscriptionJob) -> str:
    """Generate SRT subtitle file."""
    try:
        # Create filename from job ID
        filename = f"transcript_{job.id}.srt"
        output_path = os.path.join(Config.TRANSCRIPT_FOLDER, filename)
        
        def format_srt_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = seconds % 60
            milliseconds = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
        
        # Generate SRT content
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(job.segments, 1):
                f.write(f"{i}\n")
                f.write(f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}\n")
                f.write(f"{segment.speaker}: {segment.text}\n\n")
        
        logger.info(f"SRT subtitle generated: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate SRT subtitle: {e}")
        raise

# ------------------------ PLUGIN EXAMPLES ------------------------ #
@plugins.register_audio_preprocessor
def normalize_audio(audio_path: str) -> str:
    """Normalize audio volume levels."""
    # Placeholder for a real implementation
    logger.info(f"Audio preprocessing: Normalizing {audio_path}")
    return audio_path

@plugins.register_post_processor
def filter_empty_segments(segments: List[Segment]) -> List[Segment]:
    """Filter out empty segments."""
    filtered = [s for s in segments if s.text.strip()]
    logger.info(f"Post-processing: Filtered {len(segments) - len(filtered)} empty segments")
    return filtered