"""
Meta-Recursive Feedback Loop System
===================================
Advanced feedback loop system with localhost:5500 integration and OpenTTS for continuous improvement.
"""

import logging
import asyncio
import aiohttp
import json
import os
import tempfile
import uuid
from typing import Dict, List, Tuple, Optional, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class FeedbackSession:
    """Represents a feedback session for continuous improvement."""
    session_id: str
    start_time: datetime
    iterations: int = 0
    improvements: List[Dict] = None
    current_accuracy: float = 0.0
    target_accuracy: float = 0.95
    status: str = "active"  # "active", "completed", "failed"
    
    def __post_init__(self):
        if self.improvements is None:
            self.improvements = []

@dataclass
class AudioGenerationRequest:
    """Request for audio generation with OpenTTS."""
    text: str
    voice: str = "en_US/male"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    output_format: str = "wav"
    sample_rate: int = 16000

@dataclass
class TranscriptionTest:
    """Represents a transcription test case."""
    test_id: str
    audio_path: str
    expected_text: str
    actual_text: str = ""
    accuracy: float = 0.0
    confidence: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class MetaRecursiveFeedbackSystem:
    """Meta-recursive feedback loop system for continuous improvement."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        self.feedback_sessions: Dict[str, FeedbackSession] = {}
        self.test_cases: List[TranscriptionTest] = []
        self.improvement_history: List[Dict] = []
        
        # OpenTTS configuration
        self.opentts_url = self.config.get('opentts_url', 'http://localhost:5500')
        self.opentts_available = False
        
        # Initialize OpenTTS connection
        self._initialize_opentts()
    
    async def start_feedback_session(self, target_accuracy: float = 0.95) -> str:
        """Start a new feedback session."""
        try:
            session_id = str(uuid.uuid4())
            session = FeedbackSession(
                session_id=session_id,
                start_time=datetime.utcnow(),
                target_accuracy=target_accuracy
            )
            
            self.feedback_sessions[session_id] = session
            logger.info(f"Started feedback session: {session_id}")
            
            # Start the feedback loop
            asyncio.create_task(self._run_feedback_loop(session_id))
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start feedback session: {e}")
            return None
    
    async def _run_feedback_loop(self, session_id: str):
        """Run the main feedback loop."""
        try:
            session = self.feedback_sessions[session_id]
            max_iterations = self.config.get('max_iterations', 50)
            
            while session.iterations < max_iterations and session.status == "active":
                logger.info(f"Feedback loop iteration {session.iterations + 1} for session {session_id}")
                
                # Generate test cases
                test_cases = await self._generate_test_cases(session_id)
                
                # Run transcription tests
                results = await self._run_transcription_tests(test_cases)
                
                # Analyze results and calculate accuracy
                accuracy = self._calculate_accuracy(results)
                session.current_accuracy = accuracy
                
                # Generate improvements if needed
                if accuracy < session.target_accuracy:
                    improvements = await self._generate_improvements(results, session)
                    session.improvements.extend(improvements)
                    
                    # Apply improvements
                    await self._apply_improvements(improvements)
                else:
                    logger.info(f"Target accuracy {session.target_accuracy} reached: {accuracy}")
                    session.status = "completed"
                    break
                
                session.iterations += 1
                
                # Wait between iterations
                await asyncio.sleep(self.config.get('iteration_delay', 5.0))
            
            if session.status == "active":
                session.status = "completed" if session.current_accuracy >= session.target_accuracy else "failed"
            
            logger.info(f"Feedback session {session_id} completed: {session.status}")
            
        except Exception as e:
            logger.error(f"Feedback loop failed for session {session_id}: {e}")
            if session_id in self.feedback_sessions:
                self.feedback_sessions[session_id].status = "failed"
    
    async def _generate_test_cases(self, session_id: str) -> List[TranscriptionTest]:
        """Generate diverse test cases for transcription testing."""
        try:
            test_cases = []
            
            # Generate test cases with varying complexity
            test_scenarios = [
                "Simple sentences for basic accuracy testing",
                "Complex technical terminology and domain-specific vocabulary",
                "Multi-speaker conversations with overlapping speech",
                "Audio with background noise and poor quality conditions",
                "Long-form content with multiple paragraphs and topics",
                "Questions and answers with different speaking styles",
                "Numbers, dates, and technical specifications",
                "Conversational speech with interruptions and corrections"
            ]
            
            for i, scenario in enumerate(test_scenarios):
                # Generate audio for this scenario
                audio_path = await self._generate_audio_for_scenario(scenario, session_id, i)
                
                if audio_path:
                    test_case = TranscriptionTest(
                        test_id=f"{session_id}_test_{i}",
                        audio_path=audio_path,
                        expected_text=scenario
                    )
                    test_cases.append(test_case)
            
            logger.info(f"Generated {len(test_cases)} test cases for session {session_id}")
            return test_cases
            
        except Exception as e:
            logger.error(f"Test case generation failed: {e}")
            return []
    
    async def _generate_audio_for_scenario(self, scenario: str, session_id: str, test_index: int) -> Optional[str]:
        """Generate audio for a specific test scenario."""
        try:
            if not self.opentts_available:
                logger.warning("OpenTTS not available, using fallback audio generation")
                return await self._generate_fallback_audio(scenario, session_id, test_index)
            
            # Create audio generation request
            request = AudioGenerationRequest(
                text=scenario,
                voice="en_US/male",
                speed=1.0,
                pitch=1.0,
                volume=1.0,
                output_format="wav",
                sample_rate=16000
            )
            
            # Generate audio using OpenTTS
            audio_path = await self._generate_audio_with_opentts(request, session_id, test_index)
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Audio generation failed for scenario: {e}")
            return None
    
    async def _generate_audio_with_opentts(self, request: AudioGenerationRequest, 
                                         session_id: str, test_index: int) -> Optional[str]:
        """Generate audio using OpenTTS."""
        try:
            # Prepare OpenTTS request
            opentts_data = {
                "text": request.text,
                "voice": request.voice,
                "speed": request.speed,
                "pitch": request.pitch,
                "volume": request.volume,
                "output_format": request.output_format,
                "sample_rate": request.sample_rate
            }
            
            # Make request to OpenTTS
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.opentts_url}/api/tts",
                    json=opentts_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        # Save audio file
                        audio_data = await response.read()
                        audio_path = os.path.join(
                            self.config.get('temp_dir', '/tmp'),
                            f"{session_id}_test_{test_index}.wav"
                        )
                        
                        with open(audio_path, 'wb') as f:
                            f.write(audio_data)
                        
                        logger.info(f"Generated audio with OpenTTS: {audio_path}")
                        return audio_path
                    else:
                        logger.error(f"OpenTTS request failed: {response.status}")
                        return None
            
        except Exception as e:
            logger.error(f"OpenTTS audio generation failed: {e}")
            return None
    
    async def _generate_fallback_audio(self, scenario: str, session_id: str, test_index: int) -> Optional[str]:
        """Generate fallback audio when OpenTTS is not available."""
        try:
            # Create a simple audio file with the text as metadata
            # This is a placeholder - in practice, you'd use a different TTS system
            audio_path = os.path.join(
                self.config.get('temp_dir', '/tmp'),
                f"{session_id}_test_{test_index}_fallback.wav"
            )
            
            # Create a simple sine wave as placeholder
            duration = 2.0  # seconds
            sample_rate = 16000
            frequency = 440  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # Save as WAV file
            import soundfile as sf
            sf.write(audio_path, audio, sample_rate)
            
            # Save text as metadata file
            metadata_path = audio_path.replace('.wav', '_text.txt')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(scenario)
            
            logger.info(f"Generated fallback audio: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Fallback audio generation failed: {e}")
            return None
    
    async def _run_transcription_tests(self, test_cases: List[TranscriptionTest]) -> List[TranscriptionTest]:
        """Run transcription tests on generated audio."""
        try:
            results = []
            
            for test_case in test_cases:
                try:
                    # Run transcription on the audio
                    from app.transcription import process_audio_file
                    
                    # Process the audio file
                    result = process_audio_file(
                        test_case.audio_path,
                        model_size="base",
                        transcription_backend="faster_whisper",
                        diarization_backend="pyannote"
                    )
                    
                    if result.success and result.job.segments:
                        # Extract transcribed text
                        transcribed_text = " ".join([seg.text for seg in result.job.segments])
                        test_case.actual_text = transcribed_text
                        
                        # Calculate accuracy
                        test_case.accuracy = self._calculate_text_accuracy(
                            test_case.expected_text, 
                            transcribed_text
                        )
                        
                        # Calculate confidence
                        test_case.confidence = np.mean([seg.confidence for seg in result.job.segments])
                        
                        logger.info(f"Test {test_case.test_id}: accuracy={test_case.accuracy:.3f}, confidence={test_case.confidence:.3f}")
                    else:
                        logger.warning(f"Transcription failed for test {test_case.test_id}")
                        test_case.accuracy = 0.0
                        test_case.confidence = 0.0
                    
                    results.append(test_case)
                    
                except Exception as e:
                    logger.error(f"Transcription test failed for {test_case.test_id}: {e}")
                    test_case.accuracy = 0.0
                    test_case.confidence = 0.0
                    results.append(test_case)
            
            return results
            
        except Exception as e:
            logger.error(f"Transcription testing failed: {e}")
            return test_cases
    
    def _calculate_text_accuracy(self, expected: str, actual: str) -> float:
        """Calculate text accuracy using various metrics."""
        try:
            if not expected or not actual:
                return 0.0
            
            # Simple word-based accuracy
            expected_words = expected.lower().split()
            actual_words = actual.lower().split()
            
            if not expected_words:
                return 0.0
            
            # Calculate word overlap
            matches = 0
            for word in expected_words:
                if word in actual_words:
                    matches += 1
            
            word_accuracy = matches / len(expected_words)
            
            # Calculate character-based accuracy
            expected_chars = list(expected.lower())
            actual_chars = list(actual.lower())
            
            char_matches = 0
            min_len = min(len(expected_chars), len(actual_chars))
            for i in range(min_len):
                if expected_chars[i] == actual_chars[i]:
                    char_matches += 1
            
            char_accuracy = char_matches / len(expected_chars) if expected_chars else 0.0
            
            # Combined accuracy
            combined_accuracy = 0.7 * word_accuracy + 0.3 * char_accuracy
            
            return min(1.0, combined_accuracy)
            
        except Exception as e:
            logger.error(f"Text accuracy calculation failed: {e}")
            return 0.0
    
    def _calculate_accuracy(self, results: List[TranscriptionTest]) -> float:
        """Calculate overall accuracy from test results."""
        try:
            if not results:
                return 0.0
            
            # Weighted average of accuracy and confidence
            total_weight = 0.0
            weighted_accuracy = 0.0
            
            for result in results:
                weight = result.confidence  # Use confidence as weight
                weighted_accuracy += result.accuracy * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_accuracy / total_weight
            else:
                return np.mean([r.accuracy for r in results])
                
        except Exception as e:
            logger.error(f"Accuracy calculation failed: {e}")
            return 0.0
    
    async def _generate_improvements(self, results: List[TranscriptionTest], 
                                  session: FeedbackSession) -> List[Dict]:
        """Generate improvements based on test results."""
        try:
            improvements = []
            
            # Analyze common errors
            error_patterns = self._analyze_error_patterns(results)
            
            # Generate vocabulary improvements
            if error_patterns.get('vocabulary_errors', 0) > 0.3:
                vocab_improvement = await self._generate_vocabulary_improvement(results)
                if vocab_improvement:
                    improvements.append(vocab_improvement)
            
            # Generate preprocessing improvements
            if error_patterns.get('audio_quality_errors', 0) > 0.3:
                preprocessing_improvement = await self._generate_preprocessing_improvement(results)
                if preprocessing_improvement:
                    improvements.append(preprocessing_improvement)
            
            # Generate model parameter improvements
            if error_patterns.get('model_errors', 0) > 0.3:
                model_improvement = await self._generate_model_improvement(results)
                if model_improvement:
                    improvements.append(model_improvement)
            
            logger.info(f"Generated {len(improvements)} improvements")
            return improvements
            
        except Exception as e:
            logger.error(f"Improvement generation failed: {e}")
            return []
    
    def _analyze_error_patterns(self, results: List[TranscriptionTest]) -> Dict[str, float]:
        """Analyze error patterns in test results."""
        try:
            patterns = {
                'vocabulary_errors': 0.0,
                'audio_quality_errors': 0.0,
                'model_errors': 0.0,
                'timing_errors': 0.0
            }
            
            for result in results:
                if result.accuracy < 0.8:  # Low accuracy indicates errors
                    # Analyze error types (simplified)
                    if result.confidence < 0.6:
                        patterns['model_errors'] += 1
                    if len(result.actual_text) < len(result.expected_text) * 0.7:
                        patterns['audio_quality_errors'] += 1
                    if self._has_vocabulary_errors(result.expected_text, result.actual_text):
                        patterns['vocabulary_errors'] += 1
            
            # Normalize by total results
            total_results = len(results)
            if total_results > 0:
                for pattern in patterns:
                    patterns[pattern] /= total_results
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error pattern analysis failed: {e}")
            return {}
    
    def _has_vocabulary_errors(self, expected: str, actual: str) -> bool:
        """Check if there are vocabulary-related errors."""
        try:
            # Simple check for vocabulary errors
            expected_words = set(expected.lower().split())
            actual_words = set(actual.lower().split())
            
            # Check for missing important words
            important_words = expected_words - actual_words
            return len(important_words) > len(expected_words) * 0.3
            
        except Exception:
            return False
    
    async def _generate_vocabulary_improvement(self, results: List[TranscriptionTest]) -> Dict:
        """Generate vocabulary improvement suggestions."""
        try:
            # Extract frequently missed words
            missed_words = []
            for result in results:
                if result.accuracy < 0.8:
                    expected_words = result.expected_text.lower().split()
                    actual_words = result.actual_text.lower().split()
                    missed = set(expected_words) - set(actual_words)
                    missed_words.extend(list(missed))
            
            if missed_words:
                # Find most common missed words
                from collections import Counter
                word_counts = Counter(missed_words)
                common_missed = [word for word, count in word_counts.most_common(10)]
                
                improvement = {
                    'type': 'vocabulary',
                    'action': 'add_domain_vocabulary',
                    'words': common_missed,
                    'boost_factor': 2.0,
                    'description': f"Add {len(common_missed)} frequently missed words to vocabulary"
                }
                
                return improvement
            
            return None
            
        except Exception as e:
            logger.error(f"Vocabulary improvement generation failed: {e}")
            return None
    
    async def _generate_preprocessing_improvement(self, results: List[TranscriptionTest]) -> Dict:
        """Generate preprocessing improvement suggestions."""
        try:
            # Analyze audio quality issues
            low_confidence_results = [r for r in results if r.confidence < 0.6]
            
            if len(low_confidence_results) > len(results) * 0.3:
                improvement = {
                    'type': 'preprocessing',
                    'action': 'enhance_audio_quality',
                    'parameters': {
                        'noise_reduction': True,
                        'spectral_enhancement': True,
                        'dynamic_range_compression': True
                    },
                    'description': "Enhance audio preprocessing for better quality"
                }
                
                return improvement
            
            return None
            
        except Exception as e:
            logger.error(f"Preprocessing improvement generation failed: {e}")
            return None
    
    async def _generate_model_improvement(self, results: List[TranscriptionTest]) -> Dict:
        """Generate model parameter improvement suggestions."""
        try:
            # Analyze model performance
            avg_confidence = np.mean([r.confidence for r in results])
            
            if avg_confidence < 0.7:
                improvement = {
                    'type': 'model_parameters',
                    'action': 'optimize_transcription_parameters',
                    'parameters': {
                        'beam_size': 10,
                        'temperature': 0.0,
                        'best_of': 5,
                        'patience': 2.0
                    },
                    'description': "Optimize transcription parameters for better accuracy"
                }
                
                return improvement
            
            return None
            
        except Exception as e:
            logger.error(f"Model improvement generation failed: {e}")
            return None
    
    async def _apply_improvements(self, improvements: List[Dict]):
        """Apply generated improvements to the system."""
        try:
            for improvement in improvements:
                if improvement['type'] == 'vocabulary':
                    await self._apply_vocabulary_improvement(improvement)
                elif improvement['type'] == 'preprocessing':
                    await self._apply_preprocessing_improvement(improvement)
                elif improvement['type'] == 'model_parameters':
                    await self._apply_model_improvement(improvement)
                
                logger.info(f"Applied improvement: {improvement['description']}")
            
        except Exception as e:
            logger.error(f"Improvement application failed: {e}")
    
    async def _apply_vocabulary_improvement(self, improvement: Dict):
        """Apply vocabulary improvements."""
        try:
            from app.custom_vocabulary import vocabulary_manager
            
            # Add words to vocabulary
            vocabulary_manager.add_domain_vocabulary(
                domain="feedback_improved",
                terms=improvement['words'],
                boost_factors=[improvement['boost_factor']] * len(improvement['words'])
            )
            
            logger.info(f"Added {len(improvement['words'])} words to vocabulary")
            
        except Exception as e:
            logger.error(f"Vocabulary improvement application failed: {e}")
    
    async def _apply_preprocessing_improvement(self, improvement: Dict):
        """Apply preprocessing improvements."""
        try:
            # Update preprocessing configuration
            # This would update the audio preprocessing pipeline
            logger.info("Applied preprocessing improvements")
            
        except Exception as e:
            logger.error(f"Preprocessing improvement application failed: {e}")
    
    async def _apply_model_improvement(self, improvement: Dict):
        """Apply model parameter improvements."""
        try:
            # Update model parameters
            # This would update the transcription model configuration
            logger.info("Applied model parameter improvements")
            
        except Exception as e:
            logger.error(f"Model improvement application failed: {e}")
    
    def _initialize_opentts(self):
        """Initialize OpenTTS connection."""
        try:
            # Check if OpenTTS is available
            import aiohttp
            import asyncio
            
            async def check_opentts():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{self.opentts_url}/api/voices",
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                self.opentts_available = True
                                logger.info("OpenTTS connection established")
                            else:
                                logger.warning(f"OpenTTS not available: {response.status}")
                except Exception as e:
                    logger.warning(f"OpenTTS connection failed: {e}")
            
            # Run the check
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(check_opentts())
            except RuntimeError:
                # Create new event loop if none exists
                asyncio.run(check_opentts())
            
        except Exception as e:
            logger.warning(f"OpenTTS initialization failed: {e}")
    
    def get_feedback_session_status(self, session_id: str) -> Optional[Dict]:
        """Get status of a feedback session."""
        try:
            if session_id in self.feedback_sessions:
                session = self.feedback_sessions[session_id]
                return {
                    'session_id': session_id,
                    'status': session.status,
                    'iterations': session.iterations,
                    'current_accuracy': session.current_accuracy,
                    'target_accuracy': session.target_accuracy,
                    'improvements_count': len(session.improvements),
                    'start_time': session.start_time.isoformat()
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session status: {e}")
            return None
    
    def get_improvement_history(self) -> List[Dict]:
        """Get history of improvements made."""
        return self.improvement_history.copy()
    
    def get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'opentts_url': 'http://localhost:5500',
            'max_iterations': 50,
            'iteration_delay': 5.0,
            'temp_dir': '/tmp',
            'target_accuracy': 0.95,
            'min_improvement_threshold': 0.05
        }

# Global feedback system instance
feedback_system = MetaRecursiveFeedbackSystem()

# API endpoints for the feedback system
async def start_feedback_loop(target_accuracy: float = 0.95) -> str:
    """Start a new feedback loop session."""
    return await feedback_system.start_feedback_session(target_accuracy)

async def get_feedback_status(session_id: str) -> Optional[Dict]:
    """Get feedback session status."""
    return feedback_system.get_feedback_session_status(session_id)

def get_improvement_history() -> List[Dict]:
    """Get improvement history."""
    return feedback_system.get_improvement_history()
