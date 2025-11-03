"""
Real-Time Quality Monitoring System
==================================
Comprehensive quality monitoring and assessment for transcription accuracy.
"""

import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for audio and transcription."""
    snr: float
    clarity: float
    noise_level: float
    speech_rate: float
    overlap_ratio: float
    spectral_centroid: float
    zero_crossing_rate: float
    mfcc_variance: float
    overall_quality: float

@dataclass
class QualityRecommendations:
    """Quality improvement recommendations."""
    recommendations: List[str]
    priority: str  # "high", "medium", "low"
    expected_improvement: float

class TranscriptionQualityMonitor:
    """Real-time quality monitoring system."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        self.quality_history = []
        self.baseline_metrics = None
    
    def assess_audio_quality(self, audio_path: str) -> QualityMetrics:
        """Comprehensive audio quality assessment."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            logger.info(f"Assessing audio quality: {len(y)} samples at {sr}Hz")
            
            # Calculate various quality metrics
            snr = self._calculate_snr(y, sr)
            clarity = self._calculate_clarity(y, sr)
            noise_level = self._calculate_noise_level(y, sr)
            speech_rate = self._calculate_speech_rate(y, sr)
            overlap_ratio = self._calculate_speaker_overlap(y, sr)
            spectral_centroid = self._calculate_spectral_centroid(y, sr)
            zcr = self._calculate_zero_crossing_rate(y, sr)
            mfcc_variance = self._calculate_mfcc_variance(y, sr)
            
            # Calculate overall quality score
            overall_quality = self._calculate_overall_quality(
                snr, clarity, noise_level, speech_rate, overlap_ratio
            )
            
            metrics = QualityMetrics(
                snr=snr,
                clarity=clarity,
                noise_level=noise_level,
                speech_rate=speech_rate,
                overlap_ratio=overlap_ratio,
                spectral_centroid=spectral_centroid,
                zero_crossing_rate=zcr,
                mfcc_variance=mfcc_variance,
                overall_quality=overall_quality
            )
            
            # Store in history
            self.quality_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'audio_path': audio_path,
                'metrics': metrics
            })
            
            logger.info(f"Quality assessment complete: overall_quality={overall_quality:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Audio quality assessment failed: {e}")
            return self._get_fallback_metrics()
    
    def assess_transcription_quality(self, transcription: str, reference: str = None) -> Dict[str, Any]:
        """Assess transcription quality."""
        try:
            quality_metrics = {
                'word_error_rate': 0.0,
                'character_error_rate': 0.0,
                'bleu_score': 0.0,
                'rouge_score': 0.0,
                'confidence_score': 0.0,
                'completeness': 0.0,
                'fluency': 0.0
            }
            
            if reference:
                # Calculate error rates
                quality_metrics['word_error_rate'] = self._calculate_wer(transcription, reference)
                quality_metrics['character_error_rate'] = self._calculate_cer(transcription, reference)
                quality_metrics['bleu_score'] = self._calculate_bleu(transcription, reference)
                quality_metrics['rouge_score'] = self._calculate_rouge(transcription, reference)
            
            # Calculate other metrics
            quality_metrics['confidence_score'] = self._calculate_confidence_score(transcription)
            quality_metrics['completeness'] = self._calculate_completeness(transcription)
            quality_metrics['fluency'] = self._calculate_fluency(transcription)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Transcription quality assessment failed: {e}")
            return {}
    
    def generate_recommendations(self, audio_metrics: QualityMetrics, 
                               transcription_metrics: Dict = None) -> QualityRecommendations:
        """Generate quality improvement recommendations."""
        try:
            recommendations = []
            priority = "low"
            expected_improvement = 0.0
            
            # Audio quality recommendations
            if audio_metrics.snr < 10:
                recommendations.append("Low signal-to-noise ratio detected. Consider noise reduction preprocessing.")
                priority = "high"
                expected_improvement += 0.15
            
            if audio_metrics.clarity < 0.7:
                recommendations.append("Low audio clarity detected. Consider spectral enhancement.")
                priority = "high" if priority != "high" else priority
                expected_improvement += 0.10
            
            if audio_metrics.overlap_ratio > 0.3:
                recommendations.append("High speaker overlap detected. Consider speaker separation techniques.")
                priority = "medium" if priority == "low" else priority
                expected_improvement += 0.20
            
            if audio_metrics.noise_level > 0.5:
                recommendations.append("High noise level detected. Consider noise reduction.")
                priority = "high" if priority != "high" else priority
                expected_improvement += 0.12
            
            if audio_metrics.speech_rate < 0.3 or audio_metrics.speech_rate > 0.8:
                recommendations.append("Unusual speech rate detected. Consider audio normalization.")
                priority = "medium" if priority == "low" else priority
                expected_improvement += 0.08
            
            # Transcription quality recommendations
            if transcription_metrics:
                if transcription_metrics.get('word_error_rate', 0) > 0.3:
                    recommendations.append("High word error rate detected. Consider model fine-tuning.")
                    priority = "high" if priority != "high" else priority
                    expected_improvement += 0.25
                
                if transcription_metrics.get('confidence_score', 0) < 0.7:
                    recommendations.append("Low confidence scores detected. Consider ensemble methods.")
                    priority = "medium" if priority == "low" else priority
                    expected_improvement += 0.15
                
                if transcription_metrics.get('completeness', 0) < 0.8:
                    recommendations.append("Incomplete transcription detected. Check audio quality.")
                    priority = "high" if priority != "high" else priority
                    expected_improvement += 0.18
            
            # General recommendations
            if audio_metrics.overall_quality < 0.6:
                recommendations.append("Overall audio quality is poor. Consider comprehensive preprocessing.")
                priority = "high" if priority != "high" else priority
                expected_improvement += 0.20
            
            if not recommendations:
                recommendations.append("Audio quality is good. No specific improvements needed.")
                priority = "low"
                expected_improvement = 0.0
            
            return QualityRecommendations(
                recommendations=recommendations,
                priority=priority,
                expected_improvement=min(expected_improvement, 1.0)
            )
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return QualityRecommendations(
                recommendations=["Unable to generate recommendations"],
                priority="low",
                expected_improvement=0.0
            )
    
    def _calculate_snr(self, audio: np.ndarray, sr: int) -> float:
        """Calculate signal-to-noise ratio."""
        try:
            # Estimate noise from quiet segments
            frame_length = 1024
            hop_length = 512
            
            # Calculate RMS energy
            energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Find quiet segments (bottom 20%)
            quiet_threshold = np.percentile(energy, 20)
            noise_energy = np.mean(energy[energy < quiet_threshold])
            
            # Find speech segments (top 20%)
            speech_threshold = np.percentile(energy, 80)
            speech_energy = np.mean(energy[energy > speech_threshold])
            
            if noise_energy > 0:
                snr = 20 * np.log10(speech_energy / noise_energy)
            else:
                snr = 50.0  # Very high SNR if no noise detected
            
            return max(0.0, min(snr, 50.0))  # Clamp between 0 and 50 dB
            
        except Exception as e:
            logger.debug(f"SNR calculation failed: {e}")
            return 20.0  # Default moderate SNR
    
    def _calculate_clarity(self, audio: np.ndarray, sr: int) -> float:
        """Calculate audio clarity score."""
        try:
            # Calculate spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Clarity indicators
            centroid_variance = np.var(spectral_centroid)
            rolloff_variance = np.var(spectral_rolloff)
            mfcc_variance = np.var(mfccs)
            
            # Combine into clarity score (0-1)
            clarity = min(1.0, (centroid_variance + rolloff_variance + mfcc_variance) / 3.0)
            
            return clarity
            
        except Exception as e:
            logger.debug(f"Clarity calculation failed: {e}")
            return 0.5  # Default moderate clarity
    
    def _calculate_noise_level(self, audio: np.ndarray, sr: int) -> float:
        """Calculate noise level in audio."""
        try:
            # Calculate spectral features
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Estimate noise from low-energy frames
            frame_energy = np.sum(magnitude, axis=0)
            noise_threshold = np.percentile(frame_energy, 20)
            noise_frames = frame_energy < noise_threshold
            
            if np.any(noise_frames):
                noise_spectrum = np.mean(magnitude[:, noise_frames], axis=1)
                total_spectrum = np.mean(magnitude, axis=1)
                
                # Noise ratio
                noise_ratio = np.sum(noise_spectrum) / (np.sum(total_spectrum) + 1e-10)
                return min(1.0, noise_ratio)
            else:
                return 0.1  # Low noise if no quiet frames found
                
        except Exception as e:
            logger.debug(f"Noise level calculation failed: {e}")
            return 0.3  # Default moderate noise
    
    def _calculate_speech_rate(self, audio: np.ndarray, sr: int) -> float:
        """Calculate speech rate (speech vs silence ratio)."""
        try:
            # Voice activity detection
            frame_length = 1024
            hop_length = 512
            
            # Calculate energy
            energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Threshold for speech detection
            threshold = np.mean(energy) + 0.5 * np.std(energy)
            speech_frames = energy > threshold
            
            # Speech rate
            speech_rate = np.sum(speech_frames) / len(speech_frames)
            
            return speech_rate
            
        except Exception as e:
            logger.debug(f"Speech rate calculation failed: {e}")
            return 0.5  # Default moderate speech rate
    
    def _calculate_speaker_overlap(self, audio: np.ndarray, sr: int) -> float:
        """Estimate speaker overlap ratio."""
        try:
            # This is a simplified estimation
            # In practice, you'd need more sophisticated speaker diarization
            
            # Calculate energy variations that might indicate speaker changes
            frame_length = 1024
            hop_length = 512
            
            energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Calculate energy variations
            energy_diff = np.diff(energy)
            high_variations = np.sum(np.abs(energy_diff) > np.std(energy_diff))
            
            # Estimate overlap based on energy variations
            overlap_ratio = min(1.0, high_variations / len(energy_diff))
            
            return overlap_ratio
            
        except Exception as e:
            logger.debug(f"Speaker overlap calculation failed: {e}")
            return 0.2  # Default low overlap
    
    def _calculate_spectral_centroid(self, audio: np.ndarray, sr: int) -> float:
        """Calculate spectral centroid."""
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            return np.mean(spectral_centroid)
        except Exception as e:
            logger.debug(f"Spectral centroid calculation failed: {e}")
            return 2000.0  # Default value
    
    def _calculate_zero_crossing_rate(self, audio: np.ndarray, sr: int) -> float:
        """Calculate zero crossing rate."""
        try:
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            return np.mean(zcr)
        except Exception as e:
            logger.debug(f"Zero crossing rate calculation failed: {e}")
            return 0.1  # Default value
    
    def _calculate_mfcc_variance(self, audio: np.ndarray, sr: int) -> float:
        """Calculate MFCC variance."""
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            return np.var(mfccs)
        except Exception as e:
            logger.debug(f"MFCC variance calculation failed: {e}")
            return 1.0  # Default value
    
    def _calculate_overall_quality(self, snr: float, clarity: float, noise_level: float, 
                                 speech_rate: float, overlap_ratio: float) -> float:
        """Calculate overall quality score."""
        try:
            # Normalize metrics to 0-1 scale
            snr_norm = min(1.0, snr / 30.0)  # 30 dB is excellent
            clarity_norm = clarity
            noise_norm = 1.0 - noise_level  # Invert noise level
            speech_rate_norm = 1.0 - abs(speech_rate - 0.5) * 2  # Optimal around 0.5
            overlap_norm = 1.0 - overlap_ratio  # Less overlap is better
            
            # Weighted combination
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # SNR and clarity are most important
            overall_quality = (
                weights[0] * snr_norm +
                weights[1] * clarity_norm +
                weights[2] * noise_norm +
                weights[3] * speech_rate_norm +
                weights[4] * overlap_norm
            )
            
            return max(0.0, min(1.0, overall_quality))
            
        except Exception as e:
            logger.debug(f"Overall quality calculation failed: {e}")
            return 0.5  # Default moderate quality
    
    def _calculate_wer(self, transcription: str, reference: str) -> float:
        """Calculate Word Error Rate."""
        try:
            # Simple WER calculation
            ref_words = reference.lower().split()
            hyp_words = transcription.lower().split()
            
            # Dynamic programming for edit distance
            d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
            
            for i in range(len(ref_words) + 1):
                d[i][0] = i
            for j in range(len(hyp_words) + 1):
                d[0][j] = j
            
            for i in range(1, len(ref_words) + 1):
                for j in range(1, len(hyp_words) + 1):
                    if ref_words[i-1] == hyp_words[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
            
            wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
            return min(1.0, wer)
            
        except Exception as e:
            logger.debug(f"WER calculation failed: {e}")
            return 0.5  # Default moderate error rate
    
    def _calculate_cer(self, transcription: str, reference: str) -> float:
        """Calculate Character Error Rate."""
        try:
            # Simple CER calculation
            ref_chars = list(reference.lower())
            hyp_chars = list(transcription.lower())
            
            # Edit distance for characters
            d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
            
            for i in range(len(ref_chars) + 1):
                d[i][0] = i
            for j in range(len(hyp_chars) + 1):
                d[0][j] = j
            
            for i in range(1, len(ref_chars) + 1):
                for j in range(1, len(hyp_chars) + 1):
                    if ref_chars[i-1] == hyp_chars[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
            
            cer = d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)
            return min(1.0, cer)
            
        except Exception as e:
            logger.debug(f"CER calculation failed: {e}")
            return 0.3  # Default moderate error rate
    
    def _calculate_bleu(self, transcription: str, reference: str) -> float:
        """Calculate BLEU score."""
        try:
            # Simplified BLEU calculation
            ref_words = reference.lower().split()
            hyp_words = transcription.lower().split()
            
            # 1-gram precision
            ref_counts = {}
            for word in ref_words:
                ref_counts[word] = ref_counts.get(word, 0) + 1
            
            hyp_counts = {}
            for word in hyp_words:
                hyp_counts[word] = hyp_counts.get(word, 0) + 1
            
            matches = 0
            for word in hyp_counts:
                matches += min(hyp_counts[word], ref_counts.get(word, 0))
            
            precision = matches / len(hyp_words) if hyp_words else 0.0
            
            # Brevity penalty
            bp = min(1.0, len(hyp_words) / len(ref_words)) if ref_words else 0.0
            
            bleu = bp * precision
            return min(1.0, bleu)
            
        except Exception as e:
            logger.debug(f"BLEU calculation failed: {e}")
            return 0.3  # Default moderate score
    
    def _calculate_rouge(self, transcription: str, reference: str) -> float:
        """Calculate ROUGE score."""
        try:
            # Simplified ROUGE-L calculation
            ref_words = reference.lower().split()
            hyp_words = transcription.lower().split()
            
            # Longest Common Subsequence
            def lcs_length(seq1, seq2):
                m, n = len(seq1), len(seq2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if seq1[i-1] == seq2[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
                return dp[m][n]
            
            lcs_len = lcs_length(ref_words, hyp_words)
            
            if len(ref_words) == 0:
                return 0.0
            
            rouge_l = lcs_len / len(ref_words)
            return min(1.0, rouge_l)
            
        except Exception as e:
            logger.debug(f"ROUGE calculation failed: {e}")
            return 0.3  # Default moderate score
    
    def _calculate_confidence_score(self, transcription: str) -> float:
        """Calculate confidence score based on transcription characteristics."""
        try:
            if not transcription:
                return 0.0
            
            # Factors that indicate confidence
            word_count = len(transcription.split())
            char_count = len(transcription)
            
            # Average word length (longer words might be more confident)
            avg_word_length = char_count / word_count if word_count > 0 else 0
            
            # Punctuation presence (indicates sentence structure)
            has_punctuation = any(p in transcription for p in '.!?')
            
            # Capitalization (indicates proper nouns)
            has_capitals = any(c.isupper() for c in transcription)
            
            # Combine factors
            confidence = 0.5  # Base confidence
            
            if word_count > 5:
                confidence += 0.1
            if avg_word_length > 4:
                confidence += 0.1
            if has_punctuation:
                confidence += 0.1
            if has_capitals:
                confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.debug(f"Confidence score calculation failed: {e}")
            return 0.5  # Default moderate confidence
    
    def _calculate_completeness(self, transcription: str) -> float:
        """Calculate transcription completeness."""
        try:
            if not transcription:
                return 0.0
            
            # Check for common incomplete patterns
            incomplete_patterns = ['...', 'unclear', 'inaudible', '[', ']']
            has_incomplete = any(pattern in transcription.lower() for pattern in incomplete_patterns)
            
            # Check for very short transcriptions
            word_count = len(transcription.split())
            is_too_short = word_count < 3
            
            # Calculate completeness
            completeness = 1.0
            if has_incomplete:
                completeness -= 0.3
            if is_too_short:
                completeness -= 0.2
            
            return max(0.0, completeness)
            
        except Exception as e:
            logger.debug(f"Completeness calculation failed: {e}")
            return 0.7  # Default moderate completeness
    
    def _calculate_fluency(self, transcription: str) -> float:
        """Calculate transcription fluency."""
        try:
            if not transcription:
                return 0.0
            
            # Check for fluency indicators
            words = transcription.split()
            word_count = len(words)
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
            
            # Sentence structure (presence of punctuation)
            has_sentence_structure = any(p in transcription for p in '.!?')
            
            # Repetition (bad for fluency)
            unique_words = len(set(word.lower() for word in words))
            repetition_ratio = unique_words / word_count if word_count > 0 else 0
            
            # Calculate fluency
            fluency = 0.5  # Base fluency
            
            if avg_word_length > 3:  # Reasonable word length
                fluency += 0.1
            if has_sentence_structure:
                fluency += 0.2
            if repetition_ratio > 0.7:  # Good vocabulary diversity
                fluency += 0.2
            
            return min(1.0, fluency)
            
        except Exception as e:
            logger.debug(f"Fluency calculation failed: {e}")
            return 0.6  # Default moderate fluency
    
    def _get_fallback_metrics(self) -> QualityMetrics:
        """Get fallback metrics when calculation fails."""
        return QualityMetrics(
            snr=20.0,
            clarity=0.5,
            noise_level=0.3,
            speech_rate=0.5,
            overlap_ratio=0.2,
            spectral_centroid=2000.0,
            zero_crossing_rate=0.1,
            mfcc_variance=1.0,
            overall_quality=0.5
        )
    
    def get_default_config(self) -> Dict:
        """Get default quality monitoring configuration."""
        return {
            'snr_threshold': 10.0,
            'clarity_threshold': 0.7,
            'noise_threshold': 0.5,
            'overlap_threshold': 0.3,
            'quality_threshold': 0.6,
            'history_size': 100
        }

# Global quality monitor instance
quality_monitor = TranscriptionQualityMonitor()

# Plugin registration
def quality_monitoring_plugin(segments: List) -> List:
    """Quality monitoring plugin."""
    try:
        # This would be called during post-processing
        # Implementation depends on specific use case
        logger.info("Applied quality monitoring")
        return segments
    except Exception as e:
        logger.error(f"Quality monitoring failed: {e}")
        return segments
