"""
Multi-Model Ensemble Transcription System
=======================================
Advanced ensemble approach combining multiple transcription models for improved accuracy.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os
from collections import defaultdict
import asyncio
import concurrent.futures
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionCandidate:
    """Represents a transcription candidate with metadata."""
    text: str
    confidence: float
    model_name: str
    segments: List[Dict]
    language: str = "en"
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

@dataclass
class EnsembleConfig:
    """Configuration for ensemble transcription."""
    models: List[str]
    weights: Dict[str, float]
    voting_method: str = "confidence_weighted"  # "confidence_weighted", "majority", "consensus"
    min_confidence: float = 0.5
    max_candidates: int = 5
    temporal_alignment: bool = True
    language_detection: bool = True

class EnsembleTranscription:
    """Multi-model ensemble transcription system."""
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or self.get_default_config()
        self.models = {}
        self.results_cache = {}
        
        # Initialize models
        self._initialize_models()
    
    def transcribe_ensemble(self, audio_path: str, domain: str = None) -> List[Dict]:
        """Perform ensemble transcription using multiple models."""
        try:
            logger.info(f"Starting ensemble transcription for {audio_path}")
            
            # Generate candidates from multiple models
            candidates = self._generate_candidates(audio_path, domain)
            
            if not candidates:
                logger.warning("No transcription candidates generated")
                return []
            
            # Combine candidates using ensemble method
            ensemble_result = self._combine_candidates(candidates)
            
            logger.info(f"Ensemble transcription complete: {len(ensemble_result)} segments")
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Ensemble transcription failed: {e}")
            return []
    
    def _generate_candidates(self, audio_path: str, domain: str = None) -> List[TranscriptionCandidate]:
        """Generate transcription candidates from multiple models."""
        candidates = []
        
        # Run models in parallel for efficiency
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_model = {}
            
            for model_name in self.config.models:
                if model_name in self.models:
                    future = executor.submit(
                        self._transcribe_with_model, 
                        model_name, 
                        audio_path, 
                        domain
                    )
                    future_to_model[future] = model_name
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        candidates.append(result)
                        logger.info(f"Generated candidate from {model_name}")
                except Exception as e:
                    logger.error(f"Model {model_name} failed: {e}")
        
        return candidates
    
    def _transcribe_with_model(self, model_name: str, audio_path: str, domain: str = None) -> Optional[TranscriptionCandidate]:
        """Transcribe using a specific model."""
        try:
            model = self.models[model_name]
            
            # Get model-specific configuration
            model_config = self._get_model_config(model_name, domain)
            
            # Perform transcription
            if model_name == 'whisper':
                result = self._transcribe_with_whisper(model, audio_path, model_config)
            elif model_name == 'faster_whisper':
                result = self._transcribe_with_faster_whisper(model, audio_path, model_config)
            else:
                logger.warning(f"Unknown model: {model_name}")
                return None
            
            if result:
                return TranscriptionCandidate(
                    text=result.get('text', ''),
                    confidence=result.get('confidence', 0.0),
                    model_name=model_name,
                    segments=result.get('segments', []),
                    language=result.get('language', 'en')
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Transcription with {model_name} failed: {e}")
            return None
    
    def _transcribe_with_whisper(self, model, audio_path: str, config: Dict) -> Dict:
        """Transcribe using OpenAI Whisper."""
        try:
            result = model.transcribe(
                audio_path,
                temperature=config.get('temperature', 0.0),
                beam_size=config.get('beam_size', 5),
                best_of=config.get('best_of', 1),
                patience=config.get('patience', 1.0),
                length_penalty=config.get('length_penalty', 1.0),
                repetition_penalty=config.get('repetition_penalty', 1.0),
                no_repeat_ngram_size=config.get('no_repeat_ngram_size', 0),
                condition_on_previous_text=config.get('condition_on_previous_text', True),
                prompt=config.get('prompt', None)
            )
            
            # Calculate average confidence
            segments = result.get('segments', [])
            avg_confidence = np.mean([seg.get('confidence', 0.0) for seg in segments]) if segments else 0.0
            
            return {
                'text': result.get('text', ''),
                'confidence': avg_confidence,
                'segments': segments,
                'language': result.get('language', 'en')
            }
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {}
    
    def _transcribe_with_faster_whisper(self, model, audio_path: str, config: Dict) -> Dict:
        """Transcribe using Faster Whisper."""
        try:
            segments, info = model.transcribe(
                audio_path,
                beam_size=config.get('beam_size', 5),
                temperature=config.get('temperature', 0.0),
                best_of=config.get('best_of', 1),
                patience=config.get('patience', 1.0),
                length_penalty=config.get('length_penalty', 1.0),
                repetition_penalty=config.get('repetition_penalty', 1.0),
                no_repeat_ngram_size=config.get('no_repeat_ngram_size', 0),
                condition_on_previous_text=config.get('condition_on_previous_text', True),
                prompt=config.get('prompt', None)
            )
            
            # Convert segments to list format
            segments_list = []
            total_confidence = 0.0
            
            for segment in segments:
                seg_dict = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'confidence': getattr(segment, 'confidence', 0.8)
                }
                segments_list.append(seg_dict)
                total_confidence += seg_dict['confidence']
            
            avg_confidence = total_confidence / len(segments_list) if segments_list else 0.0
            
            return {
                'text': ' '.join([seg['text'] for seg in segments_list]),
                'confidence': avg_confidence,
                'segments': segments_list,
                'language': info.language if hasattr(info, 'language') else 'en'
            }
            
        except Exception as e:
            logger.error(f"Faster Whisper transcription failed: {e}")
            return {}
    
    def _combine_candidates(self, candidates: List[TranscriptionCandidate]) -> List[Dict]:
        """Combine transcription candidates using ensemble method."""
        try:
            if not candidates:
                return []
            
            if len(candidates) == 1:
                return candidates[0].segments
            
            if self.config.voting_method == "confidence_weighted":
                return self._confidence_weighted_combination(candidates)
            elif self.config.voting_method == "majority":
                return self._majority_voting_combination(candidates)
            elif self.config.voting_method == "consensus":
                return self._consensus_combination(candidates)
            else:
                logger.warning(f"Unknown voting method: {self.config.voting_method}")
                return candidates[0].segments
                
        except Exception as e:
            logger.error(f"Candidate combination failed: {e}")
            return candidates[0].segments if candidates else []
    
    def _confidence_weighted_combination(self, candidates: List[TranscriptionCandidate]) -> List[Dict]:
        """Combine candidates using confidence-weighted voting."""
        try:
            # Sort candidates by confidence
            sorted_candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)
            
            # Use highest confidence candidate as base
            base_candidate = sorted_candidates[0]
            base_segments = base_candidate.segments
            
            # Apply weights from other candidates
            weighted_segments = []
            
            for i, segment in enumerate(base_segments):
                weighted_text = segment['text']
                weighted_confidence = segment['confidence']
                
                # Apply weights from other candidates
                for candidate in sorted_candidates[1:]:
                    if i < len(candidate.segments):
                        other_segment = candidate.segments[i]
                        weight = self.config.weights.get(candidate.model_name, 0.5)
                        
                        # Weighted combination of text (simplified)
                        if other_segment['confidence'] > segment['confidence'] * 0.8:
                            weighted_confidence = max(weighted_confidence, other_segment['confidence'] * weight)
                
                weighted_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': weighted_text,
                    'confidence': weighted_confidence,
                    'ensemble_models': [c.model_name for c in sorted_candidates]
                })
            
            return weighted_segments
            
        except Exception as e:
            logger.error(f"Confidence-weighted combination failed: {e}")
            return candidates[0].segments if candidates else []
    
    def _majority_voting_combination(self, candidates: List[TranscriptionCandidate]) -> List[Dict]:
        """Combine candidates using majority voting."""
        try:
            # For each time segment, find the majority vote
            all_segments = []
            for candidate in candidates:
                all_segments.extend(candidate.segments)
            
            # Group segments by time overlap
            grouped_segments = self._group_segments_by_time(all_segments)
            
            # For each group, find majority text
            majority_segments = []
            for group in grouped_segments:
                if not group:
                    continue
                
                # Count text occurrences
                text_counts = defaultdict(int)
                for segment in group:
                    text_counts[segment['text']] += 1
                
                # Find majority text
                majority_text = max(text_counts.items(), key=lambda x: x[1])[0]
                
                # Use average timing and confidence
                avg_start = np.mean([s['start'] for s in group])
                avg_end = np.mean([s['end'] for s in group])
                avg_confidence = np.mean([s['confidence'] for s in group])
                
                majority_segments.append({
                    'start': avg_start,
                    'end': avg_end,
                    'text': majority_text,
                    'confidence': avg_confidence,
                    'ensemble_models': [c.model_name for c in candidates]
                })
            
            return majority_segments
            
        except Exception as e:
            logger.error(f"Majority voting combination failed: {e}")
            return candidates[0].segments if candidates else []
    
    def _consensus_combination(self, candidates: List[TranscriptionCandidate]) -> List[Dict]:
        """Combine candidates using consensus method."""
        try:
            # Find segments that appear in multiple candidates
            consensus_segments = []
            
            # Group segments by time overlap
            all_segments = []
            for candidate in candidates:
                all_segments.extend(candidate.segments)
            
            grouped_segments = self._group_segments_by_time(all_segments)
            
            for group in grouped_segments:
                if len(group) < 2:  # Need at least 2 candidates for consensus
                    continue
                
                # Find consensus text (appears in multiple candidates)
                text_counts = defaultdict(int)
                for segment in group:
                    text_counts[segment['text']] += 1
                
                # Only use segments with consensus (appears in multiple candidates)
                consensus_texts = [text for text, count in text_counts.items() if count >= 2]
                
                if consensus_texts:
                    # Use the most frequent consensus text
                    consensus_text = max(consensus_texts, key=lambda x: text_counts[x])
                    
                    # Calculate average timing and confidence
                    consensus_segments_in_group = [s for s in group if s['text'] == consensus_text]
                    
                    if consensus_segments_in_group:
                        avg_start = np.mean([s['start'] for s in consensus_segments_in_group])
                        avg_end = np.mean([s['end'] for s in consensus_segments_in_group])
                        avg_confidence = np.mean([s['confidence'] for s in consensus_segments_in_group])
                        
                        consensus_segments.append({
                            'start': avg_start,
                            'end': avg_end,
                            'text': consensus_text,
                            'confidence': avg_confidence,
                            'ensemble_models': [c.model_name for c in candidates]
                        })
            
            return consensus_segments
            
        except Exception as e:
            logger.error(f"Consensus combination failed: {e}")
            return candidates[0].segments if candidates else []
    
    def _group_segments_by_time(self, segments: List[Dict], overlap_threshold: float = 0.5) -> List[List[Dict]]:
        """Group segments by time overlap."""
        try:
            if not segments:
                return []
            
            # Sort segments by start time
            sorted_segments = sorted(segments, key=lambda x: x['start'])
            
            groups = []
            current_group = [sorted_segments[0]]
            
            for segment in sorted_segments[1:]:
                # Check if segment overlaps with current group
                group_end = max(s['end'] for s in current_group)
                overlap = min(segment['end'], group_end) - max(segment['start'], current_group[0]['start'])
                overlap_ratio = overlap / (segment['end'] - segment['start'])
                
                if overlap_ratio >= overlap_threshold:
                    current_group.append(segment)
                else:
                    groups.append(current_group)
                    current_group = [segment]
            
            groups.append(current_group)
            return groups
            
        except Exception as e:
            logger.error(f"Segment grouping failed: {e}")
            return [segments]
    
    def _get_model_config(self, model_name: str, domain: str = None) -> Dict:
        """Get model-specific configuration."""
        base_config = {
            'temperature': 0.0,
            'beam_size': 5,
            'best_of': 1,
            'patience': 1.0,
            'length_penalty': 1.0,
            'repetition_penalty': 1.0,
            'no_repeat_ngram_size': 0,
            'condition_on_previous_text': True,
            'prompt': None
        }
        
        # Add domain-specific configuration
        if domain:
            base_config['prompt'] = f"Domain: {domain}"
        
        return base_config
    
    def _initialize_models(self):
        """Initialize transcription models."""
        try:
            # Initialize Whisper
            if 'whisper' in self.config.models:
                try:
                    import whisper
                    self.models['whisper'] = whisper.load_model("base")
                    logger.info("Initialized Whisper model")
                except ImportError:
                    logger.warning("Whisper not available")
            
            # Initialize Faster Whisper
            if 'faster_whisper' in self.config.models:
                try:
                    from faster_whisper import WhisperModel
                    self.models['faster_whisper'] = WhisperModel("base", device="cpu", compute_type="int8")
                    logger.info("Initialized Faster Whisper model")
                except ImportError:
                    logger.warning("Faster Whisper not available")
            
            logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
    
    def get_default_config(self) -> EnsembleConfig:
        """Get default ensemble configuration."""
        return EnsembleConfig(
            models=['whisper', 'faster_whisper'],
            weights={'whisper': 0.6, 'faster_whisper': 0.4},
            voting_method='confidence_weighted',
            min_confidence=0.5,
            max_candidates=5,
            temporal_alignment=True,
            language_detection=True
        )

# Global ensemble instance
ensemble_transcription = EnsembleTranscription()

# Plugin registration
def ensemble_transcription_plugin(audio_path: str) -> List[Dict]:
    """Ensemble transcription plugin."""
    return ensemble_transcription.transcribe_ensemble(audio_path)
