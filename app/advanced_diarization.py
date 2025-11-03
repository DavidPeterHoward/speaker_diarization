"""
Advanced Speaker Diarization with Improved Accuracy
==================================================
Enhanced speaker diarization with advanced clustering and segment-level reassignment.
"""

import logging
import numpy as np
import librosa
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import uuid
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedSpeakerDiarization:
    """Advanced speaker diarization with improved accuracy."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        self.encoder = None
        self.scaler = StandardScaler()
        
    def diarize_speakers(self, audio_path: str) -> List[Tuple[float, float, str]]:
        """Perform advanced speaker diarization."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            logger.info(f"Processing audio: {len(y)} samples at {sr}Hz")
            
            # Extract embeddings with overlapping windows
            embeddings, timestamps = self.extract_overlapping_embeddings(y, sr)
            
            if len(embeddings) < 2:
                logger.warning("Insufficient audio for diarization")
                return [(0.0, len(y)/sr, "SPEAKER_0")]
            
            # Determine optimal number of speakers
            optimal_speakers = self.find_optimal_speaker_count(embeddings)
            logger.info(f"Detected optimal speaker count: {optimal_speakers}")
            
            # Perform advanced clustering
            speaker_labels = self.advanced_clustering(embeddings, optimal_speakers)
            
            # Convert to speaker turns with temporal smoothing
            speaker_turns = self.temporal_smoothing(speaker_labels, timestamps)
            
            # Apply segment-level reassignment
            speaker_turns = self.segment_level_reassignment(speaker_turns, embeddings, timestamps)
            
            logger.info(f"Diarization complete: {len(speaker_turns)} speaker turns")
            return speaker_turns
            
        except Exception as e:
            logger.error(f"Advanced diarization failed: {e}")
            return self._fallback_diarization(audio_path)
    
    def extract_overlapping_embeddings(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, List[float]]:
        """Extract embeddings with overlapping windows for better accuracy."""
        try:
            # Initialize encoder if needed
            if self.encoder is None:
                self._initialize_encoder()
            
            window_size = self.config['window_size']
            step_size = self.config['step_size']
            min_clusters = self.config['min_clusters']
            max_clusters = self.config['max_clusters']
            
            embeddings = []
            timestamps = []
            
            # Extract embeddings with overlapping windows
            for i in range(0, len(audio) - int(window_size * sr), int(step_size * sr)):
                start_time = i / sr
                end_time = (i + int(window_size * sr)) / sr
                
                # Get audio segment
                segment = audio[i:i + int(window_size * sr)]
                
                # Skip segments that are too short
                if len(segment) < sr:  # Less than 1 second
                    continue
                
                # Extract embedding
                try:
                    embedding = self._extract_embedding(segment, sr)
                    if embedding is not None:
                        embeddings.append(embedding)
                        timestamps.append((start_time, end_time))
                except Exception as e:
                    logger.debug(f"Failed to extract embedding for segment {start_time}-{end_time}: {e}")
                    continue
            
            if not embeddings:
                logger.warning("No embeddings extracted")
                return np.array([]), []
            
            embeddings_array = np.array(embeddings)
            logger.info(f"Extracted {len(embeddings)} embeddings")
            
            return embeddings_array, timestamps
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return np.array([]), []
    
    def find_optimal_speaker_count(self, embeddings: np.ndarray) -> int:
        """Find optimal number of speakers using multiple metrics."""
        try:
            if len(embeddings) < 2:
                return 1
            
            max_speakers = min(len(embeddings) // 3, 8)
            min_speakers = 2
            
            if max_speakers < min_speakers:
                return min_speakers
            
            scores = []
            
            for n_speakers in range(min_speakers, max_speakers + 1):
                try:
                    # Try different clustering algorithms
                    for algorithm in ['kmeans', 'agglomerative', 'spectral']:
                        labels = self._cluster_speakers(embeddings, n_speakers, algorithm)
                        
                        if len(set(labels)) < 2:  # Skip if only one cluster
                            continue
                        
                        # Calculate metrics
                        sil_score = silhouette_score(embeddings, labels)
                        ch_score = calinski_harabasz_score(embeddings, labels)
                        
                        # Combined score with weights
                        combined_score = 0.7 * sil_score + 0.3 * (ch_score / 1000)
                        scores.append((combined_score, n_speakers, algorithm))
                        
                except Exception as e:
                    logger.debug(f"Clustering failed for {n_speakers} speakers: {e}")
                    continue
            
            if not scores:
                return 2  # Default fallback
            
            # Select best result
            best_score, best_n_speakers, best_algorithm = max(scores)
            logger.info(f"Best clustering: {best_n_speakers} speakers, {best_algorithm}, score: {best_score:.3f}")
            
            return best_n_speakers
            
        except Exception as e:
            logger.error(f"Optimal speaker count detection failed: {e}")
            return 2
    
    def advanced_clustering(self, embeddings: np.ndarray, n_speakers: int) -> np.ndarray:
        """Perform advanced clustering with multiple algorithms."""
        try:
            # Normalize embeddings
            embeddings_scaled = self.scaler.fit_transform(embeddings)
            
            # Try multiple clustering algorithms
            clustering_results = []
            
            # K-Means
            try:
                kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
                labels_kmeans = kmeans.fit_predict(embeddings_scaled)
                score_kmeans = silhouette_score(embeddings_scaled, labels_kmeans)
                clustering_results.append((score_kmeans, labels_kmeans, 'kmeans'))
            except Exception as e:
                logger.debug(f"K-Means clustering failed: {e}")
            
            # Agglomerative Clustering
            try:
                agg = AgglomerativeClustering(n_clusters=n_speakers, linkage='ward')
                labels_agg = agg.fit_predict(embeddings_scaled)
                score_agg = silhouette_score(embeddings_scaled, labels_agg)
                clustering_results.append((score_agg, labels_agg, 'agglomerative'))
            except Exception as e:
                logger.debug(f"Agglomerative clustering failed: {e}")
            
            # Spectral Clustering
            try:
                spectral = SpectralClustering(n_clusters=n_speakers, random_state=42)
                labels_spectral = spectral.fit_predict(embeddings_scaled)
                score_spectral = silhouette_score(embeddings_scaled, labels_spectral)
                clustering_results.append((score_spectral, labels_spectral, 'spectral'))
            except Exception as e:
                logger.debug(f"Spectral clustering failed: {e}")
            
            if not clustering_results:
                # Fallback to simple clustering
                logger.warning("All clustering algorithms failed, using fallback")
                return np.zeros(len(embeddings), dtype=int)
            
            # Select best clustering result
            best_score, best_labels, best_algorithm = max(clustering_results)
            logger.info(f"Best clustering algorithm: {best_algorithm} with score: {best_score:.3f}")
            
            return best_labels
            
        except Exception as e:
            logger.error(f"Advanced clustering failed: {e}")
            return np.zeros(len(embeddings), dtype=int)
    
    def temporal_smoothing(self, labels: np.ndarray, timestamps: List[Tuple[float, float]]) -> List[Tuple[float, float, str]]:
        """Apply temporal smoothing to reduce speaker switching noise."""
        try:
            if len(labels) != len(timestamps):
                logger.warning("Mismatch between labels and timestamps")
                return []
            
            # Apply median filtering for temporal smoothing
            window_size = 3
            smoothed_labels = []
            
            for i in range(len(labels)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(labels), i + window_size // 2 + 1)
                
                # Get window of labels
                window_labels = labels[start_idx:end_idx]
                
                # Use most frequent label in window
                unique, counts = np.unique(window_labels, return_counts=True)
                most_frequent = unique[np.argmax(counts)]
                smoothed_labels.append(most_frequent)
            
            # Convert to speaker turns
            speaker_turns = []
            current_speaker = None
            current_start = None
            
            for i, (label, (start, end)) in enumerate(zip(smoothed_labels, timestamps)):
                speaker_id = f"SPEAKER_{label}"
                
                if current_speaker is None:
                    current_speaker = speaker_id
                    current_start = start
                elif current_speaker != speaker_id:
                    # Speaker change - add previous turn
                    speaker_turns.append((current_start, start, current_speaker))
                    current_speaker = speaker_id
                    current_start = start
                
                # Add final turn
                if i == len(smoothed_labels) - 1:
                    speaker_turns.append((current_start, end, current_speaker))
            
            return speaker_turns
            
        except Exception as e:
            logger.error(f"Temporal smoothing failed: {e}")
            return []
    
    def segment_level_reassignment(self, speaker_turns: List[Tuple[float, float, str]], 
                                 embeddings: np.ndarray, timestamps: List[Tuple[float, float]]) -> List[Tuple[float, float, str]]:
        """Apply segment-level speaker reassignment for improved accuracy."""
        try:
            if not speaker_turns or len(embeddings) == 0:
                return speaker_turns
            
            # Group segments by speaker
            speaker_groups = defaultdict(list)
            for start, end, speaker in speaker_turns:
                speaker_groups[speaker].append((start, end))
            
            # Analyze each segment for potential reassignment
            reassigned_turns = []
            
            for start, end, speaker in speaker_turns:
                # Find corresponding embedding
                segment_embedding = self._find_segment_embedding(start, end, embeddings, timestamps)
                
                if segment_embedding is not None:
                    # Calculate similarity to all speaker groups
                    best_speaker = self._find_best_speaker_for_segment(
                        segment_embedding, speaker_groups, embeddings, timestamps
                    )
                    
                    if best_speaker and best_speaker != speaker:
                        logger.debug(f"Reassigned segment {start}-{end} from {speaker} to {best_speaker}")
                        reassigned_turns.append((start, end, best_speaker))
                    else:
                        reassigned_turns.append((start, end, speaker))
                else:
                    reassigned_turns.append((start, end, speaker))
            
            return reassigned_turns
            
        except Exception as e:
            logger.error(f"Segment-level reassignment failed: {e}")
            return speaker_turns
    
    def _initialize_encoder(self):
        """Initialize voice encoder."""
        try:
            # Try to import and initialize real encoder
            from resemblyzer import VoiceEncoder
            self.encoder = VoiceEncoder(device="cpu")
            logger.info("Initialized real Resemblyzer encoder")
        except ImportError:
            # Fallback to mock encoder
            from app.mock_backends import MockVoiceEncoder
            self.encoder = MockVoiceEncoder(device="cpu")
            logger.info("Using mock Resemblyzer encoder")
    
    def _extract_embedding(self, segment: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extract voice embedding from audio segment."""
        try:
            if self.encoder is None:
                return None
            
            # Preprocess segment
            from resemblyzer import preprocess_wav
            processed_segment = preprocess_wav(segment, source_sr=sr)
            
            # Extract embedding
            embedding = self.encoder.embed_utterance(processed_segment)
            return embedding
            
        except Exception as e:
            logger.debug(f"Embedding extraction failed: {e}")
            return None
    
    def _cluster_speakers(self, embeddings: np.ndarray, n_speakers: int, algorithm: str) -> np.ndarray:
        """Cluster speakers using specified algorithm."""
        if algorithm == 'kmeans':
            kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
            return kmeans.fit_predict(embeddings)
        elif algorithm == 'agglomerative':
            agg = AgglomerativeClustering(n_clusters=n_speakers, linkage='ward')
            return agg.fit_predict(embeddings)
        elif algorithm == 'spectral':
            spectral = SpectralClustering(n_clusters=n_speakers, random_state=42)
            return spectral.fit_predict(embeddings)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    
    def _find_segment_embedding(self, start: float, end: float, embeddings: np.ndarray, 
                               timestamps: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """Find embedding for a specific time segment."""
        try:
            for i, (ts_start, ts_end) in enumerate(timestamps):
                if ts_start <= start <= ts_end or ts_start <= end <= ts_end:
                    return embeddings[i]
            return None
        except Exception:
            return None
    
    def _find_best_speaker_for_segment(self, embedding: np.ndarray, speaker_groups: Dict, 
                                     embeddings: np.ndarray, timestamps: List[Tuple[float, float]]) -> Optional[str]:
        """Find the best speaker for a segment based on similarity."""
        try:
            best_speaker = None
            best_similarity = -1
            
            for speaker, segments in speaker_groups.items():
                # Calculate average similarity to this speaker's segments
                similarities = []
                
                for seg_start, seg_end in segments:
                    seg_embedding = self._find_segment_embedding(seg_start, seg_end, embeddings, timestamps)
                    if seg_embedding is not None:
                        similarity = np.dot(embedding, seg_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(seg_embedding)
                        )
                        similarities.append(similarity)
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    if avg_similarity > best_similarity and avg_similarity > 0.7:
                        best_similarity = avg_similarity
                        best_speaker = speaker
            
            return best_speaker
            
        except Exception as e:
            logger.debug(f"Best speaker detection failed: {e}")
            return None
    
    def _fallback_diarization(self, audio_path: str) -> List[Tuple[float, float, str]]:
        """Fallback diarization when advanced methods fail."""
        try:
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(y) / sr
            
            # Simple fallback: split into 2 speakers
            mid_point = duration / 2
            return [
                (0.0, mid_point, "SPEAKER_0"),
                (mid_point, duration, "SPEAKER_1")
            ]
        except Exception:
            return [(0.0, 10.0, "SPEAKER_0")]
    
    def get_default_config(self) -> Dict:
        """Get default diarization configuration."""
        return {
            'window_size': 3.0,  # seconds
            'step_size': 1.0,     # seconds
            'min_clusters': 1,
            'max_clusters': 8,
            'similarity_threshold': 0.7,
            'temporal_smoothing': True,
            'segment_reassignment': True
        }

# Plugin registration
def advanced_speaker_diarization(audio_path: str) -> List[Tuple[float, float, str]]:
    """Advanced speaker diarization plugin."""
    diarizer = AdvancedSpeakerDiarization()
    return diarizer.diarize_speakers(audio_path)
