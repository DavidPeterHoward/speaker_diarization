#!/usr/bin/env python3
"""
Speaker Fine-Tuning API Endpoints
REST API for speaker identification, similarity scoring, and fine-tuning
"""

from flask import Blueprint, request, jsonify
import asyncio
from typing import Dict, Any, List
from pathlib import Path
import logging

from .speaker_fine_tuning import (
    SpeakerFineTuningSystem,
    create_speaker_profile_api,
    identify_speaker_api,
    fine_tune_speaker_model_api,
    get_similarity_matrix_api
)

logger = logging.getLogger(__name__)

# Create blueprint
speaker_bp = Blueprint('speaker', __name__)

# Global system instance (in production, this would be managed better)
speaker_system = None

def get_speaker_system() -> SpeakerFineTuningSystem:
    """Get or create speaker fine-tuning system instance"""
    global speaker_system
    if speaker_system is None:
        config = {
            'feature_extraction': {'sample_rate': 16000},
            'similarity_scoring': {'use_gpu': True},
            'fine_tuning': {'max_epochs': 10}
        }
        speaker_system = SpeakerFineTuningSystem(config)
    return speaker_system

@speaker_bp.route('/health', methods=['GET'])
def speaker_health():
    """Health check for speaker fine-tuning system"""
    return jsonify({
        'status': 'healthy',
        'service': 'speaker_fine_tuning',
        'capabilities': [
            'speaker_profile_creation',
            'speaker_identification',
            'similarity_scoring',
            'speaker_specific_fine_tuning'
        ]
    })

@speaker_bp.route('/profile/create', methods=['POST'])
async def create_speaker_profile():
    """
    Create speaker profile from audio files

    Expected JSON payload:
    {
        "speaker_id": "speaker_001",
        "audio_files": ["/path/to/audio1.wav", "/path/to/audio2.wav"],
        "metadata": {
            "name": "John Doe",
            "gender": "male",
            "accent": "american"
        }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        speaker_id = data.get('speaker_id')
        audio_files = data.get('audio_files', [])
        metadata = data.get('metadata', {})

        if not speaker_id:
            return jsonify({'error': 'speaker_id is required'}), 400

        if not audio_files:
            return jsonify({'error': 'audio_files list is required'}), 400

        # Validate audio files exist
        missing_files = []
        for audio_file in audio_files:
            if not Path(audio_file).exists():
                missing_files.append(audio_file)

        if missing_files:
            return jsonify({
                'error': f'Audio files not found: {missing_files}'
            }), 400

        # Create speaker profile
        result = await create_speaker_profile_api(speaker_id, audio_files)

        # Add metadata if provided
        if metadata:
            result['speaker_profile']['metadata'] = metadata

        logger.info(f"Created speaker profile for {speaker_id} with {len(audio_files)} audio files")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error creating speaker profile: {e}")
        return jsonify({'error': str(e)}), 500

@speaker_bp.route('/identify', methods=['POST'])
async def identify_speaker():
    """
    Identify speaker in audio segment

    Expected JSON payload:
    {
        "audio_file": "/path/to/audio_segment.wav",
        "confidence_threshold": 0.75,
        "include_alternatives": true
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        audio_file = data.get('audio_file')
        confidence_threshold = data.get('confidence_threshold', 0.75)
        include_alternatives = data.get('include_alternatives', True)

        if not audio_file:
            return jsonify({'error': 'audio_file is required'}), 400

        if not Path(audio_file).exists():
            return jsonify({'error': f'Audio file not found: {audio_file}'}), 400

        # Identify speaker
        result = await identify_speaker_api(audio_file, confidence_threshold)

        # Optionally remove alternatives if not requested
        if not include_alternatives and 'identification' in result:
            result['identification'].pop('alternatives', None)
            result['identification'].pop('all_similarities', None)

        logger.info(f"Speaker identification completed for {audio_file}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error identifying speaker: {e}")
        return jsonify({'error': str(e)}), 500

@speaker_bp.route('/similarity/matrix', methods=['GET'])
async def get_similarity_matrix():
    """
    Get similarity matrix for all known speakers

    Query parameters:
    - format: "full" or "summary" (default: "full")
    - threshold: minimum similarity to include (default: 0.0)
    """
    try:
        format_type = request.args.get('format', 'full')
        threshold = float(request.args.get('threshold', 0.0))

        # Get similarity matrix
        result = await get_similarity_matrix_api()

        if format_type == 'summary':
            # Create summary format
            matrix = result.get('similarity_matrix', {})
            summary = {
                'total_speakers': len(matrix),
                'speaker_ids': list(matrix.keys()),
                'highly_similar_pairs': [],
                'average_similarity': 0.0,
                'similarity_distribution': {}
            }

            all_similarities = []
            for speaker1, similarities in matrix.items():
                for speaker2, similarity in similarities.items():
                    if speaker1 != speaker2 and similarity >= threshold:
                        all_similarities.append(similarity)
                        if similarity >= 0.9:
                            summary['highly_similar_pairs'].append({
                                'speaker1': speaker1,
                                'speaker2': speaker2,
                                'similarity': similarity
                            })

            if all_similarities:
                summary['average_similarity'] = sum(all_similarities) / len(all_similarities)
                summary['similarity_distribution'] = {
                    'high': len([s for s in all_similarities if s >= 0.8]),
                    'medium': len([s for s in all_similarities if 0.6 <= s < 0.8]),
                    'low': len([s for s in all_similarities if s < 0.6])
                }

            result['similarity_matrix'] = summary

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error getting similarity matrix: {e}")
        return jsonify({'error': str(e)}), 500

@speaker_bp.route('/similarity/compare', methods=['POST'])
async def compare_speakers():
    """
    Compare similarity between two specific speakers

    Expected JSON payload:
    {
        "speaker1_id": "speaker_001",
        "speaker2_id": "speaker_002",
        "detailed_analysis": true
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        speaker1_id = data.get('speaker1_id')
        speaker2_id = data.get('speaker2_id')
        detailed_analysis = data.get('detailed_analysis', True)

        if not speaker1_id or not speaker2_id:
            return jsonify({'error': 'Both speaker1_id and speaker2_id are required'}), 400

        # Get speaker system
        system = get_speaker_system()

        # Check if speakers exist
        if speaker1_id not in system.speaker_profiles:
            return jsonify({'error': f'Speaker profile not found: {speaker1_id}'}), 404

        if speaker2_id not in system.speaker_profiles:
            return jsonify({'error': f'Speaker profile not found: {speaker2_id}'}), 404

        # Compute similarity
        similarity_score = await system.compute_speaker_similarity(speaker1_id, speaker2_id)

        result = {
            'speaker1_id': speaker1_id,
            'speaker2_id': speaker2_id,
            'similarity_score': similarity_score,
            'similarity_category': get_similarity_category(similarity_score)
        }

        if detailed_analysis:
            # Add detailed comparison
            profile1 = system.speaker_profiles[speaker1_id]['voice_characteristics']
            profile2 = system.speaker_profiles[speaker2_id]['voice_characteristics']

            result['detailed_comparison'] = {
                'feature_similarities': {},
                'key_differences': [],
                'confidence_assessment': assess_similarity_confidence(similarity_score, profile1, profile2)
            }

            # Compare key features
            key_features = ['fundamental_frequency', 'spectral_centroid', 'voice_activity_ratio']
            for feature in key_features:
                val1 = profile1.get(feature, 0)
                val2 = profile2.get(feature, 0)
                if val1 != 0 and val2 != 0:
                    diff_percent = abs(val1 - val2) / max(abs(val1), abs(val2))
                    result['detailed_comparison']['feature_similarities'][feature] = 1.0 - diff_percent

        return jsonify({
            'success': True,
            'comparison': result
        })

    except Exception as e:
        logger.error(f"Error comparing speakers: {e}")
        return jsonify({'error': str(e)}), 500

@speaker_bp.route('/fine-tune', methods=['POST'])
async def fine_tune_speaker_model():
    """
    Fine-tune transcription model for specific speaker

    Expected JSON payload:
    {
        "speaker_id": "speaker_001",
        "training_files": ["/path/to/audio1.wav", "/path/to/audio2.wav"],
        "validation_files": ["/path/to/val1.wav"],
        "model_config": {
            "base_model": "whisper_base",
            "learning_rate": 1e-5,
            "epochs": 10
        }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        speaker_id = data.get('speaker_id')
        training_files = data.get('training_files', [])
        validation_files = data.get('validation_files', [])
        model_config = data.get('model_config', {})

        if not speaker_id:
            return jsonify({'error': 'speaker_id is required'}), 400

        if not training_files:
            return jsonify({'error': 'training_files list is required'}), 400

        # Validate training files exist
        missing_files = []
        for audio_file in training_files:
            if not Path(audio_file).exists():
                missing_files.append(audio_file)

        if missing_files:
            return jsonify({
                'error': f'Training files not found: {missing_files}'
            }), 400

        # Validate validation files if provided
        if validation_files:
            missing_val_files = []
            for audio_file in validation_files:
                if not Path(audio_file).exists():
                    missing_val_files.append(audio_file)

            if missing_val_files:
                return jsonify({
                    'error': f'Validation files not found: {missing_val_files}'
                }), 400

        # Fine-tune model
        result = await fine_tune_speaker_model_api(speaker_id, training_files)

        # Add validation results if validation files provided
        if validation_files:
            result['fine_tuning']['validation_performed'] = True
            result['fine_tuning']['validation_files_count'] = len(validation_files)
        else:
            result['fine_tuning']['validation_performed'] = False

        logger.info(f"Fine-tuning completed for speaker {speaker_id}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error fine-tuning speaker model: {e}")
        return jsonify({'error': str(e)}), 500

@speaker_bp.route('/profiles', methods=['GET'])
def list_speaker_profiles():
    """
    List all speaker profiles

    Query parameters:
    - detailed: include full profile data (default: false)
    - speaker_id: filter by specific speaker
    """
    try:
        system = get_speaker_system()
        detailed = request.args.get('detailed', 'false').lower() == 'true'
        speaker_id_filter = request.args.get('speaker_id')

        profiles = {}

        for speaker_id, profile in system.speaker_profiles.items():
            if speaker_id_filter and speaker_id != speaker_id_filter:
                continue

            if detailed:
                profiles[speaker_id] = profile
            else:
                # Summary view
                profiles[speaker_id] = {
                    'speaker_id': speaker_id,
                    'sample_count': profile.get('sample_count', 0),
                    'confidence_score': profile.get('confidence_score', 0.0),
                    'created_at': profile.get('created_at'),
                    'has_fine_tuned_model': speaker_id in system.speaker_models
                }

        return jsonify({
            'success': True,
            'total_profiles': len(profiles),
            'profiles': profiles
        })

    except Exception as e:
        logger.error(f"Error listing speaker profiles: {e}")
        return jsonify({'error': str(e)}), 500

@speaker_bp.route('/models', methods=['GET'])
def list_speaker_models():
    """
    List all fine-tuned speaker models

    Query parameters:
    - speaker_id: filter by specific speaker
    - include_performance: include performance metrics (default: true)
    """
    try:
        system = get_speaker_system()
        speaker_id_filter = request.args.get('speaker_id')
        include_performance = request.args.get('include_performance', 'true').lower() == 'true'

        models = {}

        for speaker_id, model_info in system.speaker_models.items():
            if speaker_id_filter and speaker_id != speaker_id_filter:
                continue

            models[speaker_id] = {
                'speaker_id': speaker_id,
                'model_path': model_info.get('model_path'),
                'fine_tuned_at': model_info.get('fine_tuned_at'),
                'training_samples': model_info.get('training_samples', 0)
            }

            if include_performance:
                models[speaker_id]['performance_metrics'] = model_info.get('performance_metrics', {})

        return jsonify({
            'success': True,
            'total_models': len(models),
            'models': models
        })

    except Exception as e:
        logger.error(f"Error listing speaker models: {e}")
        return jsonify({'error': str(e)}), 500

@speaker_bp.route('/cluster', methods=['POST'])
async def cluster_speakers():
    """
    Cluster speakers based on similarity

    Expected JSON payload:
    {
        "similarity_threshold": 0.8,
        "clustering_method": "similarity_based"
    }
    """
    try:
        data = request.get_json() or {}
        similarity_threshold = data.get('similarity_threshold', 0.8)
        clustering_method = data.get('clustering_method', 'similarity_based')

        system = get_speaker_system()

        if clustering_method == 'similarity_based':
            clusters = await system.cluster_speakers_by_similarity(similarity_threshold)
        else:
            return jsonify({'error': f'Unsupported clustering method: {clustering_method}'}), 400

        # Analyze clusters
        cluster_analysis = {
            'total_clusters': len(clusters),
            'largest_cluster_size': max(len(members) for members in clusters.values()) if clusters else 0,
            'average_cluster_size': sum(len(members) for members in clusters.values()) / len(clusters) if clusters else 0,
            'single_speaker_clusters': len([c for c in clusters.values() if len(c) == 1])
        }

        return jsonify({
            'success': True,
            'clustering': {
                'method': clustering_method,
                'threshold': similarity_threshold,
                'clusters': clusters,
                'analysis': cluster_analysis
            }
        })

    except Exception as e:
        logger.error(f"Error clustering speakers: {e}")
        return jsonify({'error': str(e)}), 500

# Helper functions
def get_similarity_category(similarity: float) -> str:
    """Categorize similarity score"""
    if similarity >= 0.9:
        return "identical"
    elif similarity >= 0.8:
        return "very_similar"
    elif similarity >= 0.7:
        return "similar"
    elif similarity >= 0.6:
        return "moderately_similar"
    elif similarity >= 0.5:
        return "somewhat_similar"
    else:
        return "different"

def assess_similarity_confidence(similarity: float, profile1: Dict[str, Any], profile2: Dict[str, Any]) -> Dict[str, Any]:
    """Assess confidence in similarity assessment"""
    confidence_factors = []

    # Sample count confidence
    sample_count = min(
        profile1.get('sample_count', 1),
        profile2.get('sample_count', 1)
    )
    sample_confidence = min(1.0, sample_count / 5.0)  # 5+ samples = high confidence
    confidence_factors.append(('sample_size', sample_confidence))

    # Feature completeness confidence
    features1 = set(profile1.keys())
    features2 = set(profile2.keys())
    common_features = len(features1.intersection(features2))
    total_features = len(features1.union(features2))
    completeness_confidence = common_features / total_features if total_features > 0 else 0.0
    confidence_factors.append(('feature_completeness', completeness_confidence))

    # Overall confidence
    base_confidence = similarity  # Higher similarity = higher confidence
    weighted_confidence = sum(factor[1] for factor in confidence_factors) / len(confidence_factors)
    overall_confidence = (base_confidence + weighted_confidence) / 2

    return {
        'overall_confidence': overall_confidence,
        'confidence_factors': dict(confidence_factors),
        'confidence_level': 'high' if overall_confidence >= 0.8 else 'medium' if overall_confidence >= 0.6 else 'low'
    }

# Register the blueprint in the main app
def register_speaker_api(app):
    """Register speaker API blueprint with the Flask app"""
    app.register_blueprint(speaker_bp, url_prefix='/api/speaker')
    logger.info("Speaker fine-tuning API registered at /api/speaker")
