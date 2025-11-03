"""
Enhanced API Endpoints
======================
API endpoints for the enhanced transcription system with all improvements.
"""

import logging
from flask import Blueprint, request, jsonify, current_app
import asyncio
from typing import Dict, List, Optional, Any
import os
import uuid
from datetime import datetime

# Import enhanced system
from app.enhanced_transcription import (
    process_audio_enhanced,
    start_continuous_improvement,
    get_system_status,
    add_domain_vocabulary,
    get_improvement_recommendations
)

# Import individual components
from app.meta_recursive_feedback import (
    start_feedback_loop,
    get_feedback_status,
    get_improvement_history
)
from app.machine_learning_finetuning import (
    start_fine_tuning,
    add_feedback_data,
    evaluate_finetuned_model,
    transcribe_with_finetuned_model
)
from app.custom_vocabulary import vocabulary_manager
from app.quality_monitoring import quality_monitor

logger = logging.getLogger(__name__)

# Create enhanced API blueprint
enhanced_api = Blueprint('enhanced_api', __name__, url_prefix='/api/enhanced')

@enhanced_api.route('/process', methods=['POST'])
async def process_audio():
    """Process audio with all enhancements."""
    try:
        data = request.get_json()
        audio_path = data.get('audio_path')
        domain = data.get('domain')
        use_feedback_loop = data.get('use_feedback_loop', False)
        use_finetuned_model = data.get('use_finetuned_model')
        
        if not audio_path:
            return jsonify({'error': 'audio_path is required'}), 400
        
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        # Process with enhanced system
        result = await process_audio_enhanced(
            audio_path=audio_path,
            domain=domain,
            use_feedback_loop=use_feedback_loop,
            use_finetuned_model=use_finetuned_model
        )
        
        if result.success:
            response_data = {
                'success': True,
                'message': result.message,
                'job_id': result.job.id if result.job else None,
                'segments': [seg.to_dict() for seg in result.job.segments] if result.job else [],
                'enhancement_metadata': getattr(result, 'enhancement_metadata', {})
            }
            return jsonify(response_data)
        else:
            return jsonify({
                'success': False,
                'error': result.message
            }), 500
            
    except Exception as e:
        logger.error(f"Enhanced processing failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/improvement/start', methods=['POST'])
async def start_improvement():
    """Start continuous improvement process."""
    try:
        data = request.get_json()
        target_accuracy = data.get('target_accuracy', 0.95)
        
        session_id = await start_continuous_improvement(target_accuracy)
        
        if session_id:
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Continuous improvement started'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to start continuous improvement'
            }), 500
            
    except Exception as e:
        logger.error(f"Improvement startup failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/feedback/start', methods=['POST'])
async def start_feedback():
    """Start feedback loop session."""
    try:
        data = request.get_json()
        target_accuracy = data.get('target_accuracy', 0.95)
        
        session_id = await start_feedback_loop(target_accuracy)
        
        if session_id:
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Feedback loop started'
            })
        else:
            return jsonify({'error': 'Failed to start feedback loop'}), 500
            
    except Exception as e:
        logger.error(f"Feedback loop startup failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/feedback/status/<session_id>')
async def get_feedback_status_endpoint(session_id):
    """Get feedback session status."""
    try:
        status = await get_feedback_status(session_id)
        
        if status:
            return jsonify(status)
        else:
            return jsonify({'error': 'Session not found'}), 404
            
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/feedback/history')
async def get_feedback_history():
    """Get feedback improvement history."""
    try:
        history = get_improvement_history()
        return jsonify({'history': history})
        
    except Exception as e:
        logger.error(f"History retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/finetuning/start', methods=['POST'])
async def start_finetuning():
    """Start fine-tuning process."""
    try:
        model_name = await start_fine_tuning()
        
        if model_name:
            return jsonify({
                'success': True,
                'model_name': model_name,
                'message': 'Fine-tuning started'
            })
        else:
            return jsonify({'error': 'Failed to start fine-tuning'}), 500
            
    except Exception as e:
        logger.error(f"Fine-tuning startup failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/finetuning/transcribe', methods=['POST'])
async def transcribe_finetuned():
    """Transcribe using fine-tuned model."""
    try:
        data = request.get_json()
        audio_path = data.get('audio_path')
        model_name = data.get('model_name')
        
        if not audio_path:
            return jsonify({'error': 'audio_path is required'}), 400
        
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        transcription = await transcribe_with_finetuned_model(audio_path, model_name)
        
        return jsonify({
            'success': True,
            'transcription': transcription
        })
        
    except Exception as e:
        logger.error(f"Fine-tuned transcription failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/vocabulary/add', methods=['POST'])
async def add_vocabulary():
    """Add domain vocabulary."""
    try:
        data = request.get_json()
        domain = data.get('domain')
        terms = data.get('terms', [])
        boost_factors = data.get('boost_factors', [])
        
        if not domain or not terms:
            return jsonify({'error': 'domain and terms are required'}), 400
        
        success = await add_domain_vocabulary(domain, terms, boost_factors)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Added {len(terms)} terms to {domain} vocabulary'
            })
        else:
            return jsonify({'error': 'Failed to add vocabulary'}), 500
            
    except Exception as e:
        logger.error(f"Vocabulary addition failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/vocabulary/domains')
async def get_vocabulary_domains():
    """Get available vocabulary domains."""
    try:
        domains = list(vocabulary_manager.domains.keys())
        return jsonify({'domains': domains})
        
    except Exception as e:
        logger.error(f"Domain retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/quality/assess', methods=['POST'])
async def assess_quality():
    """Assess audio quality."""
    try:
        data = request.get_json()
        audio_path = data.get('audio_path')
        
        if not audio_path:
            return jsonify({'error': 'audio_path is required'}), 400
        
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        # Assess audio quality
        metrics = quality_monitor.assess_audio_quality(audio_path)
        recommendations = quality_monitor.generate_recommendations(metrics)
        
        return jsonify({
            'success': True,
            'metrics': {
                'snr': metrics.snr,
                'clarity': metrics.clarity,
                'noise_level': metrics.noise_level,
                'speech_rate': metrics.speech_rate,
                'overlap_ratio': metrics.overlap_ratio,
                'overall_quality': metrics.overall_quality
            },
            'recommendations': recommendations.recommendations,
            'priority': recommendations.priority,
            'expected_improvement': recommendations.expected_improvement
        })
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/recommendations/<audio_path>')
async def get_recommendations(audio_path):
    """Get improvement recommendations for audio."""
    try:
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        recommendations = await get_improvement_recommendations(audio_path)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Recommendation retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/status')
async def get_status():
    """Get comprehensive system status."""
    try:
        status = await get_system_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/models/finetuned')
async def get_finetuned_models():
    """Get available fine-tuned models."""
    try:
        from app.machine_learning_finetuning import fine_tuning_system
        models = fine_tuning_system.get_available_models()
        return jsonify({'models': models})
        
    except Exception as e:
        logger.error(f"Model retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/models/training-history')
async def get_training_history():
    """Get fine-tuning training history."""
    try:
        from app.machine_learning_finetuning import fine_tuning_system
        history = fine_tuning_system.get_training_history()
        return jsonify({'history': history})
        
    except Exception as e:
        logger.error(f"Training history retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/feedback/data', methods=['POST'])
async def add_feedback_data():
    """Add feedback data for fine-tuning."""
    try:
        data = request.get_json()
        feedback_data = data.get('feedback_data', [])
        
        if not feedback_data:
            return jsonify({'error': 'feedback_data is required'}), 400
        
        await add_feedback_data(feedback_data)
        
        return jsonify({
            'success': True,
            'message': f'Added {len(feedback_data)} feedback examples'
        })
        
    except Exception as e:
        logger.error(f"Feedback data addition failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/evaluate', methods=['POST'])
async def evaluate_model():
    """Evaluate a fine-tuned model."""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        test_examples = data.get('test_examples', [])
        
        if not model_name or not test_examples:
            return jsonify({'error': 'model_name and test_examples are required'}), 400
        
        metrics = await evaluate_finetuned_model(model_name, test_examples)
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@enhanced_api.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@enhanced_api.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Health check endpoint
@enhanced_api.route('/health')
async def health_check():
    """Health check endpoint."""
    try:
        status = await get_system_status()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': status.get('enhancement_components', {})
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500
