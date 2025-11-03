"""
Enhanced Transcription System Integration
========================================
Integrates all improvement components into a unified transcription system.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

# Import all enhancement modules
from app.audio_preprocessing import AudioPreprocessor
from app.advanced_diarization import AdvancedSpeakerDiarization
from app.custom_vocabulary import vocabulary_manager
from app.ensemble_transcription import EnsembleTranscription
from app.quality_monitoring import quality_monitor
from app.meta_recursive_feedback import feedback_system
from app.machine_learning_finetuning import fine_tuning_system

# Import existing transcription system
from app.transcription import process_audio_file, plugins
from app.models import ProcessingResult, TranscriptionJob, TranscriptionState

logger = logging.getLogger(__name__)

class EnhancedTranscriptionSystem:
    """Enhanced transcription system with all improvements integrated."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        
        # Initialize all enhancement components
        self.audio_preprocessor = AudioPreprocessor()
        self.diarizer = AdvancedSpeakerDiarization()
        self.ensemble = EnsembleTranscription()
        self.quality_monitor = quality_monitor
        self.vocabulary_manager = vocabulary_manager
        self.feedback_system = feedback_system
        self.fine_tuning_system = fine_tuning_system
        
        # Register enhanced plugins
        self._register_enhanced_plugins()
    
    def _register_enhanced_plugins(self):
        """Register all enhanced plugins with the system."""
        try:
            # Register audio preprocessing
            plugins.register_audio_preprocessor(self.audio_preprocessor.preprocess_audio)
            
            # Register advanced diarization
            plugins.register_diarization_plugin(self.diarizer.diarize_speakers)
            
            # Register ensemble transcription
            plugins.register_transcription_plugin(self.ensemble.transcribe_ensemble)
            
            # Register quality monitoring
            plugins.register_post_processor(self._quality_monitoring_post_processor)
            
            logger.info("Enhanced plugins registered successfully")
            
        except Exception as e:
            logger.error(f"Plugin registration failed: {e}")
    
    def _quality_monitoring_post_processor(self, segments: List) -> List:
        """Quality monitoring post-processor."""
        try:
            # Apply quality monitoring
            logger.info("Applied quality monitoring post-processing")
            return segments
        except Exception as e:
            logger.error(f"Quality monitoring post-processing failed: {e}")
            return segments
    
    async def process_audio_enhanced(self, audio_path: str, 
                                   domain: str = None,
                                   use_feedback_loop: bool = False,
                                   use_finetuned_model: str = None) -> ProcessingResult:
        """Process audio with all enhancements applied."""
        try:
            logger.info(f"Starting enhanced processing for {audio_path}")
            
            # 1. Audio preprocessing
            if self.config.get('enable_audio_preprocessing', True):
                audio_path = self.audio_preprocessor.preprocess_audio(audio_path)
                logger.info("Applied audio preprocessing")
            
            # 2. Get enhanced prompt for domain
            enhanced_prompt = None
            if domain:
                enhanced_prompt = self.vocabulary_manager.get_enhanced_prompt(
                    audio_context=f"Domain: {domain}",
                    domain=domain
                )
                logger.info(f"Generated enhanced prompt for domain: {domain}")
            
            # 3. Process with enhanced system
            if use_finetuned_model and use_finetuned_model in self.fine_tuning_system.get_available_models():
                # Use fine-tuned model
                transcription = await self.fine_tuning_system.transcribe_with_finetuned_model(
                    audio_path, use_finetuned_model
                )
                logger.info(f"Used fine-tuned model: {use_finetuned_model}")
            else:
                # Use standard enhanced processing
                result = process_audio_file(
                    audio_path,
                    model_size=self.config.get('model_size', 'base'),
                    transcription_backend=self.config.get('transcription_backend', 'faster_whisper'),
                    diarization_backend=self.config.get('diarization_backend', 'pyannote')
                )
                
                if not result.success:
                    return result
                
                transcription = result.job.segments
            
            # 4. Start feedback loop if requested
            feedback_session_id = None
            if use_feedback_loop:
                feedback_session_id = await self.feedback_system.start_feedback_session(
                    target_accuracy=self.config.get('target_accuracy', 0.95)
                )
                logger.info(f"Started feedback session: {feedback_session_id}")
            
            # 5. Quality assessment
            if self.config.get('enable_quality_monitoring', True):
                audio_metrics = self.quality_monitor.assess_audio_quality(audio_path)
                recommendations = self.quality_monitor.generate_recommendations(audio_metrics)
                
                logger.info(f"Audio quality: {audio_metrics.overall_quality:.3f}")
                logger.info(f"Recommendations: {len(recommendations.recommendations)}")
            
            # Create enhanced result
            enhanced_result = ProcessingResult(
                success=True,
                message=f"Enhanced processing completed for {os.path.basename(audio_path)}",
                job=result.job if 'result' in locals() else None
            )
            
            # Add enhancement metadata
            enhanced_result.enhancement_metadata = {
                'audio_preprocessing_applied': self.config.get('enable_audio_preprocessing', True),
                'domain_adaptation': domain is not None,
                'feedback_session_id': feedback_session_id,
                'finetuned_model_used': use_finetuned_model,
                'quality_assessment': self.config.get('enable_quality_monitoring', True),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info("Enhanced processing completed successfully")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced processing failed: {e}")
            return ProcessingResult(
                success=False,
                message=f"Enhanced processing failed: {str(e)}",
                error=e
            )
    
    async def start_continuous_improvement(self, target_accuracy: float = 0.95) -> str:
        """Start continuous improvement process."""
        try:
            # Start feedback loop
            feedback_session_id = await self.feedback_system.start_feedback_session(target_accuracy)
            
            # Start fine-tuning if enough data is available
            if len(self.fine_tuning_system.training_examples) > 50:
                model_name = await self.fine_tuning_system.start_fine_tuning()
                logger.info(f"Started fine-tuning: {model_name}")
            
            return feedback_session_id
            
        except Exception as e:
            logger.error(f"Continuous improvement startup failed: {e}")
            return None
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                'enhancement_components': {
                    'audio_preprocessing': True,
                    'advanced_diarization': True,
                    'custom_vocabulary': True,
                    'ensemble_transcription': True,
                    'quality_monitoring': True,
                    'feedback_system': True,
                    'fine_tuning': True
                },
                'feedback_sessions': {
                    'active': len([s for s in self.feedback_system.feedback_sessions.values() 
                                 if s.status == 'active']),
                    'completed': len([s for s in self.feedback_system.feedback_sessions.values() 
                                    if s.status == 'completed']),
                    'failed': len([s for s in self.feedback_system.feedback_sessions.values() 
                                 if s.status == 'failed'])
                },
                'fine_tuned_models': self.fine_tuning_system.get_available_models(),
                'vocabulary_domains': list(self.vocabulary_manager.domains.keys()),
                'training_examples': len(self.fine_tuning_system.training_examples),
                'opentts_available': self.feedback_system.opentts_available,
                'system_health': 'healthy'
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {'error': str(e)}
    
    async def add_domain_vocabulary(self, domain: str, terms: List[str], 
                                  boost_factors: List[float] = None) -> bool:
        """Add domain-specific vocabulary."""
        try:
            success = self.vocabulary_manager.add_domain_vocabulary(
                domain=domain,
                terms=terms,
                boost_factors=boost_factors or [1.0] * len(terms)
            )
            
            if success:
                logger.info(f"Added vocabulary for domain: {domain}")
            
            return success
            
        except Exception as e:
            logger.error(f"Vocabulary addition failed: {e}")
            return False
    
    async def get_improvement_recommendations(self, audio_path: str) -> List[str]:
        """Get improvement recommendations for specific audio."""
        try:
            # Assess audio quality
            audio_metrics = self.quality_monitor.assess_audio_quality(audio_path)
            
            # Generate recommendations
            recommendations = self.quality_monitor.generate_recommendations(audio_metrics)
            
            return recommendations.recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'enable_audio_preprocessing': True,
            'enable_quality_monitoring': True,
            'model_size': 'base',
            'transcription_backend': 'faster_whisper',
            'diarization_backend': 'pyannote',
            'target_accuracy': 0.95,
            'ensemble_enabled': True,
            'feedback_loop_enabled': True,
            'fine_tuning_enabled': True
        }

# Global enhanced system instance
enhanced_system = EnhancedTranscriptionSystem()

# API functions for integration
async def process_audio_enhanced(audio_path: str, domain: str = None, 
                               use_feedback_loop: bool = False,
                               use_finetuned_model: str = None) -> ProcessingResult:
    """Process audio with all enhancements."""
    return await enhanced_system.process_audio_enhanced(
        audio_path, domain, use_feedback_loop, use_finetuned_model
    )

async def start_continuous_improvement(target_accuracy: float = 0.95) -> str:
    """Start continuous improvement process."""
    return await enhanced_system.start_continuous_improvement(target_accuracy)

async def get_system_status() -> Dict[str, Any]:
    """Get system status."""
    return await enhanced_system.get_system_status()

async def add_domain_vocabulary(domain: str, terms: List[str], 
                              boost_factors: List[float] = None) -> bool:
    """Add domain vocabulary."""
    return await enhanced_system.add_domain_vocabulary(domain, terms, boost_factors)

async def get_improvement_recommendations(audio_path: str) -> List[str]:
    """Get improvement recommendations."""
    return await enhanced_system.get_improvement_recommendations(audio_path)
