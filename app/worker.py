"""
AudioTranscribe: Background Worker
----------------------------------
Worker process for processing transcription jobs asynchronously.
Run with: python -m rq worker transcription
Or use: rq worker transcription
"""

import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_queue import redis_conn, is_background_processing_available
from transcription import process_audio_file
from models import get_job, save_job, TranscriptionState

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_transcription_job(job_id: str, file_path: str, model_size: str, 
                               transcription_backend: str, diarization_backend: str) -> dict:
    """
    Process a transcription job in the background.
    This function is called by RQ worker.
    """
    logger.info(f"Processing transcription job {job_id} for file {file_path}")
    
    try:
        # Get job from database
        job = get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found in database")
            return {'success': False, 'error': 'Job not found'}
        
        # Update job state to processing
        job.state = TranscriptionState.PROCESSING
        save_job(job)
        
        # Process the audio file
        result = process_audio_file(
            file_path,
            model_size=model_size,
            transcription_backend=transcription_backend,
            diarization_backend=diarization_backend
        )
        
        # Return result (which includes updated job)
        if result.success:
            logger.info(f"Job {job_id} completed successfully")
            return {
                'success': True,
                'job_id': job_id,
                'message': result.message
            }
        else:
            logger.error(f"Job {job_id} failed: {result.message}")
            # Update job state in database
            if result.job:
                result.job.state = TranscriptionState.FAILED
                result.job.error = result.message
                save_job(result.job)
            
            return {
                'success': False,
                'job_id': job_id,
                'error': result.message
            }
            
    except Exception as e:
        logger.exception(f"Error processing job {job_id}: {e}")
        # Update job state in database
        try:
            job = get_job(job_id)
            if job:
                job.state = TranscriptionState.FAILED
                job.error = str(e)
                save_job(job)
        except Exception as db_error:
            logger.error(f"Failed to update job status: {db_error}")
        
        return {
            'success': False,
            'job_id': job_id,
            'error': str(e)
        }

if __name__ == '__main__':
    if not is_background_processing_available():
        logger.error("Background processing not available. Redis/RQ not configured.")
        logger.error("To enable background processing:")
        logger.error("1. Install Redis: https://redis.io/download")
        logger.error("2. Install Python packages: pip install redis rq")
        logger.error("3. Set REDIS_URL environment variable (default: redis://localhost:6379/0)")
        logger.error("4. Start Redis server")
        sys.exit(1)
    
    logger.info("Starting RQ worker for transcription jobs")
    logger.info("Worker will process jobs from 'transcription' queue")
    
    # Import and start worker
    from rq import Worker, Queue, Connection
    
    with Connection(redis_conn):
        worker = Worker(['transcription'])
        worker.work()

