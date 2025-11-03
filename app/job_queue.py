"""
AudioTranscribe: Background Job Queue
-------------------------------------
Background job processing using Redis Queue (RQ) for async transcription tasks.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import Redis and RQ
REDIS_AVAILABLE = False
RQ_AVAILABLE = False
redis_conn = None
Queue = None

try:
    import redis
    REDIS_AVAILABLE = True
    logger.info("Redis client available")
except ImportError:
    logger.warning("Redis not available - background jobs will not work. Install with: pip install redis")

try:
    from rq import Queue, Worker
    RQ_AVAILABLE = True
    logger.info("RQ (Redis Queue) available")
except ImportError:
    logger.warning("RQ not available - background jobs will not work. Install with: pip install rq")

# Initialize Redis connection if available
if REDIS_AVAILABLE and RQ_AVAILABLE:
    try:
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        redis_conn = redis.from_url(redis_url, socket_connect_timeout=5)
        # Test connection
        redis_conn.ping()
        logger.info(f"Connected to Redis at {redis_url}")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}. Background jobs disabled.")
        logger.warning("To enable background jobs: 1) Install Redis 2) Set REDIS_URL environment variable")
        REDIS_AVAILABLE = False
        redis_conn = None

# Create job queue if Redis is available
job_queue = None
if REDIS_AVAILABLE and RQ_AVAILABLE and redis_conn:
    try:
        job_queue = Queue('transcription', connection=redis_conn)
        logger.info("Background job queue initialized")
    except Exception as e:
        logger.warning(f"Failed to create job queue: {e}")
        job_queue = None

def is_background_processing_available() -> bool:
    """Check if background processing is available."""
    return job_queue is not None

def enqueue_job(job_function, *args, **kwargs):
    """Enqueue a job for background processing."""
    if not is_background_processing_available():
        raise RuntimeError("Background processing not available. Redis/RQ not configured.")
    
    try:
        job = job_queue.enqueue(job_function, *args, **kwargs, timeout='1h')
        logger.info(f"Job {job.id} enqueued for background processing")
        return job.id
    except Exception as e:
        logger.error(f"Failed to enqueue job: {e}")
        raise

def get_job_status(rq_job_id: str) -> Optional[dict]:
    """Get the status of a background RQ job by RQ job ID."""
    if not is_background_processing_available():
        return None
    
    try:
        from rq.job import Job
        job = Job.fetch(rq_job_id, connection=redis_conn)
        
        if job.is_finished:
            result = job.result
            return {
                'status': 'completed',
                'result': result if result else None,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'ended_at': job.ended_at.isoformat() if job.ended_at else None,
            }
        elif job.is_failed:
            return {
                'status': 'failed',
                'error': str(job.exc_info) if job.exc_info else 'Unknown error',
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'ended_at': job.ended_at.isoformat() if job.ended_at else None,
            }
        elif job.is_started:
            return {
                'status': 'processing',
                'started_at': job.started_at.isoformat() if job.started_at else None,
            }
        else:
            return {
                'status': 'queued',
                'created_at': job.created_at.isoformat() if job.created_at else None,
            }
    except Exception as e:
        logger.error(f"Failed to get RQ job status for {rq_job_id}: {e}")
        return None

