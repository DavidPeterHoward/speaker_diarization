"""
AudioTranscribe: Flask Web Application
-------------------------------------
Flask routes and web application functionality.
"""

# Suppress non-critical warnings before importing dependencies
try:
    from .suppress_warnings import suppress_dependency_warnings
    suppress_dependency_warnings()
except ImportError:
    # Fallback if suppress_warnings doesn't exist
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from .models import (
    Config, Speaker, TranscriptionJob, TranscriptionState,
    get_job, list_jobs, list_speakers, rename_speaker, allowed_file, save_job, hash_file
)
from .transcription import (
    TRANSCRIPTION_BACKENDS, DIARIZATION_BACKENDS,
    process_audio_file, generate_json_output, generate_srt_subtitle
)
from .validation import (
    FileValidator, InputValidator, ValidationError, SecurityError
)
from .error_handlers import register_error_handlers, create_error_response

# Get logger
logger = logging.getLogger(__name__)

# Try to import background job queue
try:
    from app.job_queue import is_background_processing_available, enqueue_job, get_job_status as get_bg_job_status
    BACKGROUND_JOBS_AVAILABLE = is_background_processing_available()
except ImportError:
    BACKGROUND_JOBS_AVAILABLE = False
    logger.info("Background job processing not available")

# Try to import meta-recursive API
try:
    from .meta_recursive_api import meta_recursive_bp
    META_RECURSIVE_AVAILABLE = True
    logger.info("Meta-recursive API available")
except ImportError as e:
    logger.warning(f"Meta-recursive API not available: {e}")
    META_RECURSIVE_AVAILABLE = False
    meta_recursive_bp = None

# logger defined above

# ------------------------ FLASK WEB APPLICATION ------------------------ #
import os
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = Config.SECRET_KEY
logger.info(f"Template directory: {template_dir}")

# Health Check Route
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    try:
        return jsonify({
            'status': 'healthy',
            'service': 'AudioTranscribe',
            'version': '1.0.0',
            'timestamp': time.time()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 500

# API Routes
@app.route('/api/jobs')
def list_jobs_api() -> Tuple[Dict[str, Any], int]:
    """API endpoint to list jobs with pagination."""
    try:
        # Get pagination parameters with validation
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Validate parameters
        limit = InputValidator.validate_number(limit, min_value=1, max_value=1000)
        offset = InputValidator.validate_number(offset, min_value=0)
        
        jobs = list_jobs(limit=limit, offset=offset)
        return jsonify({
            'success': True,
            'jobs': [job.to_dict() for job in jobs],
            'pagination': {
                'limit': limit,
                'offset': offset,
                'count': len(jobs)
            }
        }), 200
    except ValidationError as e:
        return create_error_response(str(e), 400, 'ValidationError')
    except Exception as e:
        logger.exception(f"Error listing jobs: {e}")
        return create_error_response('Internal server error', 500, 'DatabaseError')

@app.route('/api/jobs/<job_id>')
def get_job_api(job_id: str) -> Tuple[Dict[str, Any], int]:
    """API endpoint to get a specific job with validation."""
    try:
        # Validate job_id format (UUID)
        job_id = InputValidator.validate_string(
            job_id, min_length=1, max_length=100, 
            pattern=r'^[a-fA-F0-9-]{36}$'  # UUID pattern
        )
        
        job = get_job(job_id)
        if not job:
            return create_error_response(f"Job {job_id} not found", 404, 'JobNotFound')
        
        return jsonify({
            'success': True,
            'job': job.to_dict()
        }), 200
    except ValidationError as e:
        return create_error_response(f"Invalid job ID: {e}", 400, 'ValidationError')
    except Exception as e:
        logger.exception(f"Error getting job {job_id}: {e}")
        return create_error_response('Internal server error', 500, 'DatabaseError')

@app.route('/api/speakers')
def list_speakers_api():
    """API endpoint to list speakers."""
    try:
        speakers = list_speakers()
        return jsonify({
            'success': True,
            'speakers': [speaker.to_dict() for speaker in speakers]
        })
    except Exception as e:
        logger.exception(f"Error listing speakers: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/speakers/<speaker_id>/rename', methods=['POST'])
def rename_speaker_api(speaker_id: str) -> Tuple[Dict[str, Any], int]:
    """API endpoint to rename a speaker with comprehensive validation."""
    try:
        # Validate speaker_id
        speaker_id = InputValidator.validate_string(
            speaker_id, min_length=1, max_length=100
        )
        
        # Validate request content type
        if not request.is_json:
            return create_error_response(
                'Content-Type must be application/json', 400, 'InvalidContentType'
            )
        
        data = request.get_json()
        if not data:
            return create_error_response('Invalid JSON data', 400, 'InvalidJSON')
        
        # Validate required fields
        if 'new_name' not in data:
            return create_error_response(
                'Missing required field: new_name', 400, 'MissingField'
            )
        
        # Validate and sanitize new_name
        new_name = InputValidator.validate_string(
            data['new_name'], 
            min_length=1, 
            max_length=100,
            allowed_chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-'
        )
        new_name = InputValidator.sanitize_html(new_name)
        
        # Attempt to rename speaker
        if not rename_speaker(speaker_id, new_name):
            return create_error_response(
                f"Speaker {speaker_id} not found", 404, 'SpeakerNotFound'
            )
        
        return jsonify({
            'success': True,
            'message': f"Speaker renamed to {new_name}"
        }), 200
        
    except ValidationError as e:
        return create_error_response(str(e), 400, 'ValidationError')
    except Exception as e:
        logger.exception(f"Error renaming speaker {speaker_id}: {e}")
        return create_error_response('Internal server error', 500, 'DatabaseError')

@app.route('/api/backend-status')
def backend_status_api():
    """API endpoint to get the backend status."""
    try:
        transcription_status = {}
        for name, info in TRANSCRIPTION_BACKENDS.items():
            transcription_status[name] = {
                'available': info['available'],
                'real_available': info.get('real_available', False),
                'name': info['name']
            }
        
        diarization_status = {}
        for name, info in DIARIZATION_BACKENDS.items():
            diarization_status[name] = {
                'available': info['available'],
                'real_available': info.get('real_available', False),
                'name': info['name']
            }
        
        return jsonify({
            'success': True,
            'transcription_backends': transcription_status,
            'diarization_backends': diarization_status,
            'background_jobs_available': BACKGROUND_JOBS_AVAILABLE
        })
    except Exception as e:
        logger.exception(f"Error getting backend status: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/jobs/<job_id>/status')
def get_job_status_api(job_id: str):
    """API endpoint to get job status (for polling background jobs)."""
    try:
        # Validate job_id format
        job_id = InputValidator.validate_string(
            job_id, min_length=1, max_length=100,
            pattern=r'^[a-fA-F0-9-]{36}$'  # UUID pattern
        )
        
        # Get job from database
        job = get_job(job_id)
        if not job:
            return create_error_response(f"Job {job_id} not found", 404, 'JobNotFound')
        
        # Note: RQ job status requires the RQ job ID, not our job_id
        # For now, we rely on database state which is updated by the worker
        # Future enhancement: store RQ job ID in job metadata for direct status checks
        
        response = {
            'success': True,
            'job': job.to_dict(),
            'state': job.state.value if hasattr(job.state, 'value') else str(job.state)
        }
        
        if bg_status:
            response['background_status'] = bg_status
        
        return jsonify(response), 200
        
    except ValidationError as e:
        return create_error_response(str(e), 400, 'ValidationError')
    except Exception as e:
        logger.exception(f"Error getting job status: {e}")
        return create_error_response('Internal server error', 500, 'DatabaseError')

@app.route('/upload', methods=['POST'])
def upload_file() -> Tuple[Dict[str, Any], int]:
    """Endpoint to upload and process an audio file with comprehensive security."""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return create_error_response('No file provided', 400, 'NoFile')
        
        file = request.files['file']
        
        if not file or file.filename == '':
            return create_error_response('No file selected', 400, 'EmptyFile')
        
        # Comprehensive file validation
        try:
            safe_filename, extension = FileValidator.validate_upload(file)
        except (ValidationError, SecurityError) as e:
            return create_error_response(str(e), 400, 'FileValidationError')
        
        # Validate processing parameters
        try:
            model_size = InputValidator.validate_choice(
                request.form.get('model_size', Config.DEFAULT_MODEL_SIZE),
                ['tiny', 'base', 'small', 'medium', 'large']
            )
            transcription_backend = InputValidator.validate_choice(
                request.form.get('transcription_backend', Config.DEFAULT_TRANSCRIPTION_BACKEND),
                list(TRANSCRIPTION_BACKENDS.keys())
            )
            diarization_backend = InputValidator.validate_choice(
                request.form.get('diarization_backend', Config.DEFAULT_DIARIZATION_BACKEND),
                list(DIARIZATION_BACKENDS.keys())
            )
        except ValidationError as e:
            return create_error_response(f'Invalid parameter: {e}', 400, 'ParameterError')
        
        # Generate secure file path
        unique_filename = f"{uuid.uuid4()}_{safe_filename}"
        file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        
        # Ensure upload directory exists and is secure
        try:
            from pathlib import Path
            Path(Config.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f'Failed to create upload directory: {e}')
            return create_error_response('Server configuration error', 500, 'ServerError')
        
        # Save file securely
        try:
            file.save(file_path)
            logger.info(f"File uploaded: {unique_filename} ({file.content_length} bytes)")
        except OSError as e:
            logger.error(f'Failed to save uploaded file: {e}')
            return create_error_response('Failed to save file', 500, 'FileSystemError')
        
        # Additional file content validation after saving
        try:
            FileValidator.validate_file_content(file_path)
        except ValidationError as e:
            # Clean up invalid file
            try:
                os.unlink(file_path)
            except OSError:
                pass
            return create_error_response(f'Invalid file content: {e}', 400, 'ContentValidationError')
        
        # Create transcription job record
        job_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        from .models import TranscriptionJob, TranscriptionState
        
        job = TranscriptionJob(
            id=job_id,
            file_path=file_path,
            file_hash=hash_file(file_path),
            state=TranscriptionState.PENDING,
            timestamp=timestamp,
            model_size=model_size,
            transcription_backend=transcription_backend,
            diarization_backend=diarization_backend
        )
        save_job(job)
        
        # Check if background processing is available
        use_background = request.form.get('use_background', 'true').lower() in ('true', '1', 'yes')
        
        if BACKGROUND_JOBS_AVAILABLE and use_background:
            # Process in background
            try:
                from worker import process_transcription_job
                bg_job_id = enqueue_job(
                    process_transcription_job,
                    job_id,
                    file_path,
                    model_size,
                    transcription_backend,
                    diarization_backend
                )
                
                logger.info(f"Job {job_id} queued for background processing (RQ job: {bg_job_id})")
                
                return jsonify({
                    'success': True,
                    'message': 'File uploaded and queued for processing',
                    'job': job.to_dict(),
                    'background': True,
                    'job_id': job_id
                }), 202  # 202 Accepted for async processing
                
            except Exception as bg_error:
                logger.warning(f"Background processing failed, falling back to synchronous: {bg_error}")
                # Fall through to synchronous processing
        
        # Process synchronously (fallback or if background disabled)
        try:
            result = process_audio_file(
                file_path, 
                model_size=model_size,
                transcription_backend=transcription_backend,
                diarization_backend=diarization_backend
            )
            
            if result.success:
                return jsonify({
                    'success': True,
                    'message': result.message,
                    'job': result.job.to_dict() if result.job else None,
                    'background': False
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': result.message,
                    'error_type': 'ProcessingError'
                }), 500
                
        except Exception as processing_error:
            logger.exception(f'Audio processing failed: {processing_error}')
            # Clean up file on processing failure
            try:
                os.unlink(file_path)
            except OSError:
                pass
            return create_error_response(
                'Audio processing failed', 500, 'ProcessingError'
            )
            
    except RequestEntityTooLarge:
        return create_error_response(
            f'File too large. Maximum size: {Config.MAX_UPLOAD_SIZE_MB}MB', 
            413, 'FileTooLarge'
        )
    except Exception as e:
        logger.exception(f"Upload error: {e}")
        return create_error_response('Upload failed', 500, 'UploadError')

@app.route('/download/<job_id>/<format>')
def download_file(job_id: str, format: str):
    """Download a transcript in the specified format with security validation."""
    try:
        # Validate job_id format
        job_id = InputValidator.validate_string(
            job_id, min_length=1, max_length=100,
            pattern=r'^[a-fA-F0-9-]{36}$'  # UUID pattern
        )
        
        # Validate format parameter
        format = InputValidator.validate_choice(
            format, ['html', 'json', 'srt']
        )
        
        job = get_job(job_id)
        if not job:
            return create_error_response(f"Job {job_id} not found", 404, 'JobNotFound')
        
        if job.state != TranscriptionState.COMPLETED:
            return create_error_response(
                f"Job {job_id} is not completed yet (status: {job.state})", 
                400, 'JobNotReady'
            )
        
        # Generate and serve files with proper security checks
        try:
            if format == 'html':
                if job.output_path and os.path.exists(job.output_path):
                    # Validate path is within allowed directory
                    from pathlib import Path
                    file_path = Path(job.output_path).resolve()
                    allowed_dir = Path(Config.TRANSCRIPT_FOLDER).resolve()
                    
                    if not str(file_path).startswith(str(allowed_dir)):
                        raise SecurityError('Invalid file path')
                    
                    return send_file(file_path, mimetype='text/html')
                else:
                    return create_error_response('HTML output not available', 404, 'FileNotFound')
                    
            elif format == 'json':
                json_path = generate_json_output(job)
                return send_file(json_path, mimetype='application/json', 
                               as_attachment=True, download_name=f"transcript_{job_id}.json")
                               
            elif format == 'srt':
                srt_path = generate_srt_subtitle(job)
                return send_file(srt_path, mimetype='text/plain', 
                               as_attachment=True, download_name=f"transcript_{job_id}.srt")
        
        except SecurityError as e:
            logger.warning(f'Security error in download: {e}')
            return create_error_response('Access denied', 403, 'SecurityError')
        except Exception as generation_error:
            logger.error(f'Failed to generate {format} output: {generation_error}')
            return create_error_response(
                f'Failed to generate {format} output', 500, 'GenerationError'
            )
            
    except ValidationError as e:
        return create_error_response(str(e), 400, 'ValidationError')
    except Exception as e:
        logger.exception(f"Download error: {e}")
        return create_error_response('Download failed', 500, 'DownloadError')

@app.route('/')
def index():
    """Render the homepage."""
    # Get available backends
    transcription_backends = [
        {
            'id': name, 
            'name': info['name'], 
            'available': info['available'],
            'real_available': info.get('real_available', False),
            'status': 'real' if info.get('real_available', False) else 'mock' if info['available'] else 'unavailable'
        }
        for name, info in TRANSCRIPTION_BACKENDS.items()
    ]
    
    diarization_backends = [
        {
            'id': name, 
            'name': info['name'], 
            'available': info['available'],
            'real_available': info.get('real_available', False),
            'status': 'real' if info.get('real_available', False) else 'mock' if info['available'] else 'unavailable'
        }
        for name, info in DIARIZATION_BACKENDS.items()
    ]
    
    # Get data to pass to template
    data = {
        'transcription_backends': transcription_backends,
        'diarization_backends': diarization_backends,
        'default_model_size': Config.DEFAULT_MODEL_SIZE,
        'default_transcription_backend': Config.DEFAULT_TRANSCRIPTION_BACKEND,
        'default_diarization_backend': Config.DEFAULT_DIARIZATION_BACKEND,
        'max_upload_size': Config.MAX_UPLOAD_SIZE_MB
    }
    
    return render_template('index.html', **data)

# Register error handlers
register_error_handlers(app)

# Register meta-recursive API blueprint if available
if META_RECURSIVE_AVAILABLE and meta_recursive_bp:
    app.register_blueprint(meta_recursive_bp, url_prefix='/api')
    logger.info("Meta-recursive API blueprint registered")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)