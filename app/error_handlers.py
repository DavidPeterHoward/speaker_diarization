"""
AudioTranscribe: Error Handlers
-------------------------------
Centralized error handling for the Flask application.
"""

import logging
import traceback
from flask import jsonify, render_template_string, request
from werkzeug.exceptions import HTTPException
from typing import Tuple, Union, Dict, Any

logger = logging.getLogger(__name__)


class AudioTranscribeError(Exception):
    """Base exception for AudioTranscribe application."""
    status_code = 500
    message = "An error occurred"
    
    def __init__(self, message: str = None, status_code: int = None, payload: Dict = None):
        super().__init__()
        if message is not None:
            self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        rv = dict(self.payload)
        rv['success'] = False
        rv['message'] = self.message
        rv['error_type'] = self.__class__.__name__
        return rv


class ProcessingError(AudioTranscribeError):
    """Error during audio processing."""
    status_code = 500
    message = "Audio processing failed"


class TranscriptionError(ProcessingError):
    """Error during transcription."""
    message = "Transcription failed"


class DiarizationError(ProcessingError):
    """Error during speaker diarization."""
    message = "Speaker diarization failed"


class ValidationError(AudioTranscribeError):
    """Input validation error."""
    status_code = 400
    message = "Invalid input"


class FileError(AudioTranscribeError):
    """File-related error."""
    status_code = 400
    message = "File operation failed"


class DatabaseError(AudioTranscribeError):
    """Database operation error."""
    status_code = 500
    message = "Database operation failed"


class BackendError(AudioTranscribeError):
    """Backend service error."""
    status_code = 503
    message = "Backend service unavailable"


class AuthenticationError(AudioTranscribeError):
    """Authentication error."""
    status_code = 401
    message = "Authentication required"


class AuthorizationError(AudioTranscribeError):
    """Authorization error."""
    status_code = 403
    message = "Access denied"


class RateLimitError(AudioTranscribeError):
    """Rate limit exceeded."""
    status_code = 429
    message = "Rate limit exceeded"


def handle_audio_transcribe_error(error: AudioTranscribeError) -> Tuple[Union[str, Dict], int]:
    """
    Handle custom AudioTranscribe errors.
    
    Args:
        error: The AudioTranscribeError instance
        
    Returns:
        Response tuple (content, status_code)
    """
    # Log the error
    logger.error(
        f"{error.__class__.__name__}: {error.message}",
        extra={
            'error_type': error.__class__.__name__,
            'status_code': error.status_code,
            'payload': error.payload
        }
    )
    
    # Return JSON for API requests
    if request.path.startswith('/api/') or request.headers.get('Accept') == 'application/json':
        return jsonify(error.to_dict()), error.status_code
    
    # Return HTML for browser requests
    return render_error_page(error.status_code, error.message), error.status_code


def handle_http_exception(error: HTTPException) -> Tuple[Union[str, Dict], int]:
    """
    Handle Werkzeug HTTP exceptions.
    
    Args:
        error: The HTTPException instance
        
    Returns:
        Response tuple (content, status_code)
    """
    # Log the error
    logger.warning(
        f"HTTP {error.code}: {error.description}",
        extra={
            'status_code': error.code,
            'path': request.path,
            'method': request.method
        }
    )
    
    # Return JSON for API requests
    if request.path.startswith('/api/') or request.headers.get('Accept') == 'application/json':
        return jsonify({
            'success': False,
            'message': error.description,
            'status_code': error.code
        }), error.code
    
    # Return HTML for browser requests
    return render_error_page(error.code, error.description), error.code


def handle_generic_exception(error: Exception) -> Tuple[Union[str, Dict], int]:
    """
    Handle generic Python exceptions.
    
    Args:
        error: The Exception instance
        
    Returns:
        Response tuple (content, status_code)
    """
    # Log the full traceback
    logger.exception(
        f"Unhandled exception: {str(error)}",
        extra={
            'exception_type': type(error).__name__,
            'traceback': traceback.format_exc(),
            'path': request.path,
            'method': request.method
        }
    )
    
    # Don't expose internal errors in production
    message = "An unexpected error occurred. Please try again later."
    
    # Return JSON for API requests
    if request.path.startswith('/api/') or request.headers.get('Accept') == 'application/json':
        return jsonify({
            'success': False,
            'message': message,
            'error_type': 'InternalServerError'
        }), 500
    
    # Return HTML for browser requests
    return render_error_page(500, message), 500


def render_error_page(status_code: int, message: str) -> str:
    """
    Render an error page HTML.
    
    Args:
        status_code: HTTP status code
        message: Error message
        
    Returns:
        HTML string
    """
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Error {{ status_code }}</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
                display: flex;
                min-height: 100vh;
                align-items: center;
                justify-content: center;
            }
            .error-container {
                text-align: center;
                padding: 40px;
                background: #f9f9f9;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #d32f2f;
                margin-bottom: 10px;
                font-size: 48px;
            }
            h2 {
                color: #666;
                margin-bottom: 20px;
                font-weight: normal;
            }
            p {
                margin-bottom: 30px;
                color: #777;
            }
            a {
                display: inline-block;
                padding: 10px 20px;
                background: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background 0.3s;
            }
            a:hover {
                background: #45a049;
            }
            .error-code {
                font-size: 72px;
                font-weight: bold;
                color: #e0e0e0;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="error-container">
            <div class="error-code">{{ status_code }}</div>
            <h2>{{ error_title }}</h2>
            <p>{{ message }}</p>
            <a href="/">Go to Homepage</a>
        </div>
    </body>
    </html>
    """
    
    error_titles = {
        400: "Bad Request",
        401: "Authentication Required",
        403: "Access Denied",
        404: "Page Not Found",
        405: "Method Not Allowed",
        429: "Too Many Requests",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable"
    }
    
    return render_template_string(
        template,
        status_code=status_code,
        error_title=error_titles.get(status_code, "Error"),
        message=message
    )


def register_error_handlers(app):
    """
    Register error handlers with the Flask app.
    
    Args:
        app: Flask application instance
    """
    # Handle custom AudioTranscribe errors
    @app.errorhandler(AudioTranscribeError)
    def handle_custom_error(error):
        return handle_audio_transcribe_error(error)
    
    # Handle specific HTTP errors
    @app.errorhandler(400)
    def handle_bad_request(error):
        return handle_http_exception(error)
    
    @app.errorhandler(401)
    def handle_unauthorized(error):
        return handle_http_exception(error)
    
    @app.errorhandler(403)
    def handle_forbidden(error):
        return handle_http_exception(error)
    
    @app.errorhandler(404)
    def handle_not_found(error):
        return handle_http_exception(error)
    
    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        return handle_http_exception(error)
    
    @app.errorhandler(429)
    def handle_rate_limit(error):
        return handle_http_exception(error)
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        return handle_http_exception(error)
    
    @app.errorhandler(503)
    def handle_service_unavailable(error):
        return handle_http_exception(error)
    
    # Handle generic exceptions
    @app.errorhandler(Exception)
    def handle_exception(error):
        # Pass through HTTP exceptions
        if isinstance(error, HTTPException):
            return handle_http_exception(error)
        return handle_generic_exception(error)
    
    logger.info("Error handlers registered")


def create_error_response(
    message: str,
    status_code: int = 500,
    error_type: str = None,
    details: Dict = None
) -> Tuple[Dict, int]:
    """
    Create a standardized error response.
    
    Args:
        message: Error message
        status_code: HTTP status code
        error_type: Type of error
        details: Additional error details
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        'success': False,
        'message': message,
        'status_code': status_code
    }
    
    if error_type:
        response['error_type'] = error_type
    
    if details:
        response['details'] = details
    
    return response, status_code