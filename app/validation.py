"""
AudioTranscribe: Input Validation and Security
----------------------------------------------
Comprehensive input validation and security utilities.
"""

import os
import re
import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from werkzeug.utils import secure_filename as werkzeug_secure_filename
from werkzeug.datastructures import FileStorage
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class FileValidator:
    """Validates file uploads and paths for security."""
    
    # Default allowed extensions
    ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'wma', 'opus'}
    
    # Maximum file sizes (in MB)
    MAX_FILE_SIZE_MB = {
        'default': 100,
        'wav': 500,  # WAV files can be larger
        'mp3': 100,
        'flac': 200
    }
    
    # Dangerous patterns in filenames
    DANGEROUS_PATTERNS = [
        r'\.\.',  # Directory traversal
        r'[<>:"|?*]',  # Windows forbidden characters
        r'[\x00-\x1f]',  # Control characters
        r'^\.+$',  # Hidden files
        r'^CON$|^PRN$|^AUX$|^NUL$',  # Windows reserved names
        r'^COM[1-9]$|^LPT[1-9]$'  # Windows device names
    ]
    
    @classmethod
    def validate_filename(cls, filename: str) -> str:
        """
        Validate and sanitize a filename.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
            
        Raises:
            ValidationError: If filename is invalid
        """
        if not filename:
            raise ValidationError("Filename cannot be empty")
        
        # Check length
        if len(filename) > 255:
            raise ValidationError("Filename too long (max 255 characters)")
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                raise SecurityError(f"Dangerous pattern detected in filename: {pattern}")
        
        # Use werkzeug's secure_filename
        safe_filename = werkzeug_secure_filename(filename)
        
        if not safe_filename:
            raise ValidationError("Invalid filename after sanitization")
        
        return safe_filename
    
    @classmethod
    def validate_file_extension(cls, filename: str, allowed_extensions: Optional[set] = None) -> str:
        """
        Validate file extension.
        
        Args:
            filename: Filename to check
            allowed_extensions: Set of allowed extensions (uses default if None)
            
        Returns:
            The file extension (lowercase)
            
        Raises:
            ValidationError: If extension is not allowed
        """
        if '.' not in filename:
            raise ValidationError("File has no extension")
        
        extension = filename.rsplit('.', 1)[1].lower()
        allowed = allowed_extensions or cls.ALLOWED_AUDIO_EXTENSIONS
        
        if extension not in allowed:
            raise ValidationError(
                f"File type '.{extension}' not allowed. "
                f"Allowed types: {', '.join(sorted(allowed))}"
            )
        
        return extension
    
    @classmethod
    def validate_file_size(cls, file_size: int, extension: str = 'default') -> None:
        """
        Validate file size.
        
        Args:
            file_size: Size in bytes
            extension: File extension for specific limits
            
        Raises:
            ValidationError: If file is too large
        """
        max_size_mb = cls.MAX_FILE_SIZE_MB.get(extension, cls.MAX_FILE_SIZE_MB['default'])
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            raise ValidationError(
                f"File too large: {file_size / (1024*1024):.1f}MB "
                f"(max {max_size_mb}MB for .{extension} files)"
            )
    
    @classmethod
    def validate_file_content(cls, file_path: str) -> bool:
        """
        Validate file content matches its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If content doesn't match extension
        """
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if not mime_type:
            logger.warning(f"Could not determine MIME type for {file_path}")
            return True  # Allow if we can't determine
        
        # Check if it's an audio file
        if not mime_type.startswith('audio/'):
            raise ValidationError(f"File content is not audio: {mime_type}")
        
        return True
    
    @classmethod
    def validate_upload(cls, file: FileStorage, allowed_extensions: Optional[set] = None) -> Tuple[str, str]:
        """
        Comprehensive validation of uploaded file.
        
        Args:
            file: Werkzeug FileStorage object
            allowed_extensions: Set of allowed extensions
            
        Returns:
            Tuple of (safe_filename, extension)
            
        Raises:
            ValidationError: If file is invalid
            SecurityError: If security issue detected
        """
        # Check if file exists
        if not file or file.filename == '':
            raise ValidationError("No file provided")
        
        # Validate filename
        safe_filename = cls.validate_filename(file.filename)
        
        # Validate extension
        extension = cls.validate_file_extension(safe_filename, allowed_extensions)
        
        # Check file size (approximate from stream)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        cls.validate_file_size(file_size, extension)
        
        return safe_filename, extension


class PathValidator:
    """Validates file system paths for security."""
    
    @staticmethod
    def validate_path(path: str, base_dir: Optional[str] = None) -> Path:
        """
        Validate a file system path.
        
        Args:
            path: Path to validate
            base_dir: Base directory to restrict access to
            
        Returns:
            Validated Path object
            
        Raises:
            SecurityError: If path is outside base_dir or contains traversal
        """
        # Convert to Path object
        path_obj = Path(path).resolve()
        
        # Check for directory traversal
        if '..' in path_obj.parts:
            raise SecurityError("Directory traversal detected")
        
        # If base_dir provided, ensure path is within it
        if base_dir:
            base_path = Path(base_dir).resolve()
            try:
                path_obj.relative_to(base_path)
            except ValueError:
                raise SecurityError(f"Path '{path}' is outside allowed directory")
        
        return path_obj
    
    @staticmethod
    def ensure_safe_directory(directory: str) -> Path:
        """
        Ensure directory exists and is safe to use.
        
        Args:
            directory: Directory path
            
        Returns:
            Path object for the directory
            
        Raises:
            SecurityError: If directory is unsafe
        """
        dir_path = Path(directory).resolve()
        
        # Create if doesn't exist
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Check if writable
        if not os.access(dir_path, os.W_OK):
            raise SecurityError(f"Directory '{directory}' is not writable")
        
        return dir_path


class InputValidator:
    """Validates general input parameters."""
    
    @staticmethod
    def validate_string(
        value: str,
        min_length: int = 0,
        max_length: int = 1000,
        pattern: Optional[str] = None,
        allowed_chars: Optional[str] = None
    ) -> str:
        """
        Validate a string input.
        
        Args:
            value: String to validate
            min_length: Minimum length
            max_length: Maximum length
            pattern: Regex pattern to match
            allowed_chars: String of allowed characters
            
        Returns:
            Validated string
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value).__name__}")
        
        # Check length
        if len(value) < min_length:
            raise ValidationError(f"String too short (min {min_length} characters)")
        if len(value) > max_length:
            raise ValidationError(f"String too long (max {max_length} characters)")
        
        # Check pattern
        if pattern and not re.match(pattern, value):
            raise ValidationError(f"String doesn't match required pattern")
        
        # Check allowed characters
        if allowed_chars:
            for char in value:
                if char not in allowed_chars:
                    raise ValidationError(f"Character '{char}' not allowed")
        
        return value
    
    @staticmethod
    def validate_choice(value: Any, choices: List[Any]) -> Any:
        """
        Validate value is in allowed choices.
        
        Args:
            value: Value to check
            choices: List of allowed values
            
        Returns:
            The value if valid
            
        Raises:
            ValidationError: If value not in choices
        """
        if value not in choices:
            raise ValidationError(
                f"Invalid choice '{value}'. "
                f"Allowed: {', '.join(map(str, choices))}"
            )
        return value
    
    @staticmethod
    def validate_number(
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        allow_float: bool = True
    ) -> Union[int, float]:
        """
        Validate a numeric value.
        
        Args:
            value: Number to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_float: Whether to allow float values
            
        Returns:
            Validated number
            
        Raises:
            ValidationError: If validation fails
        """
        # Type check
        if not allow_float and not isinstance(value, int):
            raise ValidationError(f"Expected integer, got {type(value).__name__}")
        
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Expected number, got {type(value).__name__}")
        
        # Range check
        if min_value is not None and value < min_value:
            raise ValidationError(f"Value {value} below minimum {min_value}")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"Value {value} above maximum {max_value}")
        
        return value
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """
        Remove potentially dangerous HTML/JavaScript from text.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove script tags and content
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove on* event handlers
        text = re.sub(r'\bon\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
        
        # Remove javascript: protocol
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # Remove other dangerous tags
        dangerous_tags = ['iframe', 'object', 'embed', 'applet', 'meta', 'link']
        for tag in dangerous_tags:
            text = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(f'<{tag}[^>]*/?>', '', text, flags=re.IGNORECASE)
        
        return text


class HashValidator:
    """Validates and generates secure hashes."""
    
    @staticmethod
    def generate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
        """
        Generate hash of a file.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm to use
            
        Returns:
            Hex digest of the hash
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    @staticmethod
    def validate_hash(hash_value: str, expected_length: int = 64) -> str:
        """
        Validate a hash string.
        
        Args:
            hash_value: Hash to validate
            expected_length: Expected length (64 for SHA256)
            
        Returns:
            Validated hash
            
        Raises:
            ValidationError: If hash is invalid
        """
        # Check if it's a hex string
        if not re.match(r'^[a-fA-F0-9]+$', hash_value):
            raise ValidationError("Invalid hash format (not hexadecimal)")
        
        # Check length
        if len(hash_value) != expected_length:
            raise ValidationError(f"Invalid hash length (expected {expected_length})")
        
        return hash_value.lower()


def validate_request_data(data: Dict[str, Any], schema: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Validate request data against a schema.
    
    Args:
        data: Data to validate
        schema: Validation schema
        
    Returns:
        Validated data
        
    Raises:
        ValidationError: If validation fails
    
    Example schema:
        {
            'name': {'type': 'string', 'required': True, 'max_length': 100},
            'age': {'type': 'number', 'required': False, 'min_value': 0, 'max_value': 150},
            'email': {'type': 'string', 'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$'}
        }
    """
    validated = {}
    
    for field, rules in schema.items():
        # Check if required
        if rules.get('required', False) and field not in data:
            raise ValidationError(f"Required field '{field}' is missing")
        
        if field not in data:
            continue
        
        value = data[field]
        field_type = rules.get('type', 'string')
        
        # Validate based on type
        if field_type == 'string':
            value = InputValidator.validate_string(
                value,
                min_length=rules.get('min_length', 0),
                max_length=rules.get('max_length', 1000),
                pattern=rules.get('pattern')
            )
        elif field_type == 'number':
            value = InputValidator.validate_number(
                value,
                min_value=rules.get('min_value'),
                max_value=rules.get('max_value'),
                allow_float=rules.get('allow_float', True)
            )
        elif field_type == 'choice':
            value = InputValidator.validate_choice(value, rules.get('choices', []))
        
        validated[field] = value
    
    return validated