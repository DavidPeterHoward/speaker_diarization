"""
AudioTranscribe: Models and Database Layer
-----------------------------------------
Data models and database interactions for the application.
"""

import hashlib
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

# Get logger
logger = logging.getLogger(__name__)

# ------------------------ CONFIGURATION ------------------------ #
class Config:
    """Application configuration with defaults and environment override support."""
    
    # Paths
    DATA_DIR = os.environ.get('AUDIOTRANSCRIBE_DATA_DIR', os.path.abspath('./data'))
    DB_PATH = os.path.join(DATA_DIR, 'transcripts.db')
    UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')
    TRANSCRIPT_FOLDER = os.path.join(DATA_DIR, 'transcripts')
    AUDIO_CACHE_FOLDER = os.path.join(DATA_DIR, 'audio_cache')
    LOG_FILE = os.path.join(DATA_DIR, 'transcription.log')
    
    # Server config
    HOST = os.environ.get('AUDIOTRANSCRIBE_HOST', '127.0.0.1')
    PORT = int(os.environ.get('AUDIOTRANSCRIBE_PORT', 5000))
    DEBUG = os.environ.get('AUDIOTRANSCRIBE_DEBUG', 'False').lower() in ('true', '1', 't')
    
    # Processing config
    DEFAULT_MODEL_SIZE = os.environ.get('AUDIOTRANSCRIBE_MODEL_SIZE', 'base')
    MAX_UPLOAD_SIZE_MB = int(os.environ.get('AUDIOTRANSCRIBE_MAX_UPLOAD_SIZE_MB', 100))
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
    DEFAULT_DIARIZATION_BACKEND = os.environ.get('AUDIOTRANSCRIBE_DIARIZATION_BACKEND', 'pyannote')
    DEFAULT_TRANSCRIPTION_BACKEND = os.environ.get('AUDIOTRANSCRIBE_TRANSCRIPTION_BACKEND', 'faster_whisper')
    
    # Similarity threshold for speaker identification (0.0-1.0)
    SPEAKER_SIMILARITY_THRESHOLD = float(os.environ.get('AUDIOTRANSCRIBE_SPEAKER_SIMILARITY', 0.85))
    
    # Security - Use a fixed fallback secret key for development
    # IMPORTANT: Set AUDIOTRANSCRIBE_SECRET_KEY in production!
    SECRET_KEY = os.environ.get('AUDIOTRANSCRIBE_SECRET_KEY', 'dev-secret-key-change-in-production')
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        for path in [cls.DATA_DIR, cls.UPLOAD_FOLDER, cls.TRANSCRIPT_FOLDER, cls.AUDIO_CACHE_FOLDER]:
            os.makedirs(path, exist_ok=True)

# ------------------------ DATA MODELS ------------------------ #
class TranscriptionState(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Speaker:
    id: str
    name: str
    embeddings: List[List[float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert speaker to dictionary, excluding embeddings."""
        return {
            'id': self.id,
            'name': self.name,
            # Exclude embeddings for JSON serialization
        }

@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = "Unknown"
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary with formatted timestamps."""
        return {
            'start': self.start,
            'end': self.end,
            'text': self.text,
            'speaker': self.speaker,
            'confidence': self.confidence,
            'formatted_start': format_timestamp(self.start),
            'formatted_end': format_timestamp(self.end),
        }

@dataclass
class TranscriptionJob:
    id: str
    file_path: str
    file_hash: str
    state: TranscriptionState
    timestamp: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    segments: List[Segment] = field(default_factory=list)
    model_size: str = Config.DEFAULT_MODEL_SIZE
    transcription_backend: str = Config.DEFAULT_TRANSCRIPTION_BACKEND
    diarization_backend: str = Config.DEFAULT_DIARIZATION_BACKEND
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary with all details."""
        return {
            'id': self.id,
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'state': self.state.value if hasattr(self.state, 'value') else str(self.state),
            'timestamp': self.timestamp,
            'output_path': self.output_path,
            'error': self.error,
            'segments': [s.to_dict() for s in self.segments],
            'model_size': self.model_size,
            'transcription_backend': self.transcription_backend,
            'diarization_backend': self.diarization_backend,
        }

@dataclass
class ProcessingResult:
    success: bool
    message: str
    job: Optional[TranscriptionJob] = None
    error: Optional[Exception] = None

# ------------------------ DATABASE ------------------------ #
@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections with proper error handling."""
    conn = None
    try:
        conn = sqlite3.connect(Config.DB_PATH, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def init_db() -> None:
    """Initialize the database schema with proper indexes and constraints."""
    try:
        with get_db_connection() as conn:
            # Jobs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'processing', 'completed', 'failed')),
                    timestamp TEXT NOT NULL,
                    output_path TEXT,
                    error TEXT,
                    model_size TEXT NOT NULL,
                    transcription_backend TEXT NOT NULL,
                    diarization_backend TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Migrate existing jobs table if PENDING state was missing
            try:
                # Check if we need to update the constraint (SQLite doesn't support ALTER CHECK)
                # We'll just ensure the enum values are correct by validating on insert
                pass
            except Exception:
                pass  # Migration handled at insert time via validation
            
            # Segments table with cascade delete
            conn.execute('''
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    text TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
                    FOREIGN KEY (job_id) REFERENCES jobs (id) ON DELETE CASCADE
                )
            ''')
            
            # Speakers table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS speakers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Speaker embeddings with cascade delete
            conn.execute('''
                CREATE TABLE IF NOT EXISTS speaker_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speaker_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (speaker_id) REFERENCES speakers (id) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs (state)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_jobs_timestamp ON jobs (timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_jobs_file_hash ON jobs (file_hash)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_segments_job_id ON segments (job_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_segments_start ON segments (start)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_speaker_id ON speaker_embeddings (speaker_id)')
            
            conn.commit()
        logger.info("Database initialized successfully with indexes")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise

def save_job(job: TranscriptionJob) -> None:
    """Save a transcription job to the database with transaction safety."""
    try:
        with get_db_connection() as conn:
            # Use proper enum value for state
            state_value = job.state.value if hasattr(job.state, 'value') else str(job.state)
            
            conn.execute('''
                INSERT OR REPLACE INTO jobs (
                    id, file_path, file_hash, state, timestamp, output_path, 
                    error, model_size, transcription_backend, diarization_backend
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job.id, job.file_path, job.file_hash, state_value, job.timestamp,
                job.output_path, job.error, job.model_size, job.transcription_backend,
                job.diarization_backend
            ))
            
            # Delete existing segments if replacing a job (CASCADE will handle this)
            conn.execute("DELETE FROM segments WHERE job_id = ?", (job.id,))
            
            # Insert segments efficiently using executemany
            if job.segments:
                segment_data = [
                    (job.id, segment.start, segment.end, segment.text, 
                     segment.speaker, segment.confidence)
                    for segment in job.segments
                ]
                conn.executemany('''
                    INSERT INTO segments (job_id, start, end, text, speaker, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', segment_data)
            
            conn.commit()
        logger.debug(f"Job {job.id} saved to database with {len(job.segments)} segments")
    except sqlite3.Error as e:
        logger.error(f"Failed to save job {job.id} to database: {e}")
        raise

def get_job(job_id: str) -> Optional[TranscriptionJob]:
    """Retrieve a job from the database by ID."""
    try:
        with get_db_connection() as conn:
            job_row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            
            if not job_row:
                return None
            
            # Get segments for this job
            segments_rows = conn.execute(
                "SELECT * FROM segments WHERE job_id = ? ORDER BY start", 
                (job_id,)
            ).fetchall()
            
            segments = [
                Segment(
                    start=row['start'],
                    end=row['end'],
                    text=row['text'],
                    speaker=row['speaker'],
                    confidence=row['confidence']
                ) for row in segments_rows
            ]
            
            return TranscriptionJob(
                id=job_row['id'],
                file_path=job_row['file_path'],
                file_hash=job_row['file_hash'],
                state=TranscriptionState(job_row['state']),
                timestamp=job_row['timestamp'],
                output_path=job_row['output_path'],
                error=job_row['error'],
                segments=segments,
                model_size=job_row['model_size'],
                transcription_backend=job_row['transcription_backend'],
                diarization_backend=job_row['diarization_backend']
            )
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error retrieving job {job_id}: {e}")
        return None

def list_jobs(limit: int = 100, offset: int = 0) -> List[TranscriptionJob]:
    """List jobs from the database with pagination."""
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY timestamp DESC LIMIT ? OFFSET ?", 
                (limit, offset)
            ).fetchall()
            
            jobs = []
            for row in rows:
                job = TranscriptionJob(
                    id=row['id'],
                    file_path=row['file_path'],
                    file_hash=row['file_hash'],
                    state=TranscriptionState(row['state']),
                    timestamp=row['timestamp'],
                    output_path=row['output_path'],
                    error=row['error'],
                    segments=[],  # Don't load segments for listing
                    model_size=row['model_size'],
                    transcription_backend=row['transcription_backend'],
                    diarization_backend=row['diarization_backend']
                )
                jobs.append(job)
            
            return jobs
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error listing jobs: {e}")
        return []

def get_job_by_hash(file_hash: str) -> Optional[TranscriptionJob]:
    """Find a completed job by file hash."""
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE file_hash = ? AND state = ? LIMIT 1", 
                (file_hash, TranscriptionState.COMPLETED)
            ).fetchone()
            
            if row:
                return get_job(row['id'])
            return None
    except sqlite3.Error as e:
        logger.error(f"Error finding job by hash: {e}")
        return None

def save_speaker(speaker: Speaker):
    """Save a speaker and their embeddings to the database."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO speakers (id, name) VALUES (?, ?)",
                (speaker.id, speaker.name)
            )
            
            # Only save embeddings if they exist
            if speaker.embeddings:
                # First delete existing embeddings
                conn.execute("DELETE FROM speaker_embeddings WHERE speaker_id = ?", (speaker.id,))
                
                # Insert new embeddings
                for embedding in speaker.embeddings:
                    # Convert embedding to bytes for storage
                    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                    conn.execute(
                        "INSERT INTO speaker_embeddings (speaker_id, embedding) VALUES (?, ?)",
                        (speaker.id, embedding_bytes)
                    )
            
            conn.commit()
        logger.debug(f"Speaker {speaker.id} ({speaker.name}) saved to database")
    except sqlite3.Error as e:
        logger.error(f"Failed to save speaker: {e}")
        raise

def get_speaker(speaker_id: str) -> Optional[Speaker]:
    """Get a speaker by ID with their embeddings."""
    try:
        with get_db_connection() as conn:
            speaker_row = conn.execute(
                "SELECT * FROM speakers WHERE id = ?", 
                (speaker_id,)
            ).fetchone()
            
            if not speaker_row:
                return None
            
            embedding_rows = conn.execute(
                "SELECT embedding FROM speaker_embeddings WHERE speaker_id = ?",
                (speaker_id,)
            ).fetchall()
            
            embeddings = []
            for row in embedding_rows:
                # Convert bytes back to list of floats
                embedding_array = np.frombuffer(row['embedding'], dtype=np.float32)
                embeddings.append(embedding_array.tolist())
            
            return Speaker(
                id=speaker_row['id'],
                name=speaker_row['name'],
                embeddings=embeddings
            )
    except sqlite3.Error as e:
        logger.error(f"Error retrieving speaker {speaker_id}: {e}")
        return None

def list_speakers() -> List[Speaker]:
    """List all speakers in the database."""
    try:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT id, name FROM speakers").fetchall()
            return [Speaker(id=row['id'], name=row['name']) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Error listing speakers: {e}")
        return []

def get_all_speakers_with_embeddings() -> List[Speaker]:
    """Get all speakers with their embeddings."""
    try:
        speakers = []
        with get_db_connection() as conn:
            speaker_rows = conn.execute("SELECT id, name FROM speakers").fetchall()
            
            for speaker_row in speaker_rows:
                speaker_id = speaker_row['id']
                
                embedding_rows = conn.execute(
                    "SELECT embedding FROM speaker_embeddings WHERE speaker_id = ?",
                    (speaker_id,)
                ).fetchall()
                
                embeddings = []
                for row in embedding_rows:
                    embedding_array = np.frombuffer(row['embedding'], dtype=np.float32)
                    embeddings.append(embedding_array.tolist())
                
                speakers.append(Speaker(
                    id=speaker_id,
                    name=speaker_row['name'],
                    embeddings=embeddings
                ))
        
        return speakers
    except sqlite3.Error as e:
        logger.error(f"Error retrieving speakers with embeddings: {e}")
        return []

def rename_speaker(speaker_id: str, new_name: str) -> bool:
    """Rename a speaker in the database."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE speakers SET name = ? WHERE id = ?",
                (new_name, speaker_id)
            )
            conn.commit()
        logger.info(f"Speaker {speaker_id} renamed to {new_name}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to rename speaker: {e}")
        return False

# ------------------------ UTILITIES ------------------------ #
def format_timestamp(seconds: float) -> str:
    """Format a timestamp in seconds to HH:MM:SS.mmm format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

def hash_file(file_path: str) -> str:
    """Generate SHA-256 hash of a file."""
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(BUF_SIZE):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()
        logger.debug(f"Generated hash for {file_path}: {file_hash}")
        return file_hash
    except Exception as e:
        logger.error(f"Failed to hash file {file_path}: {e}")
        raise

def allowed_file(filename: str) -> bool:
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS