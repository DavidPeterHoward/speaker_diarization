#!/usr/bin/env python3
"""Check backend status"""

import os
os.environ['AUDIOTRANSCRIBE_FORCE_MOCK'] = '1'

from app.transcription import TRANSCRIPTION_BACKENDS, DIARIZATION_BACKENDS

print('Transcription backends:')
for name, backend in TRANSCRIPTION_BACKENDS.items():
    print(f'  {name}: available={backend["available"]}, real_available={backend.get("real_available", "unknown")}')

print('\nDiarization backends:')
for name, backend in DIARIZATION_BACKENDS.items():
    print(f'  {name}: available={backend["available"]}, real_available={backend.get("real_available", "unknown")}')

print(f'\nAUDIOTRANSCRIBE_FORCE_MOCK = {os.environ.get("AUDIOTRANSCRIBE_FORCE_MOCK")}')
