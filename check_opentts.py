#!/usr/bin/env python3
import requests

try:
    response = requests.get('http://localhost:8012/voices', timeout=5)
    if response.status_code == 200:
        print('OpenTTS is running!')
        try:
            voices = response.json()
            print(f'Response type: {type(voices)}')

            if isinstance(voices, list):
                print(f'Available voices: {len(voices)}')
                sample_voices = voices[:3]
                print(f'Sample voices: {sample_voices}')
            elif isinstance(voices, dict):
                voices_list = voices.get('voices', voices)
                if isinstance(voices_list, list):
                    print(f'Available voices: {len(voices_list)}')
                    sample_voices = voices_list[:3]
                    print(f'Sample voices: {sample_voices}')
                else:
                    print(f'Voices data: {voices_list}')
            else:
                print(f'Unexpected response format: {voices}')

        except Exception as json_error:
            print(f'Could not parse JSON response: {json_error}')
            print(f'Raw response: {response.text[:200]}...')
    else:
        print(f'OpenTTS responded with status: {response.status_code}')
        print(f'Response: {response.text[:200]}...')
except Exception as e:
    print(f'OpenTTS not accessible: {e}')
