#!/usr/bin/env python3
import requests

# Check various endpoints
endpoints = ['/', '/voices', '/tts', '/docs', '/openapi.json', '/api/voices', '/api/tts']

for endpoint in endpoints:
    try:
        response = requests.get(f'http://localhost:8012{endpoint}', timeout=5)
        print(f'{endpoint}: {response.status_code} - {response.headers.get("content-type", "unknown")}')
        if response.status_code == 200 and len(response.text) < 500:
            print(f'  Content: {response.text[:200]}...')
    except Exception as e:
        print(f'{endpoint}: Error - {e}')
