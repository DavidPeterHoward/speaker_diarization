#!/usr/bin/env python3
import time
import requests
import sys

print('Checking if AudioTranscribe server is running...')
time.sleep(5)

try:
    response = requests.get('http://localhost:5000/health', timeout=10)
    if response.status_code in (200, 404):
        print('✓ Server is running!')
        print(f'  Health check: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            print(f'  Service: {data.get("service", "unknown")}')
            print(f'  Status: {data.get("status", "unknown")}')
        sys.exit(0)
    else:
        print(f'✗ Server returned unexpected status: {response.status_code}')
        sys.exit(1)
except Exception as e:
    print(f'✗ Server not accessible: {e}')
    sys.exit(1)
