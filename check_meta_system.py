#!/usr/bin/env python3
import time
import requests

print('Checking AudioTranscribe with Meta-Recursive System...')
time.sleep(5)

# Check health
try:
    response = requests.get('http://localhost:5000/health', timeout=5)
    print(f'✓ Health check: {response.status_code}')
except Exception as e:
    print(f'✗ Health check failed: {e}')
    exit(1)

# Check meta-recursive endpoints
meta_endpoints = [
    ('status', 'http://localhost:5000/api/meta-review/status'),
    ('diagnostics', 'http://localhost:5000/api/meta-review/diagnostics'),
    ('recommendations', 'http://localhost:5000/api/meta-review/recommendations')
]

for name, url in meta_endpoints:
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f'✓ Meta API {name}: {response.status_code}')
        else:
            print(f'✗ Meta API {name}: {response.status_code} - {response.text[:50]}...')
    except Exception as e:
        print(f'✗ Meta API {name} failed: {e}')

print('Meta-recursive system check complete!')
