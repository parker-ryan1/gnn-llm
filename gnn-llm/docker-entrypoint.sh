#!/bin/bash
set -e

echo "Starting Graph-GNN-RAG Pipeline..."

# Wait for database to be ready
echo "Waiting for database connection..."
python -c "
import time
import sys
from sqlalchemy import create_engine
from config import Config

config = Config()
max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        engine = create_engine(config.DATABASE_URL)
        connection = engine.connect()
        connection.close()
        print('Database connection successful!')
        break
    except Exception as e:
        retry_count += 1
        print(f'Database connection attempt {retry_count}/{max_retries} failed: {e}')
        time.sleep(2)

if retry_count >= max_retries:
    print('Failed to connect to database after maximum retries')
    sys.exit(1)
"

# Wait for Ollama to be ready
echo "Waiting for Ollama service..."
python -c "
import time
import requests
import sys
from config import Config

config = Config()
max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        response = requests.get(f'{config.OLLAMA_BASE_URL}/api/tags', timeout=5)
        if response.status_code == 200:
            print('Ollama service is ready!')
            break
    except Exception as e:
        retry_count += 1
        print(f'Ollama connection attempt {retry_count}/{max_retries} failed: {e}')
        time.sleep(2)

if retry_count >= max_retries:
    print('Failed to connect to Ollama after maximum retries')
    sys.exit(1)
"

# Check if Mistral model is available
echo "Checking Mistral model availability..."
python -c "
import requests
import json
from config import Config

config = Config()
try:
    response = requests.get(f'{config.OLLAMA_BASE_URL}/api/tags')
    models = response.json()
    model_names = [model['name'] for model in models.get('models', [])]
    
    if any('mistral' in name for name in model_names):
        print('Mistral model is available!')
    else:
        print('Warning: Mistral model not found. Available models:', model_names)
except Exception as e:
    print(f'Could not check available models: {e}')
"

echo "All services are ready. Starting main application..."

# Execute the main command
exec "$@"