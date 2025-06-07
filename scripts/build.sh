#!/bin/bash

# Create necessary directories
mkdir -p /opt/llm/inference-service/{app,scripts,logs,config,docker}

# Create config file if it doesn't exist
if [ ! -f /opt/llm/inference-service/config/service_config.yaml ]; then
  cat > /opt/llm/inference-service/config/service_config.yaml << YAML
models_dir: /opt/llm/models/ollama/models/blobs
manifests_dir: /opt/llm/models/ollama/models/manifests/registry.ollama.ai/library
llama_cpp_path: /opt/llm/models/ollama-custom-models/llama.cpp
default_context_size: 4096
default_model: null
tensor_split: '0.25,0.25,0.25,0.25'
gpu_layers: 999
threads: 8
batch_size: 512
YAML
fi

# Create Dockerfile
cat > /opt/llm/inference-service/docker/Dockerfile << DOCKERFILE
FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=\$CONDA_DIR/bin:\$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cmake \\
    git \\
    wget \\
    curl \\
    python3-dev \\
    python3-pip \\
    nvidia-cuda-toolkit \\
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \\
    && bash miniconda.sh -b -p \$CONDA_DIR \\
    && rm miniconda.sh

# Create working directories
RUN mkdir -p /opt/llm/inference-service/{app,logs,config} \\
    && mkdir -p /opt/llm/models/ollama-custom-models/llama.cpp

# Set up Conda environment
RUN conda create -n llm-service python=3.10 -y \\
    && echo "source activate llm-service" > ~/.bashrc
ENV PATH /opt/conda/envs/llm-service/bin:\$PATH

# Install Python dependencies
RUN /bin/bash -c "source activate llm-service && pip install flask gunicorn pyyaml requests"

# Clone and build llama.cpp
WORKDIR /opt/llm/models/ollama-custom-models
RUN git clone https://github.com/ggerganov/llama.cpp \\
    && cd llama.cpp \\
    && mkdir -p build \\
    && cd build \\
    && cmake .. -DGGML_CUDA=ON \\
    && make -j\$(nproc)

# Copy application files
COPY app/ /opt/llm/inference-service/app/
COPY config/ /opt/llm/inference-service/config/

# Expose API port
EXPOSE 8000

# Set working directory
WORKDIR /opt/llm/inference-service

# Start the service
CMD ["/bin/bash", "-c", "source activate llm-service && cd app && gunicorn --bind 0.0.0.0:8000 wsgi:app --workers 1 --threads 4"]
DOCKERFILE

# Create docker-compose.yml
cat > /opt/llm/inference-service/docker/docker-compose.yml << COMPOSE
version: '3.8'

services:
  llm-inference:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: llm-inference
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - /opt/llm/models/ollama/models:/opt/llm/models/ollama/models
      - ../logs:/opt/llm/inference-service/logs
      - ../config:/opt/llm/inference-service/config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
COMPOSE

# Create app.py
cat > /opt/llm/inference-service/app/app.py << 'PYTHON'
from flask import Flask, request, jsonify, Response, stream_with_context
import subprocess
import os
import yaml
import json
import threading
import logging
import tempfile
import time
import uuid
import re
from queue import Queue, Empty

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/llm/inference-service/logs/server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Active model instances
active_models = {}
model_locks = {}
config_path = '/opt/llm/inference-service/config/service_config.yaml'

def load_config():
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        'models_dir': '/opt/llm/models/ollama/models/blobs',
        'manifests_dir': '/opt/llm/models/ollama/models/manifests/registry.ollama.ai/library',
        'llama_cpp_path': '/opt/llm/models/ollama-custom-models/llama.cpp',
        'default_context_size': 4096,
        'default_model': None,
        'tensor_split': '0.25,0.25,0.25,0.25',
        'gpu_layers': 999,
        'threads': 8,
        'batch_size': 512
    }

config = load_config()

class OllamaModelMapper:
    def __init__(self, config_path='/opt/llm/inference-service/config/service_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = self.config['models_dir']
        self.manifests_dir = self.config['manifests_dir']
        self.model_cache = {}
        self._load_model_mappings()
    
    def _load_model_mappings(self):
        """Load all model mappings from Ollama manifests"""
        self.model_mappings = {}
        
        if not os.path.exists(self.manifests_dir):
            logger.warning(f"Manifests directory not found: {self.manifests_dir}")
            return
            
        for model_dir in os.listdir(self.manifests_dir):
            model_path = os.path.join(self.manifests_dir, model_dir)
            if os.path.isdir(model_path):
                for tag_file in os.listdir(model_path):
                    tag_path = os.path.join(model_path, tag_file)
                    if os.path.isfile(tag_path):
                        try:
                            with open(tag_path, 'r') as f:
                                manifest_content = f.read()
                                manifest = json.loads(manifest_content)
                                if 'layers' in manifest:
                                    for layer in manifest['layers']:
                                        if layer.get('type') == 'weights':
                                            model_id = f"{model_dir}/{tag_file}"
                                            blob = layer.get('digest', '').replace('sha256:', '')
                                            if blob:
                                                blob_path = os.path.join(self.models_dir, f"sha256-{blob}")
                                                if os.path.exists(blob_path):
                                                    self.model_mappings[model_id] = blob_path
                        except Exception as e:
                            logger.error(f"Error loading manifest {tag_path}: {str(e)}")
    
    def get_model_path(self, model_id):
        """Get the blob path for a given model ID"""
        if model_id in self.model_mappings:
            return self.model_mappings[model_id]
            
        # Check if it's a direct blob reference
        if model_id.startswith('sha256-'):
            blob_path = os.path.join(self.models_dir, model_id)
            if os.path.exists(blob_path):
                return blob_path
                
        # Try to find by partial match
        for key, path in self.model_mappings.items():
            if model_id in key:
                return path
                
        return None
    
    def list_models(self):
        """List all available models"""
        models = []
        
        # Include all mappings from manifests
        for model_id, path in self.model_mappings.items():
            size = os.path.getsize(path) if os.path.exists(path) else 0
            quant = "unknown"
            
            # Try to determine quantization from model name
            if "q4_k_m" in model_id:
                quant = "q4_k_m"
            elif "q5_k_m" in model_id:
                quant = "q5_k_m"
            elif "q6_k" in model_id:
                quant = "q6_k"
            elif "q8_0" in model_id:
                quant = "q8_0"
            elif "f16" in model_id:
                quant = "f16"
                
            models.append({
                "id": model_id,
                "name": model_id,
                "path": path,
                "size": size,
                "quantization": quant,
                "modelFormat": "gguf"
            })
            
        return models

model_mapper = OllamaModelMapper()

def save_config():
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

@app.route('/api/models', methods=['GET'])
def list_models():
    models = model_mapper.list_models()
    return jsonify({'models': models})

@app.route('/api/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    messages = data.get('messages', [])
    model_name = data.get('model')
    stream = data.get('stream', False)
    
    # Extract options
    options = data.get('options', {})
    context_size = options.get('context_size', config['default_context_size'])
    temperature = options.get('temperature', 0.7)
    max_tokens = options.get('max_tokens', 512)
    tensor_split = options.get('tensor_split', config['tensor_split'])
    gpu_layers = options.get('gpu_layers', config['gpu_layers'])
    threads = options.get('threads', config['threads'])
    batch_size = options.get('batch_size', config['batch_size'])
    
    if not messages:
        return jsonify({'error': 'No messages provided'}), 400
    
    if not model_name:
        if not config['default_model']:
            return jsonify({'error': 'No model specified and no default model set'}), 400
        model_name = config['default_model']
    
    model_path = model_mapper.get_model_path(model_name)
    if not model_path:
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    # Format prompt from messages
    system_prompt = next((msg['content'] for msg in messages if msg['role'] == 'system'), "")
    
    prompt = ""
    if system_prompt:
        prompt += f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    
    for msg in messages:
        if msg['role'] == 'system':
            continue
        
        content = msg['content']
        
        if msg['role'] == 'user':
            if prompt and not prompt.endswith("[/INST]"):
                prompt += f"[INST] {content} [/INST]"
            else:
                prompt += f"{content} [/INST]"
        elif msg['role'] == 'assistant':
            prompt += f" {content} </s><s>"
    
    # Ensure proper closing
    if not prompt.endswith("[/INST]"):
        prompt += "[/INST]"
        
    # Create temporary file for the prompt
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
        temp.write(prompt)
        prompt_file = temp.name
    
    # Generate a response ID
    response_id = str(uuid.uuid4())
    
    if stream:
        return Response(
            stream_with_context(generate_streaming_response(
                model_path, prompt_file, prompt, response_id, 
                context_size, temperature, max_tokens, 
                tensor_split, gpu_layers, threads, batch_size
            )),
            content_type='text/event-stream'
        )
    else:
        try:
            # Prepare the command
            cmd = [
                f"{config['llama_cpp_path']}/build/bin/llama-cli",
                "--model", model_path,
                "--ctx-size", str(context_size),
                "--temp", str(temperature),
                "--n-predict", str(max_tokens),
                "--tensor-split", tensor_split,
                "--n-gpu-layers", str(gpu_layers),
                "--threads", str(threads),
                "--batch-size", str(batch_size),
                "--file", prompt_file,
                "--color", "0"
            ]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Remove temp file
            os.unlink(prompt_file)
            
            if result.returncode != 0:
                logger.error(f"Error generating response: {result.stderr}")
                return jsonify({
                    'error': 'Failed to generate response', 
                    'details': result.stderr
                }), 500
            
            # Parse the output - get only the generated text, not the prompt
            generated_text = result.stdout.split('[/INST]')[-1].strip()
            
            # Format response in OpenAI-compatible format
            response = {
                'id': response_id,
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': model_name,
                'choices': [
                    {
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': generated_text
                        },
                        'finish_reason': 'stop'
                    }
                ],
                'usage': {
                    'prompt_tokens': len(prompt.split()),
                    'completion_tokens': len(generated_text.split()),
                    'total_tokens': len(prompt.split()) + len(generated_text.split())
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.exception("Error in generate endpoint")
            return jsonify({'error': str(e)}), 500

def generate_streaming_response(model_path, prompt_file, prompt, response_id, 
                               context_size, temperature, max_tokens,
                               tensor_split, gpu_layers, threads, batch_size):
    """Generate a streaming response compatible with OpenAI format"""
    
    cmd = [
        f"{config['llama_cpp_path']}/build/bin/llama-cli",
        "--model", model_path,
        "--ctx-size", str(context_size),
        "--temp", str(temperature),
        "--n-predict", str(max_tokens),
        "--tensor-split", tensor_split,
        "--n-gpu-layers", str(gpu_layers),
        "--threads", str(threads),
        "--batch-size", str(batch_size),
        "--file", prompt_file,
        "--color", "0"
    ]
    
    try:
        # Create a process and set up a queue for the output
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, bufsize=1
        )
        
        # Flag to track if we've passed the prompt
        passed_prompt = False
        accumulated_text = ""
        
        # Yield the first event to establish the connection
        yield f"data: {json.dumps({
            'id': response_id,
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': os.path.basename(model_path),
            'choices': [
                {
                    'index': 0,
                    'delta': {
                        'role': 'assistant'
                    },
                    'finish_reason': None
                }
            ]
        })}\n\n"
        
        # Iterate through the process output
        for line in process.stdout:
            # Check if we've passed the prompt
            if not passed_prompt and '[/INST]' in line:
                # Extract only the part after [/INST]
                parts = line.split('[/INST]')
                if len(parts) > 1:
                    line = parts[1]
                    passed_prompt = True
                else:
                    continue
            elif not passed_prompt:
                continue
            
            # Send each chunk
            yield f"data: {json.dumps({
                'id': response_id,
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': os.path.basename(model_path),
                'choices': [
                    {
                        'index': 0,
                        'delta': {
                            'content': line
                        },
                        'finish_reason': None
                    }
                ]
            })}\n\n"
            
            accumulated_text += line
            
        # Send the final chunk with finish_reason
        yield f"data: {json.dumps({
            'id': response_id,
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': os.path.basename(model_path),
            'choices': [
                {
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'stop'
                }
            ]
        })}\n\n"
        
        # End of stream
        yield "data: [DONE]\n\n"
        
        # Clean up
        process.terminate()
        os.unlink(prompt_file)
        
    except Exception as e:
        logger.exception("Error in streaming response")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
        if os.path.exists(prompt_file):
            os.unlink(prompt_file)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Legacy endpoint for compatibility"""
    return chat_completions()

@app.route('/api/generate', methods=['POST'])
def generate():
    """Legacy endpoint for compatibility"""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Convert to chat format
    prompt = data.get('prompt', '')
    messages = [{"role": "user", "content": prompt}]
    
    # Extract other parameters
    model_name = data.get('model')
    context_size = data.get('context_size', config['default_context_size'])
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 512)
    
    # Format for chat_completions
    chat_data = {
        'messages': messages,
        'model': model_name,
        'options': {
            'context_size': context_size,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'tensor_split': data.get('tensor_split', config['tensor_split']),
            'gpu_layers': data.get('gpu_layers', config['gpu_layers']),
            'threads': data.get('threads', config['threads']),
            'batch_size': data.get('batch_size', config['batch_size'])
        },
        'stream': data.get('stream', False)
    }
    
    # Use the request context with the new data
    with app.request_context(request.environ):
        request.json = chat_data
        return chat_completions()

@app.route('/api/config', methods=['GET', 'PUT'])
def manage_config():
    if request.method == 'GET':
        return jsonify(config)
    
    elif request.method == 'PUT':
        data = request.json
        for key, value in data.items():
            if key in config:
                config[key] = value
        
        save_config()
        return jsonify({'message': 'Configuration updated', 'config': config})

@app.route('/api/models/default', methods=['PUT'])
def set_default_model():
    data = request.json
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({'error': 'No model specified'}), 400
    
    model_path = model_mapper.get_model_path(model_name)
    if not model_path:
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    config['default_model'] = model_name
    save_config()
    
    return jsonify({'message': f'Default model set to {model_name}', 'default_model': model_name})

# OpenWebUI compatibility endpoints
@app.route('/api/tags', methods=['GET'])
def list_tags():
    # Return empty tags list for compatibility
    return jsonify({'tags': []})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    if not os.path.exists(config_path):
        save_config()
    app.run(host='0.0.0.0', port=8000, debug=True)
PYTHON

# Create wsgi.py
cat > /opt/llm/inference-service/app/wsgi.py << 'PYTHON'
from app import app

if __name__ == "__main__":
    app.run()
PYTHON

chmod +x /opt/llm/inference-service/scripts/build.sh
echo "Build script created successfully."
