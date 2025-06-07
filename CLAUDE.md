# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM inference service that provides an Ollama-compatible API server for serving Large Language Models using llama.cpp as the backend. The service is designed to run models efficiently with multi-GPU support and provides OpenAI-compatible endpoints.

## Architecture

The service consists of:
- **Flask-based API server** (`app/ollama_compatible_server.py`) that implements Ollama API endpoints
- **llama.cpp integration** for high-performance model inference with GPU acceleration
- **Model management system** using Ollama manifest format
- **Docker deployment** with GPU passthrough support

Key components:
- API endpoints handle chat completions, model listing, and configuration management
- Models are stored in GGUF format in `/opt/llm/models/ollama/models/`
- Configuration is managed via `config/service_config.yaml`
- Supports tensor splitting across multiple GPUs for large models

## Common Development Commands

### Starting the Server

```bash
# Refactored modular server (recommended)
cd /opt/llm/inference-service
python ollama_server/main.py \
  --model-dir /opt/llm/models/ollama/models \
  --llama-cpp-dir /opt/llm/models/ollama-custom-models/llama.cpp/build \
  --port 8000

# Legacy monolithic server
python app/ollama_compatible_server.py \
  --model-dir /opt/llm/models/ollama/models \
  --llama-cpp-dir /opt/llm/models/ollama-custom-models/llama.cpp/build \
  --port 8000

# Using convenience script
./scripts/start-ollama-compatible.sh
```

### Docker Operations

```bash
# Build and start with Docker
cd /opt/llm/inference-service/docker
docker-compose build
docker-compose up -d

# View logs
docker logs -f llm-inference

# Using convenience script
./scripts/start-docker.sh
```

### Testing the API

```bash
# List available models
curl http://localhost:8000/api/models

# Test chat completion
curl -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4-q8_0-16K-custom:latest",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Check health
curl http://localhost:8000/health
```

### Building the Project

```bash
# Initial setup (creates directory structure and configs)
./scripts/build.sh
```

## Configuration

The service uses `config/service_config.yaml` with these key settings:
- `models_dir`: Path to model blob storage
- `manifests_dir`: Path to Ollama manifests
- `llama_cpp_path`: Path to llama.cpp binary
- `tensor_split`: GPU memory allocation (e.g., "0.25,0.25,0.25,0.25" for 4 GPUs)
- `gpu_layers`: Number of layers to offload to GPU (999 for all)
- `default_context_size`: Default context window size

## Model Management

Models are stored in Ollama format:
- Blob files in `/opt/llm/models/ollama/models/blobs/`
- Manifests in `/opt/llm/models/ollama/models/manifests/`
- Model conversion tools in `/opt/llm/inference-service/models/model-conversion/`

The service supports various quantized GGUF models including Llama, Phi, and Mixtral variants.

## New Features (Refactored Version)

### Think Tag Preservation
The refactored server now properly preserves `<think>...</think>` tags in responses for reasoning models like phi4-reasoning:
- Ollama API formats preserve think tags by default (matches official Ollama behavior)
- OpenAI and other formats strip think tags for clean responses
- Configurable per API format

### Modular Architecture
The codebase has been refactored into a clean, maintainable structure:
- `ollama_server/` - Main package with proper separation of concerns
- `ollama_server/models/` - Model management and discovery
- `ollama_server/core/` - Core inference functionality 
- `ollama_server/adapters/` - API format adapters
- `ollama_server/api/` - Flask routes and request handlers
- `ollama_server/utils/` - Utilities including response processing

### Testing
Run the test suite to verify functionality:
```bash
python test_refactored_server.py
```

## Debugging

- Server logs: `/opt/llm/inference-service/logs/server.log`
- Enable debug mode: `python ollama_server/main.py --debug`
- Check GPU utilization with `nvidia-smi` during inference
- Monitor requests via dashboard: `http://localhost:8000/dashboard`