#!/bin/bash
cd /opt/llm/inference-service
mkdir -p logs

# Run the API server
python app/ollama_compatible_server.py \
  --model-dir /opt/llm/models/ollama/models \
  --llama-cpp-dir /opt/llm/models/ollama-custom-models/llama.cpp/build \
  --log-dir /opt/llm/inference-service/logs \
  --port 8000 \
  --default-tensor-split "0.25,0.25,0.25,0.25"
