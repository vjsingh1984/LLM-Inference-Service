#!/bin/bash

cd /opt/llm/inference-service/docker

# Build the Docker image
docker-compose build

# Run the container
docker-compose up -d

echo "LLM Inference Service started on port 8000."
echo "To view logs: docker logs -f llm-inference"
echo ""
echo "To test, run:"
echo 'curl -X POST http://localhost:8000/api/chat/completions -H "Content-Type: application/json" -d '\''{"model":"phi-4-q8_0-16K-custom:latest","messages":[{"role":"user","content":"What is quantum computing?"}],"options":{"tensor_split":"0.25,0.25,0.25,0.25","gpu_layers":999}}'\'''
