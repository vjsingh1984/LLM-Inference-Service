FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ollama_api.py .

# Create required directories
RUN mkdir -p /opt/llm/inference-service/logs

# Set up volumes
VOLUME ["/opt/llm/models/ollama/models", "/opt/llm/models/ollama-custom-models/llama.cpp"]

# Expose port
EXPOSE 8000

# Start the API server
CMD ["python", "ollama_api.py"]
