#!/bin/bash
# Copyright 2025 LLM Inference Service Contributors
# Licensed under the Apache License, Version 2.0

# User-level service installation (no sudo required)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SERVICE_NAME="llm-inference"
SERVICE_FILE="llm-inference-user.service"

echo -e "${BLUE}🔧 Installing LLM Inference Service (User Level)${NC}"
echo "==============================================="

# Create user systemd directory
mkdir -p ~/.config/systemd/user

# Create user service file
cat > ~/.config/systemd/user/$SERVICE_NAME.service << EOF
[Unit]
Description=LLM Inference Service - Ollama Compatible API
Documentation=https://github.com/your-org/llm-inference-service
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/llm/inference-service
ExecStart=/usr/bin/python3 -m ollama_server.main --port 11435 --host 0.0.0.0 --model-dir /opt/llm/models/ollama/models --llama-cpp-dir /opt/llm/models/ollama-custom-models/llama.cpp/build --log-dir /opt/llm/inference-service/logs --default-tensor-split 0.25,0.25,0.25,0.25
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Environment variables
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_VISIBLE_DEVICES=0,1,2,3
Environment=NVIDIA_VISIBLE_DEVICES=all

[Install]
WantedBy=default.target
EOF

echo -e "${BLUE}🔄 Reloading systemd user daemon...${NC}"
systemctl --user daemon-reload

echo -e "${BLUE}🔗 Enabling service...${NC}"
systemctl --user enable $SERVICE_NAME

echo -e "${BLUE}🚀 Starting service...${NC}"
systemctl --user start $SERVICE_NAME

# Wait for service to start
sleep 3

# Check service status
if systemctl --user is-active --quiet $SERVICE_NAME; then
    echo -e "${GREEN}✅ Service installed and started successfully!${NC}"
    echo ""
    echo -e "${BLUE}📊 Service Status:${NC}"
    systemctl --user status $SERVICE_NAME --no-pager -l
    echo ""
    echo -e "${BLUE}🌐 Service URLs:${NC}"
    echo "  API Server:   http://localhost:11435"
    echo "  Health Check: http://localhost:11435/health"
    echo ""
    echo -e "${BLUE}📋 Useful Commands:${NC}"
    echo "  View status:    systemctl --user status $SERVICE_NAME"
    echo "  View logs:      journalctl --user -u $SERVICE_NAME -f"
    echo "  Stop service:   systemctl --user stop $SERVICE_NAME"
    echo "  Start service:  systemctl --user start $SERVICE_NAME"
    echo "  Restart:        systemctl --user restart $SERVICE_NAME"
    echo "  Disable:        systemctl --user disable $SERVICE_NAME"
else
    echo -e "${RED}❌ Service failed to start!${NC}"
    echo -e "${YELLOW}💡 Check logs with: journalctl --user -u $SERVICE_NAME -n 50${NC}"
    systemctl --user status $SERVICE_NAME --no-pager -l || true
    exit 1
fi