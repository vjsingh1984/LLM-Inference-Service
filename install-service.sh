#!/bin/bash
# Copyright 2025 LLM Inference Service Contributors
# Licensed under the Apache License, Version 2.0

# Service installation script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SERVICE_NAME="llm-inference"
SERVICE_FILE="llm-inference.service"
INSTALL_DIR="/opt/llm/inference-service"

echo -e "${BLUE}🔧 Installing LLM Inference Service${NC}"
echo "========================================"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}❌ This script should not be run as root${NC}"
   echo "Please run as your regular user with sudo for system operations"
   exit 1
fi

# Check if we're in the right directory
if [[ ! -f "$SERVICE_FILE" ]]; then
    echo -e "${RED}❌ Service file not found: $SERVICE_FILE${NC}"
    echo "Please run this script from the inference service directory"
    exit 1
fi

# Check if Python dependencies are installed
echo -e "${BLUE}📦 Checking Python dependencies...${NC}"
if ! python3 -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Installing Python dependencies...${NC}"
    pip3 install --user -r requirements.txt
fi

# Create log directory if it doesn't exist
mkdir -p logs

# Check if systemd is available
if ! command -v systemctl &> /dev/null; then
    echo -e "${RED}❌ systemctl not found. This system doesn't appear to use systemd.${NC}"
    exit 1
fi

# Stop service if it's already running
if systemctl is-active --quiet $SERVICE_NAME 2>/dev/null; then
    echo -e "${YELLOW}⏹️  Stopping existing service...${NC}"
    sudo systemctl stop $SERVICE_NAME
fi

# Copy service file to systemd directory
echo -e "${BLUE}📋 Installing service file...${NC}"
sudo cp $SERVICE_FILE /etc/systemd/system/

# Reload systemd daemon
echo -e "${BLUE}🔄 Reloading systemd daemon...${NC}"
sudo systemctl daemon-reload

# Enable the service
echo -e "${BLUE}🔗 Enabling service...${NC}"
sudo systemctl enable $SERVICE_NAME

# Start the service
echo -e "${BLUE}🚀 Starting service...${NC}"
sudo systemctl start $SERVICE_NAME

# Wait a moment for service to start
sleep 3

# Check service status
if systemctl is-active --quiet $SERVICE_NAME; then
    echo -e "${GREEN}✅ Service installed and started successfully!${NC}"
    echo ""
    echo -e "${BLUE}📊 Service Status:${NC}"
    sudo systemctl status $SERVICE_NAME --no-pager -l
    echo ""
    echo -e "${BLUE}🌐 Service URLs:${NC}"
    echo "  API Server:   http://localhost:11435"
    echo "  Health Check: http://localhost:11435/health"
    echo ""
    echo -e "${BLUE}📋 Useful Commands:${NC}"
    echo "  View status:    sudo systemctl status $SERVICE_NAME"
    echo "  View logs:      sudo journalctl -u $SERVICE_NAME -f"
    echo "  Stop service:   sudo systemctl stop $SERVICE_NAME"
    echo "  Start service:  sudo systemctl start $SERVICE_NAME"
    echo "  Restart:        sudo systemctl restart $SERVICE_NAME"
    echo "  Disable:        sudo systemctl disable $SERVICE_NAME"
else
    echo -e "${RED}❌ Service failed to start!${NC}"
    echo -e "${YELLOW}💡 Check logs with: sudo journalctl -u $SERVICE_NAME -n 50${NC}"
    exit 1
fi