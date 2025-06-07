#!/bin/bash
# Copyright 2025 LLM Inference Service Contributors
# Licensed under the Apache License, Version 2.0

# Service uninstallation script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SERVICE_NAME="llm-inference"

echo -e "${BLUE}🗑️  Uninstalling LLM Inference Service${NC}"
echo "========================================"

# Check if systemd is available
if ! command -v systemctl &> /dev/null; then
    echo -e "${RED}❌ systemctl not found. This system doesn't appear to use systemd.${NC}"
    exit 1
fi

# Stop service if it's running
if systemctl is-active --quiet $SERVICE_NAME 2>/dev/null; then
    echo -e "${YELLOW}⏹️  Stopping service...${NC}"
    sudo systemctl stop $SERVICE_NAME
fi

# Disable the service
if systemctl is-enabled --quiet $SERVICE_NAME 2>/dev/null; then
    echo -e "${BLUE}🔗 Disabling service...${NC}"
    sudo systemctl disable $SERVICE_NAME
fi

# Remove service file
if [[ -f "/etc/systemd/system/$SERVICE_NAME.service" ]]; then
    echo -e "${BLUE}🗑️  Removing service file...${NC}"
    sudo rm /etc/systemd/system/$SERVICE_NAME.service
fi

# Reload systemd daemon
echo -e "${BLUE}🔄 Reloading systemd daemon...${NC}"
sudo systemctl daemon-reload

# Reset failed state if any
sudo systemctl reset-failed $SERVICE_NAME 2>/dev/null || true

echo -e "${GREEN}✅ Service uninstalled successfully!${NC}"
echo ""
echo -e "${BLUE}📋 Cleanup Complete:${NC}"
echo "  Service stopped and disabled"
echo "  Service file removed from /etc/systemd/system/"
echo "  systemd daemon reloaded"
echo ""
echo -e "${YELLOW}💡 Note: Log files in /opt/llm/inference-service/logs are preserved${NC}"