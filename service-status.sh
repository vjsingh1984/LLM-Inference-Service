#!/bin/bash
# Copyright 2025 LLM Inference Service Contributors
# Licensed under the Apache License, Version 2.0

# Service status and management script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SERVICE_NAME="llm-inference"

echo -e "${BLUE}📊 LLM Inference Service Status${NC}"
echo "=================================="

# Check if systemd is available
if ! command -v systemctl &> /dev/null; then
    echo -e "${RED}❌ systemctl not found. This system doesn't appear to use systemd.${NC}"
    exit 1
fi

# Check if service exists
if ! systemctl list-unit-files | grep -q "^$SERVICE_NAME.service"; then
    echo -e "${RED}❌ Service not installed${NC}"
    echo -e "${YELLOW}💡 Run './install-service.sh' to install the service${NC}"
    exit 1
fi

# Service status
echo -e "${BLUE}🔍 Service Status:${NC}"
if systemctl is-active --quiet $SERVICE_NAME; then
    echo -e "${GREEN}  ✅ Running${NC}"
else
    echo -e "${RED}  ❌ Stopped${NC}"
fi

if systemctl is-enabled --quiet $SERVICE_NAME; then
    echo -e "${GREEN}  ✅ Enabled (auto-start)${NC}"
else
    echo -e "${YELLOW}  ⚠️  Disabled${NC}"
fi

echo ""

# Detailed status
echo -e "${BLUE}📋 Detailed Status:${NC}"
sudo systemctl status $SERVICE_NAME --no-pager -l || true

echo ""

# Port check
echo -e "${BLUE}🌐 Port Status:${NC}"
if ss -tuln | grep -q ":11435 "; then
    echo -e "${GREEN}  ✅ Port 11435 is listening${NC}"
    
    # Test health endpoint
    if command -v curl &> /dev/null; then
        echo -e "${BLUE}🏥 Health Check:${NC}"
        if curl -s --max-time 5 http://localhost:11435/health > /dev/null; then
            echo -e "${GREEN}  ✅ Service is healthy${NC}"
        else
            echo -e "${RED}  ❌ Health check failed${NC}"
        fi
    fi
else
    echo -e "${RED}  ❌ Port 11435 is not listening${NC}"
fi

echo ""

# Recent logs
echo -e "${BLUE}📝 Recent Logs (last 10 lines):${NC}"
sudo journalctl -u $SERVICE_NAME -n 10 --no-pager || true

echo ""

# Service URLs
echo -e "${BLUE}🌐 Service URLs:${NC}"
echo "  API Server:   http://localhost:11435"
echo "  Health Check: http://localhost:11435/health"

echo ""

# Management commands
echo -e "${BLUE}📋 Management Commands:${NC}"
echo "  Start:    sudo systemctl start $SERVICE_NAME"
echo "  Stop:     sudo systemctl stop $SERVICE_NAME"
echo "  Restart:  sudo systemctl restart $SERVICE_NAME"
echo "  Logs:     sudo journalctl -u $SERVICE_NAME -f"
echo "  Status:   ./service-status.sh"