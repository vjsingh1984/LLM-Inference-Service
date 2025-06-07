# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM inference service that provides an Ollama-compatible API server for serving Large Language Models using llama.cpp as the backend. The service is designed to run models efficiently with multi-GPU support and provides OpenAI-compatible endpoints.

## Architecture

The service consists of:
- **Flask-based API server** that implements Ollama API endpoints
- **llama.cpp integration** for high-performance model inference with GPU acceleration
- **Model management system** using Ollama manifest format
- **Docker deployment** with GPU passthrough support

Key components:
- API endpoints handle chat completions, model listing, and configuration management
- Models are stored in GGUF format in `/opt/llm/models/ollama/models/`
- Configuration is managed via `config/service_config.yaml`
- Supports tensor splitting across multiple GPUs for large models

## Dashboard Implementation Plan

### High Priority Features (Implement First)
1. ✅ GPU monitoring utilities (`ollama_server/utils/gpu_monitor.py`)
2. 🔄 Real-Time GPU Monitoring Panel with live metrics
3. 🔄 Weight Distribution Visualizer with interactive controls
4. 🔄 Model Performance Analytics dashboard
5. 🔄 API Endpoint Health Monitor

### Medium Priority Features
6. 🔄 Dynamic Configuration Panel for live config changes
7. 🔄 Hardware Optimization Insights
8. 🔄 Production Monitoring features
9. 🔄 Interactive Model Explorer

### Low Priority Features
10. 🔄 Cost-Effectiveness Calculator
11. 🔄 Documentation updates

## TODO Items

### Phase 1: Foundation (High Priority)
- [x] Create GPU monitoring utilities with nvidia-ml-py support
- [ ] Create Jinja2 template system for dashboard components
- [ ] Implement Real-Time GPU Monitoring Panel
  - [ ] Live GPU utilization display (dual RTX 4090 focus)
  - [ ] Memory usage visualization per card
  - [ ] Temperature and power monitoring with alerts
  - [ ] PCIe bandwidth utilization tracking
- [ ] Build Weight Distribution Visualizer
  - [ ] Interactive tensor split configuration UI
  - [ ] Visual weight distribution sliders
  - [ ] Preset configurations (Equal, Memory-based, Custom)
  - [ ] Live validation and updates without restart

### Phase 2: Analytics & Monitoring (High Priority)
- [ ] Create Model Performance Analytics
  - [ ] Context length utilization tracking (4K → 131K detection)
  - [ ] Token generation rate metrics
  - [ ] Memory efficiency per model
  - [ ] Request latency breakdown visualization
- [ ] Implement API Endpoint Health Monitor
  - [ ] Multi-API status board (OpenAI, Ollama, vLLM, HuggingFace)
  - [ ] Response time tracking per endpoint
  - [ ] Error rate monitoring
  - [ ] Concurrent request visualization

### Phase 3: Advanced Features (Medium Priority)
- [ ] Build Dynamic Configuration Panel
  - [ ] Live tensor split adjustment interface
  - [ ] Context size scaling controls
  - [ ] Batch size optimization settings
  - [ ] GPU layer distribution controls
- [ ] Add Hardware Optimization Insights
  - [ ] Consumer GPU intelligence dashboard
  - [ ] PCIe topology visualization
  - [ ] Memory bandwidth utilization analysis
  - [ ] Optimal split recommendations engine
- [ ] Implement Production Monitoring
  - [ ] Request queue visualization
  - [ ] Service uptime tracking
  - [ ] Performance trend analysis
  - [ ] Alert system for hardware issues

### Phase 4: Enhancement Features (Low Priority)
- [ ] Create Interactive Model Explorer
  - [ ] Auto-detected vs reported context lengths
  - [ ] Model parameter discovery interface
  - [ ] Quantization impact analysis
  - [ ] GPU allocation recommendations
- [ ] Build Cost-Effectiveness Calculator
  - [ ] Dual RTX 4090 vs H100 cost comparison
  - [ ] Performance per dollar metrics
  - [ ] Power efficiency calculations
  - [ ] Hardware upgrade recommendations

## Technical Implementation Guidelines

### Template System
- Use Jinja2 templates for all HTML components
- Create modular template components for reusability
- Implement template inheritance for consistent layouts
- Store templates in `ollama_server/templates/`

### Dashboard Architecture
- Separate frontend components into logical modules
- Use WebSocket or Server-Sent Events for real-time updates
- Implement responsive design for various screen sizes
- Add dark/light theme support

### API Integration
- Create dedicated dashboard API endpoints
- Implement proper error handling and fallbacks
- Add caching for expensive operations (GPU metrics)
- Support both REST and WebSocket protocols

### Performance Considerations
- Cache GPU metrics to avoid excessive nvidia-smi calls
- Use background threads for continuous monitoring
- Implement efficient data structures for metrics storage
- Add compression for real-time data transfer

## Common Development Commands

### Starting the Enhanced Dashboard
```bash
# Install additional dependencies
pip install jinja2 pynvml websocket-client

# Start with enhanced monitoring
python ollama_server/main.py \
  --model-dir /opt/llm/models/ollama/models \
  --llama-cpp-dir /opt/llm/models/ollama-custom-models/llama.cpp/build \
  --port 8000 \
  --enable-gpu-monitoring \
  --dashboard-updates real-time
```

### Testing Dashboard Components
```bash
# Test GPU monitoring
python -c "from ollama_server.utils.gpu_monitor import gpu_monitor; print(gpu_monitor.to_dict())"

# Test template rendering
curl http://localhost:8000/dashboard

# Test real-time metrics
curl http://localhost:8000/api/dashboard/gpu-metrics
```

## Dashboard Features Specification

### Real-Time GPU Panel
- **Dual RTX 4090 Focus**: Optimized for dual high-end consumer GPU setups
- **Live Metrics**: 1-second update intervals for utilization, memory, temperature
- **Visual Indicators**: Color-coded status (green/yellow/red) based on thresholds
- **Historical Data**: 5-minute rolling history with trend analysis

### Weight Distribution Control
- **Interactive Sliders**: Drag-and-drop tensor split configuration
- **Validation**: Real-time validation of split ratios
- **Presets**: Quick access to common configurations (50/50, 60/40, etc.)
- **Live Updates**: Apply changes without service restart

### Model Analytics
- **Context Detection**: Show true vs reported context lengths
- **Performance Tracking**: Token/second, latency, throughput metrics
- **Memory Efficiency**: VRAM usage optimization suggestions
- **Model Comparison**: Side-by-side performance analysis

### Hardware Insights
- **PCIe Analysis**: Bandwidth utilization across multiple slots
- **Consumer GPU Intelligence**: RTX 4090/4080 optimization tips
- **Power Efficiency**: Watts per token calculations
- **Cooling Monitoring**: Temperature trends and thermal throttling detection

## Important Notes

- Dashboard focuses on consumer GPU optimization (RTX 4090, RTX 4080, etc.)
- Emphasizes cost-effective alternatives to datacenter hardware
- Provides enterprise-grade monitoring for home lab setups
- Supports mixed GPU configurations and dynamic workload balancing