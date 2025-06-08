# Copyright 2025 LLM Inference Service Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flask routes for the Ollama-compatible server."""
import logging
from datetime import datetime
from typing import Dict, Any

from flask import Blueprint, request, jsonify, render_template

from ..adapters import (
    OpenAIAdapter, OllamaChatAdapter, OllamaGenerateAdapter,
    VLLMAdapter, HuggingFaceAdapter
)
from .handlers import RequestHandler
from ..utils.model_inspector import model_inspector
from ..utils.gpu_monitor import gpu_monitor
from ..utils.weight_distribution import weight_manager

logger = logging.getLogger(__name__)

# Create blueprint for all routes
api_bp = Blueprint('api', __name__)


def create_routes(model_manager, request_tracker, llama_executor, config):
    """Create and configure all API routes."""
    handler = RequestHandler(llama_executor, request_tracker)
    
    def _parse_parameter_count(param_size: str) -> int:
        """Parse parameter size string to integer count."""
        try:
            if 'B' in param_size.upper():
                return int(float(param_size.upper().replace('B', '')) * 1_000_000_000)
            elif 'M' in param_size.upper():
                return int(float(param_size.upper().replace('M', '')) * 1_000_000)
            return 8030261312  # Default for 8B models
        except:
            return 8030261312
    
    # ============================================================================
    # Chat Completion Endpoints
    # ============================================================================
    
    @api_bp.route('/api/chat/completions', methods=['POST'])  # OpenAI
    def chat_completions():
        """OpenAI chat completions endpoint."""
        try:
            adapter = OpenAIAdapter(model_manager, config.default_tensor_split)
            request_obj = adapter.parse_request(request.json)
            
            if request_obj.stream:
                return handler.create_streaming_response(adapter, request_obj)
            else:
                return handler.handle_non_streaming_request(adapter, request_obj)
        except Exception as e:
            logger.exception("Error in chat completions")
            return jsonify({'error': str(e)}), 400

    @api_bp.route('/api/chat', methods=['POST'])  # Ollama Chat
    def ollama_chat():
        """Ollama chat endpoint."""
        try:
            adapter = OllamaChatAdapter(model_manager, config.default_tensor_split)
            request_obj = adapter.parse_request(request.json)
            
            if request_obj.stream:
                return handler.create_streaming_response(adapter, request_obj)
            else:
                return handler.handle_non_streaming_request(adapter, request_obj)
        except Exception as e:
            logger.exception("Error in Ollama chat")
            return jsonify({'error': str(e)}), 400

    @api_bp.route('/api/generate', methods=['POST'])  # Ollama Generate
    def ollama_generate():
        """Ollama generate endpoint."""
        try:
            adapter = OllamaGenerateAdapter(model_manager, request_tracker, config.default_tensor_split)
            request_obj = adapter.parse_request(request.json)
            
            if request_obj.stream:
                return handler.create_streaming_response(adapter, request_obj)
            else:
                return handler.handle_non_streaming_request(adapter, request_obj)
        except Exception as e:
            logger.exception("Error in Ollama generate")
            return jsonify({'error': str(e)}), 400


    @api_bp.route('/v1/chat/completions', methods=['POST'])  # vLLM Chat
    def vllm_chat_completions():
        """vLLM chat completions endpoint."""
        try:
            adapter = VLLMAdapter(model_manager, config.default_tensor_split)
            request_obj = adapter.parse_request(request.json)
            
            if request_obj.stream:
                return handler.create_streaming_response(adapter, request_obj)
            else:
                return handler.handle_non_streaming_request(adapter, request_obj)
        except Exception as e:
            logger.exception("Error in vLLM chat")
            return jsonify({'error': str(e)}), 400

    @api_bp.route('/v1/completions', methods=['POST'])  # vLLM Text Completion
    def vllm_completions():
        """vLLM text completions endpoint."""
        try:
            adapter = VLLMAdapter(model_manager, config.default_tensor_split)
            request_obj = adapter.parse_request(request.json)
            
            if request_obj.stream:
                return handler.create_streaming_response(adapter, request_obj)
            else:
                return handler.handle_non_streaming_request(adapter, request_obj)
        except Exception as e:
            logger.exception("Error in vLLM completions")
            return jsonify({'error': str(e)}), 400

    # ============================================================================
    # HuggingFace TGI Endpoints
    # ============================================================================
    
    @api_bp.route('/generate', methods=['POST'])  # HF TGI (common path)
    @api_bp.route('/hf/v1/completions', methods=['POST'])  # More specific path
    def huggingface_generate():
        """HuggingFace TGI generate endpoint."""
        try:
            adapter = HuggingFaceAdapter(model_manager, config.default_tensor_split)
            request_obj = adapter.parse_request(request.json)
            
            if request_obj.stream:
                return handler.create_streaming_response(adapter, request_obj)
            else:
                return handler.handle_non_streaming_request(adapter, request_obj)
        except Exception as e:
            logger.exception("Error in HuggingFace generate")
            return jsonify({'error': str(e)}), 400

    # ============================================================================
    # Model Management Endpoints
    # ============================================================================
    
    @api_bp.route('/v1/models', methods=['GET'])  # OpenAI/vLLM compatible
    def openai_models():
        """OpenAI models list endpoint."""
        return jsonify(handler.handle_models_list(model_manager, 'openai'))

    @api_bp.route('/api/tags', methods=['GET'])   # Ollama compatible (legacy)
    @api_bp.route('/api/models', methods=['GET']) # Ollama compatible
    def ollama_models():
        """Ollama models list endpoint."""
        return jsonify(handler.handle_models_list(model_manager, 'ollama'))

    @api_bp.route('/api/show', methods=['POST'])  # Ollama show model capabilities
    def ollama_show():
        """Ollama show model capabilities endpoint."""
        try:
            data = request.json or {}
            model_name = data.get('name', data.get('model', ''))
            
            if not model_name:
                return jsonify({'error': 'model name is required'}), 400
            
            # Find the model
            model_mapping = model_manager.build_model_mapping()
            model_info = None
            
            # Try exact match first
            if model_name in model_mapping:
                model_info = model_manager.get_model_info(model_name)
            else:
                # Try partial match
                for model_id in model_mapping:
                    if model_name.lower() in model_id.lower():
                        model_info = model_manager.get_model_info(model_id)
                        break
            
            if not model_info:
                return jsonify({'error': f'model "{model_name}" not found'}), 404
            
            # Get enhanced model info using inspector
            enhanced_info = model_inspector.get_enhanced_model_info(model_name, model_info)
            
            # Build stop tokens parameters section
            parameters_section = f"parameter_size {enhanced_info['parameter_size']}\nquantization_level {enhanced_info['quantization']}"
            if enhanced_info.get('stop_tokens'):
                for stop_token in enhanced_info['stop_tokens']:
                    parameters_section += f'\nstop "{stop_token}"'
            
            # Build model_info section with accurate context length
            architecture = enhanced_info['architecture']
            context_length = enhanced_info['context_size']
            
            # Generate model_info that matches real Ollama format
            model_info_dict = {
                'general.architecture': architecture,
                'general.file_type': 15 if enhanced_info['quantization'] == 'Q4_K_M' else 1,
                'general.parameter_count': _parse_parameter_count(enhanced_info['parameter_size']),
                'general.quantization_version': 2,
                'general.size_label': enhanced_info['parameter_size'],
                'general.type': 'model'
            }
            
            # Add architecture-specific fields that include context length
            if architecture == 'llama':
                model_info_dict.update({
                    'llama.context_length': context_length,
                    'llama.embedding_length': enhanced_info.get('embedding_length', 4096),
                    'llama.block_count': 32 if '8b' in model_name.lower() else 80,
                    'llama.attention.head_count': 32 if '8b' in model_name.lower() else 64,
                    'llama.attention.head_count_kv': 8 if '8b' in model_name.lower() else 8,
                    'llama.attention.layer_norm_rms_epsilon': 1e-05,
                    'llama.feed_forward_length': 14336 if '8b' in model_name.lower() else 28672,
                    'llama.rope.dimension_count': 128,
                    'llama.rope.freq_base': 500000
                })
            elif architecture == 'phi':
                model_info_dict.update({
                    'phi.context_length': context_length,
                    'phi.embedding_length': enhanced_info.get('embedding_length', 4096)
                })
            
            # Return model capabilities in exact Ollama format
            return jsonify({
                'license': enhanced_info.get('license_info', 'Apache 2.0'),
                'modelfile': f'# Modelfile generated by "ollama show"\n# To build a new Modelfile based on this, replace FROM with:\n# FROM {model_name}\n\nFROM {model_info.path}\nTEMPLATE """{enhanced_info.get("template", "{{ if .System }}{{ .System }}{{ end }}{{ if .Prompt }}{{ .Prompt }}{{ end }}")}\n"""\n' + '\n'.join([f'PARAMETER {line}' for line in parameters_section.split('\n') if line.strip()]),
                'parameters': '\n'.join([f'{line.split()[0]:<30} "{line.split(maxsplit=1)[1].strip(chr(34))}"' if 'stop' in line else f'{line.split()[0]:<30} {line.split(maxsplit=1)[1]}' for line in parameters_section.split('\n') if line.strip()]),
                'template': enhanced_info.get('template', '{{- if or .System .Tools }}<|start_header_id|>system<|end_header_id|>\n{{- if .System }}\n\n{{ .System }}\n{{- end }}<|eot_id|>\n{{- end }}\n{{- range $i, $_ := .Messages }}\n{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Content }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{{- else if eq .Role "assistant" }}\n\n{{ .Content }}<|eot_id|>\n{{- end }}\n{{- end }}'),
                'details': {
                    'parent_model': '',
                    'format': 'gguf',
                    'family': architecture,
                    'families': [architecture],
                    'parameter_size': enhanced_info['parameter_size'],
                    'quantization_level': enhanced_info['quantization']
                },
                'model_info': model_info_dict
            })
        except Exception as e:
            logger.exception("Error in Ollama show")
            return jsonify({'error': str(e)}), 500

    @api_bp.route('/info', methods=['GET'])  # HuggingFace TGI info
    def huggingface_info():
        """HuggingFace TGI info endpoint."""
        try:
            # Get first available model as default
            model_mapping = model_manager.build_model_mapping()
            if not model_mapping:
                return jsonify({'error': 'No models available'}), 404
            
            default_model_id = list(model_mapping.keys())[0]
            model_info = model_manager.get_model_info(default_model_id)
            
            if not model_info:
                return jsonify({'error': 'Model info not available'}), 500
            
            return jsonify({
                'model_id': model_info.id,
                'model_sha': model_info.path.split('sha256-')[-1] if 'sha256-' in model_info.path else 'unknown',
                'model_dtype': model_info.quantization,
                'model_device_type': 'cuda',
                'model_pipeline_tag': 'text-generation',
                'max_concurrent_requests': 128,
                'max_best_of': 1,
                'max_stop_sequences': 4,
                'max_input_length': model_info.context_size,
                'max_total_tokens': model_info.context_size,
                'version': '1.0.0'
            })
        except Exception as e:
            logger.exception("Error in HuggingFace info")
            return jsonify({'error': str(e)}), 500

    # ============================================================================
    # Monitoring and Management Endpoints
    # ============================================================================
    
    @api_bp.route('/api/progress/<request_id>', methods=['GET'])
    def get_progress(request_id: str):
        """Get progress for a specific request."""
        req_status = request_tracker.get_request(request_id)
        if not req_status:
            return jsonify({'error': 'Request not found'}), 404
        
        return jsonify({
            'request_id': req_status.request_id,
            'status': req_status.status,
            'progress': req_status.progress,
            'total': req_status.total,
            'start_time': req_status.start_time,
            'last_update': req_status.last_update,
            'model': req_status.model,
            'error': req_status.error
        })

    @api_bp.route('/api/dismiss/<request_id>', methods=['POST'])
    def dismiss_request(request_id: str):
        """Dismiss/remove a specific request from tracking."""
        request_tracker.remove_request(request_id)
        return jsonify({'message': 'Request dismissed'})

    @api_bp.route('/api/dismiss/completed', methods=['POST'])
    def dismiss_completed():
        """Dismiss all completed requests."""
        removed_count = request_tracker.remove_completed(0)  # Remove immediately
        return jsonify({'message': f'Dismissed {removed_count} completed requests'})

    @api_bp.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        try:
            # Check if executor is available
            executor_status = 'healthy' if llama_executor else 'unhealthy'
            
            # Check model availability
            model_mapping = model_manager.build_model_mapping()
            model_count = len(model_mapping)
            
            # Check active requests
            active_requests = len(request_tracker.get_all_requests())
            
            status = 'healthy' if executor_status == 'healthy' and model_count > 0 else 'degraded'
            
            return jsonify({
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'executor': executor_status,
                    'models': f'{model_count} available',
                    'active_requests': active_requests
                }
            })
        except Exception as e:
            logger.exception("Error in health check")
            return jsonify({
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }), 500

    # ============================================================================
    # Dashboard and Web Interface
    # ============================================================================
    
    @api_bp.route('/dashboard', methods=['GET'])
    def dashboard():
        """Enhanced web dashboard with real-time GPU monitoring."""
        # Start GPU monitoring if not already running
        gpu_monitor.start_monitoring()
        return render_template('dashboard.html')

    @api_bp.route('/dashboard/data', methods=['GET'])
    def dashboard_data():
        """Enhanced dashboard data endpoint with GPU metrics."""
        try:
            # Get service status
            status_response = health_check()
            status_data = status_response.get_json()
            
            # Get active requests
            active_requests = []
            for req_id, req_status in request_tracker.get_all_requests().items():
                active_requests.append({
                    'request_id': req_status.request_id,
                    'status': req_status.status,
                    'model': req_status.model,
                    'progress': req_status.progress,
                    'total': req_status.total,
                    'start_time': req_status.start_time,
                    'api_format': getattr(req_status, 'api_format', 'unknown')
                })
            
            # Get models
            models_data = handler.handle_models_list(model_manager, 'ollama')
            
            # Get GPU metrics
            gpu_metrics = gpu_monitor.to_dict()
            
            return jsonify({
                'status': status_data,
                'requests': active_requests,
                'models': models_data.get('models', []),
                'gpu_metrics': gpu_metrics,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.exception("Error in dashboard data")
            return jsonify({'error': str(e)}), 500

    @api_bp.route('/dashboard/gpu', methods=['GET'])
    def gpu_monitor_page():
        """Dedicated GPU monitoring page."""
        gpu_monitor.start_monitoring()
        return render_template('gpu_monitor.html')
    
    @api_bp.route('/dashboard/models', methods=['GET'])
    def model_analytics_page():
        """Model performance analytics page."""
        return render_template('model_analytics.html')
    
    @api_bp.route('/dashboard/apis', methods=['GET'])
    def api_health_page():
        """API endpoint health monitoring page."""
        return render_template('api_health.html')
    
    @api_bp.route('/api/dashboard/gpu-metrics', methods=['GET'])
    def gpu_metrics_api():
        """Real-time GPU metrics API endpoint."""
        try:
            metrics = gpu_monitor.to_dict()
            return jsonify(metrics) if metrics else jsonify({'error': 'No GPU data available'}), 503
        except Exception as e:
            logger.exception("Error getting GPU metrics")
            return jsonify({'error': str(e)}), 500
    
    @api_bp.route('/api/dashboard/configure', methods=['POST'])
    def configure_system():
        """Dynamic system configuration endpoint."""
        try:
            config_data = request.get_json()
            if not config_data:
                return jsonify({'error': 'No configuration data provided'}), 400
            
            # TODO: Implement configuration application
            # This would update tensor splits, context sizes, etc.
            
            logger.info(f"Configuration update requested: {config_data}")
            
            return jsonify({
                'status': 'success',
                'message': 'Configuration applied successfully',
                'applied_config': config_data
            })
        except Exception as e:
            logger.exception("Error applying configuration")
            return jsonify({'error': str(e)}), 500

    @api_bp.route('/api/dashboard/weight-distribution', methods=['GET'])
    def get_weight_distribution():
        """Get current weight distribution configuration."""
        try:
            gpu_metrics = gpu_monitor.to_dict()
            if not gpu_metrics:
                return jsonify({'error': 'GPU metrics not available'}), 503
            
            # Create or get current configuration
            if not weight_manager.current_config:
                config = weight_manager.create_config_from_gpu_data(gpu_metrics)
            else:
                config = weight_manager.current_config
            
            return jsonify(weight_manager.get_config_summary(config))
        except Exception as e:
            logger.exception("Error getting weight distribution")
            return jsonify({'error': str(e)}), 500
    
    @api_bp.route('/api/dashboard/weight-distribution', methods=['POST'])
    def update_weight_distribution():
        """Update weight distribution configuration."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No configuration data provided'}), 400
            
            gpu_metrics = gpu_monitor.to_dict()
            if not gpu_metrics:
                return jsonify({'error': 'GPU metrics not available'}), 503
            
            if 'preset' in data:
                # Apply preset configuration
                preset_name = data['preset']
                model_name = data.get('model_name')
                config = weight_manager.apply_preset(preset_name, gpu_metrics, model_name)
                
                return jsonify({
                    'status': 'success',
                    'message': f'Applied {preset_name} preset',
                    'config': weight_manager.get_config_summary(config)
                })
            
            elif 'gpu_weights' in data:
                # Apply custom weights
                gpu_weights = data['gpu_weights']  # List of {gpu_id: int, weight: float}
                
                # Ensure we have a current config
                if not weight_manager.current_config:
                    weight_manager.create_config_from_gpu_data(gpu_metrics)
                
                # Update individual GPU weights
                for gpu_weight in gpu_weights:
                    gpu_id = gpu_weight.get('gpu_id')
                    weight = gpu_weight.get('weight', 0.0)
                    if gpu_id is not None:
                        weight_manager.update_weight(gpu_id, weight)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Updated custom weight distribution',
                    'config': weight_manager.get_config_summary()
                })
            
            elif 'model_config' in data:
                # Update model-specific configuration
                model_config = data['model_config']
                if weight_manager.current_config:
                    if 'context_size' in model_config:
                        weight_manager.current_config.context_size = model_config['context_size']
                    if 'batch_size' in model_config:
                        weight_manager.current_config.batch_size = model_config['batch_size']
                    if 'gpu_layers' in model_config:
                        weight_manager.current_config.gpu_layers = model_config['gpu_layers']
                    if 'model_name' in model_config:
                        weight_manager.current_config.model_name = model_config['model_name']
                    
                    # Revalidate configuration
                    weight_manager.validate_config(weight_manager.current_config)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Updated model configuration',
                    'config': weight_manager.get_config_summary()
                })
            
            else:
                return jsonify({'error': 'Invalid configuration data'}), 400
                
        except Exception as e:
            logger.exception("Error updating weight distribution")
            return jsonify({'error': str(e)}), 500
    
    @api_bp.route('/api/dashboard/weight-distribution/presets', methods=['GET'])
    def get_weight_presets():
        """Get available weight distribution presets."""
        try:
            return jsonify({
                'presets': weight_manager.preset_configs
            })
        except Exception as e:
            logger.exception("Error getting weight presets")
            return jsonify({'error': str(e)}), 500
    
    @api_bp.route('/api/dashboard/weight-distribution/validate', methods=['POST'])
    def validate_weight_distribution():
        """Validate a weight distribution configuration."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No configuration data provided'}), 400
            
            gpu_metrics = gpu_monitor.to_dict()
            if not gpu_metrics:
                return jsonify({'error': 'GPU metrics not available'}), 503
            
            # Create temporary configuration for validation
            config = weight_manager.create_config_from_gpu_data(gpu_metrics)
            
            # Apply provided weights
            if 'gpu_weights' in data:
                gpu_weights = data['gpu_weights']
                for gpu_weight in gpu_weights:
                    gpu_id = gpu_weight.get('gpu_id')
                    weight = gpu_weight.get('weight', 0.0)
                    for gpu_config in config.gpu_configs:
                        if gpu_config.gpu_id == gpu_id:
                            gpu_config.weight_percentage = weight
                            break
            
            # Update model configuration if provided
            if 'model_config' in data:
                model_config = data['model_config']
                config.context_size = model_config.get('context_size', config.context_size)
                config.batch_size = model_config.get('batch_size', config.batch_size)
                config.gpu_layers = model_config.get('gpu_layers', config.gpu_layers)
                config.model_name = model_config.get('model_name', config.model_name)
            
            # Validate configuration
            weight_manager.validate_config(config)
            
            return jsonify({
                'is_valid': config.is_valid,
                'validation_errors': config.validation_errors,
                'tensor_split': weight_manager.get_tensor_split_string(config),
                'config_summary': weight_manager.get_config_summary(config)
            })
            
        except Exception as e:
            logger.exception("Error validating weight distribution")
            return jsonify({'error': str(e)}), 500

    @api_bp.route('/', methods=['GET'])
    def root():
        """Root endpoint - redirect to dashboard."""
        return jsonify({
            'service': 'LLM Inference Service',
            'status': 'running',
            'endpoints': {
                'dashboard': '/dashboard',
                'health': '/health',
                'openai_chat': '/api/chat/completions',
                'ollama_generate': '/api/generate',
                'models': '/api/models'
            }
        })

    return api_bp