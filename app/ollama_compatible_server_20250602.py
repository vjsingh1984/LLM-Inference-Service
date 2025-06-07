#!/usr/bin/env python3
import argparse
import flask
from flask import Flask, request, jsonify, Response, stream_with_context
import subprocess
import os
import tempfile
import time
import uuid
import json
import logging
import threading
import re
import queue
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Generator, Tuple
from datetime import datetime

# Configure argument parser
parser = argparse.ArgumentParser(description='Run an Ollama-compatible API server with GPU sharding')
parser.add_argument('--model-dir', type=str, default='/opt/llm/models/ollama/models',
                    help='Path to the Ollama models directory (default: /opt/llm/models/ollama/models)')
parser.add_argument('--llama-cpp-dir', type=str, default='/opt/llm/models/ollama-custom-models/llama.cpp/build',
                    help='Path to the llama.cpp build directory (default: /opt/llm/models/ollama-custom-models/llama.cpp/build)')
parser.add_argument('--log-dir', type=str, default='/opt/llm/inference-service/logs',
                    help='Path to the logs directory (default: /opt/llm/inference-service/logs)')
parser.add_argument('--port', type=int, default=8000,
                    help='Port to run the API server on (default: 8000)')
parser.add_argument('--host', type=str, default='0.0.0.0',
                    help='Host to run the API server on (default: 0.0.0.0)')
parser.add_argument('--debug', action='store_true',
                    help='Run the API server in debug mode')
parser.add_argument('--default-tensor-split', type=str, default='0.25,0.25,0.25,0.25',
                    help='Default tensor split for GPU sharding (default: 0.25,0.25,0.25,0.25)')

args = parser.parse_args()

# Set up paths based on arguments
MODELS_BASE_DIR = args.model_dir
MODELS_DIR = os.path.join(MODELS_BASE_DIR, 'blobs')
MANIFESTS_DIR = os.path.join(MODELS_BASE_DIR, 'manifests/registry.ollama.ai/library')
LLAMA_CPP_DIR = args.llama_cpp_dir
LOG_DIR = args.log_dir

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not args.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'server.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# ================================
# Core Data Structures
# ================================

@dataclass
class InternalRequest:
    """Unified internal request format"""
    request_id: str
    model_name: str
    prompt: str
    context_size: int
    max_tokens: int
    temperature: float
    threads: int
    tensor_split: str
    gpu_layers: int
    stream: bool
    additional_params: Dict[str, Any]

@dataclass
class ModelInfo:
    """Model information structure"""
    id: str
    name: str
    path: str
    context_size: int
    size: int
    parameter_size: str
    quantization: str
    modified_at: str

@dataclass
class RequestStatus:
    """Request status tracking"""
    request_id: str
    status: str  # loading, generating, completed, error
    progress: int
    total: int
    output: str
    start_time: float
    last_update: float
    model: str
    options: Dict[str, Any]
    error: Optional[str] = None
    actual_tokens: int = 0
    context_size: int = 0
    prompt_tokens: int = 0

# ================================
# Model Management
# ================================

class ModelManager:
    """Handles model discovery, context detection, and caching"""
    
    def __init__(self):
        self._model_cache: Dict[str, ModelInfo] = {}
        self._context_cache: Dict[str, int] = {}
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return len(text) // 4  # Rough approximation
    
    def read_manifest_metadata(self, model_id: str) -> Dict[str, Any]:
        """Read metadata from Ollama manifest files"""
        try:
            if ':' in model_id:
                model_name, tag = model_id.split(':', 1)
            else:
                model_name = model_id
                tag = 'latest'
            
            manifest_path = os.path.join(MANIFESTS_DIR, model_name, tag)
            if not os.path.exists(manifest_path):
                return {}
            
            with open(manifest_path, 'r') as f:
                manifest_content = f.read()
            
            try:
                manifest = json.loads(manifest_content)
                
                # Extract config from manifest
                config = {}
                if 'config' in manifest:
                    config = manifest['config']
                
                # Look for model config in layers
                for layer in manifest.get('layers', []):
                    if layer.get('mediaType') == 'application/vnd.ollama.image.model':
                        # This layer contains model configuration
                        # In Ollama, model parameters are often in the config
                        pass
                
                return config
            except json.JSONDecodeError:
                # Fallback to regex parsing
                config = {}
                
                # Look for common parameters in the manifest text
                patterns = {
                    'num_ctx': r'"num_ctx":\s*(\d+)',
                    'context_length': r'"context_length":\s*(\d+)',
                    'max_position_embeddings': r'"max_position_embeddings":\s*(\d+)',
                }
                
                for key, pattern in patterns.items():
                    match = re.search(pattern, manifest_content)
                    if match:
                        config[key] = int(match.group(1))
                
                return config
        except Exception as e:
            logger.debug(f"Error reading manifest metadata for {model_id}: {e}")
            return {}
    
    def detect_context_size(self, model_path: str, model_name: str = "") -> int:
        """Detect actual context size for a model using manifest-based approach"""
        cache_key = f"{model_path}:{model_name}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        # Try manifest metadata first (more reliable than GGUF parsing)
        try:
            manifest_metadata = self.read_manifest_metadata(model_name)
            
            # Look for context length in manifest
            context_keys = ['num_ctx', 'context_length', 'max_position_embeddings', 'n_ctx_train']
            for key in context_keys:
                if key in manifest_metadata:
                    context_size = int(manifest_metadata[key])
                    logger.info(f"Found context size in manifest: {context_size} (key: {key})")
                    self._context_cache[cache_key] = context_size
                    return context_size
        except Exception as e:
            logger.debug(f"Error reading manifest metadata: {e}")
        
        # Model name and filename analysis
        model_lower = model_name.lower()
        filename_lower = Path(model_path).name.lower()
        
        # Look for explicit context size indicators in name
        context_patterns = [
            (r'(\d+)k(?:_?ctx|_?context)?', lambda x: int(x) * 1024),  # "32k", "4k_ctx" -> 32768, 4096
            (r'ctx(\d+)', lambda x: int(x)),                           # "ctx4096" -> 4096
            (r'(\d+)ctx', lambda x: int(x)),                           # "8192ctx" -> 8192
            (r'context(\d+)', lambda x: int(x)),                       # "context2048" -> 2048
            (r'(\d+)k-custom', lambda x: int(x) * 1024),               # "16K-custom" -> 16384
            (r'(\d+)k\b', lambda x: int(x) * 1024),                    # "128K" -> 131072
        ]
        
        for pattern, converter in context_patterns:
            for text in [model_lower, filename_lower]:
                match = re.search(pattern, text)
                if match:
                    try:
                        context_size = converter(match.group(1))
                        logger.info(f"Detected context size from name: {context_size} (pattern: {pattern})")
                        self._context_cache[cache_key] = context_size
                        return context_size
                    except:
                        continue
        
        # Model family-based defaults (improved based on actual model behavior)
        family_defaults = {
            # Phi models - specific handling
            'phi-4': 16384,     # Phi-4 actual context from logs
            'phi-3': 131072,    # Phi-3 has larger context
            'phi4': 16384,      # Alternative naming
            'phi3': 131072,     # Alternative naming
            'phi': 4096,        # Conservative default for older Phi
            
            # Llama family
            'llama3.3': 131072, # Llama 3.3 with 128K context
            'llama3.2': 131072, # Llama 3.2 with extended context
            'llama3': 8192,     # Llama 3 default
            'llama2': 4096,     # Llama 2 default
            'llama': 4096,      # Generic Llama
            'codellama': 16384, # Code Llama typically larger
            
            # Other families
            'mistral': 32768,   # Mistral v0.3+ has 32k
            'mixtral': 32768,   # Mixtral 8x7B/8x22B
            'qwen': 32768,      # Qwen 2.5
            'deepseek': 16384,  # DeepSeek Coder/R1
            'gemma': 8192,      # Gemma 2
            'starcoder': 8192,  # StarCoder
            'granite': 8192,    # Granite
            'devstral': 16384,  # Devstral
            'orca': 4096,       # Orca
            'tinyllama': 2048,  # TinyLlama
            'nomic': 8192,      # Nomic embeddings
        }
        
        # Try to match model family with improved matching
        for family, default_ctx in family_defaults.items():
            # Check both model name and filename
            for text in [model_lower, filename_lower]:
                if family in text:
                    # Special handling for Phi models
                    if family.startswith('phi'):
                        if 'phi-4' in text or 'phi4' in text:
                            context_size = 16384  # Phi-4 specific
                        elif 'phi-3' in text or 'phi3' in text:
                            context_size = 131072  # Phi-3 specific
                        else:
                            context_size = default_ctx
                    else:
                        context_size = default_ctx
                    
                    logger.info(f"Using family default for {family}: {context_size}")
                    self._context_cache[cache_key] = context_size
                    return context_size
        
        # Size-based estimation as last resort (more conservative)
        try:
            file_size = os.path.getsize(model_path)
            size_gb = file_size / (1024**3)
            
            if size_gb > 50:      # > 50GB (70B+ models)
                default_ctx = 4096
            elif size_gb > 25:    # > 25GB (30B+ models)  
                default_ctx = 8192
            elif size_gb > 15:    # > 15GB (13B+ models)
                default_ctx = 8192
            elif size_gb > 7:     # > 7GB (7B+ models)
                default_ctx = 8192
            elif size_gb > 3:     # > 3GB (3B+ models)
                default_ctx = 8192
            else:                 # Small models
                default_ctx = 4096
                
            logger.info(f"Using size-based default context: {default_ctx} (file size: {size_gb:.1f}GB)")
            self._context_cache[cache_key] = default_ctx
            return default_ctx
        except Exception as e:
            logger.debug(f"Error getting file size: {e}")
        
        # Final conservative fallback
        logger.warning(f"Could not detect context size for {model_path}, using conservative default: 4096")
        self._context_cache[cache_key] = 4096
        return 4096
    
    def build_model_mapping(self) -> Dict[str, str]:
        """Build mapping of model names to file paths"""
        mapping = {}
        
        try:
            if not os.path.exists(MANIFESTS_DIR) or not os.path.exists(MODELS_DIR):
                return mapping
                
            for model_dir_name in os.listdir(MANIFESTS_DIR):
                model_dir_path = os.path.join(MANIFESTS_DIR, model_dir_name)
                if not os.path.isdir(model_dir_path):
                    continue
                    
                for tag_file_name in os.listdir(model_dir_path):
                    tag_file_path = os.path.join(model_dir_path, tag_file_name)
                    if not os.path.isfile(tag_file_path):
                        continue
                        
                    model_id = f"{model_dir_name}:{tag_file_name}"
                    
                    try:
                        with open(tag_file_path, 'r') as f:
                            manifest_content = f.read()
                        
                        # Try JSON parsing
                        try:
                            manifest = json.loads(manifest_content)
                            for layer in manifest.get('layers', []):
                                media_type = layer.get('mediaType', '')
                                if 'model' in media_type or layer.get('type') == 'weights':
                                    digest = layer.get('digest', '')
                                    if digest.startswith('sha256:'):
                                        digest = digest[7:]
                                        blob_path = os.path.join(MODELS_DIR, f"sha256-{digest}")
                                        if os.path.exists(blob_path):
                                            mapping[model_id] = blob_path
                                            break
                        except json.JSONDecodeError:
                            # Regex fallback
                            digest_match = re.search(r'"digest":\s*"sha256:([^"]+)".*?"mediaType":\s*"[^"]*model', manifest_content)
                            if digest_match:
                                digest = digest_match.group(1)
                                blob_path = os.path.join(MODELS_DIR, f"sha256-{digest}")
                                if os.path.exists(blob_path):
                                    mapping[model_id] = blob_path
                    except Exception as e:
                        logger.warning(f"Error parsing manifest for {model_id}: {e}")
            
            logger.info(f"Built model mapping with {len(mapping)} entries")
            return mapping
        except Exception as e:
            logger.exception(f"Error building model mapping: {e}")
            return {}
    
    def find_model_path(self, model_name: str) -> Optional[str]:
        """Find the file path for a given model name"""
        mapping = self.build_model_mapping()
        
        # Direct lookup
        if model_name in mapping:
            return mapping[model_name]
        
        # Try with latest tag
        if ':' not in model_name:
            latest_name = f"{model_name}:latest"
            if latest_name in mapping:
                return mapping[latest_name]
        
        # Partial matching
        for mapped_name, path in mapping.items():
            base_name = mapped_name.split(':')[0] if ':' in mapped_name else mapped_name
            if model_name.startswith(base_name) or base_name.startswith(model_name):
                return path
        
        return None
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get comprehensive model information"""
        if model_name in self._model_cache:
            return self._model_cache[model_name]
        
        model_path = self.find_model_path(model_name)
        if not model_path:
            return None
        
        try:
            context_size = self.detect_context_size(model_path, model_name)
            size = os.path.getsize(model_path)
            
            # Determine parameter size and quantization from model name
            model_lower = model_name.lower()
            
            # Parameter size detection
            if "671b" in model_lower:
                parameter_size = "671B"
            elif "235b" in model_lower:
                parameter_size = "235B"
            elif "70b" in model_lower:
                parameter_size = "70B"
            elif "34b" in model_lower or "33b" in model_lower:
                parameter_size = "33B"
            elif "30b" in model_lower:
                parameter_size = "30B"
            elif "27b" in model_lower:
                parameter_size = "27B"
            elif "22b" in model_lower:
                parameter_size = "22B"
            elif "15b" in model_lower:
                parameter_size = "15B"
            elif "13b" in model_lower or "14b" in model_lower:
                parameter_size = "13B"
            elif "8x7b" in model_lower:
                parameter_size = "8x7B"
            elif "8x22b" in model_lower:
                parameter_size = "8x22B"
            elif "7b" in model_lower or "8b" in model_lower:
                parameter_size = "7B"
            elif "3b" in model_lower or "3.8b" in model_lower:
                parameter_size = "3B"
            elif "2b" in model_lower:
                parameter_size = "2B"
            elif "1.5b" in model_lower:
                parameter_size = "1.5B"
            elif "1.1b" in model_lower:
                parameter_size = "1.1B"
            elif "1b" in model_lower:
                parameter_size = "1B"
            else:
                parameter_size = "Unknown"
            
            # Quantization detection
            if "q8_0" in model_lower:
                quantization = "Q8_0"
            elif "q6_k" in model_lower:
                quantization = "Q6_K"
            elif "q5_k_m" in model_lower:
                quantization = "Q5_K_M"
            elif "q4_k_m" in model_lower:
                quantization = "Q4_K_M"
            elif "f16" in model_lower:
                quantization = "F16"
            elif "fp16" in model_lower:
                quantization = "FP16"
            else:
                quantization = "Unknown"
            
            model_info = ModelInfo(
                id=model_name,
                name=model_name.split(':')[0] if ':' in model_name else model_name,
                path=model_path,
                context_size=context_size,
                size=size,
                parameter_size=parameter_size,
                quantization=quantization,
                modified_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(os.path.getmtime(model_path)))
            )
            
            self._model_cache[model_name] = model_info
            return model_info
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return None

# ================================
# Request Adapters
# ================================

class RequestAdapter(ABC):
    """Abstract base class for API format adapters"""
    
    @abstractmethod
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        """Parse API-specific request format to internal format"""
        pass
    
    @abstractmethod
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format response in API-specific format"""
        pass

class OpenAIAdapter(RequestAdapter):
    """Adapter for OpenAI ChatCompletions API format"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        messages = data.get('messages', [])
        model_name = data.get('model')
        
        if not messages or not model_name:
            raise ValueError("Missing required fields: messages and model")
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Get model info for context size
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
        
        # Extract options
        options = data.get('options', {})
        user_context = options.get('context_size', data.get('max_tokens'))
        context_size = min(user_context, model_info.context_size) if user_context else model_info.context_size
        
        return InternalRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt,
            context_size=context_size,
            max_tokens=data.get('max_tokens', 512),
            temperature=data.get('temperature', 0.8),
            threads=options.get('threads', 8),
            tensor_split=options.get('tensor_split', args.default_tensor_split),
            gpu_layers=options.get('gpu_layers', 999),
            stream=data.get('stream', False),
            additional_params={'api_format': 'openai', 'messages': messages}
        )
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert OpenAI messages format to prompt"""
        prompt = ""
        system_message = None
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
                break
        
        if system_message:
            prompt += f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
        
        conversation_started = False
        for msg in messages:
            if msg['role'] == 'system':
                continue
            
            content = msg.get('content', '')
            if msg['role'] == 'user':
                if not conversation_started:
                    if system_message:
                        prompt += f"{content} [/INST]"
                    else:
                        prompt += f"<s>[INST] {content} [/INST]"
                    conversation_started = True
                else:
                    prompt += f"<s>[INST] {content} [/INST]"
            elif msg['role'] == 'assistant':
                prompt += f" {content} </s>"
        
        if not prompt.endswith("[/INST]") and not prompt.endswith("</s>"):
            prompt += "[/INST]"
        
        return prompt
    
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format response in OpenAI format"""
        return {
            'id': request.request_id,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': request.model_name,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': output
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': self.model_manager.estimate_tokens(request.prompt),
                'completion_tokens': self.model_manager.estimate_tokens(output),
                'total_tokens': self.model_manager.estimate_tokens(request.prompt + output)
            }
        }

class OllamaChatAdapter(RequestAdapter):
    """Adapter for Ollama Chat API format"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        messages = data.get('messages', [])
        model_name = data.get('model')
        
        if not messages or not model_name:
            raise ValueError("Missing required fields: messages and model")
        
        # Convert messages to prompt (same as OpenAI)
        prompt = self._messages_to_prompt(messages)
        
        # Get model info
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
        
        # Extract Ollama options
        options = data.get('options', {})
        user_context = data.get('context', options.get('num_ctx'))
        context_size = min(user_context, model_info.context_size) if user_context else model_info.context_size
        
        return InternalRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt,
            context_size=context_size,
            max_tokens=data.get('num_predict', options.get('num_predict', 512)),
            temperature=data.get('temperature', options.get('temperature', 0.8)),
            threads=data.get('num_thread', options.get('num_thread', 24)),
            tensor_split=data.get('tensor_split', options.get('tensor_split', args.default_tensor_split)),
            gpu_layers=data.get('n_gpu_layers', options.get('numa', 999)),
            stream=data.get('stream', False),
            additional_params={'api_format': 'ollama_chat', 'messages': messages}
        )
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert messages to prompt (reuse OpenAI logic)"""
        return OpenAIAdapter(self.model_manager)._messages_to_prompt(messages)
    
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format response in Ollama chat format"""
        return {
            'model': request.model_name,
            'created_at': time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            'message': {
                'role': 'assistant',
                'content': output
            },
            'done': True
        }

class OllamaGenerateAdapter(RequestAdapter):
    """Adapter for Ollama Generate API format"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        prompt = data.get('prompt', '')
        model_name = data.get('model')
        
        if not prompt or not model_name:
            raise ValueError("Missing required fields: prompt and model")
        
        # Get model info
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
        
        # Extract options
        user_context = data.get('context', 8192)
        context_size = min(user_context, model_info.context_size) if user_context else model_info.context_size
        
        return InternalRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt,
            context_size=context_size,
            max_tokens=data.get('num_predict', 512),
            temperature=data.get('temperature', 0.8),
            threads=data.get('num_thread', 24),
            tensor_split=data.get('tensor_split', args.default_tensor_split),
            gpu_layers=data.get('n_gpu_layers', 999),
            stream=data.get('stream', False),
            additional_params={'api_format': 'ollama_generate'}
        )
    
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format response in Ollama generate format"""
        return {
            'model': request.model_name,
            'created_at': time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            'response': output,
            'done': True
        }

class ClaudeAdapter(RequestAdapter):
    """Adapter for Claude (Anthropic) API format"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        messages = data.get('messages', [])
        model_name = data.get('model')
        
        if not messages or not model_name:
            raise ValueError("Missing required fields: messages and model")
        
        # Convert Claude messages to prompt
        prompt = self._claude_messages_to_prompt(messages)
        
        # Get model info
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
        
        # Extract Claude-specific options
        user_context = data.get('max_tokens_to_sample', data.get('max_tokens'))
        context_size = min(user_context, model_info.context_size) if user_context else model_info.context_size
        
        return InternalRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt,
            context_size=context_size,
            max_tokens=data.get('max_tokens', data.get('max_tokens_to_sample', 512)),
            temperature=data.get('temperature', 0.8),
            threads=data.get('threads', 8),
            tensor_split=data.get('tensor_split', args.default_tensor_split),
            gpu_layers=data.get('gpu_layers', 999),
            stream=data.get('stream', False),
            additional_params={
                'api_format': 'claude',
                'messages': messages,
                'system': data.get('system', ''),
                'stop_sequences': data.get('stop_sequences', [])
            }
        )
    
    def _claude_messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert Claude messages format to prompt"""
        prompt = ""
        
        # Claude format: system message separate, then alternating user/assistant
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'user':
                prompt += f"\n\nHuman: {content}"
            elif role == 'assistant':
                prompt += f"\n\nAssistant: {content}"
            elif role == 'system':
                # System messages in Claude are typically handled separately
                prompt = f"System: {content}\n\n" + prompt
        
        # Ensure prompt ends ready for assistant response
        if not prompt.endswith("Assistant:"):
            prompt += "\n\nAssistant:"
        
        return prompt
    
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format response in Claude format"""
        return {
            'id': request.request_id,
            'type': 'message',
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': output
                }
            ],
            'model': request.model_name,
            'stop_reason': 'end_turn',
            'stop_sequence': None,
            'usage': {
                'input_tokens': self.model_manager.estimate_tokens(request.prompt),
                'output_tokens': self.model_manager.estimate_tokens(output)
            }
        }

class HuggingFaceAdapter(RequestAdapter):
    """Adapter for HuggingFace Text Generation Inference (TGI) API format"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        # HuggingFace TGI format
        inputs = data.get('inputs', '')
        model_name = data.get('model', 'default')  # TGI often doesn't specify model in request
        
        if not inputs:
            raise ValueError("Missing required field: inputs")
        
        # Get model info
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info:
            # For HF, try to use any available model if not specified
            mapping = self.model_manager.build_model_mapping()
            if mapping:
                model_name = list(mapping.keys())[0]
                model_info = self.model_manager.get_model_info(model_name)
            if not model_info:
                raise ValueError(f"No models available")
        
        # Extract HuggingFace parameters
        parameters = data.get('parameters', {})
        options = data.get('options', {})
        
        user_context = parameters.get('max_length', options.get('max_length'))
        context_size = min(user_context, model_info.context_size) if user_context else model_info.context_size
        
        return InternalRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=inputs,
            context_size=context_size,
            max_tokens=parameters.get('max_new_tokens', parameters.get('max_tokens', 512)),
            temperature=parameters.get('temperature', 0.8),
            threads=options.get('threads', 8),
            tensor_split=options.get('tensor_split', args.default_tensor_split),
            gpu_layers=options.get('gpu_layers', 999),
            stream=data.get('stream', False),
            additional_params={
                'api_format': 'huggingface',
                'do_sample': parameters.get('do_sample', True),
                'top_p': parameters.get('top_p', 0.9),
                'top_k': parameters.get('top_k', 50),
                'repetition_penalty': parameters.get('repetition_penalty', 1.0),
                'stop': parameters.get('stop', [])
            }
        )
    
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format response in HuggingFace TGI format"""
        return {
            'generated_text': output,
            'details': {
                'finish_reason': 'length',
                'generated_tokens': self.model_manager.estimate_tokens(output),
                'seed': None,
                'prefill': [],
                'tokens': []
            }
        }

class VLLMAdapter(RequestAdapter):
    """Adapter for vLLM API format"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        # vLLM can handle both completions and chat formats
        if 'messages' in data:
            # Chat format
            messages = data.get('messages', [])
            prompt = self._messages_to_prompt(messages)
        else:
            # Completion format
            prompt = data.get('prompt', '')
        
        model_name = data.get('model')
        
        if not prompt or not model_name:
            raise ValueError("Missing required fields")
        
        # Get model info
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
        
        # Extract vLLM parameters
        user_context = data.get('max_tokens', data.get('max_length'))
        context_size = min(user_context, model_info.context_size) if user_context else model_info.context_size
        
        return InternalRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt,
            context_size=context_size,
            max_tokens=data.get('max_tokens', 512),
            temperature=data.get('temperature', 0.8),
            threads=data.get('threads', 8),
            tensor_split=data.get('tensor_split', args.default_tensor_split),
            gpu_layers=data.get('gpu_layers', 999),
            stream=data.get('stream', False),
            additional_params={
                'api_format': 'vllm',
                'top_p': data.get('top_p', 0.9),
                'top_k': data.get('top_k', -1),
                'frequency_penalty': data.get('frequency_penalty', 0.0),
                'presence_penalty': data.get('presence_penalty', 0.0),
                'stop': data.get('stop', []),
                'messages': data.get('messages')
            }
        )
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert messages to prompt (similar to OpenAI)"""
        return OpenAIAdapter(self.model_manager)._messages_to_prompt(messages)
    
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format response in vLLM format"""
        if request.additional_params.get('messages'):
            # Chat completion format
            return {
                'id': request.request_id,
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': request.model_name,
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': output
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': self.model_manager.estimate_tokens(request.prompt),
                    'completion_tokens': self.model_manager.estimate_tokens(output),
                    'total_tokens': self.model_manager.estimate_tokens(request.prompt + output)
                }
            }
        else:
            # Completion format
            return {
                'id': request.request_id,
                'object': 'text_completion',
                'created': int(time.time()),
                'model': request.model_name,
                'choices': [{
                    'text': output,
                    'index': 0,
                    'logprobs': None,
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': self.model_manager.estimate_tokens(request.prompt),
                    'completion_tokens': self.model_manager.estimate_tokens(output),
                    'total_tokens': self.model_manager.estimate_tokens(request.prompt + output)
                }
            }

# ================================
# Request Tracker
# ================================

class RequestTracker:
    """Manages active requests for dashboard and progress tracking"""
    
    def __init__(self):
        self.active_requests: Dict[str, RequestStatus] = {}
        self._lock = threading.Lock()
    
    def add_request(self, request: InternalRequest) -> None:
        """Add a new request to tracking"""
        with self._lock:
            self.active_requests[request.request_id] = RequestStatus(
                request_id=request.request_id,
                status='loading',
                progress=0,
                total=request.max_tokens,
                output='',
                start_time=time.time(),
                last_update=time.time(),
                model=request.model_name,
                options=request.__dict__.copy(),
                context_size=request.context_size
            )
    
    def update_request(self, request_id: str, **kwargs) -> None:
        """Update request status"""
        with self._lock:
            if request_id in self.active_requests:
                for key, value in kwargs.items():
                    setattr(self.active_requests[request_id], key, value)
                self.active_requests[request_id].last_update = time.time()
    
    def get_request(self, request_id: str) -> Optional[RequestStatus]:
        """Get request status"""
        with self._lock:
            return self.active_requests.get(request_id)
    
    def remove_request(self, request_id: str) -> None:
        """Remove request from tracking"""
        with self._lock:
            self.active_requests.pop(request_id, None)
    
    def get_all_requests(self) -> Dict[str, RequestStatus]:
        """Get all active requests (copy)"""
        with self._lock:
            return self.active_requests.copy()
    
    def remove_completed(self) -> int:
        """Remove all completed requests"""
        with self._lock:
            to_remove = [
                req_id for req_id, req in self.active_requests.items()
                if req.status in ['completed', 'error']
            ]
            for req_id in to_remove:
                del self.active_requests[req_id]
            return len(to_remove)

# ================================
# LLAMA Executor
# ================================

class LLAMAExecutor:
    """Unified LLAMA CLI execution with streaming support"""
    
    def __init__(self, llama_cli_path: str, model_manager: ModelManager, request_tracker: RequestTracker):
        self.llama_cli_path = llama_cli_path
        self.model_manager = model_manager
        self.request_tracker = request_tracker
    
    def execute_request(self, request: InternalRequest) -> Generator[str, None, None]:
        """Execute LLAMA CLI request with streaming output"""
        model_info = self.model_manager.get_model_info(request.model_name)
        if not model_info:
            raise ValueError(f"Model {request.model_name} not found")
        
        # Validate context size against detected model context
        detected_context = model_info.context_size
        if request.context_size > detected_context:
            logger.warning(f"Requested context {request.context_size} exceeds model limit {detected_context}, reducing to model limit")
            request.context_size = detected_context
        
        prompt_tokens = self.model_manager.estimate_tokens(request.prompt)
        available_tokens = request.context_size - prompt_tokens - 100
        
        if available_tokens <= 0:
            error_msg = f"Prompt too long ({prompt_tokens} tokens) for context size ({request.context_size}). Model supports up to {detected_context} tokens."
            self.request_tracker.update_request(request.request_id, status='error', error=error_msg)
            yield f"Error: {error_msg}"
            return
        
        # Adjust max_tokens if needed
        if request.max_tokens > available_tokens:
            original_max = request.max_tokens
            request.max_tokens = max(50, available_tokens)
            logger.warning(f"Reduced max_tokens from {original_max} to {request.max_tokens} to fit in context")
        
        # Track request start
        self.request_tracker.add_request(request)
        
        # Create temporary prompt file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
            temp.write(request.prompt)
            prompt_file = temp.name
        
        try:
            # Build command with validated parameters
            cmd = [
                self.llama_cli_path,
                "-m", model_info.path,
                "-f", prompt_file,
                "-c", str(request.context_size),
                "-n", str(request.max_tokens),
                "-t", str(request.threads),
                "--no-display-prompt"
            ]
            
            if request.tensor_split:
                cmd.extend(["-ts", request.tensor_split])
            if request.gpu_layers:
                cmd.extend(["-ngl", str(request.gpu_layers)])
            if request.temperature != 0.8:
                cmd.extend(["--temp", str(request.temperature)])
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # Start process
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1
            )
            
            self.request_tracker.update_request(request.request_id, status='generating')
            
            # Monitor stderr in background
            def monitor_stderr():
                for line in process.stderr:
                    logger.info(f"STDERR: {line.strip()}")
                    
                    # Check for specific error conditions
                    if "out of memory" in line.lower() or "cudamalloc failed" in line.lower():
                        error_msg = f"GPU out of memory. Try reducing context size (current: {request.context_size}) or max_tokens (current: {request.max_tokens})"
                        self.request_tracker.update_request(
                            request.request_id, status='error', error=error_msg
                        )
                    elif "prompt is too long" in line.lower():
                        self.request_tracker.update_request(
                            request.request_id, status='error', 
                            error=f"Context exceeded: {line.strip()}"
                        )
                    elif "failed to load model" in line.lower():
                        self.request_tracker.update_request(
                            request.request_id, status='error',
                            error=f"Failed to load model: {line.strip()}"
                        )
                    elif "unable to load model" in line.lower():
                        self.request_tracker.update_request(
                            request.request_id, status='error',
                            error=f"Unable to load model: {line.strip()}"
                        )
            
            threading.Thread(target=monitor_stderr, daemon=True).start()
            
            # Stream output
            full_output = ""
            tokens_generated = 0
            
            for line in process.stdout:
                current_status = self.request_tracker.get_request(request.request_id)
                if current_status and current_status.status == 'error':
                    break
                
                full_output += line
                print(line)
                tokens_generated += 1
                
                self.request_tracker.update_request(
                    request.request_id,
                    progress=tokens_generated,
                    output=full_output,
                    actual_tokens=len(full_output.split())
                )
                
                yield line
            
            # Wait for completion
            process.wait()
            
            # Check final status
            final_status = self.request_tracker.get_request(request.request_id)
            if final_status and final_status.status != 'error':
                self.request_tracker.update_request(
                    request.request_id, status='completed', 
                    progress=request.max_tokens
                )
            
        except Exception as e:
            logger.exception(f"Error executing LLAMA: {e}")
            self.request_tracker.update_request(
                request.request_id, status='error', error=str(e)
            )
            yield f"Error: {str(e)}"
        finally:
            if os.path.exists(prompt_file):
                os.unlink(prompt_file)

# ================================
# Global Instances
# ================================

model_manager = ModelManager()
request_tracker = RequestTracker()

# Find LLAMA CLI executable
def find_llama_executable():
    """Find the llama.cpp executable"""
    for name in ['llama-cli', 'main']:
        exec_path = os.path.join(LLAMA_CPP_DIR, 'bin', name)
        if os.path.isfile(exec_path) and os.access(exec_path, os.X_OK):
            return exec_path
    
    for root, dirs, files in os.walk(LLAMA_CPP_DIR):
        for file in files:
            if file in ['llama-cli', 'main']:
                exec_path = os.path.join(root, file)
                if os.access(exec_path, os.X_OK):
                    return exec_path
    
    raise RuntimeError(f"Could not find llama.cpp executable in {LLAMA_CPP_DIR}")

try:
    LLAMA_CLI = find_llama_executable()
    llama_executor = LLAMAExecutor(LLAMA_CLI, model_manager, request_tracker)
    logger.info(f"Using llama.cpp executable: {LLAMA_CLI}")
except Exception as e:
    logger.error(f"Failed to find llama.cpp executable: {e}")
    sys.exit(1)

# ================================
# API Endpoints
# ================================

def create_streaming_response(adapter: RequestAdapter, request: InternalRequest):
    """Create streaming response for any adapter"""
    def generate_stream():
        api_format = request.additional_params.get('api_format')
        
        if api_format == 'openai':
            # OpenAI streaming format
            yield f"data: {json.dumps({'id': request.request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            
            for chunk in llama_executor.execute_request(request):
                if chunk.startswith('Error:'):
                    yield f"data: {json.dumps({'id': request.request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model_name, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': 'stop'}]})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'id': request.request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model_name, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
            else:
                yield f"data: {json.dumps({'id': request.request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            
            yield "data: [DONE]\n\n"
        
        elif api_format == 'claude':
            # Claude streaming format
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': request.request_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': request.model_name, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
            
            for chunk in llama_executor.execute_request(request):
                if chunk.startswith('Error:'):
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': chunk}})}\n\n"
                    break
                else:
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': chunk}})}\n\n"
            else:
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        
        elif api_format == 'vllm':
            # vLLM streaming format (OpenAI-compatible)
            if request.additional_params.get('messages'):
                # Chat completion streaming
                yield f"data: {json.dumps({'id': request.request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
                
                for chunk in llama_executor.execute_request(request):
                    if chunk.startswith('Error:'):
                        yield f"data: {json.dumps({'id': request.request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model_name, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': 'stop'}]})}\n\n"
                        break
                    else:
                        yield f"data: {json.dumps({'id': request.request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model_name, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                else:
                    yield f"data: {json.dumps({'id': request.request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            else:
                # Text completion streaming
                for chunk in llama_executor.execute_request(request):
                    if chunk.startswith('Error:'):
                        yield f"data: {json.dumps({'id': request.request_id, 'object': 'text_completion', 'created': int(time.time()), 'model': request.model_name, 'choices': [{'text': chunk, 'index': 0, 'finish_reason': 'stop'}]})}\n\n"
                        break
                    else:
                        yield f"data: {json.dumps({'id': request.request_id, 'object': 'text_completion', 'created': int(time.time()), 'model': request.model_name, 'choices': [{'text': chunk, 'index': 0, 'finish_reason': None}]})}\n\n"
                else:
                    yield f"data: {json.dumps({'id': request.request_id, 'object': 'text_completion', 'created': int(time.time()), 'model': request.model_name, 'choices': [{'text': '', 'index': 0, 'finish_reason': 'stop'}]})}\n\n"
            
            yield "data: [DONE]\n\n"
        
        elif api_format == 'huggingface':
            # HuggingFace streaming format
            full_text = ""
            for chunk in llama_executor.execute_request(request):
                if chunk.startswith('Error:'):
                    yield json.dumps({'error': chunk}) + '\n'
                    break
                else:
                    full_text += chunk
                    yield json.dumps({'token': {'text': chunk, 'logprob': 0.0, 'id': 0}, 'generated_text': None, 'details': None}) + '\n'
            else:
                # Final response
                response = adapter.format_response(full_text, request)
                yield json.dumps(response) + '\n'
        
        else:
            # Ollama streaming format
            for chunk in llama_executor.execute_request(request):
                if chunk.startswith('Error:'):
                    response = adapter.format_response(chunk, request)
                    response['error'] = chunk
                    yield json.dumps(response) + '\n'
                    break
                else:
                    if api_format == 'ollama_chat':
                        yield json.dumps({'model': request.model_name, 'created_at': time.strftime("%Y-%m-%dT%H:%M:%S%z"), 'message': {'role': 'assistant', 'content': chunk}, 'done': False}) + '\n'
                    else:  # ollama_generate
                        yield json.dumps({'model': request.model_name, 'created_at': time.strftime("%Y-%m-%dT%H:%M:%S%z"), 'response': chunk, 'done': False}) + '\n'
            else:
                final_response = adapter.format_response('', request)
                yield json.dumps(final_response) + '\n'
        
        # Cleanup after delay
        def cleanup():
            time.sleep(300)
            request_tracker.remove_request(request.request_id)
        threading.Thread(target=cleanup, daemon=True).start()
    
    # Determine content type based on API format
    if request.additional_params.get('api_format') in ['openai', 'vllm']:
        content_type = 'text/event-stream'
    elif request.additional_params.get('api_format') == 'claude':
        content_type = 'text/event-stream'
    else:
        content_type = 'application/json'
    
    response = Response(generate_stream(), mimetype=content_type)
    response.headers['X-Request-ID'] = request.request_id
    return response

@app.route('/api/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint"""
    try:
        adapter = OpenAIAdapter(model_manager)
        request_obj = adapter.parse_request(request.json)
        
        if request_obj.stream:
            return create_streaming_response(adapter, request_obj)
        else:
            output = ""
            for chunk in llama_executor.execute_request(request_obj):
                if chunk.startswith('Error:'):
                    return jsonify({'error': chunk}), 400
                output += chunk
            
            response_data = adapter.format_response(output, request_obj)
            resp = jsonify(response_data)
            resp.headers['X-Request-ID'] = request_obj.request_id
            return resp
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Error in chat_completions")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Ollama-compatible chat endpoint"""
    try:
        adapter = OllamaChatAdapter(model_manager)
        request_obj = adapter.parse_request(request.json)
        
        if request_obj.stream:
            return create_streaming_response(adapter, request_obj)
        else:
            output = ""
            for chunk in llama_executor.execute_request(request_obj):
                if chunk.startswith('Error:'):
                    return jsonify({'error': chunk}), 400
                output += chunk
            
            return jsonify(adapter.format_response(output, request_obj))
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Error in chat")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    """Ollama-compatible generate endpoint"""
    try:
        adapter = OllamaGenerateAdapter(model_manager)
        request_obj = adapter.parse_request(request.json)
        
        if request_obj.stream:
            return create_streaming_response(adapter, request_obj)
        else:
            output = ""
            for chunk in llama_executor.execute_request(request_obj):
                if chunk.startswith('Error:'):
                    return jsonify({'error': chunk}), 400
                output += chunk
            
            return jsonify(adapter.format_response(output, request_obj))
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Error in generate")
        return jsonify({'error': str(e)}), 500

@app.route('/v1/messages', methods=['POST'])
def claude_messages():
    """Claude (Anthropic) API compatible endpoint"""
    try:
        adapter = ClaudeAdapter(model_manager)
        request_obj = adapter.parse_request(request.json)
        
        if request_obj.stream:
            return create_streaming_response(adapter, request_obj)
        else:
            output = ""
            for chunk in llama_executor.execute_request(request_obj):
                if chunk.startswith('Error:'):
                    return jsonify({'error': {'type': 'api_error', 'message': chunk}}), 400
                output += chunk
            
            return jsonify(adapter.format_response(output, request_obj))
    
    except ValueError as e:
        return jsonify({'error': {'type': 'invalid_request_error', 'message': str(e)}}), 400
    except Exception as e:
        logger.exception("Error in claude_messages")
        return jsonify({'error': {'type': 'api_error', 'message': str(e)}}), 500

@app.route('/generate', methods=['POST'])
@app.route('/v1/completions', methods=['POST'])
def huggingface_generate():
    """HuggingFace Text Generation Inference (TGI) compatible endpoint"""
    try:
        adapter = HuggingFaceAdapter(model_manager)
        request_obj = adapter.parse_request(request.json)
        
        if request_obj.stream:
            return create_streaming_response(adapter, request_obj)
        else:
            output = ""
            for chunk in llama_executor.execute_request(request_obj):
                if chunk.startswith('Error:'):
                    return jsonify({'error': chunk}), 400
                output += chunk
            
            # HuggingFace returns array of results
            response_data = adapter.format_response(output, request_obj)
            return jsonify([response_data])  # HF expects array
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Error in huggingface_generate")
        return jsonify({'error': str(e)}), 500

@app.route('/v1/chat/completions', methods=['POST'])
def vllm_chat_completions():
    """vLLM chat completions endpoint (OpenAI-compatible but with vLLM extensions)"""
    try:
        adapter = VLLMAdapter(model_manager)
        request_obj = adapter.parse_request(request.json)
        
        if request_obj.stream:
            return create_streaming_response(adapter, request_obj)
        else:
            output = ""
            for chunk in llama_executor.execute_request(request_obj):
                if chunk.startswith('Error:'):
                    return jsonify({'error': chunk}), 400
                output += chunk
            
            return jsonify(adapter.format_response(output, request_obj))
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Error in vllm_chat_completions")
        return jsonify({'error': str(e)}), 500

@app.route('/v1/completions', methods=['POST'])
def vllm_completions():
    """vLLM text completions endpoint"""
    try:
        # Force completion mode by removing messages if present
        data = request.json.copy()
        if 'messages' in data:
            del data['messages']
        
        adapter = VLLMAdapter(model_manager)
        request_obj = adapter.parse_request(data)
        
        if request_obj.stream:
            return create_streaming_response(adapter, request_obj)
        else:
            output = ""
            for chunk in llama_executor.execute_request(request_obj):
                if chunk.startswith('Error:'):
                    return jsonify({'error': chunk}), 400
                output += chunk
            
            return jsonify(adapter.format_response(output, request_obj))
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Error in vllm_completions")
        return jsonify({'error': str(e)}), 500

# Additional utility endpoints for compatibility

@app.route('/v1/models', methods=['GET'])
def vllm_list_models():
    """vLLM-compatible models endpoint"""
    try:
        mapping = model_manager.build_model_mapping()
        models = []
        
        for model_id, model_path in mapping.items():
            if not os.path.exists(model_path):
                continue
            
            model_info = model_manager.get_model_info(model_id)
            if model_info:
                models.append({
                    "id": model_info.id,
                    "object": "model",
                    "created": int(os.path.getmtime(model_info.path)),
                    "owned_by": "local",
                    "permission": [],
                    "root": model_info.id,
                    "parent": None
                })
        
        return jsonify({"object": "list", "data": models})
    except Exception as e:
        logger.exception("Error listing vLLM models")
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def huggingface_info():
    """HuggingFace TGI info endpoint"""
    try:
        mapping = model_manager.build_model_mapping()
        available_models = list(mapping.keys())
        
        return jsonify({
            "model_id": available_models[0] if available_models else "unknown",
            "model_sha": "unknown",
            "model_dtype": "float16",
            "model_device_type": "cuda",
            "model_pipeline_tag": "text-generation",
            "max_concurrent_requests": 128,
            "max_best_of": 2,
            "max_stop_sequences": 4,
            "max_input_length": 32768,
            "max_total_tokens": 32768,
            "waiting_served_ratio": 1.2,
            "max_batch_total_tokens": 32768,
            "max_waiting_tokens": 20,
            "validation_workers": 2,
            "version": "1.0.0"
        })
    except Exception as e:
        logger.exception("Error getting HF info")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tags', methods=['GET'])
@app.route('/api/models', methods=['GET'])
def list_models():
    """List all available models with actual context information"""
    try:
        mapping = model_manager.build_model_mapping()
        models = []
        
        for model_id, model_path in mapping.items():
            if not os.path.exists(model_path):
                continue
            
            model_info = model_manager.get_model_info(model_id)
            if model_info:
                models.append({
                    "id": model_info.id,
                    "name": model_info.name,
                    "model": model_info.id,
                    "tag": model_info.id.split(':')[1] if ':' in model_info.id else 'latest',
                    "path": model_info.path,
                    "size": model_info.size,
                    "context_size": model_info.context_size,
                    "modelFormat": "gguf",
                    "modified_at": model_info.modified_at,
                    "parameter_size": model_info.parameter_size,
                    "quantization_level": model_info.quantization,
                    "digest": os.path.basename(model_info.path).replace("sha256-", ""),
                    "details": {
                        "format": "gguf",
                        "family": model_info.name,
                        "parameter_size": model_info.parameter_size,
                        "quantization_level": model_info.quantization,
                        "context_length": model_info.context_size
                    }
                })
        
        return jsonify({'models': models})
    except Exception as e:
        logger.exception("Error listing models")
        return jsonify({'error': str(e)}), 500

# Progress and management endpoints
@app.route('/api/progress/<request_id>', methods=['GET'])
def check_progress(request_id):
    """Check progress of a request"""
    req_status = request_tracker.get_request(request_id)
    if not req_status:
        return jsonify({'error': 'Request not found'}), 404
    
    response_data = req_status.__dict__.copy()
    response_data['elapsed_time'] = time.time() - req_status.start_time
    
    if req_status.progress > 0 and req_status.progress < req_status.total:
        tokens_per_second = req_status.progress / response_data['elapsed_time']
        remaining_tokens = req_status.total - req_status.progress
        response_data['estimated_time_remaining'] = remaining_tokens / tokens_per_second
    else:
        response_data['estimated_time_remaining'] = None
    
    return jsonify(response_data)

@app.route('/api/dismiss/<request_id>', methods=['POST'])
def dismiss_request(request_id):
    """Dismiss a specific request"""
    request_tracker.remove_request(request_id)
    return jsonify({'success': True})

@app.route('/api/dismiss/completed', methods=['POST'])
def dismiss_completed():
    """Dismiss all completed requests"""
    count = request_tracker.remove_completed()
    return jsonify({'success': True, 'dismissed': count})

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Dashboard with manual refresh only"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Universal LLM API Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            :root {
                --primary: #4a6baf;
                --primary-light: #6987cc;
                --secondary: #38b2ac;
                --dark: #2d3748;
                --light: #f7fafc;
                --danger: #e53e3e;
                --success: #38a169;
                --warning: #d69e2e;
                --gray: #718096;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f0f4f8;
                color: var(--dark);
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            
            header {
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
                padding: 1rem 2rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                position: sticky;
                top: 0;
                z-index: 10;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            header h1 {
                margin: 0;
                font-size: 1.5rem;
            }
            
            .header-subtitle {
                font-size: 0.875rem;
                opacity: 0.9;
                margin-top: 4px;
            }
            
            .header-actions {
                display: flex;
                gap: 10px;
                align-items: center;
            }
            
            .btn {
                background-color: rgba(255,255,255,0.2);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.875rem;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            
            .btn:hover {
                background-color: rgba(255,255,255,0.3);
                transform: translateY(-1px);
            }
            
            .btn-primary {
                background-color: var(--secondary);
            }
            
            .btn-primary:hover {
                background-color: #319795;
            }
            
            .stats-bar {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            
            .stat-card {
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 4px solid var(--primary);
                transition: transform 0.2s;
            }
            
            .stat-card:hover {
                transform: translateY(-2px);
            }
            
            .stat-card h3 {
                margin: 0 0 8px 0;
                color: var(--gray);
                font-size: 0.875rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .stat-card p {
                margin: 0;
                font-size: 2rem;
                font-weight: 700;
                color: var(--dark);
            }
            
            .api-indicator {
                display: inline-block;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 0.7rem;
                font-weight: 600;
                text-transform: uppercase;
                margin-left: 8px;
            }
            
            .api-openai { background: #10b981; color: white; }
            .api-ollama { background: #f59e0b; color: white; }
            .api-claude { background: #8b5cf6; color: white; }
            .api-huggingface { background: #ef4444; color: white; }
            .api-vllm { background: #06b6d4; color: white; }
            
            .request-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            
            .request-card {
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                overflow: hidden;
                transition: all 0.3s ease;
                border: 1px solid #e2e8f0;
            }
            
            .request-card:hover {
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                transform: translateY(-4px);
            }
            
            .card-header {
                padding: 16px 20px;
                background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary) 100%);
                color: white;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .card-title {
                font-weight: 600;
                font-size: 0.95rem;
            }
            
            .model-badge {
                background: rgba(255,255,255,0.2);
                padding: 4px 10px;
                border-radius: 15px;
                font-size: 0.75rem;
                font-weight: 600;
            }
            
            .card-body {
                padding: 20px;
            }
            
            .params-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
                margin-bottom: 16px;
            }
            
            .param-item {
                background: #f8fafc;
                padding: 10px 14px;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }
            
            .param-label {
                color: var(--gray);
                font-size: 0.75rem;
                margin-bottom: 4px;
                font-weight: 500;
            }
            
            .param-value {
                font-weight: 600;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                font-size: 0.85rem;
            }
            
            .progress-container {
                margin: 16px 0;
            }
            
            .progress-bar {
                height: 8px;
                background: #e2e8f0;
                border-radius: 4px;
                overflow: hidden;
                margin: 8px 0;
            }
            
            .progress-fill {
                height: 100%;
                border-radius: 4px;
                transition: width 0.5s ease;
            }
            
            .progress-label {
                display: flex;
                justify-content: space-between;
                font-size: 0.875rem;
                color: var(--gray);
                margin-bottom: 4px;
            }
            
            .status-indicator {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 4px 10px;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
            }
            
            .status-loading {
                background: #dbeafe;
                color: #1e40af;
            }
            
            .status-generating {
                background: #fef3c7;
                color: #92400e;
            }
            
            .status-completed {
                background: #d1fae5;
                color: #065f46;
            }
            
            .status-error {
                background: #fee2e2;
                color: #991b1b;
            }
            
            .empty-state {
                grid-column: 1 / -1;
                text-align: center;
                padding: 60px 20px;
                color: var(--gray);
            }
            
            .empty-state svg {
                width: 80px;
                height: 80px;
                margin-bottom: 20px;
                color: #cbd5e0;
            }
            
            .empty-state h3 {
                font-size: 1.25rem;
                margin-bottom: 8px;
            }
            
            .loading {
                opacity: 0.6;
                pointer-events: none;
            }
            
            @media (max-width: 768px) {
                .request-grid {
                    grid-template-columns: 1fr;
                }
                
                .params-grid {
                    grid-template-columns: 1fr;
                }
                
                .stats-bar {
                    grid-template-columns: repeat(2, 1fr);
                }
            }
        </style>
        <script>
            function refreshDashboard() {
                const container = document.getElementById('requests');
                container.classList.add('loading');
                
                fetch('/dashboard/data')
                    .then(response => response.json())
                    .then(data => {
                        updateDashboardData(data);
                        container.classList.remove('loading');
                    })
                    .catch(error => {
                        console.error('Error refreshing dashboard:', error);
                        container.classList.remove('loading');
                    });
            }
            
            function updateDashboardData(data) {
                const container = document.getElementById('requests');
                
                // Update stats
                const stats = { active: 0, completed: 0, error: 0 };
                
                Object.values(data).forEach(req => {
                    if (req.status === 'loading' || req.status === 'generating') {
                        stats.active++;
                    } else if (req.status === 'completed') {
                        stats.completed++;
                    } else if (req.status === 'error') {
                        stats.error++;
                    }
                });
                
                document.getElementById('active-count').textContent = stats.active;
                document.getElementById('completed-count').textContent = stats.completed;
                document.getElementById('error-count').textContent = stats.error;
                document.getElementById('total-count').textContent = Object.keys(data).length;
                
                // Show empty state if no requests
                if (Object.keys(data).length === 0) {
                    container.innerHTML = `
                        <div class="empty-state">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            <h3>No active requests</h3>
                            <p>API requests will appear here. Click refresh to update manually.</p>
                        </div>
                    `;
                    return;
                }
                
                // Create request cards
                container.innerHTML = '';
                
                for (const [id, req] of Object.entries(data)) {
                    const percent = Math.min(100, Math.floor((req.progress / req.total) * 100)) || 0;
                    const statusClass = getStatusClass(req.status);
                    const progressColor = getProgressColor(req.status, percent);
                    const apiFormat = req.options?.additional_params?.api_format || 'unknown';
                    const apiClass = getApiClass(apiFormat);
                    
                    const div = document.createElement('div');
                    div.className = 'request-card';
                    div.innerHTML = `
                        <div class="card-header">
                            <div>
                                <div class="card-title">Request ${id.substring(0, 8)}...</div>
                                <span class="api-indicator ${apiClass}">${formatApiName(apiFormat)}</span>
                            </div>
                            <span class="model-badge">${req.model || 'Unknown'}</span>
                        </div>
                        <div class="card-body">
                            <div class="params-grid">
                                <div class="param-item">
                                    <div class="param-label">Status</div>
                                    <div class="param-value">
                                        <span class="status-indicator ${statusClass}">${req.status}</span>
                                    </div>
                                </div>
                                <div class="param-item">
                                    <div class="param-label">Context Size</div>
                                    <div class="param-value">${req.context_size || 'N/A'}</div>
                                </div>
                                <div class="param-item">
                                    <div class="param-label">Elapsed Time</div>
                                    <div class="param-value">${formatTime(req.elapsed_time)}</div>
                                </div>
                                <div class="param-item">
                                    <div class="param-label">Est. Remaining</div>
                                    <div class="param-value">${req.status === 'completed' ? 'Done' : formatTime(req.estimated_time_remaining)}</div>
                                </div>
                            </div>
                            
                            <div class="progress-container">
                                <div class="progress-label">
                                    <span>Generation Progress</span>
                                    <span>${req.progress} / ${req.total} tokens (${percent}%)</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${percent}%; background-color: ${progressColor}"></div>
                                </div>
                            </div>
                        </div>
                    `;
                    container.appendChild(div);
                }
            }
            
            function getStatusClass(status) {
                const classes = {
                    'loading': 'status-loading',
                    'generating': 'status-generating', 
                    'completed': 'status-completed',
                    'error': 'status-error'
                };
                return classes[status] || '';
            }
            
            function getProgressColor(status, percent) {
                if (status === 'error') return '#ef4444';
                if (status === 'completed') return '#10b981';
                if (status === 'loading') return '#3b82f6';
                
                if (percent < 30) return '#f59e0b';
                if (percent < 70) return '#10b981';
                return '#059669';
            }
            
            function getApiClass(apiFormat) {
                const classes = {
                    'openai': 'api-openai',
                    'ollama_chat': 'api-ollama',
                    'ollama_generate': 'api-ollama', 
                    'claude': 'api-claude',
                    'huggingface': 'api-huggingface',
                    'vllm': 'api-vllm'
                };
                return classes[apiFormat] || '';
            }
            
            function formatApiName(apiFormat) {
                const names = {
                    'openai': 'OpenAI',
                    'ollama_chat': 'Ollama',
                    'ollama_generate': 'Ollama',
                    'claude': 'Claude', 
                    'huggingface': 'HF',
                    'vllm': 'vLLM'
                };
                return names[apiFormat] || 'Unknown';
            }
            
            function formatTime(seconds) {
                if (!seconds && seconds !== 0) return 'N/A';
                
                if (seconds < 60) {
                    return `${seconds.toFixed(1)}s`;
                } else if (seconds < 3600) {
                    const minutes = Math.floor(seconds / 60);
                    const remainingSeconds = seconds % 60;
                    return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
                } else {
                    const hours = Math.floor(seconds / 3600);
                    const minutes = Math.floor((seconds % 3600) / 60);
                    return `${hours}h ${minutes}m`;
                }
            }
            
            function clearAllCompleted() {
                fetch('/api/dismiss/completed', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            refreshDashboard();
                        }
                    });
            }
            
            // Only refresh on page load and manual button clicks
            document.addEventListener('DOMContentLoaded', refreshDashboard);
        </script>
    </head>
    <body>
        <header>
            <div>
                <h1>🚀 Universal LLM API Dashboard</h1>
                <div class="header-subtitle">Supporting OpenAI, Ollama, Claude, HuggingFace & vLLM APIs</div>
            </div>
            <div class="header-actions">
                <button class="btn" onclick="refreshDashboard()">🔄 Refresh</button>
                <button class="btn btn-primary" onclick="clearAllCompleted()">🗑️ Clear Completed</button>
                <button class="btn" onclick="window.open('/', '_blank')">📋 API Info</button>
            </div>
        </header>
        
        <div class="container">
            <div class="stats-bar">
                <div class="stat-card">
                    <h3>Active Requests</h3>
                    <p id="active-count">0</p>
                </div>
                <div class="stat-card">
                    <h3>Completed</h3>
                    <p id="completed-count">0</p>
                </div>
                <div class="stat-card">
                    <h3>Errors</h3>
                    <p id="error-count">0</p>
                </div>
                <div class="stat-card">
                    <h3>Total</h3>
                    <p id="total-count">0</p>
                </div>
            </div>
            
            <div class="request-grid" id="requests">
                <div class="empty-state">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <h3>No active requests</h3>
                    <p>API requests will appear here. Click refresh to update manually.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

@app.route('/dashboard/data', methods=['GET'])
def dashboard_data():
    """API endpoint for dashboard data"""
    all_requests = request_tracker.get_all_requests()
    response_data = {}
    
    for req_id, req_status in all_requests.items():
        data = req_status.__dict__.copy()
        data['elapsed_time'] = time.time() - req_status.start_time
        
        if req_status.progress > 0 and req_status.progress < req_status.total:
            tokens_per_second = req_status.progress / data['elapsed_time']
            remaining_tokens = req_status.total - req_status.progress
            data['estimated_time_remaining'] = remaining_tokens / tokens_per_second
        else:
            data['estimated_time_remaining'] = None
        
        response_data[req_id] = data
    
    return jsonify(response_data)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    mapping = model_manager.build_model_mapping()
    return jsonify({
        'status': 'ok',
        'models_available': len(mapping),
        'active_requests': len(request_tracker.get_all_requests()),
        'supported_apis': [
            'OpenAI Chat Completions (/api/chat/completions)',
            'Ollama Chat (/api/chat)',
            'Ollama Generate (/api/generate)',
            'Claude Messages (/v1/messages)',
            'HuggingFace TGI (/generate, /v1/completions)',
            'vLLM Chat (/v1/chat/completions)',
            'vLLM Completions (/v1/completions)'
        ],
        'llama_executable': LLAMA_CLI,
        'default_tensor_split': args.default_tensor_split
    })

@app.route('/', methods=['GET'])
def api_info():
    """API information and available endpoints"""
    mapping = model_manager.build_model_mapping()
    available_models = list(mapping.keys())[:5]  # Show first 5 models
    
    return jsonify({
        'service': 'Universal LLM API Server',
        'version': '1.0.0',
        'description': 'Unified API server supporting multiple LLM API formats',
        'available_models': available_models,
        'total_models': len(mapping),
        'endpoints': {
            'OpenAI Compatible': {
                'chat_completions': '/api/chat/completions',
                'models': '/api/models'
            },
            'Ollama Compatible': {
                'chat': '/api/chat',
                'generate': '/api/generate',
                'tags': '/api/tags',
                'models': '/api/models'
            },
            'Claude Compatible': {
                'messages': '/v1/messages'
            },
            'HuggingFace TGI Compatible': {
                'generate': '/generate',
                'completions': '/v1/completions',
                'info': '/info'
            },
            'vLLM Compatible': {
                'chat_completions': '/v1/chat/completions',
                'completions': '/v1/completions',
                'models': '/v1/models'
            },
            'Management': {
                'health': '/health',
                'dashboard': '/dashboard',
                'progress': '/api/progress/<request_id>',
                'dismiss': '/api/dismiss/<request_id>'
            }
        },
        'features': [
            'Real-time model context detection',
            'Automatic prompt truncation',
            'GPU sharding support',
            'Streaming responses',
            'Progress tracking',
            'Web dashboard',
            'Multi-API compatibility'
        ]
    })

# ================================
# Main
# ================================

if __name__ == "__main__":
    logger.info(f"Starting modular LLM server on {args.host}:{args.port}")
    logger.info("Supported API formats:")
    logger.info("  • OpenAI: POST /api/chat/completions")
    logger.info("  • Ollama Chat: POST /api/chat")
    logger.info("  • Ollama Generate: POST /api/generate")
    logger.info("  • Claude: POST /v1/messages")
    logger.info("  • HuggingFace TGI: POST /generate")
    logger.info("  • vLLM: POST /v1/chat/completions")
    logger.info("  • Dashboard: GET /dashboard")
    logger.info("  • API Info: GET /")
    
    # Display available models on startup with context info
    mapping = model_manager.build_model_mapping()
    if mapping:
        logger.info(f"Found {len(mapping)} models:")
        successful_models = 0
        for i, model_id in enumerate(list(mapping.keys())[:5]):  # Show first 5
            try:
                model_info = model_manager.get_model_info(model_id)
                if model_info:
                    size_gb = model_info.size / (1024**3)
                    logger.info(f"  • {model_id}")
                    logger.info(f"    - Context: {model_info.context_size:,} tokens")
                    logger.info(f"    - Size: {size_gb:.1f}GB ({model_info.parameter_size}, {model_info.quantization})")
                    successful_models += 1
            except Exception as e:
                logger.debug(f"Error getting info for {model_id}: {e}")
        
        if len(mapping) > 5:
            logger.info(f"  ... and {len(mapping) - 5} more models")
        
        if successful_models > 0:
            # Show context detection summary
            context_summary = {}
            for model_id in list(mapping.keys()):
                try:
                    model_info = model_manager.get_model_info(model_id)
                    if model_info:
                        ctx = model_info.context_size
                        if ctx not in context_summary:
                            context_summary[ctx] = 0
                        context_summary[ctx] += 1
                except:
                    pass
            
            if context_summary:
                logger.info("Context size distribution:")
                for ctx_size, count in sorted(context_summary.items()):
                    logger.info(f"  • {ctx_size:,} tokens: {count} model(s)")
        else:
            logger.warning("Could not load model information for any models")
    else:
        logger.warning("No models found! Check your model directory configuration.")
        logger.warning(f"Model directory: {MODELS_BASE_DIR}")
        logger.warning(f"Manifests directory: {MANIFESTS_DIR}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

# ================================
# Usage Examples (for testing)
# ================================

"""
Example API calls for testing:

1. OpenAI Format:
curl -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3:8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

2. Ollama Chat Format:
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3:8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

3. Claude Format:
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3:8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

4. HuggingFace TGI Format:
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "Hello, how are you?",
    "parameters": {"max_new_tokens": 100, "temperature": 0.8}
  }'

5. vLLM Format:
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3:8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

6. List Models:
curl http://localhost:8000/api/models
curl http://localhost:8000/v1/models

7. Health Check:
curl http://localhost:8000/health

8. API Info:
curl http://localhost:8000/

9. Dashboard:
Open http://localhost:8000/dashboard in browser
"""
