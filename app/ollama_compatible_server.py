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
# import queue # Not actively used, consider removing if no plans
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
parser.add_argument('--default-tensor-split', type=str, default='0.25,0.25,0.25,0.25', # Example, might not be universally optimal
                    help='Default tensor split for GPU sharding (default: 0.25,0.25,0.25,0.25). Set to None or empty to disable.')

args = parser.parse_args()
if args.default_tensor_split and args.default_tensor_split.lower() in ['none', '']:
    args.default_tensor_split = None


# Set up paths based on arguments
MODELS_BASE_DIR = Path(args.model_dir)
MODELS_DIR = MODELS_BASE_DIR / 'blobs'
MANIFESTS_DIR = MODELS_BASE_DIR / 'manifests/registry.ollama.ai/library'
LLAMA_CPP_DIR = Path(args.llama_cpp_dir)
LOG_DIR = Path(args.log_dir)

# Create log directory if it doesn't exist
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not args.debug else logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'server.log'),
        logging.StreamHandler(sys.stdout) # Ensure logs go to stdout for container environments
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
    tensor_split: Optional[str]
    gpu_layers: int
    stream: bool
    additional_params: Dict[str, Any]

@dataclass
class ModelInfo:
    """Model information structure"""
    id: str              # Full model ID, e.g., "llama3:70b-instruct"
    name: str            # Base name, e.g., "llama3"
    path: str            # Filesystem path to the model blob
    context_size: int    # Detected context window size
    size: int            # File size in bytes
    parameter_size: str  # Estimated parameter size, e.g., "70B"
    quantization: str    # Detected quantization, e.g., "Q4_K_M"
    modified_at: str     # ISO 8601 timestamp of model file modification

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
    options: Dict[str, Any] # Copy of InternalRequest fields
    error: Optional[str] = None
    actual_tokens: int = 0
    context_size: int = 0 # Actual context_size used for the request
    prompt_tokens: int = 0

# ================================
# Model Management
# ================================

class ModelManager:
    def __init__(self):
        self._model_cache: Dict[str, ModelInfo] = {}
        self._context_cache: Dict[str, int] = {}
        
    def estimate_tokens(self, text: str) -> int:
        if not text: return 0
        # A very rough heuristic: average 4 chars per token. Can be improved.
        return len(text) // 4 
    
    def read_manifest_metadata(self, model_id: str) -> Dict[str, Any]:
        try:
            model_name_part, tag_part = model_id.split(':', 1) if ':' in model_id else (model_id, 'latest')
            manifest_path = MANIFESTS_DIR / model_name_part / tag_part
            
            if not manifest_path.exists():
                logger.debug(f"Manifest file not found for {model_id} at {manifest_path}")
                return {}
            
            with manifest_path.open('r', encoding='utf-8') as f:
                manifest_content = f.read()
            
            manifest_data = json.loads(manifest_content)
            config = manifest_data.get('config', {})
            param_str = config.get('parameters', '')
            if isinstance(param_str, str):
                for line in param_str.splitlines():
                    parts = line.split()
                    if len(parts) == 2:
                        key, value_str = parts[0], parts[1]
                        # Try to convert to int if it's a numeric parameter we care about
                        if key in ['num_ctx', 'num_gpu_layers', 'num_batch', 'n_ctx'] and value_str.isdigit():
                            config[key] = int(value_str)
                            logger.debug(f"Parsed from manifest config.parameters: {key}={config[key]} for {model_id}")
            return config
        except json.JSONDecodeError:
            logger.warning(f"JSONDecodeError for manifest {model_id}. Content snippet: {manifest_content[:200]}")
            return {} # Or try regex fallback if critical
        except Exception as e:
            logger.error(f"Error reading manifest metadata for {model_id}: {e}")
            return {}

    def get_context_length(self, model_id, mainfest_dir="/opt/llm/models/ollama/models/",blob_dir="/opt/llm/models/ollama/models/blobs"):
        # Load manifest JSON
        model_name_part, tag_part = model_id.split(':', 1) if ':' in model_id else (model_id, 'latest')
        manifest_path = MANIFESTS_DIR / model_name_part / tag_part
        with manifest_path.open('r', encoding='utf-8') as f:
            manifest_content = f.read()
            manifest_data = json.loads(manifest_content)

        print(json.dumps(manifest_data,indent=2))
        # Find the params layer
        params_layer = next(
            (layer for layer in manifest_data.get('layers', [])
             if layer['mediaType'] == 'application/vnd.ollama.image.params'),
            None
        )

        if not params_layer:
            raise ValueError("No 'params' layer found in the manifest.")

        # Extract digest and format blob file path
        digest = params_layer['digest'].replace('sha256:', '')
        blob_path = os.path.join(blob_dir, f'sha256-{digest}')

        # Load the JSON in the params blob
        if not os.path.exists(blob_path):
            raise FileNotFoundError(f"Blob file not found: {blob_path}")

        with open(blob_path, 'r') as f:
            params = json.load(f)

        context_length = params.get('context_length')
        if context_length is None:
            raise KeyError("No 'context_length' found in the params JSON.")
    
        return context_length

    
    def detect_context_size(self, model_path_str: str, model_id: str) -> int:
        cache_key = f"{model_path_str}:{model_id}"

        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        default_context_size = 16384
        detected_size = default_context_size

        # 1. Try manifest metadata (most reliable for Ollama models)
        manifest_metadata = self.read_manifest_metadata(model_id)
        context_keys_from_manifest = ['num_ctx', 'n_ctx', 'context_length', 'max_position_embeddings', 'n_ctx_train', 'max_context_length']
        for key in context_keys_from_manifest:
            if key in manifest_metadata and isinstance(manifest_metadata[key], int) and manifest_metadata[key] > 0:
                detected_size = manifest_metadata[key]
                logger.info(f"Context size for {model_id} from manifest key '{key}': {detected_size}")
                self._context_cache[cache_key] = detected_size
                return detected_size

        # 2. Filename/Model ID analysis (case-insensitive for 'k')
        # model_id is like "llama3:70b-instruct-128k"
        # model_path_str is like "/path/to/blobs/sha256-..."
        
        # Use model_id for name-based patterns first, as it's more descriptive
        model_id_lower = model_id.lower() # e.g., "llama3:70b-instruct-128k"
        filename_lower = Path(model_path_str).name.lower() # e.g., "sha256-..."

        context_patterns = [
            (r'(\d+)[kK](?:-ctx|-context)?', lambda x: int(x) * 1024), # 128K, 32k, 8k-ctx
            (r'ctx(\d+)', lambda x: int(x)),                           # ctx4096
            (r'(\d+)ctx', lambda x: int(x)),                           # 4096ctx
        ]
        
        # Check model_id_lower first, then filename_lower
        for text_source_name, text_source_val in [("model_id", model_id_lower), ("filename", filename_lower)]:
            for pattern_regex, converter_func in context_patterns:
                match = re.search(pattern_regex, text_source_val)
                if match:
                    try:
                        val = converter_func(match.group(1))
                        if val > detected_size: # Prefer larger context if multiple hints found
                            detected_size = val
                            logger.info(f"Context size for {model_id} from '{text_source_name}' ('{text_source_val[:30]}...') pattern '{pattern_regex}': {detected_size}")
                    except ValueError:
                        logger.warning(f"ValueError converting matched context size from '{text_source_val}' with pattern '{pattern_regex}'")
                        continue
        
        if detected_size != default_context_size: # If any pattern matched and updated
            self._context_cache[cache_key] = detected_size
            return detected_size

        # 3. Model family-based defaults (Order from more specific to general)
        # This uses model_id_lower which includes tags and quant info that might have family names.
        family_defaults = {
            # Most specific first
            'llama3-70b': 8192, 'llama-3.3-70b': 131072, # Llama 3.3 70B can be 128K
            'llama3-8b': 8192,
            'llama3': 8192, # General Llama3
            'codellama-34b': 16384, 'codellama-13b': 16384, 'codellama-7b': 16384,
            'codellama': 16384, # General Codellama
            'llama2-70b': 4096, 'llama2-13b': 4096, 'llama2-7b': 4096,
            'llama2': 4096, # General Llama2
            'phi-3-mini-128k': 131072, 'phi3-mini-128k': 131072,
            'phi-3': 4096, 'phi3': 4096, # Default Phi3 (e.g. mini 4k)
            'mistral-7b': 32768, 'mistral': 32768, # Mistral base, often 32k now
            'mixtral-8x22b': 65536, 'mixtral-8x7b': 32768,
            'mixtral': 32768, # General Mixtral
            'qwen2-72b': 32768, 'qwen2-7b': 32768, 'qwen2': 32768,
            'gemma-7b': 8192, 'gemma-2b': 8192, 'gemma': 8192,
            'deepseek-coder-33b': 16384, 'deepseek-coder': 16384, # Deepseek Coder often has larger context
            'starcoder': 8192
        }
        for family_key, family_ctx_size in family_defaults.items():
            if family_key in model_id_lower:
                detected_size = family_ctx_size
                logger.info(f"Context size for {model_id} from family default '{family_key}': {detected_size}")
                self._context_cache[cache_key] = detected_size
                return detected_size
        
        logger.warning(f"Could not reliably determine context size for {model_id} (path: {model_path_str}), using default: {default_context_size}")
        self._context_cache[cache_key] = default_context_size
        return default_context_size

    def build_model_mapping(self) -> Dict[str, str]:
        mapping = {}
        if not MANIFESTS_DIR.exists() or not MODELS_DIR.exists():
            logger.warning(f"Manifests ({MANIFESTS_DIR}) or Models ({MODELS_DIR}) dir not found.")
            return mapping
                
        for model_dir_name_path in MANIFESTS_DIR.iterdir():
            if not model_dir_name_path.is_dir(): continue
            model_name_part = model_dir_name_path.name
            for tag_file_path in model_dir_name_path.iterdir():
                if not tag_file_path.is_file(): continue
                tag_part = tag_file_path.name
                model_id = f"{model_name_part}:{tag_part}"
                try:
                    with tag_file_path.open('r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    for layer in manifest.get('layers', []):
                        media_type = layer.get('mediaType', '')
                        if 'model' in media_type or 'gguf' in media_type or layer.get('digest', '').endswith(".gguf"):
                            digest = layer.get('digest', '')
                            if digest.startswith('sha256:'):
                                digest_val = digest.split(':')[1]
                                blob_path = MODELS_DIR / f"sha256-{digest_val}"
                                if blob_path.exists():
                                    mapping[model_id] = str(blob_path)
                                    logger.debug(f"Mapped model {model_id} to {blob_path}")
                                    break 
                                else:
                                    logger.warning(f"Blob path {blob_path} for model {model_id} does not exist (Digest: {digest}).")
                except Exception as e:
                    logger.warning(f"Error parsing manifest for {model_id} at {tag_file_path}: {e}")
        
        logger.info(f"Built model mapping with {len(mapping)} entries.")
        return mapping
    
    def find_model_path(self, model_id_query: str) -> Optional[str]:
        # Rebuild mapping each time for simplicity, can be cached if performance becomes an issue
        current_mapping = self.build_model_mapping()
        
        if model_id_query in current_mapping:
            return current_mapping[model_id_query]
        
        if ':' not in model_id_query: # e.g., user provides "llama3"
            latest_id_query = f"{model_id_query}:latest"
            if latest_id_query in current_mapping:
                return current_mapping[latest_id_query]
            
            # Try partial match: if "llama3" is query, match "llama3:70b", "llama3:8b" (prefer longer context or larger if multiple)
            # This part can be sophisticated. For now, simple prefix match.
            best_match_path = None
            best_match_id = None
            for mapped_id, path_str in current_mapping.items():
                if mapped_id.startswith(model_id_query + ":"):
                    if best_match_path is None: # First match
                        best_match_path = path_str
                        best_match_id = mapped_id
                    else:
                        # Add logic to prefer, e.g. "latest" tag, or larger model if names allow
                        if ":latest" in mapped_id and (best_match_id and ":latest" not in best_match_id) :
                             best_match_path = path_str
                             best_match_id = mapped_id
            if best_match_path:
                logger.info(f"Partial match for '{model_id_query}': using '{best_match_id}' -> {best_match_path}")
                return best_match_path

        # Fallback: treat model_id_query as a potential direct file path
        potential_path = Path(model_id_query)
        if potential_path.is_file() and potential_path.name.lower().endswith(".gguf"):
            logger.info(f"Treating model_id_query '{model_id_query}' as a direct file path.")
            return str(potential_path)

        logger.warning(f"Model path not found for '{model_id_query}'. Checked mapping, 'latest' tag, partial matches, and direct path.")
        return None
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]: # model_id is e.g. "llama3:latest"
        if model_id in self._model_cache:
            return self._model_cache[model_id]
        
        model_path_str = self.find_model_path(model_id)
        if not model_path_str:
            logger.error(f"Model path not found for {model_id} in get_model_info.")
            return None
        
        model_path = Path(model_path_str)
        
        try:
            context_size = self.detect_context_size(str(model_path), model_id) # Pass model_id for richer name parsing
            file_size = model_path.stat().st_size
            
            # Use model_id (which is like "llama3:70b-instruct") for parsing params/quant
            # as it's more descriptive than the blob filename.
            name_for_parsing = model_id.lower()

            param_size_str = "Unknown"
            # Order from more specific (e.g., 8x22b) to less specific (e.g., 7b before b)
            # Ensure regexes are specific enough (e.g. \b for word boundaries if needed)
            param_patterns = {
                "8x22B": r"8x22b", "8x7B": r"8x7b",
                "671B": r"671b", "235B": r"235b",
                "70B": r"70b", "65B": r"65b", "40B": r"40b", "34B": r"34b", "33B": r"33b", "30B": r"30b",
                "27B": r"27b", "22B": r"22b", "15B": r"15b", "13B": r"13b",
                "8B": r"8b", "7B": r"7b", "3.8B": r"3\.8b", "3B": r"3b",
                "2B": r"2b", "1.5B": r"1\.5b", "1.1B": r"1\.1b", "1B": r"1b"
            } # Keys are labels, values are regex patterns
            for label, pattern in param_patterns.items():
                if re.search(pattern, name_for_parsing):
                    param_size_str = label
                    break 
            
            quant_str = "Unknown"
            # Order by typical preference or commonality.
            quant_patterns = {
                "F16": r"f16", "FP16": r"fp16", "F32": r"f32", # Unquantized last or first?
                "Q8_0": r"q8_0", "Q6_K": r"q6_k", "Q5_K_M": r"q5_k_m", "Q5_0": r"q5_0",
                "Q4_K_M": r"q4_k_m", "Q4_0": r"q4_0", "Q3_K_L": r"q3_k_l", "Q3_K_M": r"q3_k_m", "Q3_K_S": r"q3_k_s",
                "Q2_K": r"q2_k",
                # Add other quants like Q5_K_S, Q4_K_S etc. if needed
            }
            for label, pattern in quant_patterns.items():
                 if re.search(pattern, name_for_parsing):
                    quant_str = label
                    break
            
            model_info_obj = ModelInfo(
                id=model_id,
                name=model_id.split(':')[0] if ':' in model_id else model_id,
                path=str(model_path),
                context_size=context_size,
                size=file_size,
                parameter_size=param_size_str,
                quantization=quant_str,
                modified_at=datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
            )
            
            self._model_cache[model_id] = model_info_obj
            return model_info_obj
        except Exception as e:
            logger.exception(f"Error getting model info for {model_id} (path: {model_path_str}): {e}")
            return None


# ================================
# Request Adapters (Implementations assumed to be mostly correct from previous, focusing on main flow)
# ================================
# (Adapters: OpenAIAdapter, OllamaChatAdapter, OllamaGenerateAdapter, ClaudeAdapter, HuggingFaceAdapter, VLLMAdapter)
# Small corrections for consistency:
# - Use os.cpu_count() or a default for threads, e.g., `data.get('threads', os.cpu_count() or 4)`
# - Ensure gpu_layers defaults to -1 (all layers for llama.cpp) if not specified or if API uses different name.
# - `tensor_split` should use `args.default_tensor_split` if available.

class RequestAdapter(ABC):
    @abstractmethod
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest: pass
    @abstractmethod
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]: pass

class OpenAIAdapter(RequestAdapter): # Covers OpenAI & generic vLLM Chat
    def __init__(self, model_manager: ModelManager): self.model_manager = model_manager
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        messages = data.get('messages', [])
        model_name = data.get('model')
        if not messages or not model_name: raise ValueError("Missing required fields: messages and model")
        
        prompt = self._messages_to_prompt(messages, model_name) # Pass model_name for model-specific templating
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info: raise ValueError(f"Model {model_name} not found or info unavailable.")
        
        num_threads = data.get('threads', data.get('n_threads', os.cpu_count() or 4)) # Common variations
        
        return InternalRequest(
            request_id=str(uuid.uuid4()), model_name=model_name, prompt=prompt,
            context_size=model_info.context_size, 
            max_tokens=data.get('max_tokens', 512), 
            temperature=data.get('temperature', 0.8), 
            threads=num_threads,
            tensor_split=data.get('tensor_split', args.default_tensor_split), 
            gpu_layers=data.get('gpu_layers', data.get('n_gpu_layers', -1)),
            stream=data.get('stream', False),
            additional_params={'api_format': 'openai', 'messages': messages} # or 'vllm_chat' if specific handling
        )

    def _messages_to_prompt(self, messages: List[Dict[str, Any]], model_id: str) -> str:
        # Basic Llama3/Mistral Instruct style. Can be enhanced.
        # Example: <s>[INST] System Prompt [/INST]</s>\n<s>[INST] UserP1 [/INST] AsstR1 </s>\n<s>[INST] UserP2 [/INST]
        # Simplified for now:
        # USER: ... \nASSISTANT: ... \nUSER: ... \nASSISTANT:
        prompt_str = ""
        # A simple generic prompt formatter
        # TODO: Make this model-aware based on model_id (e.g. check for 'instruct' or family)
        is_instruct_model = "instruct" in model_id.lower() # Basic heuristic

        for msg in messages:
            role = msg.get('role', 'user').lower()
            content = msg.get('content', '')
            if role == 'system':
                 prompt_str += f"{content}\n\n" # System prompt at the beginning
            elif role == 'user':
                prompt_str += f"USER: {content}\n"
            elif role == 'assistant':
                prompt_str += f"ASSISTANT: {content}\n"
        
        prompt_str += "ASSISTANT:" # Model should continue from here
        return prompt_str.strip()

    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        return {
            'id': request.request_id, 'object': 'chat.completion', 'created': int(time.time()),
            'model': request.model_name,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': output.strip()}, 'finish_reason': 'stop'}],
            'usage': {
                'prompt_tokens': self.model_manager.estimate_tokens(request.prompt),
                'completion_tokens': self.model_manager.estimate_tokens(output),
                'total_tokens': self.model_manager.estimate_tokens(request.prompt + output)
            }
        }

# OllamaChatAdapter (similar to OpenAIAdapter but uses Ollama's option names)
class OllamaChatAdapter(RequestAdapter):
    def __init__(self, model_manager: ModelManager): self.model_manager = model_manager
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        messages = data.get('messages', [])
        model_name = data.get('model')
        if not messages or not model_name: raise ValueError("Missing required fields: messages and model")

        prompt = OpenAIAdapter(self.model_manager)._messages_to_prompt(messages, model_name)
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info: raise ValueError(f"Model {model_name} not found.")

        options = data.get('options', {})
        context_size_req = options.get('num_ctx', model_info.context_size)
        
        return InternalRequest(
            request_id=str(uuid.uuid4()), model_name=model_name, prompt=prompt,
            context_size=min(context_size_req, model_info.context_size), # Respect model limit
            max_tokens=options.get('num_predict', 512),
            temperature=options.get('temperature', 0.8),
            threads=options.get('num_thread', os.cpu_count() or 4),
            tensor_split=options.get('tensor_split', args.default_tensor_split),
            gpu_layers=options.get('num_gpu_layers', -1),
            stream=data.get('stream', False),
            additional_params={'api_format': 'ollama_chat', 'messages': messages}
        )
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        return {
            'model': request.model_name, 'created_at': datetime.now().isoformat(),
            'message': {'role': 'assistant', 'content': output.strip()},
            'done': True
        }

# OllamaGenerateAdapter
class OllamaGenerateAdapter(RequestAdapter):
    def __init__(self, model_manager: ModelManager): self.model_manager = model_manager
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        prompt = data.get('prompt', '')
        model_name = data.get('model')
        if not prompt or not model_name: raise ValueError("Missing prompt/model for Ollama generate.")
        
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info: raise ValueError(f"Model {model_name} not found.")
        options = data.get('options', {})
        context_size_req = options.get('num_ctx', model_info.context_size)

        return InternalRequest(
            request_id=str(uuid.uuid4()), model_name=model_name, prompt=prompt,
            context_size=min(context_size_req, model_info.context_size),
            max_tokens=options.get('num_predict', 512),
            temperature=options.get('temperature', 0.8),
            threads=options.get('num_thread', os.cpu_count() or 4),
            tensor_split=options.get('tensor_split', args.default_tensor_split),
            gpu_layers=options.get('num_gpu_layers', -1),
            stream=data.get('stream', False),
            additional_params={'api_format': 'ollama_generate'}
        )
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        # Example of adding more fields if llama.cpp provides them (e.g., via stderr parsing or future JSON output)
        final_status = request_tracker.get_request(request.request_id)
        resp_dict = {
            'model': request.model_name, 'created_at': datetime.now().isoformat(),
            'response': output.strip(), 'done': True
        }
        if final_status: # Add stats if available
            resp_dict.update({
                'total_duration': int((final_status.last_update - final_status.start_time) * 1e9), # ns
                'prompt_eval_count': final_status.prompt_tokens,
                'eval_count': final_status.actual_tokens,
                # 'load_duration', 'eval_duration' etc. would require more detailed parsing from llama.cpp
            })
        return resp_dict

# ClaudeAdapter
class ClaudeAdapter(RequestAdapter):
    def __init__(self, model_manager: ModelManager): self.model_manager = model_manager
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        messages = data.get('messages', [])
        model_name = data.get('model')
        if not messages or not model_name: raise ValueError("Missing messages/model for Claude.")
        
        prompt = self._claude_messages_to_prompt(messages, data.get('system'))
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info: raise ValueError(f"Model {model_name} not found.")

        return InternalRequest(
            request_id=str(uuid.uuid4()), model_name=model_name, prompt=prompt,
            context_size=model_info.context_size,
            max_tokens=data.get('max_tokens', 512), # Claude uses 'max_tokens'
            temperature=data.get('temperature', 0.7), # Claude default temp often 0.7 or 1.0
            threads=os.cpu_count() or 4,
            tensor_split=args.default_tensor_split,
            gpu_layers=-1,
            stream=data.get('stream', False),
            additional_params={'api_format': 'claude', 'messages': messages, 'system': data.get('system')}
        )
    def _claude_messages_to_prompt(self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> str:
        prompt_str = ""
        if system_prompt: prompt_str += f"{system_prompt}\n\n" # Anthropic recommends two newlines after system
        for msg in messages:
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            if role == 'user': prompt_str += f"Human: {content}\n\n"
            elif role == 'assistant': prompt_str += f"Assistant: {content}\n\n"
        prompt_str += "Assistant:" # Ready for model completion
        return prompt_str.strip()
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        return {
            'id': request.request_id, 'type': 'message', 'role': 'assistant',
            'content': [{'type': 'text', 'text': output.strip()}],
            'model': request.model_name, 'stop_reason': 'end_turn', 'stop_sequence': None,
            'usage': {'input_tokens': self.model_manager.estimate_tokens(request.prompt),
                      'output_tokens': self.model_manager.estimate_tokens(output)}
        }

# HuggingFaceAdapter
class HuggingFaceAdapter(RequestAdapter):
    def __init__(self, model_manager: ModelManager): self.model_manager = model_manager
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        inputs = data.get('inputs', '')
        model_name_req = data.get('model') # TGI server usually knows its model, client might not send
        
        model_name_to_use = model_name_req
        if not model_name_to_use: # If client doesn't specify, try to pick one
            all_models = self.model_manager.build_model_mapping()
            if all_models: model_name_to_use = list(all_models.keys())[0]
            else: raise ValueError("No model specified in TGI request and no default available.")
        
        if not inputs: raise ValueError("Missing 'inputs' for HuggingFace TGI.")
        model_info = self.model_manager.get_model_info(model_name_to_use)
        if not model_info: raise ValueError(f"Model {model_name_to_use} not found.")

        parameters = data.get('parameters', {})
        return InternalRequest(
            request_id=str(uuid.uuid4()), model_name=model_name_to_use, prompt=inputs,
            context_size=model_info.context_size,
            max_tokens=parameters.get('max_new_tokens', 512),
            temperature=parameters.get('temperature', 0.8),
            threads=os.cpu_count() or 4,
            tensor_split=args.default_tensor_split,
            gpu_layers=-1,
            stream=data.get('stream', False),
            additional_params={'api_format': 'huggingface', 'parameters': parameters}
        )
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        return {'generated_text': output.strip()} # TGI non-stream usually list of this

# VLLMAdapter (Can handle both chat and completion based on payload)
class VLLMAdapter(RequestAdapter):
    def __init__(self, model_manager: ModelManager): self.model_manager = model_manager
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        model_name = data.get('model')
        if not model_name: raise ValueError("Missing 'model' for vLLM request.")
        
        api_format_type = ''
        prompt_str = ''
        additional_params_base = {'original_payload': data}

        if 'messages' in data: # Chat style
            messages = data.get('messages', [])
            if not messages: raise ValueError("Empty 'messages' for vLLM chat.")
            prompt_str = OpenAIAdapter(self.model_manager)._messages_to_prompt(messages, model_name)
            api_format_type = 'vllm_chat'
            additional_params_base['messages'] = messages
        elif 'prompt' in data: # Completion style
            prompt_str = data.get('prompt', '')
            if not prompt_str: raise ValueError("Empty 'prompt' for vLLM completion.")
            api_format_type = 'vllm_completion'
        else:
            raise ValueError("vLLM request must contain 'messages' or 'prompt'.")
        
        additional_params_base['api_format'] = api_format_type
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info: raise ValueError(f"Model {model_name} not found.")

        return InternalRequest(
            request_id=str(uuid.uuid4()), model_name=model_name, prompt=prompt_str,
            context_size=model_info.context_size,
            max_tokens=data.get('max_tokens', 512),
            temperature=data.get('temperature', 0.8),
            threads=os.cpu_count() or 4,
            tensor_split=args.default_tensor_split,
            gpu_layers=-1,
            stream=data.get('stream', False),
            additional_params=additional_params_base
        )
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        api_format = request.additional_params.get('api_format', 'vllm_completion')
        if api_format == 'vllm_chat':
            return OpenAIAdapter(self.model_manager).format_response(output, request) # Reuse OpenAI format
        else: # vLLM completion mimics OpenAI text_completion
            return {
                'id': request.request_id, 'object': 'text_completion', 'created': int(time.time()),
                'model': request.model_name,
                'choices': [{'text': output.strip(), 'index': 0, 'logprobs': None, 'finish_reason': 'length'}],
                'usage': { 'prompt_tokens': self.model_manager.estimate_tokens(request.prompt),
                           'completion_tokens': self.model_manager.estimate_tokens(output),
                           'total_tokens': self.model_manager.estimate_tokens(request.prompt + output)}
            }


# ================================
# Request Tracker
# ================================
class RequestTracker:
    def __init__(self):
        self.active_requests: Dict[str, RequestStatus] = {}
        self._lock = threading.Lock()

    def add_request(self, request: InternalRequest) -> None:
        with self._lock:
            if request.request_id in self.active_requests:
                logger.warning(f"[{request.request_id}] Attempted to add existing request to tracker.")
                return
            
            # Corrected: Access global model_manager instance
            prompt_toks = 0
            if 'model_manager' in globals() and model_manager: # Check if global exists
                 prompt_toks = model_manager.estimate_tokens(request.prompt)
            else:
                 logger.error("Global 'model_manager' not found during RequestTracker.add_request. Prompt tokens will be 0.")

            self.active_requests[request.request_id] = RequestStatus(
                request_id=request.request_id, status='loading', progress=0,
                total=request.max_tokens, output='', start_time=time.time(),
                last_update=time.time(), model=request.model_name,
                options=request.__dict__.copy(),
                context_size=request.context_size,
                prompt_tokens=prompt_toks
            )
            logger.info(f"[{request.request_id}] Added to tracker. Model: {request.model_name}, Prompt tokens: {prompt_toks}, Max new tokens: {request.max_tokens}")

    def update_request(self, request_id: str, **kwargs) -> None:
        with self._lock:
            if request_id in self.active_requests:
                for key, value in kwargs.items():
                    setattr(self.active_requests[request_id], key, value)
                self.active_requests[request_id].last_update = time.time()
            else:
                logger.warning(f"[{request_id}] Update attempt for non-existent request in tracker.")
    
    def get_request(self, request_id: str) -> Optional[RequestStatus]:
        with self._lock: return self.active_requests.get(request_id)
    
    def remove_request(self, request_id: str) -> None:
        with self._lock:
            if self.active_requests.pop(request_id, None):
                logger.info(f"[{request_id}] Removed from tracker.")
            else:
                logger.debug(f"[{request_id}] Attempt to remove non-existent or already removed request.")
    
    def get_all_requests(self) -> Dict[str, RequestStatus]:
        with self._lock: return self.active_requests.copy()

    def remove_completed(self, older_than_seconds: int = 300) -> int: # Add age threshold
        removed_count = 0
        with self._lock:
            current_time = time.time()
            to_remove = []
            for req_id, req_status in self.active_requests.items():
                is_finished = req_status.status in ['completed', 'error']
                is_old = (current_time - req_status.last_update) > older_than_seconds
                if is_finished and is_old:
                    to_remove.append(req_id)
            
            for req_id in to_remove:
                del self.active_requests[req_id]
                removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} finished requests older than {older_than_seconds}s from tracker.")
        return removed_count

# ================================
# LLAMA Executor (Refactored for clarity)
# ================================
class LLAMAExecutor:
    def __init__(self, llama_cli_path: str, model_manager_ref: ModelManager, request_tracker_ref: RequestTracker):
        self.llama_cli_path = llama_cli_path
        self.model_manager = model_manager_ref # Store references
        self.request_tracker = request_tracker_ref

    def _validate_and_prepare_request(self, request: InternalRequest) -> Optional[str]:
        model_info = self.model_manager.get_model_info(request.model_name)
        if not model_info:
            err_msg = f"Model {request.model_name} not found by ModelManager."
            # Tracker updated by caller if request was added
            return err_msg

        detected_model_context = model_info.context_size
        if request.context_size <= 0 or request.context_size > detected_model_context:
            logger.warning(
                f"[{request.request_id}] Requested context_size {request.context_size} (for llama.cpp -c) is invalid or exceeds model limit {detected_model_context}. "
                f"Adjusting to {detected_model_context}."
            )
            request.context_size = detected_model_context
        
        req_status = self.request_tracker.get_request(request.request_id)
        prompt_tokens = req_status.prompt_tokens if req_status else self.model_manager.estimate_tokens(request.prompt)
        
        safety_buffer = 100 # For BOS/EOS, template overhead, etc.
        available_for_generation = request.context_size - prompt_tokens - safety_buffer

        if available_for_generation <= 0:
            err_msg = (f"Prompt too long ({prompt_tokens} tokens) for context window ({request.context_size}). "
                       f"Model supports {detected_model_context}. Need ~{prompt_tokens + safety_buffer} for prompt & buffer. "
                       f"Available for generation: {available_for_generation} tokens.")
            return err_msg

        if request.max_tokens > available_for_generation:
            original_max_tokens = request.max_tokens
            request.max_tokens = max(10, available_for_generation) 
            logger.warning(
                f"[{request.request_id}] Adjusted max_tokens from {original_max_tokens} to {request.max_tokens} "
                f"to fit within available generation space ({available_for_generation} tokens)."
            )
        
        if req_status: # Update tracker with potentially adjusted values
            self.request_tracker.update_request(
                request.request_id, options=request.__dict__.copy(),
                context_size=request.context_size, total=request.max_tokens
            )
        return None

    def _execute_shared(self, request: InternalRequest, is_streaming: bool) -> Tuple[Optional[subprocess.Popen], Optional[str], Optional[str]]:
        """Shared logic to build command and start process. Returns (process, prompt_file_path, error_message)"""
        self.request_tracker.add_request(request)
        validation_error = self._validate_and_prepare_request(request)
        if validation_error:
            logger.error(f"[{request.request_id}] Validation failed: {validation_error}")
            self.request_tracker.update_request(request.request_id, status='error', error=validation_error)
            return None, None, validation_error

        model_info = self.model_manager.get_model_info(request.model_name) # Safe, checked in validation

        # Create temporary prompt file with UTF-8 encoding
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix=".txt") as temp_prompt_file:
                temp_prompt_file.write(request.prompt)
                prompt_file_path = temp_prompt_file.name
        except Exception as e_tmp:
            err_msg = f"Failed to create temporary prompt file: {e_tmp}"
            logger.exception(f"[{request.request_id}] {err_msg}")
            self.request_tracker.update_request(request.request_id, status='error', error=err_msg)
            return None, None, err_msg

        logger.info(f"[{request.request_id}] Prompt (first 100 chars): '{request.prompt[:100].replace(chr(10), ' ')}...' (File: {prompt_file_path})")
        
        cmd = [
            self.llama_cli_path, "-m", model_info.path, "-f", prompt_file_path,
            "-c", str(request.context_size), "-n", str(request.max_tokens),
            "-t", str(request.threads), "--temp", str(request.temperature),
            "--no-display-prompt", # Avoids llama.cpp printing the prompt back
            "-mg", "3",
            "-no-cnv", # Removes conversation mode and exits after processing prompt
            "-ngl", "999"
        ]
        if request.tensor_split: cmd.extend(["-ts", request.tensor_split])
        # NGL: 0 means no layers on GPU, >0 means specific number. -1 or not present means auto/all.
        if request.gpu_layers is not None and request.gpu_layers >= 0: 
             cmd.extend(["-ngl", str(request.gpu_layers)])
        # Add other llama.cpp params as needed, e.g., --repeat-penalty, --top-k, --top-p
        # if request.additional_params.get('top_k'): cmd.extend(["--top-k", str(request.additional_params['top_k'])])

        logger.info(f"[{request.request_id}] Executing ({'stream' if is_streaming else 'non-stream'}): {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                text=True, bufsize=1, encoding='utf-8', errors='replace'
            )
            self.request_tracker.update_request(request.request_id, status='generating')
            return process, prompt_file_path, None
        except Exception as e_popen:
            err_msg = f"Failed to start llama.cpp process: {e_popen}"
            logger.exception(f"[{request.request_id}] {err_msg}")
            self.request_tracker.update_request(request.request_id, status='error', error=err_msg)
            if os.path.exists(prompt_file_path): os.unlink(prompt_file_path)
            return None, prompt_file_path, err_msg


    def execute_request_streaming(self, request: InternalRequest) -> Generator[str, None, None]:
        process, prompt_file_path, error_msg = self._execute_shared(request, is_streaming=True)
        if error_msg or not process or not prompt_file_path: # error_msg implies process is None
            yield f"Error: {error_msg or 'Process initialization failed'}"
            return

        stderr_lines: List[str] = []
        def _monitor_stderr_stream():
            for line in process.stderr: # type: ignore[union-attr]
                line_strip = line.strip()
                logger.info(f"[{request.request_id}] STDERR (stream): {line_strip}")
                stderr_lines.append(line_strip)
                if len(stderr_lines) > 20: stderr_lines.pop(0) # Keep recent lines
                # Update tracker with error if specific phrases found
                # This helps dashboard show error status quickly
                # (More robust error detection would parse structured errors if llama.cpp provides them)
                if "error" in line_strip.lower() or "failed" in line_strip.lower():
                    if not (self.request_tracker.get_request(request.request_id) or {}).get('error'): # Avoid overwriting more specific errors
                        self.request_tracker.update_request(request.request_id, status='error', error=f"llama.cpp stderr: {line_strip[:200]}")
        
        stderr_thread = threading.Thread(target=_monitor_stderr_stream, daemon=True)
        stderr_thread.start()

        full_output = ""
        generated_chunk_count = 0
        try:
            for chunk in process.stdout: # type: ignore[union-attr]
                req_stat = self.request_tracker.get_request(request.request_id)
                if req_stat and req_stat.status == 'error':
                    logger.warning(f"[{request.request_id}] Error status from tracker, breaking stdout (stream). Error: {req_stat.error}")
                    break
                
                # logger.debug(f"[{request.request_id}] Stream chunk: {repr(chunk)}") # Can be very verbose
                full_output += chunk
                generated_chunk_count += 1 # This is not token count, but output emission count
                self.request_tracker.update_request(request.request_id, progress=generated_chunk_count, output=full_output) # Progress is arbitrary here
                yield chunk
            
            process.wait(timeout=10) # Wait with timeout for process to finish after stdout closes
        except subprocess.TimeoutExpired:
            logger.warning(f"[{request.request_id}] Timeout waiting for llama.cpp process to exit after stdout closed (stream). Killing.")
            process.kill()
        except Exception as e_read:
            logger.exception(f"[{request.request_id}] Exception reading stdout (stream): {e_read}")
            self.request_tracker.update_request(request.request_id, status='error', error=f"Stream read error: {e_read}")
            yield f"Error: Stream read error: {e_read}" # Yield error to client stream
        finally:
            stderr_thread.join(timeout=2.0)
            if process.poll() is None: process.kill() # Ensure it's terminated

            actual_gen_tokens = self.model_manager.estimate_tokens(full_output)
            final_status_obj = self.request_tracker.get_request(request.request_id)

            if final_status_obj and final_status_obj.status != 'error':
                if process.returncode != 0:
                    err_detail = f"LLAMA CLI exited with code {process.returncode}. Last stderr: {' | '.join(stderr_lines)}"
                    logger.error(f"[{request.request_id}] {err_detail}")
                    self.request_tracker.update_request(request.request_id, status='error', error=err_detail, output=full_output, actual_tokens=actual_gen_tokens)
                    # Don't yield error here if stream already ended, client gets completion signal from adapter
                else:
                    logger.info(f"[{request.request_id}] Streaming completed successfully. Output len: {len(full_output)}, Estimated tokens: {actual_gen_tokens}")
                    self.request_tracker.update_request(request.request_id, status='completed', output=full_output, progress=request.max_tokens, actual_tokens=actual_gen_tokens)
            elif final_status_obj : # Already in error state
                 logger.info(f"[{request.request_id}] Streaming ended (already in error state: {final_status_obj.error}). Output len: {len(full_output)}")
                 self.request_tracker.update_request(request.request_id, output=full_output, actual_tokens=actual_gen_tokens)


            if os.path.exists(prompt_file_path):
                try: os.unlink(prompt_file_path)
                except OSError as unlink_err: logger.error(f"[{request.request_id}] Error unlinking temp prompt file {prompt_file_path}: {unlink_err}")


    def execute_request_non_streaming(self, request: InternalRequest) -> Tuple[str, Optional[str]]:
        process, prompt_file_path, error_msg = self._execute_shared(request, is_streaming=False)
        if error_msg or not process or not prompt_file_path:
            return "", error_msg or 'Process initialization failed'

        full_output = ""
        full_stderr = ""
        try:
            # Timeout needs to be generous for large models/long prompts/many tokens
            # e.g., (max_tokens / avg_tps) + model_load_time + prompt_eval_time
            # For now, a fixed large timeout. Consider making this dynamic.
            communicate_timeout_seconds = 30 * 60 # 30 minutes
            logger.info(f"[{request.request_id}] Calling process.communicate(timeout={communicate_timeout_seconds}s) for non-streaming.")
            
            stdout_data, stderr_data = process.communicate(timeout=communicate_timeout_seconds)
            full_output = stdout_data.strip()
            full_stderr = stderr_data.strip()

            if full_stderr: 
                logger.info(f"[{request.request_id}] STDERR (non-stream, first 500): {full_stderr[:500]}")
            
            actual_gen_tokens = self.model_manager.estimate_tokens(full_output)
            
            if process.returncode != 0:
                err_detail = f"LLAMA CLI exited with code {process.returncode}. Stderr: {full_stderr[:1000]}" # Truncate long stderr
                logger.error(f"[{request.request_id}] {err_detail}")
                self.request_tracker.update_request(request.request_id, status='error', error=err_detail, output=full_output, actual_tokens=actual_gen_tokens)
                return full_output, err_detail # Return output even on error, adapter might use it

            logger.info(f"[{request.request_id}] Non-streaming completed. Output len: {len(full_output)}, Estimated tokens: {actual_gen_tokens}")
            self.request_tracker.update_request(request.request_id, status='completed', output=full_output, progress=request.max_tokens, actual_tokens=actual_gen_tokens)
            return full_output, None

        except subprocess.TimeoutExpired:
            process.kill()
            # Try to get any final output
            try: final_stdout, final_stderr = process.communicate(timeout=5) 
            except: pass # Ignore errors on this final attempt
            else: full_output += final_stdout.strip(); full_stderr += final_stderr.strip()
            
            timeout_err = f"LLAMA CLI process timed out after {communicate_timeout_seconds}s."
            logger.error(f"[{request.request_id}] {timeout_err}")
            self.request_tracker.update_request(request.request_id, status='error', error=timeout_err, output=full_output.strip())
            return full_output.strip(), timeout_err
        except Exception as e_comm:
            logger.exception(f"[{request.request_id}] Exception during communicate (non-stream): {e_comm}")
            err_comm = f"Server error during non-streaming execution: {e_comm}"
            self.request_tracker.update_request(request.request_id, status='error', error=err_comm)
            return "", err_comm
        finally:
            if process.poll() is None: process.kill() # Ensure it's terminated
            if os.path.exists(prompt_file_path):
                try: os.unlink(prompt_file_path)
                except OSError as unlink_err: logger.error(f"[{request.request_id}] Error unlinking temp prompt file {prompt_file_path}: {unlink_err}")


# ================================
# Global Instances
# ================================
model_manager = ModelManager()
request_tracker = RequestTracker()

def find_llama_executable():
    common_names = ['llama-cli', 'main']
    # Preferring paths that indicate a build structure (e.g. /bin/)
    # LLAMA_CPP_DIR is now a Path object
    search_paths = [ LLAMA_CPP_DIR / 'bin' / name for name in common_names ] + \
                   [ LLAMA_CPP_DIR / name for name in common_names ]
    
    for exec_path in search_paths:
        if exec_path.is_file() and os.access(exec_path, os.X_OK):
            logger.info(f"Found llama.cpp executable at: {exec_path}")
            return str(exec_path)
    
    logger.warning(f"Common llama.cpp executable paths not found, starting deep search in {LLAMA_CPP_DIR}...")
    for root, _, files in os.walk(LLAMA_CPP_DIR):
        for file_name in files:
            if file_name in common_names:
                exec_path = Path(root) / file_name
                if exec_path.is_file() and os.access(exec_path, os.X_OK):
                    logger.info(f"Found llama.cpp executable via deep search: {exec_path}")
                    return str(exec_path)
    
    raise RuntimeError(f"Could not find llama.cpp executable (tried {', '.join(common_names)}) in {LLAMA_CPP_DIR} or its subdirectories.")

LLAMA_CLI_PATH: Optional[str] = None
llama_executor: Optional[LLAMAExecutor] = None
try:
    LLAMA_CLI_PATH = find_llama_executable()
    llama_executor = LLAMAExecutor(LLAMA_CLI_PATH, model_manager, request_tracker) # Pass refs
    logger.info(f"Using llama.cpp executable: {LLAMA_CLI_PATH}")
except Exception as e_init:
    logger.critical(f"Failed to initialize LLAMAExecutor: {e_init}", exc_info=True)
    # Server can still start to serve /api/models, etc., but generation will fail.
    # Consider sys.exit(1) if llama.cpp is absolutely essential for any functionality.


# ================================
# API Endpoints & Streaming Logic (Assumed mostly correct, minor logging tweaks)
# ================================
# create_streaming_response and handle_non_streaming_request structure is kept.
# Ensure they correctly use the refactored LLAMAExecutor methods.
def create_streaming_response(adapter: RequestAdapter, request_obj: InternalRequest):
    req_id = request_obj.request_id 
    api_format = request_obj.additional_params.get('api_format', 'unknown')
    logger.info(f"[{req_id}] Creating streaming response for API: {api_format}, Model: {request_obj.model_name}")

    if not llama_executor: # Safety check if executor failed to initialize
        def error_gen(): yield "Error: LLM executor not available."; request_tracker.update_request(req_id, status='error', error="LLM executor not available.")
        return Response(stream_with_context(error_gen()), mimetype='text/plain', status=503)

    def generate_stream_content():
        # ... (streaming logic for different API formats, calling llama_executor.execute_request_streaming)
        # This part is complex and format-specific. Assuming it's largely functional from before.
        # Key is that `llama_executor.execute_request_streaming(request_obj)` is the source of chunks.
        # Example for OpenAI/vLLM Chat SSE:
        if api_format == 'openai' or api_format == 'vllm_chat':
            yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request_obj.model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            for chunk in llama_executor.execute_request_streaming(request_obj): # type: ignore
                if chunk.startswith('Error:'):
                    error_content = chunk.split("Error:", 1)[1].strip()
                    yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': error_content}, 'finish_reason': 'error'}]})}\n\n"
                    break
                yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': chunk}}]})}\n\n"
            else: # After loop finishes (no break)
                 yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
        
        elif api_format == 'ollama_generate':
            for chunk in llama_executor.execute_request_streaming(request_obj): # type: ignore
                if chunk.startswith('Error:'):
                    yield json.dumps({'model': request_obj.model_name, 'response': chunk, 'done': True, 'error': chunk}) + '\n'
                    break
                yield json.dumps({'model': request_obj.model_name, 'created_at': datetime.now().isoformat(), 'response': chunk, 'done': False}) + '\n'
            else:
                final_status = request_tracker.get_request(req_id)
                final_msg = {'model': request_obj.model_name, 'created_at': datetime.now().isoformat(), 'response': '', 'done': True}
                if final_status: final_msg.update({'total_duration': int((final_status.last_update - final_status.start_time) * 1e9), 'eval_count': final_status.actual_tokens})
                yield json.dumps(final_msg) + '\n'
        # ... other formats ...
        else: # Fallback for unstyled stream
            logger.warning(f"[{req_id}] No specific streaming formatter for {api_format}, sending raw chunks.")
            for chunk in llama_executor.execute_request_streaming(request_obj): # type: ignore
                 yield chunk

        # Delayed cleanup for the request tracker entry
        def delayed_tracker_cleanup():
            time.sleep(60) # Keep status for 1 minute after stream ends
            request_tracker.remove_request(req_id)
        threading.Thread(target=delayed_tracker_cleanup, daemon=True).start()

    content_type = 'text/event-stream' # Default for SSE
    if api_format in ['ollama_chat', 'ollama_generate', 'huggingface']:
        content_type = 'application/x-ndjson'
    
    return Response(stream_with_context(generate_stream_content()), mimetype=content_type, headers={'X-Request-ID': req_id})


def handle_non_streaming_request(adapter: RequestAdapter, request_obj: InternalRequest):
    req_id = request_obj.request_id
    api_format = request_obj.additional_params.get('api_format', 'unknown')
    logger.info(f"[{req_id}] Handling non-streaming request. API: {api_format}, Model: {request_obj.model_name}")

    if not llama_executor:
        return jsonify({'error': 'LLM executor not available.'}), 503

    output, error_msg = llama_executor.execute_request_non_streaming(request_obj)
    
    if error_msg:
        logger.error(f"[{req_id}] Non-streaming call to executor failed: {error_msg}. Output (if any): '{output[:100]}...'")
        # API-specific error formatting (simplified here)
        err_resp_payload = {'error': error_msg}
        if api_format == 'claude': err_resp_payload = {'type': 'error', 'error': {'type': 'api_error', 'message': error_msg}}
        return jsonify(err_resp_payload), 500 # Use 500 for server-side execution errors

    logger.info(f"[{req_id}] Non-streaming executor call successful. Output len: {len(output)}. Formatting for {api_format}.")
    response_data = adapter.format_response(output, request_obj)
    
    # HF TGI often returns a list for non-streaming.
    if api_format == 'huggingface' and not isinstance(response_data, list):
        response_data = [response_data]
        
    resp = jsonify(response_data)
    resp.headers['X-Request-ID'] = req_id
    
    # Delayed cleanup
    def delayed_tracker_cleanup():
        time.sleep(60) 
        request_tracker.remove_request(req_id)
    threading.Thread(target=delayed_tracker_cleanup, daemon=True).start()
    
    return resp


@app.route('/api/chat/completions', methods=['POST']) # OpenAI
def chat_completions():
    try:
        adapter = OpenAIAdapter(model_manager) # Or a VLLMAdapter if distinguishing
        request_obj = adapter.parse_request(request.json)
        if request_obj.stream: return create_streaming_response(adapter, request_obj)
        else: return handle_non_streaming_request(adapter, request_obj)
    except ValueError as e: return jsonify({'error': str(e)}), 400
    except Exception as e: logger.exception(f"Unhandled error in /api/chat/completions"); return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/chat', methods=['POST']) # Ollama Chat
def ollama_chat():
    try:
        adapter = OllamaChatAdapter(model_manager)
        request_obj = adapter.parse_request(request.json)
        if request_obj.stream: return create_streaming_response(adapter, request_obj)
        else: return handle_non_streaming_request(adapter, request_obj)
    except ValueError as e: return jsonify({'error': str(e)}), 400
    except Exception as e: logger.exception(f"Unhandled error in /api/chat"); return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/generate', methods=['POST']) # Ollama Generate
def ollama_generate():
    try:
        adapter = OllamaGenerateAdapter(model_manager)
        request_obj = adapter.parse_request(request.json)
        if request_obj.stream: return create_streaming_response(adapter, request_obj)
        else: return handle_non_streaming_request(adapter, request_obj)
    except ValueError as e: return jsonify({'error': str(e)}), 400
    except Exception as e: logger.exception(f"Unhandled error in /api/generate"); return jsonify({'error': 'Internal server error'}), 500

@app.route('/v1/messages', methods=['POST']) # Claude
def claude_messages():
    try:
        adapter = ClaudeAdapter(model_manager)
        request_obj = adapter.parse_request(request.json)
        if request_obj.stream: return create_streaming_response(adapter, request_obj)
        else: return handle_non_streaming_request(adapter, request_obj)
    except ValueError as e: return jsonify({'type': 'error', 'error': {'type': 'invalid_request_error', 'message': str(e)}}), 400
    except Exception as e: logger.exception(f"Unhandled error in /v1/messages"); return jsonify({'type': 'error', 'error': {'type': 'api_error', 'message': 'Internal server error'}}), 500

@app.route('/generate', methods=['POST']) # HF TGI (common path)
@app.route('/hf/v1/completions', methods=['POST']) # More specific path
def huggingface_generate_tgi():
    try:
        adapter = HuggingFaceAdapter(model_manager)
        request_obj = adapter.parse_request(request.json)
        if request_obj.stream: return create_streaming_response(adapter, request_obj)
        else: return handle_non_streaming_request(adapter, request_obj)
    except ValueError as e: return jsonify({'error': str(e)}), 400
    except Exception as e: logger.exception(f"Unhandled error in HuggingFace TGI endpoint"); return jsonify({'error': 'Internal server error'}), 500

@app.route('/v1/chat/completions', methods=['POST']) # vLLM Chat
def vllm_chat_completions():
    try:
        adapter = VLLMAdapter(model_manager)
        request_obj = adapter.parse_request(request.json)
        if request_obj.additional_params.get('api_format') != 'vllm_chat':
             return jsonify({'error': "This endpoint expects 'messages' for vLLM chat. Use /v1/completions for 'prompt'."}), 400
        if request_obj.stream: return create_streaming_response(adapter, request_obj)
        else: return handle_non_streaming_request(adapter, request_obj)
    except ValueError as e: return jsonify({'error': str(e)}), 400
    except Exception as e: logger.exception(f"Unhandled error in /v1/chat/completions (vLLM)"); return jsonify({'error': 'Internal server error'}), 500

@app.route('/v1/completions', methods=['POST']) # vLLM Text Completion
def vllm_completions():
    try:
        adapter = VLLMAdapter(model_manager)
        request_obj = adapter.parse_request(request.json)
        if request_obj.additional_params.get('api_format') != 'vllm_completion':
             return jsonify({'error': "This endpoint expects 'prompt' for vLLM completion. Use /v1/chat/completions for 'messages'."}), 400
        if request_obj.stream: return create_streaming_response(adapter, request_obj)
        else: return handle_non_streaming_request(adapter, request_obj)
    except ValueError as e: return jsonify({'error': str(e)}), 400
    except Exception as e: logger.exception(f"Unhandled error in /v1/completions (vLLM)"); return jsonify({'error': 'Internal server error'}), 500

# Management & Info Endpoints (Assumed mostly correct, structure kept)
# ... /v1/models, /api/tags, /api/models, /info ... (Structure from previous answer)

@app.route('/v1/models', methods=['GET']) # OpenAI/vLLM compatible
@app.route('/api/tags', methods=['GET'])   # Ollama compatible (tags is similar to models list)
@app.route('/api/models', methods=['GET']) # Ollama compatible new route
def list_models_endpoint(): # Renamed function to avoid conflict if Flask auto-names based on path
    try:
        mapping = model_manager.build_model_mapping() # Consider caching this for a short period
        models_list = []
        for model_id_from_manifest in mapping.keys():
            info = model_manager.get_model_info(model_id_from_manifest) # This also uses a cache
            if info:
                if request.path == '/v1/models': # OpenAI /v1/models format
                    models_list.append({
                        "id": info.id, "object": "model", 
                        "created": int(datetime.fromisoformat(info.modified_at.replace("Z", "+00:00")).timestamp()),
                        "owned_by": "organization-owner" # Or "user"
                    })
                else: # Ollama /api/tags or /api/models format
                    models_list.append({
                        "name": info.id, 
                        "model": info.id, 
                        "modified_at": info.modified_at,
                        "size": info.size,
                        "digest": Path(info.path).name.replace("sha256-",""),
                        "details": {
                            "parameter_size": info.parameter_size,
                            "quantization_level": info.quantization,
                            "family": info.name, 
                            "format": "gguf", # Assuming all are GGUF
                            "context_length": info.context_size
                        }
                    })
        
        if request.path == '/v1/models':
            return jsonify({"object": "list", "data": models_list})
        else: # Ollama format
            return jsonify({"models": models_list})
    except Exception as e:
        logger.exception("Error in list_models_endpoint")
        return jsonify({'error': 'Internal server error while listing models'}), 500

@app.route('/info', methods=['GET']) # HuggingFace TGI info
def huggingface_info_endpoint():
    first_model_id = "unknown"
    first_model_info_obj = None
    model_map = model_manager.build_model_mapping() # Cache this?
    if model_map:
        try:
            first_map_id = list(model_map.keys())[0]
            first_model_info_obj = model_manager.get_model_info(first_map_id)
            if first_model_info_obj: first_model_id = first_model_info_obj.id
        except IndexError: pass # No models in map
    
    max_len = first_model_info_obj.context_size if first_model_info_obj else 4096
    return jsonify({
        "model_id": first_model_id,
        "model_dtype": first_model_info_obj.quantization if first_model_info_obj else "unknown",
        "model_device_type": "cuda", # Assuming GPU
        "max_input_length": max_len,
        "max_total_tokens": max_len,
        "version": "custom-llama.cpp-server-1.1.1" # Version of this server
    })


@app.route('/dashboard', methods=['GET'])
def dashboard():
    # The long HTML string. Ensure no `\` immediately followed by ` ` ` causing SyntaxWarning.
    # My previous version of HTML should be mostly fine.
    # The SyntaxWarning might be from the example curl commands at the bottom.
    # For brevity, I'll use a placeholder here. Use the full HTML from your working version.
    # Make sure all JavaScript `\` character are properly handled for Python strings
    # (e.g. `\\` for literal backslash in JS, or just `\${var}` which is fine in Python non-f-string)
    html_content = r"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Universal LLM API Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            :root { --primary: #4a6baf; --primary-light: #6987cc; --secondary: #38b2ac; --dark: #2d3748; --light: #f7fafc; --danger: #e53e3e; --success: #38a169; --warning: #d69e2e; --gray: #718096; --border-color: #e2e8f0; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #f0f4f8; color: var(--dark); font-size:14px; }
            .container { max-width: 1600px; margin: 0 auto; padding: 15px; }
            header { background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%); color: white; padding: 0.8rem 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); position: sticky; top: 0; z-index: 1000; display: flex; justify-content: space-between; align-items: center; }
            header h1 { margin: 0; font-size: 1.3rem; } .header-subtitle { font-size: 0.8rem; opacity: 0.9; margin-top: 3px; }
            .header-actions button { background-color: rgba(255,255,255,0.15); font-size:0.8rem; padding: 6px 12px;} .header-actions button:hover { background-color: rgba(255,255,255,0.25); }
            .stats-bar { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 15px 0; }
            .stat-card { background: white; border-radius: 6px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); border-left: 3px solid var(--primary); }
            .stat-card h3 { margin: 0 0 6px 0; color: var(--gray); font-size: 0.75rem; font-weight: 600; text-transform: uppercase; } .stat-card p { margin: 0; font-size: 1.6rem; font-weight: 700; }
            .request-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 15px; margin-top: 15px; }
            .request-card { background: white; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid var(--border-color); }
            .card-header { padding: 10px 15px; background: #f9fafb; border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; }
            .card-header .card-title { color: var(--primary); } /* Changed title color */
            .card-title { font-weight: 600; font-size: 0.85rem; } .model-badge { background: #eef2ff; color: var(--primary); padding: 3px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 500; }
            .api-indicator { padding: 2px 6px; border-radius: 3px; font-size: 0.6rem; font-weight: 600; text-transform: uppercase; margin-left: 6px; vertical-align: middle; }
            .api-openai { background: #10b981; color: white; } .api-ollama-chat, .api-ollama-generate { background: #f59e0b; color: white; } .api-claude { background: #8b5cf6; color: white; } .api-huggingface { background: #ef4444; color: white; } .api-vllm-chat, .api-vllm-completion { background: #06b6d4; color: white; } .api-unknown { background: var(--gray); color: white; }
            .card-body { padding: 15px; }
            .params-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; font-size: 0.75rem; }
            .param-item { background: #fdfdff; padding: 6px 10px; border-radius: 4px; border: 1px solid #f0f0f5; }
            .param-label { color: var(--gray); font-size: 0.65rem; margin-bottom: 2px; font-weight: 500; } .param-value { font-weight: 500; font-size:0.7rem; word-break: break-all; }
            .progress-container { margin: 12px 0; } .progress-bar { height: 5px; background: var(--border-color); border-radius: 2.5px; overflow: hidden; margin: 5px 0; }
            .progress-fill { height: 100%; border-radius: 2.5px; transition: width 0.3s ease, background-color 0.3s ease; }
            .progress-label { display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--gray); margin-bottom: 2px; }
            .status-indicator { padding: 2px 7px; border-radius: 4px; font-size: 0.65rem; font-weight: 600; text-transform: capitalize; }
            .status-loading { background: #eff6ff; color: #3b82f6; } .status-generating { background: #fefce8; color: #ca8a04; } .status-completed { background: #f0fdf4; color: #16a34a; } .status-error { background: #fef2f2; color: #dc2626; }
            .error-message { color: var(--danger); font-size: 0.7rem; margin-top: 6px; padding: 6px; background: #fff1f1; border: 1px solid #ffd0d0; border-radius: 3px; word-break: break-word; max-height: 100px; overflow-y: auto;}
            .empty-state { grid-column: 1 / -1; text-align: center; padding: 40px 15px; color: var(--gray); } .empty-state svg { width: 50px; height: 50px; margin-bottom: 12px; color: #d1d5db; }
            .empty-state h3 { font-size: 1rem; margin-bottom: 5px; } .empty-state p { font-size: 0.85rem;}
            .loading { opacity: 0.5; pointer-events: none; }
            /* Ensure the rest of your CSS from previous answer is here */
        </style>
        <script>
            // Ensure JS is exactly as from the previous working version, escaping ` correctly for Python string if needed
            // All ` in JS should be literal in Python string. All \${JS_VAR} should be literal \${JS_VAR} in Python string.
            // No \\` unless JS itself needs an escaped backtick.
            // The JS from the previous response should be fine.
            // For brevity, an outline:
            function refreshDashboard() {
                document.getElementById('requests').classList.add('loading');
                fetch('/dashboard/data')
                    .then(res => res.json()).then(updateDashboardData)
                    .catch(err => console.error('Dashboard fetch error:', err))
                    .finally(() => document.getElementById('requests').classList.remove('loading'));
            }
            function updateDashboardData(data) {
                const container = document.getElementById('requests');
                const stats = { active: 0, completed: 0, error: 0, loading: 0, generating: 0 };
                Object.values(data).forEach(req => {
                    if (req.status === 'loading') stats.loading++;
                    else if (req.status === 'generating') stats.generating++;
                    if (req.status === 'loading' || req.status === 'generating') stats.active++;
                    else if (req.status === 'completed') stats.completed++;
                    else if (req.status === 'error') stats.error++;
                });
                document.getElementById('active-count').textContent = stats.active;
                document.getElementById('loading-count').textContent = stats.loading;
                document.getElementById('generating-count').textContent = stats.generating;
                document.getElementById('completed-count').textContent = stats.completed;
                document.getElementById('error-count').textContent = stats.error;
                document.getElementById('total-count').textContent = Object.keys(data).length;

                if (Object.keys(data).length === 0) {
                    container.innerHTML = '<div class="empty-state"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg><h3>No requests</h3><p>API requests will appear here.</p></div>';
                    return;
                }
                container.innerHTML = '';
                Object.entries(data).sort(([,a],[,b]) => b.start_time - a.start_time).forEach(([id, req]) => {
                    const percent = (req.total > 0 && req.progress > 0) ? Math.min(100, Math.floor((req.progress / req.total) * 100)) : 0;
                    const apiFormat = req.options?.additional_params?.api_format || 'unknown';
                    const errorHtml = req.error ? \`<div class="error-message"><strong>Error:</strong> \${escapeHtml(String(req.error).substring(0,250))}\${String(req.error).length > 250 ? '...' : ''}</div>\` : '';
                    const card = document.createElement('div');
                    card.className = 'request-card';
                    card.innerHTML = \`
                        <div class="card-header">
                            <div><span class="card-title">Req \${id.substring(0, 8)}</span><span class="api-indicator api-\${apiFormat.replace(/_/g,'-')}">\${formatApiName(apiFormat)}</span></div>
                            <span class="model-badge">\${req.model || 'N/A'}</span>
                        </div>
                        <div class="card-body">
                            <div class="params-grid">
                                <div class="param-item"><div class="param-label">Status</div><div class="param-value"><span class="status-indicator status-\${req.status}">\${req.status}</span></div></div>
                                <div class="param-item"><div class="param-label">Ctx / Max New</div><div class="param-value">\${req.context_size || 'N/A'} / \${req.options?.max_tokens || req.total || 'N/A'}</div></div>
                                <div class="param-item"><div class="param-label">Elapsed</div><div class="param-value">\${formatTime(req.elapsed_time)}</div></div>
                                <div class="param-item"><div class="param-label">Est. Rem.</div><div class="param-value">\${(req.status === 'completed' || req.status === 'error') ? '-' : formatTime(req.estimated_time_remaining)}</div></div>
                                <div class="param-item"><div class="param-label">Prompt Tokens</div><div class="param-value">\${req.prompt_tokens || 'N/A'}</div></div>
                                <div class="param-item"><div class="param-label">Gen. Tokens</div><div class="param-value">\${req.actual_tokens || 'N/A'}</div></div>
                            </div>
                            \${req.status !== 'loading' ? \`
                            <div class="progress-container">
                                <div class="progress-label"><span>Progress</span><span>\${req.progress} / \${req.total} (\${percent}%)</span></div>
                                <div class="progress-bar"><div class="progress-fill" style="width: \${percent}%; background-color: \${getProgressColor(req.status, percent)};"></div></div>
                            </div>\` : ''}
                            \${errorHtml}
                        </div>\`;
                    container.appendChild(card);
                });
            }
            function getProgressColor(status, percent) { if (status === 'error') return 'var(--danger)'; if (status === 'completed') return 'var(--success)'; if (status === 'loading') return 'var(--primary-light)'; return percent < 30 ? 'var(--warning)' : (percent < 70 ? 'var(--primary)' : 'var(--success)');}
            function formatApiName(apiFormat) { const n={'openai':'OpenAI','ollama_chat':'Ollama C','ollama_generate':'Ollama G','claude':'Claude','huggingface':'HF TGI','vllm_chat':'vLLM C','vllm_completion':'vLLM T'}; return n[apiFormat]||apiFormat; }
            function formatTime(s) { if(s==null||s<0)return 'N/A'; if(s<1)return s.toFixed(2)+'s'; const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sc=Math.floor(s%60); return (h>0?h+'h ':'')+(m>0?m+'m ':'')+(sc>0?sc+'s':'');}
            function escapeHtml(unsafe){ return unsafe.replace(/[&<>"']/g, m=>({'&':'&','<':'<','>':'>','"':'"',"'":'''})[m]); }
            function clearAllCompleted() { fetch('/api/dismiss/completed', { method: 'POST' }).then(refreshDashboard); }
            document.addEventListener('DOMContentLoaded', refreshDashboard);
            // setInterval(refreshDashboard, 15000); // Optional: auto-refresh
        </script>
    </head>
    <body>
        <header>
            <div><h1>🚀 Universal LLM API Dashboard</h1><div class="header-subtitle">llama.cpp backed - Serving OpenAI, Ollama, Claude, HuggingFace & vLLM APIs</div></div>
            <div class="header-actions">
                <button class="btn" onclick="refreshDashboard()">🔄 Refresh</button>
                <button class="btn btn-primary" onclick="clearAllCompleted()">🗑️ Clear Finished</button>
                <button class="btn" onclick="window.open('/', '_blank')">📋 API Info</button>
            </div>
        </header>
        <div class="container">
            <div class="stats-bar">
                <div class="stat-card" style="border-left-color: var(--primary);"><h3>Active</h3><p id="active-count">0</p></div>
                <div class="stat-card" style="border-left-color: #60a5fa;"><h3>Loading</h3><p id="loading-count">0</p></div>
                <div class="stat-card" style="border-left-color: var(--warning);"><h3>Generating</h3><p id="generating-count">0</p></div>
                <div class="stat-card" style="border-left-color: var(--success);"><h3>Completed</h3><p id="completed-count">0</p></div>
                <div class="stat-card" style="border-left-color: var(--danger);"><h3>Errors</h3><p id="error-count">0</p></div>
                <div class="stat-card" style="border-left-color: var(--gray);"><h3>Total Tracked</h3><p id="total-count">0</p></div>
            </div>
            <div class="request-grid" id="requests"></div>
        </div>
    </body></html>
    """ # End of dashboard HTML
    return html_content

@app.route('/dashboard/data', methods=['GET'])
def dashboard_data_endpoint(): # Renamed
    all_requests = request_tracker.get_all_requests()
    response_data = {}
    current_time = time.time()
    for req_id, req_status in all_requests.items():
        data = req_status.__dict__.copy()
        data['elapsed_time'] = current_time - req_status.start_time
        if req_status.status == 'generating' and req_status.progress > 0 and data['elapsed_time'] > 0.1:
            # Progress for streaming is chunk count, for non-streaming it's set to max_tokens upon completion
            # ETR is tricky if progress isn't token-based. For dashboard, it's a rough guide.
            # If req_status.total is max_tokens and req_status.progress is actual tokens generated:
            if req_status.total > 0 : # total is max_tokens
                 # For streaming, progress is chunk count, not token count, so ETR is not accurate
                 # For non-streaming, progress is 0 until done, then total.
                 # This ETR is more of a placeholder for this dashboard structure.
                 tokens_per_second = req_status.actual_tokens / data['elapsed_time'] if req_status.actual_tokens > 0 else 0
                 if tokens_per_second > 0:
                     remaining_tokens = req_status.total - req_status.actual_tokens
                     data['estimated_time_remaining'] = remaining_tokens / tokens_per_second if remaining_tokens > 0 else 0
                 else: data['estimated_time_remaining'] = None
            else: data['estimated_time_remaining'] = None
        else:
            data['estimated_time_remaining'] = None
        response_data[req_id] = data
    return jsonify(response_data)

@app.route('/api/progress/<request_id>', methods=['GET'])
def check_progress(request_id):
    req_status = request_tracker.get_request(request_id)
    if not req_status: return jsonify({'error': 'Request not found'}), 404
    # Similar logic to dashboard_data_endpoint for ETR if needed
    return jsonify(req_status.__dict__)

@app.route('/api/dismiss/<request_id>', methods=['POST'])
def dismiss_request_endpoint(request_id): # Renamed
    request_tracker.remove_request(request_id)
    return jsonify({'success': True, 'message': f'Request {request_id} dismissed.'})

@app.route('/api/dismiss/completed', methods=['POST'])
def dismiss_completed_requests_endpoint(): # Renamed
    # Dismiss finished (completed/error) requests older than, e.g., 5 minutes (300s)
    count = request_tracker.remove_completed(older_than_seconds=300) 
    return jsonify({'success': True, 'dismissed_count': count})

@app.route('/health', methods=['GET'])
def health_check_endpoint(): # Renamed
    active_req_count = len(request_tracker.get_all_requests())
    try: model_map_len = len(model_manager.build_model_mapping()) # Can be slow if uncached
    except Exception: model_map_len = -1 # Indicate error fetching models

    status = 'ok'
    if not LLAMA_CLI_PATH: status = 'error_no_llama_cli'
    elif model_map_len == 0: status = 'warning_no_models_detected'
    elif model_map_len < 0: status = 'error_listing_models'


    return jsonify({
        'status': status,
        'models_available': model_map_len if model_map_len >=0 else "Error",
        'active_requests': active_req_count,
        'llama_executable': LLAMA_CLI_PATH or "Not found",
        'default_tensor_split': args.default_tensor_split or "Not set",
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def api_info_endpoint(): # Renamed
    try: mapping_len = len(model_manager.build_model_mapping())
    except: mapping_len = "Error fetching"
    return jsonify({
        'service': 'Universal LLM API Server (llama.cpp backed)',
        'version': '1.1.1', 
        'total_models_detected': mapping_len,
        # ... other info from previous version ...
        'docs': "See /dashboard and code for endpoint details. Check /health for status.",
    })


# ================================
# Main
# ================================

if __name__ == "__main__":
    logger.info(f"Starting Universal LLM server. Host: {args.host}, Port: {args.port}, Debug: {args.debug}")
    logger.info(f"Model Base Dir: {MODELS_BASE_DIR}, Llama.cpp Build Dir: {LLAMA_CPP_DIR}")
    
    if not llama_executor:
        logger.error("LLAMAExecutor could not be initialized. Generation endpoints will fail.")
    
    # Initial model scan and log
    try:
        mapping = model_manager.build_model_mapping()
        if mapping:
            logger.info(f"Found {len(mapping)} models. Sample (up to 3):")
            for i, model_id in enumerate(list(mapping.keys())[:min(3, len(mapping))]):
                info = model_manager.get_model_info(model_id)
                if info:
                    logger.info(f"  - ID: {info.id}, Path: ...{info.path[-30:]}, Context: {info.context_size}, "
                                f"Size: {(info.size/(1024**3)):.2f}GB, Params: {info.parameter_size}, Quant: {info.quantization}")
                else:
                    logger.warning(f"  - Could not get detailed info for model ID: {model_id}")
        else:
            logger.warning(f"No models found in Ollama structure at {MODELS_BASE_DIR}. Ensure manifests and blobs are correctly placed.")
    except Exception as e_scan:
        logger.exception(f"Error during initial model scan: {e_scan}")

    if not args.debug:
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.setLevel(logging.WARNING)

    logger.info("Server startup complete. Ready for requests.")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True) # threaded=True is default but explicit

# ================================
# Usage Examples (for testing) - Removed problematic backslashes
# ================================

"""
Example API calls for testing (run these in your terminal, not as Python code):

1. OpenAI Format (non-streaming):
curl -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3:latest",
    "messages": [{"role": "user", "content": "Hello! Write a short poem."}],
    "stream": false,
    "max_tokens": 50
  }'

2. Ollama Generate Format (streaming):
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3:latest",
    "prompt": "Why is the sky blue?",
    "stream": true,
    "options": {"num_predict": 100}
  }'

3. Claude Format (non-streaming):
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "llama3:latest",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 30,
    "stream": false
  }'

4. List Models (Ollama format):
curl http://localhost:8000/api/models

5. Health Check:
curl http://localhost:8000/health

6. API Info:
curl http://localhost:8000/

7. Dashboard:
Open http://localhost:8000/dashboard in your web browser
"""
