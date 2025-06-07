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

"""Ollama API format adapters."""
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from .base import RequestAdapter
from .openai import OpenAIAdapter
from ..core.schemas import InternalRequest
from ..models.manager import ModelManager


class OllamaChatAdapter(RequestAdapter):
    """Adapter for Ollama chat API format."""
    
    def __init__(self, model_manager: ModelManager, default_tensor_split: str = None):
        super().__init__(model_manager, default_tensor_split)

    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        """Parse Ollama chat request."""
        messages = data.get('messages', [])
        model_name = data.get('model')
        if not messages or not model_name:
            raise ValueError("Missing required fields: messages and model")

        # Reuse OpenAI's message parsing logic
        openai_adapter = OpenAIAdapter(self.model_manager, self.default_tensor_split)
        prompt = openai_adapter._messages_to_prompt(messages, model_name)
        
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found.")

        options = data.get('options', {})
        # Get accurate context size using inspector
        accurate_context_size = self.get_accurate_context_size(model_name)
        context_size_req = options.get('num_ctx', accurate_context_size)
        
        return InternalRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt,
            context_size=min(context_size_req, accurate_context_size),
            max_tokens=options.get('num_predict', 512),
            temperature=options.get('temperature', 0.8),
            threads=options.get('num_thread', os.cpu_count() or 4),
            tensor_split=options.get('tensor_split', self.default_tensor_split),
            gpu_layers=options.get('num_gpu_layers', -1),
            stream=data.get('stream', False),
            additional_params={'api_format': 'ollama_chat', 'messages': messages}
        )

    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format response in Ollama chat format."""
        return {
            'model': request.model_name,
            'created_at': datetime.now().isoformat(),
            'message': {
                'role': 'assistant',
                'content': output.strip()
            },
            'done': True
        }


class OllamaGenerateAdapter(RequestAdapter):
    """Adapter for Ollama generate API format."""
    
    def __init__(self, model_manager: ModelManager, request_tracker, default_tensor_split: str = None):
        super().__init__(model_manager, default_tensor_split)
        self.request_tracker = request_tracker

    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        """Parse Ollama generate request."""
        prompt = data.get('prompt', '')
        model_name = data.get('model')
        if not prompt or not model_name:
            raise ValueError("Missing prompt/model for Ollama generate.")
        
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found.")
            
        options = data.get('options', {})
        # Get accurate context size using inspector
        accurate_context_size = self.get_accurate_context_size(model_name)
        context_size_req = options.get('num_ctx', accurate_context_size)

        return InternalRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt,
            context_size=min(context_size_req, accurate_context_size),
            max_tokens=options.get('num_predict', 512),
            temperature=options.get('temperature', 0.8),
            threads=options.get('num_thread', os.cpu_count() or 4),
            tensor_split=options.get('tensor_split', self.default_tensor_split),
            gpu_layers=options.get('num_gpu_layers', -1),
            stream=data.get('stream', False),
            additional_params={'api_format': 'ollama_generate'}
        )

    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format response in Ollama generate format."""
        final_status = self.request_tracker.get_request(request.request_id)
        resp_dict = {
            'model': request.model_name,
            'created_at': datetime.now().isoformat(),
            'response': output.strip(),
            'done': True
        }
        
        if final_status:
            resp_dict.update({
                'total_duration': int((final_status.last_update - final_status.start_time) * 1e9),
                'prompt_eval_count': final_status.prompt_tokens,
                'eval_count': final_status.actual_tokens,
            })
        
        return resp_dict