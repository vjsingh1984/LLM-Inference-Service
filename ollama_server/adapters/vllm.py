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

"""vLLM API format adapter."""
import os
import time
import uuid
from typing import Dict, Any

from .base import RequestAdapter
from .openai import OpenAIAdapter
from ..core.schemas import InternalRequest
from ..models.manager import ModelManager


class VLLMAdapter(RequestAdapter):
    """Adapter for vLLM API format (supports both chat and completion)."""
    
    def __init__(self, model_manager: ModelManager, default_tensor_split: str = None):
        super().__init__(model_manager, default_tensor_split)

    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        """Parse vLLM request (chat or completion)."""
        model_name = data.get('model')
        
        # If no model specified, use the smallest available model
        if not model_name:
            model_name = self.get_smallest_model()
        
        api_format_type = ''
        prompt_str = ''
        additional_params_base = {'original_payload': data}

        if 'messages' in data:  # Chat style
            messages = data.get('messages', [])
            if not messages:
                raise ValueError("Empty 'messages' for vLLM chat.")
            
            openai_adapter = OpenAIAdapter(self.model_manager, self.default_tensor_split)
            prompt_str = openai_adapter._messages_to_prompt(messages, model_name)
            api_format_type = 'vllm_chat'
            additional_params_base['messages'] = messages
            
        elif 'prompt' in data:  # Completion style
            prompt_str = data.get('prompt', '')
            if not prompt_str:
                raise ValueError("Empty 'prompt' for vLLM completion.")
            api_format_type = 'vllm_completion'
        else:
            raise ValueError("vLLM request must contain 'messages' or 'prompt'.")
        
        additional_params_base['api_format'] = api_format_type
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found.")

        return InternalRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt_str,
            context_size=model_info.context_size,
            max_tokens=data.get('max_tokens', 512),
            temperature=data.get('temperature', 0.8),
            threads=os.cpu_count() or 4,
            tensor_split=self.default_tensor_split,
            gpu_layers=-1,
            stream=data.get('stream', False),
            additional_params=additional_params_base
        )

    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format response in vLLM format."""
        api_format = request.additional_params.get('api_format', 'vllm_completion')
        
        if api_format == 'vllm_chat':
            # Reuse OpenAI format for chat
            openai_adapter = OpenAIAdapter(self.model_manager, self.default_tensor_split)
            return openai_adapter.format_response(output, request, streaming)
        else:  # vLLM completion
            return {
                'id': request.request_id,
                'object': 'text_completion',
                'created': int(time.time()),
                'model': request.model_name,
                'choices': [{
                    'text': output.strip(),
                    'index': 0,
                    'logprobs': None,
                    'finish_reason': 'length'
                }],
                'usage': {
                    'prompt_tokens': self.model_manager.estimate_tokens(request.prompt),
                    'completion_tokens': self.model_manager.estimate_tokens(output),
                    'total_tokens': self.model_manager.estimate_tokens(request.prompt + output)
                }
            }