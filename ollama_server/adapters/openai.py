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

"""OpenAI API format adapter."""
import os
import time
import uuid
from typing import Dict, List, Any

from .base import RequestAdapter
from ..core.schemas import InternalRequest
from ..models.manager import ModelManager


class OpenAIAdapter(RequestAdapter):
    """Adapter for OpenAI chat completion API format."""
    
    def __init__(self, model_manager: ModelManager, default_tensor_split: str = None):
        self.model_manager = model_manager
        self.default_tensor_split = default_tensor_split
    
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        """Parse OpenAI chat completion request."""
        messages = data.get('messages', [])
        model_name = data.get('model')
        if not messages or not model_name:
            raise ValueError("Missing required fields: messages and model")
        
        prompt = self._messages_to_prompt(messages, model_name)
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found or info unavailable.")
        
        num_threads = data.get('threads', data.get('n_threads', os.cpu_count() or 4))
        
        return InternalRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt,
            context_size=model_info.context_size,
            max_tokens=data.get('max_tokens', 512),
            temperature=data.get('temperature', 0.8),
            threads=num_threads,
            tensor_split=data.get('tensor_split', self.default_tensor_split),
            gpu_layers=data.get('gpu_layers', data.get('n_gpu_layers', -1)),
            stream=data.get('stream', False),
            additional_params={'api_format': 'openai', 'messages': messages}
        )

    def _messages_to_prompt(self, messages: List[Dict[str, Any]], model_id: str) -> str:
        """Convert messages to prompt format."""
        prompt_str = ""
        is_instruct_model = "instruct" in model_id.lower()

        for msg in messages:
            role = msg.get('role', 'user').lower()
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_str += f"{content}\n\n"
            elif role == 'user':
                prompt_str += f"USER: {content}\n"
            elif role == 'assistant':
                prompt_str += f"ASSISTANT: {content}\n"
        
        prompt_str += "ASSISTANT:"
        return prompt_str.strip()

    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format response in OpenAI format."""
        return {
            'id': request.request_id,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': request.model_name,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': output.strip()
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': self.model_manager.estimate_tokens(request.prompt),
                'completion_tokens': self.model_manager.estimate_tokens(output),
                'total_tokens': self.model_manager.estimate_tokens(request.prompt + output)
            }
        }