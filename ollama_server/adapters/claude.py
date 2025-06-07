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

"""Claude API format adapter."""
import os
import uuid
from typing import Dict, List, Any, Optional

from .base import RequestAdapter
from ..core.schemas import InternalRequest
from ..models.manager import ModelManager


class ClaudeAdapter(RequestAdapter):
    """Adapter for Claude/Anthropic API format."""
    
    def __init__(self, model_manager: ModelManager, default_tensor_split: str = None):
        self.model_manager = model_manager
        self.default_tensor_split = default_tensor_split

    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        """Parse Claude API request."""
        messages = data.get('messages', [])
        model_name = data.get('model')
        if not messages or not model_name:
            raise ValueError("Missing messages/model for Claude.")
        
        prompt = self._claude_messages_to_prompt(messages, data.get('system'))
        model_info = self.model_manager.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found.")

        return InternalRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt,
            context_size=model_info.context_size,
            max_tokens=data.get('max_tokens', 512),
            temperature=data.get('temperature', 0.7),
            threads=os.cpu_count() or 4,
            tensor_split=self.default_tensor_split,
            gpu_layers=-1,
            stream=data.get('stream', False),
            additional_params={
                'api_format': 'claude',
                'messages': messages,
                'system': data.get('system')
            }
        )

    def _claude_messages_to_prompt(self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> str:
        """Convert Claude messages to prompt format."""
        prompt_str = ""
        if system_prompt:
            prompt_str += f"{system_prompt}\n\n"
            
        for msg in messages:
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            if role == 'user':
                prompt_str += f"Human: {content}\n\n"
            elif role == 'assistant':
                prompt_str += f"Assistant: {content}\n\n"
        
        prompt_str += "Assistant:"
        return prompt_str.strip()

    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format response in Claude format."""
        return {
            'id': request.request_id,
            'type': 'message',
            'role': 'assistant',
            'content': [{'type': 'text', 'text': output.strip()}],
            'model': request.model_name,
            'stop_reason': 'end_turn',
            'stop_sequence': None,
            'usage': {
                'input_tokens': self.model_manager.estimate_tokens(request.prompt),
                'output_tokens': self.model_manager.estimate_tokens(output)
            }
        }