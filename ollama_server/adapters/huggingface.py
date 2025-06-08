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

"""HuggingFace TGI API format adapter."""
import os
import uuid
from typing import Dict, Any

from .base import RequestAdapter
from ..core.schemas import InternalRequest
from ..models.manager import ModelManager


class HuggingFaceAdapter(RequestAdapter):
    """Adapter for HuggingFace Text Generation Inference (TGI) API format."""
    
    def __init__(self, model_manager: ModelManager, default_tensor_split: str = None):
        super().__init__(model_manager, default_tensor_split)

    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        """Parse HuggingFace TGI request."""
        inputs = data.get('inputs', '')
        model_name_req = data.get('model')  # TGI server usually knows its model
        
        model_name_to_use = model_name_req
        if not model_name_to_use:  # If client doesn't specify, use the smallest available model
            model_name_to_use = self.get_smallest_model()
        
        if not inputs:
            raise ValueError("Missing 'inputs' for HuggingFace TGI.")
            
        model_info = self.model_manager.get_model_info(model_name_to_use)
        if not model_info:
            raise ValueError(f"Model {model_name_to_use} not found.")

        parameters = data.get('parameters', {})
        return InternalRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name_to_use,
            prompt=inputs,
            context_size=model_info.context_size,
            max_tokens=parameters.get('max_new_tokens', 512),
            temperature=parameters.get('temperature', 0.8),
            threads=os.cpu_count() or 4,
            tensor_split=self.default_tensor_split,
            gpu_layers=-1,
            stream=data.get('stream', False),
            additional_params={'api_format': 'huggingface', 'parameters': parameters}
        )

    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format response in HuggingFace TGI format."""
        return {'generated_text': output.strip()}