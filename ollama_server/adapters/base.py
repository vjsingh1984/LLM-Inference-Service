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

"""Base adapter for request/response format conversion."""
from abc import ABC, abstractmethod
from typing import Dict, Any

from ..core.schemas import InternalRequest
from ..utils.model_inspector import model_inspector


class RequestAdapter(ABC):
    """Base class for converting between different API formats and internal format."""
    
    def __init__(self, model_manager, default_tensor_split=None):
        """Initialize adapter with model manager."""
        self.model_manager = model_manager
        self.default_tensor_split = default_tensor_split
    
    def get_accurate_context_size(self, model_name: str) -> int:
        """Get accurate context size for a model using inspector."""
        try:
            # Get base model info
            model_info = self.model_manager.get_model_info(model_name)
            if not model_info:
                return 4096  # Fallback
            
            # Get enhanced info with accurate context size
            enhanced_info = model_inspector.get_enhanced_model_info(model_name, model_info)
            return enhanced_info.get('context_size', model_info.context_size)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not get accurate context size for {model_name}: {e}")
            return 4096  # Fallback
    
    @abstractmethod
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        """Parse external API request into internal format."""
        pass
    
    @abstractmethod
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format internal response into external API format."""
        pass