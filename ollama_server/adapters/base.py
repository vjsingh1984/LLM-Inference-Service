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
    
    def get_smallest_model(self) -> str:
        """Get the smallest available model for better compatibility."""
        all_models = self.model_manager.build_model_mapping()
        if not all_models:
            raise ValueError("No models available")
        
        # Find the smallest model by looking for size indicators in model names
        # Order: 1b < 1.1b < 1.5b < 2b < 3b < 7b < 8b < 12b < 13b < 14b < 22b < 30b < 33b < 70b < etc.
        size_priorities = {
            '1b': 1, '1.1b': 2, '1.5b': 3, 
            '2b': 4, '3b': 5, '7b': 6, '8b': 7,
            '12b': 8, '13b': 9, '14b': 10, '15b': 11,
            '22b': 12, '27b': 13, '30b': 14, '33b': 15,
            '70b': 16, '235b': 17, '671b': 18
        }
        
        smallest_model = None
        smallest_priority = float('inf')
        
        for model_name in all_models.keys():
            model_lower = model_name.lower()
            found_size = False
            
            # Check for size indicators in the model name
            for size, priority in size_priorities.items():
                if f':{size}' in model_lower or f'-{size}' in model_lower or f'{size}-' in model_lower:
                    if priority < smallest_priority:
                        smallest_priority = priority
                        smallest_model = model_name
                        found_size = True
                    break
            
            # If no size found, assume it's small and give it low priority
            if not found_size and smallest_priority == float('inf'):
                smallest_model = model_name
                smallest_priority = 99
        
        return smallest_model if smallest_model else list(all_models.keys())[0]
    
    @abstractmethod
    def parse_request(self, data: Dict[str, Any]) -> InternalRequest:
        """Parse external API request into internal format."""
        pass
    
    @abstractmethod
    def format_response(self, output: str, request: InternalRequest, streaming: bool = False) -> Dict[str, Any]:
        """Format internal response into external API format."""
        pass