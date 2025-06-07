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

"""Response processing utilities including think tag handling."""
import re
from typing import Dict, Any, Tuple


class ThinkTagProcessor:
    """Handles processing of think tags in model responses."""
    
    @staticmethod
    def extract_think_content(text: str) -> Tuple[str, str]:
        """Extract think content and clean response.
        
        Args:
            text: Raw response text that may contain think tags
            
        Returns:
            Tuple of (think_content, clean_response)
        """
        # Pattern to match <think>...</think> tags
        think_pattern = r'<think>(.*?)</think>'
        
        # Extract all think content
        think_matches = re.findall(think_pattern, text, re.DOTALL)
        think_content = '\n'.join(think_matches) if think_matches else ''
        
        # Remove think tags from the response
        clean_response = re.sub(think_pattern, '', text, flags=re.DOTALL)
        
        # Clean up extra whitespace
        clean_response = re.sub(r'\n\s*\n', '\n', clean_response).strip()
        
        return think_content, clean_response
    
    @staticmethod
    def should_preserve_think_tags(api_format: str, preserve_think: bool = True) -> bool:
        """Determine if think tags should be preserved based on API format.
        
        Args:
            api_format: The API format being used
            preserve_think: Global setting for think tag preservation
            
        Returns:
            True if think tags should be preserved
        """
        if not preserve_think:
            return False
            
        # Preserve think tags for Ollama format by default
        # Other formats may want clean responses
        return api_format in ['ollama_generate', 'ollama_chat']
    
    @staticmethod
    def process_response_with_think_tags(text: str, api_format: str, preserve_think: bool = True) -> str:
        """Process response text handling think tags appropriately.
        
        Args:
            text: Raw response text
            api_format: API format being used
            preserve_think: Whether to preserve think tags
            
        Returns:
            Processed response text
        """
        if not preserve_think or not ThinkTagProcessor.should_preserve_think_tags(api_format, preserve_think):
            # Remove think tags and return clean response
            _, clean_response = ThinkTagProcessor.extract_think_content(text)
            return clean_response
        
        # Preserve the original text with think tags
        return text
    
    @staticmethod
    def format_response_with_thinking(response_data: Dict[str, Any], raw_output: str, api_format: str) -> Dict[str, Any]:
        """Format response including thinking information if appropriate.
        
        Args:
            response_data: Base response data
            raw_output: Raw model output
            api_format: API format being used
            
        Returns:
            Enhanced response data with thinking information
        """
        think_content, clean_response = ThinkTagProcessor.extract_think_content(raw_output)
        
        if think_content and api_format in ['ollama_generate', 'ollama_chat']:
            # Add thinking information to Ollama responses
            if api_format == 'ollama_generate':
                response_data['thinking'] = think_content
            elif api_format == 'ollama_chat':
                if 'message' in response_data:
                    response_data['message']['thinking'] = think_content
        
        return response_data


def preserve_think_tags_in_streaming(chunk: str, api_format: str) -> str:
    """Process streaming chunk while preserving think tags if appropriate.
    
    Args:
        chunk: Streaming response chunk
        api_format: API format being used
        
    Returns:
        Processed chunk
    """
    # For Ollama formats, preserve think tags in streaming
    if api_format in ['ollama_generate', 'ollama_chat']:
        return chunk
    
    # For other formats, you might want to filter out think tags
    # but preserve the content between them
    if '<think>' in chunk or '</think>' in chunk:
        # For OpenAI/other formats, we might want to stream the thinking
        # content but not the tags themselves
        chunk = chunk.replace('<think>', '').replace('</think>', '')
    
    return chunk