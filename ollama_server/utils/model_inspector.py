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

"""Model inspection utilities for getting accurate model parameters."""
import logging
import subprocess
import json
import re
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DetailedModelInfo:
    """Extended model information with accurate parameters."""
    architecture: str
    context_length: int
    embedding_length: int
    parameters: str
    quantization: str
    capabilities: list
    stop_tokens: list
    license_info: str


class ModelInspector:
    """Utility for inspecting models and getting accurate parameters."""
    
    def __init__(self, ollama_command: str = "ollama"):
        """Initialize with ollama command path."""
        self.ollama_command = ollama_command
        self._model_cache = {}
    
    def get_ollama_model_info(self, model_name: str) -> Optional[DetailedModelInfo]:
        """Get detailed model info from Ollama CLI."""
        # Check cache first
        if model_name in self._model_cache:
            return self._model_cache[model_name]
        
        try:
            # Run ollama show command
            result = subprocess.run(
                [self.ollama_command, "show", model_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.warning(f"Ollama show failed for {model_name}: {result.stderr}")
                return None
            
            # Parse the output
            model_info = self._parse_ollama_show_output(result.stdout)
            if model_info:
                self._model_cache[model_name] = model_info
                logger.info(f"Successfully cached model info for {model_name}")
            
            return model_info
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout when querying Ollama for model {model_name}")
            return None
        except FileNotFoundError:
            logger.warning(f"Ollama command not found: {self.ollama_command}")
            return None
        except Exception as e:
            logger.exception(f"Error getting Ollama model info for {model_name}: {e}")
            return None
    
    def _parse_ollama_show_output(self, output: str) -> Optional[DetailedModelInfo]:
        """Parse the output from ollama show command."""
        try:
            lines = output.strip().split('\n')
            
            # Initialize default values
            architecture = "llama"
            context_length = 4096  # Default fallback
            embedding_length = 4096
            parameters = "Unknown"
            quantization = "Unknown"
            capabilities = []
            stop_tokens = []
            license_info = ""
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Section headers
                if line == "Model":
                    current_section = "model"
                    continue
                elif line == "Capabilities":
                    current_section = "capabilities"
                    continue
                elif line == "Parameters":
                    current_section = "parameters"
                    continue
                elif line == "License":
                    current_section = "license"
                    continue
                
                # Parse model section
                if current_section == "model":
                    if "architecture" in line:
                        architecture = line.split()[-1]
                    elif "parameters" in line:
                        parameters = line.split()[-1]
                    elif "context length" in line:
                        context_str = line.split()[-1]
                        # Handle numbers like "131072" or "131K"
                        context_length = self._parse_size_string(context_str)
                    elif "embedding length" in line:
                        embedding_str = line.split()[-1]
                        embedding_length = self._parse_size_string(embedding_str)
                    elif "quantization" in line:
                        quantization = line.split()[-1]
                
                # Parse capabilities section
                elif current_section == "capabilities":
                    if line and not line.startswith(" "):
                        capabilities.append(line)
                
                # Parse parameters section (stop tokens)
                elif current_section == "parameters":
                    if "stop" in line:
                        # Extract stop token from quoted string
                        match = re.search(r'"([^"]*)"', line)
                        if match:
                            stop_tokens.append(match.group(1))
                
                # Parse license section
                elif current_section == "license":
                    license_info += line + " "
            
            return DetailedModelInfo(
                architecture=architecture,
                context_length=context_length,
                embedding_length=embedding_length,
                parameters=parameters,
                quantization=quantization,
                capabilities=capabilities,
                stop_tokens=stop_tokens,
                license_info=license_info.strip()
            )
            
        except Exception as e:
            logger.exception(f"Error parsing Ollama show output: {e}")
            return None
    
    def _parse_size_string(self, size_str: str) -> int:
        """Parse size strings like '131072', '131K', '8B', etc."""
        try:
            # Remove any whitespace
            size_str = size_str.strip()
            
            # If it's just a number, return it
            if size_str.isdigit():
                return int(size_str)
            
            # Handle suffixes
            multipliers = {
                'K': 1024,
                'M': 1024 * 1024,
                'B': 1024 * 1024 * 1024,  # For parameters
                'G': 1024 * 1024 * 1024
            }
            
            for suffix, multiplier in multipliers.items():
                if size_str.upper().endswith(suffix):
                    number_part = size_str[:-1]
                    try:
                        number = float(number_part)
                        # For context length, B suffix means billion tokens, not bytes
                        if suffix == 'B' and 'context' in str(size_str).lower():
                            return int(number * 1000000000)
                        else:
                            return int(number * multiplier)
                    except ValueError:
                        break
            
            # If we can't parse it, return a reasonable default
            logger.warning(f"Could not parse size string: {size_str}")
            return 4096
            
        except Exception as e:
            logger.exception(f"Error parsing size string {size_str}: {e}")
            return 4096
    
    def get_enhanced_model_info(self, model_name: str, base_model_info: Any) -> Dict[str, Any]:
        """Get enhanced model info by combining base info with Ollama inspection."""
        # Get detailed info from Ollama
        detailed_info = self.get_ollama_model_info(model_name)
        
        # Start with base model info
        enhanced_info = {
            'id': base_model_info.id,
            'name': base_model_info.name,
            'path': base_model_info.path,
            'size': base_model_info.size,
            'modified_at': base_model_info.modified_at,
            'parameter_size': base_model_info.parameter_size,
            'quantization': base_model_info.quantization,
            'context_size': base_model_info.context_size,
        }
        
        # Enhance with detailed info if available
        if detailed_info:
            enhanced_info.update({
                'architecture': detailed_info.architecture,
                'context_size': detailed_info.context_length,  # Use accurate context length
                'embedding_length': detailed_info.embedding_length,
                'capabilities': detailed_info.capabilities,
                'stop_tokens': detailed_info.stop_tokens,
                'license_info': detailed_info.license_info,
                'quantization': detailed_info.quantization,
                'parameter_size': detailed_info.parameters,
            })
            
            logger.info(f"Enhanced model info for {model_name}: context_size={detailed_info.context_length}")
        else:
            # Fallback: infer architecture from name
            architecture = 'llama'  # Default
            if 'phi' in model_name.lower():
                architecture = 'phi'
            elif 'mixtral' in model_name.lower():
                architecture = 'mixtral'
            elif 'qwen' in model_name.lower():
                architecture = 'qwen'
            
            enhanced_info['architecture'] = architecture
            enhanced_info['capabilities'] = ['completion']
            enhanced_info['stop_tokens'] = []
            enhanced_info['license_info'] = 'Unknown'
        
        return enhanced_info


# Global instance
model_inspector = ModelInspector()