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

"""Configuration management for the Ollama-compatible server."""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ServerConfig:
    """Server configuration parameters."""
    model_dir: Path
    llama_cpp_dir: Path
    log_dir: Path
    port: int
    host: str
    debug: bool
    default_tensor_split: Optional[str]
    
    @property
    def models_base_dir(self) -> Path:
        """Base directory for models."""
        return Path(self.model_dir)
    
    @property
    def models_dir(self) -> Path:
        """Directory containing model blobs."""
        return self.models_base_dir / 'blobs'
    
    @property
    def manifests_dir(self) -> Path:
        """Directory containing Ollama manifests."""
        return self.models_base_dir / 'manifests/registry.ollama.ai/library'


def parse_arguments() -> ServerConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run an Ollama-compatible API server with GPU sharding')
    parser.add_argument('--model-dir', type=str, default='/opt/llm/models/ollama/models',
                        help='Path to the Ollama models directory (default: /opt/llm/models/ollama/models)')
    parser.add_argument('--llama-cpp-dir', type=str, default='/opt/llm/models/ollama-custom-models/llama.cpp/build',
                        help='Path to the llama.cpp build directory (default: /opt/llm/models/ollama-custom-models/llama.cpp/build)')
    parser.add_argument('--log-dir', type=str, default='/opt/llm/inference-service/logs',
                        help='Path to the logs directory (default: /opt/llm/inference-service/logs)')
    parser.add_argument('--port', type=int, default=11435,
                        help='Port to run the API server on (default: 11435)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the API server on (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Run the API server in debug mode')
    parser.add_argument('--default-tensor-split', type=str, default='0.25,0.25,0.25,0.25',
                        help='Default tensor split for GPU sharding (default: 0.25,0.25,0.25,0.25). Set to None or empty to disable.')

    args = parser.parse_args()
    
    # Handle tensor split special cases
    if args.default_tensor_split and args.default_tensor_split.lower() in ['none', '']:
        args.default_tensor_split = None
    
    return ServerConfig(
        model_dir=Path(args.model_dir),
        llama_cpp_dir=Path(args.llama_cpp_dir),
        log_dir=Path(args.log_dir),
        port=args.port,
        host=args.host,
        debug=args.debug,
        default_tensor_split=args.default_tensor_split
    )