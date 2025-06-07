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

"""Core data structures for the Ollama-compatible server."""
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class InternalRequest:
    """Unified internal request format"""
    request_id: str
    model_name: str
    prompt: str
    context_size: int
    max_tokens: int
    temperature: float
    threads: int
    tensor_split: Optional[str]
    gpu_layers: int
    stream: bool
    additional_params: Dict[str, Any]


@dataclass
class RequestStatus:
    """Request status tracking"""
    request_id: str
    status: str  # loading, generating, completed, error
    progress: int
    total: int
    output: str
    start_time: float
    last_update: float
    model: str
    options: Dict[str, Any]  # Copy of InternalRequest fields
    error: Optional[str] = None
    actual_tokens: int = 0
    context_size: int = 0  # Actual context_size used for the request
    prompt_tokens: int = 0