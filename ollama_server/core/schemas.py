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
    completion_time: Optional[float] = None  # Time when request completed
    prompt_eval_duration: Optional[int] = None  # Time spent evaluating prompt (nanoseconds)
    eval_duration: Optional[int] = None  # Time spent generating response (nanoseconds)
    load_duration: Optional[int] = None  # Time spent loading model (nanoseconds)
    
    @property
    def duration(self) -> float:
        """Calculate duration based on completion status"""
        if self.completion_time and self.status in ['completed', 'error']:
            return self.completion_time - self.start_time
        # For active requests, calculate live duration from current time
        import time
        return time.time() - self.start_time
    
    @property
    def total_tokens_per_second(self) -> float:
        """Calculate total tokens per second using Ollama total_duration"""
        # Don't calculate throughput for error requests or very short durations
        if self.status == 'error' or self.duration < 0.1:  # Less than 100ms
            return 0.0
            
        total_duration_seconds = self.duration  # Use overall duration
        total_tokens = self.prompt_tokens + self.actual_tokens  # prompt + generated
        if total_duration_seconds > 0 and total_tokens > 0:
            return total_tokens / total_duration_seconds
        return 0.0
    
    @property
    def generated_tokens_per_second(self) -> float:
        """Calculate generated tokens per second using Ollama eval_duration if available"""
        # Don't calculate throughput for error requests or when no tokens generated
        if self.status == 'error' or self.actual_tokens == 0:
            return 0.0
            
        if self.eval_duration and self.eval_duration > 0 and self.actual_tokens > 0:
            # Use precise Ollama timing: eval_count / eval_duration * 10^9
            return (self.actual_tokens / self.eval_duration) * 1e9
        elif self.duration > 0.1 and self.actual_tokens > 0:  # At least 100ms duration
            # Fallback to duration-based calculation
            return self.actual_tokens / self.duration
        return 0.0