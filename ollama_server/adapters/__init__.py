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

"""API format adapters for different LLM service interfaces."""

from .base import RequestAdapter
from .openai import OpenAIAdapter
from .ollama import OllamaChatAdapter, OllamaGenerateAdapter
from .vllm import VLLMAdapter
from .huggingface import HuggingFaceAdapter

__all__ = [
    'RequestAdapter',
    'OpenAIAdapter',
    'OllamaChatAdapter',
    'OllamaGenerateAdapter',
    'VLLMAdapter',
    'HuggingFaceAdapter',
]