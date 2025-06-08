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

"""Request handlers for the API endpoints."""
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any

from flask import Response, jsonify, stream_with_context

from ..adapters.base import RequestAdapter
from ..core.schemas import InternalRequest
from ..utils.response_processing import preserve_think_tags_in_streaming, ThinkTagProcessor
from ..utils.api_metrics import api_metrics

logger = logging.getLogger(__name__)


class RequestHandler:
    """Handles request processing for different API formats."""
    
    def __init__(self, llama_executor, request_tracker):
        self.llama_executor = llama_executor
        self.request_tracker = request_tracker

    def create_streaming_response(self, adapter: RequestAdapter, request_obj: InternalRequest) -> Response:
        """Create a streaming response for the request."""
        req_id = request_obj.request_id 
        api_format = request_obj.additional_params.get('api_format', 'unknown')
        logger.info(f"[{req_id}] Creating streaming response for API: {api_format}, Model: {request_obj.model_name}")
        
        # Start timing for metrics
        start_time = time.time()

        if not self.llama_executor:
            def error_gen():
                # Record failed request
                response_time = (time.time() - start_time) * 1000
                api_metrics.record_request(api_format, 'failure', response_time)
                yield "Error: LLM executor not available."
                self.request_tracker.update_request(req_id, status='error', error="LLM executor not available.")
            return Response(stream_with_context(error_gen()), mimetype='text/plain', status=503)

        def generate_stream_content():
            """Generate streaming content based on API format."""
            if api_format == 'openai' or api_format == 'vllm_chat':
                # OpenAI/vLLM Chat SSE format
                yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request_obj.model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
                
                for chunk in self.llama_executor.execute_request_streaming(request_obj):
                    if chunk.startswith('Error:'):
                        error_content = chunk.split("Error:", 1)[1].strip()
                        yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': error_content}, 'finish_reason': 'error'}]})}\n\n"
                        break
                    
                    # Process chunk for think tags
                    processed_chunk = preserve_think_tags_in_streaming(chunk, api_format)
                    chunk_data = {'id': req_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': processed_chunk}}]}
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                else:
                    # Stream completed successfully
                    yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                
                yield "data: [DONE]\n\n"
            
            elif api_format == 'ollama_generate':
                # Ollama generate format
                for chunk in self.llama_executor.execute_request_streaming(request_obj):
                    if chunk.startswith('Error:'):
                        yield json.dumps({'model': request_obj.model_name, 'response': chunk, 'done': True, 'error': chunk}) + '\n'
                        break
                    # Preserve think tags for Ollama generate format
                    processed_chunk = preserve_think_tags_in_streaming(chunk, api_format)
                    yield json.dumps({'model': request_obj.model_name, 'created_at': datetime.now().isoformat(), 'response': processed_chunk, 'done': False}) + '\n'
                else:
                    # Stream completed
                    final_status = self.request_tracker.get_request(req_id)
                    final_msg = {'model': request_obj.model_name, 'created_at': datetime.now().isoformat(), 'response': '', 'done': True}
                    if final_status:
                        final_msg.update({
                            'total_duration': int((final_status.last_update - final_status.start_time) * 1e9),
                            'eval_count': final_status.actual_tokens
                        })
                    yield json.dumps(final_msg) + '\n'
            
            elif api_format == 'ollama_chat':
                # Ollama chat format
                for chunk in self.llama_executor.execute_request_streaming(request_obj):
                    if chunk.startswith('Error:'):
                        yield json.dumps({'model': request_obj.model_name, 'message': {'role': 'assistant', 'content': chunk}, 'done': True, 'error': chunk}) + '\n'
                        break
                    # Preserve think tags for Ollama chat format
                    processed_chunk = preserve_think_tags_in_streaming(chunk, api_format)
                    yield json.dumps({'model': request_obj.model_name, 'message': {'role': 'assistant', 'content': processed_chunk}, 'done': False}) + '\n'
                else:
                    yield json.dumps({'model': request_obj.model_name, 'message': {'role': 'assistant', 'content': ''}, 'done': True}) + '\n'
            
            else:
                # Fallback for other formats
                logger.warning(f"[{req_id}] No specific streaming formatter for {api_format}, sending raw chunks.")
                for chunk in self.llama_executor.execute_request_streaming(request_obj):
                    yield chunk

            # Record successful streaming request
            response_time = (time.time() - start_time) * 1000
            api_metrics.record_request(api_format, 'success', response_time)
            
            # Schedule cleanup
            self._schedule_cleanup(req_id)

        # Determine content type based on API format
        content_type = 'text/event-stream'  # Default for SSE
        if api_format in ['ollama_chat', 'ollama_generate', 'huggingface']:
            content_type = 'application/x-ndjson'
        
        return Response(stream_with_context(generate_stream_content()), 
                       mimetype=content_type, 
                       headers={'X-Request-ID': req_id})

    def handle_non_streaming_request(self, adapter: RequestAdapter, request_obj: InternalRequest) -> tuple:
        """Handle non-streaming request."""
        req_id = request_obj.request_id
        api_format = request_obj.additional_params.get('api_format', 'unknown')
        logger.info(f"[{req_id}] Handling non-streaming request. API: {api_format}, Model: {request_obj.model_name}")

        # Start timing for metrics
        start_time = time.time()

        if not self.llama_executor:
            # Record failed request
            response_time = (time.time() - start_time) * 1000
            api_metrics.record_request(api_format, 'failure', response_time)
            return jsonify({'error': 'LLM executor not available.'}), 503

        output, error_msg = self.llama_executor.execute_request_non_streaming(request_obj)
        
        if error_msg:
            logger.error(f"[{req_id}] Non-streaming call to executor failed: {error_msg}. Output (if any): '{output[:100]}...'")
            # Record failed request
            response_time = (time.time() - start_time) * 1000
            api_metrics.record_request(api_format, 'failure', response_time)
            # API-specific error formatting
            err_resp_payload = {'error': error_msg}
            return jsonify(err_resp_payload), 500

        logger.info(f"[{req_id}] Non-streaming executor call successful. Output len: {len(output)}. Formatting for {api_format}.")
        
        # Process output for think tags
        processed_output = ThinkTagProcessor.process_response_with_think_tags(output, api_format)
        response_data = adapter.format_response(processed_output, request_obj)
        
        # Add thinking information if available and appropriate
        response_data = ThinkTagProcessor.format_response_with_thinking(response_data, output, api_format)
        
        # HF TGI often returns a list for non-streaming
        if api_format == 'huggingface' and not isinstance(response_data, list):
            response_data = [response_data]
            
        resp = jsonify(response_data)
        resp.headers['X-Request-ID'] = req_id
        
        # Record successful request
        response_time = (time.time() - start_time) * 1000
        api_metrics.record_request(api_format, 'success', response_time)
        
        # Schedule cleanup
        self._schedule_cleanup(req_id)
        
        return resp

    def _schedule_cleanup(self, req_id: str) -> None:
        """Schedule cleanup of request tracker entry."""
        def delayed_tracker_cleanup():
            time.sleep(60)  # Keep status for 1 minute after completion
            self.request_tracker.remove_request(req_id)
        
        threading.Thread(target=delayed_tracker_cleanup, daemon=True).start()

    def handle_models_list(self, model_manager, api_format: str = 'openai') -> Dict[str, Any]:
        """Handle model listing requests."""
        try:
            model_mapping = model_manager.build_model_mapping()
            model_list = []
            
            for model_id, model_path in model_mapping.items():
                model_info = model_manager.get_model_info(model_id)
                if model_info:
                    if api_format == 'openai':
                        model_entry = {
                            'id': model_info.id,
                            'object': 'model',
                            'created': int(datetime.fromisoformat(model_info.modified_at).timestamp()),
                            'owned_by': 'local',
                            'permission': [],
                            'root': model_info.id,
                            'parent': None
                        }
                    else:  # Ollama format
                        model_entry = {
                            'name': model_info.id,
                            'size': model_info.size,
                            'digest': model_path.split('sha256-')[-1] if 'sha256-' in model_path else 'unknown',
                            'details': {
                                'format': 'gguf',
                                'family': model_info.name,
                                'families': [model_info.name],
                                'parameter_size': model_info.parameter_size,
                                'quantization_level': model_info.quantization
                            },
                            'modified_at': model_info.modified_at
                        }
                    model_list.append(model_entry)
            
            if api_format == 'openai':
                return {'object': 'list', 'data': model_list}
            else:
                return {'models': model_list}
                
        except Exception as e:
            logger.exception("Error listing models")
            return {'error': f'Failed to list models: {str(e)}'}