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

"""LLAMA.cpp execution management."""
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Generator, Tuple, List

from .schemas import InternalRequest
from .request_tracker import RequestTracker
from ..models.manager import ModelManager

logger = logging.getLogger(__name__)


class LLAMAExecutor:
    """Manages execution of llama.cpp for inference."""
    
    def __init__(self, llama_cli_path: str, model_manager: ModelManager, request_tracker: RequestTracker):
        self.llama_cli_path = llama_cli_path
        self.model_manager = model_manager
        self.request_tracker = request_tracker

    def _validate_and_prepare_request(self, request: InternalRequest) -> Optional[str]:
        """Validate request parameters and adjust if needed."""
        model_info = self.model_manager.get_model_info(request.model_name)
        if not model_info:
            return f"Model {request.model_name} not found by ModelManager."

        detected_model_context = model_info.context_size
        if request.context_size <= 0 or request.context_size > detected_model_context:
            logger.warning(
                f"[{request.request_id}] Requested context_size {request.context_size} is invalid or exceeds model limit {detected_model_context}. "
                f"Adjusting to {detected_model_context}."
            )
            request.context_size = detected_model_context
        
        req_status = self.request_tracker.get_request(request.request_id)
        prompt_tokens = req_status.prompt_tokens if req_status else self.model_manager.estimate_tokens(request.prompt)
        
        safety_buffer = 100  # For BOS/EOS, template overhead, etc.
        available_for_generation = request.context_size - prompt_tokens - safety_buffer

        if available_for_generation <= 0:
            return (f"Prompt too long ({prompt_tokens} tokens) for context window ({request.context_size}). "
                   f"Model supports {detected_model_context}. Need ~{prompt_tokens + safety_buffer} for prompt & buffer. "
                   f"Available for generation: {available_for_generation} tokens.")

        if request.max_tokens > available_for_generation:
            original_max_tokens = request.max_tokens
            request.max_tokens = max(10, available_for_generation) 
            logger.warning(
                f"[{request.request_id}] Adjusted max_tokens from {original_max_tokens} to {request.max_tokens} "
                f"to fit within available generation space ({available_for_generation} tokens)."
            )
        
        if req_status:  # Update tracker with potentially adjusted values
            self.request_tracker.update_request(
                request.request_id, options=request.__dict__.copy(),
                context_size=request.context_size, total=request.max_tokens
            )
        return None

    def _execute_shared(self, request: InternalRequest, is_streaming: bool) -> Tuple[Optional[subprocess.Popen], Optional[str], Optional[str]]:
        """Shared logic to build command and start process. Returns (process, prompt_file_path, error_message)"""
        self.request_tracker.add_request(request)
        validation_error = self._validate_and_prepare_request(request)
        if validation_error:
            logger.error(f"[{request.request_id}] Validation failed: {validation_error}")
            self.request_tracker.update_request(request.request_id, status='error', error=validation_error)
            return None, None, validation_error

        model_info = self.model_manager.get_model_info(request.model_name)

        # Create temporary prompt file with UTF-8 encoding
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix=".txt") as temp_prompt_file:
                temp_prompt_file.write(request.prompt)
                prompt_file_path = temp_prompt_file.name
        except Exception as e_tmp:
            err_msg = f"Failed to create temporary prompt file: {e_tmp}"
            logger.exception(f"[{request.request_id}] {err_msg}")
            self.request_tracker.update_request(request.request_id, status='error', error=err_msg)
            return None, None, err_msg

        logger.info(f"[{request.request_id}] Prompt (first 100 chars): '{request.prompt[:100].replace(chr(10), ' ')}...' (File: {prompt_file_path})")
        
        cmd = [
            self.llama_cli_path, "-m", model_info.path, "-f", prompt_file_path,
            "-c", str(request.context_size), "-n", str(request.max_tokens),
            "-t", str(request.threads), "--temp", str(request.temperature),
            "--no-display-prompt",  # Avoids llama.cpp printing the prompt back
            "-mg", "3",
            "-no-cnv",  # Removes conversation mode and exits after processing prompt
            "-ngl", "999"
        ]
        if request.tensor_split:
            cmd.extend(["-ts", request.tensor_split])
        if request.gpu_layers is not None and request.gpu_layers >= 0: 
            cmd.extend(["-ngl", str(request.gpu_layers)])

        logger.info(f"[{request.request_id}] Executing ({'stream' if is_streaming else 'non-stream'}): {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                text=True, bufsize=1, encoding='utf-8', errors='replace'
            )
            self.request_tracker.update_request(request.request_id, status='generating')
            return process, prompt_file_path, None
        except Exception as e_popen:
            err_msg = f"Failed to start llama.cpp process: {e_popen}"
            logger.exception(f"[{request.request_id}] {err_msg}")
            self.request_tracker.update_request(request.request_id, status='error', error=err_msg)
            if os.path.exists(prompt_file_path):
                os.unlink(prompt_file_path)
            return None, prompt_file_path, err_msg

    def execute_request_streaming(self, request: InternalRequest) -> Generator[str, None, None]:
        """Execute request with streaming output."""
        process, prompt_file_path, error_msg = self._execute_shared(request, is_streaming=True)
        if error_msg or not process or not prompt_file_path:
            yield f"Error: {error_msg or 'Process initialization failed'}"
            return

        stderr_lines: List[str] = []
        def _monitor_stderr_stream():
            for line in process.stderr:
                line_strip = line.strip()
                logger.info(f"[{request.request_id}] STDERR (stream): {line_strip}")
                stderr_lines.append(line_strip)
                if len(stderr_lines) > 20:
                    stderr_lines.pop(0)
                if "error" in line_strip.lower() or "failed" in line_strip.lower():
                    if not (self.request_tracker.get_request(request.request_id) or {}).get('error'):
                        self.request_tracker.update_request(request.request_id, status='error', error=f"llama.cpp stderr: {line_strip[:200]}")
        
        stderr_thread = threading.Thread(target=_monitor_stderr_stream, daemon=True)
        stderr_thread.start()

        full_output = ""
        generated_chunk_count = 0
        try:
            for chunk in process.stdout:
                req_stat = self.request_tracker.get_request(request.request_id)
                if req_stat and req_stat.status == 'error':
                    logger.warning(f"[{request.request_id}] Error status from tracker, breaking stdout (stream). Error: {req_stat.error}")
                    break
                
                full_output += chunk
                generated_chunk_count += 1
                self.request_tracker.update_request(request.request_id, progress=generated_chunk_count, output=full_output)
                yield chunk
            
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning(f"[{request.request_id}] Timeout waiting for llama.cpp process to exit after stdout closed (stream). Killing.")
            process.kill()
        except Exception as e_read:
            logger.exception(f"[{request.request_id}] Exception reading stdout (stream): {e_read}")
            self.request_tracker.update_request(request.request_id, status='error', error=f"Stream read error: {e_read}")
            yield f"Error: Stream read error: {e_read}"
        finally:
            stderr_thread.join(timeout=2.0)
            if process.poll() is None:
                process.kill()

            actual_gen_tokens = self.model_manager.estimate_tokens(full_output)
            final_status_obj = self.request_tracker.get_request(request.request_id)

            if final_status_obj and final_status_obj.status != 'error':
                if process.returncode != 0:
                    err_detail = f"LLAMA CLI exited with code {process.returncode}. Last stderr: {' | '.join(stderr_lines)}"
                    logger.error(f"[{request.request_id}] {err_detail}")
                    self.request_tracker.update_request(request.request_id, status='error', error=err_detail, output=full_output, actual_tokens=actual_gen_tokens)
                else:
                    logger.info(f"[{request.request_id}] Streaming completed successfully. Output len: {len(full_output)}, Generated tokens: {actual_gen_tokens}")
                    # Estimate Ollama timing data (in nanoseconds)
                    total_duration_ns = int((final_status_obj.last_update - final_status_obj.start_time) * 1e9)
                    eval_duration_ns = int(total_duration_ns * 0.8)  # Estimate generation took 80% of time
                    prompt_eval_duration_ns = int(total_duration_ns * 0.1)  # Estimate prompt eval took 10%
                    load_duration_ns = int(total_duration_ns * 0.1)  # Estimate load took 10%
                    
                    self.request_tracker.update_request(
                        request.request_id, 
                        status='completed', 
                        output=full_output, 
                        progress=request.max_tokens, 
                        actual_tokens=actual_gen_tokens,
                        eval_duration=eval_duration_ns,
                        prompt_eval_duration=prompt_eval_duration_ns,
                        load_duration=load_duration_ns
                    )
            elif final_status_obj:
                 logger.info(f"[{request.request_id}] Streaming ended (already in error state: {final_status_obj.error}). Output len: {len(full_output)}")
                 self.request_tracker.update_request(request.request_id, output=full_output, actual_tokens=actual_gen_tokens)

            if os.path.exists(prompt_file_path):
                try:
                    os.unlink(prompt_file_path)
                except OSError as unlink_err:
                    logger.error(f"[{request.request_id}] Error unlinking temp prompt file {prompt_file_path}: {unlink_err}")

    def execute_request_non_streaming(self, request: InternalRequest) -> Tuple[str, Optional[str]]:
        """Execute request without streaming."""
        process, prompt_file_path, error_msg = self._execute_shared(request, is_streaming=False)
        if error_msg or not process or not prompt_file_path:
            return "", error_msg or 'Process initialization failed'

        full_output = ""
        full_stderr = ""
        try:
            communicate_timeout_seconds = 30 * 60  # 30 minutes
            logger.info(f"[{request.request_id}] Calling process.communicate(timeout={communicate_timeout_seconds}s) for non-streaming.")
            
            stdout_data, stderr_data = process.communicate(timeout=communicate_timeout_seconds)
            full_output = stdout_data.strip()
            full_stderr = stderr_data.strip()

            if full_stderr: 
                logger.info(f"[{request.request_id}] STDERR (non-stream, first 500): {full_stderr[:500]}")
            
            actual_gen_tokens = self.model_manager.estimate_tokens(full_output)
            
            if process.returncode != 0:
                err_detail = f"LLAMA CLI exited with code {process.returncode}. Stderr: {full_stderr[:500]}"
                logger.error(f"[{request.request_id}] {err_detail}")
                self.request_tracker.update_request(request.request_id, status='error', error=err_detail, output=full_output, actual_tokens=actual_gen_tokens)
                return full_output, err_detail
            else:
                logger.info(f"[{request.request_id}] Non-streaming completed successfully. Output len: {len(full_output)}, Generated tokens: {actual_gen_tokens}")
                # Get final status for timing calculation
                final_status_obj = self.request_tracker.get_request(request.request_id)
                if final_status_obj:
                    # Estimate Ollama timing data (in nanoseconds)
                    total_duration_ns = int((time.time() - final_status_obj.start_time) * 1e9)
                    eval_duration_ns = int(total_duration_ns * 0.8)  # Estimate generation took 80% of time
                    prompt_eval_duration_ns = int(total_duration_ns * 0.1)  # Estimate prompt eval took 10%
                    load_duration_ns = int(total_duration_ns * 0.1)  # Estimate load took 10%
                    
                    self.request_tracker.update_request(
                        request.request_id, 
                        status='completed', 
                        output=full_output, 
                        progress=request.max_tokens, 
                        actual_tokens=actual_gen_tokens,
                        eval_duration=eval_duration_ns,
                        prompt_eval_duration=prompt_eval_duration_ns,
                        load_duration=load_duration_ns
                    )
                else:
                    self.request_tracker.update_request(request.request_id, status='completed', output=full_output, progress=request.max_tokens, actual_tokens=actual_gen_tokens)
                return full_output, None
        except subprocess.TimeoutExpired as e_timeout:
            logger.error(f"[{request.request_id}] Timeout during process.communicate(): {e_timeout}")
            process.kill()
            self.request_tracker.update_request(request.request_id, status='error', error=f"Timeout: {e_timeout}")
            return "", f"Timeout: {e_timeout}"
        except Exception as e_comm:
            logger.exception(f"[{request.request_id}] Exception during process.communicate(): {e_comm}")
            self.request_tracker.update_request(request.request_id, status='error', error=f"Communication error: {e_comm}")
            return "", f"Communication error: {e_comm}"
        finally:
            if os.path.exists(prompt_file_path):
                try:
                    os.unlink(prompt_file_path)
                except OSError as unlink_err:
                    logger.error(f"[{request.request_id}] Error unlinking temp prompt file {prompt_file_path}: {unlink_err}")


def find_llama_executable(llama_cpp_dir: Path) -> str:
    """Find the llama.cpp executable in the build directory."""
    common_names = ['llama-cli', 'llama.cpp', 'main', 'llama']
    search_dirs = [
        llama_cpp_dir,
        llama_cpp_dir / 'bin', 
        llama_cpp_dir / 'build/bin', 
        llama_cpp_dir / 'examples',
        llama_cpp_dir / 'build'
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for exe_name in common_names:
            exe_path = search_dir / exe_name
            if exe_path.is_file() and os.access(exe_path, os.X_OK):
                logger.info(f"Found llama.cpp executable: {exe_path}")
                return str(exe_path)
    
    # Deep search if not found in common locations
    logger.warning(f"Common llama.cpp executable paths not found, starting deep search in {llama_cpp_dir}...")
    for root, _, files in os.walk(llama_cpp_dir):
        for file_name in files:
            if file_name in common_names:
                exec_path = Path(root) / file_name
                if exec_path.is_file() and os.access(exec_path, os.X_OK):
                    logger.info(f"Found llama.cpp executable via deep search: {exec_path}")
                    return str(exec_path)
    
    raise RuntimeError(f"Could not find llama.cpp executable (tried {', '.join(common_names)}) in {llama_cpp_dir} or its subdirectories.")