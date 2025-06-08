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

"""Request tracking functionality."""
import logging
import threading
import time
from typing import Dict, Optional, List
from collections import deque

from .schemas import InternalRequest, RequestStatus

logger = logging.getLogger(__name__)


class RequestTracker:
    """Tracks active requests and their status."""
    
    def __init__(self, model_manager):
        self.active_requests: Dict[str, RequestStatus] = {}
        self.completed_requests: deque = deque(maxlen=10)  # Keep last 10 completed requests
        self._lock = threading.Lock()
        self.model_manager = model_manager

    def add_request(self, request: InternalRequest) -> None:
        """Add a new request to tracking."""
        with self._lock:
            if request.request_id in self.active_requests:
                logger.warning(f"[{request.request_id}] Attempted to add existing request to tracker.")
                return
            
            prompt_toks = self.model_manager.estimate_tokens(request.prompt)

            self.active_requests[request.request_id] = RequestStatus(
                request_id=request.request_id,
                status='loading',
                progress=0,
                total=request.max_tokens,
                output='',
                start_time=time.time(),
                last_update=time.time(),
                model=request.model_name,
                options=request.__dict__.copy(),
                context_size=request.context_size,
                prompt_tokens=prompt_toks
            )
            logger.info(f"[{request.request_id}] Added to tracker. Model: {request.model_name}, "
                       f"Prompt tokens: {prompt_toks}, Max new tokens: {request.max_tokens}")

    def update_request(self, request_id: str, **kwargs) -> None:
        """Update request status."""
        with self._lock:
            if request_id in self.active_requests:
                request_status = self.active_requests[request_id]
                
                # Check if status is being updated to completed or error
                if 'status' in kwargs and kwargs['status'] in ['completed', 'error']:
                    if not request_status.completion_time:
                        kwargs['completion_time'] = time.time()
                
                for key, value in kwargs.items():
                    setattr(request_status, key, value)
                request_status.last_update = time.time()
                
                # If request is now completed/error, move it to completed_requests deque
                if request_status.status in ['completed', 'error']:
                    # Add to completed requests deque (automatically removes oldest if full)
                    self.completed_requests.append(request_status)
                    # Remove from active requests after a short delay
                    # Schedule removal in 3 seconds to allow UI to update
                    threading.Timer(3.0, self._delayed_remove, args=[request_id]).start()
            else:
                logger.warning(f"[{request_id}] Update attempt for non-existent request in tracker.")
    
    def get_request(self, request_id: str) -> Optional[RequestStatus]:
        """Get request status."""
        with self._lock:
            return self.active_requests.get(request_id)
    
    def remove_request(self, request_id: str) -> None:
        """Remove request from tracking."""
        with self._lock:
            if self.active_requests.pop(request_id, None):
                logger.info(f"[{request_id}] Removed from tracker.")
            else:
                logger.debug(f"[{request_id}] Attempt to remove non-existent or already removed request.")
    
    def _delayed_remove(self, request_id: str) -> None:
        """Remove completed request from active tracking after a delay."""
        with self._lock:
            if request_id in self.active_requests:
                request_status = self.active_requests[request_id]
                # Only remove if still in completed/error state
                if request_status.status in ['completed', 'error']:
                    del self.active_requests[request_id]
                    logger.info(f"[{request_id}] Removed from tracker.")
    
    def get_all_requests(self) -> Dict[str, RequestStatus]:
        """Get all active requests."""
        with self._lock:
            return self.active_requests.copy()
    
    def get_recent_requests(self) -> List[RequestStatus]:
        """Get active requests plus last 10 completed requests for display."""
        with self._lock:
            # Combine active requests with recent completed requests
            recent_requests = list(self.active_requests.values())
            recent_requests.extend(list(self.completed_requests))
            
            # Sort by start_time, most recent first
            recent_requests.sort(key=lambda x: x.start_time, reverse=True)
            
            return recent_requests[:20]  # Limit to 20 total for display

    def remove_completed(self, older_than_seconds: int = 300) -> int:
        """Remove completed requests older than specified seconds."""
        removed_count = 0
        with self._lock:
            current_time = time.time()
            to_remove = []
            for req_id, req_status in self.active_requests.items():
                is_finished = req_status.status in ['completed', 'error']
                is_old = (current_time - req_status.last_update) > older_than_seconds
                if is_finished and is_old:
                    to_remove.append(req_id)
            
            for req_id in to_remove:
                # Move to completed_requests before removing if not already there
                req_status = self.active_requests[req_id]
                if req_status not in self.completed_requests:
                    self.completed_requests.append(req_status)
                del self.active_requests[req_id]
                removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} finished requests older than {older_than_seconds}s from tracker.")
        return removed_count