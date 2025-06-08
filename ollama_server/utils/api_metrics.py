"""
API metrics tracking for real endpoint usage statistics.
Provides real-time metrics instead of simulated data.
"""

import logging
import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class APIEndpointMetrics:
    """Metrics for a specific API endpoint."""
    name: str
    path: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_request_time: Optional[str] = None
    status: str = "unknown"  # healthy, warning, error, unknown
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100.0
    
    @property
    def requests_per_minute(self) -> float:
        # This is simplified - in a real implementation, you'd track time windows
        return min(self.total_requests, 60)  # Cap at 60 for display purposes

class APIMetricsTracker:
    """Tracks real API usage metrics across different endpoints."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.endpoints: Dict[str, APIEndpointMetrics] = {
            "openai": APIEndpointMetrics("OpenAI API", "/v1/chat/completions"),
            "ollama_chat": APIEndpointMetrics("Ollama Chat", "/api/chat"),
            "ollama_generate": APIEndpointMetrics("Ollama Generate", "/api/generate"),
            "vllm_chat": APIEndpointMetrics("vLLM Chat", "/v1/chat/completions"),
            "vllm_completion": APIEndpointMetrics("vLLM Completion", "/v1/completions"),
            "huggingface": APIEndpointMetrics("HuggingFace TGI", "/generate")
        }
        
        # Track recent response times for averaging
        self.recent_response_times: Dict[str, deque] = {
            endpoint: deque(maxlen=100) for endpoint in self.endpoints.keys()
        }
        
        # Track request timestamps for rate calculation
        self.recent_requests: Dict[str, deque] = {
            endpoint: deque(maxlen=100) for endpoint in self.endpoints.keys()
        }
        
    def record_request(self, endpoint_key: str, response_time_ms: float, success: bool = True):
        """Record a request for the specified endpoint."""
        with self._lock:
            if endpoint_key not in self.endpoints:
                logger.warning(f"Unknown endpoint key: {endpoint_key}")
                return
            
            endpoint = self.endpoints[endpoint_key]
            current_time = datetime.now()
            
            # Update basic counters
            endpoint.total_requests += 1
            if success:
                endpoint.successful_requests += 1
            else:
                endpoint.failed_requests += 1
            
            # Update response time tracking
            self.recent_response_times[endpoint_key].append(response_time_ms)
            endpoint.avg_response_time = sum(self.recent_response_times[endpoint_key]) / len(self.recent_response_times[endpoint_key])
            
            # Update request rate tracking
            self.recent_requests[endpoint_key].append(current_time)
            
            # Update last request time
            endpoint.last_request_time = current_time.isoformat()
            
            # Update status based on metrics
            self._update_endpoint_status(endpoint_key)
    
    def _update_endpoint_status(self, endpoint_key: str):
        """Update endpoint status based on recent metrics."""
        endpoint = self.endpoints[endpoint_key]
        
        if endpoint.total_requests == 0:
            endpoint.status = "unknown"
        elif endpoint.error_rate > 20:
            endpoint.status = "error"
        elif endpoint.error_rate > 5 or endpoint.avg_response_time > 5000:
            endpoint.status = "warning"
        else:
            endpoint.status = "healthy"
    
    def get_requests_per_minute(self, endpoint_key: str) -> int:
        """Calculate requests per minute for the last minute."""
        if endpoint_key not in self.recent_requests:
            return 0
        
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent_count = sum(1 for req_time in self.recent_requests[endpoint_key] 
                          if req_time > one_minute_ago)
        return recent_count
    
    def get_metrics_summary(self) -> Dict[str, any]:
        """Get a summary of all endpoint metrics."""
        with self._lock:
            summary = {}
            
            for key, endpoint in self.endpoints.items():
                # Update requests per minute
                rpm = self.get_requests_per_minute(key)
                
                summary[key] = {
                    "name": endpoint.name,
                    "path": endpoint.path,
                    "status": endpoint.status,
                    "total_requests": endpoint.total_requests,
                    "success_rate": round(endpoint.success_rate, 1),
                    "error_rate": round(endpoint.error_rate, 1),
                    "avg_response_time": round(endpoint.avg_response_time, 1),
                    "requests_per_minute": rpm,
                    "last_request_time": endpoint.last_request_time
                }
            
            return summary
    
    def get_endpoint_data_for_charts(self) -> List[Dict[str, any]]:
        """Get endpoint data formatted for dashboard charts."""
        with self._lock:
            endpoints_data = []
            
            for key, endpoint in self.endpoints.items():
                rpm = self.get_requests_per_minute(key)
                
                endpoints_data.append({
                    "name": endpoint.name,
                    "path": endpoint.path,
                    "status": endpoint.status,
                    "responseTime": round(endpoint.avg_response_time) if endpoint.avg_response_time > 0 else 0,
                    "requestsPerMin": rpm,
                    "successRate": round(endpoint.success_rate),
                    "errorRate": round(endpoint.error_rate),
                    "lastCheck": endpoint.last_request_time or datetime.now().isoformat(),
                    "color": self._get_endpoint_color(key)
                })
            
            return endpoints_data
    
    def _get_endpoint_color(self, endpoint_key: str) -> str:
        """Get color for endpoint based on type."""
        colors = {
            "openai": "#10b981",
            "ollama_chat": "#f59e0b", 
            "ollama_generate": "#f59e0b",
            "vllm_chat": "#06b6d4",
            "vllm_completion": "#0ea5e9",
            "huggingface": "#ef4444"
        }
        return colors.get(endpoint_key, "#6b7280")
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            for endpoint in self.endpoints.values():
                endpoint.total_requests = 0
                endpoint.successful_requests = 0
                endpoint.failed_requests = 0
                endpoint.avg_response_time = 0.0
                endpoint.last_request_time = None
                endpoint.status = "unknown"
            
            for deque_list in self.recent_response_times.values():
                deque_list.clear()
            
            for deque_list in self.recent_requests.values():
                deque_list.clear()

# Global metrics tracker instance
api_metrics = APIMetricsTracker()