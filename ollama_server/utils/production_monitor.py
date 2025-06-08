"""
Production monitoring system for LLM Inference Service.

This module provides comprehensive monitoring capabilities for production deployments,
including performance tracking, error monitoring, resource usage analytics, and
automated alerting systems.
"""

import time
import threading
import logging
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    duration_seconds: int = 60  # How long threshold must be exceeded
    enabled: bool = True


@dataclass
class Alert:
    """Production alert."""
    id: str
    alert_type: str  # 'warning', 'critical', 'info'
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: str
    resolved: bool = False
    resolution_timestamp: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: str
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_utilization_avg: float
    gpu_memory_usage_percent: float
    active_requests: int
    completed_requests_per_minute: float
    average_response_time_ms: float
    error_rate_percent: float
    tokens_per_second: float
    disk_usage_percent: float
    network_io_mbps: float


@dataclass
class ProductionSummary:
    """Production system summary."""
    uptime_hours: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    peak_response_time_ms: float
    total_tokens_generated: int
    average_tokens_per_second: float
    peak_concurrent_requests: int
    system_health_score: float
    active_alerts: List[Alert]
    recent_metrics: List[PerformanceMetrics]


class ProductionMonitor:
    """Comprehensive production monitoring system."""
    
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.alerts = {}  # alert_id -> Alert
        self.alert_history = deque(maxlen=1000)
        
        # Performance tracking
        self.request_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.token_counts = deque(maxlen=100)
        self.concurrent_requests_peak = 0
        
        # System state tracking
        self.start_time = datetime.now()
        self.last_metric_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens_generated = 0
        
        # Alert thresholds (configurable)
        self.alert_thresholds = {
            'cpu_usage': AlertThreshold('cpu_usage', 80.0, 95.0, 300),
            'memory_usage': AlertThreshold('memory_usage', 85.0, 95.0, 300),
            'gpu_utilization': AlertThreshold('gpu_utilization', 90.0, 98.0, 180),
            'gpu_memory': AlertThreshold('gpu_memory', 90.0, 98.0, 180),
            'error_rate': AlertThreshold('error_rate', 5.0, 15.0, 120),
            'response_time': AlertThreshold('response_time', 10000.0, 30000.0, 180),
            'disk_usage': AlertThreshold('disk_usage', 80.0, 95.0, 600),
            'active_requests': AlertThreshold('active_requests', 50, 100, 300)
        }
        
        # Thresholds for alert state tracking
        self.threshold_violations = defaultdict(list)
        
        logger.info("Production monitor initialized")
    
    def start_monitoring(self):
        """Start production monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop production monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Production monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                    self._check_alerts(metrics)
                
                time.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Shorter sleep on error
    
    def _collect_metrics(self) -> Optional[PerformanceMetrics]:
        """Collect current system metrics."""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # GPU metrics (if available)
            gpu_utilization = 0.0
            gpu_memory_usage = 0.0
            
            try:
                from .gpu_monitor import gpu_monitor
                gpu_data = gpu_monitor.to_dict()
                if gpu_data and gpu_data.get('gpus'):
                    gpu_utils = [gpu.get('utilization_percent', 0) for gpu in gpu_data['gpus']]
                    gpu_mems = [gpu.get('memory_used_percent', 0) for gpu in gpu_data['gpus']]
                    gpu_utilization = statistics.mean(gpu_utils) if gpu_utils else 0
                    gpu_memory_usage = statistics.mean(gpu_mems) if gpu_mems else 0
            except Exception as e:
                logger.debug(f"GPU metrics not available: {e}")
            
            # Request metrics
            current_time = time.time()
            time_window = current_time - 60  # Last minute
            
            recent_requests = [t for t in self.request_times if t > time_window]
            requests_per_minute = len(recent_requests)
            
            avg_response_time = statistics.mean(recent_requests) if recent_requests else 0
            
            # Error rate calculation
            total_recent = self.total_requests - getattr(self, '_last_total_requests', 0)
            error_recent = sum(count for timestamp, count in self.error_counts.items() 
                             if isinstance(timestamp, float) and timestamp > time_window)
            error_rate = (error_recent / max(total_recent, 1)) * 100 if total_recent > 0 else 0
            
            # Token generation rate
            recent_tokens = [t for t in self.token_counts if t > time_window]
            tokens_per_second = sum(recent_tokens) / 60 if recent_tokens else 0
            
            # Network throughput (rough estimate)
            network_bytes_per_sec = getattr(network, 'bytes_sent', 0) + getattr(network, 'bytes_recv', 0)
            network_mbps = (network_bytes_per_sec / 1024 / 1024) if hasattr(network, 'bytes_sent') else 0
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory.percent,
                gpu_utilization_avg=gpu_utilization,
                gpu_memory_usage_percent=gpu_memory_usage,
                active_requests=self._get_active_requests_count(),
                completed_requests_per_minute=requests_per_minute,
                average_response_time_ms=avg_response_time,
                error_rate_percent=error_rate,
                tokens_per_second=tokens_per_second,
                disk_usage_percent=disk.percent,
                network_io_mbps=network_mbps
            )
            
            self._last_total_requests = self.total_requests
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return None
    
    def _get_active_requests_count(self) -> int:
        """Get current active request count."""
        try:
            # This would be injected by the main application
            if hasattr(self, 'request_tracker'):
                return len(self.request_tracker.get_all_requests())
            return 0
        except:
            return 0
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check metrics against alert thresholds."""
        current_time = time.time()
        
        # Define metric checks
        checks = {
            'cpu_usage': metrics.cpu_usage_percent,
            'memory_usage': metrics.memory_usage_percent,
            'gpu_utilization': metrics.gpu_utilization_avg,
            'gpu_memory': metrics.gpu_memory_usage_percent,
            'error_rate': metrics.error_rate_percent,
            'response_time': metrics.average_response_time_ms,
            'disk_usage': metrics.disk_usage_percent,
            'active_requests': metrics.active_requests
        }
        
        for metric_name, current_value in checks.items():
            threshold = self.alert_thresholds.get(metric_name)
            if not threshold or not threshold.enabled:
                continue
            
            # Check for threshold violations
            alert_type = None
            threshold_value = None
            
            if current_value >= threshold.critical_threshold:
                alert_type = 'critical'
                threshold_value = threshold.critical_threshold
            elif current_value >= threshold.warning_threshold:
                alert_type = 'warning'
                threshold_value = threshold.warning_threshold
            
            if alert_type:
                # Track violation duration
                violations = self.threshold_violations[metric_name]
                violations.append(current_time)
                
                # Remove old violations outside duration window
                cutoff_time = current_time - threshold.duration_seconds
                violations[:] = [t for t in violations if t > cutoff_time]
                
                # Check if we've been in violation long enough
                if len(violations) >= (threshold.duration_seconds / 60):  # Assuming 1-minute intervals
                    alert_id = f"{metric_name}_{alert_type}"
                    
                    if alert_id not in self.alerts or self.alerts[alert_id].resolved:
                        # Create new alert
                        alert = Alert(
                            id=alert_id,
                            alert_type=alert_type,
                            metric_name=metric_name,
                            current_value=current_value,
                            threshold_value=threshold_value,
                            message=f"{metric_name.replace('_', ' ').title()} {alert_type}: {current_value:.1f} exceeds {threshold_value:.1f}",
                            timestamp=datetime.now().isoformat()
                        )
                        
                        self.alerts[alert_id] = alert
                        self.alert_history.append(alert)
                        logger.warning(f"Alert triggered: {alert.message}")
            else:
                # No violation, clear tracking and resolve alerts
                self.threshold_violations[metric_name].clear()
                
                # Resolve any active alerts for this metric
                for alert_type in ['warning', 'critical']:
                    alert_id = f"{metric_name}_{alert_type}"
                    if alert_id in self.alerts and not self.alerts[alert_id].resolved:
                        self.alerts[alert_id].resolved = True
                        self.alerts[alert_id].resolution_timestamp = datetime.now().isoformat()
                        logger.info(f"Alert resolved: {alert_id}")
    
    def record_request(self, response_time_ms: float, success: bool, tokens_generated: int = 0):
        """Record a completed request."""
        current_time = time.time()
        
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.request_times.append(response_time_ms)
        else:
            self.failed_requests += 1
            self.error_counts[current_time] += 1
        
        if tokens_generated > 0:
            self.total_tokens_generated += tokens_generated
            self.token_counts.append(tokens_generated)
    
    def record_concurrent_requests(self, count: int):
        """Record peak concurrent requests."""
        self.concurrent_requests_peak = max(self.concurrent_requests_peak, count)
    
    def get_production_summary(self) -> ProductionSummary:
        """Get comprehensive production summary."""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds() / 3600
            
            # Calculate system health score (0-100)
            health_factors = []
            
            if self.metrics_history:
                latest = self.metrics_history[-1]
                
                # CPU health (lower usage = higher score)
                cpu_score = max(0, 100 - latest.cpu_usage_percent)
                health_factors.append(cpu_score)
                
                # Memory health
                memory_score = max(0, 100 - latest.memory_usage_percent) 
                health_factors.append(memory_score)
                
                # Error rate health
                error_score = max(0, 100 - (latest.error_rate_percent * 10))
                health_factors.append(error_score)
                
                # Response time health (under 5s = 100, over 30s = 0)
                response_score = max(0, min(100, 100 - ((latest.average_response_time_ms - 5000) / 250)))
                health_factors.append(response_score)
            
            # Alert health (no active alerts = 100)
            active_alert_count = len([a for a in self.alerts.values() if not a.resolved])
            alert_score = max(0, 100 - (active_alert_count * 20))
            health_factors.append(alert_score)
            
            system_health_score = statistics.mean(health_factors) if health_factors else 50.0
            
            # Calculate averages
            avg_response_time = statistics.mean(self.request_times) if self.request_times else 0
            peak_response_time = max(self.request_times) if self.request_times else 0
            avg_tokens_per_second = self.total_tokens_generated / max(uptime * 3600, 1)
            
            # Get recent metrics (last 6 hours)
            six_hours_ago = datetime.now() - timedelta(hours=6)
            recent_metrics = [
                m for m in list(self.metrics_history)
                if datetime.fromisoformat(m.timestamp) > six_hours_ago
            ]
            
            # Get active alerts
            active_alerts = [a for a in self.alerts.values() if not a.resolved]
            
            return ProductionSummary(
                uptime_hours=uptime,
                total_requests=self.total_requests,
                successful_requests=self.successful_requests,
                failed_requests=self.failed_requests,
                average_response_time_ms=avg_response_time,
                peak_response_time_ms=peak_response_time,
                total_tokens_generated=self.total_tokens_generated,
                average_tokens_per_second=avg_tokens_per_second,
                peak_concurrent_requests=self.concurrent_requests_peak,
                system_health_score=system_health_score,
                active_alerts=active_alerts,
                recent_metrics=recent_metrics[-60:]  # Last hour
            )
            
        except Exception as e:
            logger.error(f"Error generating production summary: {e}")
            return ProductionSummary(
                uptime_hours=0, total_requests=0, successful_requests=0,
                failed_requests=0, average_response_time_ms=0, peak_response_time_ms=0,
                total_tokens_generated=0, average_tokens_per_second=0,
                peak_concurrent_requests=0, system_health_score=0,
                active_alerts=[], recent_metrics=[]
            )
    
    def get_metrics_for_timeframe(self, hours: int = 24) -> List[PerformanceMetrics]:
        """Get metrics for specified timeframe."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            m for m in list(self.metrics_history)
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
    
    def configure_alert_threshold(self, metric_name: str, warning: float, critical: float, 
                                duration: int = 60, enabled: bool = True):
        """Configure alert threshold for a metric."""
        if metric_name in self.alert_thresholds:
            threshold = self.alert_thresholds[metric_name]
            threshold.warning_threshold = warning
            threshold.critical_threshold = critical
            threshold.duration_seconds = duration
            threshold.enabled = enabled
            logger.info(f"Updated alert threshold for {metric_name}")
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss/acknowledge an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolution_timestamp = datetime.now().isoformat()
            logger.info(f"Alert {alert_id} dismissed manually")
            return True
        return False
    
    def export_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Export metrics data for analysis."""
        metrics = self.get_metrics_for_timeframe(hours)
        summary = self.get_production_summary()
        
        return {
            'export_timestamp': datetime.now().isoformat(),
            'timeframe_hours': hours,
            'summary': asdict(summary),
            'metrics': [asdict(m) for m in metrics],
            'alert_thresholds': {k: asdict(v) for k, v in self.alert_thresholds.items()},
            'alert_history': [asdict(a) for a in list(self.alert_history)[-100:]]  # Last 100 alerts
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert monitor state to dictionary."""
        summary = self.get_production_summary()
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'monitoring_active': self.is_monitoring,
            'summary': asdict(summary),
            'latest_metrics': asdict(latest_metrics) if latest_metrics else None,
            'active_alert_count': len([a for a in self.alerts.values() if not a.resolved]),
            'metrics_history_length': len(self.metrics_history),
            'alert_thresholds_configured': len(self.alert_thresholds)
        }


# Global production monitor instance
production_monitor = ProductionMonitor()