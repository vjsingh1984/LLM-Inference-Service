"""
GPU monitoring utilities for real-time hardware metrics.
Supports NVIDIA GPUs with nvidia-ml-py integration.
"""

import logging
import subprocess
import json
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class GPUMetrics:
    """GPU metrics data structure."""
    gpu_id: int
    name: str
    utilization_gpu: float  # 0-100%
    utilization_memory: float  # 0-100%
    memory_used: int  # MB
    memory_total: int  # MB
    temperature: int  # Celsius
    power_draw: float  # Watts
    power_limit: float  # Watts
    fan_speed: Optional[int] = None  # 0-100% (if available)
    clock_graphics: Optional[int] = None  # MHz
    clock_memory: Optional[int] = None  # MHz
    pcie_generation: Optional[int] = None
    pcie_width: Optional[int] = None

@dataclass
class SystemMetrics:
    """System-wide metrics."""
    timestamp: str
    gpus: List[GPUMetrics]
    total_memory_used: int
    total_memory_available: int
    driver_version: str
    cuda_version: str

class GPUMonitor:
    """Real-time GPU monitoring with caching for performance."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.last_update = 0
        self.cached_metrics: Optional[SystemMetrics] = None
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Try to initialize nvidia-ml-py
        self.nvidia_ml_available = self._init_nvidia_ml()
        
    def _init_nvidia_ml(self) -> bool:
        """Try to initialize nvidia-ml-py for better performance."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            logger.info("nvidia-ml-py initialized successfully")
            return True
        except ImportError:
            logger.warning("pynvml not available, falling back to nvidia-smi")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize pynvml: {e}, falling back to nvidia-smi")
            return False
    
    def _get_gpu_metrics_nvidia_ml(self) -> List[GPUMetrics]:
        """Get GPU metrics using nvidia-ml-py (faster)."""
        gpus = []
        
        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Basic info
                name = self.pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # Utilization
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_gpu = util.gpu
                util_memory = util.memory
                
                # Memory
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = mem_info.used // (1024 * 1024)  # Convert to MB
                memory_total = mem_info.total // (1024 * 1024)
                
                # Temperature
                try:
                    temperature = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = 0
                
                # Power
                try:
                    power_draw = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    power_limit = self.pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                except:
                    power_draw = 0.0
                    power_limit = 0.0
                
                # Optional metrics
                fan_speed = None
                try:
                    fan_speed = self.pynvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    pass
                
                clock_graphics = None
                clock_memory = None
                try:
                    clock_graphics = self.pynvml.nvmlDeviceGetClockInfo(handle, self.pynvml.NVML_CLOCK_GRAPHICS)
                    clock_memory = self.pynvml.nvmlDeviceGetClockInfo(handle, self.pynvml.NVML_CLOCK_MEM)
                except:
                    pass
                
                # PCIe info
                pcie_generation = None
                pcie_width = None
                try:
                    pcie_info = self.pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
                    pcie_generation = pcie_info
                    pcie_width = self.pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
                except:
                    pass
                
                gpu_metrics = GPUMetrics(
                    gpu_id=i,
                    name=name,
                    utilization_gpu=util_gpu,
                    utilization_memory=util_memory,
                    memory_used=memory_used,
                    memory_total=memory_total,
                    temperature=temperature,
                    power_draw=power_draw,
                    power_limit=power_limit,
                    fan_speed=fan_speed,
                    clock_graphics=clock_graphics,
                    clock_memory=clock_memory,
                    pcie_generation=pcie_generation,
                    pcie_width=pcie_width
                )
                
                gpus.append(gpu_metrics)
                
        except Exception as e:
            logger.error(f"Error getting GPU metrics via nvidia-ml: {e}")
            raise
            
        return gpus
    
    def _get_gpu_metrics_nvidia_smi(self) -> List[GPUMetrics]:
        """Get GPU metrics using nvidia-smi (fallback)."""
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit,fan.speed,clocks.gr,clocks.mem,pcie.link.gen.current,pcie.link.width.current',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                raise Exception(f"nvidia-smi failed: {result.stderr}")
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 14:
                    continue
                
                # Parse values with error handling
                def safe_int(value, default=0):
                    try:
                        return int(float(value)) if value != '[Not Supported]' else default
                    except:
                        return default
                
                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value != '[Not Supported]' else default
                    except:
                        return default
                
                gpu_metrics = GPUMetrics(
                    gpu_id=safe_int(parts[0]),
                    name=parts[1],
                    utilization_gpu=safe_float(parts[2]),
                    utilization_memory=safe_float(parts[3]),
                    memory_used=safe_int(parts[4]),
                    memory_total=safe_int(parts[5]),
                    temperature=safe_int(parts[6]),
                    power_draw=safe_float(parts[7]),
                    power_limit=safe_float(parts[8]),
                    fan_speed=safe_int(parts[9]) if parts[9] != '[Not Supported]' else None,
                    clock_graphics=safe_int(parts[10]) if parts[10] != '[Not Supported]' else None,
                    clock_memory=safe_int(parts[11]) if parts[11] != '[Not Supported]' else None,
                    pcie_generation=safe_int(parts[12]) if parts[12] != '[Not Supported]' else None,
                    pcie_width=safe_int(parts[13]) if parts[13] != '[Not Supported]' else None
                )
                
                gpus.append(gpu_metrics)
            
            return gpus
            
        except Exception as e:
            logger.error(f"Error getting GPU metrics via nvidia-smi: {e}")
            return []
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get NVIDIA driver and CUDA version."""
        driver_version = "Unknown"
        cuda_version = "Unknown"
        
        try:
            # Get driver version
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                driver_version = result.stdout.strip().split('\n')[0]
        except:
            pass
        
        try:
            # Get CUDA version
            if self.nvidia_ml_available:
                cuda_version = self.pynvml.nvmlSystemGetCudaDriverVersion_v2()
                cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
            else:
                result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'release' in line:
                            cuda_version = line.split('release')[1].split(',')[0].strip()
                            break
        except:
            pass
        
        return {
            'driver_version': driver_version,
            'cuda_version': cuda_version
        }
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get current GPU metrics (cached if recent)."""
        current_time = time.time()
        
        with self._lock:
            # Return cached data if recent enough
            if (self.cached_metrics and 
                current_time - self.last_update < self.update_interval):
                return self.cached_metrics
        
        # Update metrics
        try:
            if self.nvidia_ml_available:
                gpus = self._get_gpu_metrics_nvidia_ml()
            else:
                gpus = self._get_gpu_metrics_nvidia_smi()
            
            if not gpus:
                return None
            
            system_info = self._get_system_info()
            
            total_memory_used = sum(gpu.memory_used for gpu in gpus)
            total_memory_available = sum(gpu.memory_total for gpu in gpus)
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                gpus=gpus,
                total_memory_used=total_memory_used,
                total_memory_available=total_memory_available,
                driver_version=system_info['driver_version'],
                cuda_version=system_info['cuda_version']
            )
            
            with self._lock:
                self.cached_metrics = metrics
                self.last_update = current_time
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
            return None
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("GPU monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                self.get_current_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def to_dict(self) -> Optional[Dict[str, Any]]:
        """Convert current metrics to dictionary for JSON serialization."""
        metrics = self.get_current_metrics()
        if metrics:
            return asdict(metrics)
        return None

# Global GPU monitor instance
gpu_monitor = GPUMonitor(update_interval=1.0)