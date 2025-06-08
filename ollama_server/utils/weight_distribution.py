"""
Weight Distribution Visualizer and Tensor Split Configuration Management.
Provides API-level control over GPU weight distribution with real-time validation.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class GPUWeightConfig:
    """GPU weight configuration for tensor splitting."""
    gpu_id: int
    weight_percentage: float
    memory_total: int  # MB
    memory_available: int  # MB
    name: str
    utilization: float = 0.0
    temperature: int = 0
    power_draw: float = 0.0

@dataclass
class TensorSplitConfig:
    """Complete tensor split configuration."""
    gpu_configs: List[GPUWeightConfig]
    total_weight: float
    model_name: Optional[str] = None
    context_size: int = 4096
    batch_size: int = 512
    gpu_layers: int = 999
    created_at: str = ""
    is_valid: bool = True
    validation_errors: List[str] = None

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

class WeightDistributionManager:
    """Manages GPU weight distribution and tensor split configurations."""
    
    def __init__(self):
        self.current_config: Optional[TensorSplitConfig] = None
        self.preset_configs: Dict[str, Dict[str, Any]] = {
            'equal': {
                'name': 'Equal Split',
                'description': 'Distribute weights equally across all GPUs',
                'strategy': 'equal'
            },
            'memory-based': {
                'name': 'Memory Based',
                'description': 'Distribute based on available GPU memory',
                'strategy': 'memory_proportional'
            },
            'performance-based': {
                'name': 'Performance Based',
                'description': 'Distribute based on GPU performance characteristics',
                'strategy': 'performance_weighted'
            },
            'primary-secondary': {
                'name': 'Primary/Secondary',
                'description': 'Favor the most powerful GPU with fallback',
                'strategy': 'primary_secondary'
            }
        }
    
    def create_config_from_gpu_data(self, gpu_data: Dict[str, Any], model_name: Optional[str] = None) -> TensorSplitConfig:
        """Create tensor split configuration from GPU monitoring data."""
        if not gpu_data or 'gpus' not in gpu_data:
            raise ValueError("Invalid GPU data provided")
        
        gpu_configs = []
        for gpu in gpu_data['gpus']:
            gpu_config = GPUWeightConfig(
                gpu_id=gpu['gpu_id'],
                weight_percentage=0.0,  # Will be calculated
                memory_total=gpu['memory_total'],
                memory_available=gpu['memory_total'] - gpu['memory_used'],
                name=gpu['name'],
                utilization=gpu.get('utilization_gpu', 0.0),
                temperature=gpu.get('temperature', 0),
                power_draw=gpu.get('power_draw', 0.0)
            )
            gpu_configs.append(gpu_config)
        
        # Default to equal split
        equal_weight = 100.0 / len(gpu_configs) if gpu_configs else 0.0
        for config in gpu_configs:
            config.weight_percentage = equal_weight
        
        tensor_config = TensorSplitConfig(
            gpu_configs=gpu_configs,
            total_weight=100.0,
            model_name=model_name
        )
        
        self.validate_config(tensor_config)
        return tensor_config
    
    def apply_preset(self, preset_name: str, gpu_data: Dict[str, Any], 
                    model_name: Optional[str] = None) -> TensorSplitConfig:
        """Apply a preset configuration strategy."""
        if preset_name not in self.preset_configs:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        config = self.create_config_from_gpu_data(gpu_data, model_name)
        strategy = self.preset_configs[preset_name]['strategy']
        
        if strategy == 'equal':
            config = self._apply_equal_split(config)
        elif strategy == 'memory_proportional':
            config = self._apply_memory_based_split(config)
        elif strategy == 'performance_weighted':
            config = self._apply_performance_based_split(config)
        elif strategy == 'primary_secondary':
            config = self._apply_primary_secondary_split(config)
        
        self.validate_config(config)
        self.current_config = config
        return config
    
    def _apply_equal_split(self, config: TensorSplitConfig) -> TensorSplitConfig:
        """Apply equal weight distribution."""
        if not config.gpu_configs:
            return config
        
        equal_weight = 100.0 / len(config.gpu_configs)
        for gpu_config in config.gpu_configs:
            gpu_config.weight_percentage = equal_weight
        
        config.total_weight = 100.0
        return config
    
    def _apply_memory_based_split(self, config: TensorSplitConfig) -> TensorSplitConfig:
        """Apply memory-proportional weight distribution."""
        if not config.gpu_configs:
            return config
        
        total_memory = sum(gpu.memory_total for gpu in config.gpu_configs)
        
        for gpu_config in config.gpu_configs:
            gpu_config.weight_percentage = (gpu_config.memory_total / total_memory) * 100.0
        
        config.total_weight = 100.0
        return config
    
    def _apply_performance_based_split(self, config: TensorSplitConfig) -> TensorSplitConfig:
        """Apply performance-weighted distribution (considers memory + utilization)."""
        if not config.gpu_configs:
            return config
        
        # Calculate performance score based on memory and inverse utilization
        scores = []
        for gpu_config in config.gpu_configs:
            # Higher memory is better, lower utilization is better
            memory_score = gpu_config.memory_available
            utilization_penalty = (100 - gpu_config.utilization) / 100.0
            performance_score = memory_score * utilization_penalty
            scores.append(performance_score)
        
        total_score = sum(scores)
        
        for i, gpu_config in enumerate(config.gpu_configs):
            gpu_config.weight_percentage = (scores[i] / total_score) * 100.0 if total_score > 0 else 0.0
        
        config.total_weight = 100.0
        return config
    
    def _apply_primary_secondary_split(self, config: TensorSplitConfig) -> TensorSplitConfig:
        """Apply primary/secondary split (favor the most capable GPU)."""
        if not config.gpu_configs:
            return config
        
        if len(config.gpu_configs) == 1:
            config.gpu_configs[0].weight_percentage = 100.0
        elif len(config.gpu_configs) == 2:
            # For dual GPU setup (like dual RTX 4090), favor the less utilized one
            primary_idx = 0
            secondary_idx = 1
            
            if config.gpu_configs[1].utilization < config.gpu_configs[0].utilization:
                primary_idx = 1
                secondary_idx = 0
            
            config.gpu_configs[primary_idx].weight_percentage = 70.0
            config.gpu_configs[secondary_idx].weight_percentage = 30.0
        else:
            # For 3+ GPUs, use graduated distribution
            sorted_gpus = sorted(config.gpu_configs, 
                               key=lambda x: (x.memory_available, -x.utilization), 
                               reverse=True)
            
            weights = [50.0, 30.0, 15.0, 5.0]  # Primary gets 50%, etc.
            remaining_weight = 100.0
            
            for i, gpu_config in enumerate(sorted_gpus):
                if i < len(weights):
                    gpu_config.weight_percentage = weights[i]
                    remaining_weight -= weights[i]
                else:
                    # Distribute remaining weight equally among remaining GPUs
                    remaining_gpus = len(sorted_gpus) - len(weights)
                    gpu_config.weight_percentage = remaining_weight / remaining_gpus if remaining_gpus > 0 else 0.0
        
        config.total_weight = sum(gpu.weight_percentage for gpu in config.gpu_configs)
        return config
    
    def validate_config(self, config: TensorSplitConfig) -> bool:
        """Validate tensor split configuration and populate validation errors."""
        config.validation_errors = []
        config.is_valid = True
        
        if not config.gpu_configs:
            config.validation_errors.append("No GPU configurations provided")
            config.is_valid = False
            return False
        
        # Check total weight
        total_weight = sum(gpu.weight_percentage for gpu in config.gpu_configs)
        if abs(total_weight - 100.0) > 0.1:
            config.validation_errors.append(f"Total weight must equal 100% (current: {total_weight:.1f}%)")
            config.is_valid = False
        
        # Check individual GPU weights
        for gpu_config in config.gpu_configs:
            if gpu_config.weight_percentage < 0:
                config.validation_errors.append(f"GPU {gpu_config.gpu_id} weight cannot be negative")
                config.is_valid = False
            elif gpu_config.weight_percentage > 100:
                config.validation_errors.append(f"GPU {gpu_config.gpu_id} weight cannot exceed 100%")
                config.is_valid = False
        
        # Check memory availability for estimated model size
        estimated_model_memory = self._estimate_model_memory(config.context_size, config.model_name)
        
        for gpu_config in config.gpu_configs:
            required_memory = (estimated_model_memory * gpu_config.weight_percentage / 100.0)
            if required_memory > gpu_config.memory_available:
                config.validation_errors.append(
                    f"GPU {gpu_config.gpu_id} insufficient memory: "
                    f"need {required_memory:.0f}MB, have {gpu_config.memory_available}MB"
                )
                config.is_valid = False
        
        # Temperature warnings
        for gpu_config in config.gpu_configs:
            if gpu_config.temperature > 85:
                config.validation_errors.append(
                    f"GPU {gpu_config.gpu_id} temperature critical: {gpu_config.temperature}°C"
                )
            elif gpu_config.temperature > 80:
                config.validation_errors.append(
                    f"GPU {gpu_config.gpu_id} temperature high: {gpu_config.temperature}°C"
                )
        
        return config.is_valid
    
    def _estimate_model_memory(self, context_size: int, model_name: Optional[str] = None) -> float:
        """Estimate model memory requirements in MB."""
        base_memory = 1000  # Base overhead
        
        # Context size contribution
        context_memory = (context_size / 1024) * 50  # Rough estimate
        
        # Model size contribution (if we can infer from name)
        model_memory = 4000  # Default for medium model
        if model_name:
            name_lower = model_name.lower()
            if '70b' in name_lower:
                model_memory = 40000
            elif '30b' in name_lower or '33b' in name_lower:
                model_memory = 20000
            elif '13b' in name_lower:
                model_memory = 8000
            elif '7b' in name_lower:
                model_memory = 4000
            elif '3b' in name_lower:
                model_memory = 2000
        
        return base_memory + context_memory + model_memory
    
    def update_weight(self, gpu_id: int, new_weight: float) -> bool:
        """Update weight for a specific GPU and rebalance others."""
        if not self.current_config:
            return False
        
        # Find the target GPU
        target_gpu = None
        for gpu_config in self.current_config.gpu_configs:
            if gpu_config.gpu_id == gpu_id:
                target_gpu = gpu_config
                break
        
        if not target_gpu:
            return False
        
        # Calculate the difference
        old_weight = target_gpu.weight_percentage
        weight_diff = new_weight - old_weight
        
        # Update target GPU
        target_gpu.weight_percentage = new_weight
        
        # Redistribute the difference among other GPUs proportionally
        other_gpus = [gpu for gpu in self.current_config.gpu_configs if gpu.gpu_id != gpu_id]
        if other_gpus and weight_diff != 0:
            total_other_weight = sum(gpu.weight_percentage for gpu in other_gpus)
            
            if total_other_weight > 0:
                for gpu_config in other_gpus:
                    proportion = gpu_config.weight_percentage / total_other_weight
                    gpu_config.weight_percentage -= weight_diff * proportion
                    gpu_config.weight_percentage = max(0, gpu_config.weight_percentage)
        
        # Ensure total is 100%
        total_weight = sum(gpu.weight_percentage for gpu in self.current_config.gpu_configs)
        if total_weight > 0:
            for gpu_config in self.current_config.gpu_configs:
                gpu_config.weight_percentage = (gpu_config.weight_percentage / total_weight) * 100.0
        
        self.validate_config(self.current_config)
        return True
    
    def get_tensor_split_string(self, config: Optional[TensorSplitConfig] = None) -> str:
        """Generate tensor split string for llama.cpp (e.g., '0.5,0.5')."""
        config = config or self.current_config
        if not config or not config.gpu_configs:
            return ""
        
        weights = [gpu.weight_percentage / 100.0 for gpu in config.gpu_configs]
        return ','.join(f"{weight:.3f}" for weight in weights)
    
    def get_config_summary(self, config: Optional[TensorSplitConfig] = None) -> Dict[str, Any]:
        """Get a summary of the current configuration for API responses."""
        config = config or self.current_config
        if not config:
            return {'error': 'No configuration available'}
        
        return {
            'tensor_split': self.get_tensor_split_string(config),
            'gpu_count': len(config.gpu_configs),
            'total_weight': config.total_weight,
            'model_name': config.model_name,
            'context_size': config.context_size,
            'gpu_layers': config.gpu_layers,
            'batch_size': config.batch_size,
            'is_valid': config.is_valid,
            'validation_errors': config.validation_errors,
            'created_at': config.created_at,
            'gpus': [
                {
                    'gpu_id': gpu.gpu_id,
                    'name': gpu.name,
                    'weight_percentage': round(gpu.weight_percentage, 2),
                    'memory_total': gpu.memory_total,
                    'memory_available': gpu.memory_available,
                    'utilization': gpu.utilization,
                    'temperature': gpu.temperature
                }
                for gpu in config.gpu_configs
            ]
        }
    
    def save_config(self, name: str, config: Optional[TensorSplitConfig] = None) -> bool:
        """Save configuration as a custom preset."""
        config = config or self.current_config
        if not config:
            return False
        
        # TODO: Implement persistent storage of custom presets
        logger.info(f"Saving tensor split configuration '{name}': {self.get_tensor_split_string(config)}")
        return True
    
    def to_dict(self, config: Optional[TensorSplitConfig] = None) -> Dict[str, Any]:
        """Convert configuration to dictionary for JSON serialization."""
        config = config or self.current_config
        if not config:
            return {}
        
        return asdict(config)

# Global weight distribution manager instance
weight_manager = WeightDistributionManager()