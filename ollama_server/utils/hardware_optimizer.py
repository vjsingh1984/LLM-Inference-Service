"""
Hardware optimization insights and recommendations for LLM inference.
Analyzes GPU configuration, model requirements, and system performance to provide
intelligent optimization suggestions.
"""

import logging
import psutil
import platform
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class OptimizationRecommendation:
    """Single optimization recommendation."""
    category: str  # 'performance', 'memory', 'cost', 'stability'
    priority: str  # 'high', 'medium', 'low'
    title: str
    description: str
    action: str
    impact: str  # Expected impact description
    implementation_difficulty: str  # 'easy', 'moderate', 'difficult'
    estimated_improvement: str  # e.g., "15-25% faster inference"

@dataclass
class SystemAnalysis:
    """Complete system analysis for optimization."""
    timestamp: str
    cpu_analysis: Dict[str, Any]
    memory_analysis: Dict[str, Any]
    gpu_analysis: Dict[str, Any]
    model_analysis: Dict[str, Any]
    recommendations: List[OptimizationRecommendation]
    overall_score: float  # 0-100 optimization score

class HardwareOptimizer:
    """Analyzes hardware configuration and provides optimization insights."""
    
    def __init__(self, gpu_monitor=None, model_manager=None):
        self.gpu_monitor = gpu_monitor
        self.model_manager = model_manager
        
    def analyze_system(self, current_config: Dict[str, Any] = None) -> SystemAnalysis:
        """Perform comprehensive system analysis."""
        try:
            # Gather system information
            cpu_analysis = self._analyze_cpu()
            memory_analysis = self._analyze_memory()
            gpu_analysis = self._analyze_gpu(current_config)
            model_analysis = self._analyze_models()
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                cpu_analysis, memory_analysis, gpu_analysis, model_analysis, current_config
            )
            
            # Calculate overall optimization score
            overall_score = self._calculate_optimization_score(
                cpu_analysis, memory_analysis, gpu_analysis, model_analysis
            )
            
            return SystemAnalysis(
                timestamp=datetime.now().isoformat(),
                cpu_analysis=cpu_analysis,
                memory_analysis=memory_analysis,
                gpu_analysis=gpu_analysis,
                model_analysis=model_analysis,
                recommendations=recommendations,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.exception("Error analyzing system")
            return self._create_fallback_analysis()
    
    def _analyze_cpu(self) -> Dict[str, Any]:
        """Analyze CPU configuration and performance."""
        try:
            cpu_count = psutil.cpu_count(logical=True)
            cpu_count_physical = psutil.cpu_count(logical=False)
            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get CPU architecture info
            try:
                import cpuinfo
                cpu_info = cpuinfo.get_cpu_info()
                cpu_brand = cpu_info.get('brand_raw', 'Unknown')
                cpu_arch = cpu_info.get('arch', 'Unknown')
            except ImportError:
                cpu_brand = platform.processor()
                cpu_arch = platform.machine()
            
            return {
                'logical_cores': cpu_count,
                'physical_cores': cpu_count_physical,
                'current_frequency_mhz': cpu_freq.current if cpu_freq else 0,
                'max_frequency_mhz': cpu_freq.max if cpu_freq else 0,
                'current_usage_percent': cpu_percent,
                'brand': cpu_brand,
                'architecture': cpu_arch,
                'hyperthreading_available': cpu_count > cpu_count_physical,
                'performance_rating': self._rate_cpu_performance(cpu_count, cpu_freq, cpu_brand)
            }
        except Exception as e:
            logger.warning(f"Error analyzing CPU: {e}")
            return {'error': str(e), 'performance_rating': 'unknown'}
    
    def _analyze_memory(self) -> Dict[str, Any]:
        """Analyze system memory configuration."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'usage_percent': memory.percent,
                'swap_total_gb': round(swap.total / (1024**3), 2),
                'swap_used_gb': round(swap.used / (1024**3), 2),
                'memory_pressure': 'high' if memory.percent > 85 else 'medium' if memory.percent > 70 else 'low',
                'adequacy_for_llm': self._assess_memory_adequacy(memory.total)
            }
        except Exception as e:
            logger.warning(f"Error analyzing memory: {e}")
            return {'error': str(e)}
    
    def _analyze_gpu(self, current_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze GPU configuration and utilization."""
        try:
            if not self.gpu_monitor:
                return {'error': 'GPU monitor not available'}
            
            gpu_metrics = self.gpu_monitor.get_current_metrics()
            if not gpu_metrics or not gpu_metrics.gpus:
                return {'error': 'No GPU data available'}
            
            gpus = gpu_metrics.gpus
            tensor_split = current_config.get('tensor_split', '0.25,0.25,0.25,0.25') if current_config else '0.25,0.25,0.25,0.25'
            splits = [float(x.strip()) for x in tensor_split.split(',')]
            
            # Analyze each GPU
            gpu_details = []
            total_vram = 0
            total_used_vram = 0
            max_temp = 0
            avg_utilization = 0
            
            for i, gpu in enumerate(gpus):
                allocation = splits[i] if i < len(splits) else 0
                gpu_details.append({
                    'id': gpu.gpu_id,
                    'name': gpu.name,
                    'memory_total_gb': round(gpu.memory_total / 1024, 2),
                    'memory_used_gb': round(gpu.memory_used / 1024, 2),
                    'memory_utilization_percent': round((gpu.memory_used / gpu.memory_total) * 100, 1),
                    'gpu_utilization_percent': gpu.utilization_gpu,
                    'temperature_c': gpu.temperature,
                    'tensor_allocation_percent': round(allocation * 100, 1),
                    'power_draw_watts': gpu.power_draw,
                    'power_limit_watts': gpu.power_limit,
                    'power_efficiency': round((gpu.power_draw / gpu.power_limit) * 100, 1) if gpu.power_limit > 0 else 0,
                    'thermal_status': 'critical' if gpu.temperature > 85 else 'warning' if gpu.temperature > 75 else 'good'
                })
                
                total_vram += gpu.memory_total
                total_used_vram += gpu.memory_used
                max_temp = max(max_temp, gpu.temperature)
                avg_utilization += gpu.utilization_gpu
            
            avg_utilization = avg_utilization / len(gpus) if gpus else 0
            
            # Assess GPU configuration quality
            config_quality = self._assess_gpu_configuration(gpus, splits)
            
            return {
                'gpu_count': len(gpus),
                'total_vram_gb': round(total_vram / 1024, 2),
                'total_used_vram_gb': round(total_used_vram / 1024, 2),
                'vram_utilization_percent': round((total_used_vram / total_vram) * 100, 1) if total_vram > 0 else 0,
                'average_gpu_utilization': round(avg_utilization, 1),
                'max_temperature': max_temp,
                'thermal_status': 'critical' if max_temp > 85 else 'warning' if max_temp > 75 else 'good',
                'current_tensor_split': tensor_split,
                'gpu_details': gpu_details,
                'configuration_quality': config_quality,
                'driver_version': gpu_metrics.driver_version,
                'cuda_version': gpu_metrics.cuda_version
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing GPU: {e}")
            return {'error': str(e)}
    
    def _analyze_models(self) -> Dict[str, Any]:
        """Analyze model inventory and requirements."""
        try:
            if not self.model_manager:
                return {'error': 'Model manager not available'}
            
            model_mapping = self.model_manager.build_model_mapping()
            
            # Categorize models by size and context
            model_categories = {
                'small': [],  # < 7B parameters
                'medium': [],  # 7B - 30B parameters  
                'large': [],  # 30B - 70B parameters
                'xlarge': []  # > 70B parameters
            }
            
            context_sizes = []
            total_models = len(model_mapping)
            
            for model_id, model_path in model_mapping.items():
                try:
                    model_info = self.model_manager.get_model_info(model_id)
                    if model_info:
                        # Categorize by parameter size
                        param_size = model_info.parameter_size.lower()
                        if 'b' in param_size:
                            param_num = float(param_size.replace('b', ''))
                            if param_num < 7:
                                model_categories['small'].append(model_id)
                            elif param_num < 30:
                                model_categories['medium'].append(model_id)
                            elif param_num < 70:
                                model_categories['large'].append(model_id)
                            else:
                                model_categories['xlarge'].append(model_id)
                        
                        # Track context sizes
                        context_size = self.model_manager.get_context_size(model_id)
                        if context_size:
                            context_sizes.append(context_size)
                            
                except Exception as e:
                    logger.debug(f"Error analyzing model {model_id}: {e}")
                    continue
            
            avg_context_size = sum(context_sizes) / len(context_sizes) if context_sizes else 0
            max_context_size = max(context_sizes) if context_sizes else 0
            
            # Calculate model diversity score
            diversity_score = self._calculate_model_diversity(model_categories)
            
            return {
                'total_models': total_models,
                'model_categories': {k: len(v) for k, v in model_categories.items()},
                'model_examples': {k: v[:3] for k, v in model_categories.items()},  # First 3 of each category
                'average_context_size': round(avg_context_size),
                'max_context_size': max_context_size,
                'diversity_score': diversity_score,
                'optimization_potential': self._assess_model_optimization_potential(model_categories, context_sizes)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing models: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, cpu_analysis: Dict, memory_analysis: Dict, 
                                gpu_analysis: Dict, model_analysis: Dict, 
                                current_config: Dict = None) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # GPU recommendations
        if gpu_analysis.get('gpu_count', 0) > 0:
            recommendations.extend(self._gpu_recommendations(gpu_analysis, current_config))
        
        # Memory recommendations
        recommendations.extend(self._memory_recommendations(memory_analysis, model_analysis))
        
        # CPU recommendations
        recommendations.extend(self._cpu_recommendations(cpu_analysis, current_config))
        
        # Model-specific recommendations
        recommendations.extend(self._model_recommendations(model_analysis, gpu_analysis))
        
        # Configuration recommendations
        if current_config:
            recommendations.extend(self._config_recommendations(current_config, gpu_analysis))
        
        # Sort by priority and impact
        return sorted(recommendations, key=lambda x: (
            {'high': 0, 'medium': 1, 'low': 2}[x.priority],
            {'difficult': 2, 'moderate': 1, 'easy': 0}[x.implementation_difficulty]
        ))
    
    def _gpu_recommendations(self, gpu_analysis: Dict, current_config: Dict = None) -> List[OptimizationRecommendation]:
        """Generate GPU-specific recommendations."""
        recommendations = []
        
        # Check thermal issues
        if gpu_analysis.get('thermal_status') == 'critical':
            recommendations.append(OptimizationRecommendation(
                category='stability',
                priority='high',
                title='Critical GPU Temperature Warning',
                description=f"GPU temperature is {gpu_analysis.get('max_temperature')}°C, which may cause throttling",
                action='Improve cooling or reduce GPU load',
                impact='Prevent thermal throttling and maintain consistent performance',
                implementation_difficulty='moderate',
                estimated_improvement='Prevent 10-30% performance degradation'
            ))
        
        # Check VRAM utilization
        vram_util = gpu_analysis.get('vram_utilization_percent', 0)
        if vram_util < 30:
            recommendations.append(OptimizationRecommendation(
                category='performance',
                priority='medium',
                title='Underutilized GPU Memory',
                description=f"Only {vram_util}% of GPU memory is being used",
                action='Consider loading larger models or increasing context sizes',
                impact='Better utilize available hardware resources',
                implementation_difficulty='easy',
                estimated_improvement='Enable 2-4x larger models'
            ))
        elif vram_util > 90:
            recommendations.append(OptimizationRecommendation(
                category='stability',
                priority='high',
                title='GPU Memory Near Capacity',
                description=f"GPU memory utilization is {vram_util}%",
                action='Reduce model size, context length, or optimize tensor split',
                impact='Prevent out-of-memory errors and crashes',
                implementation_difficulty='easy',
                estimated_improvement='Ensure stable operation'
            ))
        
        # Check tensor split optimization
        if gpu_analysis.get('gpu_count', 0) > 1:
            config_quality = gpu_analysis.get('configuration_quality', {})
            if config_quality.get('balance_score', 100) < 70:
                recommendations.append(OptimizationRecommendation(
                    category='performance',
                    priority='medium',
                    title='Suboptimal Tensor Split Configuration',
                    description='Current tensor split doesn\'t optimally utilize all GPUs',
                    action='Adjust tensor split based on GPU memory and capabilities',
                    impact='Better distribute workload across all GPUs',
                    implementation_difficulty='easy',
                    estimated_improvement='10-20% performance improvement'
                ))
        
        return recommendations
    
    def _memory_recommendations(self, memory_analysis: Dict, model_analysis: Dict) -> List[OptimizationRecommendation]:
        """Generate memory-specific recommendations."""
        recommendations = []
        
        memory_pressure = memory_analysis.get('memory_pressure', 'low')
        if memory_pressure == 'high':
            recommendations.append(OptimizationRecommendation(
                category='stability',
                priority='high',
                title='High System Memory Pressure',
                description=f"System memory usage is {memory_analysis.get('usage_percent')}%",
                action='Close unnecessary applications or add more RAM',
                impact='Prevent system swapping and maintain performance',
                implementation_difficulty='moderate',
                estimated_improvement='Avoid 50-90% performance degradation'
            ))
        
        # Check memory adequacy for LLM workloads
        adequacy = memory_analysis.get('adequacy_for_llm', 'unknown')
        if adequacy == 'insufficient':
            recommendations.append(OptimizationRecommendation(
                category='performance',
                priority='medium',
                title='Insufficient System Memory for Large Models',
                description='System RAM may be limiting for very large model operations',
                action='Consider upgrading to 64GB+ RAM for optimal large model performance',
                impact='Enable smooth operation of 70B+ parameter models',
                implementation_difficulty='difficult',
                estimated_improvement='Enable 2-4x larger model deployment'
            ))
        
        return recommendations
    
    def _cpu_recommendations(self, cpu_analysis: Dict, current_config: Dict = None) -> List[OptimizationRecommendation]:
        """Generate CPU-specific recommendations."""
        recommendations = []
        
        cpu_usage = cpu_analysis.get('current_usage_percent', 0)
        if cpu_usage > 80:
            recommendations.append(OptimizationRecommendation(
                category='performance',
                priority='medium',
                title='High CPU Utilization',
                description=f"CPU usage is {cpu_usage}%",
                action='Consider reducing CPU threads or optimizing background processes',
                impact='Free up CPU resources for inference processing',
                implementation_difficulty='easy',
                estimated_improvement='5-15% better response times'
            ))
        
        # Check thread configuration
        logical_cores = cpu_analysis.get('logical_cores', 0)
        if current_config and logical_cores > 0:
            configured_threads = current_config.get('threads', 32)
            if configured_threads > logical_cores * 2:
                recommendations.append(OptimizationRecommendation(
                    category='performance',
                    priority='low',
                    title='Excessive Thread Configuration',
                    description=f"Configured {configured_threads} threads on {logical_cores}-core system",
                    action=f'Reduce thread count to {logical_cores} or {logical_cores * 2}',
                    impact='Reduce context switching overhead',
                    implementation_difficulty='easy',
                    estimated_improvement='3-8% efficiency improvement'
                ))
        
        return recommendations
    
    def _model_recommendations(self, model_analysis: Dict, gpu_analysis: Dict) -> List[OptimizationRecommendation]:
        """Generate model-specific recommendations."""
        recommendations = []
        
        # Check model diversity
        diversity_score = model_analysis.get('diversity_score', 0)
        if diversity_score < 0.3:
            recommendations.append(OptimizationRecommendation(
                category='cost',
                priority='low',
                title='Limited Model Size Diversity',
                description='Most models are of similar size - consider diversifying',
                action='Add smaller models for quick tasks and larger models for complex reasoning',
                impact='Optimize resource usage for different task types',
                implementation_difficulty='moderate',
                estimated_improvement='20-40% cost reduction for simple tasks'
            ))
        
        # Check context size optimization
        max_context = model_analysis.get('max_context_size', 0)
        total_vram = gpu_analysis.get('total_vram_gb', 0)
        if max_context > 65536 and total_vram < 32:
            recommendations.append(OptimizationRecommendation(
                category='performance',
                priority='medium',
                title='Context Size vs VRAM Mismatch',
                description=f"Models support up to {max_context} context but VRAM may be limiting",
                action='Consider context-optimized model variants or adjust default context size',
                impact='Better match model capabilities to hardware constraints',
                implementation_difficulty='easy',
                estimated_improvement='Avoid memory bottlenecks'
            ))
        
        return recommendations
    
    def _config_recommendations(self, current_config: Dict, gpu_analysis: Dict) -> List[OptimizationRecommendation]:
        """Generate configuration-specific recommendations."""
        recommendations = []
        
        # Check GPU layers configuration
        gpu_layers = current_config.get('gpu_layers', 999)
        gpu_count = gpu_analysis.get('gpu_count', 0)
        if gpu_layers == 999 and gpu_count > 0:
            recommendations.append(OptimizationRecommendation(
                category='performance',
                priority='low',
                title='GPU Layer Configuration Optimization',
                description='Using maximum GPU layers - consider optimizing for specific models',
                action='Fine-tune GPU layer count based on model size and available VRAM',
                impact='Optimize memory usage and inference speed',
                implementation_difficulty='moderate',
                estimated_improvement='5-10% memory efficiency'
            ))
        
        # Check batch size optimization
        batch_size = current_config.get('batch_size', 512)
        total_vram = gpu_analysis.get('total_vram_gb', 0)
        if batch_size < 256 and total_vram > 16:
            recommendations.append(OptimizationRecommendation(
                category='performance',
                priority='medium',
                title='Suboptimal Batch Size',
                description=f"Batch size of {batch_size} may be too small for available VRAM",
                action='Consider increasing batch size to 512-1024 for better throughput',
                impact='Improve token generation throughput',
                implementation_difficulty='easy',
                estimated_improvement='15-30% faster processing'
            ))
        
        return recommendations
    
    def _calculate_optimization_score(self, cpu_analysis: Dict, memory_analysis: Dict, 
                                    gpu_analysis: Dict, model_analysis: Dict) -> float:
        """Calculate overall system optimization score (0-100)."""
        scores = []
        weights = []
        
        # CPU score (20% weight)
        cpu_score = 100
        if cpu_analysis.get('current_usage_percent', 0) > 80:
            cpu_score -= 30
        if cpu_analysis.get('performance_rating') == 'poor':
            cpu_score -= 20
        scores.append(max(0, cpu_score))
        weights.append(20)
        
        # Memory score (25% weight)
        memory_score = 100
        memory_pressure = memory_analysis.get('memory_pressure', 'low')
        if memory_pressure == 'high':
            memory_score -= 40
        elif memory_pressure == 'medium':
            memory_score -= 20
        scores.append(max(0, memory_score))
        weights.append(25)
        
        # GPU score (40% weight)
        gpu_score = 100
        if gpu_analysis.get('thermal_status') == 'critical':
            gpu_score -= 50
        elif gpu_analysis.get('thermal_status') == 'warning':
            gpu_score -= 20
        
        vram_util = gpu_analysis.get('vram_utilization_percent', 50)
        if vram_util > 90:
            gpu_score -= 30
        elif vram_util < 20:
            gpu_score -= 15
        
        avg_gpu_util = gpu_analysis.get('average_gpu_utilization', 0)
        if avg_gpu_util > 90:
            gpu_score -= 20
        elif avg_gpu_util < 10:
            gpu_score -= 10
        
        scores.append(max(0, gpu_score))
        weights.append(40)
        
        # Model score (15% weight)
        model_score = 100
        diversity = model_analysis.get('diversity_score', 1.0)
        if diversity < 0.3:
            model_score -= 20
        scores.append(max(0, model_score))
        weights.append(15)
        
        # Calculate weighted average
        total_weight = sum(weights)
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        return round(weighted_sum / total_weight, 1)
    
    # Helper methods
    
    def _rate_cpu_performance(self, cpu_count: int, cpu_freq, cpu_brand: str) -> str:
        """Rate CPU performance for LLM workloads."""
        if cpu_count >= 16 and cpu_freq and cpu_freq.max > 3000:
            return 'excellent'
        elif cpu_count >= 8 and cpu_freq and cpu_freq.max > 2500:
            return 'good'
        elif cpu_count >= 4:
            return 'adequate'
        else:
            return 'poor'
    
    def _assess_memory_adequacy(self, total_memory_bytes: int) -> str:
        """Assess if system memory is adequate for LLM workloads."""
        total_gb = total_memory_bytes / (1024**3)
        if total_gb >= 64:
            return 'excellent'
        elif total_gb >= 32:
            return 'good'
        elif total_gb >= 16:
            return 'adequate'
        else:
            return 'insufficient'
    
    def _assess_gpu_configuration(self, gpus: List, tensor_splits: List[float]) -> Dict[str, Any]:
        """Assess quality of current GPU configuration."""
        if not gpus:
            return {'balance_score': 0, 'utilization_score': 0}
        
        # Calculate balance score based on how well tensor split matches GPU capabilities
        balance_score = 100
        if len(tensor_splits) != len(gpus):
            balance_score -= 30
        
        # Check if allocation matches relative GPU memory
        total_vram = sum(gpu.memory_total for gpu in gpus)
        for i, gpu in enumerate(gpus):
            if i < len(tensor_splits):
                expected_allocation = gpu.memory_total / total_vram
                actual_allocation = tensor_splits[i]
                deviation = abs(expected_allocation - actual_allocation)
                balance_score -= deviation * 100
        
        # Calculate utilization score
        avg_utilization = sum(gpu.utilization_gpu for gpu in gpus) / len(gpus)
        utilization_score = 100 - abs(50 - avg_utilization)  # Optimal around 50%
        
        return {
            'balance_score': max(0, balance_score),
            'utilization_score': max(0, utilization_score),
            'thermal_score': 100 - max(0, max(gpu.temperature for gpu in gpus) - 70) * 2
        }
    
    def _calculate_model_diversity(self, model_categories: Dict[str, List]) -> float:
        """Calculate model diversity score (0-1)."""
        total_models = sum(len(models) for models in model_categories.values())
        if total_models == 0:
            return 0
        
        # Calculate entropy-based diversity
        import math
        diversity = 0
        for models in model_categories.values():
            if models:
                p = len(models) / total_models
                diversity -= p * math.log2(p)
        
        # Normalize to 0-1 scale (max entropy for 4 categories is 2)
        return diversity / 2
    
    def _assess_model_optimization_potential(self, model_categories: Dict, context_sizes: List[int]) -> str:
        """Assess potential for model configuration optimization."""
        large_models = len(model_categories.get('large', [])) + len(model_categories.get('xlarge', []))
        total_models = sum(len(models) for models in model_categories.values())
        
        if large_models / total_models > 0.5 if total_models > 0 else False:
            return 'high'
        elif len(set(context_sizes)) < 3:
            return 'medium'
        else:
            return 'low'
    
    def _create_fallback_analysis(self) -> SystemAnalysis:
        """Create a fallback analysis when full analysis fails."""
        return SystemAnalysis(
            timestamp=datetime.now().isoformat(),
            cpu_analysis={'error': 'Analysis failed'},
            memory_analysis={'error': 'Analysis failed'},
            gpu_analysis={'error': 'Analysis failed'},
            model_analysis={'error': 'Analysis failed'},
            recommendations=[
                OptimizationRecommendation(
                    category='stability',
                    priority='medium',
                    title='System Analysis Unavailable',
                    description='Unable to perform complete system analysis',
                    action='Check system monitoring capabilities and permissions',
                    impact='Enable comprehensive optimization insights',
                    implementation_difficulty='moderate',
                    estimated_improvement='Enable full optimization potential'
                )
            ],
            overall_score=50.0
        )

# Global optimizer instance
hardware_optimizer = HardwareOptimizer()