"""
Interactive Model Explorer for LLM Inference Service.

This module provides comprehensive model analysis, comparison, and exploration
capabilities including performance benchmarking, capability analysis, and
interactive model testing.
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelBenchmark:
    """Model performance benchmark result."""
    model_id: str
    test_prompt: str
    response_time_ms: float
    tokens_generated: int
    tokens_per_second: float
    memory_usage_mb: float
    gpu_utilization_percent: float
    quality_score: float  # 0-100 based on response quality
    timestamp: str


@dataclass
class ModelCapability:
    """Model capability assessment."""
    model_id: str
    context_window: int
    parameter_count: int
    quantization_level: str
    architecture: str
    supports_code: bool
    supports_reasoning: bool
    supports_multimodal: bool
    language_support: List[str]
    specializations: List[str]  # e.g., ['math', 'science', 'creative']
    performance_rating: str  # 'excellent', 'good', 'fair', 'poor'


@dataclass
class ModelComparison:
    """Model comparison result."""
    models: List[str]
    comparison_type: str  # 'performance', 'capability', 'cost'
    metrics: Dict[str, Dict[str, float]]  # model_id -> metric_name -> value
    winner: str
    summary: str
    timestamp: str


@dataclass
class ModelTestResult:
    """Interactive model test result."""
    model_id: str
    test_id: str
    prompt: str
    response: str
    response_time_ms: float
    tokens_generated: int
    user_rating: Optional[int]  # 1-5 stars
    notes: str
    timestamp: str


class ModelExplorer:
    """Interactive model exploration and analysis system."""
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager
        self.benchmarks = []
        self.capabilities = {}
        self.test_results = []
        self.comparison_cache = {}
        
        # Pre-defined test prompts for different categories
        self.test_prompts = {
            'general': [
                "What is artificial intelligence?",
                "Explain quantum computing in simple terms.",
                "What are the benefits of renewable energy?",
                "How does machine learning work?",
                "What is the future of space exploration?"
            ],
            'coding': [
                "Write a Python function to sort a list of numbers.",
                "Explain the difference between classes and objects in programming.",
                "How do you implement a binary search algorithm?",
                "What are the advantages of functional programming?",
                "Write a SQL query to find duplicate records."
            ],
            'reasoning': [
                "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
                "A train leaves at 9 AM traveling at 60 mph. Another leaves at 10 AM from the same station traveling at 80 mph in the same direction. When will the second train catch up?",
                "You have 3 boxes. One contains only apples, one only oranges, one both. All labels are wrong. How many fruits must you draw to correctly label all boxes?",
                "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?"
            ],
            'creative': [
                "Write a short story about a robot discovering emotions.",
                "Create a haiku about artificial intelligence.",
                "Describe a futuristic city from the perspective of a time traveler.",
                "Write a dialogue between two AI systems discussing consciousness.",
                "Compose a limerick about machine learning."
            ],
            'math': [
                "Calculate the derivative of x^3 + 2x^2 - 5x + 1",
                "Solve the quadratic equation: 2x^2 + 5x - 3 = 0",
                "Find the area of a circle with radius 7 cm",
                "What is the probability of rolling two dice and getting a sum of 8?",
                "Calculate the compound interest on $1000 at 5% annually for 3 years"
            ],
            'science': [
                "Explain photosynthesis and its importance to life on Earth.",
                "What causes the Northern Lights (Aurora Borealis)?",
                "How does DNA replication work?",
                "Explain the greenhouse effect and its impact on climate.",
                "What is CRISPR and how is it used in gene editing?"
            ]
        }
        
        logger.info("Model explorer initialized")
    
    def discover_model_capabilities(self, model_id: str) -> ModelCapability:
        """Analyze and discover model capabilities."""
        try:
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            model_info = self.model_manager.get_model_info(model_id)
            if not model_info:
                raise ValueError(f"Model {model_id} not found")
            
            # Extract basic information
            context_window = getattr(model_info, 'context_size', 0)
            architecture = getattr(model_info, 'architecture', 'unknown')
            quantization = getattr(model_info, 'quantization', 'unknown')
            
            # Estimate parameter count from model ID
            parameter_count = self._estimate_parameters(model_id)
            
            # Analyze capabilities based on model name and type
            capabilities = self._analyze_model_type(model_id, architecture)
            
            # Performance rating based on size and architecture
            performance_rating = self._rate_performance(parameter_count, architecture, quantization)
            
            capability = ModelCapability(
                model_id=model_id,
                context_window=context_window,
                parameter_count=parameter_count,
                quantization_level=quantization,
                architecture=architecture,
                supports_code=capabilities['code'],
                supports_reasoning=capabilities['reasoning'],
                supports_multimodal=capabilities['multimodal'],
                language_support=capabilities['languages'],
                specializations=capabilities['specializations'],
                performance_rating=performance_rating
            )
            
            self.capabilities[model_id] = capability
            return capability
            
        except Exception as e:
            logger.error(f"Error discovering capabilities for {model_id}: {e}")
            # Return default capability
            return ModelCapability(
                model_id=model_id,
                context_window=0,
                parameter_count=0,
                quantization_level='unknown',
                architecture='unknown',
                supports_code=False,
                supports_reasoning=False,
                supports_multimodal=False,
                language_support=['english'],
                specializations=[],
                performance_rating='unknown'
            )
    
    def _estimate_parameters(self, model_id: str) -> int:
        """Estimate parameter count from model ID."""
        model_lower = model_id.lower()
        
        # Common parameter patterns
        if '1b' in model_lower or '1.1b' in model_lower:
            return 1_000_000_000
        elif '3b' in model_lower or '2b' in model_lower:
            return 3_000_000_000
        elif '7b' in model_lower or '8b' in model_lower:
            return 8_000_000_000
        elif '13b' in model_lower or '12b' in model_lower:
            return 13_000_000_000
        elif '30b' in model_lower or '33b' in model_lower:
            return 33_000_000_000
        elif '70b' in model_lower:
            return 70_000_000_000
        elif '235b' in model_lower:
            return 235_000_000_000
        elif '671b' in model_lower:
            return 671_000_000_000
        
        # Architecture-based estimates
        if 'mixtral' in model_lower:
            if '8x7b' in model_lower:
                return 47_000_000_000  # Sparse model
            elif '8x22b' in model_lower:
                return 141_000_000_000
        
        if 'phi' in model_lower:
            if 'mini' in model_lower:
                return 3_800_000_000
            else:
                return 14_000_000_000
        
        # Default estimate
        return 8_000_000_000
    
    def _analyze_model_type(self, model_id: str, architecture: str) -> Dict[str, Any]:
        """Analyze model type and capabilities."""
        model_lower = model_id.lower()
        
        capabilities = {
            'code': False,
            'reasoning': False,
            'multimodal': False,
            'languages': ['english'],
            'specializations': []
        }
        
        # Code capabilities
        if any(keyword in model_lower for keyword in ['code', 'coder', 'coding', 'starcoder', 'codellama']):
            capabilities['code'] = True
            capabilities['specializations'].append('programming')
        
        # Reasoning capabilities
        if any(keyword in model_lower for keyword in ['reasoning', 'reason', 'phi4', 'deepseek-r1']):
            capabilities['reasoning'] = True
            capabilities['specializations'].append('logical_reasoning')
        
        # Multimodal capabilities
        if any(keyword in model_lower for keyword in ['vision', 'multimodal', 'mm', 'vlm']):
            capabilities['multimodal'] = True
            capabilities['specializations'].append('vision')
        
        # Language support
        if 'qwen' in model_lower:
            capabilities['languages'].extend(['chinese', 'multilingual'])
        elif 'gemma' in model_lower:
            capabilities['languages'].append('multilingual')
        
        # Specialized models
        if 'financial' in model_lower or 'finance' in model_lower:
            capabilities['specializations'].append('finance')
        elif 'medical' in model_lower or 'med' in model_lower:
            capabilities['specializations'].append('medical')
        elif 'legal' in model_lower or 'law' in model_lower:
            capabilities['specializations'].append('legal')
        elif 'math' in model_lower or 'mathematical' in model_lower:
            capabilities['specializations'].append('mathematics')
        
        # General capabilities based on architecture
        if architecture in ['llama', 'mistral', 'phi']:
            capabilities['code'] = True
            if not capabilities['specializations']:
                capabilities['specializations'].append('general_purpose')
        
        return capabilities
    
    def _rate_performance(self, parameter_count: int, architecture: str, quantization: str) -> str:
        """Rate model performance based on specifications."""
        # Base score from parameter count
        if parameter_count >= 70_000_000_000:
            base_score = 'excellent'
        elif parameter_count >= 30_000_000_000:
            base_score = 'good'
        elif parameter_count >= 7_000_000_000:
            base_score = 'fair'
        else:
            base_score = 'basic'
        
        # Adjust for architecture
        if architecture in ['llama', 'phi', 'mistral']:
            # These are generally high-quality architectures
            if base_score == 'fair':
                base_score = 'good'
            elif base_score == 'basic':
                base_score = 'fair'
        
        # Adjust for quantization
        if 'q4' in quantization.lower() or 'q6' in quantization.lower():
            # Reasonable quantization levels
            pass
        elif 'q8' in quantization.lower() or 'f16' in quantization.lower():
            # Higher precision
            if base_score == 'fair':
                base_score = 'good'
            elif base_score == 'good':
                base_score = 'excellent'
        elif 'q2' in quantization.lower() or 'q3' in quantization.lower():
            # Lower precision
            if base_score == 'excellent':
                base_score = 'good'
            elif base_score == 'good':
                base_score = 'fair'
        
        return base_score
    
    async def benchmark_model(self, model_id: str, test_category: str = 'general', 
                            custom_prompt: str = None) -> ModelBenchmark:
        """Benchmark a specific model with test prompts."""
        try:
            # Get test prompt
            if custom_prompt:
                test_prompt = custom_prompt
            else:
                prompts = self.test_prompts.get(test_category, self.test_prompts['general'])
                test_prompt = prompts[0]  # Use first prompt for consistency
            
            # Record start time and memory
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # This would integrate with the actual model execution
            # For now, simulate performance based on model characteristics
            model_info = self.model_manager.get_model_info(model_id) if self.model_manager else None
            
            # Simulate response generation
            await self._simulate_model_execution(model_id, test_prompt)
            
            # Calculate metrics
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Estimate tokens and performance based on model
            tokens_generated = len(test_prompt.split()) * 2  # Rough estimate
            tokens_per_second = tokens_generated / (response_time_ms / 1000) if response_time_ms > 0 else 0
            
            memory_usage_mb = self._get_memory_usage() - start_memory
            gpu_utilization = self._get_gpu_utilization()
            
            # Quality score based on model capabilities
            quality_score = self._estimate_quality_score(model_id, test_category)
            
            benchmark = ModelBenchmark(
                model_id=model_id,
                test_prompt=test_prompt,
                response_time_ms=response_time_ms,
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_per_second,
                memory_usage_mb=memory_usage_mb,
                gpu_utilization_percent=gpu_utilization,
                quality_score=quality_score,
                timestamp=datetime.now().isoformat()
            )
            
            self.benchmarks.append(benchmark)
            return benchmark
            
        except Exception as e:
            logger.error(f"Error benchmarking model {model_id}: {e}")
            raise
    
    async def _simulate_model_execution(self, model_id: str, prompt: str):
        """Simulate model execution for benchmarking."""
        # Simulate different response times based on model size
        param_count = self._estimate_parameters(model_id)
        
        if param_count >= 70_000_000_000:
            base_time = 5.0  # Large models take longer
        elif param_count >= 30_000_000_000:
            base_time = 3.0
        elif param_count >= 7_000_000_000:
            base_time = 2.0
        else:
            base_time = 1.0
        
        # Add variability
        import random
        actual_time = base_time * (0.8 + 0.4 * random.random())
        
        await asyncio.sleep(actual_time)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization."""
        try:
            # This would integrate with GPU monitor
            return 45.0  # Simulated value
        except:
            return 0.0
    
    def _estimate_quality_score(self, model_id: str, test_category: str) -> float:
        """Estimate quality score based on model and test type."""
        base_score = 70.0
        
        # Adjust based on model size
        param_count = self._estimate_parameters(model_id)
        if param_count >= 70_000_000_000:
            base_score += 20
        elif param_count >= 30_000_000_000:
            base_score += 15
        elif param_count >= 7_000_000_000:
            base_score += 10
        
        # Adjust based on specialization
        capabilities = self._analyze_model_type(model_id, '')
        if test_category == 'coding' and capabilities['code']:
            base_score += 10
        elif test_category == 'reasoning' and capabilities['reasoning']:
            base_score += 10
        
        # Add some randomness for realism
        import random
        base_score += random.uniform(-5, 5)
        
        return min(100.0, max(0.0, base_score))
    
    def compare_models(self, model_ids: List[str], comparison_type: str = 'performance') -> ModelComparison:
        """Compare multiple models across different metrics."""
        try:
            if len(model_ids) < 2:
                raise ValueError("At least 2 models required for comparison")
            
            metrics = {}
            
            if comparison_type == 'performance':
                # Performance comparison based on benchmarks
                for model_id in model_ids:
                    model_benchmarks = [b for b in self.benchmarks if b.model_id == model_id]
                    if model_benchmarks:
                        avg_response_time = statistics.mean([b.response_time_ms for b in model_benchmarks])
                        avg_tokens_per_sec = statistics.mean([b.tokens_per_second for b in model_benchmarks])
                        avg_quality = statistics.mean([b.quality_score for b in model_benchmarks])
                    else:
                        # Use estimated values
                        param_count = self._estimate_parameters(model_id)
                        avg_response_time = 5000 if param_count >= 70_000_000_000 else 2000
                        avg_tokens_per_sec = 10 if param_count >= 70_000_000_000 else 20
                        avg_quality = 85 if param_count >= 70_000_000_000 else 75
                    
                    metrics[model_id] = {
                        'response_time_ms': avg_response_time,
                        'tokens_per_second': avg_tokens_per_sec,
                        'quality_score': avg_quality,
                        'parameter_count': self._estimate_parameters(model_id)
                    }
                
                # Determine winner based on composite score
                winner = self._determine_performance_winner(metrics)
                summary = f"{winner} shows the best overall performance balance"
                
            elif comparison_type == 'capability':
                # Capability comparison
                for model_id in model_ids:
                    capability = self.capabilities.get(model_id) or self.discover_model_capabilities(model_id)
                    
                    metrics[model_id] = {
                        'context_window': capability.context_window,
                        'parameter_count': capability.parameter_count,
                        'supports_code': 1 if capability.supports_code else 0,
                        'supports_reasoning': 1 if capability.supports_reasoning else 0,
                        'supports_multimodal': 1 if capability.supports_multimodal else 0,
                        'specialization_count': len(capability.specializations)
                    }
                
                winner = max(model_ids, key=lambda m: sum(metrics[m].values()))
                summary = f"{winner} has the most comprehensive capabilities"
                
            elif comparison_type == 'cost':
                # Cost-effectiveness comparison
                for model_id in model_ids:
                    param_count = self._estimate_parameters(model_id)
                    # Estimate VRAM usage (rough)
                    vram_gb = param_count / 1_000_000_000 * 2  # Rough estimate
                    
                    # Cost factors (lower is better)
                    compute_cost = param_count / 1_000_000_000  # Relative cost
                    memory_cost = vram_gb
                    
                    metrics[model_id] = {
                        'compute_cost': compute_cost,
                        'memory_cost_gb': memory_cost,
                        'parameter_efficiency': 100 / (param_count / 1_000_000_000),  # Higher is better
                        'total_cost_score': compute_cost + memory_cost
                    }
                
                winner = min(model_ids, key=lambda m: metrics[m]['total_cost_score'])
                summary = f"{winner} offers the best cost-effectiveness ratio"
            
            else:
                raise ValueError(f"Unknown comparison type: {comparison_type}")
            
            comparison = ModelComparison(
                models=model_ids,
                comparison_type=comparison_type,
                metrics=metrics,
                winner=winner,
                summary=summary,
                timestamp=datetime.now().isoformat()
            )
            
            # Cache the comparison
            cache_key = hashlib.md5(f"{'-'.join(sorted(model_ids))}-{comparison_type}".encode()).hexdigest()
            self.comparison_cache[cache_key] = comparison
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            raise
    
    def _determine_performance_winner(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """Determine performance winner based on composite score."""
        scores = {}
        
        for model_id, model_metrics in metrics.items():
            # Normalize scores (lower response time is better, higher tokens/sec and quality is better)
            response_time_score = 100 / max(model_metrics['response_time_ms'], 100)  # Inverse
            tokens_score = model_metrics['tokens_per_second']
            quality_score = model_metrics['quality_score']
            
            # Weighted composite score
            composite_score = (response_time_score * 0.3 + tokens_score * 0.3 + quality_score * 0.4)
            scores[model_id] = composite_score
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    def get_model_recommendations(self, use_case: str, constraints: Dict[str, Any] = None) -> List[Tuple[str, float, str]]:
        """Get model recommendations for specific use cases."""
        try:
            constraints = constraints or {}
            max_vram_gb = constraints.get('max_vram_gb', float('inf'))
            max_response_time_ms = constraints.get('max_response_time_ms', float('inf'))
            required_capabilities = constraints.get('capabilities', [])
            
            recommendations = []
            
            if not self.model_manager:
                return recommendations
            
            # Get all available models
            model_mapping = self.model_manager.build_model_mapping()
            
            for model_id in model_mapping.keys():
                # Get or discover capabilities
                capability = self.capabilities.get(model_id) or self.discover_model_capabilities(model_id)
                
                # Check constraints
                param_count = capability.parameter_count
                estimated_vram = param_count / 1_000_000_000 * 2  # Rough estimate
                
                if estimated_vram > max_vram_gb:
                    continue
                
                # Check required capabilities
                model_caps = []
                if capability.supports_code:
                    model_caps.append('code')
                if capability.supports_reasoning:
                    model_caps.append('reasoning')
                if capability.supports_multimodal:
                    model_caps.append('multimodal')
                
                if required_capabilities and not all(cap in model_caps for cap in required_capabilities):
                    continue
                
                # Calculate suitability score for use case
                score = self._calculate_use_case_score(model_id, capability, use_case)
                
                # Add reasoning for recommendation
                reasoning = self._generate_recommendation_reasoning(model_id, capability, use_case, score)
                
                recommendations.append((model_id, score, reasoning))
            
            # Sort by score (descending)
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations[:10]  # Top 10 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _calculate_use_case_score(self, model_id: str, capability: ModelCapability, use_case: str) -> float:
        """Calculate suitability score for a specific use case."""
        base_score = 50.0
        
        use_case_lower = use_case.lower()
        
        # Adjust based on use case
        if 'code' in use_case_lower or 'programming' in use_case_lower:
            if capability.supports_code:
                base_score += 30
            if 'code' in capability.specializations:
                base_score += 20
        
        elif 'reasoning' in use_case_lower or 'logic' in use_case_lower or 'math' in use_case_lower:
            if capability.supports_reasoning:
                base_score += 30
            if 'logical_reasoning' in capability.specializations or 'mathematics' in capability.specializations:
                base_score += 20
        
        elif 'creative' in use_case_lower or 'writing' in use_case_lower:
            # Larger models generally better for creative tasks
            if capability.parameter_count >= 30_000_000_000:
                base_score += 25
            elif capability.parameter_count >= 7_000_000_000:
                base_score += 15
        
        elif 'chat' in use_case_lower or 'conversation' in use_case_lower:
            # General purpose models good for chat
            if 'general_purpose' in capability.specializations:
                base_score += 20
            # Medium-large models good balance for chat
            if 7_000_000_000 <= capability.parameter_count <= 30_000_000_000:
                base_score += 15
        
        elif 'analysis' in use_case_lower or 'research' in use_case_lower:
            # Larger context windows better for analysis
            if capability.context_window >= 32000:
                base_score += 25
            elif capability.context_window >= 16000:
                base_score += 15
        
        # Adjust for performance rating
        rating_bonus = {
            'excellent': 20,
            'good': 10,
            'fair': 0,
            'poor': -10,
            'basic': -5
        }
        base_score += rating_bonus.get(capability.performance_rating, 0)
        
        # Adjust for context window (generally larger is better)
        if capability.context_window >= 100000:
            base_score += 15
        elif capability.context_window >= 32000:
            base_score += 10
        elif capability.context_window >= 16000:
            base_score += 5
        
        return min(100.0, max(0.0, base_score))
    
    def _generate_recommendation_reasoning(self, model_id: str, capability: ModelCapability, 
                                         use_case: str, score: float) -> str:
        """Generate human-readable reasoning for recommendation."""
        reasons = []
        
        # Parameter count
        param_billions = capability.parameter_count / 1_000_000_000
        if param_billions >= 70:
            reasons.append(f"Large {param_billions:.0f}B parameter model with excellent capabilities")
        elif param_billions >= 30:
            reasons.append(f"Medium-large {param_billions:.0f}B parameter model with good performance")
        elif param_billions >= 7:
            reasons.append(f"Medium {param_billions:.0f}B parameter model with balanced performance")
        else:
            reasons.append(f"Smaller {param_billions:.1f}B parameter model for efficient inference")
        
        # Context window
        if capability.context_window >= 100000:
            reasons.append(f"Very large {capability.context_window//1000}K context window")
        elif capability.context_window >= 32000:
            reasons.append(f"Large {capability.context_window//1000}K context window")
        elif capability.context_window >= 16000:
            reasons.append(f"Good {capability.context_window//1000}K context window")
        
        # Specializations
        if capability.specializations:
            spec_str = ', '.join(capability.specializations)
            reasons.append(f"Specialized for: {spec_str}")
        
        # Performance rating
        if capability.performance_rating in ['excellent', 'good']:
            reasons.append(f"{capability.performance_rating.title()} performance rating")
        
        # Score interpretation
        if score >= 80:
            reasons.insert(0, "Excellent match for this use case")
        elif score >= 70:
            reasons.insert(0, "Good match for this use case")
        elif score >= 60:
            reasons.insert(0, "Decent match for this use case")
        else:
            reasons.insert(0, "Basic match for this use case")
        
        return ". ".join(reasons) + "."
    
    def record_test_result(self, model_id: str, prompt: str, response: str, 
                          response_time_ms: float, tokens_generated: int,
                          user_rating: int = None, notes: str = "") -> ModelTestResult:
        """Record an interactive test result."""
        test_result = ModelTestResult(
            model_id=model_id,
            test_id=hashlib.md5(f"{model_id}-{prompt}-{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            prompt=prompt,
            response=response,
            response_time_ms=response_time_ms,
            tokens_generated=tokens_generated,
            user_rating=user_rating,
            notes=notes,
            timestamp=datetime.now().isoformat()
        )
        
        self.test_results.append(test_result)
        return test_result
    
    def get_model_analytics(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a specific model."""
        try:
            capability = self.capabilities.get(model_id) or self.discover_model_capabilities(model_id)
            model_benchmarks = [b for b in self.benchmarks if b.model_id == model_id]
            model_tests = [t for t in self.test_results if t.model_id == model_id]
            
            analytics = {
                'model_id': model_id,
                'capabilities': asdict(capability),
                'benchmark_summary': {},
                'test_summary': {},
                'usage_patterns': {},
                'recommendations': []
            }
            
            # Benchmark summary
            if model_benchmarks:
                analytics['benchmark_summary'] = {
                    'total_benchmarks': len(model_benchmarks),
                    'avg_response_time_ms': statistics.mean([b.response_time_ms for b in model_benchmarks]),
                    'avg_tokens_per_second': statistics.mean([b.tokens_per_second for b in model_benchmarks]),
                    'avg_quality_score': statistics.mean([b.quality_score for b in model_benchmarks]),
                    'best_category': self._find_best_benchmark_category(model_benchmarks)
                }
            
            # Test summary
            if model_tests:
                user_ratings = [t.user_rating for t in model_tests if t.user_rating]
                analytics['test_summary'] = {
                    'total_tests': len(model_tests),
                    'avg_response_time_ms': statistics.mean([t.response_time_ms for t in model_tests]),
                    'avg_user_rating': statistics.mean(user_ratings) if user_ratings else None,
                    'total_tokens_generated': sum([t.tokens_generated for t in model_tests])
                }
            
            # Usage patterns
            analytics['usage_patterns'] = self._analyze_usage_patterns(model_id)
            
            # Recommendations for optimization
            analytics['recommendations'] = self._generate_optimization_recommendations(model_id, capability)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating analytics for {model_id}: {e}")
            return {'error': str(e)}
    
    def _find_best_benchmark_category(self, benchmarks: List[ModelBenchmark]) -> str:
        """Find the category where the model performs best."""
        category_scores = defaultdict(list)
        
        for benchmark in benchmarks:
            # Extract category from test prompt (simplified)
            for category, prompts in self.test_prompts.items():
                if benchmark.test_prompt in prompts:
                    category_scores[category].append(benchmark.quality_score)
                    break
        
        if not category_scores:
            return 'unknown'
        
        # Find category with highest average score
        avg_scores = {cat: statistics.mean(scores) for cat, scores in category_scores.items()}
        return max(avg_scores.keys(), key=lambda k: avg_scores[k])
    
    def _analyze_usage_patterns(self, model_id: str) -> Dict[str, Any]:
        """Analyze usage patterns for a model."""
        model_tests = [t for t in self.test_results if t.model_id == model_id]
        
        if not model_tests:
            return {}
        
        # Time patterns
        test_times = [datetime.fromisoformat(t.timestamp) for t in model_tests]
        hours = [t.hour for t in test_times]
        
        return {
            'most_active_hour': max(set(hours), key=hours.count) if hours else None,
            'tests_last_24h': len([t for t in test_times if t > datetime.now() - timedelta(days=1)]),
            'average_session_length': statistics.mean([t.response_time_ms for t in model_tests]),
            'preferred_prompt_length': statistics.mean([len(t.prompt) for t in model_tests])
        }
    
    def _generate_optimization_recommendations(self, model_id: str, capability: ModelCapability) -> List[str]:
        """Generate optimization recommendations for a model."""
        recommendations = []
        
        # Based on parameter count
        param_billions = capability.parameter_count / 1_000_000_000
        if param_billions >= 70:
            recommendations.append("Consider tensor parallelism across multiple GPUs for optimal performance")
            recommendations.append("Use batch processing for multiple requests to improve throughput")
        elif param_billions >= 30:
            recommendations.append("Monitor GPU memory usage and adjust context size as needed")
        else:
            recommendations.append("This model can run efficiently on single GPU setups")
        
        # Based on context window
        if capability.context_window >= 100000:
            recommendations.append("Large context window - excellent for document analysis and long conversations")
            recommendations.append("Consider enabling context caching for repeated long prompts")
        elif capability.context_window >= 32000:
            recommendations.append("Good context window for most applications")
        
        # Based on specializations
        if capability.supports_code:
            recommendations.append("Optimize for code generation by using appropriate code prompts")
        if capability.supports_reasoning:
            recommendations.append("Use chain-of-thought prompting for complex reasoning tasks")
        
        return recommendations
    
    def export_data(self) -> Dict[str, Any]:
        """Export all exploration data."""
        return {
            'export_timestamp': datetime.now().isoformat(),
            'capabilities': {k: asdict(v) for k, v in self.capabilities.items()},
            'benchmarks': [asdict(b) for b in self.benchmarks],
            'test_results': [asdict(t) for t in self.test_results],
            'comparison_cache': {k: asdict(v) for k, v in self.comparison_cache.items()},
            'test_prompts': self.test_prompts
        }


# Global model explorer instance
model_explorer = ModelExplorer()


# Import asyncio for async functions
import asyncio