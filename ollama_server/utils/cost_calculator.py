"""
Cost-Effectiveness Calculator for LLM Inference Service.

This module provides comprehensive cost analysis and optimization recommendations
for LLM deployment including hardware costs, operational expenses, and ROI calculations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class DeploymentType(Enum):
    """Deployment type options."""
    CONSUMER_GPU = "consumer_gpu"
    DATACENTER_GPU = "datacenter_gpu"
    CLOUD_INSTANCE = "cloud_instance"
    HYBRID = "hybrid"


class CostCategory(Enum):
    """Cost category types."""
    HARDWARE = "hardware"
    OPERATIONAL = "operational"
    MAINTENANCE = "maintenance"
    ENERGY = "energy"
    CLOUD = "cloud"
    DEVELOPMENT = "development"


@dataclass
class HardwareCost:
    """Hardware cost specification."""
    component_type: str  # 'gpu', 'cpu', 'memory', 'storage', 'motherboard', 'psu'
    component_name: str
    unit_cost_usd: float
    quantity: int
    lifespan_years: float
    resale_value_percent: float = 20.0  # Percentage of original cost
    power_consumption_watts: float = 0.0


@dataclass
class OperationalCost:
    """Operational cost specification."""
    cost_type: str  # 'electricity', 'cooling', 'maintenance', 'internet', 'staffing'
    monthly_cost_usd: float
    variable_per_hour: float = 0.0  # Variable cost per hour of operation
    description: str = ""


@dataclass
class CloudCost:
    """Cloud instance cost specification."""
    provider: str  # 'aws', 'gcp', 'azure', 'runpod', 'vast'
    instance_type: str
    hourly_cost_usd: float
    gpu_count: int
    gpu_type: str
    vram_gb: int
    cpu_cores: int
    memory_gb: int
    storage_gb: int
    setup_cost_usd: float = 0.0


@dataclass
class WorkloadProfile:
    """Workload profile for cost analysis."""
    requests_per_day: int
    avg_tokens_per_request: int
    peak_concurrent_requests: int
    uptime_hours_per_day: float
    seasonal_variance_factor: float = 1.0  # 1.0 = no variance, 1.5 = 50% peak variance
    growth_rate_monthly: float = 0.0  # Monthly growth rate


@dataclass
class CostAnalysis:
    """Comprehensive cost analysis result."""
    deployment_type: str
    total_cost_monthly_usd: float
    total_cost_yearly_usd: float
    cost_per_1k_tokens: float
    cost_per_request: float
    cost_breakdown: Dict[str, float]  # category -> amount
    hardware_costs: List[HardwareCost]
    operational_costs: List[OperationalCost]
    cloud_costs: List[CloudCost]
    roi_analysis: Dict[str, Any]
    optimization_recommendations: List[str]
    comparison_baselines: Dict[str, float]  # service_name -> cost_per_1k_tokens
    timestamp: str


@dataclass
class CostOptimization:
    """Cost optimization recommendation."""
    optimization_type: str
    current_cost_monthly: float
    optimized_cost_monthly: float
    savings_monthly: float
    savings_percent: float
    implementation_difficulty: str  # 'easy', 'moderate', 'difficult'
    implementation_time_days: int
    description: str
    action_items: List[str]


class CostCalculator:
    """Comprehensive cost analysis and optimization system."""
    
    def __init__(self):
        # Current market prices (updated periodically)
        self.gpu_prices = {
            # Consumer GPUs
            'RTX 4090': {'cost': 1599, 'vram': 24, 'power': 450, 'lifespan': 4},
            'RTX 4080': {'cost': 1199, 'vram': 16, 'power': 320, 'lifespan': 4},
            'RTX 3090': {'cost': 899, 'vram': 24, 'power': 350, 'lifespan': 3},
            'RTX 3080': {'cost': 599, 'vram': 12, 'power': 320, 'lifespan': 3},
            'Tesla M10': {'cost': 150, 'vram': 8, 'power': 225, 'lifespan': 6},  # Used/refurbished
            
            # Datacenter GPUs
            'H100 80GB': {'cost': 25000, 'vram': 80, 'power': 700, 'lifespan': 5},
            'A100 80GB': {'cost': 15000, 'vram': 80, 'power': 400, 'lifespan': 5},
            'A100 40GB': {'cost': 11000, 'vram': 40, 'power': 400, 'lifespan': 5},
            'V100 32GB': {'cost': 3000, 'vram': 32, 'power': 300, 'lifespan': 4}
        }
        
        # Cloud instance pricing (per hour USD)
        self.cloud_pricing = {
            'aws': {
                'p4d.24xlarge': {'cost': 32.77, 'gpus': 8, 'gpu_type': 'A100', 'vram': 320},
                'p3.16xlarge': {'cost': 24.48, 'gpus': 8, 'gpu_type': 'V100', 'vram': 128},
                'g5.48xlarge': {'cost': 16.29, 'gpus': 8, 'gpu_type': 'A10G', 'vram': 192}
            },
            'gcp': {
                'a2-highgpu-8g': {'cost': 12.00, 'gpus': 8, 'gpu_type': 'A100', 'vram': 320},
                'a2-highgpu-4g': {'cost': 6.00, 'gpus': 4, 'gpu_type': 'A100', 'vram': 160}
            },
            'runpod': {
                'RTX 4090': {'cost': 0.50, 'gpus': 1, 'gpu_type': 'RTX 4090', 'vram': 24},
                'A100 80GB': {'cost': 1.89, 'gpus': 1, 'gpu_type': 'A100', 'vram': 80}
            },
            'vast': {
                'RTX 4090': {'cost': 0.45, 'gpus': 1, 'gpu_type': 'RTX 4090', 'vram': 24},
                'RTX 3090': {'cost': 0.35, 'gpus': 1, 'gpu_type': 'RTX 3090', 'vram': 24}
            }
        }
        
        # Baseline service pricing for comparison
        self.baseline_pricing = {
            'OpenAI GPT-4': 0.03,  # per 1K tokens
            'Anthropic Claude': 0.025,
            'Google Gemini Pro': 0.00125,
            'Azure OpenAI': 0.03,
            'AWS Bedrock': 0.025,
            'Cohere': 0.015,
            'Together AI': 0.008,
            'Anyscale': 0.015
        }
        
        # Default operational costs
        self.default_operational_costs = {
            'electricity_per_kwh': 0.12,  # USD per kWh
            'cooling_factor': 0.3,  # 30% additional power for cooling
            'internet_monthly': 100,  # USD per month
            'maintenance_factor': 0.02,  # 2% of hardware cost annually
            'insurance_factor': 0.01,  # 1% of hardware cost annually
            'staffing_hourly': 50,  # USD per hour for DevOps/maintenance
            'staffing_hours_monthly': 20  # Hours per month
        }
        
        logger.info("Cost calculator initialized with current market pricing")
    
    def calculate_consumer_gpu_costs(self, gpu_config: Dict[str, Any], 
                                   workload: WorkloadProfile) -> CostAnalysis:
        """Calculate costs for consumer GPU deployment."""
        try:
            gpu_type = gpu_config.get('gpu_type', 'RTX 4090')
            gpu_count = gpu_config.get('gpu_count', 4)
            cpu_type = gpu_config.get('cpu_type', 'Intel Xeon E5-2680')
            memory_gb = gpu_config.get('memory_gb', 256)
            storage_gb = gpu_config.get('storage_gb', 2000)
            
            hardware_costs = []
            
            # GPU costs
            if gpu_type in self.gpu_prices:
                gpu_spec = self.gpu_prices[gpu_type]
                gpu_cost = HardwareCost(
                    component_type='gpu',
                    component_name=gpu_type,
                    unit_cost_usd=gpu_spec['cost'],
                    quantity=gpu_count,
                    lifespan_years=gpu_spec['lifespan'],
                    power_consumption_watts=gpu_spec['power']
                )
                hardware_costs.append(gpu_cost)
            
            # CPU cost (estimate)
            cpu_cost = HardwareCost(
                component_type='cpu',
                component_name=cpu_type,
                unit_cost_usd=800,  # Estimated cost for server CPU
                quantity=1,
                lifespan_years=5,
                power_consumption_watts=150
            )
            hardware_costs.append(cpu_cost)
            
            # Memory cost
            memory_cost_per_gb = 4  # USD per GB for server memory
            memory_cost = HardwareCost(
                component_type='memory',
                component_name=f'{memory_gb}GB DDR4',
                unit_cost_usd=memory_gb * memory_cost_per_gb,
                quantity=1,
                lifespan_years=5,
                power_consumption_watts=memory_gb * 0.5  # ~0.5W per GB
            )
            hardware_costs.append(memory_cost)
            
            # Storage cost
            storage_cost_per_gb = 0.08  # USD per GB for NVMe SSD
            storage_cost = HardwareCost(
                component_type='storage',
                component_name=f'{storage_gb}GB NVMe SSD',
                unit_cost_usd=storage_gb * storage_cost_per_gb,
                quantity=1,
                lifespan_years=5,
                power_consumption_watts=10
            )
            hardware_costs.append(storage_cost)
            
            # Motherboard and PSU
            motherboard_cost = HardwareCost(
                component_type='motherboard',
                component_name='Server Motherboard',
                unit_cost_usd=600,
                quantity=1,
                lifespan_years=7,
                power_consumption_watts=50
            )
            hardware_costs.append(motherboard_cost)
            
            total_power = sum(cost.power_consumption_watts * cost.quantity for cost in hardware_costs)
            psu_cost = HardwareCost(
                component_type='psu',
                component_name=f'{int(total_power * 1.3)}W Power Supply',
                unit_cost_usd=300,
                quantity=1,
                lifespan_years=6,
                power_consumption_watts=0  # Efficiency loss already factored
            )
            hardware_costs.append(psu_cost)
            
            # Calculate operational costs
            operational_costs = self._calculate_operational_costs(hardware_costs, workload)
            
            # Calculate total costs
            monthly_hardware_cost = self._calculate_monthly_hardware_cost(hardware_costs)
            monthly_operational_cost = sum(cost.monthly_cost_usd for cost in operational_costs)
            
            total_monthly = monthly_hardware_cost + monthly_operational_cost
            total_yearly = total_monthly * 12
            
            # Calculate per-token and per-request costs
            monthly_requests = workload.requests_per_day * 30
            monthly_tokens = monthly_requests * workload.avg_tokens_per_request
            
            cost_per_1k_tokens = (total_monthly / monthly_tokens) * 1000 if monthly_tokens > 0 else 0
            cost_per_request = total_monthly / monthly_requests if monthly_requests > 0 else 0
            
            # Cost breakdown
            cost_breakdown = {
                'hardware_amortization': monthly_hardware_cost,
                'electricity': next((c.monthly_cost_usd for c in operational_costs if c.cost_type == 'electricity'), 0),
                'cooling': next((c.monthly_cost_usd for c in operational_costs if c.cost_type == 'cooling'), 0),
                'maintenance': next((c.monthly_cost_usd for c in operational_costs if c.cost_type == 'maintenance'), 0),
                'other_operational': monthly_operational_cost - sum([
                    next((c.monthly_cost_usd for c in operational_costs if c.cost_type == t), 0)
                    for t in ['electricity', 'cooling', 'maintenance']
                ])
            }
            
            # ROI analysis
            roi_analysis = self._calculate_roi_analysis(
                total_monthly, cost_per_1k_tokens, workload, DeploymentType.CONSUMER_GPU
            )
            
            # Generate optimization recommendations
            optimizations = self._generate_consumer_gpu_optimizations(
                gpu_config, hardware_costs, operational_costs, workload
            )
            
            return CostAnalysis(
                deployment_type=DeploymentType.CONSUMER_GPU.value,
                total_cost_monthly_usd=total_monthly,
                total_cost_yearly_usd=total_yearly,
                cost_per_1k_tokens=cost_per_1k_tokens,
                cost_per_request=cost_per_request,
                cost_breakdown=cost_breakdown,
                hardware_costs=hardware_costs,
                operational_costs=operational_costs,
                cloud_costs=[],
                roi_analysis=roi_analysis,
                optimization_recommendations=[opt.description for opt in optimizations],
                comparison_baselines=self.baseline_pricing,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error calculating consumer GPU costs: {e}")
            raise
    
    def calculate_cloud_costs(self, cloud_config: Dict[str, Any], 
                            workload: WorkloadProfile) -> CostAnalysis:
        """Calculate costs for cloud deployment."""
        try:
            provider = cloud_config.get('provider', 'aws')
            instance_type = cloud_config.get('instance_type')
            
            if provider not in self.cloud_pricing:
                raise ValueError(f"Unsupported cloud provider: {provider}")
            
            if instance_type not in self.cloud_pricing[provider]:
                raise ValueError(f"Unsupported instance type: {instance_type}")
            
            instance_spec = self.cloud_pricing[provider][instance_type]
            
            cloud_cost = CloudCost(
                provider=provider,
                instance_type=instance_type,
                hourly_cost_usd=instance_spec['cost'],
                gpu_count=instance_spec['gpus'],
                gpu_type=instance_spec['gpu_type'],
                vram_gb=instance_spec['vram'],
                cpu_cores=cloud_config.get('cpu_cores', 64),
                memory_gb=cloud_config.get('memory_gb', 512),
                storage_gb=cloud_config.get('storage_gb', 1000),
                setup_cost_usd=cloud_config.get('setup_cost', 0)
            )
            
            # Calculate monthly costs
            hours_per_month = workload.uptime_hours_per_day * 30
            monthly_instance_cost = cloud_cost.hourly_cost_usd * hours_per_month
            monthly_storage_cost = (cloud_config.get('storage_gb', 1000) * 0.10)  # $0.10 per GB per month
            monthly_bandwidth_cost = 50  # Estimated bandwidth costs
            
            total_monthly = monthly_instance_cost + monthly_storage_cost + monthly_bandwidth_cost + cloud_cost.setup_cost_usd
            total_yearly = total_monthly * 12
            
            # Calculate per-token and per-request costs
            monthly_requests = workload.requests_per_day * 30
            monthly_tokens = monthly_requests * workload.avg_tokens_per_request
            
            cost_per_1k_tokens = (total_monthly / monthly_tokens) * 1000 if monthly_tokens > 0 else 0
            cost_per_request = total_monthly / monthly_requests if monthly_requests > 0 else 0
            
            # Cost breakdown
            cost_breakdown = {
                'compute_instances': monthly_instance_cost,
                'storage': monthly_storage_cost,
                'bandwidth': monthly_bandwidth_cost,
                'setup_costs': cloud_cost.setup_cost_usd
            }
            
            # ROI analysis
            roi_analysis = self._calculate_roi_analysis(
                total_monthly, cost_per_1k_tokens, workload, DeploymentType.CLOUD_INSTANCE
            )
            
            # Generate optimization recommendations
            optimizations = self._generate_cloud_optimizations(cloud_config, workload)
            
            return CostAnalysis(
                deployment_type=DeploymentType.CLOUD_INSTANCE.value,
                total_cost_monthly_usd=total_monthly,
                total_cost_yearly_usd=total_yearly,
                cost_per_1k_tokens=cost_per_1k_tokens,
                cost_per_request=cost_per_request,
                cost_breakdown=cost_breakdown,
                hardware_costs=[],
                operational_costs=[],
                cloud_costs=[cloud_cost],
                roi_analysis=roi_analysis,
                optimization_recommendations=[opt.description for opt in optimizations],
                comparison_baselines=self.baseline_pricing,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error calculating cloud costs: {e}")
            raise
    
    def compare_deployment_options(self, workload: WorkloadProfile) -> Dict[str, CostAnalysis]:
        """Compare multiple deployment options."""
        try:
            comparisons = {}
            
            # Consumer GPU option (4x RTX 4090)
            consumer_config = {
                'gpu_type': 'RTX 4090',
                'gpu_count': 4,
                'memory_gb': 128,
                'storage_gb': 2000
            }
            comparisons['Consumer GPU (4x RTX 4090)'] = self.calculate_consumer_gpu_costs(
                consumer_config, workload
            )
            
            # Consumer GPU option (4x Tesla M10) - budget option
            budget_config = {
                'gpu_type': 'Tesla M10',
                'gpu_count': 4,
                'memory_gb': 256,
                'storage_gb': 2000
            }
            comparisons['Budget Consumer (4x Tesla M10)'] = self.calculate_consumer_gpu_costs(
                budget_config, workload
            )
            
            # Cloud options
            cloud_configs = [
                {'provider': 'aws', 'instance_type': 'g5.48xlarge'},
                {'provider': 'runpod', 'instance_type': 'RTX 4090'},
                {'provider': 'vast', 'instance_type': 'RTX 4090'}
            ]
            
            for config in cloud_configs:
                provider = config['provider']
                instance = config['instance_type']
                name = f"{provider.upper()} {instance}"
                comparisons[name] = self.calculate_cloud_costs(config, workload)
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Error comparing deployment options: {e}")
            return {}
    
    def _calculate_operational_costs(self, hardware_costs: List[HardwareCost], 
                                   workload: WorkloadProfile) -> List[OperationalCost]:
        """Calculate operational costs for hardware deployment."""
        operational_costs = []
        
        # Calculate power consumption
        total_power_kw = sum(cost.power_consumption_watts * cost.quantity for cost in hardware_costs) / 1000
        
        # Add cooling power (typically 30% additional)
        total_power_with_cooling = total_power_kw * (1 + self.default_operational_costs['cooling_factor'])
        
        # Electricity cost
        hours_per_month = workload.uptime_hours_per_day * 30
        monthly_electricity_cost = (total_power_with_cooling * 
                                  self.default_operational_costs['electricity_per_kwh'] * 
                                  hours_per_month)
        
        electricity_cost = OperationalCost(
            cost_type='electricity',
            monthly_cost_usd=monthly_electricity_cost,
            description=f"Power consumption: {total_power_with_cooling:.1f}kW"
        )
        operational_costs.append(electricity_cost)
        
        # Cooling cost (already included in electricity, but tracked separately)
        cooling_cost = OperationalCost(
            cost_type='cooling',
            monthly_cost_usd=monthly_electricity_cost * self.default_operational_costs['cooling_factor'],
            description="Additional cooling and ventilation"
        )
        operational_costs.append(cooling_cost)
        
        # Internet cost
        internet_cost = OperationalCost(
            cost_type='internet',
            monthly_cost_usd=self.default_operational_costs['internet_monthly'],
            description="High-speed internet connectivity"
        )
        operational_costs.append(internet_cost)
        
        # Maintenance cost
        total_hardware_value = sum(cost.unit_cost_usd * cost.quantity for cost in hardware_costs)
        monthly_maintenance_cost = (total_hardware_value * 
                                  self.default_operational_costs['maintenance_factor'] / 12)
        
        maintenance_cost = OperationalCost(
            cost_type='maintenance',
            monthly_cost_usd=monthly_maintenance_cost,
            description="Hardware maintenance and support"
        )
        operational_costs.append(maintenance_cost)
        
        # Staffing cost
        staffing_cost = OperationalCost(
            cost_type='staffing',
            monthly_cost_usd=(self.default_operational_costs['staffing_hourly'] * 
                            self.default_operational_costs['staffing_hours_monthly']),
            description="DevOps and system administration"
        )
        operational_costs.append(staffing_cost)
        
        return operational_costs
    
    def _calculate_monthly_hardware_cost(self, hardware_costs: List[HardwareCost]) -> float:
        """Calculate monthly amortization cost for hardware."""
        monthly_cost = 0.0
        
        for cost in hardware_costs:
            total_cost = cost.unit_cost_usd * cost.quantity
            resale_value = total_cost * (cost.resale_value_percent / 100)
            depreciation = total_cost - resale_value
            monthly_depreciation = depreciation / (cost.lifespan_years * 12)
            monthly_cost += monthly_depreciation
        
        return monthly_cost
    
    def _calculate_roi_analysis(self, monthly_cost: float, cost_per_1k_tokens: float,
                              workload: WorkloadProfile, deployment_type: DeploymentType) -> Dict[str, Any]:
        """Calculate ROI analysis."""
        try:
            # Compare against baseline services
            baseline_costs = {}
            monthly_tokens = workload.requests_per_day * 30 * workload.avg_tokens_per_request
            
            for service, price_per_1k in self.baseline_pricing.items():
                baseline_monthly_cost = (monthly_tokens / 1000) * price_per_1k
                baseline_costs[service] = {
                    'monthly_cost': baseline_monthly_cost,
                    'savings_vs_self_hosted': baseline_monthly_cost - monthly_cost,
                    'cost_ratio': baseline_monthly_cost / monthly_cost if monthly_cost > 0 else 0
                }
            
            # Break-even analysis
            cheapest_baseline = min(baseline_costs.values(), key=lambda x: x['monthly_cost'])
            break_even_months = 0
            if cheapest_baseline['savings_vs_self_hosted'] > 0:
                # Calculate break-even considering upfront costs for self-hosted
                if deployment_type == DeploymentType.CONSUMER_GPU:
                    # Assume initial hardware investment
                    initial_investment = monthly_cost * 6  # Rough estimate
                    break_even_months = initial_investment / cheapest_baseline['savings_vs_self_hosted']
            
            # Scalability analysis
            scalability_score = self._calculate_scalability_score(deployment_type, workload)
            
            return {
                'baseline_comparisons': baseline_costs,
                'break_even_months': break_even_months,
                'scalability_score': scalability_score,
                'cost_efficiency_rating': self._calculate_cost_efficiency_rating(cost_per_1k_tokens),
                'total_3_year_cost': monthly_cost * 36,
                'total_5_year_cost': monthly_cost * 60
            }
            
        except Exception as e:
            logger.error(f"Error calculating ROI analysis: {e}")
            return {}
    
    def _calculate_scalability_score(self, deployment_type: DeploymentType, 
                                   workload: WorkloadProfile) -> float:
        """Calculate scalability score (0-100)."""
        base_score = 50
        
        if deployment_type == DeploymentType.CLOUD_INSTANCE:
            base_score += 30  # Cloud is more scalable
        elif deployment_type == DeploymentType.CONSUMER_GPU:
            base_score += 10  # Limited but possible scaling
        
        # Adjust for current workload vs capacity
        if workload.peak_concurrent_requests > 50:
            base_score -= 10  # High load reduces scalability
        
        # Growth factor
        if workload.growth_rate_monthly > 0.1:  # 10% monthly growth
            base_score -= 15  # High growth challenges scalability
        
        return max(0, min(100, base_score))
    
    def _calculate_cost_efficiency_rating(self, cost_per_1k_tokens: float) -> str:
        """Rate cost efficiency based on market comparison."""
        avg_baseline = statistics.mean(self.baseline_pricing.values())
        
        if cost_per_1k_tokens <= avg_baseline * 0.3:
            return 'excellent'
        elif cost_per_1k_tokens <= avg_baseline * 0.5:
            return 'very_good'
        elif cost_per_1k_tokens <= avg_baseline * 0.7:
            return 'good'
        elif cost_per_1k_tokens <= avg_baseline:
            return 'competitive'
        else:
            return 'expensive'
    
    def _generate_consumer_gpu_optimizations(self, gpu_config: Dict[str, Any],
                                          hardware_costs: List[HardwareCost],
                                          operational_costs: List[OperationalCost],
                                          workload: WorkloadProfile) -> List[CostOptimization]:
        """Generate optimization recommendations for consumer GPU deployment."""
        optimizations = []
        
        # Power optimization
        power_cost = next((c for c in operational_costs if c.cost_type == 'electricity'), None)
        if power_cost and power_cost.monthly_cost_usd > 500:
            optimizations.append(CostOptimization(
                optimization_type='power_efficiency',
                current_cost_monthly=power_cost.monthly_cost_usd,
                optimized_cost_monthly=power_cost.monthly_cost_usd * 0.8,
                savings_monthly=power_cost.monthly_cost_usd * 0.2,
                savings_percent=20.0,
                implementation_difficulty='moderate',
                implementation_time_days=7,
                description='Implement power optimization and efficient cooling',
                action_items=[
                    'Undervolt GPUs for better efficiency',
                    'Optimize cooling curves and fan profiles',
                    'Schedule workloads during off-peak electricity hours',
                    'Implement dynamic GPU scaling'
                ]
            ))
        
        # Hardware optimization
        gpu_type = gpu_config.get('gpu_type', 'RTX 4090')
        if gpu_type == 'RTX 4090' and workload.requests_per_day < 1000:
            optimizations.append(CostOptimization(
                optimization_type='hardware_rightsizing',
                current_cost_monthly=sum(c.unit_cost_usd * c.quantity for c in hardware_costs if c.component_type == 'gpu') / 48,  # Monthly amortization
                optimized_cost_monthly=sum(c.unit_cost_usd * c.quantity for c in hardware_costs if c.component_type == 'gpu') * 0.6 / 48,
                savings_monthly=sum(c.unit_cost_usd * c.quantity for c in hardware_costs if c.component_type == 'gpu') * 0.4 / 48,
                savings_percent=40.0,
                implementation_difficulty='easy',
                implementation_time_days=1,
                description='Switch to RTX 3090 or Tesla M10 for lower workloads',
                action_items=[
                    'Evaluate actual GPU utilization',
                    'Consider RTX 3090 for 80% cost savings',
                    'Tesla M10 for budget-conscious deployments',
                    'Maintain same VRAM capacity with multiple smaller GPUs'
                ]
            ))
        
        return optimizations
    
    def _generate_cloud_optimizations(self, cloud_config: Dict[str, Any],
                                    workload: WorkloadProfile) -> List[CostOptimization]:
        """Generate optimization recommendations for cloud deployment."""
        optimizations = []
        
        provider = cloud_config.get('provider', 'aws')
        
        # Spot instance optimization
        if provider in ['aws', 'gcp'] and workload.uptime_hours_per_day < 20:
            optimizations.append(CostOptimization(
                optimization_type='spot_instances',
                current_cost_monthly=0,  # Will be calculated from context
                optimized_cost_monthly=0,
                savings_monthly=0,
                savings_percent=60.0,
                implementation_difficulty='moderate',
                implementation_time_days=3,
                description='Use spot instances for 60-80% cost savings',
                action_items=[
                    'Implement fault-tolerant request handling',
                    'Use spot instance pools across multiple AZs',
                    'Implement automatic failover to on-demand instances',
                    'Schedule batch workloads during low-demand periods'
                ]
            ))
        
        # Reserved instance optimization
        if workload.uptime_hours_per_day > 20:
            optimizations.append(CostOptimization(
                optimization_type='reserved_instances',
                current_cost_monthly=0,
                optimized_cost_monthly=0,
                savings_monthly=0,
                savings_percent=30.0,
                implementation_difficulty='easy',
                implementation_time_days=1,
                description='Use 1-year reserved instances for steady workloads',
                action_items=[
                    'Analyze usage patterns over past 3 months',
                    'Purchase reserved instances for baseline capacity',
                    'Use on-demand instances for peak traffic',
                    'Consider 3-year reservations for maximum savings'
                ]
            ))
        
        return optimizations
    
    def calculate_scaling_costs(self, base_analysis: CostAnalysis, 
                              scaling_factors: List[float]) -> Dict[float, Dict[str, float]]:
        """Calculate costs at different scale factors."""
        scaling_costs = {}
        
        for factor in scaling_factors:
            scaled_monthly_cost = base_analysis.total_cost_monthly_usd
            
            # Different scaling models based on deployment type
            if base_analysis.deployment_type == DeploymentType.CONSUMER_GPU.value:
                # Hardware scales in discrete steps
                if factor <= 1.0:
                    # No scaling needed
                    pass
                elif factor <= 2.0:
                    # Need to double hardware
                    hardware_cost = base_analysis.cost_breakdown.get('hardware_amortization', 0)
                    scaled_monthly_cost += hardware_cost
                else:
                    # Multiple systems needed
                    systems_needed = int(factor)
                    hardware_cost = base_analysis.cost_breakdown.get('hardware_amortization', 0)
                    operational_factor = 0.8  # Some operational efficiency at scale
                    scaled_monthly_cost = (hardware_cost * systems_needed + 
                                         (scaled_monthly_cost - hardware_cost) * systems_needed * operational_factor)
            
            elif base_analysis.deployment_type == DeploymentType.CLOUD_INSTANCE.value:
                # Cloud scales more linearly
                scaled_monthly_cost = base_analysis.total_cost_monthly_usd * factor
                # Volume discounts at scale
                if factor > 5:
                    scaled_monthly_cost *= 0.9  # 10% volume discount
                elif factor > 10:
                    scaled_monthly_cost *= 0.85  # 15% volume discount
            
            scaling_costs[factor] = {
                'monthly_cost': scaled_monthly_cost,
                'cost_per_1k_tokens': base_analysis.cost_per_1k_tokens / factor,  # Better efficiency at scale
                'cost_per_request': base_analysis.cost_per_request / factor
            }
        
        return scaling_costs
    
    def export_cost_analysis(self, analysis: CostAnalysis) -> Dict[str, Any]:
        """Export comprehensive cost analysis data."""
        return {
            'export_timestamp': datetime.now().isoformat(),
            'analysis': asdict(analysis),
            'market_pricing': {
                'gpu_prices': self.gpu_prices,
                'cloud_pricing': self.cloud_pricing,
                'baseline_pricing': self.baseline_pricing
            },
            'methodology': {
                'hardware_amortization': 'Linear depreciation over lifespan with resale value',
                'operational_costs': 'Monthly recurring costs including power, cooling, maintenance',
                'cloud_costs': 'On-demand pricing without volume discounts',
                'comparison_baseline': 'Major LLM API providers'
            }
        }


# Global cost calculator instance
cost_calculator = CostCalculator()