from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

class EvaluationMetric(Enum):
    # Performance metrics
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    RESPONSE_LATENCY = "response_latency"
    COLD_START_TIME = "cold_start_time"
    THROUGHPUT = "throughput"
    CPU_UTILIZATION = "cpu_utilization"
    
    # Accuracy metrics
    ACCURACY = "accuracy"
    
    # Multi-agent specific metrics
    INTER_AGENT_LATENCY = "inter_agent_latency"
    MESSAGE_OVERHEAD = "message_overhead"
    AGENT_COORDINATION = "agent_coordination"
    COMMUNICATION_EFFICIENCY = "communication_efficiency"
    
    # Composability metrics
    COMPONENT_REUSABILITY = "component_reusability"
    INTEGRATION_COMPLEXITY = "integration_complexity"
    STATE_SERIALIZATION_OVERHEAD = "state_serialization_overhead"
    
    # Hybrid architecture metrics
    ADAPTER_OVERHEAD = "adapter_overhead"
    FRAMEWORK_SWITCHING_LATENCY = "framework_switching_latency"
    ACCURACY_PRESERVATION = "accuracy_preservation"

class AgentType(Enum):
    STRESS_MODELING = "stress_modeling"
    NEUROCHEMICAL_INTERACTION = "neurochemical_interaction"
    COGNITIVE_ASSESSMENT = "cognitive_assessment"
    INTERVENTION_RECOMMENDATION = "intervention_recommendation"
    TEAM_DYNAMICS = "team_dynamics"
    PATTERN_RECOGNITION = "pattern_recognition"

class ArchitectureType(Enum):
    MULTI_FRAMEWORK = "multi_framework"  # Original approach with multiple frameworks
    COMPOSABLE = "composable"            # Pure composable architecture
    HYBRID = "hybrid"                    # Composable with framework-specific adapters

@dataclass
class BenchmarkConfig:
    num_iterations: int = 10
    warm_up_iterations: int = 2
    timeout_seconds: int = 30
    memory_tracking: bool = True
    accuracy_threshold: float = 0.85
    ground_truth_path: str = "./data/ground_truth/"
    results_path: str = "./results/"
    frameworks: List[str] = None
    architectures: List[ArchitectureType] = None
    agent_types: List[AgentType] = None
    scenario_categories: List[str] = None
    inter_agent_communication_tracking: bool = True
    state_serialization_tracking: bool = True
    adapter_overhead_tracking: bool = True
    verbose: bool = False
    
    def __post_init__(self):
        if self.frameworks is None:
            self.frameworks = []
        if self.architectures is None:
            self.architectures = [arch.value for arch in ArchitectureType]
        if self.agent_types is None:
            self.agent_types = [agent.value for agent in AgentType]
        if self.scenario_categories is None:
            self.scenario_categories = ["workplace", "academic", "healthcare", "social", "emergency"]

@dataclass
class AgentMetrics:
    agent_type: AgentType
    response_time: float
    accuracy: float
    memory_usage: float
    state_size: int
    message_count: int
    
@dataclass
class ArchitectureMetrics:
    architecture_type: ArchitectureType
    execution_time: float
    memory_usage: float
    response_latency: float
    cold_start_time: float
    throughput: float
    cpu_utilization: float
    accuracy: Dict[str, float]  # Category to accuracy mapping
    inter_agent_latency: Optional[float] = None
    message_overhead: Optional[float] = None
    state_serialization_overhead: Optional[float] = None
    adapter_overhead: Optional[float] = None
    framework_switching_latency: Optional[float] = None
    accuracy_preservation: Optional[float] = None  # % of original accuracy preserved
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "architecture_type": self.architecture_type.value,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "response_latency": self.response_latency,
            "cold_start_time": self.cold_start_time,
            "throughput": self.throughput,
            "cpu_utilization": self.cpu_utilization,
            "accuracy": self.accuracy
        }
        
        # Include multi-agent metrics if available
        if self.inter_agent_latency is not None:
            result["inter_agent_latency"] = self.inter_agent_latency
        if self.message_overhead is not None:
            result["message_overhead"] = self.message_overhead
            
        # Include composability metrics if available
        if self.state_serialization_overhead is not None:
            result["state_serialization_overhead"] = self.state_serialization_overhead
            
        # Include hybrid metrics if available
        if self.adapter_overhead is not None:
            result["adapter_overhead"] = self.adapter_overhead
        if self.framework_switching_latency is not None:
            result["framework_switching_latency"] = self.framework_switching_latency
        if self.accuracy_preservation is not None:
            result["accuracy_preservation"] = self.accuracy_preservation
            
        return result
