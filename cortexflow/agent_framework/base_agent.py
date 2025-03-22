from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import time
import uuid
import json

class AgentResult:
    """Container for standardized agent processing results"""
    
    def __init__(
        self, 
        data: Dict[str, Any],
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        self.data = data
        self.status = status
        self.metadata = metadata or {}
        self.error = error
        self.timestamp = time.time()
        self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "id": self.id,
            "status": self.status,
            "data": self.data,
            "metadata": self.metadata,
            "error": self.error,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentResult':
        """Create result from dictionary"""
        result = cls(
            data=data.get("data", {}),
            status=data.get("status", "success"),
            metadata=data.get("metadata", {}),
            error=data.get("error")
        )
        result.timestamp = data.get("timestamp", time.time())
        result.id = data.get("id", str(uuid.uuid4()))
        return result
    
    @classmethod
    def error_result(cls, error_message: str, metadata: Optional[Dict[str, Any]] = None) -> 'AgentResult':
        """Create an error result"""
        return cls(
            data={},
            status="error",
            metadata=metadata or {},
            error=error_message
        )


class GlobalState:
    """
    Container for the global state shared between agents
    
    This class maintains:
    1. The original input data
    2. Results from each agent
    3. Execution metadata
    4. Shared state for direct agent-to-agent communication
    """
    
    def __init__(self, input_data: Dict[str, Any]):
        self.input = input_data
        self.results: Dict[str, AgentResult] = {}
        self.shared_state: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {
            "start_time": time.time(),
            "agent_sequence": [],
            "status": "initialized",
            "communications": []
        }
    
    def add_result(self, agent_id: str, result: AgentResult) -> None:
        """Add a result from an agent"""
        self.results[agent_id] = result
        self.metadata["agent_sequence"].append(agent_id)
    
    def get_result(self, agent_id: str) -> Optional[AgentResult]:
        """Get result from a specific agent"""
        return self.results.get(agent_id)
    
    def get_agent_data(self, agent_id: str) -> Dict[str, Any]:
        """Get data from a specific agent result"""
        result = self.get_result(agent_id)
        if result:
            return result.data
        return {}
    
    def set_shared_value(self, key: str, value: Any) -> None:
        """Set a value in the shared state"""
        self.shared_state[key] = value
    
    def get_shared_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the shared state"""
        return self.shared_state.get(key, default)
    
    def record_communication(self, from_agent: str, to_agent: str, message_type: str) -> None:
        """Record inter-agent communication"""
        communication_id = f"{from_agent}_to_{to_agent}_{message_type}"
        if communication_id not in self.metadata["communications"]:
            self.metadata["communications"].append(communication_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the global state to a dictionary"""
        return {
            "input": self.input,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "shared_state": self.shared_state,
            "metadata": self.metadata
        }
    
    def serialize(self) -> str:
        """Serialize the global state to JSON"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, serialized_state: str) -> 'GlobalState':
        """Create a GlobalState from serialized JSON"""
        data = json.loads(serialized_state)
        state = cls(data.get("input", {}))
        state.shared_state = data.get("shared_state", {})
        state.metadata = data.get("metadata", {})
        
        # Rebuild results
        for agent_id, result_data in data.get("results", {}).items():
            state.results[agent_id] = AgentResult.from_dict(result_data)
            
        return state


class BaseAgent(ABC):
    """
    Base agent interface for both composable and hybrid architectures
    
    This defines the standardized interface that all agents must implement:
    1. Initialize with a unique ID
    2. Process the global state
    3. Return results in a standardized format
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the agent (load models, resources, etc.)"""
        pass
    
    @abstractmethod
    def process(self, state: GlobalState) -> AgentResult:
        """
        Process the global state and return a result
        
        Args:
            state: The global state object
            
        Returns:
            AgentResult with the processing outcome
        """
        pass
    
    def validate_input(self, state: GlobalState) -> bool:
        """
        Validate that the global state contains required inputs
        
        Args:
            state: The global state object
            
        Returns:
            True if valid, False otherwise
        """
        return True


class AgentOrchestrator(ABC):
    """
    Base orchestrator that coordinates agent execution
    
    This handles:
    1. Agent registration and initialization
    2. Execution sequence determination
    3. State management during execution
    4. Error handling and recovery
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.initialized = False
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
    
    def initialize(self) -> None:
        """Initialize all agents"""
        for agent_id, agent in self.agents.items():
            agent.initialize()
        self.initialized = True
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent workflow
        
        Args:
            input_data: The input data to process
            
        Returns:
            The final processed output
        """
        pass
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
