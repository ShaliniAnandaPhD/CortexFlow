import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

from agent_framework.base_agent import BaseAgent, GlobalState, AgentResult
from agent_framework.framework_adapter import FrameworkAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridAgent(BaseAgent):
    """
    Hybrid agent that combines the composable interface with framework adapters
    
    This class:
    1. Maintains the standardized composable interface
    2. Uses framework-specific adapters for specialized processing
    3. Handles conversion between standardized and framework-specific formats
    4. Monitors adapter overhead and accuracy improvements
    """
    
    def __init__(
        self, 
        agent_id: str,
        adapter: FrameworkAdapter,
        required_inputs: Optional[List[str]] = None,
        required_agent_results: Optional[List[str]] = None
    ):
        super().__init__(agent_id)
        self.adapter = adapter
        self.required_inputs = required_inputs or []
        self.required_agent_results = required_agent_results or []
    
    def initialize(self) -> None:
        """Initialize the agent and its adapter"""
        # Initialize the framework adapter
        self.adapter.initialize()
        self.initialized = True
        logger.info(f"Initialized hybrid agent: {self.agent_id} with {self.adapter.framework_name} adapter")
    
    def process(self, state: GlobalState) -> AgentResult:
        """
        Process the global state using the framework adapter
        
        Args:
            state: The global state object
            
        Returns:
            AgentResult containing the processing outcome
        """
        if not self.initialized:
            self.initialize()
        
        # Validate input requirements
        if not self.validate_input(state):
            error_msg = f"Missing required inputs for agent {self.agent_id}"
            logger.error(error_msg)
            return AgentResult.error_result(error_msg)
        
        try:
            # Start timing
            start_time = time.time()
            
            # Extract relevant data from the global state
            process_input = self._prepare_input(state)
            
            # Apply the adapter to process the input
            composable_output, metrics = self.adapter.run(process_input)
            
            # Calculate total processing time
            processing_time = (time.time() - start_time) * 1000  # in ms
            
            # Create metadata about the processing
            metadata = {
                "processing_time_ms": processing_time,
                "adapter_metrics": metrics,
                "framework": self.adapter.framework_name,
                "input_size": len(json.dumps(process_input)),
                "output_size": len(json.dumps(composable_output))
            }
            
            # Update shared state (if needed)
            self._update_shared_state(state, composable_output)
            
            # Return standardized result
            return AgentResult(
                data=composable_output,
                status="success",
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Error in {self.agent_id}: {str(e)}"
            logger.exception(error_msg)
            return AgentResult.error_result(error_msg)
    
    def validate_input(self, state: GlobalState) -> bool:
        """
        Validate that the global state contains all required inputs
        
        Args:
            state: The global state object
            
        Returns:
            True if all required inputs are present, False otherwise
        """
        # Check required input fields
        for field in self.required_inputs:
            if field not in state.input:
                logger.warning(f"Missing required input field: {field}")
                return False
        
        # Check required agent results
        for agent_id in self.required_agent_results:
            if agent_id not in state.results:
                logger.warning(f"Missing required agent result: {agent_id}")
                return False
            if state.results[agent_id].status != "success":
                logger.warning(f"Required agent {agent_id} did not complete successfully")
                return False
        
        return True
    
    def _prepare_input(self, state: GlobalState) -> Dict[str, Any]:
        """
        Extract and prepare input data from global state
        
        Args:
            state: The global state object
            
        Returns:
            Dictionary containing the prepared input for processing
        """
        # Start with the original input
        prepared_input = {
            "input": state.input,
            "agent_id": self.agent_id
        }
        
        # Add results from other agents that this agent depends on
        if self.required_agent_results:
            prepared_input["agent_results"] = {}
            for agent_id in self.required_agent_results:
                if agent_id in state.results:
                    prepared_input["agent_results"][agent_id] = state.results[agent_id].data
        
        # Add any relevant shared state values
        prepared_input["shared_state"] = state.shared_state
        
        return prepared_input
    
    def _update_shared_state(self, state: GlobalState, output: Dict[str, Any]) -> None:
        """
        Update the shared state with relevant values from output
        
        Args:
            state: The global state object
            output: The processing output
        """
        # Default implementation does nothing
        # Subclasses can override to update specific shared state values
        pass


class HybridAgentOrchestrator:
    """
    Orchestrator for hybrid agents that manages execution flow
    
    This handles:
    1. Agent registration and initialization
    2. Execution sequence management
    3. Global state maintenance
    4. Adapter metrics collection
    5. Error handling and recovery
    """
    
    def __init__(self):
        self.agents: Dict[str, HybridAgent] = {}
        self.execution_sequence: List[str] = []
        self.initialized = False
        self.global_state: Optional[GlobalState] = None
    
    def register_agent(self, agent: HybridAgent) -> None:
        """Register a hybrid agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
    
    def set_execution_sequence(self, sequence: List[str]) -> None:
        """
        Set the execution sequence for the agents
        
        Args:
            sequence: List of agent IDs in execution order
        """
        # Validate that all agents in the sequence are registered
        for agent_id in sequence:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} in execution sequence is not registered")
        
        self.execution_sequence = sequence
    
    def initialize(self) -> None:
        """Initialize all agents and their adapters"""
        for agent_id, agent in self.agents.items():
            agent.initialize()
        self.initialized = True
        logger.info(f"Initialized {len(self.agents)} hybrid agents")
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent workflow
        
        Args:
            input_data: The input data to process
            
        Returns:
            The final processed output
        """
        if not self.initialized:
            self.initialize()
        
        # If no execution sequence is set, use all agents in registration order
        if not self.execution_sequence:
            self.execution_sequence = list(self.agents.keys())
            logger.info(f"Using default execution sequence: {self.execution_sequence}")
        
        # Initialize global state
        self.global_state = GlobalState(input_data)
        
        # Execute each agent in sequence
        for agent_id in self.execution_sequence:
            if agent_id not in self.agents:
                error_msg = f"Agent {agent_id} not found in registered agents"
                logger.error(error_msg)
                continue
                
            agent = self.agents[agent_id]
            logger.info(f"Executing agent: {agent_id} using {agent.adapter.framework_name}")
            
            # Process with this agent
            result = agent.process(self.global_state)
            
            # Store the result in global state
            self.global_state.add_result(agent_id, result)
            
            # Check for errors and handle accordingly
            if result.status == "error":
                logger.warning(f"Agent {agent_id} returned error: {result.error}")
                # Continue execution by default, but implementations could choose to abort
        
        # Compile final output from all agent results
        return self._compile_final_output()
    
    def _compile_final_output(self) -> Dict[str, Any]:
        """
        Compile the final output from all agent results
        
        Returns:
            Dictionary containing the compiled output
        """
        if not self.global_state:
            return {"error": "No global state available"}
        
        # Extract successful results from each agent
        output = {}
        for agent_id, result in self.global_state.results.items():
            if result.status == "success":
                output[agent_id] = result.data
        
        return output
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the execution including adapter metrics
        
        Returns:
            Dictionary containing execution metrics
        """
        if not self.global_state:
            return {"error": "No execution metrics available"}
        
        metrics = {
            "execution_sequence": self.global_state.metadata["agent_sequence"],
            "total_time_ms": 0,
            "adapter_overhead_ms": 0,
            "framework_execution_ms": 0,
            "agent_metrics": {},
            "success_rate": 0
        }
        
        # Calculate metrics
        success_count = 0
        for agent_id, result in self.global_state.results.items():
            # Skip errors
            if result.status != "success":
                continue
                
            # Get processing time
            processing_time = result.metadata.get("processing_time_ms", 0)
            
            # Get adapter metrics if available
            adapter_metrics = result.metadata.get("adapter_metrics", {})
            adapter_overhead = adapter_metrics.get("total_adapter_overhead_ms", 0)
            framework_time = adapter_metrics.get("framework_execution_ms", 0)
            
            # Store individual agent metrics
            metrics["agent_metrics"][agent_id] = {
                "processing_time_ms": processing_time,
                "adapter_overhead_ms": adapter_overhead,
                "framework_execution_ms": framework_time,
                "framework": result.metadata.get("framework", "unknown")
            }
            
            # Update totals
            metrics["total_time_ms"] += processing_time
            metrics["adapter_overhead_ms"] += adapter_overhead
            metrics["framework_execution_ms"] += framework_time
            
            # Count successes
            success_count += 1
        
        # Calculate success rate
        if self.global_state.results:
            metrics["success_rate"] = success_count / len(self.global_state.results)
        
        # Calculate overhead percentage
        if metrics["total_time_ms"] > 0:
            metrics["adapter_overhead_percentage"] = (metrics["adapter_overhead_ms"] / metrics["total_time_ms"]) * 100
        else:
            metrics["adapter_overhead_percentage"] = 0
        
        return metrics


# Example of creating specialized hybrid agents with different adapters

def create_stress_modeling_hybrid_agent(camel_adapter):
    """Create a hybrid agent for stress modeling using CAMEL"""
    return HybridAgent(
        agent_id="stress_modeling",
        adapter=camel_adapter,
        required_inputs=["stress_factors"],
        required_agent_results=[]
    )

def create_neurochemical_hybrid_agent(autogen_adapter):
    """Create a hybrid agent for neurochemical interactions using AutoGen"""
    return HybridAgent(
        agent_id="neurochemical_interaction",
        adapter=autogen_adapter,
        required_inputs=[],
        required_agent_results=["stress_modeling"]
    )

def create_cognitive_assessment_hybrid_agent(langchain_adapter):
    """Create a hybrid agent for cognitive assessment using LangChain"""
    return HybridAgent(
        agent_id="cognitive_assessment",
        adapter=langchain_adapter,
        required_inputs=[],
        required_agent_results=["stress_modeling", "neurochemical_interaction"]
    )

def create_intervention_hybrid_agent(langgraph_adapter):
    """Create a hybrid agent for intervention recommendations using LangGraph"""
    return HybridAgent(
        agent_id="intervention_recommendation",
        adapter=langgraph_adapter,
        required_inputs=[],
        required_agent_results=[
            "stress_modeling", 
            "neurochemical_interaction", 
            "cognitive_assessment"
        ]
    )
