import time
from typing import Dict, Any, List, Optional, Callable, Tuple
import json
import logging

from agent_framework.base_agent import BaseAgent, GlobalState, AgentResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComposableAgent(BaseAgent):
    """
    Composable agent implementation with standardized processing logic
    
    Each composable agent follows a consistent pattern:
    1. Take the global state as input
    2. Extract relevant data from state
    3. Apply domain-specific processing
    4. Return result in standardized format
    5. Optionally update shared state values
    """
    
    def __init__(
        self, 
        agent_id: str,
        processor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        required_inputs: Optional[List[str]] = None,
        required_agent_results: Optional[List[str]] = None
    ):
        super().__init__(agent_id)
        self.processor = processor
        self.required_inputs = required_inputs or []
        self.required_agent_results = required_agent_results or []
    
    def initialize(self) -> None:
        """Initialize the agent"""
        # Base initialization logic
        self.initialized = True
        logger.info(f"Initialized composable agent: {self.agent_id}")
    
    def process(self, state: GlobalState) -> AgentResult:
        """
        Process the global state using this agent's logic
        
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
            
            # Apply the domain-specific processing logic
            if self.processor:
                process_output = self.processor(process_input)
            else:
                process_output = self._process_internal(process_input)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # in ms
            
            # Create metadata about the processing
            metadata = {
                "processing_time_ms": processing_time,
                "input_size": len(json.dumps(process_input)),
                "output_size": len(json.dumps(process_output))
            }
            
            # Update shared state (if needed)
            self._update_shared_state(state, process_output)
            
            # Return standardized result
            return AgentResult(
                data=process_output,
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
    
    def _process_internal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default internal processing logic (to be overridden by subclasses)
        
        Args:
            input_data: The prepared input data
            
        Returns:
            Processing results as a dictionary
        """
        # This is a placeholder - subclasses should override this
        return {
            "message": f"Default processing by {self.agent_id}",
            "received_input": input_data
        }
    
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


class ComposableAgentOrchestrator:
    """
    Orchestrator for composable agents that manages execution flow
    
    This handles:
    1. Agent registration and initialization
    2. Execution sequence management
    3. Global state maintenance
    4. Error handling and recovery
    """
    
    def __init__(self):
        self.agents: Dict[str, ComposableAgent] = {}
        self.execution_sequence: List[str] = []
        self.initialized = False
        self.global_state: Optional[GlobalState] = None
    
    def register_agent(self, agent: ComposableAgent) -> None:
        """Register an agent with the orchestrator"""
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
        """Initialize all agents"""
        for agent_id, agent in self.agents.items():
            agent.initialize()
        self.initialized = True
        logger.info(f"Initialized {len(self.agents)} composable agents")
    
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
            logger.info(f"Executing agent: {agent_id}")
            
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
        Get metrics about the execution
        
        Returns:
            Dictionary containing execution metrics
        """
        if not self.global_state:
            return {"error": "No execution metrics available"}
        
        metrics = {
            "execution_sequence": self.global_state.metadata["agent_sequence"],
            "total_time_ms": 0,
            "agent_times_ms": {},
            "success_rate": 0
        }
        
        # Calculate metrics
        success_count = 0
        for agent_id, result in self.global_state.results.items():
            # Get processing time
            processing_time = result.metadata.get("processing_time_ms", 0)
            metrics["agent_times_ms"][agent_id] = processing_time
            metrics["total_time_ms"] += processing_time
            
            # Count successes
            if result.status == "success":
                success_count += 1
        
        # Calculate success rate
        if self.global_state.results:
            metrics["success_rate"] = success_count / len(self.global_state.results)
        
        return metrics


# Example specialized agent implementations for various domains

class StressModelingAgent(ComposableAgent):
    """Specialized agent for stress modeling"""
    
    def __init__(self, agent_id: str = "stress_modeling"):
        super().__init__(
            agent_id=agent_id,
            required_inputs=["stress_factors"],
            required_agent_results=[]
        )
    
    def _process_internal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process stress data and model stress levels
        
        Args:
            input_data: The prepared input data
            
        Returns:
            Stress modeling results
        """
        # Extract stress factors from input
        stress_factors = input_data["input"].get("stress_factors", [])
        context = input_data["input"].get("context", "")
        
        # Simple stress level calculation based on factors
        base_stress = 4.0  # Moderate baseline
        stress_level = base_stress + (len(stress_factors) * 1.2)
        
        # Cap at maximum of 10
        stress_level = min(10.0, stress_level)
        
        # Generate interventions based on stress level and factors
        interventions = []
        
        # Always recommend basic interventions
        interventions.append("deep_breathing")
        
        if "deadline" in stress_factors:
            interventions.append("time_management")
            interventions.append("prioritization")
            
        if "workload" in stress_factors:
            interventions.append("task_delegation")
            interventions.append("workload_distribution")
            
        if "perfectionism" in stress_factors:
            interventions.append("cognitive_restructuring")
            interventions.append("setting_realistic_goals")
            
        if stress_level > 7.0:
            interventions.append("stress_counseling")
            
        return {
            "stress_level": stress_level,
            "triggers": stress_factors,
            "recommended_interventions": interventions
        }
    
    def _update_shared_state(self, state: GlobalState, output: Dict[str, Any]) -> None:
        """Update shared state with stress level information"""
        state.set_shared_value("stress_level", output["stress_level"])
        state.set_shared_value("stress_triggers", output["triggers"])


class NeurochemicalInteractionAgent(ComposableAgent):
    """Specialized agent for neurochemical interaction modeling"""
    
    def __init__(self, agent_id: str = "neurochemical_interaction"):
        super().__init__(
            agent_id=agent_id,
            required_inputs=[],
            required_agent_results=["stress_modeling"]
        )
    
    def _process_internal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Model neurochemical interactions based on stress
        
        Args:
            input_data: The prepared input data
            
        Returns:
            Neurochemical modeling results
        """
        # Get stress results
        stress_results = input_data["agent_results"]["stress_modeling"]
        stress_level = stress_results["stress_level"]
        triggers = stress_results["triggers"]
        
        # Extract user profile if available
        user_profile = input_data["input"].get("user_profile", {})
        baseline_cortisol = user_profile.get("baseline_cortisol", 10.0)
        cortisol_reactivity = user_profile.get("cortisol_reactivity", 1.0)
        
        # Model neurochemical concentrations
        cortisol = baseline_cortisol + (stress_level * 1.5 * cortisol_reactivity)
        adrenaline = 5.0 + (stress_level * 1.2)
        dopamine = max(1.0, 15.0 - (stress_level * 0.8))
        serotonin = max(1.0, 20.0 - (stress_level * 0.6))
        
        # Identify neurochemical patterns
        patterns = []
        
        if cortisol > 20.0:
            patterns.append("acute_stress_response")
            
        if cortisol > 15.0:
            patterns.append("elevated_cortisol")
            
        if dopamine < 10.0:
            patterns.append("reward_seeking_behavior")
            
        if serotonin < 15.0:
            patterns.append("mood_regulation_challenge")
            
        # Generate recommendations
        recommendations = []
        
        if cortisol > 20.0:
            recommendations.append("physical_activity")
            recommendations.append("deep_breathing")
            
        if dopamine < 10.0:
            recommendations.append("reward_scheduling")
            
        if serotonin < 15.0:
            recommendations.append("sunlight_exposure")
            recommendations.append("positive_social_interaction")
            
        return {
            "concentrations": {
                "cortisol": cortisol,
                "adrenaline": adrenaline,
                "dopamine": dopamine,
                "serotonin": serotonin
            },
            "patterns": patterns,
            "recommendations": recommendations
        }


class CognitiveAssessmentAgent(ComposableAgent):
    """Specialized agent for cognitive assessment"""
    
    def __init__(self, agent_id: str = "cognitive_assessment"):
        super().__init__(
            agent_id=agent_id,
            required_inputs=[],
            required_agent_results=["stress_modeling", "neurochemical_interaction"]
        )
    
    def _process_internal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess cognitive impacts of stress and neurochemical state
        
        Args:
            input_data: The prepared input data
            
        Returns:
            Cognitive assessment results
        """
        # Get stress and neurochemical results
        stress_results = input_data["agent_results"]["stress_modeling"]
        neuro_results = input_data["agent_results"]["neurochemical_interaction"]
        
        stress_level = stress_results["stress_level"]
        concentrations = neuro_results["concentrations"]
        
        # Extract relevant neurochemical levels
        cortisol = concentrations["cortisol"]
        dopamine = concentrations["dopamine"]
        
        # Model cognitive impacts
        cognitive_load = min(10.0, stress_level + (cortisol - 15) / 5.0)
        attention_span = max(1.0, 10.0 - (stress_level * 0.8))
        working_memory = max(1.0, 10.0 - (cognitive_load * 0.5))
        decision_making = max(1.0, 10.0 - (cognitive_load * 0.6) + (dopamine * 0.2))
        
        # Generate recommendations
        recommendations = []
        
        if cognitive_load > 7.0:
            recommendations.append("task_simplification")
            
        if attention_span < 5.0:
            recommendations.append("pomodoro_technique")
            
        if working_memory < 5.0:
            recommendations.append("memory_aids")
            
        if decision_making < 5.0:
            recommendations.append("decision_frameworks")
            
        return {
            "cognitive_load": cognitive_load,
            "attention_span": attention_span,
            "working_memory": working_memory,
            "decision_making": decision_making,
            "recommendations": recommendations
        }


class InterventionRecommendationAgent(ComposableAgent):
    """Specialized agent for comprehensive intervention recommendations"""
    
    def __init__(self, agent_id: str = "intervention_recommendation"):
        super().__init__(
            agent_id=agent_id,
            required_inputs=[],
            required_agent_results=[
                "stress_modeling", 
                "neurochemical_interaction", 
                "cognitive_assessment"
            ]
        )
    
    def _process_internal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive intervention recommendations
        
        Args:
            input_data: The prepared input data
            
        Returns:
            Intervention recommendations
        """
        # Collect all recommendations from previous agents
        all_recommendations = []
        
        # Stress management interventions
        stress_results = input_data["agent_results"]["stress_modeling"]
        all_recommendations.extend(stress_results.get("recommended_interventions", []))
        
        # Neurochemical interventions
        neuro_results = input_data["agent_results"]["neurochemical_interaction"]
        all_recommendations.extend(neuro_results.get("recommendations", []))
        
        # Cognitive interventions
        cognitive_results = input_data["agent_results"]["cognitive_assessment"]
        all_recommendations.extend(cognitive_results.get("recommendations", []))
        
        # Remove duplicates
        unique_recommendations = list(set(all_recommendations))
        
        # Categorize interventions
        stress_reduction = []
        cognitive_support = []
        environmental = []
        
        for rec in unique_recommendations:
            if any(term in rec.lower() for term in ["breathing", "stress", "counseling"]):
                stress_reduction.append(rec)
            elif any(term in rec.lower() for term in ["cognitive", "memory", "decision", "pomodoro"]):
                cognitive_support.append(rec)
            elif any(term in rec.lower() for term in ["environment", "workspace"]):
                environmental.append(rec)
        
        # Determine priorities based on stress level
        stress_level = stress_results["stress_level"]
        priority_interventions = []
        
        if stress_level > 7.0:
            priority_interventions.append("stress_reduction")
            
        if cognitive_results["cognitive_load"] > 6.0:
            priority_interventions.append("cognitive_support")
            
        # Environmental modifications are usually low priority but still important
        priority_interventions.append("environmental_modification")
        
        # Create detailed recommendations
        detailed_recommendations = []
        
        # Add stress reduction interventions
        for rec in stress_reduction:
            detailed_recommendations.append({
                "name": rec,
                "frequency": "daily" if stress_level > 7.0 else "3x weekly",
                "duration": "15 minutes"
            })
            
        # Add cognitive support interventions
        for rec in cognitive_support:
            detailed_recommendations.append({
                "name": rec,
                "frequency": "as needed",
                "duration": "30 minutes"
            })
            
        # Add environmental interventions
        for rec in environmental:
            detailed_recommendations.append({
                "name": rec,
                "frequency": "once",
                "duration": "varies"
            })
        
        # Expected outcomes
        expected_outcomes = ["reduced_stress"]
        
        if cognitive_results["cognitive_load"] > 5.0:
            expected_outcomes.append("improved_focus")
            
        if cognitive_results["decision_making"] < 6.0:
            expected_outcomes.append("better_decision_making")
        
        return {
            "priority_interventions": priority_interventions,
            "detailed_recommendations": detailed_recommendations,
            "expected_outcomes": expected_outcomes
        }
