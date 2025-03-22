import time
import json
import psutil
from typing import Dict, Any, List, Tuple, Optional
import importlib

from architectures.base_architecture import BaseArchitecture
from agent_evaluator.metrics import AgentType

class MultiFrameworkArchitecture(BaseArchitecture):
    """
    The original multi-framework architecture using different frameworks 
    for different agent types.
    """
    
    def __init__(self):
        super().__init__()
        self.frameworks = {}
        self.framework_instances = {}
        self.framework_config = {
            AgentType.STRESS_MODELING.value: "camel",
            AgentType.NEUROCHEMICAL_INTERACTION.value: "autogen",
            AgentType.COGNITIVE_ASSESSMENT.value: "langchain",
            AgentType.INTERVENTION_RECOMMENDATION.value: "langgraph",
            AgentType.TEAM_DYNAMICS.value: "crewai",
            AgentType.PATTERN_RECOGNITION.value: "ai_town"
        }
    
    def initialize(self) -> None:
        """Initialize all frameworks needed for the architecture"""
        self.initialized = True
        
        # Load framework adapters
        for agent_type, framework_name in self.framework_config.items():
            try:
                # Attempt to load the adapter for this framework
                module_name = f"adapters.{framework_name}_adapter"
                spec = importlib.util.find_spec(module_name)
                
                if spec:
                    module = importlib.import_module(module_name)
                    adapter_class = getattr(module, f"{framework_name.capitalize()}Adapter")
                    
                    # Initialize the adapter
                    self.frameworks[agent_type] = adapter_class()
                    print(f"Loaded {framework_name} adapter for {agent_type}")
                else:
                    print(f"Warning: Adapter module {module_name} not found")
                    # Use mock adapter for demo purposes
                    self.frameworks[agent_type] = MockFrameworkAdapter(framework_name)
            except Exception as e:
                print(f"Error loading adapter for {agent_type}: {e}")
                # Use mock adapter for demo purposes
                self.frameworks[agent_type] = MockFrameworkAdapter(framework_name)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the multi-framework architecture on the input data"""
        if not self.initialized:
            self.initialize()
        
        output = {}
        agent_types = input_data.get("agent_types", list(self.framework_config.keys()))
        state = {"input": input_data, "intermediate_results": {}}
        
        # Start with stress modeling
        if AgentType.STRESS_MODELING.value in agent_types:
            start_time = time.time() * 1000  # Convert to ms
            
            framework = self.frameworks[AgentType.STRESS_MODELING.value]
            stress_input = self._prepare_agent_input(input_data, AgentType.STRESS_MODELING.value)
            
            # Execute the framework
            stress_results = framework.run(stress_input)
            
            # Measure execution
            end_time = time.time() * 1000
            duration = end_time - start_time
            
            # Log the timeline event
            self.log_timeline_event(
                agent=AgentType.STRESS_MODELING.value,
                event_type="primary_execution",
                start_time=start_time,
                duration=duration,
                metadata={"framework": framework.name}
            )
            
            # Update state and output
            state["intermediate_results"][AgentType.STRESS_MODELING.value] = stress_results
            output[AgentType.STRESS_MODELING.value] = stress_results
            
            # Log serialization when passing data between frameworks
            start_serialization = time.time() * 1000
            serialized_state = json.dumps(state)
            end_serialization = time.time() * 1000
            
            self.log_serialization(
                operation="serialize",
                state_type="intermediate_state",
                duration_ms=end_serialization - start_serialization,
                size_bytes=len(serialized_state.encode('utf-8'))
            )
        
        # Process neurochemical interactions if required
        if AgentType.NEUROCHEMICAL_INTERACTION.value in agent_types:
            start_time = time.time() * 1000
            
            framework = self.frameworks[AgentType.NEUROCHEMICAL_INTERACTION.value]
            
            # Prepare the input with the current state
            neuro_input = self._prepare_agent_input(state, AgentType.NEUROCHEMICAL_INTERACTION.value)
            
            # Execute the framework
            neuro_results = framework.run(neuro_input)
            
            # Measure execution
            end_time = time.time() * 1000
            duration = end_time - start_time
            
            # Log the timeline event
            self.log_timeline_event(
                agent=AgentType.NEUROCHEMICAL_INTERACTION.value,
                event_type="primary_execution",
                start_time=start_time,
                duration=duration,
                metadata={"framework": framework.name}
            )
            
            # Update state and output
            state["intermediate_results"][AgentType.NEUROCHEMICAL_INTERACTION.value] = neuro_results
            output[AgentType.NEUROCHEMICAL_INTERACTION.value] = neuro_results
            
            # Simulate communication between agents
            self._simulate_inter_agent_communication(
                AgentType.STRESS_MODELING.value,
                AgentType.NEUROCHEMICAL_INTERACTION.value,
                state
            )
            
            # Log serialization
            start_serialization = time.time() * 1000
            serialized_state = json.dumps(state)
            end_serialization = time.time() * 1000
            
            self.log_serialization(
                operation="serialize",
                state_type="intermediate_state",
                duration_ms=end_serialization - start_serialization,
                size_bytes=len(serialized_state.encode('utf-8'))
            )
        
        # Continue with other agent types as needed
        for agent_type in agent_types:
            if agent_type in [AgentType.STRESS_MODELING.value, AgentType.NEUROCHEMICAL_INTERACTION.value]:
                continue  # Already processed
                
            if agent_type in self.frameworks:
                start_time = time.time() * 1000
                
                framework = self.frameworks[agent_type]
                agent_input = self._prepare_agent_input(state, agent_type)
                
                # Execute the framework
                agent_results = framework.run(agent_input)
                
                # Measure execution
                end_time = time.time() * 1000
                duration = end_time - start_time
                
                # Log the timeline event
                self.log_timeline_event(
                    agent=agent_type,
                    event_type="primary_execution",
                    start_time=start_time,
                    duration=duration,
                    metadata={"framework": framework.name}
                )
                
                # Update state and output
                state["intermediate_results"][agent_type] = agent_results
                output[agent_type] = agent_results
                
                # Simulate communication with previous agents
                for prev_agent in state["intermediate_results"]:
                    if prev_agent != agent_type:
                        self._simulate_inter_agent_communication(
                            prev_agent,
                            agent_type,
                            state["intermediate_results"][prev_agent]
                        )
                
                # Log serialization
                start_serialization = time.time() * 1000
                serialized_state = json.dumps(state)
                end_serialization = time.time() * 1000
                
                self.log_serialization(
                    operation="serialize",
                    state_type="intermediate_state",
                    duration_ms=end_serialization - start_serialization,
                    size_bytes=len(serialized_state.encode('utf-8'))
                )
        
        return output
    
    def _prepare_agent_input(self, state: Dict[str, Any], agent_type: str) -> Dict[str, Any]:
        """
        Prepare input for a specific agent type, extracting relevant data from the state
        """
        # In a real implementation, this would adapt the format for each framework
        return {
            "agent_type": agent_type,
            "state": state
        }
    
    def _simulate_inter_agent_communication(self, from_agent: str, to_agent: str, message: Any) -> None:
        """
        Simulate communication between agents, measuring latency and payload size
        """
        # Serialize the message to measure payload size
        serialized_message = json.dumps(message)
        payload_size = len(serialized_message.encode('utf-8'))
        
        # Simulate some latency based on payload size
        # In a real system, this would be measured from actual communication
        latency = 50 + (payload_size / 1024) * 10  # Base latency + size-dependent component
        
        # Log the communication event
        self.log_communication(
            from_agent=from_agent,
            to_agent=to_agent,
            message=message,
            latency_ms=latency,
            payload_size_bytes=payload_size
        )


class MockFrameworkAdapter:
    """Mock adapter for demonstration purposes"""
    
    def __init__(self, name: str):
        self.name = name
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate running this framework with the given input"""
        agent_type = input_data.get("agent_type", "unknown")
        
        # Simulate processing delay
        time.sleep(0.1)
        
        # Return simulated output based on agent type
        if agent_type == AgentType.STRESS_MODELING.value:
            return {
                "stress_level": 7.5,
                "triggers": ["deadline", "workload", "interpersonal_conflict"],
                "recommended_interventions": ["breathing_exercise", "time_management", "communication_training"]
            }
        elif agent_type == AgentType.NEUROCHEMICAL_INTERACTION.value:
            return {
                "concentrations": {
                    "cortisol": 18.5,
                    "adrenaline": 12.3,
                    "dopamine": 6.7,
                    "serotonin": 4.2
                },
                "patterns": ["acute_stress_response", "reward_seeking_behavior"],
                "recommendations": ["physical_activity", "meditation", "sleep_improvement"]
            }
        elif agent_type == AgentType.COGNITIVE_ASSESSMENT.value:
            return {
                "cognitive_load": 8.2,
                "attention_span": 4.5,
                "working_memory": 6.8,
                "decision_making": 5.5,
                "recommendations": ["task_simplification", "memory_aids", "decision_frameworks"]
            }
        elif agent_type == AgentType.INTERVENTION_RECOMMENDATION.value:
            return {
                "priority_interventions": ["stress_reduction", "cognitive_support", "environmental_modification"],
                "detailed_recommendations": [
                    {"name": "breathing_exercise", "frequency": "3x daily", "duration": "5 minutes"},
                    {"name": "task_prioritization", "frequency": "daily", "duration": "15 minutes"},
                    {"name": "workspace_optimization", "frequency": "once", "duration": "30 minutes"}
                ],
                "expected_outcomes": ["reduced_stress", "improved_focus", "better_decision_making"]
            }
        elif agent_type == AgentType.TEAM_DYNAMICS.value:
            return {
                "team_stress_factors": ["communication_gaps", "unclear_roles", "deadline_pressure"],
                "interpersonal_dynamics": ["support_network_present", "conflict_avoidance", "information_siloing"],
                "recommended_team_interventions": ["role_clarification", "communication_workshop", "shared_planning_session"]
            }
        elif agent_type == AgentType.PATTERN_RECOGNITION.value:
            return {
                "identified_patterns": ["cyclical_stress_peaks", "intervention_resistance", "environmental_triggers"],
                "pattern_details": {
                    "cyclical_stress_peaks": {"frequency": "weekly", "triggers": ["monday_meetings", "friday_deadlines"]},
                    "intervention_resistance": {"areas": ["meditation", "delegation"], "reasons": ["time_pressure", "trust_issues"]},
                    "environmental_triggers": {"locations": ["open_office", "meeting_room_3"], "factors": ["noise", "interruptions"]}
                },
                "pattern_based_recommendations": ["schedule_restructuring", "environmental_modifications", "intervention_timing_optimization"]
            }
        else:
            return {"result": "No specific output for this agent type"}
