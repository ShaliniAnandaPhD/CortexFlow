import time
import json
from typing import Dict, Any, List, Tuple, Optional, Callable
import importlib

from architectures.base_architecture import BaseArchitecture
from architectures.composable_architecture import ComposableArchitecture, Agent
from agent_evaluator.metrics import AgentType
from agent_evaluator.utils import serialize_complex_state, deserialize_complex_state

class HybridArchitecture(BaseArchitecture):
    """
    A hybrid architecture that combines the composable approach with framework-specific adapters.
    
    This architecture:
    1. Uses the composable interface for all agents
    2. Leverages framework-specific adapters for specialized processing
    3. Converts between standardized state and framework-specific formats
    4. Maintains the performance benefits of composability while recovering accuracy
    """
    
    def __init__(self):
        super().__init__()
        self.agents = {}
        self.adapters = {}  # Framework-specific adapters
        self.agent_order = []
        self.framework_mapping = {
            AgentType.STRESS_MODELING.value: "camel",
            AgentType.NEUROCHEMICAL_INTERACTION.value: "autogen", 
            AgentType.COGNITIVE_ASSESSMENT.value: "langchain",
            AgentType.INTERVENTION_RECOMMENDATION.value: "langgraph",
            AgentType.TEAM_DYNAMICS.value: "crewai",
            AgentType.PATTERN_RECOGNITION.value: "ai_town"
        }
        self.global_state = {}
    
    def initialize(self) -> None:
        """Initialize the architecture with agents and adapters"""
        self.initialized = True
        
        # Initialize the composable architecture base
        composable = ComposableArchitecture()
        composable.initialize()
        
        # Load framework adapters
        for agent_type, framework_name in self.framework_mapping.items():
            try:
                # Try to load the adapter for this framework
                module_name = f"adapters.{framework_name}_adapter"
                spec = importlib.util.find_spec(module_name)
                
                if spec:
                    module = importlib.import_module(module_name)
                    adapter_class = getattr(module, f"{framework_name.capitalize()}Adapter")
                    
                    # Initialize the adapter
                    self.adapters[agent_type] = adapter_class()
                    print(f"Loaded {framework_name} adapter for {agent_type}")
                else:
                    print(f"Warning: Adapter module {module_name} not found")
                    # Use mock adapter for demo purposes
                    self.adapters[agent_type] = MockFrameworkAdapter(framework_name, agent_type)
            except Exception as e:
                print(f"Error loading adapter for {agent_type}: {e}")
                # Use mock adapter for demo purposes
                self.adapters[agent_type] = MockFrameworkAdapter(framework_name, agent_type)
        
        # Create hybrid agents that wrap adapters with the composable interface
        for agent_type in AgentType:
            agent_name = agent_type.value
            if agent_name in self.adapters:
                # Create an adapter-wrapped agent
                self.agents[agent_name] = HybridAgent(
                    agent_type=agent_name,
                    adapter=self.adapters[agent_name],
                    state_converter=self._get_state_converter(agent_name)
                )
            else:
                # Fall back to composable implementation for missing adapters
                self.agents[agent_name] = composable.agents.get(agent_name)
        
        # Define the default processing order (same as composable)
        self.agent_order = [
            AgentType.STRESS_MODELING.value,
            AgentType.NEUROCHEMICAL_INTERACTION.value,
            AgentType.COGNITIVE_ASSESSMENT.value,
            AgentType.PATTERN_RECOGNITION.value,
            AgentType.TEAM_DYNAMICS.value,
            AgentType.INTERVENTION_RECOMMENDATION.value
        ]
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the hybrid architecture on the input data"""
        if not self.initialized:
            self.initialize()
        
        # Initialize global state (identical to composable approach)
        self.global_state = {
            "input": input_data,
            "results": {},
            "metadata": {
                "start_time": time.time(),
                "agent_sequence": []
            }
        }
        
        # Determine which agents to run
        agent_types = input_data.get("agent_types", self.agent_order)
        
        # Filter and order agents
        ordered_agents = [agent for agent in self.agent_order if agent in agent_types]
        
        # Process with each agent in sequence
        for agent_type in ordered_agents:
            if agent_type in self.agents:
                agent = self.agents[agent_type]
                
                # Track start time for this agent
                start_time = time.time() * 1000  # Convert to ms
                
                # Process with this agent
                agent_result = agent.process(self.global_state)
                
                # Track end time
                end_time = time.time() * 1000
                duration = end_time - start_time
                
                # Update global state with the result
                self.global_state["results"][agent_type] = agent_result
                self.global_state["metadata"]["agent_sequence"].append(agent_type)
                
                # Log the timeline event
                self.log_timeline_event(
                    agent=agent_type,
                    event_type="hybrid_processing",
                    start_time=start_time,
                    duration=duration,
                    metadata={
                        "status": agent_result.get("status", "unknown"),
                        "framework": self.framework_mapping.get(agent_type, "unknown")
                    }
                )
                
                # Simulate communication with previous agents
                for prev_agent in self.global_state["results"]:
                    if prev_agent != agent_type:
                        # Only log communication if not already processed
                        if f"{prev_agent}_to_{agent_type}" not in self.global_state.get("metadata", {}).get("communications", []):
                            # Determine message payload - just the results for simplicity
                            message = self.global_state["results"][prev_agent]
                            
                            # Simulate communication and log it
                            self._simulate_agent_communication(prev_agent, agent_type, message)
                            
                            # Track that we've logged this communication
                            if "communications" not in self.global_state["metadata"]:
                                self.global_state["metadata"]["communications"] = []
                            self.global_state["metadata"]["communications"].append(f"{prev_agent}_to_{agent_type}")
        
        # Extract final results
        return {
            agent_type: self.global_state["results"][agent_type].get("data", {})
            for agent_type in self.global_state["results"]
        }
    
    def _get_state_converter(self, agent_type: str) -> Callable:
        """
        Get a converter function that transforms between the composable state format
        and the framework-specific format for a particular agent type
        """
        def stress_modeling_converter(state: Dict[str, Any], direction: str) -> Dict[str, Any]:
            """Convert between composable state and CAMEL framework format"""
            if direction == "to_framework":
                # Extract relevant data from composable state
                input_data = state.get("input", {})
                
                # Convert to CAMEL format
                camel_format = {
                    "system_message": "You are a stress modeling assistant.",
                    "user_message": f"Analyze the following stress factors: {input_data.get('stress_factors', [])}",
                    "task": "stress_analysis",
                    "parameters": {
                        "detailed": True,
                        "include_interventions": True
                    }
                }
                
                return camel_format
            else:  # direction == "from_framework"
                # Convert from CAMEL format to composable format
                stress_level = state.get("stress_level", 5.0)
                triggers = state.get("identified_factors", []) or state.get("triggers", [])
                interventions = state.get("suggested_interventions", []) or state.get("recommended_interventions", [])
                
                return {
                    "stress_level": stress_level,
                    "triggers": triggers,
                    "recommended_interventions": interventions
                }
        
        def neurochemical_converter(state: Dict[str, Any], direction: str) -> Dict[str, Any]:
            """Convert between composable state and AutoGen framework format"""
            if direction == "to_framework":
                # Extract relevant data
                input_data = state.get("input", {})
                stress_results = state.get("results", {}).get(AgentType.STRESS_MODELING.value, {}).get("data", {})
                
                # Format for AutoGen
                autogen_format = {
                    "task": "analyze_neurochemical_interactions",
                    "stress_level": stress_results.get("stress_level", 5.0),
                    "stress_triggers": stress_results.get("triggers", []),
                    "user_profile": input_data.get("user_profile", {}),
                    "required_outputs": ["concentrations", "patterns", "recommendations"]
                }
                
                return autogen_format
            else:  # direction == "from_framework"
                # Process AutoGen output format
                concentrations = state.get("chemical_levels", {}) or state.get("concentrations", {})
                patterns = state.get("identified_patterns", []) or state.get("patterns", [])
                recommendations = state.get("suggested_actions", []) or state.get("recommendations", [])
                
                return {
                    "concentrations": concentrations,
                    "patterns": patterns,
                    "recommendations": recommendations
                }
        
        def cognitive_assessment_converter(state: Dict[str, Any], direction: str) -> Dict[str, Any]:
            """Convert between composable state and LangChain framework format"""
            if direction == "to_framework":
                # Extract needed data
                stress_results = state.get("results", {}).get(AgentType.STRESS_MODELING.value, {}).get("data", {})
                neuro_results = state.get("results", {}).get(AgentType.NEUROCHEMICAL_INTERACTION.value, {}).get("data", {})
                
                # Format for LangChain
                langchain_format = {
                    "chain_type": "cognitive_assessment",
                    "inputs": {
                        "stress_level": stress_results.get("stress_level", 5.0),
                        "cortisol_level": neuro_results.get("concentrations", {}).get("cortisol", 15.0),
                        "dopamine_level": neuro_results.get("concentrations", {}).get("dopamine", 10.0),
                        "user_activity": state.get("input", {}).get("activity", "unknown")
                    }
                }
                
                return langchain_format
            else:  # direction == "from_framework"
                # Process LangChain output
                cognitive_load = state.get("cognitive_load", 5.0)
                attention_span = state.get("attention_score", 5.0) or state.get("attention_span", 5.0)
                working_memory = state.get("working_memory_score", 5.0) or state.get("working_memory", 5.0)
                decision_making = state.get("decision_quality", 5.0) or state.get("decision_making", 5.0)
                recommendations = state.get("improvement_strategies", []) or state.get("recommendations", [])
                
                return {
                    "cognitive_load": cognitive_load,
                    "attention_span": attention_span,
                    "working_memory": working_memory,
                    "decision_making": decision_making,
                    "recommendations": recommendations
                }
        
        def pattern_recognition_converter(state: Dict[str, Any], direction: str) -> Dict[str, Any]:
            """Convert between composable state and AI Town framework format"""
            if direction == "to_framework":
                # Gather all results so far
                results = state.get("results", {})
                
                # Format for AI Town pattern recognition
                ai_town_format = {
                    "agent_type": "pattern_recognizer",
                    "memory_length": 10,
                    "data_sources": [
                        {
                            "type": "stress",
                            "data": results.get(AgentType.STRESS_MODELING.value, {}).get("data", {})
                        },
                        {
                            "type": "neurochemical",
                            "data": results.get(AgentType.NEUROCHEMICAL_INTERACTION.value, {}).get("data", {})
                        },
                        {
                            "type": "cognitive",
                            "data": results.get(AgentType.COGNITIVE_ASSESSMENT.value, {}).get("data", {})
                        }
                    ]
                }
                
                return ai_town_format
            else:  # direction == "from_framework"
                # Process AI Town output
                patterns = state.get("detected_patterns", []) or state.get("identified_patterns", [])
                details = state.get("pattern_analytics", {}) or state.get("pattern_details", {})
                recommendations = state.get("suggested_actions", []) or state.get("pattern_based_recommendations", [])
                
                return {
                    "identified_patterns": patterns,
                    "pattern_details": details,
                    "pattern_based_recommendations": recommendations
                }
        
        def team_dynamics_converter(state: Dict[str, Any], direction: str) -> Dict[str, Any]:
            """Convert between composable state and CrewAI framework format"""
            if direction == "to_framework":
                # Extract relevant data
                input_data = state.get("input", {})
                stress_data = state.get("results", {}).get(AgentType.STRESS_MODELING.value, {}).get("data", {})
                
                # Format for CrewAI
                crewai_format = {
                    "crew_task": "team_dynamics_analysis",
                    "team_data": {
                        "size": input_data.get("team_size", 5),
                        "roles": input_data.get("team_roles", ["manager", "contributor"]),
                        "stress_factors": stress_data.get("triggers", [])
                    },
                    "individual_stress": stress_data.get("stress_level", 5.0),
                    "analysis_depth": "detailed"
                }
                
                return crewai_format
            else:  # direction == "from_framework"
                # Process CrewAI output
                team_stress = state.get("team_stress_factors", []) or state.get("stress_factors", [])
                dynamics = state.get("interpersonal_dynamics", []) or state.get("team_interactions", [])
                interventions = state.get("recommended_team_interventions", []) or state.get("team_recommendations", [])
                
                return {
                    "team_stress_factors": team_stress,
                    "interpersonal_dynamics": dynamics,
                    "recommended_team_interventions": interventions
                }
        
        def intervention_converter(state: Dict[str, Any], direction: str) -> Dict[str, Any]:
            """Convert between composable state and LangGraph framework format"""
            if direction == "to_framework":
                # Collect all results from previous agents
                results = state.get("results", {})
                
                # Format for LangGraph
                langgraph_format = {
                    "graph_type": "intervention_planning",
                    "node_data": {
                        "stress": results.get(AgentType.STRESS_MODELING.value, {}).get("data", {}),
                        "neurochemical": results.get(AgentType.NEUROCHEMICAL_INTERACTION.value, {}).get("data", {}),
                        "cognitive": results.get(AgentType.COGNITIVE_ASSESSMENT.value, {}).get("data", {}),
                        "patterns": results.get(AgentType.PATTERN_RECOGNITION.value, {}).get("data", {}),
                        "team": results.get(AgentType.TEAM_DYNAMICS.value, {}).get("data", {})
                    },
                    "execution_parameters": {
                        "max_iterations": 3,
                        "prioritize_by": ["urgency", "effectiveness"]
                    }
                }
                
                return langgraph_format
            else:  # direction == "from_framework"
                # Process LangGraph output
                priorities = state.get("priority_interventions", []) or state.get("intervention_priorities", [])
                detailed = state.get("detailed_recommendations", []) or state.get("intervention_details", [])
                outcomes = state.get("expected_outcomes", []) or state.get("projected_benefits", [])
                
                return {
                    "priority_interventions": priorities,
                    "detailed_recommendations": detailed,
                    "expected_outcomes": outcomes
                }
        
        # Return the appropriate converter based on agent type
        converters = {
            AgentType.STRESS_MODELING.value: stress_modeling_converter,
            AgentType.NEUROCHEMICAL_INTERACTION.value: neurochemical_converter,
            AgentType.COGNITIVE_ASSESSMENT.value: cognitive_assessment_converter,
            AgentType.PATTERN_RECOGNITION.value: pattern_recognition_converter,
            AgentType.TEAM_DYNAMICS.value: team_dynamics_converter,
            AgentType.INTERVENTION_RECOMMENDATION.value: intervention_converter
        }
        
        return converters.get(agent_type, lambda state, direction: state)
    
    def _simulate_agent_communication(self, from_agent: str, to_agent: str, message: Any) -> None:
        """
        Simulate communication between agents in the hybrid architecture
        """
        # Serialize the message to measure payload size
        serialized_message = json.dumps(message)
        payload_size = len(serialized_message.encode('utf-8'))
        
        # In the hybrid architecture, communication overhead is between
        # multi-framework and pure composable approaches
        latency = 20 + (payload_size / 1024) * 5  # Moderate latency
        
        # Log the communication event
        self.log_communication(
            from_agent=from_agent,
            to_agent=to_agent,
            message=message,
            latency_ms=latency,
            payload_size_bytes=payload_size
        )


class HybridAgent:
    """
    Agent that wraps a framework-specific adapter with the composable interface
    """
    
    def __init__(self, agent_type: str, adapter, state_converter: Callable):
        self.agent_type = agent_type
        self.adapter = adapter
        self.state_converter = state_converter
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the global state through the framework-specific adapter
        
        Args:
            state: The global state object
            
        Returns:
            Dict with status, data, and any errors
        """
        try:
            # Start timing for conversion
            conversion_start = time.time() * 1000
            
            # Convert state to framework-specific format
            framework_input = self.state_converter(state, "to_framework")
            
            # End timing for conversion
            conversion_end = time.time() * 1000
            conversion_time = conversion_end - conversion_start
            
            # Start timing for framework execution
            execution_start = time.time() * 1000
            
            # Run the framework-specific adapter
            framework_output = self.adapter.run(framework_input)
            
            # End timing for framework execution
            execution_end = time.time() * 1000
            execution_time = execution_end - execution_start
            
            # Start timing for conversion back
            back_conversion_start = time.time() * 1000
            
            # Convert framework output to composable format
            composable_output = self.state_converter(framework_output, "from_framework")
            
            # End timing for conversion back
            back_conversion_end = time.time() * 1000
            back_conversion_time = back_conversion_end - back_conversion_start
            
            # Calculate total adapter overhead
            total_conversion_time = conversion_time + back_conversion_time
            
            # Create an adapter event for metrics tracking
            if hasattr(self.adapter, "framework_name"):
                framework_name = self.adapter.framework_name
            else:
                framework_name = type(self.adapter).__name__
                
            # Notify about the adapter usage (this would normally be tracked by a monitoring system)
            if hasattr(state, "log_adapter_event"):
                state.log_adapter_event(
                    framework=framework_name,
                    operation=f"process_{self.agent_type}",
                    conversion_time_ms=total_conversion_time,
                    execution_time_ms=execution_time,
                    framework_switching_latency_ms=5.0  # Simulated switching latency
                )
            
            return {
                "status": "success",
                "data": composable_output,
                "errors": None,
                "metadata": {
                    "framework": framework_name,
                    "conversion_time_ms": total_conversion_time,
                    "execution_time_ms": execution_time
                }
            }
        except Exception as e:
            # Handle any errors during processing
            return {
                "status": "error",
                "data": None,
                "errors": str(e)
            }


class MockFrameworkAdapter:
    """Mock adapter for demonstration purposes"""
    
    def __init__(self, framework_name: str, agent_type: str):
        self.framework_name = framework_name
        self.agent_type = agent_type
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate running this framework with the given input"""
        # Simulate processing delay based on framework complexity
        if self.framework_name in ["autogen", "langgraph"]:
            time.sleep(0.15)  # More complex frameworks
        else:
            time.sleep(0.08)  # Simpler frameworks
        
        # Return simulated output based on agent type
        if self.agent_type == AgentType.STRESS_MODELING.value:
            return {
                "stress_level": 7.5,
                "identified_factors": ["deadline", "workload", "interpersonal_conflict"],
                "suggested_interventions": ["breathing_exercise", "time_management", "communication_training"]
            }
        elif self.agent_type == AgentType.NEUROCHEMICAL_INTERACTION.value:
            return {
                "chemical_levels": {
                    "cortisol": 18.5,
                    "adrenaline": 12.3,
                    "dopamine": 6.7,
                    "serotonin": 4.2
                },
                "identified_patterns": ["acute_stress_response", "reward_seeking_behavior"],
                "suggested_actions": ["physical_activity", "meditation", "sleep_improvement"]
            }
        elif self.agent_type == AgentType.COGNITIVE_ASSESSMENT.value:
            return {
                "cognitive_load": 8.2,
                "attention_score": 4.5,
                "working_memory_score": 6.8,
                "decision_quality": 5.5,
                "improvement_strategies": ["task_simplification", "memory_aids", "decision_frameworks"]
            }
        elif self.agent_type == AgentType.INTERVENTION_RECOMMENDATION.value:
            return {
                "intervention_priorities": ["stress_reduction", "cognitive_support", "environmental_modification"],
                "intervention_details": [
                    {"name": "breathing_exercise", "frequency": "3x daily", "duration": "5 minutes"},
                    {"name": "task_prioritization", "frequency": "daily", "duration": "15 minutes"},
                    {"name": "workspace_optimization", "frequency": "once", "duration": "30 minutes"}
                ],
                "projected_benefits": ["reduced_stress", "improved_focus", "better_decision_making"]
            }
        elif self.agent_type == AgentType.TEAM_DYNAMICS.value:
            return {
                "stress_factors": ["communication_gaps", "unclear_roles", "deadline_pressure"],
                "team_interactions": ["support_network_present", "conflict_avoidance", "information_siloing"],
                "team_recommendations": ["role_clarification", "communication_workshop", "shared_planning_session"]
            }
        elif self.agent_type == AgentType.PATTERN_RECOGNITION.value:
            return {
                "detected_patterns": ["cyclical_stress_peaks", "intervention_resistance", "environmental_triggers"],
                "pattern_analytics": {
                    "cyclical_stress_peaks": {"frequency": "weekly", "triggers": ["monday_meetings", "friday_deadlines"]},
                    "intervention_resistance": {"areas": ["meditation", "delegation"], "reasons": ["time_pressure", "trust_issues"]},
                    "environmental_triggers": {"locations": ["open_office", "meeting_room_3"], "factors": ["noise", "interruptions"]}
                },
                "suggested_actions": ["schedule_restructuring", "environmental_modifications", "intervention_timing_optimization"]
            }
        else:
            return {"result": "No specific output for this agent type"}
