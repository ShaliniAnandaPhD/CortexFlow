import time
import json
from typing import Dict, Any, List, Tuple, Optional, Callable
import importlib

from architectures.base_architecture import BaseArchitecture
from agent_evaluator.metrics import AgentType

class ComposableArchitecture(BaseArchitecture):
    """
    A composable architecture with standardized interfaces for all agent components.
    Each agent follows a unified pattern:
    1. Take a global state as input
    2. Process it through domain-specific logic
    3. Return results in a standardized format
    4. Contribute to a centralized state update
    """
    
    def __init__(self):
        super().__init__()
        self.agents = {}
        self.agent_order = []
        self.global_state = {}
    
    def initialize(self) -> None:
        """Initialize the architecture with all agent components"""
        self.initialized = True
        
        # Initialize agents for each type
        for agent_type in AgentType:
            try:
                # Try to load a specialized agent implementation if available
                module_name = f"agents.{agent_type.value}_agent"
                spec = importlib.util.find_spec(module_name)
                
                if spec:
                    module = importlib.import_module(module_name)
                    agent_class = getattr(module, f"{agent_type.value.capitalize()}Agent")
                    
                    # Initialize the agent
                    self.agents[agent_type.value] = agent_class()
                    print(f"Loaded specialized agent for {agent_type.value}")
                else:
                    # Use generic agent with specialized processing for demo
                    self.agents[agent_type.value] = Agent(agent_type.value, self._get_agent_processor(agent_type.value))
                    print(f"Created generic agent for {agent_type.value}")
            except Exception as e:
                print(f"Error loading agent for {agent_type.value}: {e}")
                # Use generic agent with specialized processing for demo
                self.agents[agent_type.value] = Agent(agent_type.value, self._get_agent_processor(agent_type.value))
        
        # Define the default processing order
        self.agent_order = [
            AgentType.STRESS_MODELING.value,
            AgentType.NEUROCHEMICAL_INTERACTION.value,
            AgentType.COGNITIVE_ASSESSMENT.value,
            AgentType.PATTERN_RECOGNITION.value,
            AgentType.TEAM_DYNAMICS.value,
            AgentType.INTERVENTION_RECOMMENDATION.value
        ]
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the composable architecture on the input data"""
        if not self.initialized:
            self.initialize()
        
        # Initialize global state
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
                    event_type="processing",
                    start_time=start_time,
                    duration=duration,
                    metadata={"status": agent_result.get("status", "unknown")}
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
    
    def _get_agent_processor(self, agent_type: str) -> Callable:
        """Get the appropriate processing function for an agent type"""
        # This would normally be implemented with actual agent-specific logic
        # Here we simulate different processing for each agent type
        
        def stress_modeling_processor(state: Dict[str, Any]) -> Dict[str, Any]:
            # Simulate stress modeling processing
            input_data = state.get("input", {})
            
            # Very simplified stress modeling logic
            stress_level = 5.0  # Default moderate stress
            
            if "stress_factors" in input_data:
                # Calculate stress based on factors
                stress_level = min(10.0, 2.0 + len(input_data["stress_factors"]) * 1.5)
            
            triggers = input_data.get("stress_factors", ["unknown"])
            
            # Determine interventions based on stress level
            interventions = []
            if stress_level > 8.0:
                interventions = ["immediate_break", "deep_breathing", "stress_counseling"]
            elif stress_level > 6.0:
                interventions = ["deep_breathing", "prioritization", "time_management"]
            elif stress_level > 4.0:
                interventions = ["mindfulness", "task_prioritization"]
            else:
                interventions = ["preventive_measures", "regular_breaks"]
            
            return {
                "stress_level": stress_level,
                "triggers": triggers,
                "recommended_interventions": interventions
            }
        
        def neurochemical_processor(state: Dict[str, Any]) -> Dict[str, Any]:
            # Simplified neurochemical interaction processing
            input_data = state.get("input", {})
            stress_results = state.get("results", {}).get(AgentType.STRESS_MODELING.value, {}).get("data", {})
            
            # Basic model - stress level affects chemical concentrations
            stress_level = stress_results.get("stress_level", 5.0)
            
            # Model simplified neurochemical concentrations based on stress
            cortisol = 10.0 + stress_level * 1.5
            adrenaline = 5.0 + stress_level * 1.2
            dopamine = 15.0 - stress_level * 0.8
            serotonin = 20.0 - stress_level * 0.6
            
            # Identify patterns
            patterns = []
            if stress_level > 7.0:
                patterns.append("acute_stress_response")
            if stress_level > 5.0:
                patterns.append("elevated_cortisol")
            if dopamine < 10.0:
                patterns.append("reduced_reward_response")
                
            # Recommendations
            recommendations = []
            if cortisol > 20.0:
                recommendations.append("stress_reduction_techniques")
            if dopamine < 10.0:
                recommendations.append("reward_system_activities")
            if serotonin < 15.0:
                recommendations.append("mood_enhancing_activities")
                
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
        
        def cognitive_assessment_processor(state: Dict[str, Any]) -> Dict[str, Any]:
            # Simplified cognitive assessment processing
            stress_results = state.get("results", {}).get(AgentType.STRESS_MODELING.value, {}).get("data", {})
            neuro_results = state.get("results", {}).get(AgentType.NEUROCHEMICAL_INTERACTION.value, {}).get("data", {})
            
            # Extract relevant factors
            stress_level = stress_results.get("stress_level", 5.0)
            cortisol = neuro_results.get("concentrations", {}).get("cortisol", 15.0)
            dopamine = neuro_results.get("concentrations", {}).get("dopamine", 10.0)
            
            # Model cognitive impacts
            cognitive_load = min(10.0, stress_level + (cortisol - 15) / 5.0)
            attention_span = max(1.0, 10.0 - stress_level * 0.8)
            working_memory = max(1.0, 10.0 - cognitive_load * 0.5)
            decision_making = max(1.0, 10.0 - cognitive_load * 0.6 + dopamine * 0.2)
            
            # Recommendations
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
        
        def pattern_recognition_processor(state: Dict[str, Any]) -> Dict[str, Any]:
            # Simplified pattern recognition
            results = state.get("results", {})
            
            # Extract data from previous agents
            stress_data = results.get(AgentType.STRESS_MODELING.value, {}).get("data", {})
            neuro_data = results.get(AgentType.NEUROCHEMICAL_INTERACTION.value, {}).get("data", {})
            cognitive_data = results.get(AgentType.COGNITIVE_ASSESSMENT.value, {}).get("data", {})
            
            # Identify patterns across domains
            patterns = []
            
            # Stress-cognitive patterns
            if stress_data.get("stress_level", 0) > 7.0 and cognitive_data.get("cognitive_load", 0) > 7.0:
                patterns.append("high_stress_cognitive_impact")
                
            # Neurochemical patterns
            if neuro_data.get("concentrations", {}).get("cortisol", 0) > 20.0:
                patterns.append("sustained_cortisol_elevation")
                
            # Environmental patterns from stress triggers
            triggers = stress_data.get("triggers", [])
            if "deadline" in triggers and "workload" in triggers:
                patterns.append("work_pressure_pattern")
                
            if len(patterns) == 0:
                patterns.append("no_significant_patterns")
                
            # Generate detailed data
            pattern_details = {}
            for pattern in patterns:
                if pattern == "high_stress_cognitive_impact":
                    pattern_details[pattern] = {
                        "stress_level": stress_data.get("stress_level", 0),
                        "cognitive_load": cognitive_data.get("cognitive_load", 0),
                        "correlation": "direct_impact"
                    }
                elif pattern == "sustained_cortisol_elevation":
                    pattern_details[pattern] = {
                        "cortisol_level": neuro_data.get("concentrations", {}).get("cortisol", 0),
                        "duration": "unknown",  # Would be based on historical data
                        "impact": "negative_cognitive_function"
                    }
                elif pattern == "work_pressure_pattern":
                    pattern_details[pattern] = {
                        "triggers": [t for t in triggers if t in ["deadline", "workload"]],
                        "frequency": "regular",  # Would be based on historical data
                        "impact": "elevated_stress_and_cortisol"
                    }
                elif pattern == "no_significant_patterns":
                    pattern_details[pattern] = {
                        "note": "Insufficient data or normal functioning"
                    }
            
            # Pattern-based recommendations
            recommendations = []
            if "high_stress_cognitive_impact" in patterns:
                recommendations.append("cognitive_load_management")
            if "sustained_cortisol_elevation" in patterns:
                recommendations.append("stress_hormone_regulation")
            if "work_pressure_pattern" in patterns:
                recommendations.append("workload_balancing")
                
            return {
                "identified_patterns": patterns,
                "pattern_details": pattern_details,
                "pattern_based_recommendations": recommendations
            }
        
        def team_dynamics_processor(state: Dict[str, Any]) -> Dict[str, Any]:
            # Simplified team dynamics processing
            input_data = state.get("input", {})
            stress_data = state.get("results", {}).get(AgentType.STRESS_MODELING.value, {}).get("data", {})
            
            # Extract team information
            team_size = input_data.get("team_size", 5)
            team_roles = input_data.get("team_roles", ["manager", "contributor"])
            
            # Identify team stress factors
            team_stress_factors = []
            individual_stress_level = stress_data.get("stress_level", 5.0)
            
            if individual_stress_level > 7.0:
                team_stress_factors.append("high_individual_stress")
            
            if "deadline" in stress_data.get("triggers", []):
                team_stress_factors.append("deadline_pressure")
                
            if team_size > 8:
                team_stress_factors.append("communication_complexity")
                
            if len(team_roles) < team_size / 2:
                team_stress_factors.append("unclear_roles")
                
            # Interpersonal dynamics
            interpersonal_dynamics = []
            
            if "conflict" in stress_data.get("triggers", []):
                interpersonal_dynamics.append("conflict_present")
            else:
                interpersonal_dynamics.append("conflict_avoidance")
                
            if "communication" in stress_data.get("triggers", []):
                interpersonal_dynamics.append("communication_gaps")
            else:
                interpersonal_dynamics.append("open_communication")
                
            # Team interventions
            team_interventions = []
            
            if "high_individual_stress" in team_stress_factors:
                team_interventions.append("team_wellness_program")
                
            if "deadline_pressure" in team_stress_factors:
                team_interventions.append("shared_planning_session")
                
            if "unclear_roles" in team_stress_factors:
                team_interventions.append("role_clarification")
                
            if "conflict_present" in interpersonal_dynamics:
                team_interventions.append("conflict_resolution_workshop")
                
            if "communication_gaps" in interpersonal_dynamics:
                team_interventions.append("communication_workshop")
                
            return {
                "team_stress_factors": team_stress_factors,
                "interpersonal_dynamics": interpersonal_dynamics,
                "recommended_team_interventions": team_interventions
            }
        
        def intervention_recommendation_processor(state: Dict[str, Any]) -> Dict[str, Any]:
            # Comprehensive intervention recommendation
            results = state.get("results", {})
            
            # Collect all recommendations from previous agents
            all_recommendations = []
            
            if AgentType.STRESS_MODELING.value in results:
                stress_interventions = results[AgentType.STRESS_MODELING.value].get("data", {}).get("recommended_interventions", [])
                all_recommendations.extend(stress_interventions)
                
            if AgentType.NEUROCHEMICAL_INTERACTION.value in results:
                neuro_recommendations = results[AgentType.NEUROCHEMICAL_INTERACTION.value].get("data", {}).get("recommendations", [])
                all_recommendations.extend(neuro_recommendations)
                
            if AgentType.COGNITIVE_ASSESSMENT.value in results:
                cognitive_recommendations = results[AgentType.COGNITIVE_ASSESSMENT.value].get("data", {}).get("recommendations", [])
                all_recommendations.extend(cognitive_recommendations)
                
            if AgentType.PATTERN_RECOGNITION.value in results:
                pattern_recommendations = results[AgentType.PATTERN_RECOGNITION.value].get("data", {}).get("pattern_based_recommendations", [])
                all_recommendations.extend(pattern_recommendations)
                
            if AgentType.TEAM_DYNAMICS.value in results:
                team_recommendations = results[AgentType.TEAM_DYNAMICS.value].get("data", {}).get("recommended_team_interventions", [])
                all_recommendations.extend(team_recommendations)
            
            # Remove duplicates
            unique_recommendations = list(set(all_recommendations))
            
            # Categorize recommendations
            stress_reduction = []
            cognitive_support = []
            team_support = []
            environmental = []
            
            # Simple categorization
            for rec in unique_recommendations:
                if any(term in rec.lower() for term in ["stress", "breathing", "meditation", "break", "wellness"]):
                    stress_reduction.append(rec)
                elif any(term in rec.lower() for term in ["cognitive", "memory", "decision", "attention", "load"]):
                    cognitive_support.append(rec)
                elif any(term in rec.lower() for term in ["team", "communication", "role", "conflict", "planning"]):
                    team_support.append(rec)
                elif any(term in rec.lower() for term in ["environment", "workspace", "optimization"]):
                    environmental.append(rec)
                else:
                    # Assign to most appropriate category
                    cognitive_support.append(rec)
            
            # Prioritize categories based on need
            priority_categories = []
            
            # Check stress level to determine priorities
            stress_level = 0
            if AgentType.STRESS_MODELING.value in results:
                stress_level = results[AgentType.STRESS_MODELING.value].get("data", {}).get("stress_level", 0)
            
            if stress_level > 7.0:
                priority_categories.append("stress_reduction")
            if stress_level > 5.0:
                priority_categories.append("cognitive_support")
            
            # Add team support if team dynamics were analyzed
            if AgentType.TEAM_DYNAMICS.value in results:
                priority_categories.append("team_support")
                
            # Always consider environmental factors
            priority_categories.append("environmental_modification")
            
            # Create detailed recommendations
            detailed_recommendations = []
            
            # Add stress reduction interventions
            for rec in stress_reduction:
                detailed_recommendations.append({
                    "name": rec,
                    "frequency": "daily" if stress_level > 7 else "3x weekly",
                    "duration": "15 minutes"
                })
                
            # Add cognitive support interventions
            for rec in cognitive_support:
                detailed_recommendations.append({
                    "name": rec,
                    "frequency": "as needed",
                    "duration": "30 minutes"
                })
                
            # Add team interventions
            for rec in team_support:
                detailed_recommendations.append({
                    "name": rec,
                    "frequency": "weekly",
                    "duration": "60 minutes"
                })
                
            # Add environmental interventions
            for rec in environmental:
                detailed_recommendations.append({
                    "name": rec,
                    "frequency": "once",
                    "duration": "varies"
                })
            
            # Expected outcomes
            expected_outcomes = []
            
            if stress_level > 5.0:
                expected_outcomes.append("reduced_stress")
            if AgentType.COGNITIVE_ASSESSMENT.value in results:
                expected_outcomes.append("improved_focus")
                expected_outcomes.append("better_decision_making")
            if AgentType.TEAM_DYNAMICS.value in results:
                expected_outcomes.append("improved_team_dynamics")
            
            return {
                "priority_interventions": priority_categories,
                "detailed_recommendations": detailed_recommendations,
                "expected_outcomes": expected_outcomes
            }
        
        # Return the appropriate processor based on agent type
        processors = {
            AgentType.STRESS_MODELING.value: stress_modeling_processor,
            AgentType.NEUROCHEMICAL_INTERACTION.value: neurochemical_processor,
            AgentType.COGNITIVE_ASSESSMENT.value: cognitive_assessment_processor,
            AgentType.PATTERN_RECOGNITION.value: pattern_recognition_processor,
            AgentType.TEAM_DYNAMICS.value: team_dynamics_processor,
            AgentType.INTERVENTION_RECOMMENDATION.value: intervention_recommendation_processor
        }
        
        return processors.get(agent_type, lambda state: {"error": "Unsupported agent type"})
    
    def _simulate_agent_communication(self, from_agent: str, to_agent: str, message: Any) -> None:
        """
        Simulate communication between agents, with minimal overhead in this architecture
        """
        # Serialize the message to measure payload size
        serialized_message = json.dumps(message)
        payload_size = len(serialized_message.encode('utf-8'))
        
        # In the composable architecture, communication overhead is minimal
        # We use a smaller base latency compared to the multi-framework approach
        latency = 5 + (payload_size / 1024) * 2  # Significantly lower latency
        
        # Log the communication event
        self.log_communication(
            from_agent=from_agent,
            to_agent=to_agent,
            message=message,
            latency_ms=latency,
            payload_size_bytes=payload_size
        )


class Agent:
    """
    Generic agent implementation for the composable architecture
    """
    
    def __init__(self, agent_type: str, processor: Callable):
        self.agent_type = agent_type
        self.processor = processor
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the global state and return results
        
        Args:
            state: The global state object
            
        Returns:
            Dict with status, data, and any errors
        """
        try:
            # Execute the processor function
            result_data = self.processor(state)
            
            return {
                "status": "success",
                "data": result_data,
                "errors": None
            }
        except Exception as e:
            # Handle any errors during processing
            return {
                "status": "error",
                "data": None,
                "errors": str(e)
            }
