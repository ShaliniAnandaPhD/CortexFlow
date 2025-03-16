"""
Haystack agent implementation for the CortexFlow system.

This module implements the Haystack agent, which is responsible for
intervention recommendation based on current stress levels and cognitive
metrics. It uses a retrieval-augmented system to suggest personalized
stress management techniques.
"""

import random
from typing import Dict, Any, Set, Optional, List, Tuple

from cortexflow.agents.base import AgentBase
from cortexflow.core.state import SimulationState
from cortexflow.core.types import (
    AgentType, 
    AgentCapability, 
    StressLevel, 
    TaskType,
    InterventionType
)


class HaystackAgent(AgentBase):
    """
    Haystack agent for intervention recommendation.
    
    This agent uses a retrieval-augmented approach to recommend appropriate
    stress management interventions based on the current simulation state
    and evidence-based techniques from the stress research literature.
    
    Attributes:
        intervention_database: Database of intervention techniques
        task_suitability: Mapping of interventions to task types
        stress_suitability: Mapping of interventions to stress levels
    """
    
    def __init__(
        self,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new Haystack agent.
        
        Args:
            enabled: Whether the agent is currently enabled
            config: Agent-specific configuration parameters
        """
        super().__init__("HaystackAgent", AgentType.KNOWLEDGE, enabled, config or {})
        
        # Initialize the intervention database
        self.intervention_database = self._init_intervention_database()
        
        # Initialize task suitability mapping
        self.task_suitability = self._init_task_suitability()
        
        # Initialize stress suitability mapping
        self.stress_suitability = self._init_stress_suitability()
    
    def _init_intervention_database(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize the database of intervention techniques.
        
        This database contains information about various stress management
        interventions, including their descriptions, efficacy, and sources.
        
        Returns:
            A dictionary of intervention techniques
        """
        return {
            InterventionType.MEDITATION: {
                "name": "Mindfulness Meditation",
                "description": "A practice of focused attention on breath, thoughts, or sensations without judgment.",
                "techniques": [
                    "Focused Breathing: Pay attention to the sensation of breathing for 2-5 minutes.",
                    "Body Scan: Systematically focus attention on different parts of the body.",
                    "Thought Observation: Observe thoughts without engaging with them.",
                    "Guided Meditation: Follow audio instructions for a meditation session."
                ],
                "efficacy": 0.75,
                "duration": "5-15 minutes",
                "sources": [
                    "Kabat-Zinn, J. (2003). Mindfulness-based interventions in context: Past, present, and future.",
                    "Creswell, J. D. (2017). Mindfulness interventions."
                ]
            },
            InterventionType.MICRO_BREAKS: {
                "name": "Micro-Breaks",
                "description": "Short breaks during work periods to prevent mental fatigue and stress buildup.",
                "techniques": [
                    "20-20-20 Rule: Every 20 minutes, look at something 20 feet away for 20 seconds.",
                    "Stretch Break: Stand up and stretch for 30-60 seconds.",
                    "Hydration Break: Take a moment to drink water.",
                    "Deep Breathing: Take 5 deep breaths with eyes closed."
                ],
                "efficacy": 0.70,
                "duration": "30 seconds - 2 minutes",
                "sources": [
                    "Henning, R. A., et al. (1997). Frequent short rest breaks from computer work.",
                    "Kim, S., et al. (2017). Effects of micro-breaks on cognitive function."
                ]
            },
            InterventionType.BIOFEEDBACK: {
                "name": "Biofeedback Techniques",
                "description": "Using awareness of physiological functions to regulate stress response.",
                "techniques": [
                    "Heart Rate Monitoring: Observe heart rate and practice slowing it.",
                    "Breath Control: Regulate breathing patterns to induce calm.",
                    "Muscle Tension Awareness: Notice and release physical tension.",
                    "Temperature Feedback: Notice skin temperature as indicator of stress."
                ],
                "efficacy": 0.80,
                "duration": "3-10 minutes",
                "sources": [
                    "Lehrer, P. M., & Gevirtz, R. (2014). Heart rate variability biofeedback.",
                    "Yucha, C., & Montgomery, D. (2008). Evidence-based practice in biofeedback."
                ]
            }
        }
    
    def _init_task_suitability(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize task suitability mapping for interventions.
        
        This mapping indicates how suitable each intervention is for
        different types of tasks based on research evidence.
        
        Returns:
            A dictionary mapping interventions to task suitability scores
        """
        return {
            InterventionType.MEDITATION: {
                TaskType.CREATIVE: 0.85,
                TaskType.ANALYTICAL: 0.70,
                TaskType.PHYSICAL: 0.60,
                TaskType.REPETITIVE: 0.75
            },
            InterventionType.MICRO_BREAKS: {
                TaskType.CREATIVE: 0.70,
                TaskType.ANALYTICAL: 0.75,
                TaskType.PHYSICAL: 0.85,
                TaskType.REPETITIVE: 0.90
            },
            InterventionType.BIOFEEDBACK: {
                TaskType.CREATIVE: 0.65,
                TaskType.ANALYTICAL: 0.80,
                TaskType.PHYSICAL: 0.75,
                TaskType.REPETITIVE: 0.65
            }
        }
    
    def _init_stress_suitability(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize stress suitability mapping for interventions.
        
        This mapping indicates how suitable each intervention is for
        different levels of stress based on research evidence.
        
        Returns:
            A dictionary mapping interventions to stress suitability scores
        """
        return {
            InterventionType.MEDITATION: {
                StressLevel.MILD: 0.70,
                StressLevel.MODERATE: 0.80,
                StressLevel.SEVERE: 0.60
            },
            InterventionType.MICRO_BREAKS: {
                StressLevel.MILD: 0.85,
                StressLevel.MODERATE: 0.75,
                StressLevel.SEVERE: 0.55
            },
            InterventionType.BIOFEEDBACK: {
                StressLevel.MILD: 0.65,
                StressLevel.MODERATE: 0.75,
                StressLevel.SEVERE: 0.85
            }
        }
    
    async def process(self, state: SimulationState) -> SimulationState:
        """
        Process the current simulation state using the Haystack agent.
        
        This method analyzes the current state and recommends appropriate
        interventions based on the current stress level, task type, and
        cognitive metrics.
        
        Args:
            state: The current simulation state
            
        Returns:
            An updated simulation state with recommended interventions
        """
        # Clone the state to avoid modifying the original
        new_state = state.clone()
        
        try:
            # Only recommend an intervention if none is currently applied
            if new_state.intervention == InterventionType.NONE:
                # Check if stress level requires intervention
                if self._should_intervene(new_state):
                    # Recommend the best intervention
                    intervention, score = self._recommend_intervention(new_state)
                    
                    # Apply the recommended intervention
                    new_state.intervention = intervention
                    
                    # Store the recommendation score in metadata
                    if "recommendations" not in new_state.metadata:
                        new_state.metadata["recommendations"] = []
                    
                    new_state.metadata["recommendations"].append({
                        "step": new_state.simulation_step,
                        "intervention": intervention,
                        "score": score,
                        "metrics": {
                            "stress_level": new_state.stress_level,
                            "cortisol_level": new_state.cortisol_level,
                            "dopamine_level": new_state.dopamine_level,
                            "productivity_score": new_state.productivity_score,
                            "memory_efficiency": new_state.memory_efficiency,
                            "decision_quality": new_state.decision_quality
                        }
                    })
                    
                    # Log the recommendation (in a real system, this would use a proper logger)
                    print(f"[HaystackAgent] Recommended intervention: {intervention} (score: {score:.2f})")
                    
                    # Get technique details
                    technique = self._select_specific_technique(intervention, new_state)
                    print(f"[HaystackAgent] Recommended technique: {technique}")
            else:
                # Evaluate the current intervention
                effectiveness = self._evaluate_intervention_effectiveness(new_state)
                
                # Store the evaluation in metadata
                if "evaluations" not in new_state.metadata:
                    new_state.metadata["evaluations"] = []
                
                new_state.metadata["evaluations"].append({
                    "step": new_state.simulation_step,
                    "intervention": new_state.intervention,
                    "effectiveness": effectiveness,
                    "metrics": {
                        "productivity_score": new_state.productivity_score,
                        "memory_efficiency": new_state.memory_efficiency,
                        "decision_quality": new_state.decision_quality
                    }
                })
                
                # Log the evaluation (in a real system, this would use a proper logger)
                print(f"[HaystackAgent] Intervention effectiveness: {effectiveness:.2f}")
            
            return new_state
            
        except Exception as e:
            # In a real implementation, this would use a proper error handling mechanism
            print(f"Haystack recommendation failed: {e}")
            # Return the original state if the recommendation fails
            return state
    
    def _should_intervene(self, state: SimulationState) -> bool:
        """
        Determine if an intervention is needed based on current state.
        
        This method analyzes the current stress level, cortisol level,
        and cognitive metrics to decide if an intervention is warranted.
        
        Args:
            state: The current simulation state
            
        Returns:
            True if an intervention is recommended, False otherwise
        """
        # Define thresholds for intervention
        thresholds = {
            StressLevel.MILD: {
                "cortisol": 60.0,
                "productivity": 80.0,
                "memory": 80.0,
                "decision": 80.0
            },
            StressLevel.MODERATE: {
                "cortisol": 70.0,
                "productivity": 75.0,
                "memory": 75.0,
                "decision": 75.0
            },
            StressLevel.SEVERE: {
                "cortisol": 80.0,
                "productivity": 70.0,
                "memory": 70.0,
                "decision": 70.0
            }
        }
        
        # Get thresholds for current stress level
        current_thresholds = thresholds.get(
            state.stress_level, thresholds[StressLevel.MODERATE]
        )
        
        # Check if any threshold is exceeded
        if (state.cortisol_level >= current_thresholds["cortisol"] or
            state.productivity_score <= current_thresholds["productivity"] or
            state.memory_efficiency <= current_thresholds["memory"] or
            state.decision_quality <= current_thresholds["decision"]):
            return True
        
        return False
    
    def _recommend_intervention(
        self,
        state: SimulationState
    ) -> Tuple[str, float]:
        """
        Recommend the most appropriate intervention for the current state.
        
        This method scores each available intervention based on task
        suitability, stress suitability, and current cognitive metrics
        to identify the most effective option.
        
        Args:
            state: The current simulation state
            
        Returns:
            A tuple of (recommended intervention, recommendation score)
        """
        # Calculate scores for each intervention
        intervention_scores = {}
        
        for intervention in [
            InterventionType.MEDITATION,
            InterventionType.MICRO_BREAKS,
            InterventionType.BIOFEEDBACK
        ]:
            # Calculate task suitability
            task_score = self.task_suitability.get(intervention, {}).get(
                state.task_type, 0.5
            )
            
            # Calculate stress suitability
            stress_score = self.stress_suitability.get(intervention, {}).get(
                state.stress_level, 0.5
            )
            
            # Calculate metric-based suitability
            metric_score = self._calculate_metric_suitability(intervention, state)
            
            # Calculate overall score (weighted average)
            overall_score = (
                task_score * 0.3 +
                stress_score * 0.4 +
                metric_score * 0.3
            )
            
            intervention_scores[intervention] = overall_score
        
        # Find the intervention with the highest score
        best_intervention = max(
            intervention_scores.items(),
            key=lambda x: x[1]
        )
        
        return best_intervention
    
    def _calculate_metric_suitability(
        self,
        intervention: str,
        state: SimulationState
    ) -> float:
        """
        Calculate metric-based suitability for an intervention.
        
        This method evaluates how well an intervention addresses the
        current cognitive deficits in the simulation state.
        
        Args:
            intervention: The intervention to evaluate
            state: The current simulation state
            
        Returns:
            A suitability score between 0 and 1
        """
        # Define intervention strengths for different metrics
        intervention_strengths = {
            InterventionType.MEDITATION: {
                "productivity": 0.6,
                "memory": 0.7,
                "decision": 0.8
            },
            InterventionType.MICRO_BREAKS: {
                "productivity": 0.8,
                "memory": 0.6,
                "decision": 0.5
            },
            InterventionType.BIOFEEDBACK: {
                "productivity": 0.7,
                "memory": 0.6,
                "decision": 0.8
            }
        }
        
        # Get strengths for the current intervention
        strengths = intervention_strengths.get(
            intervention, 
            {"productivity": 0.5, "memory": 0.5, "decision": 0.5}
        )
        
        # Calculate how much improvement is needed for each metric
        productivity_need = max(0, (100 - state.productivity_score) / 100)
        memory_need = max(0, (100 - state.memory_efficiency) / 100)
        decision_need = max(0, (100 - state.decision_quality) / 100)
        
        # Calculate the weighted match between needs and strengths
        suitability = (
            productivity_need * strengths["productivity"] +
            memory_need * strengths["memory"] +
            decision_need * strengths["decision"]
        ) / (productivity_need + memory_need + decision_need + 0.001)  # Avoid division by zero
        
        return suitability
    
    def _select_specific_technique(
        self,
        intervention: str,
        state: SimulationState
    ) -> str:
        """
        Select a specific technique within the recommended intervention.
        
        This method chooses a specific technique from the intervention
        database that is most appropriate for the current state.
        
        Args:
            intervention: The recommended intervention
            state: The current simulation state
            
        Returns:
            A specific technique recommendation
        """
        # Get the techniques for the intervention
        techniques = self.intervention_database.get(intervention, {}).get("techniques", [])
        
        if not techniques:
            return "No specific technique available"
        
        # For now, just randomly select a technique
        # In a real implementation, this would use more sophisticated matching
        return random.choice(techniques)
    
    def _evaluate_intervention_effectiveness(self, state: SimulationState) -> float:
        """
        Evaluate the effectiveness of the current intervention.
        
        This method analyzes the changes in cognitive metrics since the
        intervention was applied to estimate its effectiveness.
        
        Args:
            state: The current simulation state
            
        Returns:
            An effectiveness score between 0 and 1
        """
        # Find when the intervention was applied
        intervention_step = None
        if "recommendations" in state.metadata:
            for recommendation in reversed(state.metadata.get("recommendations", [])):
                if recommendation["intervention"] == state.intervention:
                    intervention_step = recommendation["step"]
                    break
        
        if intervention_step is None or len(state.history) <= intervention_step:
            # Can't evaluate if we don't know when it was applied or no history
            return 0.5
        
        # Get the metrics at the intervention point
        intervention_metrics = None
        for entry in state.history:
            if entry.get("step") == intervention_step:
                intervention_metrics = entry
                break
        
        if intervention_metrics is None:
            return 0.5
        
        # Calculate changes in metrics since intervention
        productivity_change = state.productivity_score - intervention_metrics.get("productivity_score", state.productivity_score)
        memory_change = state.memory_efficiency - intervention_metrics.get("memory_efficiency", state.memory_efficiency)
        decision_change = state.decision_quality - intervention_metrics.get("decision_quality", state.decision_quality)
        
        # Calculate overall effectiveness (positive changes indicate effectiveness)
        effectiveness = (
            max(0, productivity_change) +
            max(0, memory_change) +
            max(0, decision_change)
        ) / 30.0  # Normalize to 0-1 range
        
        return min(1.0, effectiveness)
    
    def get_capabilities(self) -> Set[AgentCapability]:
        """
        Return the set of capabilities this agent provides.
        
        Returns:
            A set containing the INTERVENTION_RECOMMENDATION capability
        """
        return {AgentCapability.INTERVENTION_RECOMMENDATION}


# TODO: Implement more sophisticated intervention matching
# TODO: Add support for personalized recommendations
# TODO: Implement intervention combination strategies
# TODO: Add long-term effectiveness tracking

# Summary:
# This module implements the Haystack agent, which is responsible for
# recommending stress management interventions based on the current
# simulation state. It uses a retrieval-augmented approach to match
# interventions to the current stress level, task type, and cognitive
# metrics, and provides specific techniques within each intervention
# category. The agent also evaluates the effectiveness of ongoing
# interventions.
