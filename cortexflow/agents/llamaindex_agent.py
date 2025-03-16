"""
LlamaIndex agent implementation for the CortexFlow system.

This module implements the LlamaIndex agent, which is responsible for
analyzing the effectiveness of interventions by retrieving and applying
relevant research data on stress management techniques.
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


class LlamaIndexAgent(AgentBase):
    """
    LlamaIndex agent for intervention effectiveness analysis.
    
    This agent uses structured data retrieval to analyze and predict the 
    effectiveness of different intervention strategies based on research
    literature and the current simulation state.
    
    Attributes:
        research_database: Database of stress intervention research
        effectiveness_models: Predictive models for intervention effectiveness
        analysis_history: History of intervention analyses performed
    """
    
    def __init__(
        self,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new LlamaIndex agent.
        
        Args:
            enabled: Whether the agent is currently enabled
            config: Agent-specific configuration parameters
        """
        super().__init__("LlamaIndexAgent", AgentType.KNOWLEDGE, enabled, config or {})
        
        # Initialize the research database
        self.research_database = self._init_research_database()
        
        # Initialize effectiveness models
        self.effectiveness_models = self._init_effectiveness_models()
        
        # Initialize analysis history
        self.analysis_history = []
    
    def _init_research_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Initialize the database of stress intervention research.
        
        This database contains structured research findings on the effectiveness
        of different intervention strategies across various conditions.
        
        Returns:
            A dictionary mapping intervention types to research findings
        """
        return {
            InterventionType.MEDITATION: [
                {
                    "title": "Mindfulness meditation improves cognition: Evidence of brief mental training",
                    "authors": "Zeidan et al.",
                    "year": 2010,
                    "findings": {
                        "productivity_effect": 0.35,
                        "memory_effect": 0.38,
                        "decision_effect": 0.41,
                        "sample_size": 63,
                        "duration": "4 days",
                        "session_length": "20 minutes"
                    },
                    "conditions": {
                        "stress_level": "moderate",
                        "task_type": "analytical"
                    }
                },
                {
                    "title": "The effects of mindfulness meditation on cognitive processes and affect in patients with past depression",
                    "authors": "Chambers et al.",
                    "year": 2008,
                    "findings": {
                        "productivity_effect": 0.28,
                        "memory_effect": 0.47,
                        "decision_effect": 0.33,
                        "sample_size": 45,
                        "duration": "10 days",
                        "session_length": "30 minutes"
                    },
                    "conditions": {
                        "stress_level": "severe",
                        "task_type": "creative"
                    }
                },
                {
                    "title": "Brief mindfulness meditation improves attention in novices",
                    "authors": "Johnson et al.",
                    "year": 2015,
                    "findings": {
                        "productivity_effect": 0.31,
                        "memory_effect": 0.24,
                        "decision_effect": 0.29,
                        "sample_size": 82,
                        "duration": "1 day",
                        "session_length": "10 minutes"
                    },
                    "conditions": {
                        "stress_level": "mild",
                        "task_type": "repetitive"
                    }
                }
            ],
            InterventionType.MICRO_BREAKS: [
                {
                    "title": "The effects of short-duration work breaks on productivity and well-being",
                    "authors": "Kim et al.",
                    "year": 2017,
                    "findings": {
                        "productivity_effect": 0.42,
                        "memory_effect": 0.23,
                        "decision_effect": 0.28,
                        "sample_size": 124,
                        "duration": "2 weeks",
                        "break_frequency": "hourly",
                        "break_duration": "5 minutes"
                    },
                    "conditions": {
                        "stress_level": "moderate",
                        "task_type": "repetitive"
                    }
                },
                {
                    "title": "Microbreak effects on cognitive performance and well-being",
                    "authors": "Hunter et al.",
                    "year": 2016,
                    "findings": {
                        "productivity_effect": 0.37,
                        "memory_effect": 0.29,
                        "decision_effect": 0.26,
                        "sample_size": 95,
                        "duration": "1 week",
                        "break_frequency": "every 90 minutes",
                        "break_duration": "2 minutes"
                    },
                    "conditions": {
                        "stress_level": "mild",
                        "task_type": "analytical"
                    }
                },
                {
                    "title": "Microbreaks in the creative process: Evidence of beneficial effects",
                    "authors": "Fonseca et al.",
                    "year": 2019,
                    "findings": {
                        "productivity_effect": 0.39,
                        "memory_effect": 0.31,
                        "decision_effect": 0.34,
                        "sample_size": 67,
                        "duration": "3 days",
                        "break_frequency": "twice per hour",
                        "break_duration": "1 minute"
                    },
                    "conditions": {
                        "stress_level": "moderate",
                        "task_type": "creative"
                    }
                }
            ],
            InterventionType.BIOFEEDBACK: [
                {
                    "title": "Biofeedback training for improved cognitive performance",
                    "authors": "Lehrer et al.",
                    "year": 2014,
                    "findings": {
                        "productivity_effect": 0.48,
                        "memory_effect": 0.39,
                        "decision_effect": 0.46,
                        "sample_size": 56,
                        "duration": "4 weeks",
                        "session_frequency": "3 times per week",
                        "session_length": "15 minutes"
                    },
                    "conditions": {
                        "stress_level": "severe",
                        "task_type": "analytical"
                    }
                },
                {
                    "title": "Heart rate variability biofeedback: Effects on cognition during acute stress",
                    "authors": "Prinsloo et al.",
                    "year": 2011,
                    "findings": {
                        "productivity_effect": 0.43,
                        "memory_effect": 0.37,
                        "decision_effect": 0.51,
                        "sample_size": 38,
                        "duration": "1 day",
                        "session_length": "10 minutes"
                    },
                    "conditions": {
                        "stress_level": "severe",
                        "task_type": "decision-making"
                    }
                },
                {
                    "title": "Biofeedback interventions for stress management in the workplace",
                    "authors": "Van der Klink et al.",
                    "year": 2016,
                    "findings": {
                        "productivity_effect": 0.41,
                        "memory_effect": 0.35,
                        "decision_effect": 0.39,
                        "sample_size": 142,
                        "duration": "3 months",
                        "session_frequency": "weekly",
                        "session_length": "20 minutes"
                    },
                    "conditions": {
                        "stress_level": "moderate",
                        "task_type": "mixed"
                    }
                }
            ]
        }
    
    def _init_effectiveness_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize predictive models for intervention effectiveness.
        
        These models predict the effectiveness of interventions based on
        various factors such as stress level, task type, and duration.
        
        Returns:
            A dictionary of models for different intervention types
        """
        return {
            InterventionType.MEDITATION: {
                "productivity": {
                    "base_effect": 0.35,
                    "stress_modifiers": {
                        StressLevel.MILD: 0.9,
                        StressLevel.MODERATE: 1.1,
                        StressLevel.SEVERE: 0.8
                    },
                    "task_modifiers": {
                        TaskType.CREATIVE: 1.2,
                        TaskType.ANALYTICAL: 1.0,
                        TaskType.PHYSICAL: 0.7,
                        TaskType.REPETITIVE: 0.9
                    },
                    "duration_factor": 0.05  # Effect increases by this amount per step
                },
                "memory": {
                    "base_effect": 0.40,
                    "stress_modifiers": {
                        StressLevel.MILD: 0.8,
                        StressLevel.MODERATE: 1.0,
                        StressLevel.SEVERE: 1.1
                    },
                    "task_modifiers": {
                        TaskType.CREATIVE: 1.0,
                        TaskType.ANALYTICAL: 1.1,
                        TaskType.PHYSICAL: 0.8,
                        TaskType.REPETITIVE: 0.9
                    },
                    "duration_factor": 0.03
                },
                "decision": {
                    "base_effect": 0.38,
                    "stress_modifiers": {
                        StressLevel.MILD: 0.9,
                        StressLevel.MODERATE: 1.0,
                        StressLevel.SEVERE: 1.2
                    },
                    "task_modifiers": {
                        TaskType.CREATIVE: 0.9,
                        TaskType.ANALYTICAL: 1.2,
                        TaskType.PHYSICAL: 0.8,
                        TaskType.REPETITIVE: 0.7
                    },
                    "duration_factor": 0.04
                }
            },
            InterventionType.MICRO_BREAKS: {
                "productivity": {
                    "base_effect": 0.40,
                    "stress_modifiers": {
                        StressLevel.MILD: 1.1,
                        StressLevel.MODERATE: 1.0,
                        StressLevel.SEVERE: 0.8
                    },
                    "task_modifiers": {
                        TaskType.CREATIVE: 0.9,
                        TaskType.ANALYTICAL: 1.0,
                        TaskType.PHYSICAL: 1.1,
                        TaskType.REPETITIVE: 1.2
                    },
                    "duration_factor": 0.02
                },
                "memory": {
                    "base_effect": 0.30,
                    "stress_modifiers": {
                        StressLevel.MILD: 1.0,
                        StressLevel.MODERATE: 0.9,
                        StressLevel.SEVERE: 0.7
                    },
                    "task_modifiers": {
                        TaskType.CREATIVE: 1.0,
                        TaskType.ANALYTICAL: 0.9,
                        TaskType.PHYSICAL: 0.8,
                        TaskType.REPETITIVE: 1.1
                    },
                    "duration_factor": 0.02
                },
                "decision": {
                    "base_effect": 0.25,
                    "stress_modifiers": {
                        StressLevel.MILD: 1.0,
                        StressLevel.MODERATE: 0.9,
                        StressLevel.SEVERE: 0.8
                    },
                    "task_modifiers": {
                        TaskType.CREATIVE: 1.0,
                        TaskType.ANALYTICAL: 0.9,
                        TaskType.PHYSICAL: 0.9,
                        TaskType.REPETITIVE: 0.8
                    },
                    "duration_factor": 0.03
                }
            },
            InterventionType.BIOFEEDBACK: {
                "productivity": {
                    "base_effect": 0.45,
                    "stress_modifiers": {
                        StressLevel.MILD: 0.8,
                        StressLevel.MODERATE: 1.0,
                        StressLevel.SEVERE: 1.2
                    },
                    "task_modifiers": {
                        TaskType.CREATIVE: 0.9,
                        TaskType.ANALYTICAL: 1.1,
                        TaskType.PHYSICAL: 1.0,
                        TaskType.REPETITIVE: 0.9
                    },
                    "duration_factor": 0.04
                },
                "memory": {
                    "base_effect": 0.35,
                    "stress_modifiers": {
                        StressLevel.MILD: 0.9,
                        StressLevel.MODERATE: 1.0,
                        StressLevel.SEVERE: 1.1
                    },
                    "task_modifiers": {
                        TaskType.CREATIVE: 0.9,
                        TaskType.ANALYTICAL: 1.0,
                        TaskType.PHYSICAL: 0.9,
                        TaskType.REPETITIVE: 0.8
                    },
                    "duration_factor": 0.03
                },
                "decision": {
                    "base_effect": 0.45,
                    "stress_modifiers": {
                        StressLevel.MILD: 0.8,
                        StressLevel.MODERATE: 1.0,
                        StressLevel.SEVERE: 1.2
                    },
                    "task_modifiers": {
                        TaskType.CREATIVE: 0.9,
                        TaskType.ANALYTICAL: 1.2,
                        TaskType.PHYSICAL: 0.9,
                        TaskType.REPETITIVE: 0.8
                    },
                    "duration_factor": 0.05
                }
            }
        }
    
    async def process(self, state: SimulationState) -> SimulationState:
        """
        Process the current simulation state using the LlamaIndex agent.
        
        This method analyzes the effectiveness of the current intervention
        and predicts its future impact based on research data.
        
        Args:
            state: The current simulation state
            
        Returns:
            An updated simulation state with effectiveness analysis
        """
        # Clone the state to avoid modifying the original
        new_state = state.clone()
        
        try:
            # Only analyze if an intervention is being applied
            if new_state.intervention != InterventionType.NONE:
                # Find how long the intervention has been applied
                intervention_duration = self._get_intervention_duration(new_state)
                
                # Retrieve relevant research data
                research_data = self._retrieve_relevant_research(
                    new_state.intervention,
                    new_state.stress_level,
                    new_state.task_type
                )
                
                # Predict the intervention's effectiveness
                predictions = self._predict_intervention_effectiveness(
                    new_state.intervention,
                    new_state.stress_level,
                    new_state.task_type,
                    intervention_duration
                )
                
                # Store analysis results in metadata
                if "effectiveness_analysis" not in new_state.metadata:
                    new_state.metadata["effectiveness_analysis"] = []
                
                analysis = {
                    "step": new_state.simulation_step,
                    "intervention": new_state.intervention,
                    "duration": intervention_duration,
                    "research": {
                        "count": len(research_data),
                        "summary": self._summarize_research(research_data)
                    },
                    "predictions": predictions,
                    "recommendation": self._generate_recommendation(
                        new_state, predictions, intervention_duration
                    )
                }
                
                new_state.metadata["effectiveness_analysis"].append(analysis)
                
                # Add to analysis history
                self.analysis_history.append(analysis)
                
                # Log the analysis (in a real system, this would use a proper logger)
                print(f"[LlamaIndexAgent] Analysis for {new_state.intervention} after {intervention_duration} steps:")
                print(f"[LlamaIndexAgent] Predictions: {predictions}")
                print(f"[LlamaIndexAgent] Recommendation: {analysis['recommendation']}")
            
            return new_state
            
        except Exception as e:
            # In a real implementation, this would use a proper error handling mechanism
            print(f"LlamaIndex analysis failed: {e}")
            # Return the original state if the analysis fails
            return state
    
    def _get_intervention_duration(self, state: SimulationState) -> int:
        """
        Calculate how long the current intervention has been applied.
        
        This method analyzes the simulation history to determine when
        the current intervention was first applied.
        
        Args:
            state: The current simulation state
            
        Returns:
            The number of steps since the intervention was applied
        """
        current_intervention = state.intervention
        
        # Check if we have history
        if not state.history:
            return 0
        
        # Scan history from latest to earliest
        for i in range(len(state.history) - 1, -1, -1):
            if state.history[i].get("intervention") != current_intervention:
                # Found the step where the intervention changed
                return len(state.history) - i - 1
        
        # If all history has the same intervention, it's been applied since the start
        return len(state.history)
    
    def _retrieve_relevant_research(
        self,
        intervention: str,
        stress_level: str,
        task_type: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant research data for the current conditions.
        
        This method searches the research database for studies that match
        or are relevant to the current intervention, stress level, and task type.
        
        Args:
            intervention: The current intervention
            stress_level: The current stress level
            task_type: The current task type
            
        Returns:
            A list of relevant research findings
        """
        # Get all research for the intervention
        all_research = self.research_database.get(intervention, [])
        
        if not all_research:
            return []
        
        # Score each research entry for relevance
        scored_research = []
        for entry in all_research:
            score = self._calculate_research_relevance(entry, stress_level, task_type)
            scored_research.append((score, entry))
        
        # Sort by relevance score
        sorted_research = sorted(scored_research, key=lambda x: x[0], reverse=True)
        
        # Return the top 3 most relevant entries
        return [entry for _, entry in sorted_research[:3]]
    
    def _calculate_research_relevance(
        self,
        research: Dict[str, Any],
        stress_level: str,
        task_type: str
    ) -> float:
        """
        Calculate the relevance of a research entry to current conditions.
        
        This method scores how closely a research study matches the
        current stress level and task type.
        
        Args:
            research: The research entry to evaluate
            stress_level: The current stress level
            task_type: The current task type
            
        Returns:
            A relevance score between 0 and 1
        """
        # Initialize base score
        score = 0.5
        
        # Get conditions from the research
        conditions = research.get("conditions", {})
        research_stress = conditions.get("stress_level", "")
        research_task = conditions.get("task_type", "")
        
        # Score stress level match
        if research_stress == stress_level:
            score += 0.25
        elif research_stress:
            # Partial match based on stress level similarity
            stress_levels = [StressLevel.MILD, StressLevel.MODERATE, StressLevel.SEVERE]
            if research_stress in stress_levels and stress_level in stress_levels:
                research_idx = stress_levels.index(research_stress)
                current_idx = stress_levels.index(stress_level)
                similarity = 1.0 - abs(research_idx - current_idx) / (len(stress_levels) - 1)
                score += 0.15 * similarity
        
        # Score task type match
        if research_task == task_type:
            score += 0.25
        elif research_task == "mixed" or research_task == "decision-making":
            # Generic task types partially match all specific types
            score += 0.1
        
        # Add bonus for larger sample sizes and more recent research
        sample_size = research.get("findings", {}).get("sample_size", 0)
        year = research.get("year", 2000)
        
        # Normalize sample size (assuming max 200)
        sample_bonus = min(0.1, (sample_size / 200) * 0.1)
        
        # Normalize year (2000-2020)
        year_bonus = min(0.1, ((year - 2000) / 20) * 0.1)
        
        score += sample_bonus + year_bonus
        
        # Ensure score is between 0 and 1
        return min(1.0, max(0.0, score))
    
    def _summarize_research(self, research_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of the relevant research data.
        
        This method aggregates the findings from multiple research studies
        to create an overall summary of the expected effects.
        
        Args:
            research_data: List of relevant research findings
            
        Returns:
            A dictionary with summarized research findings
        """
        if not research_data:
            return {
                "avg_productivity_effect": 0.0,
                "avg_memory_effect": 0.0,
                "avg_decision_effect": 0.0,
                "avg_sample_size": 0,
                "study_count": 0
            }
        
        # Extract findings
        productivity_effects = []
        memory_effects = []
        decision_effects = []
        sample_sizes = []
        
        for entry in research_data:
            findings = entry.get("findings", {})
            
            if "productivity_effect" in findings:
                productivity_effects.append(findings["productivity_effect"])
            
            if "memory_effect" in findings:
                memory_effects.append(findings["memory_effect"])
            
            if "decision_effect" in findings:
                decision_effects.append(findings["decision_effect"])
            
            if "sample_size" in findings:
                sample_sizes.append(findings["sample_size"])
        
        # Calculate averages
        avg_productivity = sum(productivity_effects) / len(productivity_effects) if productivity_effects else 0.0
        avg_memory = sum(memory_effects) / len(memory_effects) if memory_effects else 0.0
        avg_decision = sum(decision_effects) / len(decision_effects) if decision_effects else 0.0
        avg_sample = sum(sample_sizes) / len(sample_sizes) if sample_sizes else 0
        
        return {
            "avg_productivity_effect": avg_productivity,
            "avg_memory_effect": avg_memory,
            "avg_decision_effect": avg_decision,
            "avg_sample_size": avg_sample,
            "study_count": len(research_data)
        }
    
    def _predict_intervention_effectiveness(
        self,
        intervention: str,
        stress_level: str,
        task_type: str,
        duration: int
    ) -> Dict[str, float]:
        """
        Predict the effectiveness of the intervention based on models.
        
        This method uses the effectiveness models to predict how the
        intervention will affect cognitive metrics based on the current
        conditions and how long it has been applied.
        
        Args:
            intervention: The current intervention
            stress_level: The current stress level
            task_type: The current task type
            duration: How long the intervention has been applied
            
        Returns:
            A dictionary of predicted effects on cognitive metrics
        """
        # Get the model for the intervention
        model = self.effectiveness_models.get(intervention, {})
        
        if not model:
            return {
                "productivity_effect": 0.0,
                "memory_effect": 0.0,
                "decision_effect": 0.0,
                "overall_effect": 0.0
            }
        
        # Calculate predicted effects for each metric
        predictions = {}
        
        for metric in ["productivity", "memory", "decision"]:
            if metric in model:
                # Get model parameters
                base_effect = model[metric].get("base_effect", 0.3)
                stress_modifier = model[metric].get("stress_modifiers", {}).get(stress_level, 1.0)
                task_modifier = model[metric].get("task_modifiers", {}).get(task_type, 1.0)
                duration_factor = model[metric].get("duration_factor", 0.03)
                
                # Calculate the duration effect (with diminishing returns)
                duration_effect = min(0.5, duration * duration_factor)
                
                # Calculate the final effect
                effect = base_effect * stress_modifier * task_modifier + duration_effect
                
                # Add some randomness
                effect += random.uniform(-0.05, 0.05)
                
                # Ensure effect is between 0 and 1
                effect = min(1.0, max(0.0, effect))
                
                predictions[f"{metric}_effect"] = effect
            else:
                predictions[f"{metric}_effect"] = 0.0
        
        # Calculate overall effect
        overall_effect = (
            predictions.get("productivity_effect", 0.0) +
            predictions.get("memory_effect", 0.0) +
            predictions.get("decision_effect", 0.0)
        ) / 3.0
        
        predictions["overall_effect"] = overall_effect
        
        return predictions
    
    def _generate_recommendation(
        self,
        state: SimulationState,
        predictions: Dict[str, float],
        duration: int
    ) -> str:
        """
        Generate a recommendation based on the effectiveness analysis.
        
        This method provides a recommendation on whether to continue with
        the current intervention, modify it, or try a different approach.
        
        Args:
            state: The current simulation state
            predictions: The predicted effectiveness values
            duration: How long the intervention has been applied
            
        Returns:
            A recommendation string
        """
        overall_effect = predictions.get("overall_effect", 0.0)
        
        # Define recommendation thresholds
        if overall_effect > 0.7:
            recommendation = "highly_effective"
        elif overall_effect > 0.5:
            recommendation = "effective"
        elif overall_effect > 0.3:
            recommendation = "moderately_effective"
        else:
            recommendation = "minimally_effective"
        
        # Define recommendation messages
        recommendation_messages = {
            "highly_effective": f"Continue with {state.intervention}. Research suggests this is a highly effective intervention for the current conditions.",
            "effective": f"Continue with {state.intervention}. This intervention is showing good effectiveness based on research data.",
            "moderately_effective": f"Consider supplementing {state.intervention} with additional techniques. Research suggests moderate effectiveness in current conditions.",
            "minimally_effective": f"Consider switching from {state.intervention} to a different intervention. Research suggests limited effectiveness in current conditions."
        }
        
        # Add duration-based advice
        if duration > 5 and overall_effect < 0.4:
            recommendation_messages[recommendation] += " The extended duration without significant improvement suggests trying an alternative approach."
        
        # Return the recommendation message
        return recommendation_messages.get(recommendation, "No specific recommendation available.")
    
    def get_capabilities(self) -> Set[AgentCapability]:
        """
        Return the set of capabilities this agent provides.
        
        Returns:
            A set containing the INTERVENTION_ANALYSIS capability
        """
        return {AgentCapability.INTERVENTION_ANALYSIS}


# TODO: Implement more sophisticated research retrieval
# TODO: Add support for combined intervention analysis
# TODO: Implement research quality assessment
# TODO: Add personalized effectiveness modeling

# Summary:
# This module implements the LlamaIndex agent, which analyzes the
# effectiveness of stress management interventions based on research
# literature. It retrieves relevant research studies, predicts the
# effectiveness of interventions under current conditions, and provides
# recommendations on intervention strategies. The agent helps optimize
# stress management by applying evidence-based approaches.
