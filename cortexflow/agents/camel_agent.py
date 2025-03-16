"""
CAMEL agent implementation for the CortexFlow system.

This module implements the CAMEL agent, which is responsible for
simulating the stress dialogue between cognitive processes. It 
uses a conversational model to simulate how stress affects internal
thoughts and cognitive performance.
"""

import random
from typing import Dict, Any, Set, Optional, List, Tuple

from cortexflow.agents.base import AgentBase
from cortexflow.core.state import SimulationState
from cortexflow.core.types import AgentType, AgentCapability, StressLevel, TaskType


class CAMELAgent(AgentBase):
    """
    CAMEL agent for simulating stress dialogue.
    
    This agent simulates the internal dialogue that occurs during stress,
    including negative thoughts, self-talk patterns, and coping strategies.
    It uses a dialogue model to represent how stress manifests in cognition.
    
    Attributes:
        thought_patterns: Dictionary of stress-related thought patterns
        coping_strategies: List of cognitive coping strategies
        dialogue_history: History of simulated internal dialogue
    """
    
    def __init__(
        self,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new CAMEL agent.
        
        Args:
            enabled: Whether the agent is currently enabled
            config: Agent-specific configuration parameters
        """
        super().__init__("CAMELAgent", AgentType.PROCESSING, enabled, config or {})
        
        # Initialize thought patterns
        self.thought_patterns = self._init_thought_patterns()
        
        # Initialize coping strategies
        self.coping_strategies = self._init_coping_strategies()
        
        # Initialize dialogue history
        self.dialogue_history = []
    
    def _init_thought_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Initialize the stress-related thought patterns.
        
        These patterns represent common cognitive distortions and thought
        processes that occur under different levels of stress.
        
        Returns:
            A dictionary of stress levels and associated thought patterns
        """
        return {
            StressLevel.MILD: {
                "self_talk": [
                    "I can handle this.",
                    "This is a bit challenging, but I'll manage.",
                    "I need to focus a little more than usual.",
                    "I'm feeling a little tense, but it's okay."
                ],
                "concerns": [
                    "Will I finish this on time?",
                    "Am I doing this right?",
                    "I should be more efficient."
                ],
                "distractions": [
                    "Maybe I should check my messages quickly.",
                    "I wonder what else I need to do today.",
                    "I'm getting a bit hungry."
                ]
            },
            StressLevel.MODERATE: {
                "self_talk": [
                    "This is getting difficult.",
                    "I'm not sure I can keep up with this.",
                    "I need to work harder.",
                    "Why is this so challenging?",
                    "I'm really feeling the pressure now."
                ],
                "concerns": [
                    "I'm falling behind.",
                    "Others are probably doing better than me.",
                    "What if I make a serious mistake?",
                    "I'm not sure I'm good enough for this."
                ],
                "distractions": [
                    "I keep losing my train of thought.",
                    "It's hard to concentrate with all this pressure.",
                    "My mind keeps wandering to other problems.",
                    "I can't stop thinking about the consequences of failure."
                ]
            },
            StressLevel.SEVERE: {
                "self_talk": [
                    "I can't do this.",
                    "This is impossible.",
                    "I'm going to fail completely.",
                    "I'm not cut out for this.",
                    "Everything is falling apart.",
                    "I'm overwhelmed and can't cope."
                ],
                "concerns": [
                    "This is going to be a disaster.",
                    "Everyone will see that I'm incompetent.",
                    "I've never been good enough.",
                    "I always mess up under pressure.",
                    "My career/reputation is at stake."
                ],
                "distractions": [
                    "I can't focus at all.",
                    "My mind is completely scattered.",
                    "All I can think about is escaping this situation.",
                    "My anxiety is consuming all my mental energy.",
                    "I'm physically uncomfortable and can't ignore it."
                ]
            }
        }
    
    def _init_coping_strategies(self) -> Dict[str, List[str]]:
        """
        Initialize cognitive coping strategies.
        
        These strategies represent techniques that can be used to manage
        stress and improve cognitive performance under pressure.
        
        Returns:
            A dictionary of intervention types and associated coping strategies
        """
        return {
            "none": [
                "Just push through it.",
                "Keep working despite the stress.",
                "Try to ignore the discomfort.",
                "Focus on the task, not the feelings."
            ],
            "meditation": [
                "Take a deep breath and center yourself.",
                "Observe your thoughts without judgment.",
                "Focus on the present moment.",
                "Notice your breath moving in and out.",
                "Let go of worrying thoughts about the future."
            ],
            "micro_breaks": [
                "Take a short break to reset.",
                "Step away from the task for a moment.",
                "Stretch and move your body briefly.",
                "Look away from the screen and rest your eyes.",
                "Give yourself permission to pause."
            ],
            "biofeedback": [
                "Notice your physical tension and consciously relax.",
                "Recognize that your heart rate is elevated and focus on calming it.",
                "Pay attention to your breathing pattern and regulate it.",
                "Use the feedback to recognize and reduce your stress response.",
                "Make small adjustments based on your body's signals."
            ]
        }
    
    async def process(self, state: SimulationState) -> SimulationState:
        """
        Process the current simulation state using the CAMEL agent.
        
        This method simulates the internal dialogue that occurs during
        stress and calculates its impact on cognitive performance.
        
        Args:
            state: The current simulation state
            
        Returns:
            An updated simulation state reflecting the stress dialogue effects
        """
        # Clone the state to avoid modifying the original
        new_state = state.clone()
        
        try:
            # Generate internal dialogue based on current state
            dialogue = self._generate_dialogue(new_state)
            
            # Add the dialogue to the history
            self.dialogue_history.append(dialogue)
            
            # Calculate the cognitive impact of the dialogue
            cognitive_impact = self._calculate_dialogue_impact(dialogue, new_state)
            
            # Apply the impact to the state
            new_state = self._apply_cognitive_impact(new_state, cognitive_impact)
            
            # Log the dialogue (in a real system, this would use a proper logger)
            print(f"[CAMELAgent] Simulated dialogue: {dialogue['summary']}")
            print(f"[CAMELAgent] Cognitive impact: {cognitive_impact}")
            
            return new_state
            
        except Exception as e:
            # In a real implementation, this would use a proper error handling mechanism
            print(f"CAMEL simulation failed: {e}")
            # Return the original state if the simulation fails
            return state
    
    def _generate_dialogue(self, state: SimulationState) -> Dict[str, Any]:
        """
        Generate simulated internal dialogue based on the current state.
        
        This method creates a representation of the internal thoughts and
        self-talk that occur under the current stress conditions, including
        any coping strategies if interventions are active.
        
        Args:
            state: The current simulation state
            
        Returns:
            A dictionary containing the simulated dialogue elements
        """
        # Get the thought patterns for the current stress level
        patterns = self.thought_patterns.get(
            state.stress_level, self.thought_patterns[StressLevel.MODERATE]
        )
        
        # Get the coping strategies for the current intervention
        strategies = self.coping_strategies.get(
            state.intervention, self.coping_strategies["none"]
        )
        
        # Select random thoughts from each category
        self_talk = random.sample(
            patterns["self_talk"], 
            min(2, len(patterns["self_talk"]))
        )
        concerns = random.sample(
            patterns["concerns"], 
            min(2, len(patterns["concerns"]))
        )
        distractions = random.sample(
            patterns["distractions"], 
            min(1, len(patterns["distractions"]))
        )
        
        # Select a coping strategy if an intervention is active
        coping = []
        if state.intervention != "none":
            coping = random.sample(
                strategies, 
                min(2, len(strategies))
            )
        
        # Construct the dialogue
        dialogue = {
            "stress_level": state.stress_level,
            "task_type": state.task_type,
            "intervention": state.intervention,
            "self_talk": self_talk,
            "concerns": concerns,
            "distractions": distractions,
            "coping": coping,
            "summary": self._create_dialogue_summary(
                self_talk, concerns, distractions, coping
            )
        }
        
        return dialogue
    
    def _create_dialogue_summary(
        self,
        self_talk: List[str],
        concerns: List[str],
        distractions: List[str],
        coping: List[str]
    ) -> str:
        """
        Create a summary of the internal dialogue.
        
        This method combines the different elements of the internal dialogue
        into a coherent summary that represents the stress experience.
        
        Args:
            self_talk: List of self-talk statements
            concerns: List of concerns
            distractions: List of distractions
            coping: List of coping strategies
            
        Returns:
            A string summarizing the internal dialogue
        """
        summary_parts = []
        
        # Add self-talk
        if self_talk:
            summary_parts.append(f"Self-talk: {' '.join(self_talk)}")
        
        # Add concerns
        if concerns:
            summary_parts.append(f"Concerns: {' '.join(concerns)}")
        
        # Add distractions
        if distractions:
            summary_parts.append(f"Distractions: {' '.join(distractions)}")
        
        # Add coping
        if coping:
            summary_parts.append(f"Coping: {' '.join(coping)}")
        
        # Join all parts
        return " | ".join(summary_parts)
    
    def _calculate_dialogue_impact(
        self,
        dialogue: Dict[str, Any],
        state: SimulationState
    ) -> Dict[str, float]:
        """
        Calculate the cognitive impact of the internal dialogue.
        
        This method analyzes the simulated dialogue to determine its
        impact on productivity, memory, and decision quality.
        
        Args:
            dialogue: The simulated internal dialogue
            state: The current simulation state
            
        Returns:
            A dictionary of impact values for each cognitive metric
        """
        # Initialize base impact values
        impact = {
            "productivity": 0.0,
            "memory": 0.0,
            "decision": 0.0
        }
        
        # Calculate negative impact from self-talk, concerns, and distractions
        negative_count = len(dialogue["self_talk"]) + len(dialogue["concerns"]) + len(dialogue["distractions"])
        negative_impact = -0.5 * negative_count
        
        # Calculate positive impact from coping strategies
        positive_impact = 0.3 * len(dialogue["coping"])
        
        # Distribution of impact across cognitive metrics depends on task type
        if state.task_type == TaskType.CREATIVE:
            # Creative tasks are most affected by negative self-talk
            impact["productivity"] = negative_impact * 0.5 + positive_impact * 0.4
            impact["memory"] = negative_impact * 0.2 + positive_impact * 0.3
            impact["decision"] = negative_impact * 0.3 + positive_impact * 0.3
        elif state.task_type == TaskType.ANALYTICAL:
            # Analytical tasks are most affected by distractions
            impact["productivity"] = negative_impact * 0.3 + positive_impact * 0.3
            impact["memory"] = negative_impact * 0.3 + positive_impact * 0.3
            impact["decision"] = negative_impact * 0.4 + positive_impact * 0.4
        elif state.task_type == TaskType.PHYSICAL:
            # Physical tasks are most affected by physical distraction
            impact["productivity"] = negative_impact * 0.5 + positive_impact * 0.5
            impact["memory"] = negative_impact * 0.2 + positive_impact * 0.2
            impact["decision"] = negative_impact * 0.3 + positive_impact * 0.3
        else:  # Repetitive tasks
            # Repetitive tasks are most affected by mind wandering
            impact["productivity"] = negative_impact * 0.4 + positive_impact * 0.4
            impact["memory"] = negative_impact * 0.3 + positive_impact * 0.3
            impact["decision"] = negative_impact * 0.3 + positive_impact * 0.3
        
        # Add some randomness
        for metric in impact:
            impact[metric] += random.uniform(-0.2, 0.2)
        
        return impact
    
    def _apply_cognitive_impact(
        self,
        state: SimulationState,
        impact: Dict[str, float]
    ) -> SimulationState:
        """
        Apply the cognitive impact to the simulation state.
        
        This method updates the productivity, memory, and decision quality
        metrics based on the calculated impact of the internal dialogue.
        
        Args:
            state: The current simulation state
            impact: Dictionary of impact values for each cognitive metric
            
        Returns:
            The updated simulation state
        """
        # Apply impact to each metric
        state.productivity_score += impact["productivity"]
        state.memory_efficiency += impact["memory"]
        state.decision_quality += impact["decision"]
        
        # Ensure values stay within bounds
        state.productivity_score = max(0, min(100, state.productivity_score))
        state.memory_efficiency = max(0, min(100, state.memory_efficiency))
        state.decision_quality = max(0, min(100, state.decision_quality))
        
        return state
    
    def get_capabilities(self) -> Set[AgentCapability]:
        """
        Return the set of capabilities this agent provides.
        
        Returns:
            A set containing the STRESS_DIALOGUE capability
        """
        return {AgentCapability.STRESS_DIALOGUE}


# TODO: Implement more sophisticated dialogue generation
# TODO: Add support for personalized dialogue patterns
# TODO: Implement dialogue analysis tools
# TODO: Add support for external influences on internal dialogue

# Summary:
# This module implements the CAMEL agent, which simulates the internal dialogue
# that occurs during stress. It models how negative thoughts, self-talk, and
# coping strategies affect cognitive performance, and how different intervention
# strategies can improve this dialogue. The agent uses a conversation-based
# approach to represent the complex interplay between stress and cognition.
