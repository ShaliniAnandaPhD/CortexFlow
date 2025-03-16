"""
Type definitions and enumerations for the CortexFlow system.

This module contains all the core type definitions used throughout the system,
including enumerations for stress levels, task types, and intervention types.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any


class StressLevel(str, Enum):
    """
    Enumeration of possible stress levels for the simulation.
    
    Attributes:
        MILD: Low level of stress
        MODERATE: Medium level of stress
        SEVERE: High level of stress
    """
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class TaskType(str, Enum):
    """
    Enumeration of task types that can be simulated.
    
    Attributes:
        CREATIVE: Tasks requiring creative thinking and innovation
        ANALYTICAL: Tasks requiring logical analysis and problem-solving
        PHYSICAL: Tasks requiring physical effort and coordination
        REPETITIVE: Routine, repetitive tasks requiring minimal variation
    """
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    PHYSICAL = "physical"
    REPETITIVE = "repetitive"


class InterventionType(str, Enum):
    """
    Enumeration of intervention strategies that can be applied.
    
    Attributes:
        NONE: No intervention applied
        MEDITATION: Mindfulness meditation practices
        MICRO_BREAKS: Short breaks during work periods
        BIOFEEDBACK: Using biometric data to guide stress reduction
    """
    NONE = "none"
    MEDITATION = "meditation"
    MICRO_BREAKS = "micro_breaks"
    BIOFEEDBACK = "biofeedback"


class AgentType(str, Enum):
    """
    Enumeration of agent types based on their layer in the architecture.
    
    Attributes:
        ORCHESTRATION: Agents responsible for coordination and state management
        PROCESSING: Agents handling core processing of cognitive functions
        KNOWLEDGE: Agents providing knowledge and recommendations
        OPTIMIZATION: Agents optimizing workflows and processes
    """
    ORCHESTRATION = "orchestration"
    PROCESSING = "processing"
    KNOWLEDGE = "knowledge"
    OPTIMIZATION = "optimization"


class AgentCapability(str, Enum):
    """
    Enumeration of capabilities that agents can provide.
    
    These capabilities are used for agent discovery and routing.
    """
    STATE_MANAGEMENT = "state_management"
    RUNTIME_ENVIRONMENT = "runtime_environment"
    NEUROCHEMICAL_DIALOGUE = "neurochemical_dialogue"
    MEMORY_ANALYSIS = "memory_analysis"
    PRODUCTIVITY_ANALYSIS = "productivity_analysis"
    STRESS_DIALOGUE = "stress_dialogue"
    INTERVENTION_RECOMMENDATION = "intervention_recommendation"
    INTERVENTION_ANALYSIS = "intervention_analysis"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"


# Type aliases for improved code readability
AgentName = str
AgentConfig = Dict[str, Any]
SimulationStep = int
MetricValue = float
JSON = Dict[str, Any]

# TODO: Add more specialized type definitions as the system expands
# TODO: Consider using Pydantic models for more complex types
# TODO: Add validation functions for each enum type

# Summary:
# This module defines the core type definitions for the CortexFlow system.
# It includes enumerations for stress levels, task types, intervention types,
# agent types, and agent capabilities. It also defines type aliases for
# improved code readability.
