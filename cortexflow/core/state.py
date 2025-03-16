"""
Simulation state management for the CortexFlow system.

This module contains the SimulationState class, which represents the current
state of a stress-productivity simulation, including stress levels, cortisol
and dopamine levels, productivity metrics, and simulation history.
"""

from typing import Dict, List, Optional, Any
import copy
import json

from cortexflow.core.types import (
    StressLevel,
    TaskType,
    InterventionType,
    SimulationStep,
    MetricValue,
    JSON
)


class SimulationState:
    """
    Represents the current state of a stress-productivity simulation.
    
    This class maintains all the metrics and parameters that define the
    simulation state at a given point in time. It includes methods for
    updating, recording history, and serializing/deserializing the state.
    
    Attributes:
        stress_level: Current stress level setting
        task_type: Type of task being performed
        intervention: Current intervention being applied
        cortisol_level: Current cortisol hormone level
        dopamine_level: Current dopamine neurotransmitter level
        productivity_score: Overall productivity score
        memory_efficiency: Memory function efficiency
        decision_quality: Decision-making quality
        simulation_step: Current step in the simulation
        history: Record of all previous states
        metadata: Additional metadata about the simulation
    """
    
    def __init__(
        self,
        stress_level: str = StressLevel.MODERATE,
        task_type: str = TaskType.CREATIVE,
        intervention: str = InterventionType.NONE,
        cortisol_level: float = 50.0,
        dopamine_level: float = 50.0,
        productivity_score: float = 100.0,
        memory_efficiency: float = 100.0,
        decision_quality: float = 100.0,
        simulation_step: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new simulation state.
        
        Args:
            stress_level: Starting stress level
            task_type: Type of task being simulated
            intervention: Intervention strategy applied
            cortisol_level: Initial cortisol level (0-100)
            dopamine_level: Initial dopamine level (0-100)
            productivity_score: Initial productivity score (0-100)
            memory_efficiency: Initial memory efficiency (0-100)
            decision_quality: Initial decision quality (0-100)
            simulation_step: Starting simulation step
            history: Optional history from previous states
            metadata: Optional metadata about the simulation
        """
        # Validate enum values
        self.stress_level = self._validate_enum(stress_level, StressLevel)
        self.task_type = self._validate_enum(task_type, TaskType)
        self.intervention = self._validate_enum(intervention, InterventionType)
        
        # Set metric values with bounds checking
        self.cortisol_level = self._bound_value(cortisol_level, 0, 100)
        self.dopamine_level = self._bound_value(dopamine_level, 0, 100)
        self.productivity_score = self._bound_value(productivity_score, 0, 100)
        self.memory_efficiency = self._bound_value(memory_efficiency, 0, 100)
        self.decision_quality = self._bound_value(decision_quality, 0, 100)
        
        # Set simulation tracking variables
        self.simulation_step = max(0, simulation_step)
        self.history = history or []
        self.metadata = metadata or {}
    
    def _validate_enum(self, value: str, enum_class: Any) -> str:
        """
        Validate that a string value is a valid enum member.
        
        Args:
            value: The string value to validate
            enum_class: The enum class to validate against
            
        Returns:
            The validated enum value
            
        Raises:
            ValueError: If the value is not a valid enum member
        """
        try:
            # Try to access the enum by name
            enum_value = enum_class(value)
            return enum_value
        except ValueError:
            valid_values = [e.value for e in enum_class]
            raise ValueError(
                f"Invalid value '{value}' for {enum_class.__name__}. "
                f"Valid values are: {', '.join(valid_values)}"
            )
    
    def _bound_value(self, value: float, min_val: float, max_val: float) -> float:
        """
        Ensure a value stays within specified bounds.
        
        Args:
            value: The value to bound
            min_val: The minimum allowed value
            max_val: The maximum allowed value
            
        Returns:
            The bounded value
        """
        return max(min_val, min(max_val, value))
    
    def update_history(self) -> None:
        """
        Record the current state in the history.
        
        This creates a snapshot of all current metrics and adds it to the
        history list, then increments the simulation step counter.
        """
        # Create a snapshot of current metrics
        snapshot = {
            "step": self.simulation_step,
            "stress_level": self.stress_level,
            "cortisol_level": self.cortisol_level,
            "dopamine_level": self.dopamine_level,
            "productivity_score": self.productivity_score,
            "memory_efficiency": self.memory_efficiency,
            "decision_quality": self.decision_quality,
            "intervention": self.intervention
        }
        
        # Add the snapshot to history
        self.history.append(snapshot)
        
        # Increment the simulation step
        self.simulation_step += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the current state to a dictionary.
        
        Returns:
            A dictionary representation of the current state
        """
        return {
            "stress_level": self.stress_level,
            "task_type": self.task_type,
            "intervention": self.intervention,
            "cortisol_level": self.cortisol_level,
            "dopamine_level": self.dopamine_level,
            "productivity_score": self.productivity_score,
            "memory_efficiency": self.memory_efficiency,
            "decision_quality": self.decision_quality,
            "simulation_step": self.simulation_step,
            "history": self.history,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """
        Convert the current state to a JSON string.
        
        Returns:
            A JSON string representation of the current state
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationState':
        """
        Create a SimulationState instance from a dictionary.
        
        Args:
            data: Dictionary containing state data
            
        Returns:
            A new SimulationState instance
        """
        # Extract required parameters with defaults if missing
        return cls(
            stress_level=data.get("stress_level", StressLevel.MODERATE),
            task_type=data.get("task_type", TaskType.CREATIVE),
            intervention=data.get("intervention", InterventionType.NONE),
            cortisol_level=data.get("cortisol_level", 50.0),
            dopamine_level=data.get("dopamine_level", 50.0),
            productivity_score=data.get("productivity_score", 100.0),
            memory_efficiency=data.get("memory_efficiency", 100.0),
            decision_quality=data.get("decision_quality", 100.0),
            simulation_step=data.get("simulation_step", 0),
            history=data.get("history", []),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SimulationState':
        """
        Create a SimulationState instance from a JSON string.
        
        Args:
            json_str: JSON string containing state data
            
        Returns:
            A new SimulationState instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def clone(self) -> 'SimulationState':
        """
        Create a deep copy of the current state.
        
        Returns:
            A new SimulationState instance with identical values
        """
        return SimulationState(
            stress_level=self.stress_level,
            task_type=self.task_type,
            intervention=self.intervention,
            cortisol_level=self.cortisol_level,
            dopamine_level=self.dopamine_level,
            productivity_score=self.productivity_score,
            memory_efficiency=self.memory_efficiency,
            decision_quality=self.decision_quality,
            simulation_step=self.simulation_step,
            history=copy.deepcopy(self.history),
            metadata=copy.deepcopy(self.metadata)
        )


# TODO: Add methods for state comparison and difference calculation
# TODO: Implement more sophisticated history tracking (e.g., compression, sampling)
# TODO: Add visualization methods directly on the state object
# TODO: Consider implementing state validation against a schema

# Summary:
# This module defines the SimulationState class, which represents the current
# state of a stress-productivity simulation. It provides methods for
# initialization, validation, updating history, and serialization. The class
# is designed to be immutable, with methods returning new instances rather
# than modifying the current instance.
