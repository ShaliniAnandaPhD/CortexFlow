"""
Base agent implementation and interfaces for the CortexFlow system.

This module defines the base agent class and interfaces that all agent
implementations must follow, ensuring consistent behavior across different
agent types.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set

from cortexflow.core.state import SimulationState
from cortexflow.core.types import AgentCapability, AgentType


class AgentBase(ABC):
    """
    Abstract base class for all agents in the CortexFlow system.
    
    This class defines the common interface that all agents must implement,
    including methods for processing simulation state and reporting capabilities.
    
    Attributes:
        name: Unique name of the agent
        agent_type: Type of agent (orchestration, processing, etc.)
        enabled: Whether the agent is currently enabled
        config: Agent-specific configuration parameters
    """
    
    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new agent.
        
        Args:
            name: Unique name of the agent
            agent_type: Type of agent (orchestration, processing, etc.)
            enabled: Whether the agent is currently enabled
            config: Agent-specific configuration parameters
        """
        self.name = name
        self.agent_type = agent_type
        self.enabled = enabled
        self.config = config or {}
    
    @abstractmethod
    async def process(self, state: SimulationState) -> SimulationState:
        """
        Process the current simulation state and return an updated state.
        
        This is the core method that all agents must implement. It takes the
        current simulation state, applies agent-specific logic, and returns
        an updated state. This method should not modify the input state but
        instead return a new instance.
        
        Args:
            state: The current simulation state
            
        Returns:
            An updated simulation state
            
        Raises:
            NotImplementedError: If the agent does not implement this method
        """
        raise NotImplementedError("Agents must implement process method")
    
    @abstractmethod
    def get_capabilities(self) -> Set[AgentCapability]:
        """
        Return the set of capabilities this agent provides.
        
        This method allows the system to discover what capabilities each agent
        provides, enabling dynamic routing and fallback mechanisms.
        
        Returns:
            A set of AgentCapability enums
            
        Raises:
            NotImplementedError: If the agent does not implement this method
        """
        raise NotImplementedError("Agents must implement get_capabilities method")
    
    def validate_config(self) -> bool:
        """
        Validate the agent's configuration.
        
        This method checks if the agent's configuration is valid and complete.
        Subclasses can override this method to implement agent-specific
        validation logic.
        
        Returns:
            True if the configuration is valid, False otherwise
        """
        # Base implementation assumes any configuration is valid
        return True
    
    def is_available(self) -> bool:
        """
        Check if the agent is available for use.
        
        This method checks if the agent is enabled and properly configured.
        Subclasses can override this method to implement more sophisticated
        availability checks, such as API connectivity tests.
        
        Returns:
            True if the agent is available, False otherwise
        """
        return self.enabled and self.validate_config()
    
    def __str__(self) -> str:
        """
        Return a string representation of the agent.
        
        Returns:
            A string representation of the agent
        """
        status = "enabled" if self.enabled else "disabled"
        return f"{self.name} ({self.agent_type}) - {status}"
    
    def __repr__(self) -> str:
        """
        Return a developer string representation of the agent.
        
        Returns:
            A detailed string representation of the agent
        """
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"agent_type='{self.agent_type}', "
                f"enabled={self.enabled}, "
                f"config={self.config})")


class AgentExecutionError(Exception):
    """
    Exception raised when an agent fails to execute.
    
    Attributes:
        agent_name: The name of the agent that failed
        message: The error message
        original_error: The original exception that caused the failure
    """
    
    def __init__(
        self,
        agent_name: str,
        message: str,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize a new AgentExecutionError.
        
        Args:
            agent_name: The name of the agent that failed
            message: The error message
            original_error: The original exception that caused the failure
        """
        self.agent_name = agent_name
        self.original_error = original_error
        self.message = message
        super().__init__(f"Agent {agent_name} failed: {message}")


class FallbackAgent(AgentBase):
    """
    A fallback agent that is used when no suitable agent is available.
    
    This agent implements a minimal "do nothing" behavior that simply returns
    the input state unchanged. It can be used as a last resort when no other
    agent is available or when a required capability is not available.
    """
    
    def __init__(self, name: str = "FallbackAgent"):
        """
        Initialize a new FallbackAgent.
        
        Args:
            name: The name of the fallback agent
        """
        super().__init__(name, AgentType.ORCHESTRATION, True, {})
    
    async def process(self, state: SimulationState) -> SimulationState:
        """
        Process the current simulation state (no-op).
        
        This implementation simply returns the input state unchanged.
        
        Args:
            state: The current simulation state
            
        Returns:
            The unchanged simulation state
        """
        return state
    
    def get_capabilities(self) -> Set[AgentCapability]:
        """
        Return an empty set of capabilities.
        
        This agent does not provide any capabilities.
        
        Returns:
            An empty set
        """
        return set()


# TODO: Add method for agent metrics collection
# TODO: Implement agent discovery and registration mechanism
# TODO: Add support for agent dependencies and ordering
# TODO: Create specialized agent base classes for different agent types

# Summary:
# This module defines the base classes and interfaces for agents in the
# CortexFlow system. It includes an abstract base class that all agents
# must inherit from, defining the common interface for processing simulation
# state and reporting capabilities. It also includes a FallbackAgent
# implementation that provides a "do nothing" behavior when no suitable
# agent is available.
