"""
Agent orchestration logic for the CortexFlow system.

This module provides the Orchestrator class, which is responsible for
coordinating the execution of agents in the simulation. It handles agent
discovery, execution planning, and state transitions.
"""

import asyncio
import logging
from typing import Dict, List, Set, Any, Optional, Tuple

from cortexflow.core.state import SimulationState
from cortexflow.core.types import AgentCapability, AgentType, AgentName
from cortexflow.agents.base import AgentBase, AgentExecutionError, FallbackAgent

# Configure logging
logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrator for coordinating agent execution in the simulation.
    
    The Orchestrator is responsible for planning and executing agent workflows,
    handling agent discovery, execution ordering, and state transitions.
    
    Attributes:
        agents: Dictionary of available agents by name
        execution_plan: List of agent names in execution order
        fallback_agent: Fallback agent for when no suitable agent is available
        capability_index: Index of agent names by capability
    """
    
    def __init__(
        self,
        agents: Optional[Dict[AgentName, AgentBase]] = None,
        execution_plan: Optional[List[AgentName]] = None
    ):
        """
        Initialize a new orchestrator.
        
        Args:
            agents: Dictionary of available agents by name
            execution_plan: Optional predefined execution plan
        """
        self.agents = agents or {}
        self.execution_plan = execution_plan or []
        self.fallback_agent = FallbackAgent()
        
        # Build capability index for agent lookup by capability
        self.capability_index = self._build_capability_index()
        
        # If no execution plan was provided, create a default one
        if not self.execution_plan:
            self.execution_plan = self._create_default_execution_plan()
        
        logger.info(f"Initialized orchestrator with {len(self.agents)} agents")
    
    def _build_capability_index(self) -> Dict[AgentCapability, List[AgentName]]:
        """
        Build an index of capabilities to agent names.
        
        This index allows for efficient lookup of agents by capability.
        
        Returns:
            A dictionary mapping capabilities to lists of agent names
        """
        index: Dict[AgentCapability, List[AgentName]] = {}
        
        for name, agent in self.agents.items():
            if not agent.is_available():
                continue
                
            for capability in agent.get_capabilities():
                if capability not in index:
                    index[capability] = []
                index[capability].append(name)
        
        return index
    
    def _create_default_execution_plan(self) -> List[AgentName]:
        """
        Create a default execution plan based on agent capabilities.
        
        This method creates an execution plan that follows the layered
        architecture of the system, with orchestration agents first, followed
        by processing, knowledge, and optimization agents.
        
        Returns:
            A list of agent names in execution order
        """
        # Define the execution order by layer and capability
        execution_order = [
            # Orchestration layer
            AgentCapability.STATE_MANAGEMENT,
            AgentCapability.RUNTIME_ENVIRONMENT,
            
            # Processing layer
            AgentCapability.NEUROCHEMICAL_DIALOGUE,
            AgentCapability.MEMORY_ANALYSIS,
            AgentCapability.STRESS_DIALOGUE,
            
            # Knowledge layer
            AgentCapability.INTERVENTION_RECOMMENDATION,
            AgentCapability.INTERVENTION_ANALYSIS,
            
            # Optimization layer
            AgentCapability.WORKFLOW_OPTIMIZATION
        ]
        
        # Create the execution plan
        execution_plan = []
        
        for capability in execution_order:
            agent_names = self.capability_index.get(capability, [])
            
            if not agent_names:
                logger.warning(f"No agent available for capability: {capability}")
                continue
            
            # For now, just take the first available agent for each capability
            # In a more sophisticated implementation, this would consider agent
            # performance, dependencies, and other factors
            execution_plan.append(agent_names[0])
        
        return execution_plan
    
    def update_execution_plan(self, state: SimulationState) -> None:
        """
        Update the execution plan based on the current simulation state.
        
        This method can be used to dynamically adjust the execution plan
        based on the current state of the simulation.
        
        Args:
            state: The current simulation state
        """
        # In a more sophisticated implementation, this would analyze the
        # current state and adjust the execution plan accordingly
        # For now, we'll just use the default execution plan
        pass
    
    async def execute_agents(self, state: SimulationState) -> SimulationState:
        """
        Execute agents according to the execution plan.
        
        This method executes each agent in the execution plan, passing
        the state from one agent to the next.
        
        Args:
            state: The initial simulation state
            
        Returns:
            The updated simulation state after all agents have been executed
        """
        current_state = state
        
        for agent_name in self.execution_plan:
            agent = self.agents.get(agent_name)
            
            if not agent or not agent.is_available():
                logger.warning(f"Agent {agent_name} is not available, skipping")
                continue
            
            try:
                logger.info(f"Executing agent: {agent_name}")
                current_state = await agent.process(current_state)
            except Exception as e:
                logger.error(f"Agent {agent_name} execution failed: {e}")
                # Continue with the current state if an agent fails
        
        return current_state
    
    async def execute_capability(
        self,
        capability: AgentCapability,
        state: SimulationState
    ) -> Tuple[SimulationState, bool]:
        """
        Execute an agent with a specific capability.
        
        This method finds an agent with the specified capability and
        executes it on the current state.
        
        Args:
            capability: The capability to execute
            state: The current simulation state
            
        Returns:
            A tuple of (updated state, success flag)
        """
        agent_names = self.capability_index.get(capability, [])
        
        if not agent_names:
            logger.warning(f"No agent available for capability: {capability}")
            return state, False
        
        # For now, just use the first available agent
        agent_name = agent_names[0]
        agent = self.agents.get(agent_name)
        
        if not agent or not agent.is_available():
            logger.warning(f"Agent {agent_name} is not available")
            return state, False
        
        try:
            logger.info(f"Executing capability {capability} with agent {agent_name}")
            new_state = await agent.process(state)
            return new_state, True
        except Exception as e:
            logger.error(f"Agent {agent_name} execution failed: {e}")
            return state, False
    
    async def execute_workflow(
        self,
        workflow: List[AgentCapability],
        state: SimulationState
    ) -> SimulationState:
        """
        Execute a workflow of capabilities.
        
        This method executes a sequence of capabilities, passing the
        state from one to the next.
        
        Args:
            workflow: List of capabilities to execute in order
            state: The initial simulation state
            
        Returns:
            The updated simulation state after the workflow has been executed
        """
        current_state = state
        
        for capability in workflow:
            new_state, success = await self.execute_capability(capability, current_state)
            current_state = new_state
            
            if not success:
                logger.warning(f"Workflow step {capability} failed")
        
        return current_state
    
    def create_intervention_workflow(
        self,
        state: SimulationState
    ) -> List[AgentCapability]:
        """
        Create a workflow for intervention recommendation and application.
        
        This method creates a workflow that recommends and applies an
        intervention based on the current simulation state.
        
        Args:
            state: The current simulation state
            
        Returns:
            A list of capabilities to execute in order
        """
        # Basic workflow: recommend an intervention, then analyze its effectiveness
        workflow = [
            AgentCapability.INTERVENTION_RECOMMENDATION,
            AgentCapability.INTERVENTION_ANALYSIS
        ]
        
        return workflow
    
    def create_stress_analysis_workflow(
        self,
        state: SimulationState
    ) -> List[AgentCapability]:
        """
        Create a workflow for stress analysis.
        
        This method creates a workflow that analyzes the effects of stress
        on productivity, memory, and decision-making.
        
        Args:
            state: The current simulation state
            
        Returns:
            A list of capabilities to execute in order
        """
        # Basic workflow: simulate stress effects, analyze memory impact,
        # simulate dialogue, and optimize workflow
        workflow = [
            AgentCapability.RUNTIME_ENVIRONMENT,
            AgentCapability.NEUROCHEMICAL_DIALOGUE,
            AgentCapability.MEMORY_ANALYSIS,
            AgentCapability.WORKFLOW_OPTIMIZATION
        ]
        
        return workflow


# TODO: Implement parallel agent execution
# TODO: Add support for conditional workflows
# TODO: Implement workflow templates for common scenarios
# TODO: Add support for workflow visualization

# Summary:
# This module provides the Orchestrator class, which coordinates the execution
# of agents in the simulation. It handles agent discovery, execution planning,
# and state transitions. The class provides methods for executing individual
# agents, capabilities, and workflows, as well as for creating workflows for
# common scenarios like intervention recommendation and stress analysis.
