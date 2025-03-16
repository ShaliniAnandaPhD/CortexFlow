"""
Main simulation engine for the CortexFlow system.

This module contains the Simulation class, which orchestrates the execution
of agents to simulate stress and productivity interactions. It handles agent
initialization, execution, and result aggregation.
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set

from cortexflow.core.state import SimulationState
from cortexflow.core.types import (
    StressLevel,
    TaskType,
    InterventionType,
    AgentCapability,
    AgentName
)
from cortexflow.agents.base import AgentBase, AgentExecutionError, FallbackAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Simulation:
    """
    Main simulation engine for the CortexFlow system.
    
    This class orchestrates the execution of agents to simulate stress and
    productivity interactions. It handles agent initialization, execution,
    and result aggregation.
    
    Attributes:
        agents: Dictionary of available agents by name
        state: Current simulation state
        config: Simulation configuration
        capability_index: Index of capabilities to agent names
    """
    
    def __init__(
        self,
        agents: Optional[Dict[AgentName, AgentBase]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new simulation.
        
        Args:
            agents: Dictionary of available agents by name
            config: Simulation configuration
        """
        self.agents = agents or {}
        self.config = config or {}
        self.state = SimulationState()
        
        # Build capability index for agent lookup by capability
        self.capability_index = self._build_capability_index()
        
        # Create fallback agent for when no suitable agent is available
        self.fallback_agent = FallbackAgent()
        
        logger.info(f"Initialized simulation with {len(self.agents)} agents")
    
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
    
    def configure(
        self,
        stress_level: str = StressLevel.MODERATE,
        task_type: str = TaskType.CREATIVE,
        intervention: str = InterventionType.NONE,
        steps: int = 10
    ) -> None:
        """
        Configure the simulation parameters.
        
        Args:
            stress_level: Stress level to simulate
            task_type: Type of task to simulate
            intervention: Intervention strategy to apply
            steps: Number of simulation steps to run
        """
        # Initialize a new state with the specified parameters
        self.state = SimulationState(
            stress_level=stress_level,
            task_type=task_type,
            intervention=intervention
        )
        
        # Store the number of steps in the configuration
        self.config["steps"] = steps
        
        logger.info(
            f"Configured simulation: stress={stress_level}, "
            f"task={task_type}, intervention={intervention}, steps={steps}"
        )
    
    async def run_step(self) -> SimulationState:
        """
        Run a single step of the simulation.
        
        This method executes all available agents in sequence, passing the
        state from one agent to the next.
        
        Returns:
            The updated simulation state after all agents have been executed
        """
        logger.info(f"Running simulation step {self.state.simulation_step}")
        
        # Create the execution plan
        execution_plan = self._create_execution_plan()
        
        # Execute agents in sequence
        current_state = self.state
        
        for agent_name in execution_plan:
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
        
        # Update the simulation state
        self.state = current_state
        
        # Record state history
        self.state.update_history()
        
        return self.state
    
    def _create_execution_plan(self) -> List[AgentName]:
        """
        Create a plan for executing agents.
        
        This method creates an ordered list of agent names to execute based
        on the current simulation state and the available agents.
        
        Returns:
            A list of agent names in execution order
        """
        # For now, we'll use a simple fixed order based on the layer architecture
        # In a real implementation, this would be more dynamic based on the state
        
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
            # In a real implementation, this would use a more sophisticated
            # selection mechanism based on the current state and agent performance
            execution_plan.append(agent_names[0])
        
        return execution_plan
    
    async def run_simulation(self, steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the full simulation for a specified number of steps.
        
        Args:
            steps: Number of steps to run, or None to use the configured value
            
        Returns:
            A dictionary containing the final state and simulation history
        """
        # Use the specified steps or the configured value
        steps_to_run = steps or self.config.get("steps", 10)
        
        logger.info(f"Running simulation for {steps_to_run} steps")
        
        start_time = time.time()
        
        # Run the specified number of steps
        for _ in range(steps_to_run):
            await self.run_step()
            
            # Add a small delay between steps for logging visibility
            await asyncio.sleep(0.1)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Simulation completed in {duration:.2f} seconds")
        
        # Prepare the results
        results = {
            "final_state": self.state.to_dict(),
            "history": self.state.history,
            "simulation_metadata": {
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration,
                "config": {
                    "stress_level": self.state.stress_level,
                    "task_type": self.state.task_type,
                    "intervention": self.state.intervention,
                    "simulation_steps": len(self.state.history),
                    "agent_count": len(self.agents)
                },
                "agents_used": list(self.agents.keys())
            }
        }
        
        return results
    
    def export_results(self, filename: str = "simulation_results.json") -> str:
        """
        Export simulation results to a JSON file.
        
        Args:
            filename: Name of the output file
            
        Returns:
            The absolute path to the output file
        """
        # Create the results dictionary
        results = {
            "final_state": self.state.to_dict(),
            "history": self.state.history,
            "simulation_metadata": {
                "config": {
                    "stress_level": self.state.stress_level,
                    "task_type": self.state.task_type,
                    "intervention": self.state.intervention,
                    "simulation_steps": len(self.state.history),
                    "agent_count": len(self.agents)
                },
                "agents_used": list(self.agents.keys())
            }
        }
        
        # Write the results to a file
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Get the absolute path to the file
        abs_path = os.path.abspath(filename)
        
        logger.info(f"Results exported to {abs_path}")
        
        return abs_path


# TODO: Implement parallel agent execution
# TODO: Add support for dynamic agent loading
# TODO: Implement more sophisticated execution planning
# TODO: Add support for simulation checkpoints and resumption

# Summary:
# This module implements the Simulation class, which orchestrates the execution
# of agents to simulate stress and productivity interactions. It handles agent
# initialization, execution, and result aggregation. The class provides methods
# for configuring and running the simulation, as well as exporting the results
# to a JSON file.
