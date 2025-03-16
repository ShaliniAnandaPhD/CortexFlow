"""
Command-line interface for the CortexFlow system.

This module provides a command-line interface for interacting with the
CortexFlow system, allowing users to run simulations, visualize results,
and manage agents.
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional

import click

from cortexflow.core.simulation import Simulation
from cortexflow.core.types import StressLevel, TaskType, InterventionType
from cortexflow.agents.base import AgentBase
from cortexflow.agents.e2b_agent import E2BAgent
from cortexflow.agents.langgraph_agent import LangGraphAgent

# Try importing optional agent implementations
# These will be None if the dependencies are not installed
try:
    from cortexflow.agents.autogen_agent import AutoGenAgent
except ImportError:
    AutoGenAgent = None

try:
    from cortexflow.agents.crewai_agent import CrewAIAgent
except ImportError:
    CrewAIAgent = None

try:
    from cortexflow.agents.camel_agent import CAMELAgent
except ImportError:
    CAMELAgent = None

try:
    from cortexflow.agents.haystack_agent import HaystackAgent
except ImportError:
    HaystackAgent = None

try:
    from cortexflow.agents.llamaindex_agent import LlamaIndexAgent
except ImportError:
    LlamaIndexAgent = None

try:
    from cortexflow.agents.opendevin_agent import OpenDevinAgent
except ImportError:
    OpenDevinAgent = None


def initialize_agents() -> Dict[str, AgentBase]:
    """
    Initialize all available agents.
    
    This function tries to initialize all the agent implementations that have
    been imported successfully. If an agent's dependencies are not installed,
    that agent will not be included.
    
    Returns:
        A dictionary of agent name to agent instance
    """
    agents = {}
    
    # Initialize the core agents
    agents["e2b"] = E2BAgent()
    agents["langgraph"] = LangGraphAgent()
    
    # Initialize optional agents if available
    if AutoGenAgent:
        agents["autogen"] = AutoGenAgent()
    
    if CrewAIAgent:
        agents["crewai"] = CrewAIAgent()
    
    if CAMELAgent:
        agents["camel"] = CAMELAgent()
    
    if HaystackAgent:
        agents["haystack"] = HaystackAgent()
    
    if LlamaIndexAgent:
        agents["llamaindex"] = LlamaIndexAgent()
    
    if OpenDevinAgent:
        agents["opendevin"] = OpenDevinAgent()
    
    return agents


def print_simulation_status(simulation: Simulation) -> None:
    """
    Print the current status of a simulation.
    
    Args:
        simulation: The simulation to print status for
    """
    state = simulation.state
    
    # Print the current simulation state
    click.echo("\nCurrent Simulation State:")
    click.echo(f"Step: {state.simulation_step}")
    click.echo(f"Stress Level: {state.stress_level}")
    click.echo(f"Task Type: {state.task_type}")
    click.echo(f"Intervention: {state.intervention}")
    click.echo(f"Cortisol Level: {state.cortisol_level:.1f}")
    click.echo(f"Dopamine Level: {state.dopamine_level:.1f}")
    click.echo(f"Productivity Score: {state.productivity_score:.1f}")
    click.echo(f"Memory Efficiency: {state.memory_efficiency:.1f}")
    click.echo(f"Decision Quality: {state.decision_quality:.1f}")


@click.group()
def cli():
    """
    CortexFlow: Multi-Agent AI System for Modeling Cognitive Processes.
    
    This CLI allows you to run simulations, visualize results, and manage agents.
    """
    pass


@cli.command()
@click.option(
    "--stress",
    type=click.Choice(["mild", "moderate", "severe"]),
    default="moderate",
    help="Stress level to simulate."
)
@click.option(
    "--task",
    type=click.Choice(["creative", "analytical", "physical", "repetitive"]),
    default="creative",
    help="Task type to simulate."
)
@click.option(
    "--intervention",
    type=click.Choice(["none", "meditation", "micro_breaks", "biofeedback"]),
    default="none",
    help="Intervention strategy to apply."
)
@click.option(
    "--steps",
    type=int,
    default=10,
    help="Number of simulation steps to run."
)
@click.option(
    "--output",
    type=str,
    default="simulation_results.json",
    help="Output file for simulation results."
)
def run(stress, task, intervention, steps, output):
    """
    Run a stress-productivity simulation.
    """
    click.echo(f"Running simulation with stress={stress}, task={task}, "
               f"intervention={intervention}, steps={steps}")
    
    # Initialize agents
    agents = initialize_agents()
    available_agents = {name: agent for name, agent in agents.items() if agent.is_available()}
    
    if not available_agents:
        click.echo("Error: No agents available. Please install at least one agent package.")
        sys.exit(1)
    
    click.echo(f"Initialized {len(available_agents)} agents: {', '.join(available_agents.keys())}")
    
    # Create and configure the simulation
    simulation = Simulation(agents=available_agents)
    simulation.configure(
        stress_level=stress,
        task_type=task,
        intervention=intervention,
        steps=steps
    )
    
    # Run the simulation
    click.echo("\nRunning Simulation...")
    
    with click.progressbar(range(steps), label="Simulation Progress") as progress_bar:
        for _ in progress_bar:
            # Run a single step
            asyncio.run(simulation.run_step())
            
            # Print status every few steps
            if simulation.state.simulation_step % 3 == 0:
                print_simulation_status(simulation)
    
    # Print final status
    click.echo("\nSimulation Complete!")
    print_simulation_status(simulation)
    
    # Export results
    output_file = simulation.export_results(output)
    click.echo(f"\nResults saved to {output_file}")
    click.echo("You can now visualize these results with the 'cortexflow visualize' command.")


@cli.command()
@click.argument(
    "results_file",
    type=click.Path(exists=True, readable=True)
)
@click.option(
    "--output",
    type=str,
    default="simulation_visualization.html",
    help="Output file for visualization."
)
def visualize(results_file, output):
    """
    Visualize simulation results from a JSON file.
    """
    click.echo(f"Visualizing results from {results_file}")
    
    # Load the results file
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        click.echo(f"Error loading results file: {e}")
        sys.exit(1)
    
    # Verify the results file has the expected structure
    if "history" not in results:
        click.echo("Error: Invalid results file. Missing 'history' key.")
        sys.exit(1)
    
    # Create a basic HTML visualization
    # In a real implementation, this would use proper visualization libraries
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CortexFlow Simulation Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            h1, h2 { color: #4472C4; }
            .container { max-width: 1000px; margin: 0 auto; }
            .chart { background: #f9f9f9; padding: 20px; margin: 20px 0; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CortexFlow Simulation Results</h1>
            
            <h2>Simulation Configuration</h2>
            <table>
                <tr><th>Setting</th><th>Value</th></tr>
    """
    
    # Add configuration settings
    metadata = results.get("simulation_metadata", {})
    config = metadata.get("config", {})
    for key, value in config.items():
        html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
    
    html += """
            </table>
            
            <h2>Simulation History</h2>
            <table>
                <tr>
                    <th>Step</th>
                    <th>Cortisol</th>
                    <th>Dopamine</th>
                    <th>Productivity</th>
                    <th>Memory</th>
                    <th>Decision</th>
                </tr>
    """
    
    # Add history rows
    for entry in results["history"]:
        html += f"""
            <tr>
                <td>{entry.get('step', 'N/A')}</td>
                <td>{entry.get('cortisol_level', 'N/A'):.1f}</td>
                <td>{entry.get('dopamine_level', 'N/A'):.1f}</td>
                <td>{entry.get('productivity_score', 'N/A'):.1f}</td>
                <td>{entry.get('memory_efficiency', 'N/A'):.1f}</td>
                <td>{entry.get('decision_quality', 'N/A'):.1f}</td>
            </tr>
        """
    
    html += """
            </table>
            
            <h2>Visualization</h2>
            <p>For more detailed visualizations, use the Google Colab notebook.</p>
            <p>Instructions: Open the Colab notebook and upload this JSON file.</p>
            
            <div class="chart">
                <p>Chart placeholder - In a real implementation, this would include interactive visualizations.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    try:
        with open(output, 'w') as f:
            f.write(html)
    except Exception as e:
        click.echo(f"Error writing visualization file: {e}")
        sys.exit(1)
    
    click.echo(f"Visualization saved to {output}")
    click.echo("Open this file in a web browser to view the results.")


@cli.command()
def agents():
    """
    List available agents and their status.
    """
    click.echo("Available Agents:")
    
    # Initialize agents
    agents = initialize_agents()
    
    # Print agent status
    for name, agent in agents.items():
        status = "available" if agent.is_available() else "unavailable"
        agent_type = agent.agent_type
        capabilities = ", ".join(str(c) for c in agent.get_capabilities())
        
        click.echo(f"- {name} ({agent_type}): {status}")
        click.echo(f"  Capabilities: {capabilities}")


def main():
    """
    Main entry point for the CLI.
    """
    cli()


if __name__ == "__main__":
    main()


# TODO: Add commands for agent management (enable/disable)
# TODO: Implement more sophisticated visualization options
# TODO: Add support for configuration files
# TODO: Implement commands for managing simulation presets

# Summary:
# This module implements the command-line interface for the CortexFlow system.
# It provides commands for running simulations, visualizing results, and
# managing agents. The CLI uses the Click library for command-line parsing
# and provides a user-friendly interface for interacting with the system.

  
