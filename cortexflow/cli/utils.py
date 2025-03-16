"""
CLI utility functions for the CortexFlow system.

This module provides utility functions for the command-line interface,
including formatting, visualization, and configuration handling.
"""

import os
import json
import sys
from typing import Dict, List, Any, Optional, Tuple
import datetime

import click
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from cortexflow.core.state import SimulationState
from cortexflow.core.types import StressLevel, TaskType, InterventionType


def load_results(results_file: str) -> Dict[str, Any]:
    """
    Load simulation results from a JSON file.
    
    Args:
        results_file: Path to the results file
        
    Returns:
        The loaded results dictionary
        
    Raises:
        click.FileError: If the file cannot be loaded
    """
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        raise click.FileError(
            results_file, f"Could not load results file: {e}"
        )


def save_config(config: Dict[str, Any], filename: str) -> str:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        filename: Name of the output file
        
    Returns:
        The absolute path to the saved file
        
    Raises:
        click.FileError: If the file cannot be saved
    """
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        return os.path.abspath(filename)
    except Exception as e:
        raise click.FileError(
            filename, f"Could not save configuration file: {e}"
        )


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        The loaded configuration dictionary
        
    Raises:
        click.FileError: If the file cannot be loaded
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise click.FileError(
            config_file, f"Could not load configuration file: {e}"
        )


def create_visualization(
    results: Dict[str, Any],
    output_file: str,
    format_type: str = "html"
) -> str:
    """
    Create a visualization of simulation results.
    
    Args:
        results: The simulation results
        output_file: Name of the output file
        format_type: Type of visualization (html, png, pdf)
        
    Returns:
        The absolute path to the created visualization
        
    Raises:
        click.UsageError: If the format type is not supported
    """
    if format_type == "html":
        return create_html_visualization(results, output_file)
    elif format_type == "png" or format_type == "pdf":
        return create_matplotlib_visualization(results, output_file, format_type)
    else:
        raise click.UsageError(f"Unsupported visualization format: {format_type}")


def create_html_visualization(results: Dict[str, Any], output_file: str) -> str:
    """
    Create an HTML visualization of simulation results.
    
    Args:
        results: The simulation results
        output_file: Name of the output file
        
    Returns:
        The absolute path to the created HTML file
    """
    # Ensure the output file has the correct extension
    if not output_file.endswith(".html"):
        output_file += ".html"
    
    # Extract history from results
    history = results.get("history", [])
    
    if not history:
        raise click.UsageError("No history data found in results")
    
    # Create HTML content
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
            .chart-container { height: 400px; }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            
            <h2>Cognitive Metrics Over Time</h2>
            <div class="chart-container">
                <canvas id="metricsChart"></canvas>
            </div>
            
            <h2>Neurochemical Levels Over Time</h2>
            <div class="chart-container">
                <canvas id="chemicalChart"></canvas>
            </div>
            
            <h2>Simulation History</h2>
            <table>
                <tr>
                    <th>Step</th>
                    <th>Cortisol</th>
                    <th>Dopamine</th>
                    <th>Productivity</th>
                    <th>Memory</th>
                    <th>Decision</th>
                    <th>Intervention</th>
                </tr>
    """
    
    # Add history rows
    for entry in history:
        html += f"""
            <tr>
                <td>{entry.get('step', 'N/A')}</td>
                <td>{entry.get('cortisol_level', 'N/A'):.1f}</td>
                <td>{entry.get('dopamine_level', 'N/A'):.1f}</td>
                <td>{entry.get('productivity_score', 'N/A'):.1f}</td>
                <td>{entry.get('memory_efficiency', 'N/A'):.1f}</td>
                <td>{entry.get('decision_quality', 'N/A'):.1f}</td>
                <td>{entry.get('intervention', 'none')}</td>
            </tr>
        """
    
    # Extract data for charts
    steps = [entry.get('step', i) for i, entry in enumerate(history)]
    productivity = [entry.get('productivity_score', 0) for entry in history]
    memory = [entry.get('memory_efficiency', 0) for entry in history]
    decision = [entry.get('decision_quality', 0) for entry in history]
    cortisol = [entry.get('cortisol_level', 0) for entry in history]
    dopamine = [entry.get('dopamine_level', 0) for entry in history]
    
    # Add chart initialization
    html += f"""
            </table>
        </div>
        
        <script>
            // Cognitive metrics chart
            const metricsCtx = document.getElementById('metricsChart').getContext('2d');
            const metricsChart = new Chart(metricsCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(steps)},
                    datasets: [
                        {{
                            label: 'Productivity',
                            data: {json.dumps(productivity)},
                            borderColor: 'rgb(255, 99, 132)',
                            tension: 0.1
                        }},
                        {{
                            label: 'Memory',
                            data: {json.dumps(memory)},
                            borderColor: 'rgb(54, 162, 235)',
                            tension: 0.1
                        }},
                        {{
                            label: 'Decision',
                            data: {json.dumps(decision)},
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            min: 0,
                            max: 100,
                            title: {{
                                display: true,
                                text: 'Score'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Simulation Step'
                            }}
                        }}
                    }}
                }}
            }});
            
            // Neurochemical levels chart
            const chemicalCtx = document.getElementById('chemicalChart').getContext('2d');
            const chemicalChart = new Chart(chemicalCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(steps)},
                    datasets: [
                        {{
                            label: 'Cortisol',
                            data: {json.dumps(cortisol)},
                            borderColor: 'rgb(255, 159, 64)',
                            tension: 0.1
                        }},
                        {{
                            label: 'Dopamine',
                            data: {json.dumps(dopamine)},
                            borderColor: 'rgb(153, 102, 255)',
                            tension: 0.1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            min: 0,
                            max: 100,
                            title: {{
                                display: true,
                                text: 'Level'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Simulation Step'
                            }}
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    try:
        with open(output_file, 'w') as f:
            f.write(html)
    except Exception as e:
        raise click.FileError(
            output_file, f"Could not save visualization file: {e}"
        )
    
    return os.path.abspath(output_file)


def create_matplotlib_visualization(
    results: Dict[str, Any],
    output_file: str,
    format_type: str
) -> str:
    """
    Create a Matplotlib visualization of simulation results.
    
    Args:
        results: The simulation results
        output_file: Name of the output file
        format_type: Type of visualization (png, pdf)
        
    Returns:
        The absolute path to the created image file
    """
    # Ensure the output file has the correct extension
    if not output_file.endswith(f".{format_type}"):
        output_file += f".{format_type}"
    
    # Extract history from results
    history = results.get("history", [])
    
    if not history:
        raise click.UsageError("No history data found in results")
    
    # Convert history to DataFrame
    df = pd.DataFrame(history)
    
    # Set up the figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('CortexFlow Simulation Results', fontsize=16)
    
    # Plot cognitive metrics
    axs[0].set_title('Cognitive Metrics Over Time')
    axs[0].plot(df['step'], df['productivity_score'], 'r-', label='Productivity')
    axs[0].plot(df['step'], df['memory_efficiency'], 'b-', label='Memory')
    axs[0].plot(df['step'], df['decision_quality'], 'g-', label='Decision')
    axs[0].set_ylabel('Score')
    axs[0].set_ylim(0, 100)
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot neurochemical levels
    axs[1].set_title('Neurochemical Levels Over Time')
    axs[1].plot(df['step'], df['cortisol_level'], 'orange', label='Cortisol')
    axs[1].plot(df['step'], df['dopamine_level'], 'purple', label='Dopamine')
    axs[1].set_xlabel('Simulation Step')
    axs[1].set_ylabel('Level')
    axs[1].set_ylim(0, 100)
    axs[1].legend()
    axs[1].grid(True)
    
    # Add intervention markers if present
    if 'intervention' in df.columns:
        interventions = df[df['intervention'] != 'none']
        if not interventions.empty:
            for idx, row in interventions.iterrows():
                axs[0].axvline(x=row['step'], color='k', linestyle='--', alpha=0.5)
                axs[1].axvline(x=row['step'], color='k', linestyle='--', alpha=0.5)
                axs[0].annotate(
                    row['intervention'],
                    xy=(row['step'], 95),
                    xytext=(row['step'] + 0.1, 95),
                    rotation=90,
                    fontsize=8
                )
    
    # Add configuration information
    metadata = results.get("simulation_metadata", {})
    config = metadata.get("config", {})
    config_text = "\n".join([f"{k}: {v}" for k, v in config.items()])
    fig.text(0.02, 0.02, f"Configuration:\n{config_text}", fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    try:
        plt.savefig(output_file, format=format_type, dpi=300, bbox_inches='tight')
    except Exception as e:
        raise click.FileError(
            output_file, f"Could not save visualization file: {e}"
        )
    
    return os.path.abspath(output_file)


def format_table(data: List[Dict[str, Any]], headers: List[str]) -> str:
    """
    Format data as a text table.
    
    Args:
        data: List of dictionaries containing the data
        headers: List of headers for the table
        
    Returns:
        The formatted table as a string
    """
    if not data:
        return "No data available"
    
    # Determine column widths
    widths = [len(h) for h in headers]
    for row in data:
        for i, header in enumerate(headers):
            if header in row:
                widths[i] = max(widths[i], len(str(row[header])))
    
    # Create the header row
    header_row = " | ".join(f"{h:{w}s}" for h, w in zip(headers, widths))
    separator = "-+-".join("-" * w for w in widths)
    
    # Create the data rows
    data_rows = []
    for row in data:
        values = []
        for i, header in enumerate(headers):
            value = row.get(header, "")
            values.append(f"{value:{widths[i]}s}")
        data_rows.append(" | ".join(values))
    
    # Combine all rows
    return "\n".join([header_row, separator] + data_rows)


def get_timestamp() -> str:
    """
    Get a formatted timestamp for file naming.
    
    Returns:
        A formatted timestamp string
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_stress_level(ctx, param, value) -> str:
    """
    Validate that a stress level value is valid.
    
    Args:
        ctx: Click context
        param: Click parameter
        value: The value to validate
        
    Returns:
        The validated value
        
    Raises:
        click.BadParameter: If the value is not valid
    """
    valid_levels = [level.value for level in StressLevel]
    if value not in valid_levels:
        raise click.BadParameter(
            f"Invalid stress level. Valid options are: {', '.join(valid_levels)}"
        )
    return value


def validate_task_type(ctx, param, value) -> str:
    """
    Validate that a task type value is valid.
    
    Args:
        ctx: Click context
        param: Click parameter
        value: The value to validate
        
    Returns:
        The validated value
        
    Raises:
        click.BadParameter: If the value is not valid
    """
    valid_types = [task_type.value for task_type in TaskType]
    if value not in valid_types:
        raise click.BadParameter(
            f"Invalid task type. Valid options are: {', '.join(valid_types)}"
        )
    return value


def validate_intervention(ctx, param, value) -> str:
    """
    Validate that an intervention value is valid.
    
    Args:
        ctx: Click context
        param: Click parameter
        value: The value to validate
        
    Returns:
        The validated value
        
    Raises:
        click.BadParameter: If the value is not valid
    """
    valid_interventions = [intervention.value for intervention in InterventionType]
    if value not in valid_interventions:
        raise click.BadParameter(
            f"Invalid intervention. Valid options are: {', '.join(valid_interventions)}"
        )
    return value


# TODO: Add more visualization types
# TODO: Implement interactive visualizations
# TODO: Add support for simulation comparison
# TODO: Add support for custom visualization templates

# Summary:
# This module provides utility functions for the CortexFlow CLI,
# including loading and saving configuration and results files,
# creating visualizations in various formats, and formatting data
# for display in the terminal. It also includes validation functions
# for command-line parameters.
