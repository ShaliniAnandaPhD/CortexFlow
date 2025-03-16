"""
Visualization utilities for the CortexFlow system.

This module provides functions for visualizing simulation results,
including plots of metrics over time, network diagrams of agent
interactions, and comparative visualizations.
"""

import os
import json
import tempfile
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

from cortexflow.core.types import AgentType


def plot_metrics_over_time(
    history: List[Dict[str, Any]],
    output_file: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot cognitive metrics over time.
    
    Args:
        history: List of simulation state snapshots
        output_file: Optional file path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Convert history to DataFrame
    df = pd.DataFrame(history)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot metrics
    ax.plot(df['step'], df['productivity_score'], 'r-', label='Productivity')
    ax.plot(df['step'], df['memory_efficiency'], 'g-', label='Memory')
    ax.plot(df['step'], df['decision_quality'], 'b-', label='Decision')
    
    # Add stress backdrop
    ax.fill_between(
        df['step'],
        0,
        df['cortisol_level'],
        alpha=0.2,
        color='gray',
        label='Stress Level'
    )
    
    # Mark interventions
    if 'intervention' in df.columns:
        intervention_points = df[df['intervention'] != 'none']
        
        if not intervention_points.empty:
            for idx, row in intervention_points.iterrows():
                ax.axvline(x=row['step'], color='k', linestyle='--', alpha=0.5)
                ax.text(
                    row['step'] + 0.1,
                    10,
                    row['intervention'],
                    rotation=90,
                    verticalalignment='bottom'
                )
    
    # Add labels and legend
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Score / Level')
    ax.set_title('Cognitive Metrics Over Time')
    ax.legend()
    ax.grid(True)
    
    # Set y-axis limits
    ax.set_ylim(0, 105)
    
    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_neurochemical_levels(
    history: List[Dict[str, Any]],
    output_file: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot neurochemical levels over time.
    
    Args:
        history: List of simulation state snapshots
        output_file: Optional file path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Convert history to DataFrame
    df = pd.DataFrame(history)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot hormone levels
    ax.plot(df['step'], df['cortisol_level'], 'orange', label='Cortisol')
    ax.plot(df['step'], df['dopamine_level'], 'purple', label='Dopamine')
    
    # Calculate cortisol-to-dopamine ratio
    df['c_d_ratio'] = df['cortisol_level'] / df['dopamine_level']
    ax2 = ax.twinx()
    ax2.plot(df['step'], df['c_d_ratio'], 'g--', label='Cortisol/Dopamine Ratio')
    
    # Mark interventions
    if 'intervention' in df.columns:
        intervention_points = df[df['intervention'] != 'none']
        
        if not intervention_points.empty:
            for idx, row in intervention_points.iterrows():
                ax.axvline(x=row['step'], color='k', linestyle='--', alpha=0.5)
                ax.text(
                    row['step'] + 0.1,
                    10,
                    row['intervention'],
                    rotation=90,
                    verticalalignment='bottom'
                )
    
    # Add labels and legends
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Hormone Level')
    ax2.set_ylabel('Cortisol/Dopamine Ratio')
    ax.set_title('Neurochemical Levels Over Time')
    
    # Create combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Set y-axis limits
    ax.set_ylim(0, 105)
    
    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig


def create_agent_network(
    agent_interactions: Dict[str, Dict[str, float]],
    agent_types: Dict[str, str],
    output_file: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create a network diagram of agent interactions.
    
    Args:
        agent_interactions: Dictionary of agent interactions
        agent_types: Dictionary mapping agent names to types
        output_file: Optional file path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes with types
    for agent, agent_type in agent_types.items():
        G.add_node(agent, type=agent_type)
    
    # Add edges with weights
    for source, targets in agent_interactions.items():
        for target, weight in targets.items():
            G.add_edge(source, target, weight=weight)
    
    # Set up colors based on agent types
    colors = {
        AgentType.ORCHESTRATION: 'lightblue',
        AgentType.PROCESSING: 'lightgreen',
        AgentType.KNOWLEDGE: 'orange',
        AgentType.OPTIMIZATION: 'purple'
    }
    
    # Set default color for unknown types
    default_color = 'gray'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get node colors based on agent types
    node_colors = [colors.get(agent_types.get(node, ''), default_color) for node in G.nodes()]
    
    # Get edge weights
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    
    # Set up node positions using a spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(
        G, pos, width=edge_weights, alpha=0.6, edge_color='gray',
        arrowstyle='->', arrowsize=15
    )
    
    # Add legend for agent types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                  markersize=10, label=agent_type)
        for agent_type, color in colors.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set title
    ax.set_title('Agent Interaction Network')
    
    # Remove axis
    ax.axis('off')
    
    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_intervention_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'productivity_score',
    output_file: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot a comparison of different interventions.
    
    Args:
        results: Dictionary mapping intervention names to result dictionaries
        metric: The metric to compare
        output_file: Optional file path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors for different interventions
    colors = {
        'none': 'gray',
        'meditation': 'blue',
        'micro_breaks': 'green',
        'biofeedback': 'purple'
    }
    
    # Plot each intervention
    for intervention, result in results.items():
        history = result.get('history', [])
        if not history:
            continue
        
        df = pd.DataFrame(history)
        if metric not in df.columns:
            continue
        
        color = colors.get(intervention, 'black')
        ax.plot(df['step'], df[metric], color=color, label=intervention.capitalize())
    
    # Add labels and legend
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Intervention Comparison: {metric.replace("_", " ").title()}')
    ax.legend()
    ax.grid(True)
    
    # Set y-axis limits for common metrics
    if metric in ['productivity_score', 'memory_efficiency', 'decision_quality']:
        ax.set_ylim(0, 105)
    
    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig


def create_heatmap(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    value_column: str,
    title: str,
    output_file: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create a heatmap visualization.
    
    Args:
        data: DataFrame containing the data
        x_column: Column to use for the x-axis
        y_column: Column to use for the y-axis
        value_column: Column containing the values to plot
        title: Plot title
        output_file: Optional file path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Pivot the data for the heatmap
    pivot_table = data.pivot(index=y_column, columns=x_column, values=value_column)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom colormap (blue gradient)
    colors = sns.color_palette("Blues", 256)
    cmap = LinearSegmentedColormap.from_list("custom_blues", colors)
    
    # Create heatmap
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        linewidths=0.5,
        ax=ax
    )
    
    # Set title and labels
    ax.set_title(title)
    
    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig


def create_interactive_visualization(
    results: Dict[str, Any],
    output_file: str
) -> str:
    """
    Create an interactive HTML visualization.
    
    This function generates an interactive HTML file with plots
    that can be viewed in a web browser.
    
    Args:
        results: The simulation results
        output_file: File path to save the HTML
        
    Returns:
        The absolute path to the output file
    """
    # Extract history from results
    history = results.get('history', [])
    
    if not history:
        raise ValueError("No history data found in results")
    
    # Convert history to DataFrame
    df = pd.DataFrame(history)
    
    # Generate Plotly figures using a temporary Python file
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
        f.write("""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# Load the data
df = pd.DataFrame({})

# Create figure with secondary y-axis
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=('Cognitive Metrics', 'Neurochemical Levels')
)

# Add cognitive metrics traces
fig.add_trace(
    go.Scatter(x=df['step'], y=df['productivity_score'], name='Productivity', line=dict(color='red')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df['step'], y=df['memory_efficiency'], name='Memory', line=dict(color='blue')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df['step'], y=df['decision_quality'], name='Decision', line=dict(color='green')),
    row=1, col=1
)

# Add neurochemical traces
fig.add_trace(
    go.Scatter(x=df['step'], y=df['cortisol_level'], name='Cortisol', line=dict(color='orange')),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=df['step'], y=df['dopamine_level'], name='Dopamine', line=dict(color='purple')),
    row=2, col=1
)

# Add intervention markers if present
if 'intervention' in df.columns:
    interventions = df[df['intervention'] != 'none']
    for idx, row in interventions.iterrows():
        # Add a vertical line for each intervention
        fig.add_vline(
            x=row['step'],
            line_dash='dash',
            line_width=1,
            line_color='black',
            opacity=0.5,
            row=1, col=1
        )
        fig.add_vline(
            x=row['step'],
            line_dash='dash',
            line_width=1,
            line_color='black',
            opacity=0.5,
            row=2, col=1
        )
        
        # Add annotation for intervention type
        fig.add_annotation(
            x=row['step'],
            y=100,
            text=row['intervention'],
            showarrow=True,
            arrowhead=1,
            row=1, col=1
        )

# Update layout
fig.update_layout(
    title='CortexFlow Simulation Results',
    height=800,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )
)

# Update y-axes ranges
fig.update_yaxes(range=[0, 100], title_text='Score', row=1, col=1)
fig.update_yaxes(range=[0, 100], title_text='Level', row=2, col=1)
fig.update_xaxes(title_text='Simulation Step', row=2, col=1)

# Create table for metadata
metadata = {}
table_fig = go.Figure(data=[
    go.Table(
        header=dict(values=['Setting', 'Value'],
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=[list(metadata.keys()), list(metadata.values())],
                  fill_color='lavender',
                  align='left'))
])
table_fig.update_layout(title='Simulation Configuration')

# Write output to HTML file
with open('{}', 'w') as f:
    f.write('<html><head><title>CortexFlow Simulation Results</title>')
    f.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')
    f.write('</head><body>')
    f.write('<div id="plot"></div>')
    f.write('<div id="table"></div>')
    f.write('<script>')
    f.write('var plotDiv = document.getElementById("plot");')
    f.write('var tableDiv = document.getElementById("table");')
    f.write('Plotly.newPlot(plotDiv, ' + json.dumps(fig.to_dict()) + ');')
    f.write('Plotly.newPlot(tableDiv, ' + json.dumps(table_fig.to_dict()) + ');')
    f.write('</script>')
    f.write('</body></html>')
""".format(df.to_dict(), output_file))
    
    try:
        # Execute the temporary Python script
        tmp_script = f.name
        os.system(f"python {tmp_script}")
        
        # Clean up
        os.unlink(tmp_script)
        
        return os.path.abspath(output_file)
    except Exception as e:
        raise ValueError(f"Failed to create interactive visualization: {e}")


def plot_stress_intervention_effect(
    history: List[Dict[str, Any]],
    output_file: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot the relationship between stress and intervention effectiveness.
    
    Args:
        history: List of simulation state snapshots
        output_file: Optional file path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Convert history to DataFrame
    df = pd.DataFrame(history)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define stress bands
    df['stress_band'] = pd.cut(
        df['cortisol_level'],
        bins=[0, 40, 70, 100],
        labels=['Low', 'Medium', 'High']
    )
    
    # Calculate productivity changes
    df['productivity_change'] = df['productivity_score'].diff()
    
    # Group by intervention and stress band
    grouped = df.groupby(['intervention', 'stress_band'])['productivity_change'].mean().reset_index()
    
    # Filter to only include actual interventions
    grouped = grouped[grouped['intervention'] != 'none']
    
    # Create the grouped bar chart
    sns.barplot(
        x='intervention',
        y='productivity_change',
        hue='stress_band',
        data=grouped,
        ax=ax
    )
    
    # Add labels and legend
    ax.set_xlabel('Intervention Type')
    ax.set_ylabel('Average Productivity Change')
    ax.set_title('Intervention Effectiveness by Stress Level')
    
    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig


def create_effectiveness_matrix(
    results: Dict[str, Dict[str, Any]],
    output_file: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create a matrix visualization of intervention effectiveness.
    
    Args:
        results: Dictionary mapping intervention names to result dictionaries
        output_file: Optional file path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Prepare data structure for the matrix
    interventions = list(results.keys())
    metrics = ['productivity_score', 'memory_efficiency', 'decision_quality']
    
    # Calculate effectiveness for each combination
    effectiveness = {}
    
    for intervention in interventions:
        result = results[intervention]
        history = result.get('history', [])
        
        if not history:
            continue
        
        df = pd.DataFrame(history)
        
        # Calculate percentage change from start to end
        effectiveness[intervention] = {}
        for metric in metrics:
            if metric in df.columns:
                start_value = df[metric].iloc[0]
                end_value = df[metric].iloc[-1]
                pct_change = ((end_value - start_value) / start_value) * 100 if start_value else 0
                effectiveness[intervention][metric] = pct_change
    
    # Create DataFrame for the heatmap
    matrix_data = []
    for intervention, metrics_dict in effectiveness.items():
        for metric, value in metrics_dict.items():
            matrix_data.append({
                'Intervention': intervention,
                'Metric': metric.replace('_', ' ').title(),
                'Effectiveness': value
            })
    
    matrix_df = pd.DataFrame(matrix_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Create heatmap
    pivot_table = matrix_df.pivot(
        index='Intervention',
        columns='Metric',
        values='Effectiveness'
    )
    
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        ax=ax
    )
    
    # Set title
    ax.set_title('Intervention Effectiveness Matrix (% Change)')
    
    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig


# TODO: Implement interactive web visualization
# TODO: Add 3D visualization of stress-productivity relationship
# TODO: Implement agent interaction network animation
# TODO: Add support for real-time visualization during simulation

# Summary:
# This module provides functions for visualizing simulation results
# from the CortexFlow system. It includes various plots such as
# metric trends over time, neurochemical level charts, agent interaction
# networks, and comparative visualizations of different interventions.
# The module supports both static and interactive visualizations.
