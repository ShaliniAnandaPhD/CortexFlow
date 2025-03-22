import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple

def plot_bar_chart(
    title: str, 
    labels: List[str], 
    values: List[float],
    ylabel: str, 
    output_path: str, 
    threshold: Optional[float] = None,
    color: str = 'skyblue',
    sort_values: bool = False
):
    """Create a basic bar chart"""
    plt.figure(figsize=(12, 6))
    
    if sort_values:
        # Sort by values
        sorted_data = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
        labels, values = zip(*sorted_data)
    
    plt.bar(labels, values, color=color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
        plt.legend()
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_multi_architecture_comparison(
    architecture_results: Dict[str, Dict[str, float]],
    metric: str,
    output_path: str,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    threshold: Optional[float] = None,
    sort_by: Optional[str] = None
):
    """
    Compare multiple architectures across the same metric
    
    Args:
        architecture_results: Dict mapping architecture names to their metric results
        metric: Name of the metric being compared
        output_path: Path to save the visualization
        title: Optional title, defaults to metric name
        ylabel: Optional y-axis label, defaults to metric
        threshold: Optional threshold line
        sort_by: Optional architecture name to sort values by
    """
    plt.figure(figsize=(14, 8))
    
    # Prepare data
    architectures = list(architecture_results.keys())
    
    # Get all unique labels across all architectures
    all_labels = set()
    for arch_data in architecture_results.values():
        all_labels.update(arch_data.keys())
    all_labels = list(all_labels)
    
    # Sort if requested
    if sort_by and sort_by in architecture_results:
        # Get values for the architecture we're sorting by
        sort_values = []
        for label in all_labels:
            sort_values.append(architecture_results[sort_by].get(label, 0))
        
        # Sort labels by these values
        sorted_data = sorted(zip(all_labels, sort_values), key=lambda x: x[1], reverse=True)
        all_labels = [item[0] for item in sorted_data]
    
    # Number of architectures and width of bars
    n_architectures = len(architectures)
    width = 0.8 / n_architectures
    
    # Plot each architecture's data
    for i, arch in enumerate(architectures):
        # Calculate x positions for this architecture's bars
        x = np.arange(len(all_labels))
        offset = i - (n_architectures - 1) / 2
        x_pos = x + offset * width
        
        # Get values, using 0 for missing data
        values = [architecture_results[arch].get(label, 0) for label in all_labels]
        
        plt.bar(x_pos, values, width=width, label=arch)
    
    # Set labels and title
    plt.xlabel('Categories')
    plt.ylabel(ylabel if ylabel else metric)
    plt.title(title if title else f'Comparison of {metric} Across Architectures')
    plt.xticks(np.arange(len(all_labels)), all_labels, rotation=45, ha='right')
    plt.legend()
    
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
        plt.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_radar_chart(
    architecture_metrics: Dict[str, Dict[str, float]],
    metric_categories: List[str],
    output_path: str,
    title: str = "Architecture Comparison"
):
    """
    Create a radar chart comparing multiple architectures across metrics
    
    Args:
        architecture_metrics: Dict mapping architecture names to metric values
        metric_categories: List of metric categories to include
        output_path: Path to save the visualization
        title: Title for the chart
    """
    # Set up the figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of metrics
    N = len(metric_categories)
    
    # Compute angle for each metric
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Make the plot circular
    angles += angles[:1]
    metric_categories += metric_categories[:1]
    
    # Plot each architecture
    for arch_name, metrics in architecture_metrics.items():
        # Extract values for the specified metric categories
        values = [metrics.get(cat, 0) for cat in metric_categories[:-1]]
        # Make the plot circular
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, linewidth=2, label=arch_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], metric_categories[:-1])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_accuracy_preservation_heatmap(
    baseline_accuracies: Dict[str, float],
    architecture_accuracies: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Accuracy Preservation by Architecture"
):
    """
    Create a heatmap showing accuracy preservation across architectures
    
    Args:
        baseline_accuracies: Dict mapping categories to baseline accuracy values
        architecture_accuracies: Dict mapping architecture names to dicts of category accuracies
        output_path: Path to save the visualization
        title: Title for the heatmap
    """
    # Prepare data for heatmap
    categories = list(baseline_accuracies.keys())
    architectures = list(architecture_accuracies.keys())
    
    # Calculate preservation percentages
    data = []
    for arch in architectures:
        row = []
        for category in categories:
            baseline = baseline_accuracies.get(category, 0)
            if baseline > 0:
                current = architecture_accuracies[arch].get(category, 0)
                preservation = (current / baseline) * 100.0
            else:
                preservation = 0
            row.append(preservation)
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data, index=architectures, columns=categories)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".1f", cbar_kws={'label': 'Accuracy Preservation %'})
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_performance_vs_accuracy(
    architecture_metrics: Dict[str, Dict[str, float]],
    performance_metric: str,
    accuracy_metric: str,
    output_path: str,
    title: Optional[str] = None,
    performance_label: Optional[str] = None,
    accuracy_label: Optional[str] = None,
    invert_performance: bool = True  # Set True if lower values are better (like execution time)
):
    """
    Create a scatter plot of performance vs accuracy trade-offs
    
    Args:
        architecture_metrics: Dict mapping architecture names to metric values
        performance_metric: Key for the performance metric
        accuracy_metric: Key for the accuracy metric
        output_path: Path to save the visualization
        title: Optional title
        performance_label: Optional x-axis label
        accuracy_label: Optional y-axis label
        invert_performance: If True, invert performance values (for metrics where lower is better)
    """
    plt.figure(figsize=(10, 8))
    
    # Extract data
    architectures = []
    performance_values = []
    accuracy_values = []
    
    for arch, metrics in architecture_metrics.items():
        if performance_metric in metrics and accuracy_metric in metrics:
            architectures.append(arch)
            perf_value = metrics[performance_metric]
            # Invert if needed (e.g., for execution time, lower is better)
            if invert_performance:
                # Find max to normalize
                max_val = max(metrics[performance_metric] for metrics in architecture_metrics.values())
                perf_value = max_val - perf_value
            performance_values.append(perf_value)
            accuracy_values.append(metrics[accuracy_metric])
    
    # Create scatter plot
    plt.scatter(performance_values, accuracy_values, s=100)
    
    # Label each point
    for i, arch in enumerate(architectures):
        plt.annotate(arch, (performance_values[i], accuracy_values[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Set labels and title
    plt.xlabel(performance_label if performance_label else performance_metric)
    plt.ylabel(accuracy_label if accuracy_label else accuracy_metric)
    plt.title(title if title else f'{accuracy_metric} vs {performance_metric} Trade-offs')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_timeline_visualization(
    event_log: List[Dict[str, Any]],
    output_path: str,
    title: str = "Agent Execution Timeline"
):
    """
    Create a visualization of agent execution timeline
    
    Args:
        event_log: List of events with timestamps, durations, and agent info
        output_path: Path to save the visualization
        title: Title for the chart
    """
    # Extract agents and events
    agents = set()
    for event in event_log:
        if "agent" in event:
            agents.add(event["agent"])
    
    # Sort agents for consistent ordering
    agents = sorted(list(agents))
    
    # Create timeline data
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Track y-position for each agent
    y_positions = {agent: i for i, agent in enumerate(agents)}
    
    # Track colors for event types
    event_types = set(event.get("event_type", "unknown") for event in event_log)
    colors = plt.cm.tab10(np.linspace(0, 1, len(event_types)))
    event_colors = {event_type: color for event_type, color in zip(event_types, colors)}
    
    # Plot events
    for event in event_log:
        if "agent" in event and "start_time" in event and "duration" in event:
            agent = event["agent"]
            start = event["start_time"]
            duration = event["duration"]
            event_type = event.get("event_type", "unknown")
            
            y_pos = y_positions[agent]
            ax.barh(y_pos, duration, left=start, height=0.5, 
                   color=event_colors[event_type], alpha=0.8)
            
            # Add label for longer events
            if duration > (max(e["start_time"] + e["duration"] for e in event_log if "start_time" in e and "duration" in e) * 0.05):
                ax.text(start + duration/2, y_pos, event_type, 
                       ha='center', va='center', color='black', fontsize=8)
    
    # Set y-ticks to agent names
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    
    # Add legend for event types
    handles = [plt.Rectangle((0,0), 1, 1, color=event_colors[event_type]) 
              for event_type in event_colors]
    ax.legend(handles, list(event_colors.keys()), title="Event Types", 
             loc='upper right')
    
    plt.title(title)
    plt.xlabel('Time (ms)')
    plt.ylabel('Agent')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
