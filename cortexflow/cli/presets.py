"""
Simulation presets for the CortexFlow system.

This module provides predefined simulation scenarios that can be used
as starting points for experiments. Each preset includes specific
stress levels, task types, interventions, and duration.
"""

import os
import json
from typing import Dict, Any, List, Optional

import click

from cortexflow.core.types import StressLevel, TaskType, InterventionType


# Predefined simulation presets
PRESETS = {
    "creative_stress": {
        "name": "Creative Work Under Stress",
        "description": "Simulates the effects of moderate stress on creative work",
        "parameters": {
            "stress_level": StressLevel.MODERATE,
            "task_type": TaskType.CREATIVE,
            "intervention": InterventionType.NONE,
            "steps": 10
        }
    },
    "analytical_pressure": {
        "name": "Analytical Tasks Under Pressure",
        "description": "Simulates analytical problem-solving under severe stress",
        "parameters": {
            "stress_level": StressLevel.SEVERE,
            "task_type": TaskType.ANALYTICAL,
            "intervention": InterventionType.NONE,
            "steps": 12
        }
    },
    "meditation_recovery": {
        "name": "Meditation Recovery",
        "description": "Tests the effectiveness of meditation for stress recovery",
        "parameters": {
            "stress_level": StressLevel.MODERATE,
            "task_type": TaskType.CREATIVE,
            "intervention": InterventionType.MEDITATION,
            "steps": 15
        }
    },
    "breaks_repetitive": {
        "name": "Micro-Breaks for Repetitive Tasks",
        "description": "Evaluates micro-breaks during repetitive work",
        "parameters": {
            "stress_level": StressLevel.MILD,
            "task_type": TaskType.REPETITIVE,
            "intervention": InterventionType.MICRO_BREAKS,
            "steps": 12
        }
    },
    "biofeedback_high_stress": {
        "name": "Biofeedback for High Stress",
        "description": "Tests biofeedback intervention during high-stress analytical work",
        "parameters": {
            "stress_level": StressLevel.SEVERE,
            "task_type": TaskType.ANALYTICAL,
            "intervention": InterventionType.BIOFEEDBACK,
            "steps": 15
        }
    },
    "physical_labor": {
        "name": "Physical Task Simulation",
        "description": "Simulates stress effects during physical labor",
        "parameters": {
            "stress_level": StressLevel.MODERATE,
            "task_type": TaskType.PHYSICAL,
            "intervention": InterventionType.NONE,
            "steps": 12
        }
    },
    "chronic_stress": {
        "name": "Chronic Stress Pattern",
        "description": "Simulates long-term elevated stress without intervention",
        "parameters": {
            "stress_level": StressLevel.MODERATE,
            "task_type": TaskType.ANALYTICAL,
            "intervention": InterventionType.NONE,
            "steps": 20
        }
    },
    "intervention_comparison": {
        "name": "Intervention Comparison",
        "description": "Runs multiple simulations to compare different interventions",
        "parameters": {
            "stress_level": StressLevel.MODERATE,
            "task_type": TaskType.CREATIVE,
            "interventions": [
                InterventionType.NONE,
                InterventionType.MEDITATION,
                InterventionType.MICRO_BREAKS,
                InterventionType.BIOFEEDBACK
            ],
            "steps": 12
        },
        "multi_run": True
    }
}


def list_presets() -> List[Dict[str, Any]]:
    """
    Get a list of all available presets.
    
    Returns:
        A list of preset information dictionaries
    """
    return [
        {
            "id": preset_id,
            "name": preset["name"],
            "description": preset["description"]
        }
        for preset_id, preset in PRESETS.items()
    ]


def get_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific preset by ID.
    
    Args:
        preset_id: The ID of the preset to retrieve
        
    Returns:
        The preset dictionary, or None if not found
    """
    return PRESETS.get(preset_id)


def save_preset(preset_data: Dict[str, Any], preset_id: str) -> str:
    """
    Save a custom preset to the user's configuration directory.
    
    Args:
        preset_data: The preset data to save
        preset_id: The ID for the new preset
        
    Returns:
        The path to the saved preset file
        
    Raises:
        click.FileError: If the preset cannot be saved
    """
    # Get the user's configuration directory
    config_dir = os.path.expanduser("~/.config/cortexflow")
    presets_dir = os.path.join(config_dir, "presets")
    
    # Create directories if they don't exist
    os.makedirs(presets_dir, exist_ok=True)
    
    # Create the preset file path
    preset_file = os.path.join(presets_dir, f"{preset_id}.json")
    
    # Save the preset
    try:
        with open(preset_file, 'w') as f:
            json.dump(preset_data, f, indent=2)
        return preset_file
    except Exception as e:
        raise click.FileError(
            preset_file, f"Could not save preset: {e}"
        )


def load_custom_presets() -> Dict[str, Dict[str, Any]]:
    """
    Load all custom presets from the user's configuration directory.
    
    Returns:
        A dictionary of custom presets
    """
    # Get the user's configuration directory
    config_dir = os.path.expanduser("~/.config/cortexflow")
    presets_dir = os.path.join(config_dir, "presets")
    
    # Check if the directory exists
    if not os.path.exists(presets_dir):
        return {}
    
    # Load all preset files
    custom_presets = {}
    for filename in os.listdir(presets_dir):
        if filename.endswith(".json"):
            preset_id = filename[:-5]  # Remove .json extension
            preset_file = os.path.join(presets_dir, filename)
            
            try:
                with open(preset_file, 'r') as f:
                    preset_data = json.load(f)
                    custom_presets[preset_id] = preset_data
            except Exception as e:
                # Log the error but continue loading other presets
                click.echo(f"Warning: Could not load preset '{preset_id}': {e}", err=True)
    
    return custom_presets


def get_all_presets() -> Dict[str, Dict[str, Any]]:
    """
    Get all available presets, including custom presets.
    
    Returns:
        A dictionary of all presets
    """
    # Start with the built-in presets
    all_presets = PRESETS.copy()
    
    # Add custom presets
    custom_presets = load_custom_presets()
    
    # Handle potential ID conflicts
    for preset_id, preset_data in custom_presets.items():
        if preset_id in all_presets:
            # Add a suffix to the custom preset ID to avoid conflict
            custom_id = f"{preset_id}_custom"
            all_presets[custom_id] = preset_data
        else:
            all_presets[preset_id] = preset_data
    
    return all_presets


def delete_custom_preset(preset_id: str) -> bool:
    """
    Delete a custom preset.
    
    Args:
        preset_id: The ID of the preset to delete
        
    Returns:
        True if the preset was deleted, False otherwise
    """
    # Get the user's configuration directory
    config_dir = os.path.expanduser("~/.config/cortexflow")
    presets_dir = os.path.join(config_dir, "presets")
    
    # Create the preset file path
    preset_file = os.path.join(presets_dir, f"{preset_id}.json")
    
    # Check if the file exists
    if not os.path.exists(preset_file):
        return False
    
    # Delete the file
    try:
        os.remove(preset_file)
        return True
    except Exception:
        return False


def validate_preset_id(ctx, param, value) -> str:
    """
    Validate that a preset ID exists.
    
    Args:
        ctx: Click context
        param: Click parameter
        value: The value to validate
        
    Returns:
        The validated value
        
    Raises:
        click.BadParameter: If the value is not valid
    """
    all_presets = get_all_presets()
    if value not in all_presets:
        preset_ids = ", ".join(all_presets.keys())
        raise click.BadParameter(
            f"Invalid preset ID. Valid options are: {preset_ids}"
        )
    return value


# Example usage in a Click command:
#
# @cli.command()
# @click.argument("preset", callback=validate_preset_id)
# def run_preset(preset):
#     """Run a simulation using a predefined preset."""
#     preset_data = get_preset(preset)
#     if not preset_data:
#         preset_data = load_custom_presets().get(preset)
#     
#     if not preset_data:
#         click.echo(f"Error: Preset '{preset}' not found.")
#         return
#     
#     click.echo(f"Running preset: {preset_data['name']}")
#     click.echo(f"Description: {preset_data['description']}")
#     
#     # Extract parameters
#     params = preset_data.get("parameters", {})
#     
#     # Run the simulation with these parameters
#     # ...


# TODO: Add support for multi-simulation presets
# TODO: Implement preset comparison functionality
# TODO: Add support for preset modification
# TODO: Implement preset sharing functionality

# Summary:
# This module provides predefined simulation scenarios (presets) for the
# CortexFlow system. It includes various combinations of stress levels,
# task types, interventions, and durations that can be used as starting
# points for experiments. The module also supports loading, saving, and
# managing custom presets created by users.
