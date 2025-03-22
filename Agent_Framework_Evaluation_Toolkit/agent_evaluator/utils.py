import numpy as np
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Callable

# Accuracy evaluation functions
def calculate_accuracy_score(actual, expected) -> float:
    """Calculate accuracy between actual and expected outputs"""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if expected == 0:
            return 1.0 if actual == 0 else 0.0
        diff_pct = abs(actual - expected) / abs(expected)
        return max(0.0, 1.0 - diff_pct)
    elif isinstance(expected, str) and isinstance(actual, str):
        # For string comparison, we can use exact matching or implement more sophisticated
        # NLP-based similarity metrics for domain-specific evaluations
        return 1.0 if expected.strip() == actual.strip() else 0.0
    elif isinstance(expected, dict) and isinstance(actual, dict):
        # For dictionaries, we recursively calculate accuracy for each key
        # Missing keys are penalized
        scores = []
        for k, v in expected.items():
            if k in actual:
                scores.append(calculate_accuracy_score(actual.get(k), v))
            else:
                scores.append(0.0)
        return np.mean(scores) if scores else 0.0
    elif isinstance(expected, list) and isinstance(actual, list):
        # For lists, calculate set overlap as a rough measure
        # This could be enhanced with order-sensitive metrics if needed
        if not expected:
            return 1.0 if not actual else 0.0
        # Calculate Jaccard similarity for list comparison
        if len(expected) == 0 and len(actual) == 0:
            return 1.0
        intersection = len(set(expected) & set(actual))
        union = len(set(expected) | set(actual))
        return intersection / union
    else:
        # Fallback to direct equality check
        return 1.0 if expected == actual else 0.0

def calculate_domain_specific_accuracy(
    actual: Dict[str, Any], 
    expected: Dict[str, Any], 
    domain: str
) -> float:
    """Calculate domain-specific accuracy with specialized metrics"""
    if domain == "neurochemical_interaction":
        # For neurochemical models, we care about relative concentrations and patterns
        # rather than exact values
        neurochem_score = 0.0
        if "concentrations" in expected and "concentrations" in actual:
            # Check if relative proportions are preserved
            exp_conc = expected["concentrations"]
            act_conc = actual["concentrations"]
            
            # Normalize concentrations
            exp_total = sum(exp_conc.values())
            act_total = sum(act_conc.values())
            
            if exp_total > 0 and act_total > 0:
                exp_norm = {k: v/exp_total for k, v in exp_conc.items()}
                act_norm = {k: v/act_total for k, v in act_conc.items()}
                
                # Calculate difference in proportions
                chemicals = set(exp_norm.keys()) | set(act_norm.keys())
                diffs = []
                for chem in chemicals:
                    exp_val = exp_norm.get(chem, 0)
                    act_val = act_norm.get(chem, 0)
                    diffs.append(abs(exp_val - act_val))
                
                if diffs:
                    neurochem_score = 1.0 - (sum(diffs) / len(diffs))
        
        # We might also check for correct identification of patterns or thresholds
        pattern_score = 0.0
        if "patterns" in expected and "patterns" in actual:
            exp_patterns = set(expected["patterns"])
            act_patterns = set(actual["patterns"])
            if exp_patterns:
                pattern_score = len(exp_patterns & act_patterns) / len(exp_patterns)
        
        # Combine scores with domain-specific weighting
        return 0.7 * neurochem_score + 0.3 * pattern_score
    
    elif domain == "stress_modeling":
        # For stress modeling, we focus on correctly identifying stress triggers,
        # levels, and interventions
        trigger_score = 0.0
        if "triggers" in expected and "triggers" in actual:
            exp_triggers = set(expected["triggers"])
            act_triggers = set(actual["triggers"])
            if exp_triggers:
                trigger_score = len(exp_triggers & act_triggers) / len(exp_triggers)
        
        level_score = 0.0
        if "stress_level" in expected and "stress_level" in actual:
            exp_level = expected["stress_level"]
            act_level = actual["stress_level"]
            if isinstance(exp_level, (int, float)) and isinstance(act_level, (int, float)):
                # Assuming stress levels are on the same scale
                max_diff = 10.0  # Maximum possible difference on scale
                level_score = 1.0 - (abs(exp_level - act_level) / max_diff)
        
        intervention_score = 0.0
        if "recommended_interventions" in expected and "recommended_interventions" in actual:
            exp_interventions = set(expected["recommended_interventions"])
            act_interventions = set(actual["recommended_interventions"])
            if exp_interventions:
                intervention_score = len(exp_interventions & act_interventions) / len(exp_interventions)
        
        # Combine with domain-specific weighting
        return 0.3 * trigger_score + 0.3 * level_score + 0.4 * intervention_score
    
    # Add more domain-specific accuracy calculations as needed
    
    # Fallback to general accuracy calculation
    return calculate_accuracy_score(actual, expected)

# Multi-agent and composability analysis functions
def measure_inter_agent_communication(
    communication_log: List[Dict[str, Any]]
) -> Tuple[float, float, int]:
    """
    Analyze inter-agent communication patterns
    
    Args:
        communication_log: List of communication events with timestamps and sizes
        
    Returns:
        Tuple of (average_latency, total_overhead, message_count)
    """
    if not communication_log:
        return 0.0, 0.0, 0
    
    latencies = []
    total_overhead = 0
    
    for msg in communication_log:
        if "latency_ms" in msg:
            latencies.append(msg["latency_ms"])
        if "payload_size_bytes" in msg:
            total_overhead += msg["payload_size_bytes"]
    
    avg_latency = np.mean(latencies) if latencies else 0.0
    return avg_latency, total_overhead, len(communication_log)

def analyze_state_serialization(
    serialization_events: List[Dict[str, Any]]
) -> Tuple[float, float]:
    """
    Analyze state serialization overhead
    
    Args:
        serialization_events: List of serialization/deserialization events
        
    Returns:
        Tuple of (avg_serialization_time, avg_overhead_bytes)
    """
    if not serialization_events:
        return 0.0, 0.0
    
    times = []
    overheads = []
    
    for event in serialization_events:
        if "duration_ms" in event:
            times.append(event["duration_ms"])
        if "size_bytes" in event:
            overheads.append(event["size_bytes"])
    
    avg_time = np.mean(times) if times else 0.0
    avg_overhead = np.mean(overheads) if overheads else 0.0
    
    return avg_time, avg_overhead

def measure_adapter_overhead(
    adapter_events: List[Dict[str, Any]]
) -> Tuple[float, float, float]:
    """
    Measure overhead introduced by framework adapters
    
    Args:
        adapter_events: List of adapter invocation events
        
    Returns:
        Tuple of (avg_conversion_time, avg_execution_time, avg_framework_switching_latency)
    """
    if not adapter_events:
        return 0.0, 0.0, 0.0
    
    conversion_times = []
    execution_times = []
    switching_latencies = []
    
    for event in adapter_events:
        if "conversion_time_ms" in event:
            conversion_times.append(event["conversion_time_ms"])
        if "execution_time_ms" in event:
            execution_times.append(event["execution_time_ms"])
        if "framework_switching_latency_ms" in event:
            switching_latencies.append(event["framework_switching_latency_ms"])
    
    avg_conversion = np.mean(conversion_times) if conversion_times else 0.0
    avg_execution = np.mean(execution_times) if execution_times else 0.0
    avg_switching = np.mean(switching_latencies) if switching_latencies else 0.0
    
    return avg_conversion, avg_execution, avg_switching

def analyze_accuracy_preservation(
    baseline_accuracy: Dict[str, float],
    current_accuracy: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate how much of the original accuracy is preserved
    
    Args:
        baseline_accuracy: Accuracy measurements from baseline (e.g., multi-framework)
        current_accuracy: Accuracy measurements from current architecture
        
    Returns:
        Dictionary mapping categories to preservation percentages
    """
    preservation = {}
    
    for category, baseline in baseline_accuracy.items():
        if category in current_accuracy and baseline > 0:
            preservation[category] = (current_accuracy[category] / baseline) * 100.0
        else:
            preservation[category] = 0.0
    
    # Add average preservation
    if preservation:
        preservation["overall"] = np.mean(list(preservation.values()))
    else:
        preservation["overall"] = 0.0
        
    return preservation

# Performance measurement decorators
def time_execution(func: Callable) -> Callable:
    """Decorator to measure execution time of a function"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        if isinstance(result, tuple):
            return (*result, execution_time)
        else:
            return result, execution_time
    return wrapper

# JSON serialization utilities for complex objects
def serialize_complex_state(state: Dict[str, Any]) -> Tuple[str, int]:
    """
    Serialize a complex state object to JSON, measuring overhead
    
    Args:
        state: Complex state object to serialize
        
    Returns:
        Tuple of (serialized_state, size_in_bytes)
    """
    start_time = time.time()
    serialized = json.dumps(state)
    end_time = time.time()
    
    serialization_time = (end_time - start_time) * 1000  # Convert to milliseconds
    size_bytes = len(serialized.encode('utf-8'))
    
    return serialized, size_bytes, serialization_time

def deserialize_complex_state(serialized_state: str) -> Tuple[Dict[str, Any], float]:
    """
    Deserialize a JSON state string, measuring overhead
    
    Args:
        serialized_state: JSON string to deserialize
        
    Returns:
        Tuple of (deserialized_state, deserialization_time_ms)
    """
    start_time = time.time()
    state = json.loads(serialized_state)
    end_time = time.time()
    
    deserialization_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    return state, deserialization_time
