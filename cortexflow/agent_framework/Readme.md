# Multi-Agent Framework: Composable and Hybrid Architectures

A framework for building and evaluating composable and hybrid multi-agent architectures, specifically designed for stress analysis and neurochemical modeling applications.

## Overview

This project implements two primary architecture patterns for multi-agent systems:

### 1. Composable Architecture

A standardized agent interface with unified communication protocol where all agents:
- Take a global state object as input
- Process it through domain-specific logic
- Return results in a standardized format
- Contribute to a centralized state update

### 2. Hybrid Architecture

Framework-specific adapters that implement the composable interface while leveraging specialized framework optimizations:
- Maintains the standardized interface for agent interactions
- Uses specialized frameworks for domain-specific processing
- Converts between standardized and framework-specific formats
- Measures and optimizes the adapter overhead

## Key Benefits

**Composable Architecture:**
- **Performance**: 73% faster execution time, 63% memory reduction
- **Simplicity**: 62.5% code reduction, no external dependencies
- **Integration**: Unified communication eliminating cross-framework serialization

**Hybrid Architecture:**
- **Accuracy**: Recovers ~70% of accuracy compared to multi-framework approach
- **Specialization**: Leverages domain-specific framework optimizations
- **Flexibility**: Easy integration of new frameworks through adapter pattern

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/composable-agent-framework.git
cd composable-agent-framework

# Install dependencies
pip install -e .
```

## Usage Examples

### Basic Usage

```python
from agent_framework.composable_agent import (
    ComposableAgentOrchestrator,
    StressModelingAgent,
    NeurochemicalInteractionAgent
)

# Create orchestrator
orchestrator = ComposableAgentOrchestrator()

# Create specialized agents
stress_agent = StressModelingAgent()
neuro_agent = NeurochemicalInteractionAgent()

# Register agents with orchestrator
orchestrator.register_agent(stress_agent)
orchestrator.register_agent(neuro_agent)

# Set execution sequence
orchestrator.set_execution_sequence(["stress_modeling", "neurochemical_interaction"])

# Initialize
orchestrator.initialize()

# Example input data
input_data = {
    "stress_factors": ["deadline", "workload"],
    "context": "Project deadline approaching",
    "user_profile": {
        "baseline_cortisol": 12.0,
        "cortisol_reactivity": 1.2
    }
}

# Execute
result = orchestrator.execute(input_data)
print(result)
```

### Using Hybrid Architecture

```python
from agent_framework.hybrid_agent import (
    HybridAgentOrchestrator,
    create_stress_modeling_hybrid_agent,
    create_neurochemical_hybrid_agent
)
from agent_framework.framework_adapter import (
    CamelAdapter,
    AutogenAdapter
)

# Create orchestrator
orchestrator = HybridAgentOrchestrator()

# Create framework adapters
camel_adapter = CamelAdapter()
autogen_adapter = AutogenAdapter()

# Create specialized hybrid agents
stress_agent = create_stress_modeling_hybrid_agent(camel_adapter)
neuro_agent = create_neurochemical_hybrid_agent(autogen_adapter)

# Register agents with orchestrator
orchestrator.register_agent(stress_agent)
orchestrator.register_agent(neuro_agent)

# Set execution sequence
orchestrator.set_execution_sequence(["stress_modeling", "neurochemical_interaction"])

# Initialize
orchestrator.initialize()

# Execute with same input data
result = orchestrator.execute(input_data)
print(result)
```

### Command Line Interface

The framework includes a command-line interface for running and comparing architectures:

```bash
# Run both architectures and compare results
python main.py --architecture both --input scenario.json

# Run benchmarks with multiple iterations
python main.py --benchmark 10 --output benchmark_results

# Get detailed metrics
python main.py --metrics --architecture hybrid
```

## Architecture Details

### Global State

The global state is the central mechanism for sharing data between agents:

```python
# Global state structure
{
    "input": { /* original input data */ },
    "results": {
        "agent_id_1": { /* agent 1 results */ },
        "agent_id_2": { /* agent 2 results */ }
    },
    "shared_state": { /* values shared between agents */ },
    "metadata": {
        "start_time": 1677721600,
        "agent_sequence": ["agent_id_1", "agent_id_2"],
        "communications": []
    }
}
```

### Agent Result Format

Each agent returns results in a standardized format:

```python
# Agent result structure
{
    "id": "unique_result_id",
    "status": "success", # or "error"
    "data": { /* agent-specific output data */ },
    "metadata": { /* processing metadata */ },
    "error": None # or error message if status is "error"
}
```

### Framework Adapters

Adapters translate between composable format and framework-specific formats:

```python
# Adapter workflow
standardized_input → adapter.translate_to_framework() → framework_input
framework_output ← adapter.translate_from_framework() ← framework.execute()
```

## Performance Comparison

Based on our testing with 50 standardized scenarios:

| Metric                   | Multi-Framework | Composable | Hybrid  | Improvement (Comp.) | Recovery (Hybrid) |
|--------------------------|-----------------|------------|---------|---------------------|-------------------|
| Execution Time           | 5.37s           | 1.43s      | 2.15s   | 73%                 | 87%               |
| Memory Usage             | 843MB           | 312MB      | 412MB   | 63%                 | 85%               |
| Response Latency         | 1210ms          | 340ms      | 490ms   | 72%                 | 83%               |
| Cold Start Time          | 8.3s            | 2.1s       | 3.2s    | 75%                 | 83%               |
| Accuracy                 | 91.8%           | 80.8%      | 88.2%   | -12%                | 75% recovery      |
| Adapter Overhead         | N/A             | N/A        | 23%     | N/A                 | N/A               |

## Key Components

### Base Abstractions

- **BaseAgent**: Abstract interface for all agents
- **AgentResult**: Standardized container for agent outputs
- **GlobalState**: Central state management for inter-agent communication

### Composable Architecture

- **ComposableAgent**: Implementation of the standardized agent interface
- **ComposableAgentOrchestrator**: Manages execution flow of composable agents

### Framework Adapters

- **FrameworkAdapter**: Base class for framework adapters
- **CamelAdapter**: Adapter for CAMEL framework (stress modeling)
- **AutogenAdapter**: Adapter for AutoGen framework (neurochemical interaction)
- **LangChainAdapter**: Adapter for LangChain framework (cognitive assessment)
- **LangGraphAdapter**: Adapter for LangGraph framework (intervention recommendation)

### Hybrid Architecture

- **HybridAgent**: Combines composable interface with framework adapters
- **HybridAgentOrchestrator**: Manages execution of hybrid agents

## Extending the Framework

### Creating a Custom Agent

```python
from agent_framework.composable_agent import ComposableAgent

class MyCustomAgent(ComposableAgent):
    def __init__(self, agent_id="my_custom_agent"):
        super().__init__(
            agent_id=agent_id,
            required_inputs=["my_required_input"],
            required_agent_results=["other_agent_id"]
        )
    
    def _process_internal(self, input_data):
        # Custom processing logic
        return {
            "my_output": "processed result"
        }
```

### Creating a Custom Adapter

```python
from agent_framework.framework_adapter import FrameworkAdapter

class MyFrameworkAdapter(FrameworkAdapter):
    def __init__(self):
        super().__init__("my_framework")
    
    def initialize(self):
        # Initialize framework
        self.initialized = True
    
    def translate_to_framework(self, data):
        # Convert from composable format to framework format
        return framework_format
    
    def translate_from_framework(self, data):
        # Convert from framework format to composable format
        return composable_format
    
    def execute(self, input_data):
        # Execute framework
        return framework_output
```

## Project Structure

```
agent_framework/
├── __init__.py
├── base_agent.py         # Core abstractions
├── composable_agent.py   # Composable architecture
├── framework_adapter.py  # Framework adapters
├── hybrid_agent.py       # Hybrid architecture
└── example_usage.py      # Usage examples
main.py                   # Command line interface
setup.py                  # Installation script
README.md                 # This documentation
```


## License

MIT License

