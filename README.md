# CortexFlow: Multi-Agent AI System for Modeling Cognitive Processes

CortexFlow is a modular framework for orchestrating commercial AI workflow agents to model cognitive processes. This project demonstrates the integration of 8 different agent frameworks into a unified system to simulate the Yerkes-Dodson law of stress and productivity.

## Key Features

- **Layered Architecture**: Orchestrates agents in Orchestration, Processing, Knowledge, and Optimization layers
- **Flexible Agent Integration**: Plug-and-play with existing commercial AI workflow agents
- **State Management**: LangGraph-powered central coordinator for agent communication
- **Interactive Visualization**: Google Colab notebooks for result analysis
- **CLI Interface**: Simple command-line tool for running simulations

## System Architecture

CortexFlow implements a hub-and-spoke architecture with LangGraph as the central coordinator:

| Layer | Agents | Function |
|-------|--------|----------|
| **Orchestration** | LangGraph, E2B | State management and agent runtime environment |
| **Processing** | AutoGen, CrewAI, CAMEL | Neurochemical dialogue and cognitive processing |
| **Knowledge** | Haystack, LlamaIndex | Intervention recommendations and effectiveness analysis |
| **Optimization** | OpenDevin | Workflow optimization based on stress state |

## Agent Integration

CortexFlow orchestrates these commercial AI workflow agents:

| Agent | Role | Impact Score |
|-------|------|-------------|
| **LangGraph** | State Management & Coordination | 8.1/10 |
| **CrewAI** | Memory & Productivity Specialist | 7.3/10 |
| **E2B** | Real-time Agent Deployment | 7.1/10 |
| **AutoGen** | Neurochemical Dialogue Simulation | 7.1/10 |
| **LlamaIndex** | Intervention Effectiveness Analysis | 6.9/10 |
| **OpenDevin** | Workflow Optimization | 6.5/10 |
| **Haystack** | Evidence-based Intervention Recommendation | 6.3/10 |
| **CAMEL** | Stress-Cognition Dialogue | 5.9/10 |

## Simulation Results

CortexFlow successfully models the Yerkes-Dodson law, demonstrating how moderate stress improves performance while high stress impairs cognitive function.

The simulation shows:
- Initial performance degradation as stress increases
- Maximum productivity impact at peak stress (cortisol level ~75)
- Recovery phase when interventions are applied
- Different effects on memory efficiency vs. decision quality

## Quick Start

```bash
# Install the package
pip install cortexflow

# Run a basic simulation
cortexflow run --stress moderate --task creative --intervention micro_breaks

# Visualize results
cortexflow visualize results.json
```

## Installation Options

CortexFlow supports modular installation based on which agents you want to use:

```bash
# Full installation with all agents
pip install cortexflow[all]

# Minimal installation with core functionality
pip install cortexflow

# Install with specific agents
pip install cortexflow[langgraph,autogen,crewai]

# Development installation
pip install cortexflow[dev]
```

## Required Dependencies

CortexFlow has these minimum requirements:
- Python 3.8+
- Numpy 1.20+
- Pandas 1.3+
- Matplotlib 3.4+

Agent-specific dependencies are installed when selecting the corresponding installation option.

## Project Structure

```
CortexFlow/
├── agents/              # Agent implementations
├── core/                # Core simulation logic
├── cli/                 # Command-line interface
├── visualizations/      # Visualization tools
├── examples/            # Example simulations and notebooks
├── tests/               # Unit and integration tests
└── docs/                # Documentation
```

## Agent Implementation Pattern

Each agent follows a consistent interface:

```python
class AgentBase(ABC):
    @abstractmethod
    async def process(self, state: SimulationState) -> SimulationState:
        """Process the current state and return updated state."""
        pass
        
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return the list of capabilities this agent provides."""
        pass
```

This allows for dynamic agent registration and graceful fallbacks.

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/cortexflow.git
cd cortexflow

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Research Applications

CortexFlow can be applied to research in:

- Cognitive psychology and stress management
- Productivity optimization in knowledge work
- Intervention efficacy modeling
- Multi-agent AI system design patterns

## Extension Points

CortexFlow is designed for extension through these mechanisms:

1. **Custom Agents**: Implement the `AgentBase` interface to add new agent types
2. **Alternative State Models**: Extend the `SimulationState` class for different domains
3. **Visualization Extensions**: Add new visualization types in the `visualizations` module
4. **Intervention Strategies**: Extend the available interventions in the `core.types` module

## Citation

If you use CortexFlow in your research, please cite:

```bibtex
@software{yourlastname2025cortexflow,
  author = {Your Name},
  title = {CortexFlow: Multi-Agent AI System for Modeling Cognitive Processes},
  year = {2025},
  url = {https://github.com/yourusername/cortexflow}
}
```

## Related Projects

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [E2B](https://github.com/e2b-dev/e2b)
- [Haystack](https://github.com/deepset-ai/haystack)
- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [CAMEL](https://github.com/camel-ai/camel)
- [OpenDevin](https://github.com/OpenDevin/OpenDevin)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Before contributing, please:

1. Check for open issues or create a new one to discuss your intended changes
2. Fork the repository and create a feature branch
3. Follow the code style guidelines (Black, isort, mypy)
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
