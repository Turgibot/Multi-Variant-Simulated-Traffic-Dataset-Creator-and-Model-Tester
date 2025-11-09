# Multi-Variant Simulated Traffic Dataset Creator and Model Tester

A comprehensive cross-platform GUI application for SUMO-based traffic simulation, dataset generation, and model testing. This tool provides an integrated environment for creating diverse traffic datasets and evaluating prediction models.

## Features

- **Cross-Platform GUI**: Standalone executable for Windows, macOS, and Linux
- **SUMO Integration**: Run and visualize SUMO traffic simulations
- **Network Management**: Upload and manage SUMO network files
- **Simulation Configuration**: Configure traffic patterns (rush-hour, weekend, custom)
- **Dataset Generation**: Generate multiple dataset variants (trajectory, sensor-based, GNN)
- **Model Testing**: Load models, run inference, and evaluate performance
- **Visualization**: Statistical analysis and visualization of results

## Requirements

- Python 3.8+
- SUMO (Simulation of Urban MObility) - [Installation Guide](https://sumo.dlr.de/docs/Installing/index.html)
- See `requirements.txt` for Python dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Multi-Variant-Simulated-Traffic-Dataset-Creator-and-Model-Tester
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install development dependencies (optional):
```bash
pip install -r requirements-dev.txt
```

5. Install SUMO separately:
   - Follow the [SUMO installation guide](https://sumo.dlr.de/docs/Installing/index.html)
   - Ensure SUMO binaries are in your PATH

## Project Structure

```
project-root/
├── src/
│   ├── gui/              # GUI components
│   ├── simulation/       # SUMO integration
│   ├── data_collection/ # Dataset generation
│   ├── model_testing/   # Model evaluation
│   ├── visualization/   # Charts and graphs
│   └── utils/           # Utilities and helpers
├── config/              # Configuration files
├── tests/               # Unit and integration tests
├── docs/                # Documentation
├── examples/            # Example networks and configs
└── build/               # Build scripts for executables
```

## Development

### Code Formatting

The project uses `black` for code formatting:
```bash
black src/ tests/
```

### Linting

Linting is configured with `pylint` and `flake8`:
```bash
pylint src/
flake8 src/
```

### Testing

Run tests with pytest:
```bash
pytest
```

## Usage

(To be added as the application is developed)

## License

(To be determined)

## Contributing

(To be added)

## Documentation

See `PROJECT_DESIGN.md` for detailed project design and requirements.

