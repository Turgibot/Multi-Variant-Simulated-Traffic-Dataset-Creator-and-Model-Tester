# Graph Traffic Dataset Creator

A cross-platform GUI application for SUMO-based traffic simulation and graph traffic dataset generation—from simulated or real trajectories to graph-structured datasets.

## Features

- **Cross-Platform GUI**: Standalone executable for Windows, macOS, and Linux
- **SUMO Integration**: Run and visualize SUMO traffic simulations
- **Network Management**: Upload and manage SUMO network files
- **Simulation Configuration**: Configure traffic patterns (rush-hour, weekend, custom)
- **Dataset Generation**: Generate multiple dataset variants (trajectory, sensor-based, GNN)
- **Visualization**: Statistical analysis and visualization of results

## Requirements

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) (recommended for installs), or `pip` with a virtual environment
- SUMO (Simulation of Urban MObility) — [installation guide](https://sumo.dlr.de/docs/Installing/index.html)
- Python packages are listed in `requirements.txt` (runtime) and `requirements-dev.txt` (development)

## Installation

1. Clone the repository (suggested directory name):
```bash
git clone <repository-url> graph-traffic-dataset-creator
cd graph-traffic-dataset-creator
```

### Using uv (recommended)

Install uv if you do not have it yet: see [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

From the project root:

```bash
uv venv
uv pip install -r requirements.txt
```

Optional development tools (formatting, linting, tests):

```bash
uv pip install -r requirements-dev.txt
```

`uv venv` creates `.venv` in the project directory. If the repo includes a `.python-version` file, uv uses that Python version when creating the environment.

Run the GUI without activating the venv:

```bash
uv run python src/main.py
```

Or activate the environment and run with plain `python`:

```bash
source .venv/bin/activate   # Linux and macOS
# .venv\Scripts\activate    # Windows cmd
python src/main.py
```

### Using pip (alternative)

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional
python src/main.py
```

### SUMO

Install SUMO separately:

- Follow the [SUMO installation guide](https://sumo.dlr.de/docs/Installing/index.html)
- Ensure SUMO binaries are on your `PATH`

## Project Structure

```
project-root/
├── src/
│   ├── gui/              # GUI components
│   ├── simulation/       # SUMO integration
│   ├── data_collection/ # Dataset generation
│   ├── visualization/   # Charts and graphs
│   └── utils/           # Utilities and helpers
├── config/              # Configuration files
├── tests/               # Unit and integration tests
├── docs/                # Documentation
├── examples/            # Example networks and configs
└── build/               # Build scripts for executables
```

## Development

Install dev dependencies first (`uv pip install -r requirements-dev.txt` or the pip equivalent).

### Code Formatting

The project uses `black` for code formatting:

```bash
uv run black src/ tests/
# or: black src/ tests/
```

### Linting

Linting is configured with `pylint` and `flake8`:

```bash
uv run pylint src/
uv run flake8 src/
```

### Testing

Run tests with pytest:

```bash
uv run pytest
```

## Usage

(To be added as the application is developed)

## License

(To be determined)

## Contributing

(To be added)

## Documentation

See `PROJECT_DESIGN.md` for detailed project design and requirements.

