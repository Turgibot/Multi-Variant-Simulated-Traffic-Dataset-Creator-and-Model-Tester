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

- Python **3.8** (see `.python-version`; slightly newer 3.8.x is fine)
- **Linux (GUI):** system packages for Qt’s XCB plugin — at minimum **`libxcb-cursor0`** (PySide6 6.5+). See [`INSTALL_QT_DEPS.md`](INSTALL_QT_DEPS.md) for details and a fuller dependency list.
- [uv](https://docs.astral.sh/uv/) (recommended for installs), or `pip` with a virtual environment. If you install uv with the Astral script, ensure **`~/.local/bin`** is on your `PATH` (or open a new shell after install).
- SUMO (Simulation of Urban MObility) — [installation guide](https://sumo.dlr.de/docs/Installing/index.html)
- Python packages: **`requirements.txt`** (runtime) and **`requirements-dev.txt`** (development). You can also install from your own `pip freeze` output with **`uv pip install -r freeze.txt`** if you maintain one (watch for conflicting pins between heavy ML stacks and the rest of the app).

## Installation

1. Clone the repository (suggested directory name):
```bash
git clone <repository-url> graph-traffic-dataset-creator
cd graph-traffic-dataset-creator
```

### Using uv (recommended)

Install uv if you do not have it yet: see [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

On **Linux**, install Qt XCB dependencies for the GUI before running the app (see [Requirements](#requirements) and [`INSTALL_QT_DEPS.md`](INSTALL_QT_DEPS.md)).

From the project root:

```bash
uv venv --python 3.8
uv pip install -r requirements.txt
```

If uv reports that Python 3.8 is missing, install it once with: `uv python install 3.8`.

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

**Bundled with Python deps:** `requirements.txt` includes **`eclipse-sumo`** (full CLI tools under your venv). After `uv pip install -r requirements.txt`, the app auto-detects `SUMO_HOME` from that package.

**Or use a system install:** follow the [SUMO installation guide](https://sumo.dlr.de/docs/Installing/index.html) and ensure binaries are on your `PATH` or set `SUMO_HOME`.

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

