# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PySide6 GUI application for SUMO-based traffic simulation and graph traffic dataset generation. Users build simulation projects, configure traffic patterns, run SUMO simulations via TraCI, then convert vehicle trajectory data into graph-structured datasets for GNN/ML models. LLM integration (OpenAI, Anthropic, Ollama) assists with route generation.

## Commands

### Running the application
```bash
uv run python src/main.py
```

### Running tests
```bash
uv run pytest
# Single test file:
uv run pytest tests/test_basic_setup.py
```

### Formatting and linting
```bash
uv run black src/ tests/          # 100-char line length
uv run pylint src/
uv run flake8 src/
```

### Building a standalone executable
```bash
bash scripts/build_executable.sh
```

### Standalone trajectory conversion CLI
```bash
uv run python scripts/csv_to_steps.py
uv run python scripts/convert_trajectories_fast.py
```

## Architecture

### Layer structure

```
GUI Layer (src/gui/)
  └─ MainWindow → WelcomePage → SimulationPage / DatasetGenerationPage /
                                RouteGenerationPage / DatasetConversionPage /
                                DebugTrajectoryPage
Service/Utility Layer (src/utils/)
  └─ ProjectManager, SUMOConfigManager, SimulationRunner, SimulationDB,
     NetworkParser, TrajectoryConverter, csv_to_steps_runner, route_finding,
     LLMRouteParser
External Systems
  └─ SUMO (via TraCI), OpenAI/Anthropic/Ollama APIs
```

GUI pages connect to services via Qt signals. Background tasks (simulation loop, dataset conversion) run in QThread workers.

### Key modules

| Module | Role |
|---|---|
| `src/main.py` | Entry point; sets up SUMO path and Qt platform, launches MainWindow |
| `src/gui/main_window.py` | Main window + WelcomePage (project selection/creation) |
| `src/gui/simulation_page.py` | Runs SUMO via TraCI; drives SimulationRunner each timestep |
| `src/gui/dataset_conversion_page.py` | Orchestrates trajectory→graph conversion (largest file, 323 KB) |
| `src/utils/entities.py` | Core data models: Junction, Road, Vehicle, Zone |
| `src/utils/project_manager.py` | Project lifecycle (create, load, registry) |
| `src/utils/simulation_db.py` | SQLite schema (junctions, roads, vehicles, zones, vehicle_states, schedules) |
| `src/utils/simulation_runner.py` | Per-timestep update/dispatch logic for the simulation loop |
| `src/utils/network_parser.py` | Parses SUMO `.net.xml` files, coordinate projection |
| `src/utils/trajectory_converter.py` | CSV→JSON trajectory conversion |
| `src/utils/csv_to_steps_runner.py` | Optimized single-pass CSV→step-JSON conversion with multiprocessing |
| `src/utils/route_finding.py` | EdgeSpatialIndex, Dijkstra/A* pathfinding, GPS→edge matching |
| `src/utils/llm_route_parser.py` | LLM abstraction layer (OpenAI, Anthropic, Ollama, Transformers) |
| `src/utils/sumo_config_manager.py` | Reads/writes `sumo_config.json` and SUMO config files |

### Project file layout (per user project)

```
<project_root>/
  project_info.json          # name, type, created_at
  sumo_config.json           # paths to SUMO files, dataset output folder
  simulation.config.json     # vehicle counts, time ranges, zones, patterns
  simulation_run_settings.json
  config/                    # *.sumocfg, *.net.xml, *.add.xml (SUMO files)
  simulation/<name>.db       # SQLite DB with all simulation state
  datasets_*/                # Output graph dataset files
  datasets_steps_all_routes/ # Per-timestep step_*.json, labels_*.json, node_features_*.json
```

### Simulation execution flow

1. `SimulationPage` opens a TraCI connection to SUMO.
2. Each timestep: `traci.simulationStep()` → `SimulationRunner.update()` syncs vehicle/road state to SQLite → `SimulationRunner.dispatch()` inserts scheduled vehicles.
3. UI is updated with current vehicle positions.
4. On completion, trajectories and statistics are exported.

### Trajectory conversion pipeline

```
train.csv (Porto format: POLYLINE, TIMESTAMP)
  → trajectory_converter.py  (parse, trim static points)
  → route_finding.py         (GPS→edge matching via EdgeSpatialIndex)
  → csv_to_steps_runner.py   (aggregate per-timestep snapshots, multiprocessing)
  → output JSON files        (step_*.json, labels_*.json, node/edge features)
```

## Configuration

- **Black**: 100-character line length, Python 3.8 target (see `pyproject.toml`)
- **Pylint**: `missing-docstring`, `too-few-methods`, `too-many-args` are disabled
- **Python version**: 3.8 (`.python-version`)
- **Dependency management**: `uv` with `requirements.txt` / `requirements-dev.txt`

## Linux setup note

On Linux, Qt XCB platform dependencies must be installed before the GUI will launch. See `INSTALL_QT_DEPS.md` for the full package list (the key one is `libxcb-cursor0`).
