# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NEBULA is a platform for **Decentralized Federated Learning (DFL)** — collaborative ML across distributed devices without a central server. It supports three paradigms: Centralized FL (CFL), Decentralized FL (DFL), and Semi-Decentralized FL (SDFL, with dynamic role assignment).

Three main components:
- **Core** (`nebula/core/`): FL engine deployed on each node — training, aggregation, P2P communication
- **Controller** (`nebula/controller/`): FastAPI REST API that manages experiments and deploys nodes via Docker
- **Frontend** (`nebula/frontend/`): FastAPI + Jinja2 web UI for experiment setup and real-time monitoring

## Development Commands

```bash
# Setup (installs Python 3.11 via uv, syncs deps, installs pre-commit hooks)
make install
source .venv/bin/activate

# Run the platform (controller + frontend + nodes)
python app/main.py
# With options: -c <config_dir>, -p (production), -ad (advanced analytics)
# Stop: python app/main.py --stop all | --stop nodes

# Run experiments programmatically via API
python run_experiment.py

# Code quality (pre-commit: check-case-conflict, check-merge-conflict, check-toml, check-yaml, end-of-file-fixer, trailing-whitespace)
make check
# Extended checks (adds black, mypy, deptry)
make check-plus
# Check specific files
uv run pre-commit run --files <file1> <file2>

# Build wheel
make build

# Docs
make full-install && make doc-serve  # serves at http://127.0.0.1:8000

# Docker images
docker build -t nebula-core .
docker build -t nebula-controller -f nebula/controller/Dockerfile .
docker build -t nebula-frontend -f nebula/frontend/Dockerfile .
```

**Note:** Ruff linting is currently disabled in `.pre-commit-config.yaml` (commented out). `make check` runs only the basic pre-commit hooks.

Default ports: Controller 5050, Frontend 6060, Statistics 8080, Grafana 6040, Loki 6010, WAF 6050

Default frontend login: `admin` / `admin`

## Architecture

### FL Lifecycle (per round)

1. **Local Training** → 2. **Model Propagation** (broadcast to neighbors) → 3. **Aggregation** (FedAvg, Krum, Median, TrimmedMean) → 4. **Validation** → 5. **Reputation Update** → 6. **Role Adjustment** (SDFL only)

Coordinated by an event-driven system: `EventManager` (singleton, pub-sub) dispatches `NebulaEvents` (RoundStartEvent, UpdateReceivedEvent, AggregationEvent, RoundEndEvent, etc.) with async callbacks.

### Role System (Strategy Pattern)

`RoleBehavior` abstract base class in `nebula/core/noderole.py` with 7 implementations:
- **TRAINER**, **AGGREGATOR**, **TRAINER_AGGREGATOR** (most common in DFL), **PROXY**, **SERVER** (CFL), **IDLE**, **MALICIOUS**
- Each implements `extended_learning_cycle()`, `select_nodes_to_wait()`, `resolve_missing_updates()`
- Dynamic role transitions via `change_role_behavior()` for SDFL
- MALICIOUS wraps a benign behavior for attack simulation

### Core Subsystems (`nebula/core/`)

- **`engine.py`**: Main orchestrator — coordinates training, aggregation, communication, events
- **`network/`**: Async P2P layer — `CommunicationsManager`, `Connection`, `Propagator`, `Discoverer`, `Forwarder`, `Blacklist`. Protobuf serialization (`pb/`), LZ4 compression
- **`aggregation/`**: Plugin architecture with `Aggregator` base class. Strategies: FedAvg (default), Krum (Byzantine-robust), Median, TrimmedMean. `updatehandlers/` for pre/post-processing
- **`training/`**: PyTorch Lightning-based (`lightning.py`). Siamese training also available
- **`datasets/`**: Built-in: MNIST, Fashion-MNIST, EMNIST, CIFAR-10, CIFAR-100. `NebulaPartition` base class for custom datasets. IID and non-IID distributions (Dirichlet, pathological). Each dataset has a `PartitionHandler`
- **`models/`**: CNN, MLP, ResNet, MobileNet variants. Inherit from `nn.Module`
- **`config/config.py`**: JSON-based configuration loader — topology config (network structure, federation params) and participant config (per-node settings)

### Addons (`nebula/addons/`)

Extensible plugin system: `attacks/` (model/gradient/dataset poisoning), `reputation/`, `reporter/`, `trustworthiness/` (6 pillars), `networksimulation/`, `gps/`, `waf/`

### Entry Points

- `app/main.py` → `app/deployer.py`: CLI entry point, orchestrates controller + frontend + nodes
- `run_experiment.py`: Programmatic experiment deployment via NEBULA API (starts platform, deploys scenarios via REST)
- `script/run_dfl.sh`: Shell wrapper — `./run_dfl.sh <username>` (sets CUDA_VISIBLE_DEVICES=4,5,6,7)
- `DFL_2/`: Batch experiment management — `generate_experiments.py`, `run_experiments.py`, `Experiment.tsv`

## Pseudo Aggregation

Reduces communication by 50% using EMA to predict neighbor models. Alternates actual rounds (with communication) and pseudo rounds (predicted models, no communication). Configurable via frontend UI or participant JSON config. See `PSEUDO_AGGREGATION.md` for details.

## Logging

JSON-formatted metrics in `app/logs/<experiment>/`: `*_metrics.json` (complete history), `*_rounds.json` (per-round summary), `*_summary.json` (experiment summary). Also produces TensorBoard event files. See `LOGGING_GUIDE.md` for full schema.

## Key Technical Details

- All network I/O is async (asyncio-based, uvloop)
- Model weights serialized via Protocol Buffers with LZ4 compression
- PyTorch Lightning training with CUDA and Apple MPS support
- Python 3.10-3.11, managed with `uv` package manager
- Dependency groups in `pyproject.toml`: `core`, `controller`, `frontend`, `docs`
- Type hints enforced by mypy (strict mode)
- Line length: 120 characters (ruff config in `pyproject.toml`)
- No formal test suite — `test.py` at root is a scratch file
