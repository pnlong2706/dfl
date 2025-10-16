# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NEBULA (formerly Fedstellar) is a cutting-edge platform for **Decentralized Federated Learning (DFL)** that enables collaborative machine learning across distributed devices without requiring a central server, while preserving data privacy.

### What is Decentralized Federated Learning?

Unlike traditional Federated Learning (FL) which relies on a central server to coordinate training and aggregate models, **DFL** operates in a fully peer-to-peer manner where:

- **No central authority**: Nodes communicate directly with neighbors in the network
- **Privacy-preserving**: Raw data never leaves local devices; only model updates are shared
- **Fault-tolerant**: Network can continue operating even if individual nodes fail
- **Topology-agnostic**: Supports various network structures (star, ring, mesh, fully-connected)
- **Scalable**: Eliminates the central server bottleneck present in traditional FL

NEBULA uniquely supports **three federated learning paradigms**:

1. **Centralized FL (CFL)**: Traditional server-client architecture
2. **Decentralized FL (DFL)**: Peer-to-peer without central coordination
3. **Semi-Decentralized FL (SDFL)**: Hybrid approach with dynamic role assignment and self-organization

### Platform Components

- **Core** (`nebula/core/`): The fundamental federated learning engine deployed on each participating device. Handles training, model aggregation, P2P communication, and role management.
- **Controller** (`nebula/controller/`): Orchestrator service that manages experiments, deploys nodes via Docker, and provides REST API for coordination.
- **Frontend** (`nebula/frontend/`): Web-based user interface for experiment setup, real-time monitoring, and network visualization.

## Development Commands

### Initial Setup

```bash
# Clone repository
git clone https://github.com/CyberDataLab/nebula.git
cd nebula

# Install Python 3.11 and create virtual environment with uv
make install

# Activate virtual environment
source .venv/bin/activate        # Linux/MacOS
.venv\Scripts\activate           # Windows

# Verify installation
python app/main.py --version

# Install additional production dependencies (WAF, monitoring)
make install-production
```

Default frontend login credentials:
- Username: `admin`
- Password: `admin`

### Code Quality

```bash
# Run pre-commit hooks (ruff, mypy, isort)
make check

# Run extended quality checks (includes black, mypy, deptry)
make check-plus

# Run pre-commit on specific files
uv run pre-commit run --files <file1> <file2>
```

### Building & Publishing

```bash
# Build wheel package
make build

# Update dependency lock file
make lock

# Publish to PyPI (requires PYPI_TOKEN)
make publish
```

### Documentation

```bash
# Install documentation dependencies
make full-install

# Build documentation
make doc-build

# Serve documentation locally at http://127.0.0.1:8000
make doc-serve

# Test documentation build (strict mode)
make doc-test
```

### Docker Management

```bash
# Update all three docker images (interactive prompts)
make update-dockers

# Build individual images manually
docker build -t nebula-core .
docker build -t nebula-controller -f nebula/controller/Dockerfile .
docker build -t nebula-frontend -f nebula/frontend/Dockerfile .
```

### Running NEBULA

```bash
# Start the platform (controller + frontend + nodes)
python app/main.py

# Common options
python app/main.py -c <config_dir>           # Custom config directory
python app/main.py -p                        # Production mode
python app/main.py -ad                       # Enable advanced analytics
python app/main.py --stop all                # Stop all services
python app/main.py --stop nodes              # Stop only nodes
```

Default ports:
- Controller: 5050
- Frontend: 6060
- Statistics: 8080
- Grafana: 6040
- Loki: 6010
- WAF: 6050

## Architecture

### Federated Learning Lifecycle

In a typical NEBULA DFL round, the following occurs:

1. **Local Training**: Each node trains on its local dataset for N epochs
2. **Model Propagation**: Trained model updates are broadcast to connected neighbors
3. **Aggregation**: Nodes receive updates from neighbors and aggregate them using configured strategy (FedAvg, Krum, etc.)
4. **Validation**: Aggregated model is validated on local test set
5. **Reputation Update**: Node performance is tracked for trustworthiness assessment
6. **Role Adjustment** (SDFL only): Nodes may dynamically change roles based on performance/network conditions

This process is coordinated by the event-driven architecture using `NebulaEvents` (e.g., `RoundStartEvent`, `UpdateReceivedEvent`, `AggregationEvent`, `RoundEndEvent`).

### Core Engine (`nebula/core/`)

The core module contains the federated learning engine that runs on each node:

- **`engine.py`**: Main Engine class that orchestrates the entire FL process. Coordinates training, aggregation, communication, and event management.
- **`node.py`**: Node class representing a participant in the federation. Contains device information, network configuration, and participant metadata.
- **`noderole.py`**: Role-based behavior system implementing the Strategy pattern
  - Uses `RoleBehavior` abstract base class with `extended_learning_cycle()` method
  - Factory function `factory_role_behavior()` creates appropriate role instances
  - Supports dynamic role transitions via `change_role_behavior()` for SDFL
  - Each role has unique learning/aggregation logic
- **`eventmanager.py`**: Event-driven system using publish-subscribe pattern
  - Events trigger callbacks registered by different components
  - Enables loose coupling between training, aggregation, and communication layers

#### Role System

NEBULA implements a sophisticated role-based architecture supporting:

- **TRAINER**: Performs local training only
- **AGGREGATOR**: Aggregates updates from neighbors
- **TRAINER_AGGREGATOR**: Both trains and aggregates (most common in DFL)
- **PROXY**: Forwards messages without training/aggregating
- **SERVER**: Acts as central server in centralized FL
- **IDLE**: Inactive participant
- **MALICIOUS**: Simulates adversarial behavior for robustness testing

Nodes can dynamically change roles at runtime using the `change_role_behavior()` function.

#### Key Subsystems

- **`network/`**: P2P communication layer
  - `communications.py`: Main `CommunicationsManager` handles all network operations
  - `connection.py`: Connection lifecycle (connect, disconnect, health checks)
  - `propagator.py`: Broadcasts model updates to neighbors
  - `messages.py`: Message protocol definitions (protobuf-based)
  - `blacklist.py`: Malicious node filtering and reputation tracking
  - `forwarder.py`: Message routing in multi-hop networks
  - `discoverer.py`: Neighbor discovery mechanism

- **`aggregation/`**: Model aggregation strategies
  - `aggregator.py`: Base `Aggregator` class with plugin architecture
  - `fedavg.py`: Federated Averaging (default, simple weighted average)
  - `krum.py`: Krum algorithm (Byzantine-robust, selects most similar models)
  - `median.py`: Coordinate-wise median (robust to outliers)
  - `trimmedmean.py`: Trimmed mean (removes extreme values before averaging)
  - `updatehandlers/`: Pre/post-processing of model updates

- **`training/`**: Training implementations
  - `lightning.py`: PyTorch Lightning-based training loop (primary)
  - `siamese.py`: Siamese network training for specialized tasks
  - Supports custom training callbacks and metrics logging
  - Automatic checkpointing and model saving

- **`datasets/`**: Data partitioning and loading
  - Built-in: MNIST, Fashion-MNIST, EMNIST, CIFAR-10, CIFAR-100
  - `NebulaPartition`: Abstract base class for custom datasets
  - Supports IID and non-IID data distributions (Dirichlet, pathological)
  - `DataModule`: PyTorch Lightning data module wrapper
  - Each dataset has a `PartitionHandler` for data splitting across nodes

- **`models/`**: Neural network architectures
  - CNNs: Various depths and architectures per dataset
  - MLPs: Fully-connected networks for simpler tasks
  - ResNet: Deep residual networks for complex datasets
  - MobileNet variants: Lightweight models for resource-constrained devices
  - Model-agnostic design: Inherit from `nn.Module` and implement forward pass

- **`situationalawareness/`**: Runtime monitoring and adaptive behavior
  - Tracks resource usage (CPU, GPU, memory, network)
  - Monitors training metrics (loss, accuracy) in real-time
  - Enables adaptive strategies based on system conditions

- **`addons/`**: Extensible plugin system
  - `attacks/`: Adversarial attack simulation (model poisoning, gradient manipulation)
  - `reputation/`: Reputation-based trust scoring for nodes
  - `reporter/`: Metrics collection and reporting to controller
  - `trustworthiness/`: Model trustworthiness assessment
  - `waf/`: Web Application Firewall for production deployments
  - `functions.py`: Utility functions shared across addons

### Controller (`nebula/controller/`)

- **`controller.py`**: FastAPI-based REST API for experiment management
- **`scenarios.py`**: Scenario configuration and deployment logic
- **`database.py`**: SQLite database for experiment metadata
- Manages Docker containers for node deployment

### Frontend (`nebula/frontend/`)

- **`app.py`**: FastAPI web application with Jinja2 templates
- Real-time experiment monitoring
- D3.js-based network topology visualization
- TensorBoard integration for metrics

### Application Entry Point (`app/`)

- **`main.py`**: CLI entry point with argument parsing
- **`deployer.py`**: Orchestrates controller, frontend, and node deployment
- Platform-specific scripts in `linux/`, `macos/`, `windows/`

## Configuration System

NEBULA uses a JSON-based configuration system with two main configuration types:

### Topology Configuration
Defines the overall network structure and experiment parameters:
- Network topology type (star, ring, mesh, fully-connected)
- Federation settings (rounds, epochs, learning rates)
- Aggregation algorithm selection
- Data distribution strategy (IID vs non-IID)

### Participant Configuration
Defines individual node settings:
- Device information (idx, role, IP, port)
- Network configuration (neighbors, communication protocols)
- Training parameters (model, dataset, batch size, optimizer)
- Logging and metrics tracking paths

The `Config` class (`nebula/config/config.py`) loads these JSON files and provides configuration access throughout the platform. Configurations are passed to the Engine during initialization.

## Project Configuration Files

- **`pyproject.toml`**: Project metadata and dependencies
  - Separate dependency groups: `core`, `controller`, `frontend`, `docs`
  - Python 3.10-3.11 supported
  - Development tools: ruff, mypy, isort, pre-commit

- **`.pre-commit-config.yaml`**: Pre-commit hooks for code quality

- **`Makefile`**: Uses `uv` package manager (Astral's fast Python package installer)

## Testing & Development Workflow

1. Make changes to code
2. Run `make check` to verify code quality (ruff will auto-fix many issues)
3. Test locally with `python app/main.py`
4. For core changes: test with different node roles and network topologies
5. Commit following conventional-commit style
6. Sign commits with Developer Certificate of Origin

## Important Technical Details

### Architecture Patterns
- **Event-driven architecture**: `EventManager` + `NebulaEvents` enable reactive coordination
- **Strategy pattern**: Role behaviors are swappable implementations of `RoleBehavior`
- **Observer pattern**: Event subscribers react to training/aggregation events
- **Modular design**: Easy to extend with new datasets, models, aggregation algorithms

### Communication & Serialization
- Network communication is fully **asynchronous** (asyncio-based)
- Model weights serialized using **Protocol Buffers** (`nebula/core/pb/`)
- Supports compression (LZ4) for efficient network transmission
- P2P connections managed via `CommunicationsManager`

### Security & Trustworthiness
- **Reputation system**: Tracks node reliability and can exclude malicious participants
- **Attack simulation**: Built-in adversarial attack support via Adversarial Robustness Toolbox
- **Blacklist mechanism**: Automatically filters out Byzantine nodes
- **Secure communication**: TLS support for encrypted model transmission

### Performance
- **GPU acceleration**: Full PyTorch CUDA support, Apple MPS on macOS
- **Async I/O**: Non-blocking network operations for scalability
- **Efficient aggregation**: Byzantine-robust algorithms (Krum, Median, Trimmed Mean)
- **PyTorch Lightning**: Optimized training loops with automatic mixed precision support

### Code Standards
- Type hints required (enforced by mypy with strict settings)
- Line length: 120 characters (ruff configuration)
- Conventional-commit style for commit messages
- Developer Certificate of Origin required for all commits

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS. Windows supported but less tested.
- **RAM**: Minimum 8 GB, 32 GB recommended for multi-node experiments
- **Disk**: 20 GB minimum for platform + datasets + logs
- **Docker**: Engine 24.0.4+ and Compose 2.19.0+ required for node deployment
- **GPU** (optional): NVIDIA Driver 525.60.13+, CUDA 12.1, NVIDIA Container Toolkit

## License

AGPLv3 (Community Edition) - see LICENSE file. Enterprise Edition available commercially.
