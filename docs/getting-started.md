# Getting Started Guide

Welcome to the Federated Learning Based Energy-Efficient Cloud Resource Allocation project! This guide will help you get up and running quickly.

## Prerequisites

- **Python 3.9+**: Required for running the federated learning algorithms
- **Docker**: Optional, for containerized deployment
- **Azure CLI**: Optional, for cloud deployment
- **Git**: For version control

## Quick Start

### Option 1: Automated Setup (Recommended)

#### On macOS/Linux:
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

#### On Windows:
```cmd
scripts\setup.bat
```

### Option 2: Manual Setup

1. **Clone the repository** (if not done already):
   ```bash
   git clone <repository-url>
   cd ccs
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**:
   ```bash
   export PYTHONPATH=$(pwd)/src
   export FLASK_ENV=development
   ```

## Running the System

### 1. Run a Simulation

The easiest way to see the system in action is to run a simulation:

```bash
python src/simulation/run_simulation.py
```

This will:
- Create 5 federated learning clients
- Run 8 rounds of training
- Monitor energy consumption
- Generate performance reports and visualizations

### 2. Start the API Server

To run the REST API for the federated learning system:

```bash
python src/api/app.py
```

The API will be available at `http://localhost:8000`

#### Key Endpoints:
- `GET /health` - Health check
- `POST /api/v1/training/start` - Start federated learning
- `GET /api/v1/training/status` - Get training status
- `POST /api/v1/resources/allocate` - Allocate cloud resources
- `GET /api/v1/resources/status` - Get resource status

### 3. Run with Docker

For a complete environment with database and monitoring:

```bash
docker-compose up
```

This starts:
- Federated Learning API (port 8000)
- PostgreSQL database (port 5432)
- Redis cache (port 6379)
- Prometheus monitoring (port 9090)
- Grafana dashboards (port 3000)

### 4. Deploy to Azure

For cloud deployment:

```bash
# Install Azure Developer CLI
curl -fsSL https://aka.ms/install-azd.sh | bash

# Login to Azure
azd auth login

# Deploy
azd up
```

## Understanding the System

### Architecture Overview

The system consists of several key components:

1. **Federated Learning Core** (`src/federated_learning/`)
   - Implements FedAvg and energy-aware aggregation
   - Manages global and local model updates
   - Coordinates training across multiple clients

2. **Energy Monitor** (`src/energy_monitor/`)
   - Tracks CPU, memory, and GPU usage
   - Estimates power consumption
   - Provides energy efficiency metrics

3. **Resource Allocator** (`src/resource_allocator/`)
   - Intelligent cloud resource allocation
   - Energy-aware scheduling
   - Dynamic scaling capabilities

4. **API Layer** (`src/api/`)
   - REST API for system interaction
   - Training orchestration
   - Resource management endpoints

5. **Simulation Environment** (`src/simulation/`)
   - Comprehensive testing framework
   - Synthetic data generation
   - Performance visualization

### Configuration

The system is configured through `config/settings.yaml`. Key settings include:

- **Federated Learning**: Number of clients, rounds, aggregation method
- **Energy Monitoring**: Monitoring intervals, efficiency thresholds
- **Resource Allocation**: Min/max resources, scaling policies
- **Azure Settings**: Subscription, resource group, location

## Example Workflows

### Workflow 1: Basic Simulation

```bash
# Run a simple simulation
python src/simulation/run_simulation.py

# View results
open simulation_report.md
open simulation_results_*.png
```

### Workflow 2: Custom Training

```python
from src.federated_learning.core import FederatedModel, FederatedClient
from src.energy_monitor.monitor import EnergyMonitor

# Create model and clients
model = FederatedModel()
monitor = EnergyMonitor("client-1")

# Train with energy monitoring
monitor.start_monitoring()
# ... training code ...
energy_consumed = monitor.stop_monitoring()
```

### Workflow 3: Resource Management

```bash
# Start API server
python src/api/app.py

# Allocate resources (in another terminal)
curl -X POST http://localhost:8000/api/v1/resources/allocate \
  -H "Content-Type: application/json" \
  -d '{"client_id": "client-1", "cpu_cores": 4, "memory_gb": 8}'

# Check resource status
curl http://localhost:8000/api/v1/resources/status
```

## Monitoring and Observability

### Local Monitoring

- **Logs**: Check `logs/federated_learning.log`
- **Metrics**: Energy consumption data in simulation outputs
- **API**: Health endpoint at `/health`

### Production Monitoring (with Docker)

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Application Insights**: Available in Azure deployment

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes the `src` directory
2. **Port Conflicts**: Change ports in `config/settings.yaml` if needed
3. **Memory Issues**: Reduce batch size or model complexity in config
4. **Azure Deployment**: Ensure proper authentication with `azd auth login`

### Performance Optimization

1. **GPU Usage**: Enable GPU monitoring in config for CUDA-enabled systems
2. **Batch Size**: Adjust based on available memory
3. **Client Selection**: Use energy-aware client selection for efficiency
4. **Resource Allocation**: Configure appropriate min/max resource limits

## Next Steps

1. **Customize the Model**: Modify `FederatedModel` in `src/federated_learning/core.py`
2. **Add Your Data**: Replace synthetic data generation with real datasets
3. **Extend Energy Monitoring**: Add custom energy metrics for your hardware
4. **Scale to Production**: Deploy on Azure with multiple regions
5. **Integrate ML Ops**: Add model versioning and automated retraining

## Getting Help

- **Documentation**: See `docs/` directory for detailed documentation
- **Examples**: Check `examples/` for more code samples
- **Issues**: Report bugs or request features via GitHub issues
- **Community**: Join our discussions for questions and contributions

Happy federated learning! 🚀
