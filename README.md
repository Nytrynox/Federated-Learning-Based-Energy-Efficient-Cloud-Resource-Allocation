# Federated Learning Based Energy-Efficient Cloud Resource Allocation

## Overview

This project implements a federated learning system that optimizes cloud resource allocation while minimizing energy consumption. The system uses distributed machine learning to make intelligent decisions about resource provisioning across multiple cloud nodes.

## Features

- **Federated Learning Framework**: Decentralized model training across multiple clients
- **Energy Efficiency**: Smart algorithms to minimize power consumption
- **Dynamic Resource Allocation**: Real-time allocation based on demand and energy metrics
- **Azure Cloud Integration**: Seamless deployment on Microsoft Azure
- **Performance Monitoring**: Real-time tracking of energy and performance metrics
- **Simulation Environment**: Test different scenarios and configurations

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Node   │    │   Client Node   │    │   Client Node   │
│                 │    │                 │    │                 │
│ Local Model     │    │ Local Model     │    │ Local Model     │
│ Energy Monitor  │    │ Energy Monitor  │    │ Energy Monitor  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │    Federated Server       │
                    │                           │
                    │ Global Model Aggregation  │
                    │ Resource Allocation       │
                    │ Energy Optimization       │
                    └───────────────────────────┘
```

## Technology Stack

- **Python 3.9+**: Core programming language
- **PyTorch**: Deep learning framework for federated learning
- **Flask/FastAPI**: REST API for communication
- **Azure ML**: Machine learning platform
- **Azure Functions**: Serverless computing
- **Azure Container Apps**: Containerized deployment
- **Redis**: In-memory data structure store
- **PostgreSQL**: Database for metrics and logs
- **Docker**: Containerization
- **Prometheus/Grafana**: Monitoring and visualization

## Project Structure

```
ccs/
├── src/
│   ├── federated_learning/       # Core FL implementation
│   ├── energy_monitor/           # Energy tracking and optimization
│   ├── resource_allocator/       # Cloud resource management
│   ├── api/                      # REST API endpoints
│   └── simulation/               # Testing and simulation
├── infra/                        # Azure infrastructure (Bicep)
├── docker/                       # Docker configurations
├── tests/                        # Unit and integration tests
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
└── requirements.txt              # Python dependencies
```

## Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd ccs
   pip install -r requirements.txt
   ```

2. **Run Simulation**:
   ```bash
   python src/simulation/run_simulation.py
   ```

3. **Deploy to Azure**:
   ```bash
   azd up
   ```

## Configuration

The system can be configured through `config/settings.yaml` for various parameters including:
- Number of federated clients
- Energy efficiency thresholds
- Resource allocation policies
- Azure service configurations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details
