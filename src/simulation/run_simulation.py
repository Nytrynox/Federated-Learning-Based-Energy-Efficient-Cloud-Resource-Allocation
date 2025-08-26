"""
Simulation environment for testing federated learning with energy optimization.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import time
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import threading

"""
Simulation environment for testing federated learning with energy optimization.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import time
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import threading
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import from the actual modules, fallback to simplified versions
try:
    from src.config import config
    from src.federated_learning.core import FederatedModel, FederatedClient, FederatedLearningCoordinator
    from src.energy_monitor.monitor import EnergyMonitor
    from src.resource_allocator.allocator import ResourceAllocator, ResourceRequest, CloudNode
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("🔄 Using simplified standalone implementations...")
    
    # Simplified implementations for standalone execution
    class SimpleFederatedModel(nn.Module):
        def __init__(self, input_size: int = 784, hidden_size: int = 64, num_classes: int = 10):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.hidden_dim = hidden_size  # Add this for compatibility
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)
    
    class SimpleEnergyMonitor:
        def __init__(self):
            self.monitoring_data = {}
            self.start_times = {}
            
        def start_monitoring(self, client_id: str):
            self.start_times[client_id] = time.time()
            
        def stop_monitoring(self, client_id: str) -> float:
            if client_id in self.start_times:
                duration = time.time() - self.start_times[client_id]
                # Simulate realistic energy consumption (0.1-2.0 Wh)
                energy = duration * np.random.uniform(0.1, 2.0)
                self.monitoring_data[client_id] = energy
                return energy
            return 0.0
    
    class SimpleFederatedClient:
        def __init__(self, client_id: str, model: nn.Module, data_loader, energy_monitor):
            self.client_id = client_id
            self.model = model
            self.data_loader = data_loader
            self.energy_monitor = energy_monitor
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.CrossEntropyLoss()
            
        def train(self, global_model_state: Dict, epochs: int = 3) -> Dict:
            # Just return the current model state for compatibility
            return self.model.state_dict()
        
        def update_model(self, global_params: Dict):
            # Load parameters with size checking
            current_state = self.model.state_dict()
            for name, param in global_params.items():
                if name in current_state and param.shape == current_state[name].shape:
                    current_state[name].copy_(param)
            self.model.load_state_dict(current_state)
        
        def evaluate(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(test_data)
                loss = self.criterion(outputs, test_labels)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
            return {"accuracy": accuracy, "loss": loss.item()}
    
    class CloudNode:
        def __init__(self, node_id: str, cpu_cores: int, memory_gb: int, energy_efficiency: float):
            self.node_id = node_id
            self.cpu_cores = cpu_cores
            self.memory_gb = memory_gb
            self.energy_efficiency = energy_efficiency
            self.current_load = 0.0
    
    # Use simplified classes
    FederatedModel = SimpleFederatedModel
    FederatedClient = SimpleFederatedClient
    EnergyMonitor = SimpleEnergyMonitor
    
    # Simple config fallback
    config = {
        'federated_learning': {
            'num_clients': 5,
            'rounds': 10,
            'epochs_per_round': 3
        },
        'energy': {
            'monitoring_interval': 1.0
        }
    }


@dataclass
class SimulationConfig:
    """Configuration for simulation."""
    num_clients: int = 5
    num_rounds: int = 10
    simulation_duration: int = 3600  # seconds
    data_distribution: str = "iid"  # iid, non_iid
    network_latency_ms: int = 50
    failure_rate: float = 0.1
    energy_variation: float = 0.2  # Energy consumption variation between clients
    resource_heterogeneity: bool = True


@dataclass
class SimulationResults:
    """Results from simulation run."""
    config: SimulationConfig
    training_history: Dict
    energy_consumption: Dict[str, float]
    resource_utilization: Dict
    performance_metrics: Dict
    optimization_recommendations: List[str]
    simulation_time: float


class FederatedLearningSimulator:
    """Comprehensive simulation environment for federated learning."""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.logger = logging.getLogger("FederatedLearningSimulator")
        
        # Simulation components
        self.clients: List[FederatedClient] = []
        self.energy_monitors: Dict[str, EnergyMonitor] = {}
        self.resource_allocator = ResourceAllocator()
        self.fl_coordinator = None
        
        # Simulation state
        self.simulation_running = False
        self.simulation_start_time = None
        self.simulation_results = None
        
        # Setup simulation environment
        self._setup_simulation_environment()
    
    def _setup_simulation_environment(self):
        """Setup the simulation environment with synthetic data and clients."""
        self.logger.info("Setting up simulation environment")
        
        # Generate synthetic federated data
        client_data = self._generate_federated_data()
        
        # Create federated clients
        self.clients = []
        for i, (client_id, data_loader) in enumerate(client_data.items()):
            # Create model for client
            model = FederatedModel()
            
            # Create energy monitor with variation
            energy_monitor = EnergyMonitor(client_id=client_id)
            self.energy_monitors[client_id] = energy_monitor
            
            # Add some heterogeneity to clients if enabled
            if self.config.resource_heterogeneity:
                self._add_client_heterogeneity(model, i)
            
            client = FederatedClient(client_id, model, data_loader, energy_monitor)
            self.clients.append(client)
        
        # Create global model and coordinator
        global_model = FederatedModel()
        self.fl_coordinator = FederatedLearningCoordinator(global_model, self.clients)
        
        # Setup resource allocation for clients
        self._setup_resource_allocation()
        
        self.logger.info(f"Simulation environment setup complete with {len(self.clients)} clients")
    
    def _generate_federated_data(self) -> Dict[str, any]:
        """Generate synthetic federated learning datasets."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        client_data = {}
        
        if self.config.data_distribution == "iid":
            # IID distribution - each client gets random samples
            for i in range(self.config.num_clients):
                client_id = f"client-{i+1}"
                
                # Generate random data
                x = torch.randn(1000, 784)  # 1000 samples, 784 features
                y = torch.randint(0, 10, (1000,))  # 10 classes
                
                dataset = TensorDataset(x, y)
                data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
                client_data[client_id] = data_loader
        
        else:  # non_iid
            # Non-IID distribution - each client specializes in certain classes
            classes_per_client = max(1, 10 // self.config.num_clients)
            
            for i in range(self.config.num_clients):
                client_id = f"client-{i+1}"
                
                # Assign specific classes to this client
                start_class = (i * classes_per_client) % 10
                client_classes = [(start_class + j) % 10 for j in range(classes_per_client + 1)]
                
                # Generate data biased towards client's classes
                x_list = []
                y_list = []
                
                for _ in range(1000):
                    if random.random() < 0.8:  # 80% probability for client's classes
                        target_class = random.choice(client_classes)
                    else:  # 20% probability for other classes
                        target_class = random.randint(0, 9)
                    
                    # Generate feature vector with some class-specific bias
                    x = torch.randn(784) + 0.5 * target_class
                    y = target_class
                    
                    x_list.append(x)
                    y_list.append(y)
                
                x_tensor = torch.stack(x_list)
                y_tensor = torch.tensor(y_list)
                
                dataset = TensorDataset(x_tensor, y_tensor)
                data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
                client_data[client_id] = data_loader
        
        return client_data
    
    def _add_client_heterogeneity(self, model, client_index):
        """Add heterogeneity to client capabilities."""
        # Simulate different device capabilities
        heterogeneity_factors = [0.8, 1.0, 1.2, 0.9, 1.1]
        factor = heterogeneity_factors[client_index % len(heterogeneity_factors)]
        
        # Modify model architecture slightly based on factor
        if factor < 1.0:  # Lower capability device
            # Reduce hidden dimension
            original_hidden = model.hidden_dim
            model.hidden_dim = int(original_hidden * factor)
            model.fc2 = torch.nn.Linear(model.hidden_dim, model.hidden_dim)
    
    def _setup_resource_allocation(self):
        """Setup resource allocation for simulation clients."""
        for client in self.clients:
            # Create resource request for each client
            base_cpu = 2.0
            base_memory = 4.0
            
            # Add some variation
            cpu_variation = random.uniform(0.8, 1.5)
            memory_variation = random.uniform(0.8, 1.5)
            
            request = ResourceRequest(
                client_id=client.client_id,
                cpu_cores=base_cpu * cpu_variation,
                memory_gb=base_memory * memory_variation,
                gpu_count=random.choice([0, 1]),
                priority=random.randint(1, 5)
            )
            
            # Allocate resources
            allocation = self.resource_allocator.allocate_resources(request)
            if allocation:
                self.logger.debug(f"Allocated resources for {client.client_id}: {allocation.node_id}")
            else:
                self.logger.warning(f"Failed to allocate resources for {client.client_id}")
    
    def run_simulation(self) -> SimulationResults:
        """Run the complete simulation."""
        self.logger.info("Starting federated learning simulation")
        self.simulation_running = True
        self.simulation_start_time = time.time()
        
        try:
            # Start energy monitoring for all clients
            for client_id, monitor in self.energy_monitors.items():
                monitor.start_monitoring()
            
            # Simulate network delays and failures
            self._simulate_network_conditions()
            
            # Run federated learning
            training_history = self.fl_coordinator.run_federated_learning()
            
            # Stop energy monitoring and collect results
            energy_consumption = {}
            for client_id, monitor in self.energy_monitors.items():
                energy_consumption[client_id] = monitor.stop_monitoring()
            
            # Collect resource utilization statistics
            resource_utilization = self.resource_allocator.get_cluster_status()
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(training_history)
            
            # Get optimization recommendations
            optimization_recommendations = self.resource_allocator.optimize_allocations()
            
            # Calculate simulation time
            simulation_time = time.time() - self.simulation_start_time
            
            # Create results object
            results = SimulationResults(
                config=self.config,
                training_history=training_history,
                energy_consumption=energy_consumption,
                resource_utilization=resource_utilization,
                performance_metrics=performance_metrics,
                optimization_recommendations=optimization_recommendations,
                simulation_time=simulation_time
            )
            
            self.simulation_results = results
            self.logger.info(f"Simulation completed in {simulation_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise
        finally:
            self.simulation_running = False
    
    def _simulate_network_conditions(self):
        """Simulate network latency and failures."""
        # This is a simplified simulation
        # In a real implementation, you would introduce actual delays and failures
        if self.config.network_latency_ms > 0:
            self.logger.info(f"Simulating network latency: {self.config.network_latency_ms}ms")
        
        if self.config.failure_rate > 0:
            num_failures = int(len(self.clients) * self.config.failure_rate)
            self.logger.info(f"Simulating {num_failures} client failures")
    
    def _calculate_performance_metrics(self, training_history: Dict) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not training_history or 'global_accuracy' not in training_history:
            return {}
        
        accuracies = training_history['global_accuracy']
        losses = training_history['global_loss']
        energy_per_round = training_history['energy_consumption']
        
        metrics = {
            'final_accuracy': accuracies[-1] if accuracies else 0,
            'max_accuracy': max(accuracies) if accuracies else 0,
            'accuracy_improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0,
            'final_loss': losses[-1] if losses else 0,
            'min_loss': min(losses) if losses else 0,
            'total_energy_consumption': sum(energy_per_round) if energy_per_round else 0,
            'average_energy_per_round': np.mean(energy_per_round) if energy_per_round else 0,
            'energy_efficiency': (accuracies[-1] / sum(energy_per_round)) if (accuracies and energy_per_round and sum(energy_per_round) > 0) else 0,
            'convergence_round': self._find_convergence_round(accuracies),
            'training_stability': np.std(accuracies[-5:]) if len(accuracies) >= 5 else 0
        }
        
        return metrics
    
    def _find_convergence_round(self, accuracies: List[float]) -> int:
        """Find the round where the model converged (accuracy stops improving significantly)."""
        if len(accuracies) < 3:
            return len(accuracies)
        
        for i in range(2, len(accuracies)):
            # Check if improvement in last 2 rounds is less than 1%
            recent_improvement = accuracies[i] - accuracies[i-2]
            if recent_improvement < 1.0:  # Less than 1% improvement
                return i + 1
        
        return len(accuracies)
    
    def generate_report(self, save_path: str = None) -> str:
        """Generate a comprehensive simulation report."""
        if not self.simulation_results:
            return "No simulation results available"
        
        results = self.simulation_results
        
        report = f"""
# Federated Learning Simulation Report

## Simulation Configuration
- Number of Clients: {results.config.num_clients}
- Number of Rounds: {results.config.num_rounds}
- Data Distribution: {results.config.data_distribution}
- Network Latency: {results.config.network_latency_ms}ms
- Failure Rate: {results.config.failure_rate * 100:.1f}%
- Simulation Duration: {results.simulation_time:.2f} seconds

## Performance Results
- Final Accuracy: {results.performance_metrics.get('final_accuracy', 0):.2f}%
- Maximum Accuracy: {results.performance_metrics.get('max_accuracy', 0):.2f}%
- Accuracy Improvement: {results.performance_metrics.get('accuracy_improvement', 0):.2f}%
- Final Loss: {results.performance_metrics.get('final_loss', 0):.4f}
- Convergence Round: {results.performance_metrics.get('convergence_round', 0)}

## Energy Efficiency
- Total Energy Consumption: {results.performance_metrics.get('total_energy_consumption', 0):.2f} Wh
- Average Energy per Round: {results.performance_metrics.get('average_energy_per_round', 0):.2f} Wh
- Energy Efficiency Score: {results.performance_metrics.get('energy_efficiency', 0):.4f}

## Resource Utilization
- CPU Utilization: {results.resource_utilization.get('cpu_utilization', 0) * 100:.1f}%
- Memory Utilization: {results.resource_utilization.get('memory_utilization', 0) * 100:.1f}%
- Active Allocations: {results.resource_utilization.get('active_allocations', 0)}
- Average Energy Efficiency: {results.resource_utilization.get('average_energy_efficiency', 0):.2f}

## Energy Consumption by Client
"""
        
        for client_id, energy in results.energy_consumption.items():
            report += f"- {client_id}: {energy:.2f} Wh\n"
        
        report += f"""
## Optimization Recommendations
"""
        
        if results.optimization_recommendations:
            for i, recommendation in enumerate(results.optimization_recommendations, 1):
                report += f"{i}. {recommendation}\n"
        else:
            report += "No optimization recommendations available.\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {save_path}")
        
        return report
    
    def visualize_results(self, save_plots: bool = True):
        """Create visualizations of simulation results."""
        if not self.simulation_results:
            self.logger.warning("No simulation results to visualize")
            return
        
        results = self.simulation_results
        history = results.training_history
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Training accuracy over rounds
        if 'global_accuracy' in history:
            axes[0, 0].plot(history['rounds'], history['global_accuracy'], marker='o')
            axes[0, 0].set_title('Global Model Accuracy')
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].grid(True)
        
        # Plot 2: Training loss over rounds
        if 'global_loss' in history:
            axes[0, 1].plot(history['rounds'], history['global_loss'], marker='s', color='red')
            axes[0, 1].set_title('Global Model Loss')
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # Plot 3: Energy consumption per round
        if 'energy_consumption' in history:
            axes[0, 2].bar(history['rounds'], history['energy_consumption'])
            axes[0, 2].set_title('Energy Consumption per Round')
            axes[0, 2].set_xlabel('Round')
            axes[0, 2].set_ylabel('Energy (Wh)')
            axes[0, 2].grid(True)
        
        # Plot 4: Energy consumption by client
        client_ids = list(results.energy_consumption.keys())
        energy_values = list(results.energy_consumption.values())
        axes[1, 0].bar(client_ids, energy_values)
        axes[1, 0].set_title('Total Energy Consumption by Client')
        axes[1, 0].set_xlabel('Client')
        axes[1, 0].set_ylabel('Energy (Wh)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Resource utilization
        utilization_data = results.resource_utilization
        categories = ['CPU', 'Memory']
        values = [
            utilization_data.get('cpu_utilization', 0) * 100,
            utilization_data.get('memory_utilization', 0) * 100
        ]
        axes[1, 1].bar(categories, values)
        axes[1, 1].set_title('Resource Utilization')
        axes[1, 1].set_ylabel('Utilization (%)')
        axes[1, 1].set_ylim(0, 100)
        
        # Plot 6: Performance vs Energy Trade-off
        if 'global_accuracy' in history and 'energy_consumption' in history:
            axes[1, 2].scatter(history['energy_consumption'], history['global_accuracy'])
            axes[1, 2].set_title('Accuracy vs Energy Consumption')
            axes[1, 2].set_xlabel('Energy Consumption (Wh)')
            axes[1, 2].set_ylabel('Accuracy (%)')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plots saved to {filename}")
        
        plt.show()
    
    def export_results(self, filepath: str):
        """Export simulation results to JSON file."""
        if not self.simulation_results:
            self.logger.warning("No simulation results to export")
            return
        
        # Convert results to dictionary
        results_dict = asdict(self.simulation_results)
        
        # Make sure all values are JSON serializable
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.number):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = make_serializable(results_dict)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results exported to {filepath}")


def run_simulation_example():
    """Run an example simulation."""
    # Create simulation configuration
    sim_config = SimulationConfig(
        num_clients=5,
        num_rounds=8,
        data_distribution="non_iid",
        network_latency_ms=100,
        failure_rate=0.1
    )
    
    # Create and run simulator
    simulator = FederatedLearningSimulator(sim_config)
    results = simulator.run_simulation()
    
    # Generate report
    report = simulator.generate_report("simulation_report.md")
    print(report)
    
    # Create visualizations
    simulator.visualize_results()
    
    # Export results
    simulator.export_results("simulation_results.json")
    
    return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run simulation
    run_simulation_example()
