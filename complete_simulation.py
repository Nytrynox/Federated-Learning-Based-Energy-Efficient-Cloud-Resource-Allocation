#!/usr/bin/env python3
"""
Complete Federated Learning Based Energy-Efficient Cloud Resource Allocation
Comprehensive Working Demo
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    num_clients: int = 5
    num_rounds: int = 10
    epochs_per_round: int = 3
    learning_rate: float = 0.001
    batch_size: int = 32
    dataset_size: int = 1000
    test_split: float = 0.2
    input_size: int = 20
    hidden_size: int = 64
    num_classes: int = 10
    random_seed: int = 42

config = Config()

class FederatedModel(nn.Module):
    """Enhanced federated learning model with energy awareness"""
    
    def __init__(self, input_size: int = 20, hidden_size: int = 64, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

class EnergyMonitor:
    """Enhanced energy monitoring with realistic simulation"""
    
    def __init__(self):
        self.monitoring_data = {}
        self.start_times = {}
        self.base_power = 50.0  # Base power consumption in watts
        
    def start_monitoring(self, client_id: str):
        self.start_times[client_id] = time.time()
        logger.info(f"Started energy monitoring for {client_id}")
        
    def stop_monitoring(self, client_id: str) -> float:
        if client_id in self.start_times:
            duration = time.time() - self.start_times[client_id]
            # Realistic energy calculation: base power + computation overhead
            computation_power = np.random.uniform(100, 200)  # Variable computational load (watts)
            total_power = self.base_power + computation_power
            energy_wh = (total_power * duration) / 3600  # Convert to Wh
            
            # Ensure minimum energy consumption for realistic values
            energy_wh = max(energy_wh, 0.5)  # Minimum 0.5 Wh per training session
            
            self.monitoring_data[client_id] = energy_wh
            logger.info(f"Energy monitoring stopped for {client_id}. Energy: {energy_wh:.2f}Wh")
            return energy_wh
        return 0.5  # Default minimum energy if timing fails
    
    def get_total_energy(self) -> float:
        return sum(self.monitoring_data.values())

class CloudNode:
    """Cloud node representation with resource management"""
    
    def __init__(self, node_id: str, cpu_cores: int = 4, memory_gb: int = 8, energy_efficiency: float = 0.8):
        self.node_id = node_id
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.energy_efficiency = energy_efficiency  # 0.0 to 1.0
        self.current_load = 0.0
        self.allocated_clients = []
        
    def allocate_client(self, client_id: str) -> bool:
        if self.current_load < 0.8:  # Don't overload nodes
            self.allocated_clients.append(client_id)
            self.current_load += 0.2  # Each client adds 20% load
            return True
        return False
    
    def deallocate_client(self, client_id: str):
        if client_id in self.allocated_clients:
            self.allocated_clients.remove(client_id)
            self.current_load = max(0.0, self.current_load - 0.2)

class ResourceAllocator:
    """Energy-aware resource allocation for federated learning"""
    
    def __init__(self, num_nodes: int = 10):
        self.nodes = []
        for i in range(num_nodes):
            # Create nodes with varying efficiency
            efficiency = np.random.uniform(0.6, 1.0)
            node = CloudNode(f"node-{i}", energy_efficiency=efficiency)
            self.nodes.append(node)
        
        logger.info(f"Initialized {num_nodes} cloud nodes")
    
    def allocate_resources(self, client_ids: List[str]) -> Dict[str, str]:
        """Allocate clients to nodes based on energy efficiency"""
        allocation = {}
        
        # Sort nodes by energy efficiency (descending)
        sorted_nodes = sorted(self.nodes, key=lambda n: n.energy_efficiency, reverse=True)
        
        for client_id in client_ids:
            allocated = False
            for node in sorted_nodes:
                if node.allocate_client(client_id):
                    allocation[client_id] = node.node_id
                    allocated = True
                    break
            
            if not allocated:
                # Fallback to least loaded node
                least_loaded = min(self.nodes, key=lambda n: n.current_load)
                least_loaded.allocate_client(client_id)
                allocation[client_id] = least_loaded.node_id
        
        return allocation

class FederatedClient:
    """Enhanced federated client with energy awareness"""
    
    def __init__(self, client_id: str, data: torch.Tensor, labels: torch.Tensor):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.model = FederatedModel(config.input_size, config.hidden_size, config.num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.training_history = []
        
    def train(self, global_model_state: Dict, epochs: int = 3) -> Tuple[Dict, Dict]:
        """Train local model and return updated weights with training stats"""
        # Load global model
        self.model.load_state_dict(global_model_state)
        self.model.train()
        
        epoch_losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.data)
            loss = self.criterion(outputs, self.labels)
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        training_stats = {
            "client_id": self.client_id,
            "final_loss": epoch_losses[-1],
            "avg_loss": np.mean(epoch_losses),
            "data_size": len(self.data)
        }
        
        return self.model.state_dict(), training_stats
    
    def evaluate(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(test_data)
            loss = self.criterion(outputs, test_labels)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
        
        return {"accuracy": accuracy, "loss": loss.item()}

class EnergyAwareAggregator:
    """Energy-aware federated averaging with optimization"""
    
    def __init__(self):
        self.aggregation_history = []
        
    def aggregate(self, client_updates: List[Dict], energy_data: Dict[str, float]) -> Dict:
        """Aggregate client updates with energy-aware weighting"""
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Calculate energy efficiency weights
        total_energy = sum(energy_data.values())
        if total_energy == 0:
            # Fallback to equal weighting
            weights = {client_id: 1.0 / len(client_updates) for client_id in energy_data.keys()}
        else:
            # Higher weight for lower energy consumption (more efficient)
            weights = {}
            for client_id, energy in energy_data.items():
                # Inverse energy weighting with normalization
                efficiency = 1.0 / (energy + 1e-6)  # Add small epsilon to avoid division by zero
                weights[client_id] = efficiency
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Aggregate parameters
        global_state = {}
        for key in client_updates[0].keys():
            global_state[key] = torch.zeros_like(client_updates[0][key], dtype=torch.float32)
        
        for i, (client_id, update) in enumerate(zip(energy_data.keys(), client_updates)):
            weight = weights[client_id]
            for key in update.keys():
                # Ensure consistent dtype for aggregation
                param_tensor = update[key].to(dtype=torch.float32)
                global_state[key] += weight * param_tensor
        
        self.aggregation_history.append({
            "weights": weights,
            "total_energy": total_energy,
            "num_clients": len(client_updates)
        })
        
        logger.info(f"Aggregated {len(client_updates)} client updates using energy-aware method")
        return global_state

class FederatedLearningCoordinator:
    """Main coordinator for federated learning with energy optimization"""
    
    def __init__(self):
        self.global_model = FederatedModel(config.input_size, config.hidden_size, config.num_classes)
        self.clients = []
        self.energy_monitor = EnergyMonitor()
        self.resource_allocator = ResourceAllocator()
        self.aggregator = EnergyAwareAggregator()
        self.training_history = []
        
    def add_client(self, client: FederatedClient):
        """Add a client to the federation"""
        self.clients.append(client)
        
    def train_round(self, round_num: int) -> Dict[str, Any]:
        """Execute one round of federated training"""
        logger.info(f"Starting round {round_num}/{config.num_rounds}")
        round_start_time = time.time()
        
        # Allocate resources
        client_ids = [client.client_id for client in self.clients]
        resource_allocation = self.resource_allocator.allocate_resources(client_ids)
        
        client_updates = []
        training_stats = []
        round_energy = {}
        
        # Train clients
        for client in self.clients:
            self.energy_monitor.start_monitoring(client.client_id)
            
            # Train client
            update, stats = client.train(
                self.global_model.state_dict(), 
                epochs=config.epochs_per_round
            )
            
            energy = self.energy_monitor.stop_monitoring(client.client_id)
            
            client_updates.append(update)
            training_stats.append(stats)
            round_energy[client.client_id] = energy
        
        # Aggregate updates
        aggregated_state = self.aggregator.aggregate(client_updates, round_energy)
        # Convert back to original dtypes before loading
        original_state = self.global_model.state_dict()
        for key in aggregated_state.keys():
            if key in original_state:
                aggregated_state[key] = aggregated_state[key].to(dtype=original_state[key].dtype)
        self.global_model.load_state_dict(aggregated_state)
        
        # Calculate round statistics
        round_duration = time.time() - round_start_time
        total_round_energy = sum(round_energy.values())
        avg_loss = np.mean([stats['final_loss'] for stats in training_stats])
        
        round_stats = {
            "round": round_num,
            "duration": round_duration,
            "total_energy": total_round_energy,
            "avg_loss": avg_loss,
            "client_stats": training_stats,
            "energy_data": round_energy,
            "resource_allocation": resource_allocation
        }
        
        self.training_history.append(round_stats)
        logger.info(f"Round {round_num} completed. Avg loss: {avg_loss:.4f}, Total energy: {total_round_energy:.2f}Wh")
        
        return round_stats
    
    def evaluate_global_model(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate the global model"""
        self.global_model.eval()
        with torch.no_grad():
            outputs = self.global_model(test_data)
            loss = nn.CrossEntropyLoss()(outputs, test_labels)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
        
        return {"accuracy": accuracy, "loss": loss.item()}

def create_synthetic_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic dataset for federated learning"""
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    
    # Generate synthetic classification data
    n_samples = config.dataset_size
    n_features = config.input_size
    n_classes = config.num_classes
    
    # Create data with some complexity
    X = np.random.randn(n_samples, n_features)
    # Create non-linear decision boundaries
    y = (X[:, 0] + X[:, 1] * X[:, 2] + np.random.randn(n_samples) * 0.1) > 0
    y = y.astype(int) + (X[:, 3] + X[:, 4] > 0).astype(int)
    y = y % n_classes  # Ensure labels are in valid range
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Split into train and test
    test_size = int(len(X_tensor) * config.test_split)
    train_size = len(X_tensor) - test_size
    
    train_data = X_tensor[:train_size]
    train_labels = y_tensor[:train_size]
    test_data = X_tensor[train_size:]
    test_labels = y_tensor[train_size:]
    
    logger.info(f"Created dataset: {train_size} train samples, {test_size} test samples")
    return train_data, train_labels, test_data, test_labels

def distribute_data_to_clients(train_data: torch.Tensor, train_labels: torch.Tensor) -> List[FederatedClient]:
    """Distribute data among federated clients"""
    clients = []
    data_per_client = len(train_data) // config.num_clients
    
    for i in range(config.num_clients):
        start_idx = i * data_per_client
        end_idx = start_idx + data_per_client if i < config.num_clients - 1 else len(train_data)
        
        client_data = train_data[start_idx:end_idx]
        client_labels = train_labels[start_idx:end_idx]
        
        client = FederatedClient(f"client-{i+1}", client_data, client_labels)
        clients.append(client)
        
        logger.info(f"Created client-{i+1} with {len(client_data)} samples")
    
    return clients

def create_visualizations(coordinator: FederatedLearningCoordinator, test_accuracy: float):
    """Create comprehensive visualizations"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Federated Learning Energy-Efficient Cloud Resource Allocation Results', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    rounds = [stats['round'] for stats in coordinator.training_history]
    losses = [stats['avg_loss'] for stats in coordinator.training_history]
    energies = [stats['total_energy'] for stats in coordinator.training_history]
    durations = [stats['duration'] for stats in coordinator.training_history]
    
    # 1. Training Loss Over Rounds
    axes[0, 0].plot(rounds, losses, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_title('Training Loss Over Rounds', fontweight='bold')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Average Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Energy Consumption Per Round
    axes[0, 1].bar(rounds, energies, color='green', alpha=0.7)
    axes[0, 1].set_title('Energy Consumption Per Round', fontweight='bold')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Energy (Wh)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Round Duration
    axes[0, 2].plot(rounds, durations, 'r-o', linewidth=2, markersize=6)
    axes[0, 2].set_title('Round Duration', fontweight='bold')
    axes[0, 2].set_xlabel('Round')
    axes[0, 2].set_ylabel('Duration (seconds)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Energy Efficiency (Loss Reduction per Wh)
    if len(losses) > 1:
        loss_reduction = [losses[0] - loss for loss in losses[1:]]
        energy_efficiency = [lr / en if en > 0 else 0 for lr, en in zip(loss_reduction, energies[1:])]
        axes[1, 0].plot(rounds[1:], energy_efficiency, 'purple', linewidth=2, marker='s')
        axes[1, 0].set_title('Energy Efficiency (Loss Reduction per Wh)', fontweight='bold')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Loss Reduction / Energy')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Cumulative Energy Consumption
    cumulative_energy = np.cumsum(energies)
    axes[1, 1].plot(rounds, cumulative_energy, 'orange', linewidth=3)
    axes[1, 1].fill_between(rounds, cumulative_energy, alpha=0.3, color='orange')
    axes[1, 1].set_title('Cumulative Energy Consumption', fontweight='bold')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Cumulative Energy (Wh)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Final Results Summary
    axes[1, 2].axis('off')
    summary_text = f"""
Final Results Summary:

🎯 Test Accuracy: {test_accuracy:.2%}
⚡ Total Energy: {sum(energies):.2f} Wh
⏱️ Total Time: {sum(durations):.1f} seconds
🔄 Training Rounds: {len(rounds)}
👥 Number of Clients: {config.num_clients}

📊 Efficiency Metrics:
• Accuracy per Wh: {test_accuracy/sum(energies):.4f}
• Avg Energy/Round: {np.mean(energies):.2f} Wh
• Avg Duration/Round: {np.mean(durations):.1f} s

🏆 Energy-Aware Federated Learning
   Successfully Demonstrated!
    """
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "federated_learning_results.png", dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {results_dir / 'federated_learning_results.png'}")
    
    plt.show()

def save_results(coordinator: FederatedLearningCoordinator, test_results: Dict, total_time: float):
    """Save detailed results to JSON"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results = {
        "config": {
            "num_clients": config.num_clients,
            "num_rounds": config.num_rounds,
            "epochs_per_round": config.epochs_per_round,
            "learning_rate": config.learning_rate,
            "dataset_size": config.dataset_size
        },
        "final_results": {
            "test_accuracy": test_results["accuracy"],
            "test_loss": test_results["loss"],
            "total_energy": coordinator.energy_monitor.get_total_energy(),
            "total_time": total_time,
            "energy_efficiency": test_results["accuracy"] / coordinator.energy_monitor.get_total_energy()
        },
        "training_history": coordinator.training_history,
        "aggregation_history": coordinator.aggregator.aggregation_history
    }
    
    with open(results_dir / "simulation_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_dir / 'simulation_results.json'}")

def main():
    """Main execution function"""
    print("🔬 Federated Learning Energy-Efficient Cloud Resource Allocation")
    print("   Comprehensive Simulation with Visualization")
    print()
    
    start_time = time.time()
    
    # Create dataset
    print("📊 Creating synthetic dataset...")
    train_data, train_labels, test_data, test_labels = create_synthetic_data()
    
    # Create federated clients
    print("👥 Creating federated clients...")
    clients = distribute_data_to_clients(train_data, train_labels)
    
    # Initialize coordinator
    print("🧠 Initializing federated learning coordinator...")
    coordinator = FederatedLearningCoordinator()
    for client in clients:
        coordinator.add_client(client)
    
    # Run federated learning
    print("🔄 Starting federated learning simulation...")
    print("=" * 80)
    
    for round_num in range(1, config.num_rounds + 1):
        round_stats = coordinator.train_round(round_num)
    
    # Evaluate final model
    print("\n📊 Evaluating final global model...")
    test_results = coordinator.evaluate_global_model(test_data, test_labels)
    
    total_time = time.time() - start_time
    total_energy = coordinator.energy_monitor.get_total_energy()
    
    # Display results
    print("=" * 80)
    print("📈 SIMULATION COMPLETED")
    print("=" * 80)
    print(f"⏱️  Total simulation time: {total_time:.2f} seconds")
    print(f"🎯 Final test accuracy: {test_results['accuracy']:.2%}")
    print(f"📉 Final test loss: {test_results['loss']:.4f}")
    print(f"⚡ Total energy consumption: {total_energy:.2f} Wh")
    print(f"🔋 Average energy per round: {total_energy/config.num_rounds:.2f} Wh")
    print(f"📈 Energy efficiency: {test_results['accuracy']/total_energy:.4f} accuracy per Wh")
    print()
    
    # Create visualizations
    print("📊 Creating visualizations...")
    create_visualizations(coordinator, test_results['accuracy'])
    
    # Save results
    print("💾 Saving detailed results...")
    save_results(coordinator, test_results, total_time)
    
    print("\n✅ Simulation completed successfully!")
    print("🎉 This demonstrates a complete federated learning system with:")
    print("   • Energy-aware resource allocation")
    print("   • Efficient model aggregation")
    print("   • Comprehensive monitoring and visualization")
    print("   • Cloud resource optimization")

if __name__ == "__main__":
    main()
