"""
Core federated learning implementation with energy-aware optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import numpy as np
import copy
import logging
import sys
import os
from abc import ABC, abstractmethod

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import config
except ImportError:
    # Fallback configuration if config module not available
    class MockConfig:
        @property
        def federated_learning(self):
            return {
                'learning_rate': 0.01,
                'local_epochs': 5,
                'num_rounds': 10,
                'aggregation_method': 'fedavg'
            }
    config = MockConfig()


class FederatedModel(nn.Module):
    """Base neural network model for federated learning."""
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, output_dim: int = 10):
        super(FederatedModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class FederatedClient:
    """Federated learning client with energy monitoring."""
    
    def __init__(self, client_id: str, model: nn.Module, data_loader, energy_monitor=None):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.energy_monitor = energy_monitor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Get training parameters from config
        fl_config = config.federated_learning
        self.learning_rate = fl_config.get('learning_rate', 0.01)
        self.local_epochs = fl_config.get('local_epochs', 5)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.logger = logging.getLogger(f"FederatedClient-{client_id}")
    
    def train_local_model(self) -> Tuple[Dict, float]:
        """Train the local model and return updates with energy consumption."""
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        # Start energy monitoring
        if self.energy_monitor:
            self.energy_monitor.start_monitoring()
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_samples += len(data)
            
            total_loss += epoch_loss
            self.logger.info(f"Client {self.client_id} - Epoch {epoch + 1}/{self.local_epochs}, Loss: {epoch_loss:.4f}")
        
        # Stop energy monitoring and get consumption
        energy_consumed = 0.0
        if self.energy_monitor:
            energy_consumed = self.energy_monitor.stop_monitoring()
        
        # Get model parameters
        model_params = {name: param.cpu().clone() for name, param in self.model.named_parameters()}
        
        avg_loss = total_loss / (self.local_epochs * len(self.data_loader))
        
        return {
            'model_params': model_params,
            'num_samples': num_samples,
            'loss': avg_loss,
            'client_id': self.client_id
        }, energy_consumed
    
    def update_model(self, global_params: Dict):
        """Update local model with global parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_params:
                    param.copy_(global_params[name])
    
    def evaluate_model(self) -> Dict:
        """Evaluate the local model."""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.data_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }


class FederatedAggregator:
    """Federated learning aggregator with energy-aware optimization."""
    
    def __init__(self, model: nn.Module):
        self.global_model = model
        self.aggregation_method = config.federated_learning.get('aggregation_method', 'fedavg')
        self.logger = logging.getLogger("FederatedAggregator")
    
    def aggregate_models(self, client_updates: List[Dict], energy_consumptions: List[float]) -> Dict:
        """Aggregate client models with energy-aware weighting."""
        if self.aggregation_method == 'fedavg':
            return self._fed_avg_aggregation(client_updates)
        elif self.aggregation_method == 'energy_aware':
            return self._energy_aware_aggregation(client_updates, energy_consumptions)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _fed_avg_aggregation(self, client_updates: List[Dict]) -> Dict:
        """Standard FedAvg aggregation."""
        if not client_updates:
            return {}
        
        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        first_update = client_updates[0]
        
        for param_name in first_update['model_params'].keys():
            aggregated_params[param_name] = torch.zeros_like(
                first_update['model_params'][param_name]
            )
        
        # Weighted aggregation
        for update in client_updates:
            weight = update['num_samples'] / total_samples
            for param_name, param_value in update['model_params'].items():
                aggregated_params[param_name] += weight * param_value
        
        self.logger.info(f"Aggregated {len(client_updates)} client updates using FedAvg")
        return aggregated_params
    
    def _energy_aware_aggregation(self, client_updates: List[Dict], energy_consumptions: List[float]) -> Dict:
        """Energy-aware aggregation that considers both sample size and energy efficiency."""
        if not client_updates or not energy_consumptions:
            return self._fed_avg_aggregation(client_updates)
        
        # Calculate energy efficiency scores (lower energy consumption = higher efficiency)
        max_energy = max(energy_consumptions) if energy_consumptions else 1.0
        energy_efficiency_scores = [
            (max_energy - energy + 1e-6) / max_energy for energy in energy_consumptions
        ]
        
        # Calculate combined weights (sample size + energy efficiency)
        total_samples = sum(update['num_samples'] for update in client_updates)
        sample_weights = [update['num_samples'] / total_samples for update in client_updates]
        
        # Combine sample weights with energy efficiency
        combined_weights = []
        for i, update in enumerate(client_updates):
            sample_weight = sample_weights[i]
            energy_weight = energy_efficiency_scores[i]
            combined_weight = 0.7 * sample_weight + 0.3 * energy_weight  # Configurable ratio
            combined_weights.append(combined_weight)
        
        # Normalize weights
        total_weight = sum(combined_weights)
        normalized_weights = [w / total_weight for w in combined_weights]
        
        # Initialize aggregated parameters
        aggregated_params = {}
        first_update = client_updates[0]
        
        for param_name in first_update['model_params'].keys():
            aggregated_params[param_name] = torch.zeros_like(
                first_update['model_params'][param_name]
            )
        
        # Weighted aggregation with energy awareness
        for i, update in enumerate(client_updates):
            weight = normalized_weights[i]
            for param_name, param_value in update['model_params'].items():
                aggregated_params[param_name] += weight * param_value
        
        self.logger.info(
            f"Aggregated {len(client_updates)} client updates using energy-aware method. "
            f"Energy consumptions: {energy_consumptions}"
        )
        return aggregated_params
    
    def update_global_model(self, aggregated_params: Dict):
        """Update the global model with aggregated parameters."""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_params:
                    param.copy_(aggregated_params[name])


class FederatedLearningCoordinator:
    """Main coordinator for federated learning with energy optimization."""
    
    def __init__(self, global_model: nn.Module, clients: List[FederatedClient]):
        self.global_model = global_model
        self.clients = clients
        self.aggregator = FederatedAggregator(global_model)
        self.round_number = 0
        
        self.num_rounds = config.federated_learning.get('num_rounds', 10)
        self.logger = logging.getLogger("FederatedLearningCoordinator")
    
    def run_federated_learning(self) -> Dict:
        """Run the complete federated learning process."""
        self.logger.info(f"Starting federated learning with {len(self.clients)} clients for {self.num_rounds} rounds")
        
        training_history = {
            'rounds': [],
            'global_loss': [],
            'global_accuracy': [],
            'energy_consumption': [],
            'client_participation': []
        }
        
        for round_num in range(self.num_rounds):
            self.round_number = round_num + 1
            self.logger.info(f"Starting round {self.round_number}/{self.num_rounds}")
            
            # Select clients for this round (could be energy-based selection)
            selected_clients = self._select_clients()
            
            # Distribute global model to clients
            global_params = {name: param.cpu().clone() for name, param in self.global_model.named_parameters()}
            for client in selected_clients:
                client.update_model(global_params)
            
            # Local training
            client_updates = []
            energy_consumptions = []
            
            for client in selected_clients:
                self.logger.info(f"Training client {client.client_id}")
                update, energy = client.train_local_model()
                client_updates.append(update)
                energy_consumptions.append(energy)
            
            # Aggregate models
            aggregated_params = self.aggregator.aggregate_models(client_updates, energy_consumptions)
            self.aggregator.update_global_model(aggregated_params)
            
            # Evaluate global model
            global_metrics = self._evaluate_global_model()
            
            # Record training history
            training_history['rounds'].append(self.round_number)
            training_history['global_loss'].append(global_metrics['loss'])
            training_history['global_accuracy'].append(global_metrics['accuracy'])
            training_history['energy_consumption'].append(sum(energy_consumptions))
            training_history['client_participation'].append([c.client_id for c in selected_clients])
            
            self.logger.info(
                f"Round {self.round_number} completed. "
                f"Global accuracy: {global_metrics['accuracy']:.2f}%, "
                f"Global loss: {global_metrics['loss']:.4f}, "
                f"Total energy: {sum(energy_consumptions):.2f}"
            )
        
        self.logger.info("Federated learning completed successfully")
        return training_history
    
    def _select_clients(self) -> List[FederatedClient]:
        """Select clients for the current round (currently selects all)."""
        # TODO: Implement energy-aware client selection
        return self.clients
    
    def _evaluate_global_model(self) -> Dict:
        """Evaluate the global model using all clients' data."""
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        for client in self.clients:
            metrics = client.evaluate_model()
            total_correct += metrics['correct']
            total_samples += metrics['total']
            total_loss += metrics['loss'] * metrics['total']
        
        global_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
        global_loss = total_loss / total_samples if total_samples > 0 else 0
        
        return {
            'accuracy': global_accuracy,
            'loss': global_loss,
            'total_correct': total_correct,
            'total_samples': total_samples
        }
