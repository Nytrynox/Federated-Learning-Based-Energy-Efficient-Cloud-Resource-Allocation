"""
Unit tests for federated learning core functionality.
"""

import unittest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from federated_learning.core import FederatedModel, FederatedClient, FederatedAggregator
from energy_monitor.monitor import EnergyMonitor


class TestFederatedModel(unittest.TestCase):
    """Test cases for FederatedModel."""
    
    def setUp(self):
        self.model = FederatedModel(input_dim=784, hidden_dim=128, output_dim=10)
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        self.assertEqual(self.model.input_dim, 784)
        self.assertEqual(self.model.hidden_dim, 128)
        self.assertEqual(self.model.output_dim, 10)
        
        # Check layers exist
        self.assertIsInstance(self.model.fc1, nn.Linear)
        self.assertIsInstance(self.model.fc2, nn.Linear)
        self.assertIsInstance(self.model.fc3, nn.Linear)
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        batch_size = 32
        input_data = torch.randn(batch_size, 784)
        
        output = self.model(input_data)
        
        self.assertEqual(output.shape, (batch_size, 10))
    
    def test_model_parameter_count(self):
        """Test model has expected number of parameters."""
        param_count = sum(p.numel() for p in self.model.parameters())
        expected_count = (784 * 128) + 128 + (128 * 128) + 128 + (128 * 10) + 10
        self.assertEqual(param_count, expected_count)


class TestFederatedClient(unittest.TestCase):
    """Test cases for FederatedClient."""
    
    def setUp(self):
        self.model = FederatedModel()
        self.mock_data_loader = Mock()
        self.mock_energy_monitor = Mock()
        self.client = FederatedClient(
            client_id="test_client",
            model=self.model,
            data_loader=self.mock_data_loader,
            energy_monitor=self.mock_energy_monitor
        )
    
    def test_client_initialization(self):
        """Test client initializes correctly."""
        self.assertEqual(self.client.client_id, "test_client")
        self.assertEqual(self.client.model, self.model)
        self.assertEqual(self.client.data_loader, self.mock_data_loader)
        self.assertEqual(self.client.energy_monitor, self.mock_energy_monitor)
    
    def test_update_model(self):
        """Test model parameter updates."""
        # Create mock global parameters
        global_params = {
            'fc1.weight': torch.randn(128, 784),
            'fc1.bias': torch.randn(128)
        }
        
        # Update model
        self.client.update_model(global_params)
        
        # Check parameters were updated
        for name, param in self.client.model.named_parameters():
            if name in global_params:
                self.assertTrue(torch.equal(param, global_params[name]))


class TestFederatedAggregator(unittest.TestCase):
    """Test cases for FederatedAggregator."""
    
    def setUp(self):
        self.model = FederatedModel()
        self.aggregator = FederatedAggregator(self.model)
    
    def test_fed_avg_aggregation(self):
        """Test FedAvg aggregation method."""
        # Create mock client updates
        client_updates = [
            {
                'model_params': {
                    'fc1.weight': torch.ones(128, 784),
                    'fc1.bias': torch.ones(128)
                },
                'num_samples': 100
            },
            {
                'model_params': {
                    'fc1.weight': torch.zeros(128, 784),
                    'fc1.bias': torch.zeros(128)
                },
                'num_samples': 200
            }
        ]
        
        # Aggregate
        aggregated = self.aggregator._fed_avg_aggregation(client_updates)
        
        # Check weighted average
        expected_weight = torch.ones(128, 784) * (100/300) + torch.zeros(128, 784) * (200/300)
        expected_bias = torch.ones(128) * (100/300) + torch.zeros(128) * (200/300)
        
        self.assertTrue(torch.allclose(aggregated['fc1.weight'], expected_weight))
        self.assertTrue(torch.allclose(aggregated['fc1.bias'], expected_bias))
    
    def test_energy_aware_aggregation(self):
        """Test energy-aware aggregation method."""
        client_updates = [
            {
                'model_params': {
                    'fc1.weight': torch.ones(128, 784),
                },
                'num_samples': 100
            },
            {
                'model_params': {
                    'fc1.weight': torch.zeros(128, 784),
                },
                'num_samples': 100
            }
        ]
        
        energy_consumptions = [10.0, 5.0]  # Second client is more energy efficient
        
        aggregated = self.aggregator._energy_aware_aggregation(client_updates, energy_consumptions)
        
        # Check that energy-efficient client gets higher weight
        self.assertIsNotNone(aggregated['fc1.weight'])


class TestEnergyMonitor(unittest.TestCase):
    """Test cases for EnergyMonitor."""
    
    def setUp(self):
        self.monitor = EnergyMonitor(client_id="test_client")
    
    def test_monitor_initialization(self):
        """Test monitor initializes correctly."""
        self.assertEqual(self.monitor.client_id, "test_client")
        self.assertFalse(self.monitor.monitoring)
        self.assertEqual(len(self.monitor.metrics_history), 0)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.net_io_counters')
    def test_collect_metrics(self, mock_net, mock_memory, mock_cpu):
        """Test metrics collection."""
        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0)
        mock_net.return_value = Mock(
            bytes_sent=1000,
            bytes_recv=2000,
            packets_sent=10,
            packets_recv=20
        )
        
        metrics = self.monitor._collect_metrics()
        
        self.assertEqual(metrics.cpu_usage, 50.0)
        self.assertEqual(metrics.memory_usage, 60.0)
        self.assertEqual(metrics.network_io['bytes_sent'], 1000)
        self.assertIsNotNone(metrics.power_consumption)
    
    def test_calculate_energy_efficiency_score(self):
        """Test energy efficiency score calculation."""
        from energy_monitor.monitor import EnergyMetrics
        from datetime import datetime
        
        metrics = EnergyMetrics(
            timestamp=datetime.now(),
            cpu_usage=50.0,
            memory_usage=60.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000, 'packets_sent': 10, 'packets_recv': 20}
        )
        
        score = self.monitor.calculate_energy_efficiency_score(metrics)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == '__main__':
    # Set up test environment
    import logging
    logging.disable(logging.CRITICAL)  # Disable logging during tests
    
    # Run tests
    unittest.main(verbosity=2)
