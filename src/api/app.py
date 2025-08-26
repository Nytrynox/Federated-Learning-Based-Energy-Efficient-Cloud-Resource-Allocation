"""
REST API for federated learning system.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import threading
import json
from datetime import datetime
from typing import Dict, List

from ..config import config
from ..federated_learning.core import FederatedLearningCoordinator, FederatedModel, FederatedClient
from ..energy_monitor.monitor import EnergyMonitor
from ..resource_allocator.allocator import ResourceAllocator, ResourceRequest


class FederatedLearningAPI:
    """REST API for federated learning system."""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app, origins="*")
        
        # Components
        self.resource_allocator = ResourceAllocator()
        self.energy_monitors: Dict[str, EnergyMonitor] = {}
        self.fl_coordinator = None
        self.training_thread = None
        self.training_active = False
        
        # API configuration
        api_config = config.api
        self.host = api_config.get('host', '0.0.0.0')
        self.port = api_config.get('port', 8000)
        self.debug = api_config.get('debug', False)
        
        self.logger = logging.getLogger("FederatedLearningAPI")
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
        
        @self.app.route('/api/v1/training/start', methods=['POST'])
        def start_training():
            """Start federated learning training."""
            try:
                if self.training_active:
                    return jsonify({'error': 'Training already in progress'}), 400
                
                training_config = request.get_json() or {}
                
                # Initialize training
                self._initialize_training(training_config)
                
                # Start training in background thread
                self.training_thread = threading.Thread(
                    target=self._run_training,
                    daemon=True
                )
                self.training_thread.start()
                self.training_active = True
                
                return jsonify({
                    'message': 'Federated learning training started',
                    'training_id': f"train-{int(datetime.now().timestamp())}"
                })
                
            except Exception as e:
                self.logger.error(f"Error starting training: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/training/stop', methods=['POST'])
        def stop_training():
            """Stop federated learning training."""
            try:
                if not self.training_active:
                    return jsonify({'error': 'No training in progress'}), 400
                
                self.training_active = False
                
                return jsonify({'message': 'Training stop requested'})
                
            except Exception as e:
                self.logger.error(f"Error stopping training: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/training/status', methods=['GET'])
        def get_training_status():
            """Get current training status."""
            try:
                status = {
                    'training_active': self.training_active,
                    'fl_coordinator_initialized': self.fl_coordinator is not None,
                    'num_clients': len(self.energy_monitors),
                    'current_round': getattr(self.fl_coordinator, 'round_number', 0) if self.fl_coordinator else 0
                }
                
                return jsonify(status)
                
            except Exception as e:
                self.logger.error(f"Error getting training status: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/resources/allocate', methods=['POST'])
        def allocate_resources():
            """Allocate cloud resources for a client."""
            try:
                req_data = request.get_json()
                
                # Validate required fields
                required_fields = ['client_id', 'cpu_cores', 'memory_gb']
                for field in required_fields:
                    if field not in req_data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400
                
                # Create resource request
                resource_request = ResourceRequest(
                    client_id=req_data['client_id'],
                    cpu_cores=req_data['cpu_cores'],
                    memory_gb=req_data['memory_gb'],
                    gpu_count=req_data.get('gpu_count', 0),
                    storage_gb=req_data.get('storage_gb', 0),
                    network_bandwidth_mbps=req_data.get('network_bandwidth_mbps', 0),
                    priority=req_data.get('priority', 1),
                    max_energy_budget=req_data.get('max_energy_budget')
                )
                
                # Allocate resources
                allocation = self.resource_allocator.allocate_resources(resource_request)
                
                if allocation:
                    return jsonify({
                        'allocation_id': allocation.request_id,
                        'node_id': allocation.node_id,
                        'allocated_resources': {
                            str(k): v for k, v in allocation.allocated_resources.items()
                        },
                        'estimated_energy_consumption': allocation.estimated_energy_consumption,
                        'estimated_completion_time': allocation.estimated_completion_time,
                        'cost_estimate': allocation.cost_estimate
                    })
                else:
                    return jsonify({'error': 'No suitable resources available'}), 503
                
            except Exception as e:
                self.logger.error(f"Error allocating resources: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/resources/release/<allocation_id>', methods=['DELETE'])
        def release_resources(allocation_id):
            """Release allocated resources."""
            try:
                success = self.resource_allocator.release_resources(allocation_id)
                
                if success:
                    return jsonify({'message': f'Resources released for allocation {allocation_id}'})
                else:
                    return jsonify({'error': 'Allocation not found or already released'}), 404
                
            except Exception as e:
                self.logger.error(f"Error releasing resources: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/resources/scale/<allocation_id>', methods=['PUT'])
        def scale_resources(allocation_id):
            """Scale allocated resources."""
            try:
                req_data = request.get_json()
                scale_factor = req_data.get('scale_factor', 1.0)
                
                if scale_factor <= 0:
                    return jsonify({'error': 'Scale factor must be positive'}), 400
                
                success = self.resource_allocator.scale_resources(allocation_id, scale_factor)
                
                if success:
                    return jsonify({
                        'message': f'Resources scaled by factor {scale_factor} for allocation {allocation_id}'
                    })
                else:
                    return jsonify({'error': 'Scaling failed or allocation not found'}), 400
                
            except Exception as e:
                self.logger.error(f"Error scaling resources: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/resources/status', methods=['GET'])
        def get_resource_status():
            """Get cluster resource status."""
            try:
                status = self.resource_allocator.get_cluster_status()
                return jsonify(status)
                
            except Exception as e:
                self.logger.error(f"Error getting resource status: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/energy/monitor/start/<client_id>', methods=['POST'])
        def start_energy_monitoring(client_id):
            """Start energy monitoring for a client."""
            try:
                if client_id in self.energy_monitors:
                    return jsonify({'error': f'Energy monitoring already active for {client_id}'}), 400
                
                monitor = EnergyMonitor(client_id=client_id)
                monitor.start_monitoring()
                self.energy_monitors[client_id] = monitor
                
                return jsonify({'message': f'Energy monitoring started for {client_id}'})
                
            except Exception as e:
                self.logger.error(f"Error starting energy monitoring: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/energy/monitor/stop/<client_id>', methods=['POST'])
        def stop_energy_monitoring(client_id):
            """Stop energy monitoring for a client."""
            try:
                if client_id not in self.energy_monitors:
                    return jsonify({'error': f'No energy monitoring active for {client_id}'}), 404
                
                monitor = self.energy_monitors[client_id]
                total_energy = monitor.stop_monitoring()
                del self.energy_monitors[client_id]
                
                return jsonify({
                    'message': f'Energy monitoring stopped for {client_id}',
                    'total_energy_consumed': total_energy
                })
                
            except Exception as e:
                self.logger.error(f"Error stopping energy monitoring: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/energy/stats/<client_id>', methods=['GET'])
        def get_energy_stats(client_id):
            """Get energy statistics for a client."""
            try:
                if client_id not in self.energy_monitors:
                    return jsonify({'error': f'No energy monitoring active for {client_id}'}), 404
                
                monitor = self.energy_monitors[client_id]
                stats = monitor.get_energy_statistics()
                
                return jsonify(stats)
                
            except Exception as e:
                self.logger.error(f"Error getting energy stats: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/optimization/recommendations', methods=['GET'])
        def get_optimization_recommendations():
            """Get optimization recommendations."""
            try:
                recommendations = self.resource_allocator.optimize_allocations()
                
                return jsonify({
                    'recommendations': recommendations,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error getting optimization recommendations: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/simulation/generate_data', methods=['POST'])
        def generate_simulation_data():
            """Generate synthetic data for simulation."""
            try:
                req_data = request.get_json() or {}
                num_clients = req_data.get('num_clients', 5)
                samples_per_client = req_data.get('samples_per_client', 1000)
                
                # Generate synthetic federated learning data
                simulation_data = self._generate_simulation_data(num_clients, samples_per_client)
                
                return jsonify({
                    'message': 'Simulation data generated',
                    'num_clients': num_clients,
                    'samples_per_client': samples_per_client,
                    'data_summary': simulation_data
                })
                
            except Exception as e:
                self.logger.error(f"Error generating simulation data: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _initialize_training(self, training_config: Dict):
        """Initialize federated learning components."""
        # This is a simplified initialization
        # In a real implementation, you would load actual data and models
        self.logger.info("Initializing federated learning training")
        
        # Initialize dummy clients for demonstration
        num_clients = training_config.get('num_clients', 3)
        clients = []
        
        for i in range(num_clients):
            client_id = f"client-{i+1}"
            model = FederatedModel()
            
            # Create dummy data loader (in real implementation, load actual data)
            data_loader = self._create_dummy_data_loader()
            
            # Initialize energy monitor
            energy_monitor = EnergyMonitor(client_id=client_id)
            self.energy_monitors[client_id] = energy_monitor
            
            client = FederatedClient(client_id, model, data_loader, energy_monitor)
            clients.append(client)
        
        # Initialize global model
        global_model = FederatedModel()
        
        # Create coordinator
        self.fl_coordinator = FederatedLearningCoordinator(global_model, clients)
        
        self.logger.info(f"Initialized {num_clients} clients for federated learning")
    
    def _create_dummy_data_loader(self):
        """Create dummy data loader for demonstration."""
        # This would be replaced with actual data loading logic
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        # Generate random data for demonstration
        x = torch.randn(100, 784)  # 100 samples, 784 features (28x28 images)
        y = torch.randint(0, 10, (100,))  # 10 classes
        
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=32, shuffle=True)
    
    def _run_training(self):
        """Run federated learning training."""
        try:
            if self.fl_coordinator:
                self.logger.info("Starting federated learning training")
                training_history = self.fl_coordinator.run_federated_learning()
                self.logger.info("Federated learning training completed")
                
                # Store training results (in real implementation, save to database)
                self._save_training_results(training_history)
                
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
        finally:
            self.training_active = False
    
    def _save_training_results(self, training_history: Dict):
        """Save training results."""
        # In a real implementation, save to database
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                # Convert any non-serializable objects to strings
                serializable_history = self._make_serializable(training_history)
                json.dump(serializable_history, f, indent=2)
            
            self.logger.info(f"Training results saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving training results: {e}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _generate_simulation_data(self, num_clients: int, samples_per_client: int) -> Dict:
        """Generate simulation data summary."""
        return {
            'total_samples': num_clients * samples_per_client,
            'data_distribution': 'iid',  # Assuming IID for simulation
            'feature_dimensions': 784,
            'num_classes': 10,
            'clients': [f"client-{i+1}" for i in range(num_clients)]
        }
    
    def run(self):
        """Run the Flask application."""
        self.logger.info(f"Starting Federated Learning API on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug)


def create_app():
    """Create Flask app instance."""
    api = FederatedLearningAPI()
    return api.app


if __name__ == '__main__':
    api = FederatedLearningAPI()
    api.run()
