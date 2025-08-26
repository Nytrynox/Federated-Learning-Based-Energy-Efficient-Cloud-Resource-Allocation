"""
Energy monitoring and optimization module for federated learning.
"""

import psutil
import time
import threading
import logging
import sys
import os
from typing import Dict, List, Optional
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import config
except ImportError:
    # Fallback configuration if config module not available
    class MockConfig:
        @property
        def energy_monitoring(self):
            return {
                'monitoring_interval': 10,
                'energy_threshold': 0.8,
                'cpu_weight': 0.4,
                'memory_weight': 0.3,
                'network_weight': 0.3,
                'enable_gpu_monitoring': True
            }
    config = MockConfig()


@dataclass
class EnergyMetrics:
    """Data class for storing energy metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    network_io: Dict[str, int]
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    power_consumption: Optional[float] = None


class EnergyMonitor:
    """Real-time energy monitoring and optimization."""
    
    def __init__(self, client_id: str = None):
        self.client_id = client_id or "global"
        self.monitoring = False
        self.metrics_history: List[EnergyMetrics] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Configuration
        energy_config = config.energy_monitoring
        self.monitoring_interval = energy_config.get('monitoring_interval', 10)
        self.energy_threshold = energy_config.get('energy_threshold', 0.8)
        self.cpu_weight = energy_config.get('cpu_weight', 0.4)
        self.memory_weight = energy_config.get('memory_weight', 0.3)
        self.network_weight = energy_config.get('network_weight', 0.3)
        self.enable_gpu_monitoring = energy_config.get('enable_gpu_monitoring', True)
        
        self.logger = logging.getLogger(f"EnergyMonitor-{self.client_id}")
        
        # Initialize baseline metrics
        self.baseline_cpu = self._get_cpu_usage()
        self.baseline_memory = self._get_memory_usage()
        self.baseline_network = self._get_network_io()
        
        # Try to initialize GPU monitoring
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0 and self.enable_gpu_monitoring
        except ImportError:
            self.logger.warning("GPUtil not available. GPU monitoring disabled.")
            return False
        except Exception as e:
            self.logger.warning(f"GPU monitoring initialization failed: {e}")
            return False
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        return psutil.virtual_memory().percent
    
    def _get_network_io(self) -> Dict[str, int]:
        """Get current network I/O statistics."""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU usage and memory statistics."""
        if not self.gpu_available:
            return {'usage': 0.0, 'memory': 0.0}
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    'usage': gpu.load * 100,
                    'memory': gpu.memoryUtil * 100
                }
        except Exception as e:
            self.logger.warning(f"Failed to get GPU metrics: {e}")
        
        return {'usage': 0.0, 'memory': 0.0}
    
    def _estimate_power_consumption(self, cpu_usage: float, memory_usage: float, 
                                  gpu_usage: float = 0.0) -> float:
        """Estimate power consumption based on resource usage."""
        # Simplified power model (in watts)
        # These are rough estimates and should be calibrated for specific hardware
        base_power = 50  # Base system power
        cpu_power = (cpu_usage / 100) * 65  # CPU TDP approximation
        memory_power = (memory_usage / 100) * 20  # Memory power approximation
        gpu_power = (gpu_usage / 100) * 150 if self.gpu_available else 0  # GPU power
        
        total_power = base_power + cpu_power + memory_power + gpu_power
        return total_power
    
    def _collect_metrics(self) -> EnergyMetrics:
        """Collect current energy metrics."""
        cpu_usage = self._get_cpu_usage()
        memory_usage = self._get_memory_usage()
        network_io = self._get_network_io()
        
        gpu_metrics = self._get_gpu_metrics()
        gpu_usage = gpu_metrics['usage']
        gpu_memory = gpu_metrics['memory']
        
        power_consumption = self._estimate_power_consumption(
            cpu_usage, memory_usage, gpu_usage
        )
        
        return EnergyMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            network_io=network_io,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            power_consumption=power_consumption
        )
    
    def _monitoring_loop(self):
        """Main monitoring loop running in a separate thread."""
        self.logger.info(f"Energy monitoring started for {self.client_id}")
        
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for energy threshold violations
                energy_score = self.calculate_energy_efficiency_score(metrics)
                if energy_score > self.energy_threshold:
                    self.logger.warning(
                        f"Energy threshold exceeded: {energy_score:.2f} > {self.energy_threshold}"
                    )
                
                # Log metrics periodically
                if len(self.metrics_history) % 10 == 0:
                    self.logger.info(
                        f"Energy metrics - CPU: {metrics.cpu_usage:.1f}%, "
                        f"Memory: {metrics.memory_usage:.1f}%, "
                        f"Power: {metrics.power_consumption:.1f}W"
                    )
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def start_monitoring(self):
        """Start energy monitoring."""
        if self.monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.metrics_history.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self) -> float:
        """Stop energy monitoring and return total energy consumed."""
        if not self.monitoring:
            self.logger.warning("Monitoring not started")
            return 0.0
        
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        total_energy = self.calculate_total_energy_consumption()
        self.logger.info(f"Energy monitoring stopped. Total energy consumed: {total_energy:.2f} Wh")
        
        return total_energy
    
    def calculate_energy_efficiency_score(self, metrics: EnergyMetrics) -> float:
        """Calculate normalized energy efficiency score (0-1, higher = less efficient)."""
        cpu_score = metrics.cpu_usage / 100
        memory_score = metrics.memory_usage / 100
        
        # Network score based on activity (simplified)
        network_score = min(1.0, (
            metrics.network_io['bytes_sent'] + metrics.network_io['bytes_recv']
        ) / (1024 * 1024 * 100))  # Normalize to 100MB baseline
        
        # Weighted combination
        efficiency_score = (
            self.cpu_weight * cpu_score +
            self.memory_weight * memory_score +
            self.network_weight * network_score
        )
        
        return min(1.0, efficiency_score)
    
    def calculate_total_energy_consumption(self) -> float:
        """Calculate total energy consumption in Watt-hours."""
        if not self.metrics_history:
            return 0.0
        
        total_energy = 0.0
        for i in range(1, len(self.metrics_history)):
            current_metrics = self.metrics_history[i]
            previous_metrics = self.metrics_history[i - 1]
            
            time_diff = (current_metrics.timestamp - previous_metrics.timestamp).total_seconds() / 3600
            avg_power = (current_metrics.power_consumption + previous_metrics.power_consumption) / 2
            
            total_energy += avg_power * time_diff
        
        return total_energy
    
    def get_energy_statistics(self) -> Dict:
        """Get comprehensive energy statistics."""
        if not self.metrics_history:
            return {}
        
        power_values = [m.power_consumption for m in self.metrics_history if m.power_consumption]
        cpu_values = [m.cpu_usage for m in self.metrics_history]
        memory_values = [m.memory_usage for m in self.metrics_history]
        
        stats = {
            'total_energy_wh': self.calculate_total_energy_consumption(),
            'monitoring_duration_minutes': len(self.metrics_history) * self.monitoring_interval / 60,
            'num_samples': len(self.metrics_history),
            'power_stats': {
                'avg': statistics.mean(power_values) if power_values else 0,
                'min': min(power_values) if power_values else 0,
                'max': max(power_values) if power_values else 0,
                'std': statistics.stdev(power_values) if len(power_values) > 1 else 0
            },
            'cpu_stats': {
                'avg': statistics.mean(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values),
                'std': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            'memory_stats': {
                'avg': statistics.mean(memory_values),
                'min': min(memory_values),
                'max': max(memory_values),
                'std': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            }
        }
        
        return stats
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        metrics_data = []
        for metric in self.metrics_history:
            metrics_data.append({
                'timestamp': metric.timestamp.isoformat(),
                'cpu_usage': metric.cpu_usage,
                'memory_usage': metric.memory_usage,
                'network_io': metric.network_io,
                'gpu_usage': metric.gpu_usage,
                'gpu_memory': metric.gpu_memory,
                'power_consumption': metric.power_consumption
            })
        
        with open(filepath, 'w') as f:
            json.dump({
                'client_id': self.client_id,
                'metrics': metrics_data,
                'statistics': self.get_energy_statistics()
            }, f, indent=2)
        
        self.logger.info(f"Energy metrics exported to {filepath}")


class EnergyOptimizer:
    """Energy optimization strategies for federated learning."""
    
    def __init__(self):
        self.logger = logging.getLogger("EnergyOptimizer")
        energy_config = config.energy_monitoring
        self.energy_threshold = energy_config.get('energy_threshold', 0.8)
    
    def optimize_client_selection(self, clients_energy_scores: Dict[str, float]) -> List[str]:
        """Select most energy-efficient clients for federated learning round."""
        if not clients_energy_scores:
            return []
        
        # Sort clients by energy efficiency (lower score = more efficient)
        sorted_clients = sorted(
            clients_energy_scores.items(),
            key=lambda x: x[1]
        )
        
        # Select top 70% most efficient clients
        num_clients_to_select = max(1, int(len(sorted_clients) * 0.7))
        selected_clients = [client_id for client_id, _ in sorted_clients[:num_clients_to_select]]
        
        self.logger.info(
            f"Selected {len(selected_clients)} most energy-efficient clients: {selected_clients}"
        )
        
        return selected_clients
    
    def suggest_resource_adjustments(self, energy_metrics: EnergyMetrics) -> Dict[str, str]:
        """Suggest resource adjustments based on energy metrics."""
        suggestions = {}
        
        if energy_metrics.cpu_usage > 80:
            suggestions['cpu'] = "Consider reducing batch size or local epochs to decrease CPU load"
        
        if energy_metrics.memory_usage > 85:
            suggestions['memory'] = "Reduce model size or batch size to decrease memory usage"
        
        if energy_metrics.gpu_usage and energy_metrics.gpu_usage > 90:
            suggestions['gpu'] = "GPU utilization is very high, consider reducing model complexity"
        
        efficiency_score = self._calculate_efficiency_score(energy_metrics)
        if efficiency_score > self.energy_threshold:
            suggestions['general'] = "Overall energy efficiency is low, consider optimizing training parameters"
        
        return suggestions
    
    def _calculate_efficiency_score(self, metrics: EnergyMetrics) -> float:
        """Calculate overall efficiency score."""
        cpu_score = metrics.cpu_usage / 100
        memory_score = metrics.memory_usage / 100
        gpu_score = (metrics.gpu_usage or 0) / 100
        
        return (cpu_score + memory_score + gpu_score) / 3
    
    def adaptive_learning_rate(self, current_lr: float, energy_score: float) -> float:
        """Adapt learning rate based on energy efficiency."""
        if energy_score > self.energy_threshold:
            # Reduce learning rate to potentially reduce energy consumption
            new_lr = current_lr * 0.9
            self.logger.info(f"Reducing learning rate from {current_lr} to {new_lr} due to high energy consumption")
            return new_lr
        elif energy_score < 0.5:
            # Increase learning rate if energy efficiency is very good
            new_lr = min(current_lr * 1.1, 0.1)  # Cap at 0.1
            self.logger.info(f"Increasing learning rate from {current_lr} to {new_lr} due to good energy efficiency")
            return new_lr
        
        return current_lr
