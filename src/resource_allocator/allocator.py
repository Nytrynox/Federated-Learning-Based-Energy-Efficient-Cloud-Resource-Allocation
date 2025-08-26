"""
Intelligent cloud resource allocation with energy optimization.
"""

import time
import logging
import threading
import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import math

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import config
    from energy_monitor.monitor import EnergyMonitor
except ImportError:
    # Fallback configuration if modules not available
    class MockConfig:
        @property
        def resource_allocation(self):
            return {
                'min_cpu_cores': 1,
                'max_cpu_cores': 8,
                'min_memory_gb': 2,
                'max_memory_gb': 32,
                'scaling_factor': 1.2,
                'cooldown_period': 300,
                'allocation_strategy': 'energy_aware'
            }
    config = MockConfig()
    
    # Mock EnergyMonitor if not available
    class EnergyMonitor:
        def __init__(self, client_id=None):
            self.client_id = client_id


class ResourceType(Enum):
    """Types of cloud resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


class AllocationStrategy(Enum):
    """Resource allocation strategies."""
    ENERGY_AWARE = "energy_aware"
    PERFORMANCE_FIRST = "performance_first"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"


@dataclass
class ResourceRequest:
    """Resource request specification."""
    client_id: str
    cpu_cores: float
    memory_gb: float
    gpu_count: int = 0
    storage_gb: float = 0
    network_bandwidth_mbps: float = 0
    priority: int = 1  # 1-10, higher is more important
    max_energy_budget: Optional[float] = None
    deadline: Optional[datetime] = None


@dataclass
class ResourceAllocation:
    """Resource allocation result."""
    request_id: str
    client_id: str
    allocated_resources: Dict[ResourceType, float]
    node_id: str
    allocation_time: datetime
    estimated_energy_consumption: float
    estimated_completion_time: float
    cost_estimate: float


@dataclass
class CloudNode:
    """Cloud compute node representation."""
    node_id: str
    total_cpu_cores: float
    total_memory_gb: float
    total_gpu_count: int
    available_cpu_cores: float
    available_memory_gb: float
    available_gpu_count: int
    energy_efficiency_rating: float  # 0-1, higher is more efficient
    location: str
    cost_per_hour: float
    current_load: float  # 0-1
    energy_monitor: Optional[EnergyMonitor] = None


class ResourceAllocator:
    """Intelligent resource allocator with energy optimization."""
    
    def __init__(self):
        self.nodes: Dict[str, CloudNode] = {}
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        self.allocation_history: List[ResourceAllocation] = []
        
        # Configuration
        resource_config = config.resource_allocation
        self.min_cpu_cores = resource_config.get('min_cpu_cores', 1)
        self.max_cpu_cores = resource_config.get('max_cpu_cores', 8)
        self.min_memory_gb = resource_config.get('min_memory_gb', 2)
        self.max_memory_gb = resource_config.get('max_memory_gb', 32)
        self.scaling_factor = resource_config.get('scaling_factor', 1.2)
        self.cooldown_period = resource_config.get('cooldown_period', 300)
        self.allocation_strategy = AllocationStrategy(
            resource_config.get('allocation_strategy', 'energy_aware')
        )
        
        self.logger = logging.getLogger("ResourceAllocator")
        self._last_scaling_time = {}
        
        # Initialize with some default nodes (in real implementation, these would come from cloud provider)
        self._initialize_default_nodes()
    
    def _initialize_default_nodes(self):
        """Initialize with default cloud nodes."""
        default_nodes = [
            CloudNode(
                node_id="node-001",
                total_cpu_cores=8,
                total_memory_gb=32,
                total_gpu_count=1,
                available_cpu_cores=8,
                available_memory_gb=32,
                available_gpu_count=1,
                energy_efficiency_rating=0.8,
                location="eastus",
                cost_per_hour=0.12,
                current_load=0.0
            ),
            CloudNode(
                node_id="node-002",
                total_cpu_cores=16,
                total_memory_gb=64,
                total_gpu_count=2,
                available_cpu_cores=16,
                available_memory_gb=64,
                available_gpu_count=2,
                energy_efficiency_rating=0.9,
                location="westus",
                cost_per_hour=0.24,
                current_load=0.0
            ),
            CloudNode(
                node_id="node-003",
                total_cpu_cores=4,
                total_memory_gb=16,
                total_gpu_count=0,
                available_cpu_cores=4,
                available_memory_gb=16,
                available_gpu_count=0,
                energy_efficiency_rating=0.95,
                location="eastus",
                cost_per_hour=0.06,
                current_load=0.0
            )
        ]
        
        for node in default_nodes:
            self.nodes[node.node_id] = node
            # Initialize energy monitoring for each node
            node.energy_monitor = EnergyMonitor(client_id=f"node-{node.node_id}")
        
        self.logger.info(f"Initialized {len(default_nodes)} cloud nodes")
    
    def allocate_resources(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """Allocate resources based on the request and current strategy."""
        self.logger.info(f"Processing resource allocation request for client {request.client_id}")
        
        # Find suitable nodes
        suitable_nodes = self._find_suitable_nodes(request)
        if not suitable_nodes:
            self.logger.warning(f"No suitable nodes found for request from {request.client_id}")
            return None
        
        # Select best node based on strategy
        best_node = self._select_best_node(suitable_nodes, request)
        if not best_node:
            self.logger.warning(f"No optimal node selected for request from {request.client_id}")
            return None
        
        # Create allocation
        allocation = self._create_allocation(request, best_node)
        
        # Reserve resources
        self._reserve_resources(best_node, allocation)
        
        # Store allocation
        self.active_allocations[allocation.request_id] = allocation
        self.allocation_history.append(allocation)
        
        self.logger.info(
            f"Allocated resources to {request.client_id} on node {best_node.node_id}. "
            f"Estimated energy: {allocation.estimated_energy_consumption:.2f}Wh"
        )
        
        return allocation
    
    def _find_suitable_nodes(self, request: ResourceRequest) -> List[CloudNode]:
        """Find nodes that can satisfy the resource request."""
        suitable_nodes = []
        
        for node in self.nodes.values():
            if (node.available_cpu_cores >= request.cpu_cores and
                node.available_memory_gb >= request.memory_gb and
                node.available_gpu_count >= request.gpu_count):
                suitable_nodes.append(node)
        
        return suitable_nodes
    
    def _select_best_node(self, suitable_nodes: List[CloudNode], request: ResourceRequest) -> Optional[CloudNode]:
        """Select the best node based on the allocation strategy."""
        if not suitable_nodes:
            return None
        
        if self.allocation_strategy == AllocationStrategy.ENERGY_AWARE:
            return self._select_energy_aware_node(suitable_nodes, request)
        elif self.allocation_strategy == AllocationStrategy.PERFORMANCE_FIRST:
            return self._select_performance_first_node(suitable_nodes, request)
        elif self.allocation_strategy == AllocationStrategy.COST_OPTIMIZED:
            return self._select_cost_optimized_node(suitable_nodes, request)
        else:  # BALANCED
            return self._select_balanced_node(suitable_nodes, request)
    
    def _select_energy_aware_node(self, nodes: List[CloudNode], request: ResourceRequest) -> CloudNode:
        """Select node with best energy efficiency."""
        def energy_score(node: CloudNode) -> float:
            # Consider energy efficiency rating and current load
            load_penalty = node.current_load * 0.3  # Penalize high load
            efficiency_score = node.energy_efficiency_rating * (1 - load_penalty)
            
            # Consider energy budget if specified
            if request.max_energy_budget:
                estimated_energy = self._estimate_energy_consumption(node, request)
                if estimated_energy > request.max_energy_budget:
                    efficiency_score *= 0.5  # Heavily penalize budget violations
            
            return efficiency_score
        
        return max(nodes, key=energy_score)
    
    def _select_performance_first_node(self, nodes: List[CloudNode], request: ResourceRequest) -> CloudNode:
        """Select node with best performance characteristics."""
        def performance_score(node: CloudNode) -> float:
            # Consider total resources and current availability
            cpu_ratio = node.available_cpu_cores / node.total_cpu_cores
            memory_ratio = node.available_memory_gb / node.total_memory_gb
            gpu_factor = 1 + (node.total_gpu_count * 0.2)  # Bonus for GPU availability
            
            return (cpu_ratio + memory_ratio) * gpu_factor * (1 - node.current_load)
        
        return max(nodes, key=performance_score)
    
    def _select_cost_optimized_node(self, nodes: List[CloudNode], request: ResourceRequest) -> CloudNode:
        """Select node with lowest cost."""
        return min(nodes, key=lambda node: node.cost_per_hour)
    
    def _select_balanced_node(self, nodes: List[CloudNode], request: ResourceRequest) -> CloudNode:
        """Select node with balanced cost, performance, and energy efficiency."""
        def balanced_score(node: CloudNode) -> float:
            # Normalize all factors to 0-1 scale
            max_cost = max(n.cost_per_hour for n in nodes)
            cost_score = 1 - (node.cost_per_hour / max_cost) if max_cost > 0 else 1
            
            performance_score = (node.available_cpu_cores / node.total_cpu_cores + 
                               node.available_memory_gb / node.total_memory_gb) / 2
            
            energy_score = node.energy_efficiency_rating
            
            # Weighted combination
            return 0.3 * cost_score + 0.35 * performance_score + 0.35 * energy_score
        
        return max(nodes, key=balanced_score)
    
    def _create_allocation(self, request: ResourceRequest, node: CloudNode) -> ResourceAllocation:
        """Create resource allocation object."""
        request_id = f"req-{request.client_id}-{int(time.time())}"
        
        allocated_resources = {
            ResourceType.CPU: request.cpu_cores,
            ResourceType.MEMORY: request.memory_gb,
            ResourceType.GPU: request.gpu_count,
            ResourceType.STORAGE: request.storage_gb,
            ResourceType.NETWORK: request.network_bandwidth_mbps
        }
        
        estimated_energy = self._estimate_energy_consumption(node, request)
        estimated_completion_time = self._estimate_completion_time(request)
        cost_estimate = self._estimate_cost(node, request, estimated_completion_time)
        
        return ResourceAllocation(
            request_id=request_id,
            client_id=request.client_id,
            allocated_resources=allocated_resources,
            node_id=node.node_id,
            allocation_time=datetime.now(),
            estimated_energy_consumption=estimated_energy,
            estimated_completion_time=estimated_completion_time,
            cost_estimate=cost_estimate
        )
    
    def _estimate_energy_consumption(self, node: CloudNode, request: ResourceRequest) -> float:
        """Estimate energy consumption for the request on the given node."""
        # Simplified energy model
        base_energy = 10  # Base energy consumption in Wh
        cpu_energy = request.cpu_cores * 5  # 5 Wh per core
        memory_energy = request.memory_gb * 0.5  # 0.5 Wh per GB
        gpu_energy = request.gpu_count * 50  # 50 Wh per GPU
        
        total_energy = base_energy + cpu_energy + memory_energy + gpu_energy
        
        # Apply node efficiency factor
        efficiency_factor = node.energy_efficiency_rating
        return total_energy / efficiency_factor
    
    def _estimate_completion_time(self, request: ResourceRequest) -> float:
        """Estimate completion time in hours."""
        # Simplified estimation based on resource requirements
        base_time = 0.5  # 30 minutes base time
        cpu_factor = request.cpu_cores * 0.1
        memory_factor = request.memory_gb * 0.01
        
        return base_time + cpu_factor + memory_factor
    
    def _estimate_cost(self, node: CloudNode, request: ResourceRequest, completion_time: float) -> float:
        """Estimate cost for the allocation."""
        hourly_cost = node.cost_per_hour
        resource_multiplier = (request.cpu_cores / node.total_cpu_cores + 
                             request.memory_gb / node.total_memory_gb) / 2
        
        return hourly_cost * resource_multiplier * completion_time
    
    def _reserve_resources(self, node: CloudNode, allocation: ResourceAllocation):
        """Reserve resources on the selected node."""
        cpu_required = allocation.allocated_resources[ResourceType.CPU]
        memory_required = allocation.allocated_resources[ResourceType.MEMORY]
        gpu_required = allocation.allocated_resources[ResourceType.GPU]
        
        node.available_cpu_cores -= cpu_required
        node.available_memory_gb -= memory_required
        node.available_gpu_count -= int(gpu_required)
        
        # Update current load
        cpu_utilization = 1 - (node.available_cpu_cores / node.total_cpu_cores)
        memory_utilization = 1 - (node.available_memory_gb / node.total_memory_gb)
        node.current_load = max(cpu_utilization, memory_utilization)
        
        self.logger.debug(f"Reserved resources on {node.node_id}: CPU={cpu_required}, Memory={memory_required}GB")
    
    def release_resources(self, allocation_id: str) -> bool:
        """Release allocated resources."""
        if allocation_id not in self.active_allocations:
            self.logger.warning(f"Allocation {allocation_id} not found")
            return False
        
        allocation = self.active_allocations[allocation_id]
        node = self.nodes.get(allocation.node_id)
        
        if not node:
            self.logger.error(f"Node {allocation.node_id} not found")
            return False
        
        # Release resources
        cpu_to_release = allocation.allocated_resources[ResourceType.CPU]
        memory_to_release = allocation.allocated_resources[ResourceType.MEMORY]
        gpu_to_release = allocation.allocated_resources[ResourceType.GPU]
        
        node.available_cpu_cores += cpu_to_release
        node.available_memory_gb += memory_to_release
        node.available_gpu_count += int(gpu_to_release)
        
        # Ensure we don't exceed total capacity
        node.available_cpu_cores = min(node.available_cpu_cores, node.total_cpu_cores)
        node.available_memory_gb = min(node.available_memory_gb, node.total_memory_gb)
        node.available_gpu_count = min(node.available_gpu_count, node.total_gpu_count)
        
        # Update current load
        cpu_utilization = 1 - (node.available_cpu_cores / node.total_cpu_cores)
        memory_utilization = 1 - (node.available_memory_gb / node.total_memory_gb)
        node.current_load = max(cpu_utilization, memory_utilization)
        
        # Remove from active allocations
        del self.active_allocations[allocation_id]
        
        self.logger.info(f"Released resources for allocation {allocation_id}")
        return True
    
    def scale_resources(self, allocation_id: str, scale_factor: float) -> bool:
        """Scale allocated resources up or down."""
        if allocation_id not in self.active_allocations:
            self.logger.warning(f"Allocation {allocation_id} not found")
            return False
        
        # Check cooldown period
        current_time = time.time()
        if (allocation_id in self._last_scaling_time and
            current_time - self._last_scaling_time[allocation_id] < self.cooldown_period):
            self.logger.warning(f"Scaling cooldown period not met for {allocation_id}")
            return False
        
        allocation = self.active_allocations[allocation_id]
        node = self.nodes.get(allocation.node_id)
        
        if not node:
            self.logger.error(f"Node {allocation.node_id} not found")
            return False
        
        # Calculate new resource requirements
        current_cpu = allocation.allocated_resources[ResourceType.CPU]
        current_memory = allocation.allocated_resources[ResourceType.MEMORY]
        
        new_cpu = current_cpu * scale_factor
        new_memory = current_memory * scale_factor
        
        cpu_diff = new_cpu - current_cpu
        memory_diff = new_memory - current_memory
        
        # Check if scaling is possible
        if (cpu_diff > node.available_cpu_cores or 
            memory_diff > node.available_memory_gb or
            new_cpu > self.max_cpu_cores or
            new_memory > self.max_memory_gb):
            self.logger.warning(f"Cannot scale allocation {allocation_id}: insufficient resources")
            return False
        
        # Apply scaling
        node.available_cpu_cores -= cpu_diff
        node.available_memory_gb -= memory_diff
        
        allocation.allocated_resources[ResourceType.CPU] = new_cpu
        allocation.allocated_resources[ResourceType.MEMORY] = new_memory
        
        # Update load
        cpu_utilization = 1 - (node.available_cpu_cores / node.total_cpu_cores)
        memory_utilization = 1 - (node.available_memory_gb / node.total_memory_gb)
        node.current_load = max(cpu_utilization, memory_utilization)
        
        # Record scaling time
        self._last_scaling_time[allocation_id] = current_time
        
        self.logger.info(
            f"Scaled allocation {allocation_id} by factor {scale_factor}. "
            f"New resources: CPU={new_cpu}, Memory={new_memory}GB"
        )
        
        return True
    
    def get_cluster_status(self) -> Dict:
        """Get overall cluster status and statistics."""
        total_nodes = len(self.nodes)
        active_allocations = len(self.active_allocations)
        
        total_cpu = sum(node.total_cpu_cores for node in self.nodes.values())
        available_cpu = sum(node.available_cpu_cores for node in self.nodes.values())
        total_memory = sum(node.total_memory_gb for node in self.nodes.values())
        available_memory = sum(node.available_memory_gb for node in self.nodes.values())
        
        cpu_utilization = (total_cpu - available_cpu) / total_cpu if total_cpu > 0 else 0
        memory_utilization = (total_memory - available_memory) / total_memory if total_memory > 0 else 0
        
        avg_energy_efficiency = sum(node.energy_efficiency_rating for node in self.nodes.values()) / total_nodes
        
        return {
            'total_nodes': total_nodes,
            'active_allocations': active_allocations,
            'cpu_utilization': cpu_utilization,
            'memory_utilization': memory_utilization,
            'total_cpu_cores': total_cpu,
            'available_cpu_cores': available_cpu,
            'total_memory_gb': total_memory,
            'available_memory_gb': available_memory,
            'average_energy_efficiency': avg_energy_efficiency,
            'allocation_strategy': self.allocation_strategy.value
        }
    
    def optimize_allocations(self) -> List[str]:
        """Optimize existing allocations for better energy efficiency."""
        optimizations = []
        
        for allocation_id, allocation in self.active_allocations.items():
            node = self.nodes.get(allocation.node_id)
            if not node:
                continue
            
            # Check if we can find a more energy-efficient node
            current_energy = allocation.estimated_energy_consumption
            
            # Create a mock request to test other nodes
            mock_request = ResourceRequest(
                client_id=allocation.client_id,
                cpu_cores=allocation.allocated_resources[ResourceType.CPU],
                memory_gb=allocation.allocated_resources[ResourceType.MEMORY],
                gpu_count=int(allocation.allocated_resources[ResourceType.GPU])
            )
            
            suitable_nodes = self._find_suitable_nodes(mock_request)
            for candidate_node in suitable_nodes:
                if candidate_node.node_id == node.node_id:
                    continue
                
                estimated_energy = self._estimate_energy_consumption(candidate_node, mock_request)
                if estimated_energy < current_energy * 0.8:  # 20% improvement threshold
                    optimizations.append(
                        f"Move allocation {allocation_id} from {node.node_id} to {candidate_node.node_id} "
                        f"for {((current_energy - estimated_energy) / current_energy * 100):.1f}% energy savings"
                    )
                    break
        
        return optimizations
