"""
Configuration management for the Federated Learning system.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration class to load and manage application settings."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                Path(__file__).parent.parent, 
                "config", 
                "settings.yaml"
            )
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Expand environment variables
            return self._expand_env_vars(config)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def _expand_env_vars(self, obj: Any) -> Any:
        """Recursively expand environment variables in configuration."""
        if isinstance(obj, dict):
            return {key: self._expand_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Handle ${VAR} and ${VAR:default} patterns
            if obj.startswith("${") and obj.endswith("}"):
                var_expr = obj[2:-1]
                if ":" in var_expr:
                    var_name, default_value = var_expr.split(":", 1)
                    return os.getenv(var_name, default_value)
                else:
                    return os.getenv(var_expr, obj)
            return obj
        else:
            return obj
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.get(section, {})
    
    @property
    def federated_learning(self) -> Dict[str, Any]:
        """Get federated learning configuration."""
        return self.get_section('federated_learning')
    
    @property
    def energy_monitoring(self) -> Dict[str, Any]:
        """Get energy monitoring configuration."""
        return self.get_section('energy_monitoring')
    
    @property
    def resource_allocation(self) -> Dict[str, Any]:
        """Get resource allocation configuration."""
        return self.get_section('resource_allocation')
    
    @property
    def azure(self) -> Dict[str, Any]:
        """Get Azure configuration."""
        return self.get_section('azure')
    
    @property
    def api(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.get_section('api')
    
    @property
    def database(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.get_section('database')
    
    @property
    def redis(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        return self.get_section('redis')
    
    @property
    def simulation(self) -> Dict[str, Any]:
        """Get simulation configuration."""
        return self.get_section('simulation')
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get_section('logging')


# Global configuration instance
config = Config()
