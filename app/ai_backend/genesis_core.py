"""
Genesis Core module for AI backend functionality.
"""

import logging
import time
from typing import Dict, Any, Optional
import requests
from functools import lru_cache


logger = logging.getLogger(__name__)


class GenesisCore:
    """Main class for Genesis Core functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize GenesisCore with configuration."""
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        
        self.config = config
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url', 'https://api.example.com')
        self.timeout = config.get('timeout', 30)
        self.retries = config.get('retries', 3)
        self.cache = {}
        
        if not self.api_key:
            raise ValueError("API key is required in config")
            
        logger.info(f"GenesisCore initialized with base_url: {self.base_url}")
    
    def process_data(self, data: Any) -> Dict[str, Any]:
        """Process input data and return structured output."""
        if data is None:
            return {"status": "error", "message": "No data provided"}
        
        if isinstance(data, str):
            if len(data) > MAX_INPUT_SIZE:  # Large input handling
                logger.warning("Processing large input data")
            
            return {
                "status": "success",
                "processed_data": f"processed_{data}",
                "input_type": "string",
                "length": len(data)
            }
        
        if isinstance(data, dict):
            if not data:  # Empty dict
                return {"status": "success", "processed_data": {}, "input_type": "dict"}
            
            processed = {}
            for key, value in data.items():
                processed[key] = self._transform_value(value)
            
            return {
                "status": "success",
                "processed_data": processed,
                "input_type": "dict"
            }
        
        return {"status": "error", "message": f"Unsupported data type: {type(data)}"}
    
    def _transform_value(self, value: Any) -> Any:
        """Transform individual values during processing."""
        if isinstance(value, str):
            return value.upper()
        elif isinstance(value, (int, float)):
            return value * 2
        elif isinstance(value, list):
            return [self._transform_value(item) for item in value]
        return value
    
    @lru_cache(maxsize=128)
    def cached_operation(self, input_data: str) -> str:
        """Perform a cached operation."""
        time.sleep(0.1)  # Simulate work
        return f"cached_result_{input_data}"
    
    def make_api_request(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make an API request with error handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.post(
                url,
                json=data,
                timeout=self.timeout,
                headers={'Authorization': f'Bearer {self.api_key}'}
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise ValueError("Authentication failed")
            elif response.status_code == 403:
                raise PermissionError("Access forbidden")
            else:
                raise requests.RequestException(f"API request failed with status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Network connection failed")
        except requests.exceptions.Timeout:
            raise TimeoutError("Request timeout")
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data for security and format."""
        if data is None:
            return False
        
        if isinstance(data, str):
            if not data.strip():
                return False
            
            # Check for potential security issues
            dangerous_patterns = ['<script>', 'DROP TABLE', '../', '../../']
            for pattern in dangerous_patterns:
                if pattern in data:
                    return False
        
        if isinstance(data, dict):
            if 'sql_injection' in data:
                return False
        
        return True
    
    def sanitize_input(self, data: str) -> str:
        """Sanitize input to prevent security issues."""
        if not isinstance(data, str):
            return str(data)
        
        # Remove potential XSS
        data = data.replace('<script>', '').replace('</script>', '')
        
        # Remove potential SQL injection
        data = data.replace("'; DROP TABLE", "")
        data = data.replace('"; DROP TABLE', "")
        
        # Remove path traversal
        data = data.replace('../', '')
        
        return data
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "module": "genesis_core",
            "version": "1.0.0",
            "config_keys": list(self.config.keys()),
            "cache_size": len(self.cache)
        }


# Module-level functions
def initialize_genesis(config: Dict[str, Any]) -> GenesisCore:
    """Initialize and return a GenesisCore instance."""
    return GenesisCore(config)


def process_simple_data(data: str) -> str:
    """Simple data processing function."""
    if not data:
        return ""
    return f"processed_{data}"


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary."""
    required_keys = ['api_key', 'base_url']
    return all(key in config for key in required_keys)


# Constants
DEFAULT_CONFIG = {
    'base_url': 'https://api.example.com',
    'timeout': 30,
    'retries': 3
}

MAX_INPUT_SIZE = 100000
MIN_INPUT_SIZE = 1