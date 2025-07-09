import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Add the app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.ai_backend.genesis_core import (
    GenesisCore, 
    initialize_genesis, 
    process_simple_data, 
    validate_config,
    DEFAULT_CONFIG,
    MAX_INPUT_SIZE,
    MIN_INPUT_SIZE
)


class TestGenesisCoreInitialization:
    """Test class for genesis core initialization and setup."""
    
    def test_module_import(self):
        """Test that the genesis_core module can be imported without raising an ImportError."""
        try:
            import app.ai_backend.genesis_core
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import genesis_core module: {e}")
    
    def test_initialization_with_valid_config(self):
        """Test successful initialization of genesis_core with a valid configuration."""
        valid_config = {
            'api_key': 'test_key_123',
            'base_url': 'https://api.test.com',
            'timeout': 30,
            'retries': 3
        }
        
        core = GenesisCore(valid_config)
        assert core.api_key == 'test_key_123'
        assert core.base_url == 'https://api.test.com'
        assert core.timeout == 30
        assert core.retries == 3
    
    def test_initialization_with_invalid_config(self):
        """Test that initializing genesis_core with invalid configuration raises appropriate error."""
        # Test with non-dict config
        with pytest.raises(ValueError, match="Config must be a dictionary"):
            GenesisCore("invalid_config")
        
        # Test with missing API key
        with pytest.raises(ValueError, match="API key is required in config"):
            GenesisCore({'base_url': 'https://api.test.com'})
    
    def test_initialization_with_missing_config(self):
        """Test initialization when required configuration data is missing."""
        minimal_config = {'api_key': 'test_key'}
        
        core = GenesisCore(minimal_config)
        # Should use default values
        assert core.base_url == 'https://api.example.com'
        assert core.timeout == 30
        assert core.retries == 3
    
    def test_initialize_genesis_function(self):
        """Test the module-level initialize_genesis function."""
        config = {'api_key': 'test_key', 'base_url': 'https://api.test.com'}
        core = initialize_genesis(config)
        
        assert isinstance(core, GenesisCore)
        assert core.api_key == 'test_key'


class TestGenesisCoreCoreFunctionality:
    """Test class for core functionality of genesis_core module."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.mock_config = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.example.com',
            'timeout': 30,
            'retries': 3
        }
        self.core = GenesisCore(self.mock_config)
    
    def teardown_method(self):
        """Clean up after each test method."""
        self.core.cache.clear()
    
    def test_process_data_happy_path(self):
        """Test that data processing returns correct output for valid input."""
        test_data = {"input": "test_input", "type": "valid"}
        result = self.core.process_data(test_data)
        
        assert result["status"] == "success"
        assert result["input_type"] == "dict"
        assert result["processed_data"]["input"] == "TEST_INPUT"
        assert result["processed_data"]["type"] == "VALID"
    
    def test_process_data_empty_input(self):
        """Test that data processing handles empty input gracefully."""
        test_data = {}
        result = self.core.process_data(test_data)
        
        assert result["status"] == "success"
        assert result["input_type"] == "dict"
        assert result["processed_data"] == {}
    
    def test_process_data_invalid_type(self):
        """Test that data processing handles invalid types correctly."""
        test_data = ["invalid", "list", "input"]
        result = self.core.process_data(test_data)
        
        assert result["status"] == "error"
        assert "Unsupported data type" in result["message"]
    
    def test_process_data_large_input(self):
        """Test processing of large input data."""
        large_string = "x" * 10000
        result = self.core.process_data(large_string)
        
        assert result["status"] == "success"
        assert result["input_type"] == "string"
        assert result["processed_data"] == f"processed_{large_string}"
        assert result["length"] == 10000
    
    def test_process_data_unicode_input(self):
        """Test processing of Unicode input."""
        unicode_data = {"input": "测试数据🧪", "type": "unicode"}
        result = self.core.process_data(unicode_data)
        
        assert result["status"] == "success"
        assert result["processed_data"]["input"] == "测试数据🧪"
        assert result["processed_data"]["type"] == "UNICODE"
    
    def test_process_data_none_input(self):
        """Test processing of None input."""
        result = self.core.process_data(None)
        
        assert result["status"] == "error"
        assert result["message"] == "No data provided"
    
    def test_process_simple_data_function(self):
        """Test the module-level process_simple_data function."""
        result = process_simple_data("test")
        assert result == "processed_test"
        
        result = process_simple_data("")
        assert result == ""
    
    def test_cached_operation(self):
        """Test the cached operation functionality."""
        # First call should cache the result
        result1 = self.core.cached_operation("test_input")
        assert result1 == "cached_result_test_input"
        
        # Second call should return cached result (faster)
        start_time = time.time()
        result2 = self.core.cached_operation("test_input")
        execution_time = time.time() - start_time
        
        assert result2 == "cached_result_test_input"
        assert execution_time < 0.05  # Should be much faster due to caching


class TestGenesisCoreErrorHandling:
    """Test class for error handling in genesis_core module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.example.com',
            'timeout': 30
        }
        self.core = GenesisCore(self.mock_config)
    
    @patch('requests.post')
    def test_network_error_handling(self, mock_post):
        """Test network error handling during API requests."""
        mock_post.side_effect = ConnectionError("Network error")
        
        with pytest.raises(ConnectionError, match="Network connection failed"):
            self.core.make_api_request('/test')
    
    @patch('requests.post')
    def test_timeout_handling(self, mock_post):
        """Test timeout error handling during API requests."""
        mock_post.side_effect = TimeoutError("Request timeout")
        
        with pytest.raises(TimeoutError, match="Request timeout"):
            self.core.make_api_request('/test')
    
    @patch('requests.post')
    def test_authentication_error_handling(self, mock_post):
        """Test authentication error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response
        
        with pytest.raises(ValueError, match="Authentication failed"):
            self.core.make_api_request('/test')
    
    @patch('requests.post')
    def test_permission_error_handling(self, mock_post):
        """Test permission error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_post.return_value = mock_response
        
        with pytest.raises(PermissionError, match="Access forbidden"):
            self.core.make_api_request('/test')
    
    @patch('requests.post')
    def test_invalid_response_handling(self, mock_post):
        """Test handling of invalid API responses."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        with pytest.raises(Exception, match="API request failed with status 500"):
            self.core.make_api_request('/test')


class TestGenesisCoreEdgeCases:
    """Test class for edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.example.com'
        }
        self.core = GenesisCore(self.mock_config)
    
    def test_maximum_input_size(self):
        """Test processing of input at maximum size boundary."""
        max_size_input = "x" * MAX_INPUT_SIZE
        result = self.core.process_data(max_size_input)
        
        assert result["status"] == "success"
        assert result["length"] == MAX_INPUT_SIZE
    
    def test_minimum_input_size(self):
        """Test processing of inputs at minimum size boundary."""
        min_size_input = "x" * MIN_INPUT_SIZE
        result = self.core.process_data(min_size_input)
        
        assert result["status"] == "success"
        assert result["length"] == MIN_INPUT_SIZE
    
    def test_concurrent_requests(self):
        """Test thread safety with concurrent requests."""
        def process_data_worker(data):
            return self.core.process_data(f"test_{data}")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_data_worker, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        # All requests should succeed
        assert all(result["status"] == "success" for result in results)
        
        # Results should be unique
        processed_values = [result["processed_data"] for result in results]
        assert len(set(processed_values)) == len(processed_values)
    
    def test_memory_usage_large_dataset(self):
        """Test memory efficiency with large datasets."""
        large_data = {"key_" + str(i): "value_" + str(i) for i in range(1000)}
        
        # Should not raise memory errors
        result = self.core.process_data(large_data)
        assert result["status"] == "success"
        assert len(result["processed_data"]) == 1000
    
    @patch('requests.post')
    def test_rate_limiting_behavior(self, mock_post):
        """Test rate limiting error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_post.return_value = mock_response
        
        with pytest.raises(Exception, match="API request failed with status 429"):
            self.core.make_api_request('/test')


class TestGenesisCoreIntegration:
    """Test class for integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.example.com',
            'timeout': 30
        }
        self.core = GenesisCore(self.mock_config)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Test input validation
        test_data = {"input": "test_data", "type": "valid"}
        is_valid = self.core.validate_input(test_data)
        assert is_valid is True
        
        # Test data processing
        result = self.core.process_data(test_data)
        assert result["status"] == "success"
        
        # Test system info
        info = self.core.get_system_info()
        assert info["module"] == "genesis_core"
        assert info["version"] == "1.0.0"
    
    def test_configuration_loading(self):
        """Test configuration handling."""
        # Test validate_config function
        valid_config = {'api_key': 'test', 'base_url': 'https://test.com'}
        assert validate_config(valid_config) is True
        
        invalid_config = {'api_key': 'test'}  # Missing base_url
        assert validate_config(invalid_config) is False
        
        # Test default config
        assert 'base_url' in DEFAULT_CONFIG
        assert 'timeout' in DEFAULT_CONFIG
    
    @patch('app.ai_backend.genesis_core.logger')
    def test_logging_functionality(self, mock_logger):
        """Test logging functionality."""
        config = {'api_key': 'test', 'base_url': 'https://test.com'}
        GenesisCore(config)
        
        # Verify logging was called
        mock_logger.info.assert_called_once()
        args = mock_logger.info.call_args[0][0]
        assert "GenesisCore initialized" in args
    
    def test_caching_behavior(self):
        """Test caching mechanism."""
        # Test cache miss
        result1 = self.core.cached_operation("test1")
        assert result1 == "cached_result_test1"
        
        # Test cache hit
        result2 = self.core.cached_operation("test1")
        assert result2 == "cached_result_test1"
        
        # Test different input (cache miss)
        result3 = self.core.cached_operation("test2")
        assert result3 == "cached_result_test2"
    
    @patch('requests.post')
    def test_successful_api_request(self, mock_post):
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "data": "test"}
        mock_post.return_value = mock_response
        
        result = self.core.make_api_request('/test', {'key': 'value'})
        assert result["status"] == "success"
        assert result["data"] == "test"
        
        # Verify request was made correctly
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['json'] == {'key': 'value'}
        assert call_kwargs['timeout'] == 30
        assert 'Authorization' in call_kwargs['headers']


class TestGenesisCorePerformance:
    """Test class for performance-related tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.example.com'
        }
        self.core = GenesisCore(self.mock_config)
    
    def test_response_time_within_limits(self):
        """Test that data processing completes within reasonable time."""
        test_data = {"key": "value", "number": 42}
        
        start_time = time.time()
        result = self.core.process_data(test_data)
        execution_time = time.time() - start_time
        
        assert result["status"] == "success"
        assert execution_time < 1.0  # Should complete within 1 second
    
    def test_memory_usage_within_limits(self):
        """Test memory usage patterns."""
        # Process multiple datasets to check memory usage
        datasets = [
            {"key": f"value_{i}", "number": i} 
            for i in range(100)
        ]
        
        results = []
        for dataset in datasets:
            results.append(self.core.process_data(dataset))
        
        # All should succeed
        assert all(result["status"] == "success" for result in results)
    
    def test_cpu_usage_efficiency(self):
        """Test CPU usage efficiency."""
        # Test processing multiple items efficiently
        items = [f"item_{i}" for i in range(50)]
        
        start_time = time.time()
        results = [self.core.process_data(item) for item in items]
        execution_time = time.time() - start_time
        
        assert len(results) == 50
        assert execution_time < 5.0  # Should complete 50 items in under 5 seconds
    
    def test_caching_performance_improvement(self):
        """Test that caching improves performance."""
        # First call (cache miss)
        start_time = time.time()
        result1 = self.core.cached_operation("performance_test")
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = self.core.cached_operation("performance_test")
        second_call_time = time.time() - start_time
        
        assert result1 == result2
        assert second_call_time < first_call_time * 0.5  # Should be at least 50% faster


class TestGenesisCoreValidation:
    """Test class for input validation and sanitization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.example.com'
        }
        self.core = GenesisCore(self.mock_config)
    
    def test_input_validation_valid_data(self):
        """Test that valid input data is accepted."""
        valid_inputs = [
            {"key": "value"},
            {"number": 42},
            {"list": [1, 2, 3]},
            "normal_string",
            {"nested": {"data": "test"}}
        ]
        
        for input_data in valid_inputs:
            assert self.core.validate_input(input_data) is True
    
    def test_input_validation_invalid_data(self):
        """Test that invalid data is rejected."""
        invalid_inputs = [
            None,
            "",
            "   ",  # Empty string with spaces
            {"sql_injection": "test"},
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd"
        ]
        
        for input_data in invalid_inputs:
            assert self.core.validate_input(input_data) is False
    
    def test_input_sanitization(self):
        """Test input sanitization functionality."""
        dangerous_inputs = [
            ("<script>alert('xss')</script>", "alert('xss')"),
            ("'; DROP TABLE users; --", "'; users; --"),
            ("../../../etc/passwd", "etc/passwd"),
            ('"; DROP TABLE users; --', '"; users; --')
        ]
        
        for dangerous_input, expected_output in dangerous_inputs:
            sanitized = self.core.sanitize_input(dangerous_input)
            assert sanitized == expected_output
    
    def test_sanitization_preserves_safe_content(self):
        """Test that sanitization preserves safe content."""
        safe_inputs = [
            "normal text",
            "email@domain.com",
            "123-456-7890",
            "Safe content with numbers 123"
        ]
        
        for safe_input in safe_inputs:
            sanitized = self.core.sanitize_input(safe_input)
            assert sanitized == safe_input


class TestGenesisCoreUtilityFunctions:
    """Test class for utility functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.example.com'
        }
        self.core = GenesisCore(self.mock_config)
    
    def test_transform_value_function(self):
        """Test the _transform_value utility function."""
        # Test string transformation
        assert self.core._transform_value("test") == "TEST"
        
        # Test number transformation
        assert self.core._transform_value(5) == 10
        assert self.core._transform_value(2.5) == 5.0
        
        # Test list transformation
        assert self.core._transform_value([1, 2, 3]) == [2, 4, 6]
        
        # Test passthrough for other types
        assert self.core._transform_value(True) is True
        assert self.core._transform_value(None) is None
    
    def test_get_system_info(self):
        """Test system information utility."""
        info = self.core.get_system_info()
        
        assert info["module"] == "genesis_core"
        assert info["version"] == "1.0.0"
        assert isinstance(info["config_keys"], list)
        assert "api_key" in info["config_keys"]
        assert isinstance(info["cache_size"], int)
    
    def test_validate_config_function(self):
        """Test configuration validation utility."""
        # Valid config
        valid_config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com'
        }
        assert validate_config(valid_config) is True
        
        # Invalid configs
        invalid_configs = [
            {'api_key': 'test'},  # Missing base_url
            {'base_url': 'https://api.test.com'},  # Missing api_key
            {},  # Missing both
            {'other_key': 'value'}  # Wrong keys
        ]
        
        for config in invalid_configs:
            assert validate_config(config) is False


# Additional test fixtures and utilities
@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary for tests."""
    return {
        'api_key': 'test_api_key',
        'base_url': 'https://api.test.com',
        'timeout': 30,
        'retries': 3
    }


@pytest.fixture
def mock_response():
    """Create a mock HTTP response object."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"status": "success", "data": {}}
    return response


@pytest.fixture
def sample_data():
    """Return sample datasets for testing."""
    return {
        "simple": {"key": "value"},
        "complex": {
            "nested": {"data": [1, 2, 3]},
            "metadata": {"timestamp": "2023-01-01T00:00:00Z"}
        },
        "edge_cases": {
            "empty": {},
            "null_values": {"key": None},
            "unicode": {"text": "测试数据🧪"}
        }
    }


@pytest.fixture
def genesis_core_instance(mock_config):
    """Create a GenesisCore instance for testing."""
    return GenesisCore(mock_config)


# Test parametrization examples
@pytest.mark.parametrize("input_value,expected_output", [
    ("test", "processed_test"),
    ("", ""),
    ("unicode_测试", "processed_unicode_测试"),
    ("123", "processed_123")
])
def test_parameterized_processing(input_value, expected_output):
    """Test process_simple_data function with various inputs."""
    result = process_simple_data(input_value)
    assert result == expected_output


@pytest.mark.parametrize("config,should_be_valid", [
    ({'api_key': 'test', 'base_url': 'https://test.com'}, True),
    ({'api_key': 'test'}, False),
    ({'base_url': 'https://test.com'}, False),
    ({}, False),
    ({'api_key': 'test', 'base_url': 'https://test.com', 'extra': 'value'}, True)
])
def test_config_validation_parametrized(config, should_be_valid):
    """Test configuration validation with various config combinations."""
    assert validate_config(config) == should_be_valid


# Performance benchmarks
@pytest.mark.benchmark
def test_performance_benchmark(genesis_core_instance):
    """Performance benchmark test for data processing."""
    test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    
    start_time = time.time()
    for _ in range(100):
        genesis_core_instance.process_data(test_data)
    execution_time = time.time() - start_time
    
    # Should process 100 items in under 1 second
    assert execution_time < 1.0


# Integration test markers
@pytest.mark.integration
@patch('requests.post')
def test_integration_scenario(mock_post, genesis_core_instance):
    """Integration test for GenesisCore with external dependencies."""
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success", "result": "integrated"}
    mock_post.return_value = mock_response
    
    # Test full workflow
    test_data = {"integration": "test"}
    
    # Validate input
    assert genesis_core_instance.validate_input(test_data) is True
    
    # Process data
    processed = genesis_core_instance.process_data(test_data)
    assert processed["status"] == "success"
    
    # Make API request
    api_result = genesis_core_instance.make_api_request('/integration', test_data)
    assert api_result["status"] == "success"
    assert api_result["result"] == "integrated"


# Slow test markers
@pytest.mark.slow
def test_slow_operation(genesis_core_instance):
    """Test for long-running operations."""
    # Test processing many items
    large_dataset = [{"item": i} for i in range(1000)]
    
    results = []
    for item in large_dataset:
        results.append(genesis_core_instance.process_data(item))
    
    assert len(results) == 1000
    assert all(result["status"] == "success" for result in results)


# Security tests
@pytest.mark.security
def test_security_validation(genesis_core_instance):
    """Test security-related validation and sanitization."""
    malicious_inputs = [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "../../../etc/passwd",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>"
    ]
    
    for malicious_input in malicious_inputs:
        # Should not validate
        assert genesis_core_instance.validate_input(malicious_input) is False
        
        # Should be sanitized
        sanitized = genesis_core_instance.sanitize_input(malicious_input)
        assert sanitized != malicious_input
        assert '<script>' not in sanitized
        assert 'DROP TABLE' not in sanitized


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])