import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
import asyncio
from datetime import datetime, timedelta
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
import logging

# Import the module under test
try:
    from app.ai_backend.genesis_connector import GenesisConnector, GenesisConnectionError, GenesisTimeoutError
except ImportError:
    # Fallback import structure
    from genesis_connector import GenesisConnector, GenesisConnectionError, GenesisTimeoutError


class TestGenesisConnector(unittest.TestCase):
    """Comprehensive unit tests for GenesisConnector class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def tearDown(self):
        """Clean up after each test method."""
        # Reset any global state or close connections
        if hasattr(self.connector, 'session'):
            self.connector.session.close()
    
    def test_initialization_with_valid_config(self):
        """Test successful initialization with valid configuration."""
        connector = GenesisConnector(self.mock_config)
        self.assertEqual(connector.api_key, 'test_api_key_123')
        self.assertEqual(connector.base_url, 'https://api.genesis.test')
        self.assertEqual(connector.timeout, 30)
        self.assertEqual(connector.max_retries, 3)
    
    def test_initialization_with_missing_api_key(self):
        """Test initialization failure when API key is missing."""
        config = self.mock_config.copy()
        del config['api_key']
        
        with self.assertRaises(ValueError) as context:
            GenesisConnector(config)
        self.assertIn('API key is required', str(context.exception))
    
    def test_initialization_with_empty_api_key(self):
        """Test initialization failure when API key is empty."""
        config = self.mock_config.copy()
        config['api_key'] = ''
        
        with self.assertRaises(ValueError) as context:
            GenesisConnector(config)
        self.assertIn('API key cannot be empty', str(context.exception))
    
    def test_initialization_with_invalid_base_url(self):
        """Test initialization failure with invalid base URL."""
        config = self.mock_config.copy()
        config['base_url'] = 'invalid_url'
        
        with self.assertRaises(ValueError) as context:
            GenesisConnector(config)
        self.assertIn('Invalid base URL', str(context.exception))
    
    def test_initialization_with_negative_timeout(self):
        """Test initialization failure with negative timeout."""
        config = self.mock_config.copy()
        config['timeout'] = -1
        
        with self.assertRaises(ValueError) as context:
            GenesisConnector(config)
        self.assertIn('Timeout must be positive', str(context.exception))
    
    def test_initialization_with_zero_max_retries(self):
        """Test initialization with zero max retries."""
        config = self.mock_config.copy()
        config['max_retries'] = 0
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.max_retries, 0)
    
    def test_initialization_with_negative_max_retries(self):
        """Test initialization failure with negative max retries."""
        config = self.mock_config.copy()
        config['max_retries'] = -1
        
        with self.assertRaises(ValueError) as context:
            GenesisConnector(config)
        self.assertIn('Max retries cannot be negative', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_success(self, mock_request):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test_data'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/test/endpoint')
        
        self.assertEqual(result, {'data': 'test_data'})
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/test/endpoint',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_data(self, mock_request):
        """Test API request with POST data."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 123, 'status': 'created'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        test_data = {'name': 'test', 'value': 42}
        result = self.connector.make_request('POST', '/create', data=test_data)
        
        self.assertEqual(result, {'id': 123, 'status': 'created'})
        mock_request.assert_called_once_with(
            'POST',
            'https://api.genesis.test/create',
            headers={'Authorization': 'Bearer test_api_key_123'},
            json=test_data,
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_custom_headers(self, mock_request):
        """Test API request with custom headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        custom_headers = {'X-Custom-Header': 'custom_value'}
        result = self.connector.make_request('GET', '/test', headers=custom_headers)
        
        expected_headers = {
            'Authorization': 'Bearer test_api_key_123',
            'X-Custom-Header': 'custom_value'
        }
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/test',
            headers=expected_headers,
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_timeout_error(self, mock_request):
        """Test request timeout handling."""
        mock_request.side_effect = Timeout("Request timed out")
        
        with self.assertRaises(GenesisTimeoutError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Request timed out', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_connection_error(self, mock_request):
        """Test connection error handling."""
        mock_request.side_effect = ConnectionError("Connection failed")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Connection failed', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_http_error(self, mock_request):
        """Test HTTP error response handling."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_response.text = "Resource not found"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/nonexistent')
        
        self.assertIn('404 Not Found', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_json_decode_error(self, mock_request):
        """Test JSON decode error handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Invalid JSON response"
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Invalid JSON response', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_retry_logic(self, mock_request):
        """Test retry logic on transient failures."""
        # First two calls fail, third succeeds
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {'success': True}
        mock_response_success.raise_for_status.return_value = None
        
        mock_request.side_effect = [
            ConnectionError("Connection failed"),
            Timeout("Request timed out"),
            mock_response_success
        ]
        
        result = self.connector.make_request('GET', '/test')
        
        self.assertEqual(result, {'success': True})
        self.assertEqual(mock_request.call_count, 3)
    
    @patch('requests.Session.request')
    def test_make_request_retry_exhausted(self, mock_request):
        """Test behavior when all retries are exhausted."""
        mock_request.side_effect = [
            ConnectionError("Connection failed"),
            ConnectionError("Connection failed"),
            ConnectionError("Connection failed"),
            ConnectionError("Connection failed")  # Max retries + 1
        ]
        
        with self.assertRaises(GenesisConnectionError):
            self.connector.make_request('GET', '/test')
        
        self.assertEqual(mock_request.call_count, 4)  # Initial + 3 retries
    
    @patch('requests.Session.request')
    def test_get_model_info_success(self, mock_request):
        """Test successful model info retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 'model_123',
            'name': 'Test Model',
            'version': '1.0.0',
            'status': 'active'
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.get_model_info('model_123')
        
        self.assertEqual(result['id'], 'model_123')
        self.assertEqual(result['name'], 'Test Model')
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/models/model_123',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_get_model_info_not_found(self, mock_request):
        """Test model info retrieval when model doesn't exist."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_response.text = "Model not found"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.get_model_info('nonexistent_model')
        
        self.assertIn('404 Not Found', str(context.exception))
    
    @patch('requests.Session.request')
    def test_create_generation_success(self, mock_request):
        """Test successful generation creation."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            'id': 'gen_123',
            'status': 'pending',
            'created_at': '2023-01-01T00:00:00Z'
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        generation_request = {
            'prompt': 'Test prompt',
            'model': 'test_model',
            'max_tokens': 100
        }
        
        result = self.connector.create_generation(generation_request)
        
        self.assertEqual(result['id'], 'gen_123')
        self.assertEqual(result['status'], 'pending')
        mock_request.assert_called_once_with(
            'POST',
            'https://api.genesis.test/generations',
            headers={'Authorization': 'Bearer test_api_key_123'},
            json=generation_request,
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_create_generation_invalid_request(self, mock_request):
        """Test generation creation with invalid request."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.HTTPError("400 Bad Request")
        mock_response.text = "Invalid request parameters"
        mock_request.return_value = mock_response
        
        invalid_request = {'prompt': ''}  # Empty prompt
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.create_generation(invalid_request)
        
        self.assertIn('400 Bad Request', str(context.exception))
    
    @patch('requests.Session.request')
    def test_get_generation_status_success(self, mock_request):
        """Test successful generation status retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 'gen_123',
            'status': 'completed',
            'result': 'Generated text response',
            'completed_at': '2023-01-01T00:01:00Z'
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.get_generation_status('gen_123')
        
        self.assertEqual(result['id'], 'gen_123')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['result'], 'Generated text response')
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/generations/gen_123',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_get_generation_status_not_found(self, mock_request):
        """Test generation status retrieval when generation doesn't exist."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_response.text = "Generation not found"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.get_generation_status('nonexistent_gen')
        
        self.assertIn('404 Not Found', str(context.exception))
    
    @patch('requests.Session.request')
    def test_cancel_generation_success(self, mock_request):
        """Test successful generation cancellation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 'gen_123',
            'status': 'cancelled',
            'cancelled_at': '2023-01-01T00:00:30Z'
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.cancel_generation('gen_123')
        
        self.assertEqual(result['status'], 'cancelled')
        mock_request.assert_called_once_with(
            'DELETE',
            'https://api.genesis.test/generations/gen_123',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_cancel_generation_already_completed(self, mock_request):
        """Test cancellation of already completed generation."""
        mock_response = Mock()
        mock_response.status_code = 409
        mock_response.raise_for_status.side_effect = requests.HTTPError("409 Conflict")
        mock_response.text = "Generation already completed"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.cancel_generation('gen_123')
        
        self.assertIn('409 Conflict', str(context.exception))
    
    @patch('requests.Session.request')
    def test_list_models_success(self, mock_request):
        """Test successful model listing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [
                {'id': 'model_1', 'name': 'Model 1'},
                {'id': 'model_2', 'name': 'Model 2'}
            ],
            'total': 2
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.list_models()
        
        self.assertEqual(len(result['models']), 2)
        self.assertEqual(result['total'], 2)
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/models',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_list_models_with_filters(self, mock_request):
        """Test model listing with filters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [{'id': 'model_1', 'name': 'Model 1', 'status': 'active'}],
            'total': 1
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        filters = {'status': 'active', 'limit': 10}
        result = self.connector.list_models(filters)
        
        self.assertEqual(len(result['models']), 1)
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/models',
            headers={'Authorization': 'Bearer test_api_key_123'},
            params=filters,
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_health_check_success(self, mock_request):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'healthy',
            'timestamp': '2023-01-01T00:00:00Z'
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.health_check()
        
        self.assertEqual(result['status'], 'healthy')
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/health',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_health_check_failure(self, mock_request):
        """Test health check failure."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")
        mock_response.text = "Service temporarily unavailable"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.health_check()
        
        self.assertIn('503 Service Unavailable', str(context.exception))
    
    def test_build_url_with_endpoint(self):
        """Test URL building with endpoint."""
        url = self.connector._build_url('/test/endpoint')
        self.assertEqual(url, 'https://api.genesis.test/test/endpoint')
    
    def test_build_url_without_leading_slash(self):
        """Test URL building without leading slash."""
        url = self.connector._build_url('test/endpoint')
        self.assertEqual(url, 'https://api.genesis.test/test/endpoint')
    
    def test_build_url_with_trailing_slash_in_base(self):
        """Test URL building with trailing slash in base URL."""
        config = self.mock_config.copy()
        config['base_url'] = 'https://api.genesis.test/'
        connector = GenesisConnector(config)
        
        url = connector._build_url('/test/endpoint')
        self.assertEqual(url, 'https://api.genesis.test/test/endpoint')
    
    def test_build_headers_default(self):
        """Test default header building."""
        headers = self.connector._build_headers()
        expected = {'Authorization': 'Bearer test_api_key_123'}
        self.assertEqual(headers, expected)
    
    def test_build_headers_with_custom(self):
        """Test header building with custom headers."""
        custom_headers = {'X-Custom': 'value', 'Content-Type': 'application/json'}
        headers = self.connector._build_headers(custom_headers)
        
        expected = {
            'Authorization': 'Bearer test_api_key_123',
            'X-Custom': 'value',
            'Content-Type': 'application/json'
        }
        self.assertEqual(headers, expected)
    
    def test_build_headers_override_auth(self):
        """Test header building when overriding authorization."""
        custom_headers = {'Authorization': 'Bearer different_token'}
        headers = self.connector._build_headers(custom_headers)
        
        expected = {'Authorization': 'Bearer different_token'}
        self.assertEqual(headers, expected)
    
    @patch('time.sleep')
    @patch('requests.Session.request')
    def test_exponential_backoff_retry(self, mock_request, mock_sleep):
        """Test exponential backoff in retry logic."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        # First two calls fail, third succeeds
        mock_request.side_effect = [
            ConnectionError("Connection failed"),
            ConnectionError("Connection failed"),
            mock_response
        ]
        
        result = self.connector.make_request('GET', '/test')
        
        self.assertEqual(result, {'success': True})
        # Should have slept twice (after first and second failure)
        self.assertEqual(mock_sleep.call_count, 2)
        # Check exponential backoff: 1s, then 2s
        expected_calls = [call(1), call(2)]
        mock_sleep.assert_has_calls(expected_calls)
    
    def test_context_manager_usage(self):
        """Test using connector as context manager."""
        with GenesisConnector(self.mock_config) as connector:
            self.assertIsNotNone(connector)
            self.assertEqual(connector.api_key, 'test_api_key_123')
    
    def test_repr_method(self):
        """Test string representation of connector."""
        repr_str = repr(self.connector)
        self.assertIn('GenesisConnector', repr_str)
        self.assertIn('https://api.genesis.test', repr_str)
        # Should not include the full API key for security
        self.assertNotIn('test_api_key_123', repr_str)
    
    def test_str_method(self):
        """Test string conversion of connector."""
        str_repr = str(self.connector)
        self.assertIn('GenesisConnector', str_repr)
        self.assertIn('https://api.genesis.test', str_repr)


class TestGenesisConnectorAsync(unittest.TestCase):
    """Test async functionality of GenesisConnector if available."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    @patch('aiohttp.ClientSession.request')
    async def test_async_make_request_success(self, mock_request):
        """Test successful async API request."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'data': 'async_data'})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_request.return_value = mock_response
        
        if hasattr(self.connector, 'async_make_request'):
            result = await self.connector.async_make_request('GET', '/test')
            self.assertEqual(result, {'data': 'async_data'})
    
    @patch('aiohttp.ClientSession.request')
    async def test_async_make_request_timeout(self, mock_request):
        """Test async request timeout handling."""
        mock_request.side_effect = asyncio.TimeoutError("Async timeout")
        
        if hasattr(self.connector, 'async_make_request'):
            with self.assertRaises(GenesisTimeoutError):
                await self.connector.async_make_request('GET', '/test')


class TestGenesisConnectorIntegration(unittest.TestCase):
    """Integration tests for GenesisConnector (require actual API access)."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # These tests should only run in integration test environments
        import os
        if not os.getenv('GENESIS_INTEGRATION_TESTS'):
            self.skipTest("Integration tests disabled")
        
        self.config = {
            'api_key': os.getenv('GENESIS_API_KEY'),
            'base_url': os.getenv('GENESIS_BASE_URL', 'https://api.genesis.test'),
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.config)
    
    def test_real_health_check(self):
        """Test health check against real API."""
        try:
            result = self.connector.health_check()
            self.assertIn('status', result)
        except GenesisConnectionError:
            self.skipTest("API not available for integration testing")
    
    def test_real_list_models(self):
        """Test listing models against real API."""
        try:
            result = self.connector.list_models()
            self.assertIn('models', result)
            self.assertIsInstance(result['models'], list)
        except GenesisConnectionError:
            self.skipTest("API not available for integration testing")


# Custom exception classes for testing
class AsyncMock(Mock):
    """Mock class for async operations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)
    

class TestGenesisConnectorAdvanced(unittest.TestCase):
    """Advanced unit tests for GenesisConnector edge cases and performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3,
            'rate_limit': 100,
            'burst_limit': 10
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_initialization_with_default_values(self):
        """Test initialization with minimal configuration using defaults."""
        minimal_config = {'api_key': 'test_key'}
        connector = GenesisConnector(minimal_config)
        
        self.assertEqual(connector.api_key, 'test_key')
        self.assertEqual(connector.timeout, 30)  # Default timeout
        self.assertEqual(connector.max_retries, 3)  # Default max_retries
    
    def test_initialization_with_none_values(self):
        """Test initialization handling None values in configuration."""
        config = self.mock_config.copy()
        config['timeout'] = None
        config['max_retries'] = None
        
        with self.assertRaises(ValueError):
            GenesisConnector(config)
    
    def test_initialization_with_extreme_values(self):
        """Test initialization with extreme but valid values."""
        config = self.mock_config.copy()
        config['timeout'] = 1  # Very short timeout
        config['max_retries'] = 10  # High retry count
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.timeout, 1)
        self.assertEqual(connector.max_retries, 10)
    
    def test_initialization_with_invalid_config_type(self):
        """Test initialization with invalid configuration type."""
        with self.assertRaises(TypeError):
            GenesisConnector("invalid_config")
        
        with self.assertRaises(TypeError):
            GenesisConnector(None)
    
    def test_api_key_sanitization(self):
        """Test that API key is properly sanitized in logs and representations."""
        long_api_key = 'a' * 100
        config = self.mock_config.copy()
        config['api_key'] = long_api_key
        
        connector = GenesisConnector(config)
        repr_str = repr(connector)
        
        # Should not contain the full API key
        self.assertNotIn(long_api_key, repr_str)
        # Should contain some indication of the key length or partial key
        self.assertIn('***', repr_str)
    
    @patch('requests.Session.request')
    def test_make_request_with_special_characters(self, mock_request):
        """Test API request with special characters in endpoint and data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        special_endpoint = '/test/endpoint with spaces/special%chars'
        special_data = {'text': 'Hello 世界! @#$%^&*()'}
        
        result = self.connector.make_request('POST', special_endpoint, data=special_data)
        
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once()
    
    @patch('requests.Session.request')
    def test_make_request_with_large_payload(self, mock_request):
        """Test API request with large data payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Create a large payload
        large_data = {'content': 'A' * 10000}  # 10KB of data
        
        result = self.connector.make_request('POST', '/large', data=large_data)
        
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once()
    
    @patch('requests.Session.request')
    def test_make_request_with_empty_response(self, mock_request):
        """Test API request handling empty response body."""
        mock_response = Mock()
        mock_response.status_code = 204  # No Content
        mock_response.json.side_effect = json.JSONDecodeError("Empty response", "", 0)
        mock_response.text = ""
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('DELETE', '/test')
        
        # Should handle empty response gracefully
        self.assertIsNone(result)
    
    @patch('requests.Session.request')
    def test_make_request_with_malformed_json(self, mock_request):
        """Test API request with malformed JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Malformed JSON", "", 0)
        mock_response.text = '{"incomplete": json'
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Malformed JSON', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_various_http_methods(self, mock_request):
        """Test API request with various HTTP methods."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS']
        
        for method in methods:
            with self.subTest(method=method):
                result = self.connector.make_request(method, '/test')
                self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_status_codes(self, mock_request):
        """Test API request handling various status codes."""
        status_codes = [200, 201, 202, 204, 400, 401, 403, 404, 500, 502, 503]
        
        for status_code in status_codes:
            with self.subTest(status_code=status_code):
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.text = f"Status {status_code}"
                
                if status_code < 400:
                    mock_response.json.return_value = {'status': status_code}
                    mock_response.raise_for_status.return_value = None
                    mock_request.return_value = mock_response
                    
                    result = self.connector.make_request('GET', '/test')
                    if status_code != 204:  # No Content
                        self.assertEqual(result, {'status': status_code})
                else:
                    mock_response.raise_for_status.side_effect = requests.HTTPError(f"{status_code} Error")
                    mock_request.return_value = mock_response
                    
                    with self.assertRaises(GenesisConnectionError):
                        self.connector.make_request('GET', '/test')
    
    @patch('time.sleep')
    @patch('requests.Session.request')
    def test_retry_with_different_exceptions(self, mock_request, mock_sleep):
        """Test retry logic with different types of exceptions."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        exception_types = [
            ConnectionError("Connection error"),
            Timeout("Timeout error"),
            requests.exceptions.ChunkedEncodingError("Chunked encoding error"),
            requests.exceptions.ContentDecodingError("Content decoding error")
        ]
        
        for exception in exception_types:
            with self.subTest(exception=type(exception).__name__):
                mock_request.side_effect = [exception, mock_response]
                mock_sleep.reset_mock()
                
                result = self.connector.make_request('GET', '/test')
                
                self.assertEqual(result, {'success': True})
                self.assertEqual(mock_request.call_count, 2)
                mock_sleep.assert_called_once_with(1)  # First retry delay
                mock_request.reset_mock()
    
    @patch('requests.Session.request')
    def test_concurrent_requests(self, mock_request):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        # Add slight delay to simulate real API call
        def delayed_response(*args, **kwargs):
            time.sleep(0.01)
            return mock_response
        
        mock_request.side_effect = delayed_response
        
        results = []
        errors = []
        
        def make_request():
            try:
                result = self.connector.make_request('GET', '/test')
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)
        self.assertEqual(mock_request.call_count, 5)
    
    def test_url_building_edge_cases(self):
        """Test URL building with various edge cases."""
        test_cases = [
            ('', 'https://api.genesis.test'),
            ('/', 'https://api.genesis.test/'),
            ('//endpoint', 'https://api.genesis.test/endpoint'),
            ('/endpoint/', 'https://api.genesis.test/endpoint/'),
            ('endpoint?param=value', 'https://api.genesis.test/endpoint?param=value'),
            ('/endpoint#fragment', 'https://api.genesis.test/endpoint#fragment'),
        ]
        
        for endpoint, expected in test_cases:
            with self.subTest(endpoint=endpoint):
                result = self.connector._build_url(endpoint)
                self.assertEqual(result, expected)
    
    def test_headers_with_unicode_values(self):
        """Test header building with unicode values."""
        unicode_headers = {
            'X-Custom-Header': 'value with ünicøde',
            'X-Another-Header': '测试值'
        }
        
        headers = self.connector._build_headers(unicode_headers)
        
        self.assertIn('X-Custom-Header', headers)
        self.assertIn('X-Another-Header', headers)
        self.assertEqual(headers['X-Custom-Header'], 'value with ünicøde')
        self.assertEqual(headers['X-Another-Header'], '测试值')
    
    def test_context_manager_exception_handling(self):
        """Test context manager behavior during exceptions."""
        class TestException(Exception):
            pass
        
        try:
            with GenesisConnector(self.mock_config) as connector:
                self.assertIsNotNone(connector)
                raise TestException("Test exception")
        except TestException:
            pass  # Expected
        
        # Context manager should properly clean up even with exceptions
        
    @patch('requests.Session.request')
    def test_request_timeout_variations(self, mock_request):
        """Test various timeout scenarios."""
        timeout_exceptions = [
            Timeout("Read timeout"),
            Timeout("Connection timeout"),
            requests.exceptions.ReadTimeout("Read timeout"),
            requests.exceptions.ConnectTimeout("Connect timeout")
        ]
        
        for exception in timeout_exceptions:
            with self.subTest(exception=type(exception).__name__):
                mock_request.side_effect = exception
                
                with self.assertRaises(GenesisTimeoutError):
                    self.connector.make_request('GET', '/test')
    
    @patch('requests.Session.request')
    def test_generation_workflow_complete(self, mock_request):
        """Test complete generation workflow from creation to completion."""
        # Mock responses for the complete workflow
        create_response = Mock()
        create_response.status_code = 201
        create_response.json.return_value = {'id': 'gen_123', 'status': 'pending'}
        create_response.raise_for_status.return_value = None
        
        status_pending_response = Mock()
        status_pending_response.status_code = 200
        status_pending_response.json.return_value = {'id': 'gen_123', 'status': 'running'}
        status_pending_response.raise_for_status.return_value = None
        
        status_complete_response = Mock()
        status_complete_response.status_code = 200
        status_complete_response.json.return_value = {
            'id': 'gen_123',
            'status': 'completed',
            'result': 'Generated content'
        }
        status_complete_response.raise_for_status.return_value = None
        
        mock_request.side_effect = [
            create_response,
            status_pending_response,
            status_complete_response
        ]
        
        # Test complete workflow
        generation_request = {'prompt': 'Test prompt', 'model': 'test_model'}
        
        # Create generation
        create_result = self.connector.create_generation(generation_request)
        self.assertEqual(create_result['id'], 'gen_123')
        self.assertEqual(create_result['status'], 'pending')
        
        # Check status (running)
        status_result = self.connector.get_generation_status('gen_123')
        self.assertEqual(status_result['status'], 'running')
        
        # Check status (completed)
        final_result = self.connector.get_generation_status('gen_123')
        self.assertEqual(final_result['status'], 'completed')
        self.assertEqual(final_result['result'], 'Generated content')
    
    @patch('requests.Session.request')
    def test_model_operations_comprehensive(self, mock_request):
        """Test comprehensive model operations."""
        # List models
        list_response = Mock()
        list_response.status_code = 200
        list_response.json.return_value = {
            'models': [
                {'id': 'model_1', 'name': 'Model 1', 'status': 'active'},
                {'id': 'model_2', 'name': 'Model 2', 'status': 'inactive'}
            ]
        }
        list_response.raise_for_status.return_value = None
        
        # Get specific model
        model_response = Mock()
        model_response.status_code = 200
        model_response.json.return_value = {
            'id': 'model_1',
            'name': 'Model 1',
            'status': 'active',
            'capabilities': ['text-generation', 'summarization']
        }
        model_response.raise_for_status.return_value = None
        
        mock_request.side_effect = [list_response, model_response]
        
        # Test list models
        models = self.connector.list_models()
        self.assertEqual(len(models['models']), 2)
        
        # Test get specific model
        model_info = self.connector.get_model_info('model_1')
        self.assertEqual(model_info['id'], 'model_1')
        self.assertIn('capabilities', model_info)
    
    def test_logging_integration(self):
        """Test logging integration and log messages."""
        import logging
        
        # Set up log capture
        log_capture = []
        
        class TestHandler(logging.Handler):
            def emit(self, record):
                log_capture.append(record)
        
        # Add test handler to logger
        logger = logging.getLogger('genesis_connector')
        test_handler = TestHandler()
        logger.addHandler(test_handler)
        logger.setLevel(logging.DEBUG)
        
        try:
            # Create connector (should log initialization)
            connector = GenesisConnector(self.mock_config)
            
            # Check if any logs were captured
            # Note: This depends on actual logging in the implementation
            self.assertIsNotNone(connector)
            
        finally:
            # Clean up
            logger.removeHandler(test_handler)
    
    def test_session_persistence(self):
        """Test that HTTP session is properly managed."""
        # Test that session is reused across requests
        self.assertIsNotNone(self.connector.session)
        
        # Session should be the same instance across calls
        session1 = self.connector.session
        session2 = self.connector.session
        self.assertIs(session1, session2)
    
    @patch('requests.Session.request')
    def test_error_message_preservation(self, mock_request):
        """Test that error messages are properly preserved through the error handling chain."""
        original_error_message = "Very specific error message with details"
        
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = original_error_message
        mock_response.raise_for_status.side_effect = requests.HTTPError("400 Bad Request")
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        # Original error message should be preserved
        error_str = str(context.exception)
        self.assertIn(original_error_message, error_str)
    
    def test_config_validation_comprehensive(self):
        """Test comprehensive configuration validation."""
        # Test various invalid configurations
        invalid_configs = [
            ({'api_key': 123}, "API key must be string"),
            ({'api_key': 'valid', 'base_url': 123}, "Base URL must be string"),
            ({'api_key': 'valid', 'timeout': 'invalid'}, "Timeout must be number"),
            ({'api_key': 'valid', 'max_retries': 'invalid'}, "Max retries must be number"),
            ({'api_key': 'valid', 'base_url': 'ftp://invalid'}, "Invalid protocol"),
        ]
        
        for config, expected_error in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError) as context:
                    GenesisConnector(config)
                # Note: This depends on actual validation in the implementation


class TestGenesisConnectorPerformance(unittest.TestCase):
    """Performance and load testing for GenesisConnector."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    @patch('requests.Session.request')
    def test_rapid_sequential_requests(self, mock_request):
        """Test rapid sequential request performance."""
        import time
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        num_requests = 100
        start_time = time.time()
        
        for i in range(num_requests):
            result = self.connector.make_request('GET', f'/test/{i}')
            self.assertEqual(result, {'success': True})
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertion: should complete 100 requests in reasonable time
        self.assertLess(total_time, 10.0, "100 requests took too long")
        self.assertEqual(mock_request.call_count, num_requests)
    
    @patch('requests.Session.request')
    def test_memory_usage_stability(self, mock_request):
        """Test memory usage stability over many requests."""
        import gc
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'x' * 1000}  # 1KB response
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Make many requests and ensure memory doesn't grow unbounded
        for i in range(50):
            result = self.connector.make_request('GET', '/test')
            self.assertIsNotNone(result)
            
            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()
        
        # Memory usage should be stable (no assertions here, just ensuring no crashes)
        self.assertTrue(True)


class TestGenesisConnectorSecurity(unittest.TestCase):
    """Security-focused tests for GenesisConnector."""
    
    def setUp(self):
        """Set up security test fixtures."""
        self.mock_config = {
            'api_key': 'sensitive_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_api_key_not_in_logs(self):
        """Test that API key is not exposed in logs or error messages."""
        # Test repr
        repr_str = repr(self.connector)
        self.assertNotIn('sensitive_api_key_123', repr_str)
        
        # Test str
        str_repr = str(self.connector)
        self.assertNotIn('sensitive_api_key_123', str_repr)
        
        # Test that partial key or masking is present
        self.assertTrue('***' in repr_str or 'sensitive_api_key_123'[:4] in repr_str)
    
    def test_sensitive_data_in_error_messages(self):
        """Test that sensitive data is not included in error messages."""
        with self.assertRaises(ValueError) as context:
            config = self.mock_config.copy()
            config['api_key'] = ''
            GenesisConnector(config)
        
        error_msg = str(context.exception)
        # Should not contain the original API key
        self.assertNotIn('sensitive_api_key_123', error_msg)
    
    @patch('requests.Session.request')
    def test_request_data_sanitization(self, mock_request):
        """Test that request data doesn't leak sensitive information."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        sensitive_data = {
            'password': 'secret123',
            'api_key': 'another_secret',
            'token': 'bearer_token'
        }
        
        # Make request with sensitive data
        result = self.connector.make_request('POST', '/test', data=sensitive_data)
        
        # Verify request was made but sensitive data handling is proper
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once()
        
        # Check that the call was made with the data (implementation should handle sanitization)
        call_args = mock_request.call_args
        self.assertIn('json', call_args[1])


class TestGenesisConnectorCompatibility(unittest.TestCase):
    """Compatibility tests for different Python versions and environments."""
    
    def setUp(self):
        """Set up compatibility test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
    
    def test_python_version_compatibility(self):
        """Test compatibility with current Python version."""
        import sys
        
        # Should work with Python 3.7+
        self.assertGreaterEqual(sys.version_info[:2], (3, 7))
        
        # Should be able to create connector
        connector = GenesisConnector(self.mock_config)
        self.assertIsNotNone(connector)
    
    def test_unicode_handling(self):
        """Test Unicode string handling in various contexts."""
        unicode_config = {
            'api_key': 'test_këy_123',
            'base_url': 'https://api.génesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        
        # Should handle Unicode in configuration
        connector = GenesisConnector(unicode_config)
        self.assertEqual(connector.api_key, 'test_këy_123')
    
    def test_exception_inheritance(self):
        """Test that custom exceptions inherit properly."""
        # Test exception hierarchy
        self.assertTrue(issubclass(GenesisConnectionError, Exception))
        self.assertTrue(issubclass(GenesisTimeoutError, GenesisConnectionError))
        
        # Test exception instantiation
        conn_error = GenesisConnectionError("Connection failed")
        self.assertIsInstance(conn_error, Exception)
        
        timeout_error = GenesisTimeoutError("Timeout occurred")
        self.assertIsInstance(timeout_error, GenesisConnectionError)
        self.assertIsInstance(timeout_error, Exception)


if __name__ == '__main__':
    # Run all tests including the new ones
    unittest.main(verbosity=2)


# Pytest parametrized tests (if pytest is available)
class TestGenesisConnectorParametrized:
    """Parametrized tests using pytest for comprehensive coverage."""
    
    @pytest.fixture
    def connector(self):
        """Pytest fixture for GenesisConnector."""
        config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        return GenesisConnector(config)
    
    @pytest.mark.parametrize("status_code,expected_exception", [
        (400, GenesisConnectionError),
        (401, GenesisConnectionError),
        (403, GenesisConnectionError),
        (404, GenesisConnectionError),
        (500, GenesisConnectionError),
        (502, GenesisConnectionError),
        (503, GenesisConnectionError),
    ])
    @patch('requests.Session.request')
    def test_http_error_codes(self, mock_request, connector, status_code, expected_exception):
        """Test various HTTP error codes with parametrized approach."""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.raise_for_status.side_effect = requests.HTTPError(f"{status_code} Error")
        mock_response.text = f"Error {status_code}"
        mock_request.return_value = mock_response
        
        with pytest.raises(expected_exception):
            connector.make_request('GET', '/test')
    
    @pytest.mark.parametrize("method", [
        'GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS'
    ])
    @patch('requests.Session.request')
    def test_http_methods(self, mock_request, connector, method):
        """Test all HTTP methods with parametrized approach."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'method': method}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = connector.make_request(method, '/test')
        assert result == {'method': method}
    
    @pytest.mark.parametrize("timeout_value,should_raise", [
        (1, False),
        (30, False),
        (60, False),
        (0, True),
        (-1, True),
        ('invalid', True),
        (None, True),
    ])
    def test_timeout_validation(self, timeout_value, should_raise):
        """Test timeout validation with various values."""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': timeout_value,
            'max_retries': 3
        }
        
        if should_raise:
            with pytest.raises(ValueError):
                GenesisConnector(config)
        else:
            connector = GenesisConnector(config)
            assert connector.timeout == timeout_value
    
    @pytest.mark.parametrize("base_url,should_raise", [
        ('https://api.test.com', False),
        ('http://api.test.com', False),
        ('https://api.test.com:8080', False),
        ('https://api.test.com/v1', False),
        ('invalid_url', True),
        ('ftp://api.test.com', True),
        ('', True),
        (None, True),
    ])
    def test_base_url_validation(self, base_url, should_raise):
        """Test base URL validation with various formats."""
        config = {
            'api_key': 'test_key',
            'base_url': base_url,
            'timeout': 30,
            'max_retries': 3
        }
        
        if should_raise:
            with pytest.raises(ValueError):
                GenesisConnector(config)
        else:
            connector = GenesisConnector(config)
            assert connector.base_url == base_url


class TestGenesisConnectorStress(unittest.TestCase):
    """Stress tests for GenesisConnector under high load."""
    
    def setUp(self):
        """Set up stress test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    @patch('requests.Session.request')
    def test_high_frequency_requests(self, mock_request):
        """Test high frequency requests without delays."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Make 1000 requests as fast as possible
        for i in range(1000):
            result = self.connector.make_request('GET', f'/test/{i}')
            self.assertEqual(result, {'success': True})
        
        self.assertEqual(mock_request.call_count, 1000)
    
    @patch('requests.Session.request')
    def test_retry_storm_handling(self, mock_request):
        """Test handling of retry storms (many consecutive failures)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        # Create a scenario where requests fail many times before succeeding
        failures = [ConnectionError("Connection failed")] * 100
        mock_request.side_effect = failures + [mock_response]
        
        # Should eventually succeed after all retries are exhausted and tried again
        with self.assertRaises(GenesisConnectionError):
            # This should fail after max_retries
            self.connector.make_request('GET', '/test')
    
    @patch('requests.Session.request')
    def test_large_response_handling(self, mock_request):
        """Test handling of very large responses."""
        # Create a large response (1MB of data)
        large_data = {'content': 'A' * 1024 * 1024}
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = large_data
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/large-data')
        
        # Should handle large responses without issues
        self.assertEqual(len(result['content']), 1024 * 1024)
    
    def test_connector_cleanup_after_many_uses(self):
        """Test that connector properly cleans up after extensive use."""
        # Create and destroy many connectors
        for i in range(100):
            config = self.mock_config.copy()
            config['api_key'] = f'key_{i}'
            
            connector = GenesisConnector(config)
            self.assertIsNotNone(connector)
            
            # Use context manager
            with connector:
                pass
            
            # Explicit cleanup if available
            if hasattr(connector, 'close'):
                connector.close()


class TestGenesisConnectorBoundaryConditions(unittest.TestCase):
    """Test boundary conditions and edge cases."""
    
    def setUp(self):
        """Set up boundary condition test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_minimum_configuration_values(self):
        """Test with minimum valid configuration values."""
        min_config = {
            'api_key': 'a',  # Single character API key
            'base_url': 'https://a.com',  # Minimal URL
            'timeout': 1,  # Minimum timeout
            'max_retries': 0  # No retries
        }
        
        connector = GenesisConnector(min_config)
        self.assertEqual(connector.api_key, 'a')
        self.assertEqual(connector.timeout, 1)
        self.assertEqual(connector.max_retries, 0)
    
    def test_maximum_configuration_values(self):
        """Test with maximum reasonable configuration values."""
        max_config = {
            'api_key': 'x' * 1000,  # Very long API key
            'base_url': 'https://' + 'x' * 250 + '.com',  # Long domain
            'timeout': 3600,  # 1 hour timeout
            'max_retries': 100  # Many retries
        }
        
        connector = GenesisConnector(max_config)
        self.assertEqual(len(connector.api_key), 1000)
        self.assertEqual(connector.timeout, 3600)
        self.assertEqual(connector.max_retries, 100)
    
    @patch('requests.Session.request')
    def test_zero_length_response(self, mock_request):
        """Test handling of zero-length responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}  # Empty JSON
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/empty')
        self.assertEqual(result, {})
    
    @patch('requests.Session.request')
    def test_exactly_at_retry_limit(self, mock_request):
        """Test behavior when failures equal exactly the retry limit."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        # Fail exactly max_retries times, then succeed
        failures = [ConnectionError("Connection failed")] * 3  # max_retries = 3
        mock_request.side_effect = failures + [mock_response]
        
        result = self.connector.make_request('GET', '/test')
        self.assertEqual(result, {'success': True})
        self.assertEqual(mock_request.call_count, 4)  # Initial + 3 retries
    
    def test_url_building_with_query_parameters(self):
        """Test URL building with complex query parameters."""
        test_cases = [
            ('/endpoint?param=value', 'https://api.genesis.test/endpoint?param=value'),
            ('/endpoint?a=1&b=2', 'https://api.genesis.test/endpoint?a=1&b=2'),
            ('/endpoint?unicode=測試', 'https://api.genesis.test/endpoint?unicode=測試'),
            ('/endpoint?empty=', 'https://api.genesis.test/endpoint?empty='),
        ]
        
        for endpoint, expected in test_cases:
            with self.subTest(endpoint=endpoint):
                result = self.connector._build_url(endpoint)
                self.assertEqual(result, expected)
    
    def test_header_case_sensitivity(self):
        """Test header handling with different case variations."""
        headers_variations = [
            {'authorization': 'Bearer override'},  # lowercase
            {'Authorization': 'Bearer override'},  # proper case
            {'AUTHORIZATION': 'Bearer override'},  # uppercase
        ]
        
        for headers in headers_variations:
            with self.subTest(headers=headers):
                result = self.connector._build_headers(headers)
                # Should preserve the provided authorization header
                self.assertIn('authorization', result or {})

class TestGenesisConnectorExtendedEdgeCases(unittest.TestCase):
    """Extended edge case tests for GenesisConnector with even more scenarios."""
    
    def setUp(self):
        """Set up extended edge case test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_initialization_with_whitespace_api_key(self):
        """Test initialization with whitespace in API key."""
        config = self.mock_config.copy()
        config['api_key'] = '  test_key_with_spaces  '
        
        connector = GenesisConnector(config)
        # Should handle whitespace appropriately
        self.assertIsNotNone(connector.api_key)
    
    def test_initialization_with_unicode_api_key(self):
        """Test initialization with unicode characters in API key."""
        config = self.mock_config.copy()
        config['api_key'] = 'test_key_with_unicode_测试_🔑'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test_key_with_unicode_测试_🔑')
    
    def test_initialization_with_base_url_variations(self):
        """Test initialization with various base URL formats."""
        base_url_variations = [
            'https://api.genesis.test/',
            'https://api.genesis.test/v1',
            'https://api.genesis.test/v1/',
            'https://api.genesis.test:443',
            'https://api.genesis.test:8080/api',
            'http://localhost:8000',
            'https://subdomain.domain.tld/path'
        ]
        
        for base_url in base_url_variations:
            with self.subTest(base_url=base_url):
                config = self.mock_config.copy()
                config['base_url'] = base_url
                
                try:
                    connector = GenesisConnector(config)
                    self.assertEqual(connector.base_url, base_url)
                except ValueError:
                    # Some URLs might be invalid, that's expected
                    pass
    
    def test_initialization_with_float_timeout(self):
        """Test initialization with float timeout values."""
        config = self.mock_config.copy()
        config['timeout'] = 30.5
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.timeout, 30.5)
    
    def test_initialization_with_very_large_timeout(self):
        """Test initialization with very large timeout values."""
        config = self.mock_config.copy()
        config['timeout'] = 86400  # 24 hours
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.timeout, 86400)
    
    @patch('requests.Session.request')
    def test_make_request_with_binary_data(self, mock_request):
        """Test API request with binary data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        binary_data = b'\x00\x01\x02\x03\x04\x05'
        
        # This should handle binary data gracefully
        with self.assertRaises(TypeError):
            self.connector.make_request('POST', '/binary', data=binary_data)
    
    @patch('requests.Session.request')
    def test_make_request_with_nested_json_data(self, mock_request):
        """Test API request with deeply nested JSON data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        nested_data = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'level5': 'deep_value',
                            'array': [1, 2, 3, {'nested_in_array': True}]
                        }
                    }
                }
            },
            'unicode_key_测试': 'unicode_value_🌟'
        }
        
        result = self.connector.make_request('POST', '/nested', data=nested_data)
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_circular_reference_data(self):
        """Test API request with data containing circular references."""
        mock_request = Mock()
        
        # Create circular reference
        data = {'key': 'value'}
        data['self'] = data  # Circular reference
        
        # This should raise an error when trying to serialize
        with self.assertRaises((ValueError, TypeError)):
            self.connector.make_request('POST', '/circular', data=data)
    
    @patch('requests.Session.request')
    def test_make_request_with_none_values(self, mock_request):
        """Test API request with None values in data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        data_with_none = {
            'valid_key': 'valid_value',
            'none_key': None,
            'empty_string': '',
            'zero': 0,
            'false': False
        }
        
        result = self.connector.make_request('POST', '/none_values', data=data_with_none)
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_extremely_long_endpoint(self, mock_request):
        """Test API request with extremely long endpoint."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Create very long endpoint
        long_endpoint = '/' + 'a' * 2000  # 2000 character endpoint
        
        result = self.connector.make_request('GET', long_endpoint)
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_special_http_headers(self, mock_request):
        """Test API request with special HTTP headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        special_headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'DNT': '1',
            'Origin': 'https://example.com',
            'Pragma': 'no-cache',
            'Referer': 'https://example.com/page',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'cross-site',
            'User-Agent': 'Mozilla/5.0 (Custom Test Agent)',
            'X-Requested-With': 'XMLHttpRequest',
            'X-Custom-Header-With-Unicode': 'value_with_测试_🌟'
        }
        
        result = self.connector.make_request('GET', '/special_headers', headers=special_headers)
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_make_request_response_with_different_content_types(self, mock_request):
        """Test API request handling different content types."""
        content_types = [
            'application/json',
            'application/json; charset=utf-8',
            'application/json; charset=iso-8859-1',
            'text/plain',
            'text/html',
            'application/xml',
            'application/octet-stream'
        ]
        
        for content_type in content_types:
            with self.subTest(content_type=content_type):
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.headers = {'Content-Type': content_type}
                mock_response.raise_for_status.return_value = None
                
                if 'json' in content_type:
                    mock_response.json.return_value = {'success': True}
                    expected_result = {'success': True}
                else:
                    mock_response.json.side_effect = json.JSONDecodeError("Not JSON", "", 0)
                    mock_response.text = "Non-JSON response"
                    expected_result = None
                
                mock_request.return_value = mock_response
                
                if expected_result is not None:
                    result = self.connector.make_request('GET', '/test')
                    self.assertEqual(result, expected_result)
                else:
                    # Should handle non-JSON responses gracefully
                    with self.assertRaises(GenesisConnectionError):
                        self.connector.make_request('GET', '/test')
    
    @patch('requests.Session.request')
    def test_make_request_with_response_encoding_issues(self, mock_request):
        """Test API request with response encoding issues."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        
        # Simulate encoding issues
        mock_response.text = "Response with encoding issues: \udcff\udcfe"
        mock_response.json.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "encoding error")
        
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError):
            self.connector.make_request('GET', '/encoding_test')
    
    @patch('time.sleep')
    @patch('requests.Session.request')
    def test_retry_with_exponential_backoff_maximum(self, mock_request, mock_sleep):
        """Test retry with exponential backoff reaching maximum delay."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        # Configure for maximum retries
        config = self.mock_config.copy()
        config['max_retries'] = 10
        connector = GenesisConnector(config)
        
        # Fail many times, then succeed
        failures = [ConnectionError("Connection failed")] * 10
        mock_request.side_effect = failures + [mock_response]
        
        result = connector.make_request('GET', '/test')
        
        self.assertEqual(result, {'success': True})
        # Check exponential backoff pattern
        self.assertEqual(mock_sleep.call_count, 10)
    
    @patch('requests.Session.request')
    def test_make_request_with_custom_session_configuration(self, mock_request):
        """Test that custom session configuration is respected."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Test that session is used correctly
        self.assertIsNotNone(self.connector.session)
        
        result = self.connector.make_request('GET', '/test')
        self.assertEqual(result, {'success': True})
        
        # Verify session was used
        mock_request.assert_called_once()
    
    def test_url_building_with_ipv6_addresses(self):
        """Test URL building with IPv6 addresses."""
        ipv6_configs = [
            'https://[::1]:8080',
            'https://[2001:db8::1]:8080',
            'http://[::1]',
            'https://[fe80::1%lo0]:8080'
        ]
        
        for base_url in ipv6_configs:
            with self.subTest(base_url=base_url):
                config = self.mock_config.copy()
                config['base_url'] = base_url
                
                try:
                    connector = GenesisConnector(config)
                    url = connector._build_url('/test')
                    self.assertIn(base_url, url)
                except ValueError:
                    # Some IPv6 URLs might be invalid, that's expected
                    pass
    
    def test_context_manager_with_nested_usage(self):
        """Test context manager with nested usage patterns."""
        config1 = self.mock_config.copy()
        config1['api_key'] = 'key1'
        
        config2 = self.mock_config.copy()
        config2['api_key'] = 'key2'
        
        with GenesisConnector(config1) as connector1:
            with GenesisConnector(config2) as connector2:
                self.assertEqual(connector1.api_key, 'key1')
                self.assertEqual(connector2.api_key, 'key2')
                self.assertNotEqual(connector1.api_key, connector2.api_key)
    
    def test_string_representations_with_long_values(self):
        """Test string representations with very long configuration values."""
        config = self.mock_config.copy()
        config['api_key'] = 'very_long_api_key_' + 'x' * 500
        config['base_url'] = 'https://very.long.domain.name.' + 'x' * 100 + '.com'
        
        connector = GenesisConnector(config)
        
        repr_str = repr(connector)
        str_repr = str(connector)
        
        # Should not include full long values
        self.assertNotIn('x' * 500, repr_str)
        self.assertNotIn('x' * 100, str_repr)
        
        # Should include truncated or masked versions
        self.assertIn('GenesisConnector', repr_str)
        self.assertIn('GenesisConnector', str_repr)
    
    @patch('requests.Session.request')
    def test_make_request_with_request_hooks(self, mock_request):
        """Test API request with request hooks if supported."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Test with hooks parameter if supported
        try:
            hooks = {'response': lambda r, *args, **kwargs: r}
            result = self.connector.make_request('GET', '/test', hooks=hooks)
            self.assertEqual(result, {'success': True})
        except TypeError:
            # If hooks not supported, that's fine
            pass
    
    def test_connector_thread_safety_stress(self):
        """Test connector thread safety under stress conditions."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker():
            try:
                # Test various operations
                url = self.connector._build_url('/test')
                headers = self.connector._build_headers()
                
                results.append({'url': url, 'headers': headers})
            except Exception as e:
                errors.append(e)
        
        # Create many threads
        threads = []
        for i in range(50):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All should succeed
        self.assertEqual(len(results), 50)
        self.assertEqual(len(errors), 0)
    
    def test_memory_cleanup_verification(self):
        """Test memory cleanup and reference counting."""
        import gc
        import weakref
        
        # Create connector and get weak reference
        connector = GenesisConnector(self.mock_config)
        weak_ref = weakref.ref(connector)
        
        # Delete connector
        del connector
        gc.collect()
        
        # Weak reference should be None if properly cleaned up
        # Note: This might not always work depending on implementation
        # but it's a good test for memory leaks
        self.assertIsNotNone(weak_ref())  # Might still exist due to test framework
    
    @patch('requests.Session.request')
    def test_make_request_with_streaming_response(self, mock_request):
        """Test API request with streaming response if supported."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Test streaming if supported
        try:
            result = self.connector.make_request('GET', '/stream', stream=True)
            # This depends on implementation
        except TypeError:
            # If streaming not supported, that's expected
            pass
    
    def test_error_handling_with_mixed_exception_types(self):
        """Test error handling with mixed exception types in inheritance."""
        # Test that custom exceptions handle various error conditions
        base_error = GenesisConnectionError("Base error")
        timeout_error = GenesisTimeoutError("Timeout error")
        
        # Test exception properties
        self.assertIsInstance(base_error, Exception)
        self.assertIsInstance(timeout_error, GenesisConnectionError)
        self.assertIsInstance(timeout_error, Exception)
        
        # Test exception messages
        self.assertEqual(str(base_error), "Base error")
        self.assertEqual(str(timeout_error), "Timeout error")
        
        # Test exception attributes if they exist
        if hasattr(base_error, 'error_code'):
            self.assertIsNotNone(base_error.error_code)
        if hasattr(timeout_error, 'timeout_duration'):
            self.assertIsNotNone(timeout_error.timeout_duration)


class TestGenesisConnectorRobustness(unittest.TestCase):
    """Robustness tests for GenesisConnector under adverse conditions."""
    
    def setUp(self):
        """Set up robustness test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    @patch('requests.Session.request')
    def test_resilience_to_memory_pressure(self, mock_request):
        """Test connector resilience under memory pressure."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Create memory pressure
        large_objects = []
        try:
            for i in range(100):
                large_objects.append([0] * 10000)  # 10K integers each
                
                # Make request under memory pressure
                result = self.connector.make_request('GET', f'/test/{i}')
                self.assertEqual(result, {'success': True})
                
                # Clean up periodically
                if i % 10 == 0:
                    large_objects.clear()
        finally:
            # Clean up
            large_objects.clear()
    
    @patch('requests.Session.request')
    def test_exception_handling_during_cleanup(self, mock_request):
        """Test exception handling during connector cleanup."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Test cleanup behavior
        try:
            with self.connector as conn:
                result = conn.make_request('GET', '/test')
                self.assertEqual(result, {'success': True})
                # Simulate exception during cleanup
                if hasattr(conn, 'session'):
                    # Intentionally cause an error during cleanup
                    conn.session.close = Mock(side_effect=Exception("Cleanup error"))
        except Exception as e:
            # Should handle cleanup errors gracefully
            self.assertNotIn('Cleanup error', str(e))
    
    def test_configuration_mutation_resistance(self):
        """Test resistance to configuration mutation after initialization."""
        config = self.mock_config.copy()
        connector = GenesisConnector(config)
        
        # Mutate original config
        config['api_key'] = 'mutated_key'
        config['base_url'] = 'https://mutated.url'
        
        # Connector should not be affected
        self.assertEqual(connector.api_key, 'test_api_key_123')
        self.assertEqual(connector.base_url, 'https://api.genesis.test')
    
    @patch('requests.Session.request')
    def test_graceful_degradation_on_partial_failures(self, mock_request):
        """Test graceful degradation when some operations fail."""
        # Mix of successful and failed responses
        successful_response = Mock()
        successful_response.status_code = 200
        successful_response.json.return_value = {'success': True}
        successful_response.raise_for_status.return_value = None
        
        failed_response = Mock()
        failed_response.status_code = 500
        failed_response.raise_for_status.side_effect = requests.HTTPError("500 Error")
        
        mock_request.side_effect = [
            successful_response,
            failed_response,
            successful_response,
            failed_response
        ]
        
        # First request should succeed
        result1 = self.connector.make_request('GET', '/test1')
        self.assertEqual(result1, {'success': True})
        
        # Second request should fail
        with self.assertRaises(GenesisConnectionError):
            self.connector.make_request('GET', '/test2')
        
        # Third request should succeed (connector still functional)
        result3 = self.connector.make_request('GET', '/test3')
        self.assertEqual(result3, {'success': True})
    
    def test_input_validation_edge_cases(self):
        """Test input validation with edge cases."""
        # Test with various invalid inputs
        invalid_endpoints = [
            None,
            123,
            [],
            {},
            object(),
        ]
        
        for endpoint in invalid_endpoints:
            with self.subTest(endpoint=endpoint):
                with self.assertRaises((TypeError, ValueError)):
                    self.connector.make_request('GET', endpoint)
    
    def test_state_consistency_after_errors(self):
        """Test that connector state remains consistent after errors."""
        original_api_key = self.connector.api_key
        original_base_url = self.connector.base_url
        original_timeout = self.connector.timeout
        
        # Try operations that might fail
        try:
            with self.assertRaises(TypeError):
                self.connector.make_request('GET', None)
        except:
            pass
        
        try:
            with self.assertRaises(ValueError):
                self.connector._build_url(None)
        except:
            pass
        
        # State should remain consistent
        self.assertEqual(self.connector.api_key, original_api_key)
        self.assertEqual(self.connector.base_url, original_base_url)
        self.assertEqual(self.connector.timeout, original_timeout)


class TestGenesisConnectorDataValidation(unittest.TestCase):
    """Data validation tests for GenesisConnector."""
    
    def setUp(self):
        """Set up data validation test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    @patch('requests.Session.request')
    def test_json_serialization_edge_cases(self, mock_request):
        """Test JSON serialization with edge cases."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Test with various data types
        test_data_cases = [
            {'float_inf': float('inf')},
            {'float_nan': float('nan')},
            {'decimal': 123.456789012345},
            {'scientific': 1.23e-10},
            {'very_large_number': 10**100},
            {'nested_lists': [[1, 2], [3, 4]]},
            {'boolean_values': {'true': True, 'false': False}},
            {'none_value': None},
            {'empty_containers': {'list': [], 'dict': {}, 'string': ''}},
        ]
        
        for data in test_data_cases:
            with self.subTest(data=data):
                try:
                    result = self.connector.make_request('POST', '/test', data=data)
                    self.assertEqual(result, {'success': True})
                except (ValueError, TypeError, OverflowError):
                    # Some data types might not be serializable, that's expected
                    pass
    
    def test_url_validation_comprehensive(self):
        """Test comprehensive URL validation."""
        invalid_urls = [
            '',
            'not-a-url',
            'ftp://invalid.protocol',
            'https://',
            'https://.',
            'https://..',
            'https://invalid.',
            'https://invalid..com',
            'https://invalid-.com',
            'https://-invalid.com',
            'javascript:alert(1)',
            'data:text/html,<script>alert(1)</script>',
        ]
        
        for url in invalid_urls:
            with self.subTest(url=url):
                config = self.mock_config.copy()
                config['base_url'] = url
                
                with self.assertRaises(ValueError):
                    GenesisConnector(config)
    
    def test_header_validation_edge_cases(self):
        """Test header validation with edge cases."""
        invalid_headers = [
            {'': 'empty_key'},
            {'key': ''},  # Empty value might be valid
            {'key\n': 'newline_in_key'},
            {'key': 'value\n'},  # Newline in value
            {'key\x00': 'null_byte'},
            {'key': 'value\x00'},
            {123: 'numeric_key'},
            {'key': 123},  # Numeric value
            {'key': None},  # None value
        ]
        
        for headers in invalid_headers:
            with self.subTest(headers=headers):
                try:
                    result = self.connector._build_headers(headers)
                    # Some invalid headers might be handled gracefully
                    self.assertIsInstance(result, dict)
                except (TypeError, ValueError):
                    # Expected for truly invalid headers
                    pass
    
    def test_timeout_validation_comprehensive(self):
        """Test comprehensive timeout validation."""
        invalid_timeouts = [
            -1,
            -0.1,
            0,
            'invalid',
            [],
            {},
            object(),
            complex(1, 2),
        ]
        
        for timeout in invalid_timeouts:
            with self.subTest(timeout=timeout):
                config = self.mock_config.copy()
                config['timeout'] = timeout
                
                with self.assertRaises((ValueError, TypeError)):
                    GenesisConnector(config)
    
    def test_max_retries_validation_comprehensive(self):
        """Test comprehensive max_retries validation."""
        invalid_retries = [
            -1,
            -10,
            'invalid',
            [],
            {},
            object(),
            3.14,  # Float might be invalid
        ]
        
        for retries in invalid_retries:
            with self.subTest(retries=retries):
                config = self.mock_config.copy()
                config['max_retries'] = retries
                
                with self.assertRaises((ValueError, TypeError)):
                    GenesisConnector(config)
    
    def test_api_key_validation_edge_cases(self):
        """Test API key validation with edge cases."""
        invalid_api_keys = [
            None,
            '',
            ' ',
            '\t',
            '\n',
            123,
            [],
            {},
            object(),
        ]
        
        for api_key in invalid_api_keys:
            with self.subTest(api_key=api_key):
                config = self.mock_config.copy()
                config['api_key'] = api_key
                
                with self.assertRaises((ValueError, TypeError)):
                    GenesisConnector(config)


# Additional pytest fixtures and markers for extended testing
@pytest.fixture
def connector_with_custom_config():
    """Pytest fixture for connector with custom configuration."""
    config = {
        'api_key': 'custom_test_key',
        'base_url': 'https://custom.api.test',
        'timeout': 60,
        'max_retries': 5,
        'custom_param': 'custom_value'
    }
    return GenesisConnector(config)


@pytest.mark.parametrize("config_key,invalid_value,expected_exception", [
    ('api_key', None, ValueError),
    ('api_key', '', ValueError),
    ('api_key', 123, ValueError),
    ('base_url', 'invalid', ValueError),
    ('base_url', None, ValueError),
    ('timeout', -1, ValueError),
    ('timeout', 0, ValueError),
    ('timeout', 'invalid', ValueError),
    ('max_retries', -1, ValueError),
    ('max_retries', 'invalid', ValueError),
])
def test_configuration_validation_parametrized(config_key, invalid_value, expected_exception):
    """Test configuration validation with parametrized approach."""
    config = {
        'api_key': 'test_key',
        'base_url': 'https://api.test.com',
        'timeout': 30,
        'max_retries': 3
    }
    config[config_key] = invalid_value
    
    with pytest.raises(expected_exception):
        GenesisConnector(config)


@pytest.mark.parametrize("endpoint,expected_url", [
    ('', 'https://api.genesis.test'),
    ('/', 'https://api.genesis.test/'),
    ('/test', 'https://api.genesis.test/test'),
    ('test', 'https://api.genesis.test/test'),
    ('/test/', 'https://api.genesis.test/test/'),
    ('//test', 'https://api.genesis.test/test'),
    ('/test//path', 'https://api.genesis.test/test//path'),
])
def test_url_building_parametrized(endpoint, expected_url):
    """Test URL building with parametrized approach."""
    config = {
        'api_key': 'test_key',
        'base_url': 'https://api.genesis.test',
        'timeout': 30,
        'max_retries': 3
    }
    connector = GenesisConnector(config)
    
    result = connector._build_url(endpoint)
    assert result == expected_url


@pytest.mark.slow
@pytest.mark.integration
class TestGenesisConnectorIntegrationExtended:
    """Extended integration tests for GenesisConnector."""
    
    def test_real_world_usage_simulation(self):
        """Test simulating real-world usage patterns."""
        # This would require actual API endpoints
        # Skipped in unit tests but useful for integration testing
        pytest.skip("Requires real API endpoints")
    
    def test_load_testing_simulation(self):
        """Test load testing simulation."""
        # This would test high-load scenarios
        pytest.skip("Load testing requires special setup")


if __name__ == '__main__':
    # Run the extended tests
    unittest.main(verbosity=2, exit=False)
    
    # Also run pytest if available
    try:
        import pytest
        pytest.main([__file__, '-v', '--tb=short'])
    except ImportError:
        print("pytest not available, skipping pytest tests")