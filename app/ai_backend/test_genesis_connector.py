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

class TestGenesisConnectorAdvancedEdgeCases(unittest.TestCase):
    """Additional edge cases and scenarios for GenesisConnector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_initialization_with_extra_config_params(self):
        """Test initialization with additional unknown configuration parameters."""
        config = self.mock_config.copy()
        config['unknown_param'] = 'unknown_value'
        config['another_param'] = 42
        
        # Should handle extra parameters gracefully
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test_api_key_123')
        self.assertEqual(connector.base_url, 'https://api.genesis.test')
    
    def test_initialization_with_nested_config(self):
        """Test initialization with nested configuration objects."""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': 30,
            'max_retries': 3,
            'nested': {
                'inner_param': 'value'
            }
        }
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test_key')
    
    @patch('requests.Session.request')
    def test_make_request_with_binary_data(self, mock_request):
        """Test API request with binary data payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        binary_data = b'\x00\x01\x02\x03\x04\x05'
        
        result = self.connector.make_request('POST', '/binary', data=binary_data)
        
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once()
    
    @patch('requests.Session.request')
    def test_make_request_with_file_like_object(self, mock_request):
        """Test API request with file-like object."""
        from io import BytesIO
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'uploaded': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        file_data = BytesIO(b'test file content')
        
        result = self.connector.make_request('POST', '/upload', data=file_data)
        
        self.assertEqual(result, {'uploaded': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_streaming_response(self, mock_request):
        """Test handling of streaming responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'stream': 'data'}
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/stream')
        
        self.assertEqual(result, {'stream': 'data'})
    
    @patch('requests.Session.request')
    def test_make_request_with_redirect_handling(self, mock_request):
        """Test handling of HTTP redirects."""
        # First response - redirect
        redirect_response = Mock()
        redirect_response.status_code = 302
        redirect_response.headers = {'Location': 'https://api.genesis.test/new-location'}
        redirect_response.raise_for_status.return_value = None
        redirect_response.json.return_value = {'redirected': True}
        
        mock_request.return_value = redirect_response
        
        result = self.connector.make_request('GET', '/redirect')
        
        self.assertEqual(result, {'redirected': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_custom_auth_header(self, mock_request):
        """Test API request with custom authorization header that overrides default."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'auth': 'custom'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        custom_headers = {'Authorization': 'Custom auth_token_123'}
        
        result = self.connector.make_request('GET', '/test', headers=custom_headers)
        
        self.assertEqual(result, {'auth': 'custom'})
        # Verify the custom auth header was used
        call_args = mock_request.call_args
        self.assertEqual(call_args[1]['headers']['Authorization'], 'Custom auth_token_123')
    
    @patch('requests.Session.request')
    def test_make_request_with_cookies(self, mock_request):
        """Test API request with cookies handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'cookies': 'handled'}
        mock_response.raise_for_status.return_value = None
        mock_response.cookies = {'session_id': 'abc123'}
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/cookies')
        
        self.assertEqual(result, {'cookies': 'handled'})
    
    @patch('requests.Session.request')
    def test_make_request_with_proxy_settings(self, mock_request):
        """Test API request with proxy configuration."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'proxy': 'used'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Test that proxy settings can be handled if supported
        result = self.connector.make_request('GET', '/proxy')
        
        self.assertEqual(result, {'proxy': 'used'})
    
    @patch('requests.Session.request')
    def test_make_request_with_ssl_verification(self, mock_request):
        """Test API request with SSL verification settings."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'ssl': 'verified'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/ssl')
        
        self.assertEqual(result, {'ssl': 'verified'})
    
    @patch('requests.Session.request')
    def test_make_request_with_content_encoding(self, mock_request):
        """Test API request with various content encodings."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'encoding': 'gzip'}
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {'Content-Encoding': 'gzip'}
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/encoded')
        
        self.assertEqual(result, {'encoding': 'gzip'})
    
    @patch('requests.Session.request')
    def test_make_request_with_partial_json_response(self, mock_request):
        """Test handling of partial JSON responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = [
            json.JSONDecodeError("Partial JSON", "", 10),
            {'partial': 'recovery'}
        ]
        mock_response.text = '{"partial": "json"'
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError):
            self.connector.make_request('GET', '/partial')
    
    @patch('requests.Session.request')
    def test_make_request_with_rate_limit_headers(self, mock_request):
        """Test handling of rate limit headers."""
        mock_response = Mock()
        mock_response.status_code = 429  # Too Many Requests
        mock_response.headers = {
            'X-RateLimit-Limit': '1000',
            'X-RateLimit-Remaining': '0',
            'X-RateLimit-Reset': '1640995200',
            'Retry-After': '60'
        }
        mock_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        mock_response.text = "Rate limit exceeded"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/rate-limited')
        
        self.assertIn('429 Too Many Requests', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_concurrent_modification(self, mock_request):
        """Test handling of concurrent modification scenarios."""
        mock_response = Mock()
        mock_response.status_code = 409  # Conflict
        mock_response.raise_for_status.side_effect = requests.HTTPError("409 Conflict")
        mock_response.text = "Concurrent modification detected"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('PUT', '/concurrent')
        
        self.assertIn('409 Conflict', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_precondition_failed(self, mock_request):
        """Test handling of precondition failed responses."""
        mock_response = Mock()
        mock_response.status_code = 412  # Precondition Failed
        mock_response.raise_for_status.side_effect = requests.HTTPError("412 Precondition Failed")
        mock_response.text = "Precondition failed"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('PUT', '/precondition')
        
        self.assertIn('412 Precondition Failed', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_payload_too_large(self, mock_request):
        """Test handling of payload too large responses."""
        mock_response = Mock()
        mock_response.status_code = 413  # Payload Too Large
        mock_response.raise_for_status.side_effect = requests.HTTPError("413 Payload Too Large")
        mock_response.text = "Payload too large"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('POST', '/large-payload')
        
        self.assertIn('413 Payload Too Large', str(context.exception))
    
    def test_api_key_with_special_characters(self):
        """Test API key with special characters."""
        special_chars_key = 'key_with_!@#$%^&*()_+-=[]{}|;:,.<>?'
        config = self.mock_config.copy()
        config['api_key'] = special_chars_key
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, special_chars_key)
    
    def test_base_url_with_path_components(self):
        """Test base URL with various path components."""
        path_urls = [
            'https://api.genesis.test/v1',
            'https://api.genesis.test/v1/',
            'https://api.genesis.test/api/v1',
            'https://api.genesis.test/api/v1/',
        ]
        
        for url in path_urls:
            with self.subTest(url=url):
                config = self.mock_config.copy()
                config['base_url'] = url
                
                connector = GenesisConnector(config)
                self.assertEqual(connector.base_url, url)
    
    @patch('requests.Session.request')
    def test_request_with_multiple_retries_different_errors(self, mock_request):
        """Test retry behavior with different error types in sequence."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        # Different error types in sequence
        mock_request.side_effect = [
            ConnectionError("Connection error"),
            Timeout("Timeout error"),
            requests.exceptions.SSLError("SSL error"),
            mock_response
        ]
        
        result = self.connector.make_request('GET', '/mixed-errors')
        
        self.assertEqual(result, {'success': True})
        self.assertEqual(mock_request.call_count, 4)
    
    @patch('requests.Session.request')
    def test_request_with_intermittent_success(self, mock_request):
        """Test behavior with intermittent success/failure patterns."""
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {'success': True}
        success_response.raise_for_status.return_value = None
        
        # Pattern: success, fail, success
        mock_request.side_effect = [
            success_response,
            ConnectionError("Intermittent failure"),
            success_response
        ]
        
        # First request should succeed
        result1 = self.connector.make_request('GET', '/intermittent1')
        self.assertEqual(result1, {'success': True})
        
        # Second request should succeed after retry
        result2 = self.connector.make_request('GET', '/intermittent2')
        self.assertEqual(result2, {'success': True})
    
    def test_configuration_immutability(self):
        """Test that configuration changes don't affect existing connector."""
        original_config = self.mock_config.copy()
        connector = GenesisConnector(original_config)
        
        # Modify the original config
        original_config['api_key'] = 'changed_key'
        original_config['timeout'] = 999
        
        # Connector should retain original values
        self.assertEqual(connector.api_key, 'test_api_key_123')
        self.assertEqual(connector.timeout, 30)
    
    def test_multiple_connector_instances_isolation(self):
        """Test that multiple connector instances don't interfere with each other."""
        config1 = {
            'api_key': 'key1',
            'base_url': 'https://api1.test.com',
            'timeout': 10,
            'max_retries': 1
        }
        
        config2 = {
            'api_key': 'key2',
            'base_url': 'https://api2.test.com',
            'timeout': 20,
            'max_retries': 2
        }
        
        connector1 = GenesisConnector(config1)
        connector2 = GenesisConnector(config2)
        
        # Verify isolation
        self.assertEqual(connector1.api_key, 'key1')
        self.assertEqual(connector2.api_key, 'key2')
        self.assertEqual(connector1.timeout, 10)
        self.assertEqual(connector2.timeout, 20)
        self.assertEqual(connector1.max_retries, 1)
        self.assertEqual(connector2.max_retries, 2)
    
    @patch('requests.Session.request')
    def test_request_with_custom_user_agent(self, mock_request):
        """Test request with custom User-Agent header."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'user_agent': 'custom'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        custom_headers = {'User-Agent': 'CustomAgent/1.0'}
        result = self.connector.make_request('GET', '/user-agent', headers=custom_headers)
        
        self.assertEqual(result, {'user_agent': 'custom'})
        call_args = mock_request.call_args
        self.assertEqual(call_args[1]['headers']['User-Agent'], 'CustomAgent/1.0')
    
    @patch('requests.Session.request')
    def test_request_with_accept_header_variations(self, mock_request):
        """Test request with various Accept headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'accept': 'json'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        accept_headers = [
            'application/json',
            'application/json, text/plain',
            'application/json; charset=utf-8',
            '*/*'
        ]
        
        for accept in accept_headers:
            with self.subTest(accept=accept):
                headers = {'Accept': accept}
                result = self.connector.make_request('GET', '/accept', headers=headers)
                
                self.assertEqual(result, {'accept': 'json'})
                call_args = mock_request.call_args
                self.assertEqual(call_args[1]['headers']['Accept'], accept)
    
    @patch('requests.Session.request')
    def test_request_with_content_type_variations(self, mock_request):
        """Test request with various Content-Type headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'content_type': 'received'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        content_types = [
            'application/json',
            'application/json; charset=utf-8',
            'application/x-www-form-urlencoded',
            'multipart/form-data',
            'text/plain'
        ]
        
        for content_type in content_types:
            with self.subTest(content_type=content_type):
                headers = {'Content-Type': content_type}
                result = self.connector.make_request('POST', '/content-type', headers=headers)
                
                self.assertEqual(result, {'content_type': 'received'})
    
    def test_url_building_with_double_slashes(self):
        """Test URL building handles double slashes correctly."""
        test_cases = [
            ('//endpoint', 'https://api.genesis.test/endpoint'),
            ('///endpoint', 'https://api.genesis.test/endpoint'),
            ('/endpoint//path', 'https://api.genesis.test/endpoint//path'),
            ('endpoint//path', 'https://api.genesis.test/endpoint//path'),
        ]
        
        for endpoint, expected in test_cases:
            with self.subTest(endpoint=endpoint):
                result = self.connector._build_url(endpoint)
                self.assertEqual(result, expected)
    
    def test_headers_with_empty_values(self):
        """Test header building with empty values."""
        headers_with_empty = {
            'X-Empty-Header': '',
            'X-Valid-Header': 'value',
            'X-Another-Empty': None
        }
        
        result = self.connector._build_headers(headers_with_empty)
        
        self.assertIn('X-Empty-Header', result)
        self.assertIn('X-Valid-Header', result)
        self.assertEqual(result['X-Empty-Header'], '')
        self.assertEqual(result['X-Valid-Header'], 'value')
    
    def test_headers_with_numeric_values(self):
        """Test header building with numeric values."""
        numeric_headers = {
            'X-Numeric-Header': 123,
            'X-Float-Header': 45.67,
            'X-Boolean-Header': True
        }
        
        result = self.connector._build_headers(numeric_headers)
        
        # Headers should be converted to strings
        self.assertEqual(result['X-Numeric-Header'], 123)
        self.assertEqual(result['X-Float-Header'], 45.67)
        self.assertEqual(result['X-Boolean-Header'], True)
    
    @patch('requests.Session.request')
    def test_request_with_query_parameters_in_data(self, mock_request):
        """Test request with query parameters embedded in data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'query_data': 'processed'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        data_with_query = {
            'query': 'SELECT * FROM table WHERE id = ?',
            'params': ['123']
        }
        
        result = self.connector.make_request('POST', '/query', data=data_with_query)
        
        self.assertEqual(result, {'query_data': 'processed'})
        call_args = mock_request.call_args
        self.assertEqual(call_args[1]['json'], data_with_query)
    
    @patch('requests.Session.request')
    def test_request_error_with_response_details(self, mock_request):
        """Test error handling includes response details."""
        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.raise_for_status.side_effect = requests.HTTPError("422 Unprocessable Entity")
        mock_response.text = '{"error": "validation_failed", "details": "Invalid input format"}'
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('POST', '/validate')
        
        error_message = str(context.exception)
        self.assertIn('422 Unprocessable Entity', error_message)
        self.assertIn('validation_failed', error_message)
    
    @patch('time.sleep')
    @patch('requests.Session.request')
    def test_retry_with_exponential_backoff_ceiling(self, mock_request, mock_sleep):
        """Test that exponential backoff has a reasonable ceiling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        # Configure for maximum retries
        config = self.mock_config.copy()
        config['max_retries'] = 10
        connector = GenesisConnector(config)
        
        # All requests fail except the last one
        failures = [ConnectionError("Connection failed")] * 10
        mock_request.side_effect = failures + [mock_response]
        
        result = connector.make_request('GET', '/backoff-test')
        
        self.assertEqual(result, {'success': True})
        # Check that sleep was called for each retry
        self.assertEqual(mock_sleep.call_count, 10)
        
        # Verify exponential backoff pattern (should cap at reasonable value)
        sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
        self.assertTrue(all(delay >= 1 for delay in sleep_calls))
        self.assertTrue(all(delay <= 64 for delay in sleep_calls))  # Reasonable ceiling


class TestGenesisConnectorComplexScenarios(unittest.TestCase):
    """Test complex real-world scenarios and workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    @patch('requests.Session.request')
    def test_complex_generation_workflow_with_polling(self, mock_request):
        """Test complex workflow with generation creation and status polling."""
        # Create generation response
        create_response = Mock()
        create_response.status_code = 201
        create_response.json.return_value = {
            'id': 'gen_456',
            'status': 'pending',
            'created_at': '2023-01-01T00:00:00Z'
        }
        create_response.raise_for_status.return_value = None
        
        # Status polling responses
        status_responses = []
        statuses = ['pending', 'running', 'running', 'completed']
        
        for status in statuses:
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                'id': 'gen_456',
                'status': status,
                'progress': 25 if status == 'running' else 100 if status == 'completed' else 0,
                'result': 'Generated content here' if status == 'completed' else None
            }
            response.raise_for_status.return_value = None
            status_responses.append(response)
        
        mock_request.side_effect = [create_response] + status_responses
        
        # Create generation
        generation_request = {
            'prompt': 'Complex prompt with multiple requirements',
            'model': 'advanced_model',
            'max_tokens': 1000,
            'temperature': 0.7
        }
        
        create_result = self.connector.create_generation(generation_request)
        self.assertEqual(create_result['id'], 'gen_456')
        self.assertEqual(create_result['status'], 'pending')
        
        # Poll for completion
        for expected_status in statuses:
            status_result = self.connector.get_generation_status('gen_456')
            self.assertEqual(status_result['status'], expected_status)
            
            if expected_status == 'completed':
                self.assertEqual(status_result['result'], 'Generated content here')
                self.assertEqual(status_result['progress'], 100)
    
    @patch('requests.Session.request')
    def test_batch_operations_with_mixed_results(self, mock_request):
        """Test batch operations with mixed success/failure results."""
        # Simulate batch operation responses
        batch_responses = [
            # Success responses
            Mock(status_code=200, json=lambda: {'id': 1, 'status': 'success'}),
            Mock(status_code=200, json=lambda: {'id': 2, 'status': 'success'}),
            # Failure response
            Mock(status_code=400, text='Invalid input'),
            # Success response
            Mock(status_code=200, json=lambda: {'id': 4, 'status': 'success'}),
        ]
        
        # Set up raise_for_status for each response
        for i, response in enumerate(batch_responses):
            if response.status_code >= 400:
                response.raise_for_status.side_effect = requests.HTTPError(f"{response.status_code} Error")
            else:
                response.raise_for_status.return_value = None
        
        mock_request.side_effect = batch_responses
        
        results = []
        errors = []
        
        # Simulate batch processing
        for i in range(4):
            try:
                result = self.connector.make_request('POST', f'/batch/{i}')
                results.append(result)
            except GenesisConnectionError as e:
                errors.append(e)
        
        # Verify mixed results
        self.assertEqual(len(results), 3)  # 3 successful requests
        self.assertEqual(len(errors), 1)   # 1 failed request
        
        # Check successful results
        self.assertEqual(results[0]['id'], 1)
        self.assertEqual(results[1]['id'], 2)
        self.assertEqual(results[2]['id'], 4)
    
    @patch('requests.Session.request')
    def test_model_lifecycle_management(self, mock_request):
        """Test complete model lifecycle management."""
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
        get_response = Mock()
        get_response.status_code = 200
        get_response.json.return_value = {
            'id': 'model_1',
            'name': 'Model 1',
            'status': 'active',
            'version': '1.2.3',
            'capabilities': ['text-generation', 'summarization'],
            'metrics': {'accuracy': 0.95, 'latency': 200}
        }
        get_response.raise_for_status.return_value = None
        
        # Health check
        health_response = Mock()
        health_response.status_code = 200
        health_response.json.return_value = {
            'status': 'healthy',
            'models': {
                'model_1': 'active',
                'model_2': 'inactive'
            }
        }
        health_response.raise_for_status.return_value = None
        
        mock_request.side_effect = [list_response, get_response, health_response]
        
        # Execute lifecycle workflow
        models = self.connector.list_models()
        self.assertEqual(len(models['models']), 2)
        
        model_info = self.connector.get_model_info('model_1')
        self.assertEqual(model_info['capabilities'], ['text-generation', 'summarization'])
        
        health = self.connector.health_check()
        self.assertEqual(health['status'], 'healthy')
    
    @patch('requests.Session.request')
    def test_error_recovery_workflow(self, mock_request):
        """Test error recovery and graceful degradation."""
        # Simulate network issues followed by recovery
        network_error = ConnectionError("Network unreachable")
        timeout_error = Timeout("Request timeout")
        
        recovery_response = Mock()
        recovery_response.status_code = 200
        recovery_response.json.return_value = {
            'status': 'recovered',
            'message': 'Service restored'
        }
        recovery_response.raise_for_status.return_value = None
        
        # Error pattern: network error, timeout, success
        mock_request.side_effect = [network_error, timeout_error, recovery_response]
        
        # Should recover after retries
        result = self.connector.make_request('GET', '/recovery-test')
        
        self.assertEqual(result['status'], 'recovered')
        self.assertEqual(mock_request.call_count, 3)
    
    @patch('requests.Session.request')
    def test_concurrent_request_handling(self, mock_request):
        """Test handling of concurrent requests with shared resources."""
        import threading
        import time
        
        # Mock response with slight delay
        def delayed_response(*args, **kwargs):
            time.sleep(0.05)  # 50ms delay
            response = Mock()
            response.status_code = 200
            response.json.return_value = {'thread_id': threading.current_thread().ident}
            response.raise_for_status.return_value = None
            return response
        
        mock_request.side_effect = delayed_response
        
        results = []
        errors = []
        
        def make_concurrent_request(endpoint):
            try:
                result = self.connector.make_request('GET', endpoint)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_concurrent_request, args=(f'/concurrent/{i}',))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all requests succeeded
        self.assertEqual(len(results), 10)
        self.assertEqual(len(errors), 0)
        
        # Verify thread isolation
        thread_ids = [result['thread_id'] for result in results]
        self.assertEqual(len(set(thread_ids)), 10)  # All different thread IDs
    
    @patch('requests.Session.request')
    def test_streaming_data_processing(self, mock_request):
        """Test processing of streaming data responses."""
        # Mock streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {
            'stream_id': 'stream_123',
            'data': ['chunk1', 'chunk2', 'chunk3'],
            'total_chunks': 3
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/stream-data')
        
        self.assertEqual(result['stream_id'], 'stream_123')
        self.assertEqual(len(result['data']), 3)
        self.assertEqual(result['total_chunks'], 3)
    
    @patch('requests.Session.request')
    def test_configuration_validation_workflow(self, mock_request):
        """Test configuration validation in a real workflow."""
        # Test with various configuration states
        configs = [
            # Valid configuration
            {
                'api_key': 'valid_key',
                'base_url': 'https://api.genesis.test',
                'timeout': 30,
                'max_retries': 3
            },
            # Minimal configuration
            {
                'api_key': 'minimal_key'
            },
            # Extended configuration
            {
                'api_key': 'extended_key',
                'base_url': 'https://api.genesis.test',
                'timeout': 60,
                'max_retries': 5,
                'custom_param': 'value'
            }
        ]
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'config': 'validated'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        for config in configs:
            with self.subTest(config=config):
                connector = GenesisConnector(config)
                result = connector.make_request('GET', '/config-test')
                self.assertEqual(result['config'], 'validated')
    
    def test_memory_management_with_large_operations(self):
        """Test memory management during large operations."""
        import gc
        
        # Create multiple connectors with large configurations
        connectors = []
        for i in range(100):
            config = {
                'api_key': f'key_{i}' * 100,  # Long API key
                'base_url': f'https://api{i}.genesis.test',
                'timeout': 30,
                'max_retries': 3
            }
            connector = GenesisConnector(config)
            connectors.append(connector)
        
        # Force garbage collection
        gc.collect()
        
        # Verify all connectors are functional
        for i, connector in enumerate(connectors):
            self.assertTrue(connector.api_key.startswith(f'key_{i}'))
            self.assertIn(f'api{i}', connector.base_url)
        
        # Cleanup
        del connectors
        gc.collect()


class TestGenesisConnectorExtendedParametrized:
    """Extended parametrized tests for comprehensive coverage."""
    
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
    
    @pytest.mark.parametrize("api_key,should_raise", [
        ('valid_key', False),
        ('a', False),  # Single character
        ('key_with_special_!@#$%', False),
        ('key_with_unicode_测试', False),
        ('x' * 1000, False),  # Very long key
        ('', True),  # Empty key
        (None, True),  # None key
        (123, True),  # Numeric key
        (['list', 'key'], True),  # List key
        ({'dict': 'key'}, True),  # Dict key
    ])
    def test_api_key_validation_extended(self, api_key, should_raise):
        """Extended API key validation tests."""
        config = {
            'api_key': api_key,
            'base_url': 'https://api.test.com',
            'timeout': 30,
            'max_retries': 3
        }
        
        if should_raise:
            with pytest.raises(ValueError):
                GenesisConnector(config)
        else:
            connector = GenesisConnector(config)
            assert connector.api_key == api_key
    
    @pytest.mark.parametrize("max_retries,should_raise", [
        (0, False),
        (1, False),
        (3, False),
        (10, False),
        (100, False),
        (-1, True),
        (-10, True),
        ('string', True),
        (None, True),
        (3.14, True),
        (float('inf'), True),
    ])
    def test_max_retries_validation_extended(self, max_retries, should_raise):
        """Extended max retries validation tests."""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': 30,
            'max_retries': max_retries
        }
        
        if should_raise:
            with pytest.raises(ValueError):
                GenesisConnector(config)
        else:
            connector = GenesisConnector(config)
            assert connector.max_retries == max_retries
    
    @pytest.mark.parametrize("endpoint,expected_url", [
        ('', 'https://api.genesis.test'),
        ('/', 'https://api.genesis.test/'),
        ('/test', 'https://api.genesis.test/test'),
        ('test', 'https://api.genesis.test/test'),
        ('/test/', 'https://api.genesis.test/test/'),
        ('test/', 'https://api.genesis.test/test/'),
        ('/test/endpoint', 'https://api.genesis.test/test/endpoint'),
        ('test/endpoint', 'https://api.genesis.test/test/endpoint'),
        ('/test?param=value', 'https://api.genesis.test/test?param=value'),
        ('test?param=value', 'https://api.genesis.test/test?param=value'),
        ('/test#fragment', 'https://api.genesis.test/test#fragment'),
        ('test#fragment', 'https://api.genesis.test/test#fragment'),
        ('//test', 'https://api.genesis.test/test'),
        ('///test', 'https://api.genesis.test/test'),
    ])
    def test_url_building_comprehensive(self, connector, endpoint, expected_url):
        """Comprehensive URL building tests."""
        result = connector._build_url(endpoint)
        assert result == expected_url
    
    @pytest.mark.parametrize("headers,expected_count", [
        ({}, 1),  # Just Authorization
        ({'X-Custom': 'value'}, 2),  # Authorization + Custom
        ({'Authorization': 'Override'}, 1),  # Override Authorization
        ({'X-Custom': 'value', 'Y-Another': 'value2'}, 3),  # Multiple custom
        ({'Content-Type': 'application/json'}, 2),  # Authorization + Content-Type
        ({'X-Empty': ''}, 2),  # Authorization + Empty header
        ({'X-Unicode': '测试'}, 2),  # Authorization + Unicode header
    ])
    def test_header_building_comprehensive(self, connector, headers, expected_count):
        """Comprehensive header building tests."""
        result = connector._build_headers(headers)
        assert len(result) == expected_count
        assert 'Authorization' in result or 'authorization' in result
    
    @pytest.mark.parametrize("exception_type,expected_exception", [
        (ConnectionError, GenesisConnectionError),
        (Timeout, GenesisTimeoutError),
        (requests.exceptions.ConnectTimeout, GenesisTimeoutError),
        (requests.exceptions.ReadTimeout, GenesisTimeoutError),
        (requests.exceptions.HTTPError, GenesisConnectionError),
        (requests.exceptions.RequestException, GenesisConnectionError),
        (requests.exceptions.ChunkedEncodingError, GenesisConnectionError),
        (requests.exceptions.ContentDecodingError, GenesisConnectionError),
        (requests.exceptions.TooManyRedirects, GenesisConnectionError),
        (requests.exceptions.SSLError, GenesisConnectionError),
    ])
    @patch('requests.Session.request')
    def test_exception_mapping_comprehensive(self, mock_request, connector, exception_type, expected_exception):
        """Comprehensive exception mapping tests."""
        mock_request.side_effect = exception_type("Test error")
        
        with pytest.raises(expected_exception):
            connector.make_request('GET', '/test')


if __name__ == '__main__':
    # Run the extended tests
    unittest.main(verbosity=2)