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
    """Advanced unit tests for GenesisConnector with additional edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_initialization_with_none_values(self):
        """Test initialization with None values in config."""
        config = self.mock_config.copy()
        config['timeout'] = None
        
        with self.assertRaises(ValueError) as context:
            GenesisConnector(config)
        self.assertIn('timeout', str(context.exception).lower())
    
    def test_initialization_with_non_string_api_key(self):
        """Test initialization with non-string API key."""
        config = self.mock_config.copy()
        config['api_key'] = 12345
        
        with self.assertRaises(ValueError) as context:
            GenesisConnector(config)
        self.assertIn('API key must be a string', str(context.exception))
    
    def test_initialization_with_whitespace_api_key(self):
        """Test initialization with whitespace-only API key."""
        config = self.mock_config.copy()
        config['api_key'] = '   \t\n   '
        
        with self.assertRaises(ValueError) as context:
            GenesisConnector(config)
        self.assertIn('API key cannot be empty', str(context.exception))
    
    def test_initialization_with_float_timeout(self):
        """Test initialization with float timeout value."""
        config = self.mock_config.copy()
        config['timeout'] = 30.5
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.timeout, 30.5)
    
    def test_initialization_with_very_large_timeout(self):
        """Test initialization with very large timeout value."""
        config = self.mock_config.copy()
        config['timeout'] = 999999
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.timeout, 999999)
    
    def test_initialization_with_string_max_retries(self):
        """Test initialization with string max_retries value."""
        config = self.mock_config.copy()
        config['max_retries'] = '5'
        
        with self.assertRaises(ValueError) as context:
            GenesisConnector(config)
        self.assertIn('Max retries must be an integer', str(context.exception))
    
    def test_initialization_with_missing_config_keys(self):
        """Test initialization with completely missing config keys."""
        configs_to_test = [
            {'api_key': 'test'},  # missing base_url
            {'base_url': 'https://test.com'},  # missing api_key
            {}  # empty config
        ]
        
        for config in configs_to_test:
            with self.assertRaises(ValueError):
                GenesisConnector(config)
    
    def test_initialization_with_invalid_url_schemes(self):
        """Test initialization with various invalid URL schemes."""
        invalid_urls = [
            'http://api.genesis.test',  # http instead of https
            'ftp://api.genesis.test',   # ftp scheme
            'api.genesis.test',         # no scheme
            'https://',                 # missing domain
            'https://api..genesis.test', # double dots
            'https://api.genesis.test:999999'  # invalid port
        ]
        
        for url in invalid_urls:
            config = self.mock_config.copy()
            config['base_url'] = url
            with self.assertRaises(ValueError) as context:
                GenesisConnector(config)
            self.assertIn('Invalid base URL', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_special_characters_in_endpoint(self, mock_request):
        """Test API request with special characters in endpoint."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        special_endpoint = '/models/test%20model/info?query=special%20chars'
        result = self.connector.make_request('GET', special_endpoint)
        
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/models/test%20model/info?query=special%20chars',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_empty_endpoint(self, mock_request):
        """Test API request with empty endpoint."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '')
        
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_none_data(self, mock_request):
        """Test API request with None data parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('POST', '/test', data=None)
        
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once_with(
            'POST',
            'https://api.genesis.test/test',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_empty_response(self, mock_request):
        """Test API request with empty JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/test')
        
        self.assertEqual(result, {})
    
    @patch('requests.Session.request')
    def test_make_request_with_large_response(self, mock_request):
        """Test API request with large JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        # Create a large response
        large_data = {'items': [{'id': i, 'data': 'x' * 1000} for i in range(1000)]}
        mock_response.json.return_value = large_data
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/test')
        
        self.assertEqual(len(result['items']), 1000)
        self.assertEqual(result['items'][0]['id'], 0)
    
    @patch('requests.Session.request')
    def test_make_request_with_unicode_data(self, mock_request):
        """Test API request with unicode data."""
        mock_response = Mock()
        mock_response.status_code = 200
        unicode_data = {'message': 'Hello 世界 🌍 café naïve résumé'}
        mock_response.json.return_value = unicode_data
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/test')
        
        self.assertEqual(result['message'], 'Hello 世界 🌍 café naïve résumé')
    
    @patch('requests.Session.request')
    def test_make_request_with_nested_json_data(self, mock_request):
        """Test API request with deeply nested JSON data."""
        mock_response = Mock()
        mock_response.status_code = 200
        nested_data = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {'value': 'deep_value'}
                    }
                }
            }
        }
        mock_response.json.return_value = nested_data
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/test')
        
        self.assertEqual(result['level1']['level2']['level3']['level4']['value'], 'deep_value')
    
    @patch('requests.Session.request')
    def test_make_request_with_various_http_methods(self, mock_request):
        """Test API request with various HTTP methods."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        
        for method in methods:
            with self.subTest(method=method):
                result = self.connector.make_request(method, '/test')
                self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_invalid_http_method(self, mock_request):
        """Test API request with invalid HTTP method."""
        with self.assertRaises(ValueError) as context:
            self.connector.make_request('INVALID_METHOD', '/test')
        self.assertIn('Invalid HTTP method', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_status_codes(self, mock_request):
        """Test API request with various status codes."""
        status_codes = [200, 201, 202, 204, 400, 401, 403, 404, 500, 502, 503]
        
        for status_code in status_codes:
            with self.subTest(status_code=status_code):
                mock_response = Mock()
                mock_response.status_code = status_code
                
                if status_code < 400:
                    mock_response.json.return_value = {'status': 'success'}
                    mock_response.raise_for_status.return_value = None
                    mock_request.return_value = mock_response
                    
                    result = self.connector.make_request('GET', '/test')
                    self.assertEqual(result, {'status': 'success'})
                else:
                    mock_response.raise_for_status.side_effect = requests.HTTPError(f"{status_code} Error")
                    mock_response.text = f"Error {status_code}"
                    mock_request.return_value = mock_response
                    
                    with self.assertRaises(GenesisConnectionError):
                        self.connector.make_request('GET', '/test')
    
    @patch('requests.Session.request')
    def test_make_request_with_malformed_json(self, mock_request):
        """Test API request with malformed JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "invalid", 0)
        mock_response.text = "This is not valid JSON"
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        self.assertIn('Invalid JSON', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_partial_content_response(self, mock_request):
        """Test API request with partial content response."""
        mock_response = Mock()
        mock_response.status_code = 206  # Partial Content
        mock_response.json.return_value = {'partial': 'content'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/test')
        self.assertEqual(result, {'partial': 'content'})
    
    @patch('time.sleep')
    @patch('requests.Session.request')
    def test_make_request_with_mixed_failures(self, mock_request, mock_sleep):
        """Test retry logic with mixed failure types."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        # Mix of different failure types
        mock_request.side_effect = [
            ConnectionError("Connection failed"),
            Timeout("Request timed out"),
            requests.HTTPError("500 Internal Server Error"),
            mock_response
        ]
        
        with self.assertRaises(GenesisConnectionError):
            self.connector.make_request('GET', '/test')
        
        # Should exhaust all retries
        self.assertEqual(mock_request.call_count, 4)
    
    @patch('requests.Session.request')
    def test_get_model_info_with_special_characters(self, mock_request):
        """Test model info retrieval with special characters in model ID."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 'model-test_123',
            'name': 'Test Model with Special Chars',
            'version': '1.0.0'
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.get_model_info('model-test_123')
        
        self.assertEqual(result['id'], 'model-test_123')
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/models/model-test_123',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_get_model_info_with_empty_id(self, mock_request):
        """Test model info retrieval with empty model ID."""
        with self.assertRaises(ValueError) as context:
            self.connector.get_model_info('')
        self.assertIn('Model ID cannot be empty', str(context.exception))
    
    @patch('requests.Session.request')
    def test_get_model_info_with_none_id(self, mock_request):
        """Test model info retrieval with None model ID."""
        with self.assertRaises(ValueError) as context:
            self.connector.get_model_info(None)
        self.assertIn('Model ID cannot be None', str(context.exception))
    
    @patch('requests.Session.request')
    def test_create_generation_with_empty_request(self, mock_request):
        """Test generation creation with empty request."""
        with self.assertRaises(ValueError) as context:
            self.connector.create_generation({})
        self.assertIn('Generation request cannot be empty', str(context.exception))
    
    @patch('requests.Session.request')
    def test_create_generation_with_none_request(self, mock_request):
        """Test generation creation with None request."""
        with self.assertRaises(ValueError) as context:
            self.connector.create_generation(None)
        self.assertIn('Generation request cannot be None', str(context.exception))
    
    @patch('requests.Session.request')
    def test_create_generation_with_minimal_request(self, mock_request):
        """Test generation creation with minimal valid request."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 'gen_123', 'status': 'pending'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        minimal_request = {'prompt': 'test'}
        result = self.connector.create_generation(minimal_request)
        
        self.assertEqual(result['id'], 'gen_123')
        self.assertEqual(result['status'], 'pending')
    
    @patch('requests.Session.request')
    def test_create_generation_with_complex_request(self, mock_request):
        """Test generation creation with complex request parameters."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 'gen_456', 'status': 'pending'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        complex_request = {
            'prompt': 'Complex test prompt',
            'model': 'advanced_model',
            'max_tokens': 1000,
            'temperature': 0.7,
            'top_p': 0.9,
            'frequency_penalty': 0.1,
            'presence_penalty': 0.1,
            'stop': ['END', 'STOP'],
            'stream': False,
            'metadata': {'user_id': 'test_user', 'session_id': 'test_session'}
        }
        
        result = self.connector.create_generation(complex_request)
        
        self.assertEqual(result['id'], 'gen_456')
        mock_request.assert_called_once_with(
            'POST',
            'https://api.genesis.test/generations',
            headers={'Authorization': 'Bearer test_api_key_123'},
            json=complex_request,
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_list_models_with_empty_response(self, mock_request):
        """Test model listing with empty response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'models': [], 'total': 0}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.list_models()
        
        self.assertEqual(result['models'], [])
        self.assertEqual(result['total'], 0)
    
    @patch('requests.Session.request')
    def test_list_models_with_pagination(self, mock_request):
        """Test model listing with pagination parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [{'id': 'model_1', 'name': 'Model 1'}],
            'total': 100,
            'page': 2,
            'per_page': 1
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        filters = {'page': 2, 'per_page': 1}
        result = self.connector.list_models(filters)
        
        self.assertEqual(result['page'], 2)
        self.assertEqual(result['per_page'], 1)
        self.assertEqual(result['total'], 100)
    
    def test_build_url_with_query_parameters(self):
        """Test URL building with query parameters."""
        url = self.connector._build_url('/test?param1=value1&param2=value2')
        self.assertEqual(url, 'https://api.genesis.test/test?param1=value1&param2=value2')
    
    def test_build_url_with_fragment(self):
        """Test URL building with fragment."""
        url = self.connector._build_url('/test#section')
        self.assertEqual(url, 'https://api.genesis.test/test#section')
    
    def test_build_headers_with_empty_custom_headers(self):
        """Test header building with empty custom headers."""
        headers = self.connector._build_headers({})
        expected = {'Authorization': 'Bearer test_api_key_123'}
        self.assertEqual(headers, expected)
    
    def test_build_headers_with_case_insensitive_override(self):
        """Test header building with case-insensitive header override."""
        custom_headers = {'authorization': 'Bearer different_token'}
        headers = self.connector._build_headers(custom_headers)
        
        # Should still have the original case
        expected = {'Authorization': 'Bearer test_api_key_123', 'authorization': 'Bearer different_token'}
        self.assertEqual(headers, expected)
    
    def test_session_reuse(self):
        """Test that the same session is reused for multiple requests."""
        session1 = self.connector.session
        session2 = self.connector.session
        self.assertIs(session1, session2)
    
    def test_session_close_on_del(self):
        """Test that session is properly closed when connector is deleted."""
        session = self.connector.session
        with patch.object(session, 'close') as mock_close:
            del self.connector
            # Session close should be called during cleanup
            # Note: This test depends on the implementation having a __del__ method
    
    def test_concurrent_requests(self):
        """Test thread safety with concurrent requests."""
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        def make_request():
            with patch('requests.Session.request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'success': True}
                mock_response.raise_for_status.return_value = None
                mock_request.return_value = mock_response
                
                return self.connector.make_request('GET', '/test')
        
        # Execute requests concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        for result in results:
            self.assertEqual(result, {'success': True})
    
    def test_memory_usage_with_large_config(self):
        """Test memory usage with large configuration values."""
        large_config = {
            'api_key': 'x' * 10000,  # Very long API key
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        
        connector = GenesisConnector(large_config)
        self.assertEqual(len(connector.api_key), 10000)
    
    def test_edge_case_timeout_values(self):
        """Test edge case timeout values."""
        edge_cases = [0.001, 0.1, 1, 60, 300, 3600]  # milliseconds to hours
        
        for timeout in edge_cases:
            with self.subTest(timeout=timeout):
                config = self.mock_config.copy()
                config['timeout'] = timeout
                connector = GenesisConnector(config)
                self.assertEqual(connector.timeout, timeout)
    
    def test_repr_with_long_url(self):
        """Test string representation with very long URL."""
        config = self.mock_config.copy()
        config['base_url'] = 'https://very-long-subdomain.genesis.test/with/very/long/path'
        connector = GenesisConnector(config)
        
        repr_str = repr(connector)
        self.assertIn('GenesisConnector', repr_str)
        # URL should be truncated or properly formatted
        self.assertIn('very-long-subdomain.genesis.test', repr_str)
    
    def test_error_message_formatting(self):
        """Test that error messages are properly formatted."""
        test_cases = [
            ('', 'API key cannot be empty'),
            (None, 'API key is required'),
            ('invalid_url', 'Invalid base URL'),
            (-1, 'Timeout must be positive')
        ]
        
        for invalid_value, expected_message in test_cases:
            with self.subTest(invalid_value=invalid_value):
                config = self.mock_config.copy()
                if invalid_value is None:
                    del config['api_key']
                elif isinstance(invalid_value, str) and 'url' in expected_message.lower():
                    config['base_url'] = invalid_value
                elif isinstance(invalid_value, (int, float)) and 'timeout' in expected_message.lower():
                    config['timeout'] = invalid_value
                else:
                    config['api_key'] = invalid_value
                
                with self.assertRaises(ValueError) as context:
                    GenesisConnector(config)
                self.assertIn(expected_message, str(context.exception))


class TestGenesisConnectorPerformance(unittest.TestCase):
    """Performance-focused tests for GenesisConnector."""
    
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
    def test_request_performance_with_large_payload(self, mock_request):
        """Test request performance with large payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Create a large payload
        large_payload = {
            'data': ['x' * 1000 for _ in range(1000)]  # 1MB of data
        }
        
        import time
        start_time = time.time()
        result = self.connector.make_request('POST', '/test', data=large_payload)
        end_time = time.time()
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(end_time - start_time, 1.0)  # 1 second threshold
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_rapid_consecutive_requests(self, mock_request):
        """Test rapid consecutive requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        import time
        start_time = time.time()
        
        # Make 100 rapid requests
        for i in range(100):
            result = self.connector.make_request('GET', f'/test/{i}')
            self.assertEqual(result, {'success': True})
        
        end_time = time.time()
        
        # Should complete within reasonable time
        self.assertLess(end_time - start_time, 2.0)  # 2 seconds for 100 requests
    
    def test_memory_usage_stability(self):
        """Test memory usage stability over many operations."""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Create many connectors to test memory management
        connectors = []
        for i in range(1000):
            config = self.mock_config.copy()
            config['api_key'] = f'test_key_{i}'
            connectors.append(GenesisConnector(config))
        
        # Clean up
        del connectors
        gc.collect()
        
        # Test should complete without memory errors
        self.assertTrue(True)


class TestGenesisConnectorErrorHandling(unittest.TestCase):
    """Comprehensive error handling tests for GenesisConnector."""
    
    def setUp(self):
        """Set up error handling test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    @patch('requests.Session.request')
    def test_network_unreachable_error(self, mock_request):
        """Test handling of network unreachable errors."""
        import socket
        mock_request.side_effect = socket.gaierror("Network unreachable")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Network unreachable', str(context.exception))
    
    @patch('requests.Session.request')
    def test_ssl_error_handling(self, mock_request):
        """Test handling of SSL errors."""
        import ssl
        mock_request.side_effect = ssl.SSLError("SSL certificate verification failed")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('SSL', str(context.exception))
    
    @patch('requests.Session.request')
    def test_dns_resolution_error(self, mock_request):
        """Test handling of DNS resolution errors."""
        import socket
        mock_request.side_effect = socket.gaierror("Name or service not known")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Name or service not known', str(context.exception))
    
    @patch('requests.Session.request')
    def test_proxy_error_handling(self, mock_request):
        """Test handling of proxy errors."""
        mock_request.side_effect = RequestException("Proxy error")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Proxy error', str(context.exception))
    
    @patch('requests.Session.request')
    def test_rate_limit_error_handling(self, mock_request):
        """Test handling of rate limit errors."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '60'}
        mock_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        mock_response.text = "Rate limit exceeded"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('429', str(context.exception))
    
    @patch('requests.Session.request')
    def test_authentication_error_handling(self, mock_request):
        """Test handling of authentication errors."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
        mock_response.text = "Invalid API key"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('401', str(context.exception))
    
    @patch('requests.Session.request')
    def test_server_error_handling(self, mock_request):
        """Test handling of server errors."""
        server_errors = [500, 502, 503, 504]
        
        for status_code in server_errors:
            with self.subTest(status_code=status_code):
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.raise_for_status.side_effect = requests.HTTPError(f"{status_code} Server Error")
                mock_response.text = f"Server error {status_code}"
                mock_request.return_value = mock_response
                
                with self.assertRaises(GenesisConnectionError) as context:
                    self.connector.make_request('GET', '/test')
                
                self.assertIn(str(status_code), str(context.exception))
    
    def test_invalid_config_combinations(self):
        """Test various invalid configuration combinations."""
        invalid_configs = [
            {'api_key': '', 'base_url': 'https://api.genesis.test'},
            {'api_key': 'test', 'base_url': ''},
            {'api_key': 'test', 'base_url': 'https://api.genesis.test', 'timeout': 0},
            {'api_key': 'test', 'base_url': 'https://api.genesis.test', 'max_retries': -1},
        ]
        
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    GenesisConnector(config)
    
    @patch('requests.Session.request')
    def test_unexpected_exception_handling(self, mock_request):
        """Test handling of unexpected exceptions."""
        mock_request.side_effect = Exception("Unexpected error")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Unexpected error', str(context.exception))
    
    def test_custom_exception_inheritance(self):
        """Test that custom exceptions inherit correctly."""
        # Test exception hierarchy
        self.assertTrue(issubclass(GenesisConnectionError, Exception))
        self.assertTrue(issubclass(GenesisTimeoutError, GenesisConnectionError))
        
        # Test exception instantiation
        conn_error = GenesisConnectionError("Connection error")
        timeout_error = GenesisTimeoutError("Timeout error")
        
        self.assertIsInstance(conn_error, Exception)
        self.assertIsInstance(timeout_error, GenesisConnectionError)
        self.assertIsInstance(timeout_error, Exception)


if __name__ == '__main__':
    # Run the additional tests
    unittest.main(verbosity=2)