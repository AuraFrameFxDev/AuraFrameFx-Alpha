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
    
class TestGenesisConnectorEdgeCases(unittest.TestCase):
    """Additional edge case tests for GenesisConnector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_initialization_with_unicode_api_key(self):
        """Test initialization with unicode characters in API key."""
        config = self.mock_config.copy()
        config['api_key'] = 'test_ключ_123_🔑'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test_ключ_123_🔑')
    
    def test_initialization_with_very_long_api_key(self):
        """Test initialization with extremely long API key."""
        config = self.mock_config.copy()
        config['api_key'] = 'a' * 10000  # 10KB API key
        
        connector = GenesisConnector(config)
        self.assertEqual(len(connector.api_key), 10000)
    
    def test_initialization_with_various_url_formats(self):
        """Test initialization with different URL formats."""
        test_cases = [
            'https://api.genesis.test:8080',
            'https://api.genesis.test/v1',
            'https://api.genesis.test/v1/',
            'https://api-staging.genesis.test',
            'https://localhost:3000'
        ]
        
        for url in test_cases:
            config = self.mock_config.copy()
            config['base_url'] = url
            connector = GenesisConnector(config)
            self.assertEqual(connector.base_url, url)
    
    def test_initialization_with_edge_case_timeout_values(self):
        """Test initialization with edge case timeout values."""
        test_cases = [0.1, 0.001, 300, 3600]  # Very small to very large timeouts
        
        for timeout in test_cases:
            config = self.mock_config.copy()
            config['timeout'] = timeout
            connector = GenesisConnector(config)
            self.assertEqual(connector.timeout, timeout)
    
    def test_initialization_with_string_numeric_values(self):
        """Test initialization with string numeric values."""
        config = self.mock_config.copy()
        config['timeout'] = '30'
        config['max_retries'] = '3'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.timeout, 30)
        self.assertEqual(connector.max_retries, 3)
    
    def test_initialization_with_none_values(self):
        """Test initialization with None values in config."""
        config = self.mock_config.copy()
        config['timeout'] = None
        
        with self.assertRaises(ValueError):
            GenesisConnector(config)
    
    def test_initialization_with_extra_config_fields(self):
        """Test initialization ignores extra config fields."""
        config = self.mock_config.copy()
        config['extra_field'] = 'should_be_ignored'
        config['another_field'] = 123
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test_api_key_123')
        self.assertFalse(hasattr(connector, 'extra_field'))
    
    @patch('requests.Session.request')
    def test_make_request_with_very_large_payload(self, mock_request):
        """Test making request with very large data payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # 1MB payload
        large_data = {'data': 'x' * 1024 * 1024}
        
        result = self.connector.make_request('POST', '/test', data=large_data)
        
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once()
    
    @patch('requests.Session.request')
    def test_make_request_with_special_characters_in_endpoint(self, mock_request):
        """Test making request with special characters in endpoint."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        endpoints = [
            '/test/endpoint%20with%20spaces',
            '/test/endpoint-with-dashes',
            '/test/endpoint_with_underscores',
            '/test/endpoint.with.dots',
            '/test/endpoint@special#chars'
        ]
        
        for endpoint in endpoints:
            result = self.connector.make_request('GET', endpoint)
            self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_unicode_data(self, mock_request):
        """Test making request with unicode data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        unicode_data = {
            'text': 'Hello 世界 🌍',
            'emoji': '🚀🎉💡',
            'cyrillic': 'Привет мир',
            'arabic': 'مرحبا بالعالم'
        }
        
        result = self.connector.make_request('POST', '/test', data=unicode_data)
        
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once()
    
    @patch('requests.Session.request')
    def test_make_request_with_empty_response(self, mock_request):
        """Test handling of empty response body."""
        mock_response = Mock()
        mock_response.status_code = 204  # No Content
        mock_response.json.side_effect = json.JSONDecodeError("Empty response", "", 0)
        mock_response.text = ""
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Should handle empty response gracefully
        result = self.connector.make_request('DELETE', '/test')
        self.assertIsNone(result)
    
    @patch('requests.Session.request')
    def test_make_request_with_malformed_json(self, mock_request):
        """Test handling of malformed JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Malformed JSON", "", 0)
        mock_response.text = '{"invalid": json,}'
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Malformed JSON', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_nested_json_error(self, mock_request):
        """Test handling of valid JSON but with nested error information."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            'error': {
                'code': 'INVALID_REQUEST',
                'message': 'Request validation failed',
                'details': [
                    {'field': 'prompt', 'message': 'Cannot be empty'},
                    {'field': 'model', 'message': 'Invalid model ID'}
                ]
            }
        }
        mock_response.raise_for_status.side_effect = requests.HTTPError("400 Bad Request")
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('POST', '/test')
        
        self.assertIn('400 Bad Request', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_rate_limit_headers(self, mock_request):
        """Test handling of rate limit response headers."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {
            'X-RateLimit-Limit': '1000',
            'X-RateLimit-Remaining': '0',
            'X-RateLimit-Reset': '1640995200',
            'Retry-After': '3600'
        }
        mock_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        mock_response.text = "Rate limit exceeded"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('429 Too Many Requests', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_server_error_variations(self, mock_request):
        """Test handling of different server error types."""
        error_cases = [
            (500, "Internal Server Error"),
            (502, "Bad Gateway"),
            (503, "Service Unavailable"),
            (504, "Gateway Timeout"),
            (507, "Insufficient Storage"),
            (511, "Network Authentication Required")
        ]
        
        for status_code, error_message in error_cases:
            mock_response = Mock()
            mock_response.status_code = status_code
            mock_response.raise_for_status.side_effect = requests.HTTPError(f"{status_code} {error_message}")
            mock_response.text = error_message
            mock_request.return_value = mock_response
            
            with self.assertRaises(GenesisConnectionError) as context:
                self.connector.make_request('GET', '/test')
            
            self.assertIn(str(status_code), str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_redirect_responses(self, mock_request):
        """Test handling of redirect responses."""
        redirect_cases = [
            (301, "Moved Permanently"),
            (302, "Found"),
            (307, "Temporary Redirect"),
            (308, "Permanent Redirect")
        ]
        
        for status_code, redirect_type in redirect_cases:
            mock_response = Mock()
            mock_response.status_code = status_code
            mock_response.headers = {'Location': 'https://api.genesis.test/v2/test'}
            mock_response.json.return_value = {'message': f'{redirect_type}'}
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response
            
            result = self.connector.make_request('GET', '/test')
            self.assertEqual(result, {'message': f'{redirect_type}'})
    
    @patch('requests.Session.request')
    def test_concurrent_requests_thread_safety(self, mock_request):
        """Test thread safety of concurrent requests."""
        import threading
        import queue
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        results = queue.Queue()
        threads = []
        
        def make_request_thread(connector, endpoint):
            try:
                result = connector.make_request('GET', endpoint)
                results.put(result)
            except Exception as e:
                results.put(e)
        
        # Start 10 concurrent requests
        for i in range(10):
            thread = threading.Thread(
                target=make_request_thread,
                args=(self.connector, f'/test/{i}')
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all requests succeeded
        self.assertEqual(results.qsize(), 10)
        while not results.empty():
            result = results.get()
            self.assertEqual(result, {'success': True})
    
    def test_memory_usage_with_large_config(self):
        """Test memory usage with large configuration objects."""
        import sys
        
        # Create a large configuration
        large_config = self.mock_config.copy()
        large_config['large_field'] = 'x' * 1024 * 1024  # 1MB string
        
        connector = GenesisConnector(large_config)
        
        # Verify the connector doesn't store unnecessary large data
        connector_size = sys.getsizeof(connector.__dict__)
        self.assertLess(connector_size, 1024 * 1024)  # Should be much smaller than 1MB
    
    @patch('requests.Session.request')
    def test_request_with_custom_user_agent(self, mock_request):
        """Test request with custom user agent header."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        custom_headers = {'User-Agent': 'GenesisConnector/1.0 (Custom)'}
        result = self.connector.make_request('GET', '/test', headers=custom_headers)
        
        self.assertEqual(result, {'success': True})
        expected_headers = {
            'Authorization': 'Bearer test_api_key_123',
            'User-Agent': 'GenesisConnector/1.0 (Custom)'
        }
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/test',
            headers=expected_headers,
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_request_with_binary_data(self, mock_request):
        """Test request with binary data payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        binary_data = b'\x00\x01\x02\x03\x04\x05'
        
        # This should handle binary data appropriately
        result = self.connector.make_request('POST', '/upload', data={'file': binary_data})
        
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once()
    
    def test_url_building_edge_cases(self):
        """Test URL building with various edge cases."""
        test_cases = [
            ('', 'https://api.genesis.test/'),
            ('/', 'https://api.genesis.test/'),
            ('//double//slash', 'https://api.genesis.test/double/slash'),
            ('query?param=value', 'https://api.genesis.test/query?param=value'),
            ('fragment#section', 'https://api.genesis.test/fragment#section'),
            ('encoded%20path', 'https://api.genesis.test/encoded%20path')
        ]
        
        for endpoint, expected in test_cases:
            result = self.connector._build_url(endpoint)
            self.assertEqual(result, expected)
    
    def test_headers_with_none_values(self):
        """Test header building with None values."""
        custom_headers = {
            'X-Custom': 'value',
            'X-None': None,
            'X-Empty': ''
        }
        
        headers = self.connector._build_headers(custom_headers)
        
        # None values should be filtered out
        self.assertNotIn('X-None', headers)
        self.assertIn('X-Empty', headers)
        self.assertEqual(headers['X-Empty'], '')
    
    @patch('requests.Session.request')
    def test_request_method_variations(self, mock_request):
        """Test all supported HTTP methods."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        
        for method in methods:
            result = self.connector.make_request(method, '/test')
            self.assertEqual(result, {'success': True})
            
            # Verify correct method was used
            mock_request.assert_called_with(
                method,
                'https://api.genesis.test/test',
                headers={'Authorization': 'Bearer test_api_key_123'},
                timeout=30
            )
        
        self.assertEqual(mock_request.call_count, len(methods))


class TestGenesisConnectorSecurity(unittest.TestCase):
    """Security-focused tests for GenesisConnector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_api_key_not_logged_in_repr(self):
        """Test that API key is not exposed in string representations."""
        repr_str = repr(self.connector)
        str_str = str(self.connector)
        
        # API key should not appear in full
        self.assertNotIn('test_api_key_123', repr_str)
        self.assertNotIn('test_api_key_123', str_str)
        
        # Should show masked version
        self.assertIn('***', repr_str)
    
    def test_api_key_not_in_exception_messages(self):
        """Test that API key doesn't leak in exception messages."""
        with patch('requests.Session.request') as mock_request:
            mock_request.side_effect = ConnectionError("Connection failed")
            
            with self.assertRaises(GenesisConnectionError) as context:
                self.connector.make_request('GET', '/test')
            
            # API key should not appear in error message
            self.assertNotIn('test_api_key_123', str(context.exception))
    
    @patch('requests.Session.request')
    def test_sensitive_data_not_logged(self, mock_request):
        """Test that sensitive data is not logged."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        with patch('logging.getLogger') as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log
            
            sensitive_data = {
                'password': 'secret123',
                'token': 'private_token',
                'api_key': 'another_secret'
            }
            
            self.connector.make_request('POST', '/test', data=sensitive_data)
            
            # Check that sensitive data is not in any log calls
            for call in mock_log.debug.call_args_list:
                log_message = str(call)
                self.assertNotIn('secret123', log_message)
                self.assertNotIn('private_token', log_message)
    
    def test_malicious_endpoint_injection(self):
        """Test protection against endpoint injection attacks."""
        malicious_endpoints = [
            '../../../etc/passwd',
            '../../admin/delete',
            '/admin/users/../../../system/shutdown',
            'javascript:alert(1)',
            'file:///etc/passwd'
        ]
        
        for endpoint in malicious_endpoints:
            # Should handle malicious endpoints by URL construction
            url = self.connector._build_url(endpoint)
            self.assertTrue(url.startswith('https://api.genesis.test/'))
    
    def test_header_injection_prevention(self):
        """Test prevention of header injection attacks."""
        malicious_headers = {
            'X-Injected\r\nX-Evil': 'value',
            'X-CRLF\r\n\r\nGET /evil': 'injection',
            'X-Normal': 'value\r\nX-Injected: evil'
        }
        
        headers = self.connector._build_headers(malicious_headers)
        
        # Headers should be sanitized or rejected
        for key, value in headers.items():
            self.assertNotIn('\r', key)
            self.assertNotIn('\n', key)
            self.assertNotIn('\r', str(value))
            self.assertNotIn('\n', str(value))


class TestGenesisConnectorPerformance(unittest.TestCase):
    """Performance-focused tests for GenesisConnector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_connector_initialization_performance(self):
        """Test connector initialization performance."""
        import time
        
        start_time = time.time()
        
        for _ in range(100):
            GenesisConnector(self.mock_config)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should be able to create 100 connectors in reasonable time
        self.assertLess(total_time, 1.0)  # Less than 1 second
    
    @patch('requests.Session.request')
    def test_rapid_requests_performance(self, mock_request):
        """Test performance of rapid sequential requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        import time
        start_time = time.time()
        
        for i in range(50):
            self.connector.make_request('GET', f'/test/{i}')
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 50 requests efficiently
        self.assertLess(total_time, 0.5)  # Less than 0.5 seconds
    
    def test_memory_leak_prevention(self):
        """Test that connector doesn't leak memory with repeated use."""
        import gc
        import sys
        
        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and destroy many connectors
        for _ in range(100):
            connector = GenesisConnector(self.mock_config)
            del connector
        
        # Force garbage collection again
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not have significantly more objects
        object_increase = final_objects - initial_objects
        self.assertLess(object_increase, 50)  # Allow some increase but not linear


class TestGenesisConnectorErrorRecovery(unittest.TestCase):
    """Error recovery and resilience tests for GenesisConnector."""
    
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
    def test_partial_network_failure_recovery(self, mock_request):
        """Test recovery from partial network failures."""
        # Simulate intermittent network issues
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        mock_request.side_effect = [
            ConnectionError("Network unreachable"),
            ConnectionError("Connection reset"),
            mock_response  # Finally succeeds
        ]
        
        result = self.connector.make_request('GET', '/test')
        
        self.assertEqual(result, {'success': True})
        self.assertEqual(mock_request.call_count, 3)
    
    @patch('requests.Session.request')
    def test_dns_resolution_failure_recovery(self, mock_request):
        """Test recovery from DNS resolution failures."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        mock_request.side_effect = [
            ConnectionError("Name resolution failed"),
            mock_response
        ]
        
        result = self.connector.make_request('GET', '/test')
        
        self.assertEqual(result, {'success': True})
        self.assertEqual(mock_request.call_count, 2)
    
    @patch('requests.Session.request')
    def test_ssl_certificate_error_handling(self, mock_request):
        """Test handling of SSL certificate errors."""
        mock_request.side_effect = requests.exceptions.SSLError("SSL certificate verify failed")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('SSL certificate verify failed', str(context.exception))
    
    @patch('requests.Session.request')
    def test_proxy_error_handling(self, mock_request):
        """Test handling of proxy-related errors."""
        mock_request.side_effect = requests.exceptions.ProxyError("Proxy connection failed")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Proxy connection failed', str(context.exception))
    
    @patch('requests.Session.request')
    def test_chunked_encoding_error_handling(self, mock_request):
        """Test handling of chunked encoding errors."""
        mock_request.side_effect = requests.exceptions.ChunkedEncodingError("Chunked encoding error")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Chunked encoding error', str(context.exception))
    
    @patch('requests.Session.request')
    def test_content_decode_error_handling(self, mock_request):
        """Test handling of content decoding errors."""
        mock_request.side_effect = requests.exceptions.ContentDecodingError("Content decoding error")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Content decoding error', str(context.exception))


class TestGenesisConnectorConfigurationValidation(unittest.TestCase):
    """Configuration validation tests for GenesisConnector."""
    
    def test_configuration_with_boolean_values(self):
        """Test configuration with boolean values."""
        config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3,
            'verify_ssl': True,
            'use_compression': False
        }
        
        connector = GenesisConnector(config)
        # Should handle boolean values gracefully
        self.assertEqual(connector.api_key, 'test_api_key_123')
    
    def test_configuration_type_coercion(self):
        """Test automatic type coercion in configuration."""
        config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': '30',  # String that should be converted to int
            'max_retries': '3'  # String that should be converted to int
        }
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.timeout, 30)
        self.assertEqual(connector.max_retries, 3)
    
    def test_configuration_with_environment_variables(self):
        """Test configuration using environment variables."""
        import os
        
        # Mock environment variables
        with patch.dict(os.environ, {
            'GENESIS_API_KEY': 'env_api_key',
            'GENESIS_BASE_URL': 'https://env.genesis.test',
            'GENESIS_TIMEOUT': '60'
        }):
            config = {
                'api_key': os.getenv('GENESIS_API_KEY'),
                'base_url': os.getenv('GENESIS_BASE_URL'),
                'timeout': int(os.getenv('GENESIS_TIMEOUT', '30')),
                'max_retries': 3
            }
            
            connector = GenesisConnector(config)
            self.assertEqual(connector.api_key, 'env_api_key')
            self.assertEqual(connector.base_url, 'https://env.genesis.test')
            self.assertEqual(connector.timeout, 60)
    
    def test_configuration_validation_edge_cases(self):
        """Test configuration validation with edge cases."""
        base_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        
        # Test with whitespace in API key
        config = base_config.copy()
        config['api_key'] = '  test_api_key_123  '
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test_api_key_123')  # Should be stripped
        
        # Test with trailing slash in URL
        config = base_config.copy()
        config['base_url'] = 'https://api.genesis.test/'
        connector = GenesisConnector(config)
        self.assertEqual(connector.base_url, 'https://api.genesis.test')  # Should be normalized


class TestGenesisConnectorLogging(unittest.TestCase):
    """Logging behavior tests for GenesisConnector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    @patch('logging.getLogger')
    @patch('requests.Session.request')
    def test_request_logging(self, mock_request, mock_get_logger):
        """Test that requests are properly logged."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        self.connector.make_request('GET', '/test')
        
        # Should log request details (but not sensitive data)
        mock_logger.debug.assert_called()
    
    @patch('logging.getLogger')
    @patch('requests.Session.request')
    def test_error_logging(self, mock_request, mock_get_logger):
        """Test that errors are properly logged."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        mock_request.side_effect = ConnectionError("Connection failed")
        
        with self.assertRaises(GenesisConnectionError):
            self.connector.make_request('GET', '/test')
        
        # Should log error details
        mock_logger.error.assert_called()
    
    @patch('logging.getLogger')
    @patch('requests.Session.request')
    def test_retry_logging(self, mock_request, mock_get_logger):
        """Test that retries are properly logged."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        mock_request.side_effect = [
            ConnectionError("Connection failed"),
            mock_response
        ]
        
        self.connector.make_request('GET', '/test')
        
        # Should log retry attempts
        mock_logger.warning.assert_called()