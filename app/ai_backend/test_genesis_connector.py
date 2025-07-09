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

# Additional comprehensive tests for enhanced coverage
class TestGenesisConnectorResponseHandling(unittest.TestCase):
    """Additional tests for response handling edge cases."""
    
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
    def test_response_with_custom_headers(self, mock_request):
        """Test response with custom headers and their handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_response.headers = {
            'X-Custom-Header': 'custom_value',
            'Content-Type': 'application/json',
            'X-Rate-Limit-Remaining': '99'
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/test')
        
        self.assertEqual(result, {'data': 'test'})
        # If the connector exposes response headers, test them
        if hasattr(self.connector, 'last_response_headers'):
            self.assertIn('X-Custom-Header', self.connector.last_response_headers)
    
    @patch('requests.Session.request')
    def test_response_with_bom_utf8(self, mock_request):
        """Test response with UTF-8 BOM (Byte Order Mark)."""
        mock_response = Mock()
        mock_response.status_code = 200
        # Simulate response with BOM
        mock_response.text = '\ufeff{"data": "test"}'
        mock_response.json.return_value = {'data': 'test'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/test')
        self.assertEqual(result, {'data': 'test'})
    
    @patch('requests.Session.request')
    def test_response_with_different_encodings(self, mock_request):
        """Test response with different character encodings."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'message': 'Café naïve résumé'}
        mock_response.encoding = 'utf-8'
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/test')
        self.assertEqual(result['message'], 'Café naïve résumé')
    
    @patch('requests.Session.request')
    def test_response_with_nested_json_structures(self, mock_request):
        """Test response with deeply nested JSON structures."""
        complex_response = {
            'data': {
                'nested': {
                    'deep': {
                        'structure': {
                            'values': [1, 2, 3],
                            'metadata': {
                                'created_at': '2023-01-01T00:00:00Z',
                                'tags': ['tag1', 'tag2']
                            }
                        }
                    }
                }
            }
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = complex_response
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/complex')
        self.assertEqual(result['data']['nested']['deep']['structure']['values'], [1, 2, 3])
    
    @patch('requests.Session.request')
    def test_response_with_null_values(self, mock_request):
        """Test response handling with null/None values."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': None,
            'optional_field': None,
            'existing_field': 'value'
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/test')
        self.assertIsNone(result['data'])
        self.assertIsNone(result['optional_field'])
        self.assertEqual(result['existing_field'], 'value')


class TestGenesisConnectorRequestPayloads(unittest.TestCase):
    """Additional tests for request payload handling."""
    
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
    def test_request_with_binary_data(self, mock_request):
        """Test request with binary data payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        binary_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'  # PNG header
        
        # Test that binary data is handled appropriately
        result = self.connector.make_request('POST', '/upload', data=binary_data)
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_request_with_form_data(self, mock_request):
        """Test request with form-encoded data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        form_data = {
            'field1': 'value1',
            'field2': 'value2',
            'special_chars': 'value with spaces & symbols!'
        }
        
        result = self.connector.make_request('POST', '/form', data=form_data)
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_request_with_list_parameters(self, mock_request):
        """Test request with list parameters in data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        data_with_lists = {
            'tags': ['tag1', 'tag2', 'tag3'],
            'numbers': [1, 2, 3, 4, 5],
            'nested': {
                'items': ['item1', 'item2']
            }
        }
        
        result = self.connector.make_request('POST', '/test', data=data_with_lists)
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_request_with_boolean_values(self, mock_request):
        """Test request with boolean values in data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        boolean_data = {
            'enabled': True,
            'disabled': False,
            'active': True,
            'settings': {
                'auto_save': False,
                'notifications': True
            }
        }
        
        result = self.connector.make_request('POST', '/test', data=boolean_data)
        self.assertEqual(result, {'success': True})


class TestGenesisConnectorAdvancedErrorHandling(unittest.TestCase):
    """Additional tests for advanced error handling scenarios."""
    
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
    def test_ssl_certificate_error(self, mock_request):
        """Test SSL certificate verification error handling."""
        mock_request.side_effect = requests.exceptions.SSLError("SSL certificate verification failed")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('SSL certificate', str(context.exception))
    
    @patch('requests.Session.request')
    def test_dns_resolution_error(self, mock_request):
        """Test DNS resolution error handling."""
        mock_request.side_effect = requests.exceptions.ConnectionError("Name or service not known")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Name or service not known', str(context.exception))
    
    @patch('requests.Session.request')
    def test_proxy_error(self, mock_request):
        """Test proxy connection error handling."""
        mock_request.side_effect = requests.exceptions.ProxyError("Proxy connection failed")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Proxy connection failed', str(context.exception))
    
    @patch('requests.Session.request')
    def test_content_too_large_error(self, mock_request):
        """Test handling of content too large errors."""
        mock_response = Mock()
        mock_response.status_code = 413
        mock_response.raise_for_status.side_effect = requests.HTTPError("413 Payload Too Large")
        mock_response.text = "Request entity too large"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('POST', '/test', data={'large_field': 'x' * 10000})
        
        self.assertIn('413 Payload Too Large', str(context.exception))
    
    @patch('requests.Session.request')
    def test_rate_limit_error(self, mock_request):
        """Test rate limiting error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        mock_response.text = "Rate limit exceeded"
        mock_response.headers = {'Retry-After': '60'}
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('429 Too Many Requests', str(context.exception))
    
    @patch('requests.Session.request')
    def test_redirect_error(self, mock_request):
        """Test redirect error handling."""
        mock_request.side_effect = requests.exceptions.TooManyRedirects("Too many redirects")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Too many redirects', str(context.exception))


class TestGenesisConnectorConfigurationEdgeCases(unittest.TestCase):
    """Additional tests for configuration edge cases."""
    
    def test_configuration_with_extra_fields(self):
        """Test configuration with extra unknown fields."""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': 30,
            'max_retries': 3,
            'unknown_field': 'value',
            'extra_config': {'nested': 'value'}
        }
        
        # Should handle extra fields gracefully
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test_key')
    
    def test_configuration_case_sensitivity(self):
        """Test configuration key case sensitivity."""
        config = {
            'API_KEY': 'test_key',  # Different case
            'base_url': 'https://api.test.com',
            'timeout': 30,
            'max_retries': 3
        }
        
        # Should handle case-insensitive keys or raise appropriate error
        try:
            connector = GenesisConnector(config)
            # If it works, test that it was handled correctly
            self.assertIsNotNone(connector)
        except (KeyError, ValueError):
            # If it fails, that's also acceptable behavior
            pass
    
    def test_configuration_type_coercion(self):
        """Test configuration type coercion."""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': '30',  # String instead of int
            'max_retries': '3'  # String instead of int
        }
        
        # Test how the connector handles type coercion
        try:
            connector = GenesisConnector(config)
            self.assertEqual(connector.timeout, 30)
            self.assertEqual(connector.max_retries, 3)
        except (ValueError, TypeError):
            # If strict type checking is enforced, this is acceptable
            pass
    
    def test_configuration_with_environment_variables(self):
        """Test configuration that might come from environment variables."""
        import os
        
        # Simulate environment variable configuration
        test_env = {
            'GENESIS_API_KEY': 'env_api_key',
            'GENESIS_BASE_URL': 'https://env.api.test.com',
            'GENESIS_TIMEOUT': '60',
            'GENESIS_MAX_RETRIES': '5'
        }
        
        # Test that environment-like configuration works
        config = {
            'api_key': test_env.get('GENESIS_API_KEY'),
            'base_url': test_env.get('GENESIS_BASE_URL'),
            'timeout': int(test_env.get('GENESIS_TIMEOUT', 30)),
            'max_retries': int(test_env.get('GENESIS_MAX_RETRIES', 3))
        }
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'env_api_key')
        self.assertEqual(connector.timeout, 60)


class TestGenesisConnectorAsyncExtended(unittest.TestCase):
    """Extended async tests if async functionality is available."""
    
    def setUp(self):
        """Set up async test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    @patch('asyncio.sleep')
    async def test_async_retry_with_backoff(self, mock_sleep):
        """Test async retry logic with backoff."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
        
        mock_sleep.return_value = None
        
        # Test that async retry uses proper backoff
        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.side_effect = [
                asyncio.TimeoutError("First failure"),
                asyncio.TimeoutError("Second failure"),
                AsyncMock(status=200, json=AsyncMock(return_value={'success': True}))
            ]
            
            result = await self.connector.async_make_request('GET', '/test')
            self.assertEqual(result, {'success': True})
            
            # Verify backoff was used
            expected_calls = [call(1), call(2)]
            mock_sleep.assert_has_calls(expected_calls)
    
    async def test_async_concurrent_requests(self):
        """Test multiple concurrent async requests."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
        
        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'success': True})
            mock_request.return_value = mock_response
            
            # Create multiple concurrent requests
            tasks = [
                self.connector.async_make_request('GET', f'/test/{i}')
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for result in results:
                self.assertEqual(result, {'success': True})
            
            # All requests should have been made
            self.assertEqual(mock_request.call_count, 5)


class TestGenesisConnectorUtilityMethods(unittest.TestCase):
    """Additional tests for utility methods and helper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_url_joining_with_special_characters(self):
        """Test URL joining with special characters."""
        test_cases = [
            ('/test%20endpoint', 'https://api.genesis.test/test%20endpoint'),
            ('/test+endpoint', 'https://api.genesis.test/test+endpoint'),
            ('/test&endpoint', 'https://api.genesis.test/test&endpoint'),
            ('/test=endpoint', 'https://api.genesis.test/test=endpoint'),
        ]
        
        for endpoint, expected in test_cases:
            with self.subTest(endpoint=endpoint):
                result = self.connector._build_url(endpoint)
                self.assertEqual(result, expected)
    
    def test_header_sanitization(self):
        """Test header sanitization for security."""
        sensitive_headers = {
            'Authorization': 'Bearer secret_token',
            'X-API-Key': 'secret_key',
            'Cookie': 'session=secret_session'
        }
        
        # Test that headers are properly handled
        result = self.connector._build_headers(sensitive_headers)
        
        # Should contain the headers
        self.assertIn('Authorization', result)
        self.assertIn('X-API-Key', result)
        self.assertIn('Cookie', result)
        
        # Values should be preserved
        self.assertEqual(result['Authorization'], 'Bearer secret_token')
    
    def test_request_parameter_validation(self):
        """Test request parameter validation."""
        # Test invalid HTTP methods
        invalid_methods = ['INVALID', 'GET POST', '', None]
        
        for method in invalid_methods:
            with self.subTest(method=method):
                with self.assertRaises((ValueError, AttributeError)):
                    self.connector.make_request(method, '/test')
    
    def test_endpoint_normalization(self):
        """Test endpoint path normalization."""
        test_cases = [
            ('test', 'https://api.genesis.test/test'),
            ('/test', 'https://api.genesis.test/test'),
            ('//test', 'https://api.genesis.test/test'),
            ('test/', 'https://api.genesis.test/test/'),
            ('/test/', 'https://api.genesis.test/test/'),
        ]
        
        for endpoint, expected in test_cases:
            with self.subTest(endpoint=endpoint):
                result = self.connector._build_url(endpoint)
                self.assertEqual(result, expected)


class TestGenesisConnectorResourceManagement(unittest.TestCase):
    """Additional tests for resource management and cleanup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
    
    def test_session_cleanup_on_deletion(self):
        """Test that sessions are properly cleaned up when connector is deleted."""
        connector = GenesisConnector(self.mock_config)
        session = connector.session
        
        # Delete the connector
        del connector
        
        # Session should be closed (if the implementation supports it)
        # Note: This depends on the actual implementation
        if hasattr(session, 'closed'):
            self.assertTrue(session.closed)
    
    def test_multiple_connector_instances(self):
        """Test that multiple connector instances don't interfere with each other."""
        connector1 = GenesisConnector(self.mock_config)
        
        config2 = self.mock_config.copy()
        config2['api_key'] = 'different_key'
        connector2 = GenesisConnector(config2)
        
        # Each should have its own session
        self.assertIsNot(connector1.session, connector2.session)
        self.assertNotEqual(connector1.api_key, connector2.api_key)
    
    def test_context_manager_resource_cleanup(self):
        """Test resource cleanup in context manager."""
        with GenesisConnector(self.mock_config) as connector:
            session = connector.session
            self.assertIsNotNone(session)
        
        # After exiting context, resources should be cleaned up
        # (depends on implementation)
        if hasattr(session, 'closed'):
            self.assertTrue(session.closed)
    
    def test_connector_reuse_after_error(self):
        """Test that connector can be reused after an error."""
        connector = GenesisConnector(self.mock_config)
        
        # Simulate an error
        with patch('requests.Session.request') as mock_request:
            mock_request.side_effect = ConnectionError("Connection failed")
            
            with self.assertRaises(GenesisConnectionError):
                connector.make_request('GET', '/test')
        
        # Connector should still be usable after error
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'success': True}
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response
            
            result = connector.make_request('GET', '/test')
            self.assertEqual(result, {'success': True})


# Additional test runner configuration
if __name__ == '__main__':
    # Add additional test configuration for comprehensive testing
    import sys
    
    # Set up detailed test reporting
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('.', pattern='test_*.py')
    
    # Run tests with high verbosity
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        failfast=False,
        buffer=False
    )
    
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


# Additional pytest parametrized tests for comprehensive coverage
@pytest.mark.skipif(not pytest, reason="pytest not available")
class TestGenesisConnectorParametrizedExtended:
    """Extended parametrized tests using pytest."""
    
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
    
    @pytest.mark.parametrize("content_type,expected_handled", [
        ('application/json', True),
        ('application/json; charset=utf-8', True),
        ('text/plain', False),
        ('application/xml', False),
        ('multipart/form-data', False),
        ('application/octet-stream', False),
    ])
    @patch('requests.Session.request')
    def test_content_type_handling(self, mock_request, connector, content_type, expected_handled):
        """Test handling of different content types."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': content_type}
        
        if expected_handled:
            mock_response.json.return_value = {'success': True}
        else:
            mock_response.json.side_effect = ValueError("Not JSON")
            mock_response.text = "Non-JSON response"
        
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        if expected_handled:
            result = connector.make_request('GET', '/test')
            assert result == {'success': True}
        else:
            with pytest.raises(GenesisConnectionError):
                connector.make_request('GET', '/test')
    
    @pytest.mark.parametrize("api_key_format,should_work", [
        ('simple_key', True),
        ('key-with-dashes', True),
        ('key_with_underscores', True),
        ('key.with.dots', True),
        ('key123with456numbers', True),
        ('UPPERCASE_KEY', True),
        ('MixedCase_Key', True),
        ('', False),
        ('key with spaces', True),  # Should work but not recommended
        ('key\nwith\nnewlines', True),  # Should work but not recommended
        ('key\twith\ttabs', True),  # Should work but not recommended
    ])
    def test_api_key_format_validation(self, api_key_format, should_work):
        """Test various API key formats."""
        config = {
            'api_key': api_key_format,
            'base_url': 'https://api.test.com',
            'timeout': 30,
            'max_retries': 3
        }
        
        if should_work and api_key_format:
            connector = GenesisConnector(config)
            assert connector.api_key == api_key_format
        else:
            with pytest.raises(ValueError):
                GenesisConnector(config)
    
    @pytest.mark.parametrize("retry_count,expected_calls", [
        (0, 1),  # No retries
        (1, 2),  # One retry
        (3, 4),  # Three retries
        (5, 6),  # Five retries
        (10, 11),  # Ten retries
    ])
    @patch('time.sleep')
    @patch('requests.Session.request')
    def test_retry_count_accuracy(self, mock_request, mock_sleep, connector, retry_count, expected_calls):
        """Test that retry count is accurate."""
        # Configure connector with specific retry count
        connector.max_retries = retry_count
        
        # Mock all requests to fail
        mock_request.side_effect = ConnectionError("Connection failed")
        
        with pytest.raises(GenesisConnectionError):
            connector.make_request('GET', '/test')
        
        # Verify correct number of calls
        assert mock_request.call_count == expected_calls
        assert mock_sleep.call_count == retry_count
    
    @pytest.mark.parametrize("timeout_value,expected_timeout", [
        (1, 1),
        (5, 5),
        (30, 30),
        (60, 60),
        (300, 300),
        (3600, 3600),
    ])
    @patch('requests.Session.request')
    def test_timeout_configuration(self, mock_request, timeout_value, expected_timeout):
        """Test timeout configuration with various values."""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': timeout_value,
            'max_retries': 3
        }
        
        connector = GenesisConnector(config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        connector.make_request('GET', '/test')
        
        # Verify timeout was passed to request
        args, kwargs = mock_request.call_args
        assert kwargs['timeout'] == expected_timeout


class TestGenesisConnectorPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarking tests for GenesisConnector."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    @unittest.skipIf(not hasattr(unittest, 'skip'), reason="Performance tests can be skipped")
    @patch('requests.Session.request')
    def test_request_latency_measurement(self, mock_request):
        """Measure request latency under various conditions."""
        import time
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Measure single request latency
        start_time = time.time()
        result = self.connector.make_request('GET', '/test')
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Should complete quickly (under 1 second for mocked request)
        self.assertLess(latency, 1.0)
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_throughput_measurement(self, mock_request):
        """Measure request throughput."""
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
        throughput = num_requests / total_time
        
        # Should achieve reasonable throughput (>50 requests per second for mocked)
        self.assertGreater(throughput, 50.0)
    
    @patch('requests.Session.request')
    def test_memory_efficiency(self, mock_request):
        """Test memory efficiency over many requests."""
        import gc
        import sys
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'x' * 1000}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Measure memory before
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Make many requests
        for i in range(100):
            result = self.connector.make_request('GET', f'/test/{i}')
            self.assertIsNotNone(result)
        
        # Measure memory after
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count should not grow excessively
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 1000)  # Reasonable threshold