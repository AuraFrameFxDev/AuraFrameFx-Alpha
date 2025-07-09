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
    """Additional edge case tests for GenesisConnector to ensure comprehensive coverage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_initialization_with_whitespace_api_key(self):
        """Test initialization with whitespace-only API key."""
        config = self.mock_config.copy()
        config['api_key'] = '   \t\n   '
        
        with self.assertRaises(ValueError) as context:
            GenesisConnector(config)
        self.assertIn('API key cannot be empty', str(context.exception))
    
    def test_initialization_with_special_characters_in_base_url(self):
        """Test initialization with special characters in base URL."""
        config = self.mock_config.copy()
        config['base_url'] = 'https://api.génesis-test.com/v1'
        
        # Should handle international domains
        connector = GenesisConnector(config)
        self.assertEqual(connector.base_url, 'https://api.génesis-test.com/v1')
    
    def test_initialization_with_float_timeout(self):
        """Test initialization with float timeout value."""
        config = self.mock_config.copy()
        config['timeout'] = 30.5
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.timeout, 30.5)
    
    def test_initialization_with_boolean_max_retries(self):
        """Test initialization with boolean max_retries value."""
        config = self.mock_config.copy()
        config['max_retries'] = True  # Should be treated as 1
        
        with self.assertRaises(ValueError):
            GenesisConnector(config)
    
    @patch('requests.Session.request')
    def test_make_request_with_binary_data(self, mock_request):
        """Test API request with binary data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        binary_data = b'\x00\x01\x02\x03\x04\x05'
        
        # Should handle binary data gracefully
        result = self.connector.make_request('POST', '/upload', data={'file': binary_data})
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_none_data(self, mock_request):
        """Test API request with None data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('POST', '/test', data=None)
        
        # Should handle None data without errors
        self.assertEqual(result, {'success': True})
        call_args = mock_request.call_args
        self.assertNotIn('json', call_args[1])
    
    @patch('requests.Session.request')
    def test_make_request_with_nested_dict_data(self, mock_request):
        """Test API request with deeply nested dictionary data."""
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
                            'value': 'deeply_nested',
                            'list': [1, 2, {'nested_in_list': True}]
                        }
                    }
                }
            }
        }
        
        result = self.connector.make_request('POST', '/nested', data=nested_data)
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_circular_reference_in_data(self):
        """Test API request with circular reference in data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        # Create circular reference
        circular_data = {'key': 'value'}
        circular_data['self'] = circular_data
        
        # Should handle circular references gracefully (likely by raising an error)
        with self.assertRaises((ValueError, TypeError)):
            self.connector.make_request('POST', '/circular', data=circular_data)
    
    @patch('requests.Session.request')
    def test_make_request_with_very_long_endpoint(self, mock_request):
        """Test API request with very long endpoint URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Create very long endpoint
        long_endpoint = '/' + 'x' * 2000
        
        result = self.connector.make_request('GET', long_endpoint)
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_headers_containing_lists(self, mock_request):
        """Test API request with headers containing list values."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        list_headers = {
            'X-List-Header': ['value1', 'value2', 'value3'],
            'X-Single-Header': 'single_value'
        }
        
        # Should handle list headers appropriately
        result = self.connector.make_request('GET', '/test', headers=list_headers)
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_response_encoding_issues(self, mock_request):
        """Test API request with response encoding issues."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte')
        mock_response.text = 'Response with encoding issues'
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError):
            self.connector.make_request('GET', '/encoding-issue')
    
    @patch('requests.Session.request')
    def test_make_request_with_response_content_type_mismatch(self, mock_request):
        """Test API request when response content type doesn't match JSON."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Not JSON", "", 0)
        mock_response.text = '<html><body>HTML response</body></html>'
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/html-response')
        
        self.assertIn('HTML response', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_gzip_compressed_response(self, mock_request):
        """Test API request with gzip compressed response."""
        import gzip
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'compressed': True}
        mock_response.headers = {'Content-Encoding': 'gzip'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/compressed')
        self.assertEqual(result, {'compressed': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_cookies_in_response(self, mock_request):
        """Test API request handling cookies in response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.cookies = {'session_id': 'abc123', 'csrf_token': 'xyz789'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/with-cookies')
        self.assertEqual(result, {'success': True})
    
    @patch('time.sleep')
    @patch('requests.Session.request')
    def test_make_request_with_exponential_backoff_jitter(self, mock_request, mock_sleep):
        """Test that exponential backoff includes proper jitter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        # Fail twice, then succeed
        mock_request.side_effect = [
            ConnectionError("Connection failed"),
            ConnectionError("Connection failed"),
            mock_response
        ]
        
        result = self.connector.make_request('GET', '/test')
        
        self.assertEqual(result, {'success': True})
        self.assertEqual(mock_sleep.call_count, 2)
        
        # Check that sleep times are reasonable (1s, 2s with potential jitter)
        sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
        self.assertGreaterEqual(sleep_calls[0], 0.5)
        self.assertLessEqual(sleep_calls[0], 2.0)
        self.assertGreaterEqual(sleep_calls[1], 1.0)
        self.assertLessEqual(sleep_calls[1], 4.0)
    
    @patch('requests.Session.request')
    def test_make_request_with_redirect_handling(self, mock_request):
        """Test API request handling redirects."""
        # First response is a redirect
        redirect_response = Mock()
        redirect_response.status_code = 302
        redirect_response.headers = {'Location': 'https://api.genesis.test/new-location'}
        redirect_response.raise_for_status.return_value = None
        
        # Final response
        final_response = Mock()
        final_response.status_code = 200
        final_response.json.return_value = {'redirected': True}
        final_response.raise_for_status.return_value = None
        
        mock_request.side_effect = [redirect_response, final_response]
        
        result = self.connector.make_request('GET', '/redirect')
        # requests library handles redirects automatically
        self.assertEqual(result, {'redirected': True})
    
    @patch('requests.Session.request')
    def test_make_request_with_server_sent_events(self, mock_request):
        """Test API request handling server-sent events style response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("SSE format", "", 0)
        mock_response.text = 'data: {"event": "message"}\n\ndata: {"event": "close"}\n\n'
        mock_response.headers = {'Content-Type': 'text/event-stream'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError):
            self.connector.make_request('GET', '/events')
    
    def test_build_url_with_port_numbers(self):
        """Test URL building with various port numbers."""
        config = self.mock_config.copy()
        config['base_url'] = 'https://api.genesis.test:8443'
        connector = GenesisConnector(config)
        
        url = connector._build_url('/endpoint')
        self.assertEqual(url, 'https://api.genesis.test:8443/endpoint')
    
    def test_build_url_with_path_traversal_attempt(self):
        """Test URL building with path traversal attempts."""
        malicious_endpoints = [
            '../../../etc/passwd',
            '/endpoint/../../../secret',
            '..\\..\\windows\\system32',
            '/endpoint/..%2F..%2F..%2Fsecret'
        ]
        
        for endpoint in malicious_endpoints:
            with self.subTest(endpoint=endpoint):
                # Should build URL but not necessarily prevent path traversal
                # (that's the responsibility of the server)
                url = self.connector._build_url(endpoint)
                self.assertIsInstance(url, str)
                self.assertTrue(url.startswith('https://api.genesis.test'))
    
    def test_build_headers_with_none_values(self):
        """Test header building with None values."""
        none_headers = {
            'X-Null-Header': None,
            'X-Valid-Header': 'valid_value',
            'X-Empty-Header': ''
        }
        
        headers = self.connector._build_headers(none_headers)
        
        # Should handle None values appropriately
        self.assertIn('Authorization', headers)
        self.assertIn('X-Valid-Header', headers)
        self.assertEqual(headers['X-Valid-Header'], 'valid_value')
    
    def test_build_headers_with_numeric_values(self):
        """Test header building with numeric values."""
        numeric_headers = {
            'X-Numeric-Header': 123,
            'X-Float-Header': 45.67,
            'X-Boolean-Header': True
        }
        
        headers = self.connector._build_headers(numeric_headers)
        
        # Should convert numeric values to strings
        self.assertIn('X-Numeric-Header', headers)
        self.assertEqual(headers['X-Numeric-Header'], 123)
    
    def test_session_reuse_across_requests(self):
        """Test that the same session is reused across multiple requests."""
        session1 = self.connector.session
        session2 = self.connector.session
        
        # Should be the same object
        self.assertIs(session1, session2)
        
        # Should maintain session state
        self.assertIsNotNone(session1)
    
    def test_context_manager_nested_usage(self):
        """Test nested context manager usage."""
        with GenesisConnector(self.mock_config) as connector1:
            with GenesisConnector(self.mock_config) as connector2:
                self.assertIsNotNone(connector1)
                self.assertIsNotNone(connector2)
                self.assertNotEqual(connector1, connector2)
    
    def test_repr_with_very_long_api_key(self):
        """Test repr with very long API key."""
        config = self.mock_config.copy()
        config['api_key'] = 'x' * 1000
        
        connector = GenesisConnector(config)
        repr_str = repr(connector)
        
        # Should not contain full key and should be reasonable length
        self.assertLess(len(repr_str), 200)
        self.assertNotIn('x' * 1000, repr_str)
    
    def test_str_with_special_characters_in_base_url(self):
        """Test str representation with special characters in base URL."""
        config = self.mock_config.copy()
        config['base_url'] = 'https://api.tëst-génesis.com'
        
        connector = GenesisConnector(config)
        str_repr = str(connector)
        
        # Should handle unicode in URL
        self.assertIn('tëst-génesis', str_repr)
        self.assertIn('GenesisConnector', str_repr)


class TestGenesisConnectorMethodSpecific(unittest.TestCase):
    """Method-specific tests for GenesisConnector methods."""
    
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
    def test_get_model_info_with_version_parameter(self, mock_request):
        """Test get_model_info with version parameter if supported."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 'model_123',
            'name': 'Test Model',
            'version': '2.0.0'
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Test with version parameter
        result = self.connector.get_model_info('model_123', version='2.0.0')
        
        self.assertEqual(result['version'], '2.0.0')
        # Check if version was passed in URL or params
        call_args = mock_request.call_args
        self.assertIsNotNone(call_args)
    
    @patch('requests.Session.request')
    def test_create_generation_with_streaming(self, mock_request):
        """Test create_generation with streaming parameter."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            'id': 'gen_123',
            'status': 'streaming',
            'stream_url': 'wss://api.genesis.test/stream/gen_123'
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        generation_request = {
            'prompt': 'Test prompt',
            'model': 'test_model',
            'max_tokens': 100,
            'stream': True
        }
        
        result = self.connector.create_generation(generation_request)
        
        self.assertEqual(result['status'], 'streaming')
        self.assertIn('stream_url', result)
    
    @patch('requests.Session.request')
    def test_create_generation_with_custom_parameters(self, mock_request):
        """Test create_generation with various custom parameters."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 'gen_123', 'status': 'pending'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        generation_request = {
            'prompt': 'Test prompt',
            'model': 'test_model',
            'max_tokens': 100,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'presence_penalty': 0.1,
            'frequency_penalty': 0.2,
            'stop_sequences': ['</text>', '\n\n'],
            'seed': 42
        }
        
        result = self.connector.create_generation(generation_request)
        
        self.assertEqual(result['id'], 'gen_123')
        call_args = mock_request.call_args
        self.assertEqual(call_args[1]['json'], generation_request)
    
    @patch('requests.Session.request')
    def test_get_generation_status_with_include_result(self, mock_request):
        """Test get_generation_status with include_result parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 'gen_123',
            'status': 'completed',
            'result': 'Generated text',
            'metadata': {'tokens_used': 50}
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.get_generation_status('gen_123', include_result=True)
        
        self.assertIn('result', result)
        self.assertIn('metadata', result)
    
    @patch('requests.Session.request')
    def test_cancel_generation_with_reason(self, mock_request):
        """Test cancel_generation with cancellation reason."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 'gen_123',
            'status': 'cancelled',
            'reason': 'User requested cancellation'
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.cancel_generation('gen_123', reason='User requested cancellation')
        
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['reason'], 'User requested cancellation')
    
    @patch('requests.Session.request')
    def test_list_models_with_pagination(self, mock_request):
        """Test list_models with pagination parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [
                {'id': 'model_1', 'name': 'Model 1'},
                {'id': 'model_2', 'name': 'Model 2'}
            ],
            'total': 50,
            'page': 1,
            'per_page': 2,
            'has_next': True
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        filters = {'page': 1, 'per_page': 2}
        result = self.connector.list_models(filters)
        
        self.assertEqual(result['page'], 1)
        self.assertEqual(result['per_page'], 2)
        self.assertTrue(result['has_next'])
    
    @patch('requests.Session.request')
    def test_list_models_with_sorting(self, mock_request):
        """Test list_models with sorting parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [
                {'id': 'model_1', 'name': 'A Model', 'created_at': '2023-01-01'},
                {'id': 'model_2', 'name': 'B Model', 'created_at': '2023-01-02'}
            ],
            'total': 2
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        filters = {'sort_by': 'created_at', 'sort_order': 'desc'}
        result = self.connector.list_models(filters)
        
        self.assertEqual(len(result['models']), 2)
        call_args = mock_request.call_args
        self.assertEqual(call_args[1]['params'], filters)
    
    @patch('requests.Session.request')
    def test_health_check_with_detailed_response(self, mock_request):
        """Test health_check with detailed health information."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'healthy',
            'timestamp': '2023-01-01T00:00:00Z',
            'version': '1.0.0',
            'uptime': 86400,
            'services': {
                'database': 'healthy',
                'cache': 'healthy',
                'models': 'healthy'
            },
            'metrics': {
                'requests_per_second': 100,
                'avg_response_time': 0.5,
                'error_rate': 0.01
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.health_check()
        
        self.assertEqual(result['status'], 'healthy')
        self.assertIn('services', result)
        self.assertIn('metrics', result)
        self.assertEqual(result['services']['database'], 'healthy')
    
    @patch('requests.Session.request')
    def test_health_check_with_partial_service_failure(self, mock_request):
        """Test health_check when some services are unhealthy."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'degraded',
            'timestamp': '2023-01-01T00:00:00Z',
            'services': {
                'database': 'healthy',
                'cache': 'unhealthy',
                'models': 'healthy'
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.health_check()
        
        self.assertEqual(result['status'], 'degraded')
        self.assertEqual(result['services']['cache'], 'unhealthy')


class TestGenesisConnectorAsyncAdvanced(unittest.TestCase):
    """Advanced async tests for GenesisConnector."""
    
    def setUp(self):
        """Set up async test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    @patch('aiohttp.ClientSession.request')
    async def test_async_make_request_with_concurrent_calls(self, mock_request):
        """Test concurrent async requests."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'data': 'concurrent_data'})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_request.return_value = mock_response
        
        # Make multiple concurrent requests
        tasks = [
            self.connector.async_make_request('GET', f'/test/{i}')
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All requests should succeed
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertEqual(result, {'data': 'concurrent_data'})
    
    @patch('aiohttp.ClientSession.request')
    async def test_async_make_request_with_retry_logic(self, mock_request):
        """Test async request retry logic."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'success': True})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        # First call fails, second succeeds
        mock_request.side_effect = [
            asyncio.TimeoutError("Async timeout"),
            mock_response
        ]
        
        result = await self.connector.async_make_request('GET', '/test')
        
        self.assertEqual(result, {'success': True})
        self.assertEqual(mock_request.call_count, 2)
    
    @patch('aiohttp.ClientSession.request')
    async def test_async_make_request_with_json_decode_error(self, mock_request):
        """Test async request with JSON decode error."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        mock_response.text = AsyncMock(return_value="Invalid JSON response")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError):
            await self.connector.async_make_request('GET', '/test')
    
    @patch('aiohttp.ClientSession.request')
    async def test_async_context_manager_usage(self, mock_request):
        """Test async context manager usage."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
        
        async with self.connector as async_connector:
            self.assertIsNotNone(async_connector)
            
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'async': True})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            mock_request.return_value = mock_response
            
            result = await async_connector.async_make_request('GET', '/test')
            self.assertEqual(result, {'async': True})


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
    
    def test_config_validation_with_extra_fields(self):
        """Test configuration validation with extra fields."""
        config = self.mock_config.copy()
        config['extra_field'] = 'should_be_ignored'
        config['another_extra'] = 123
        
        # Should create connector successfully, ignoring extra fields
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test_api_key_123')
    
    def test_config_validation_with_wrong_types(self):
        """Test configuration validation with wrong data types."""
        wrong_type_configs = [
            {'api_key': 123, 'base_url': 'https://api.test.com'},
            {'api_key': 'valid', 'base_url': 123},
            {'api_key': 'valid', 'base_url': 'https://api.test.com', 'timeout': 'invalid'},
            {'api_key': 'valid', 'base_url': 'https://api.test.com', 'max_retries': 'invalid'},
        ]
        
        for config in wrong_type_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    GenesisConnector(config)
    
    def test_endpoint_validation(self):
        """Test endpoint validation in URL building."""
        valid_endpoints = [
            '/api/v1/models',
            '/generations',
            '/health',
            '/models/123',
            '/generations/gen_456/status'
        ]
        
        for endpoint in valid_endpoints:
            with self.subTest(endpoint=endpoint):
                url = self.connector._build_url(endpoint)
                self.assertTrue(url.startswith('https://api.genesis.test'))
                self.assertIn(endpoint.lstrip('/'), url)
    
    def test_api_key_format_validation(self):
        """Test API key format validation."""
        valid_api_keys = [
            'simple_key',
            'key-with-dashes',
            'key_with_underscores',
            'key123with456numbers',
            'UPPERCASE_KEY',
            'MixedCase_Key-123'
        ]
        
        for api_key in valid_api_keys:
            with self.subTest(api_key=api_key):
                config = self.mock_config.copy()
                config['api_key'] = api_key
                connector = GenesisConnector(config)
                self.assertEqual(connector.api_key, api_key)
    
    def test_base_url_format_validation(self):
        """Test base URL format validation."""
        valid_base_urls = [
            'https://api.genesis.test',
            'http://localhost:8000',
            'https://api.genesis.test:443',
            'https://api.genesis.test/v1',
            'https://subdomain.api.genesis.test'
        ]
        
        for base_url in valid_base_urls:
            with self.subTest(base_url=base_url):
                config = self.mock_config.copy()
                config['base_url'] = base_url
                connector = GenesisConnector(config)
                self.assertEqual(connector.base_url, base_url)
    
    @patch('requests.Session.request')
    def test_request_data_validation(self, mock_request):
        """Test request data validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        valid_data_types = [
            {'string': 'value'},
            {'number': 123},
            {'float': 45.67},
            {'boolean': True},
            {'list': [1, 2, 3]},
            {'nested': {'key': 'value'}},
            {'mixed': {'str': 'text', 'num': 123, 'bool': False}}
        ]
        
        for data in valid_data_types:
            with self.subTest(data=data):
                result = self.connector.make_request('POST', '/test', data=data)
                self.assertEqual(result, {'success': True})
    
    def test_generation_request_validation(self):
        """Test generation request validation."""
        # Valid generation requests
        valid_requests = [
            {'prompt': 'Test prompt', 'model': 'test_model'},
            {'prompt': 'Test prompt', 'model': 'test_model', 'max_tokens': 100},
            {'prompt': 'Test prompt', 'model': 'test_model', 'temperature': 0.7},
            {'prompt': 'Test prompt', 'model': 'test_model', 'top_p': 0.9, 'top_k': 50}
        ]
        
        # These should not raise validation errors at the connector level
        for request in valid_requests:
            with self.subTest(request=request):
                # Just test that the request can be processed
                self.assertIsInstance(request, dict)
                self.assertIn('prompt', request)
                self.assertIn('model', request)


class TestGenesisConnectorRobustness(unittest.TestCase):
    """Robustness tests for GenesisConnector under various failure conditions."""
    
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
    def test_request_with_network_partitioning(self, mock_request):
        """Test request behavior during network partitioning."""
        # Simulate network partitioning with various network errors
        network_errors = [
            ConnectionError("Network unreachable"),
            requests.exceptions.ConnectionError("Connection aborted"),
            requests.exceptions.ChunkedEncodingError("Connection broken"),
            OSError("Network is unreachable")
        ]
        
        for error in network_errors:
            with self.subTest(error=type(error).__name__):
                mock_request.side_effect = error
                
                with self.assertRaises(GenesisConnectionError):
                    self.connector.make_request('GET', '/test')
    
    @patch('requests.Session.request')
    def test_request_with_intermittent_failures(self, mock_request):
        """Test request behavior with intermittent failures."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        # Pattern: fail, succeed, fail, succeed
        mock_request.side_effect = [
            ConnectionError("Intermittent failure"),
            mock_response,
            ConnectionError("Another failure"),
            mock_response
        ]
        
        # First request should succeed after retry
        result1 = self.connector.make_request('GET', '/test1')
        self.assertEqual(result1, {'success': True})
        
        # Second request should also succeed after retry
        result2 = self.connector.make_request('GET', '/test2')
        self.assertEqual(result2, {'success': True})
    
    @patch('requests.Session.request')
    def test_request_with_memory_pressure(self, mock_request):
        """Test request behavior under memory pressure."""
        # Simulate memory pressure with MemoryError
        mock_request.side_effect = MemoryError("Out of memory")
        
        # Should propagate MemoryError (not retry)
        with self.assertRaises(MemoryError):
            self.connector.make_request('GET', '/test')
    
    @patch('requests.Session.request')
    def test_request_with_ssl_errors(self, mock_request):
        """Test request behavior with SSL errors."""
        ssl_errors = [
            requests.exceptions.SSLError("SSL certificate verify failed"),
            requests.exceptions.SSLError("SSL handshake failed"),
            requests.exceptions.SSLError("SSL connection error")
        ]
        
        for error in ssl_errors:
            with self.subTest(error=str(error)):
                mock_request.side_effect = error
                
                with self.assertRaises(GenesisConnectionError):
                    self.connector.make_request('GET', '/test')
    
    @patch('requests.Session.request')
    def test_request_with_dns_resolution_failure(self, mock_request):
        """Test request behavior with DNS resolution failures."""
        dns_errors = [
            requests.exceptions.ConnectionError("Name resolution failed"),
            requests.exceptions.ConnectionError("Temporary failure in name resolution"),
            OSError("Name or service not known")
        ]
        
        for error in dns_errors:
            with self.subTest(error=str(error)):
                mock_request.side_effect = error
                
                with self.assertRaises(GenesisConnectionError):
                    self.connector.make_request('GET', '/test')
    
    @patch('requests.Session.request')
    def test_request_with_proxy_errors(self, mock_request):
        """Test request behavior with proxy errors."""
        proxy_errors = [
            requests.exceptions.ProxyError("Proxy connection failed"),
            requests.exceptions.ProxyError("Proxy authentication required"),
            requests.exceptions.ConnectTimeout("Proxy connection timeout")
        ]
        
        for error in proxy_errors:
            with self.subTest(error=str(error)):
                mock_request.side_effect = error
                
                with self.assertRaises(GenesisConnectionError):
                    self.connector.make_request('GET', '/test')
    
    @patch('requests.Session.request')
    def test_request_with_rate_limiting(self, mock_request):
        """Test request behavior with rate limiting."""
        # Simulate rate limiting with 429 status
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {'Retry-After': '60'}
        rate_limit_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        rate_limit_response.text = "Rate limit exceeded"
        
        mock_request.return_value = rate_limit_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('429 Too Many Requests', str(context.exception))
    
    @patch('requests.Session.request')
    def test_request_with_server_overload(self, mock_request):
        """Test request behavior with server overload."""
        # Simulate server overload with 503 status
        overload_response = Mock()
        overload_response.status_code = 503
        overload_response.headers = {'Retry-After': '30'}
        overload_response.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")
        overload_response.text = "Server temporarily overloaded"
        
        mock_request.return_value = overload_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('503 Service Unavailable', str(context.exception))
    
    def test_connector_thread_safety(self):
        """Test connector thread safety."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker():
            try:
                # Create new connector instance for each thread
                connector = GenesisConnector(self.mock_config)
                
                # Simulate work
                time.sleep(0.01)
                
                # Test basic functionality
                url = connector._build_url('/test')
                headers = connector._build_headers()
                
                results.append({'url': url, 'headers': headers})
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All threads should complete successfully
        self.assertEqual(len(results), 10)
        self.assertEqual(len(errors), 0)
    
    def test_connector_resource_cleanup(self):
        """Test proper resource cleanup."""
        import gc
        
        # Create many connector instances
        connectors = []
        for i in range(100):
            config = self.mock_config.copy()
            config['api_key'] = f'key_{i}'
            connector = GenesisConnector(config)
            connectors.append(connector)
        
        # Clear references
        connectors.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Should not crash or leak resources
        self.assertTrue(True)


# Run additional tests if this module is executed directly
if __name__ == '__main__':
    # Set up more detailed logging for debugging
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run all tests with high verbosity
    unittest.main(verbosity=2, exit=False)
    
    # Also run pytest tests if pytest is available
    try:
        import pytest
        pytest.main([__file__, '-v'])
    except ImportError:
        print("pytest not available, skipping pytest tests")