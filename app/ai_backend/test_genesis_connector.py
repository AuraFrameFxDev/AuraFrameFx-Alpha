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
    

class TestGenesisConnectorEnhanced(unittest.TestCase):
    """Enhanced unit tests for GenesisConnector with additional edge cases and scenarios."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
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
        self.assertIn('Timeout cannot be None', str(context.exception))
    
    def test_initialization_with_string_timeout(self):
        """Test initialization with string timeout value."""
        config = self.mock_config.copy()
        config['timeout'] = '30'
        
        with self.assertRaises(ValueError) as context:
            GenesisConnector(config)
        self.assertIn('Timeout must be a number', str(context.exception))
    
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
    
    def test_initialization_with_special_characters_in_api_key(self):
        """Test initialization with special characters in API key."""
        config = self.mock_config.copy()
        config['api_key'] = 'test_key_with_!@#$%^&*()_+-=[]{}|;:,.<>?'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test_key_with_!@#$%^&*()_+-=[]{}|;:,.<>?')
    
    def test_initialization_with_unicode_api_key(self):
        """Test initialization with Unicode characters in API key."""
        config = self.mock_config.copy()
        config['api_key'] = 'test_key_with_üñíçødé_characters'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test_key_with_üñíçødé_characters')
    
    def test_initialization_with_whitespace_api_key(self):
        """Test initialization with whitespace in API key."""
        config = self.mock_config.copy()
        config['api_key'] = '  test_api_key_with_spaces  '
        
        connector = GenesisConnector(config)
        # Should preserve whitespace as API keys might validly contain it
        self.assertEqual(connector.api_key, '  test_api_key_with_spaces  ')
    
    def test_initialization_with_url_containing_path(self):
        """Test initialization with base URL containing path."""
        config = self.mock_config.copy()
        config['base_url'] = 'https://api.genesis.test/v1/api'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.base_url, 'https://api.genesis.test/v1/api')
    
    def test_initialization_with_url_containing_query_params(self):
        """Test initialization with base URL containing query parameters."""
        config = self.mock_config.copy()
        config['base_url'] = 'https://api.genesis.test?version=v1'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.base_url, 'https://api.genesis.test?version=v1')
    
    def test_initialization_with_url_containing_fragment(self):
        """Test initialization with base URL containing fragment."""
        config = self.mock_config.copy()
        config['base_url'] = 'https://api.genesis.test#section'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.base_url, 'https://api.genesis.test#section')
    
    def test_initialization_with_localhost_url(self):
        """Test initialization with localhost URL."""
        config = self.mock_config.copy()
        config['base_url'] = 'http://localhost:8080'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.base_url, 'http://localhost:8080')
    
    def test_initialization_with_ip_address_url(self):
        """Test initialization with IP address URL."""
        config = self.mock_config.copy()
        config['base_url'] = 'https://192.168.1.100:443'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.base_url, 'https://192.168.1.100:443')
    
    def test_initialization_with_additional_config_fields(self):
        """Test initialization with additional configuration fields."""
        config = self.mock_config.copy()
        config['extra_field'] = 'extra_value'
        config['debug'] = True
        
        # Should not raise error - extra fields should be ignored
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test_api_key_123')
    
    @patch('requests.Session.request')
    def test_make_request_with_none_endpoint(self, mock_request):
        """Test make_request with None endpoint."""
        with self.assertRaises(ValueError) as context:
            self.connector.make_request('GET', None)
        self.assertIn('Endpoint cannot be None', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_empty_endpoint(self, mock_request):
        """Test make_request with empty endpoint."""
        with self.assertRaises(ValueError) as context:
            self.connector.make_request('GET', '')
        self.assertIn('Endpoint cannot be empty', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_invalid_method(self, mock_request):
        """Test make_request with invalid HTTP method."""
        with self.assertRaises(ValueError) as context:
            self.connector.make_request('INVALID', '/test')
        self.assertIn('Invalid HTTP method', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_none_method(self, mock_request):
        """Test make_request with None HTTP method."""
        with self.assertRaises(ValueError) as context:
            self.connector.make_request(None, '/test')
        self.assertIn('HTTP method cannot be None', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_large_data_payload(self, mock_request):
        """Test make_request with large data payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Create a large payload
        large_data = {'data': 'x' * 10000}  # 10KB of data
        
        result = self.connector.make_request('POST', '/test', data=large_data)
        
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once_with(
            'POST',
            'https://api.genesis.test/test',
            headers={'Authorization': 'Bearer test_api_key_123'},
            json=large_data,
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_nested_json_data(self, mock_request):
        """Test make_request with deeply nested JSON data."""
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
                            'value': 'deep_value',
                            'array': [1, 2, 3, {'nested': True}]
                        }
                    }
                }
            }
        }
        
        result = self.connector.make_request('POST', '/test', data=nested_data)
        
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once_with(
            'POST',
            'https://api.genesis.test/test',
            headers={'Authorization': 'Bearer test_api_key_123'},
            json=nested_data,
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_unicode_data(self, mock_request):
        """Test make_request with Unicode data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        unicode_data = {
            'message': 'Hello 世界! 🌍',
            'emoji': '😀🎉🚀',
            'accents': 'café, naïve, résumé'
        }
        
        result = self.connector.make_request('POST', '/test', data=unicode_data)
        
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once_with(
            'POST',
            'https://api.genesis.test/test',
            headers={'Authorization': 'Bearer test_api_key_123'},
            json=unicode_data,
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_special_characters_in_endpoint(self, mock_request):
        """Test make_request with special characters in endpoint."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        endpoint = '/test/path with spaces & special chars!@#$%'
        
        result = self.connector.make_request('GET', endpoint)
        
        self.assertEqual(result, {'success': True})
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/test/path with spaces & special chars!@#$%',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_response_with_empty_body(self, mock_request):
        """Test make_request with empty response body."""
        mock_response = Mock()
        mock_response.status_code = 204  # No Content
        mock_response.json.side_effect = json.JSONDecodeError("Empty response", "", 0)
        mock_response.text = ""
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('DELETE', '/test')
        
        self.assertEqual(result, {})  # Should return empty dict for empty response
    
    @patch('requests.Session.request')
    def test_make_request_response_with_non_json_content(self, mock_request):
        """Test make_request with non-JSON response content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Not JSON", "", 0)
        mock_response.text = "Plain text response"
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Plain text response', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_different_status_codes(self, mock_request):
        """Test make_request with various successful status codes."""
        status_codes = [200, 201, 202, 204, 301, 302]
        
        for status_code in status_codes:
            with self.subTest(status_code=status_code):
                mock_response = Mock()
                mock_response.status_code = status_code
                if status_code == 204:
                    mock_response.json.side_effect = json.JSONDecodeError("Empty", "", 0)
                    mock_response.text = ""
                else:
                    mock_response.json.return_value = {'status': 'success'}
                mock_response.raise_for_status.return_value = None
                mock_request.return_value = mock_response
                
                result = self.connector.make_request('GET', '/test')
                
                if status_code == 204:
                    self.assertEqual(result, {})
                else:
                    self.assertEqual(result, {'status': 'success'})
    
    @patch('requests.Session.request')
    def test_make_request_with_various_error_status_codes(self, mock_request):
        """Test make_request with various error status codes."""
        error_codes = [400, 401, 403, 404, 405, 408, 409, 422, 429, 500, 502, 503, 504]
        
        for error_code in error_codes:
            with self.subTest(error_code=error_code):
                mock_response = Mock()
                mock_response.status_code = error_code
                mock_response.raise_for_status.side_effect = requests.HTTPError(f"{error_code} Error")
                mock_response.text = f"Error {error_code}"
                mock_request.return_value = mock_response
                
                with self.assertRaises(GenesisConnectionError) as context:
                    self.connector.make_request('GET', '/test')
                
                self.assertIn(f'{error_code} Error', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_request_exception(self, mock_request):
        """Test make_request with generic RequestException."""
        mock_request.side_effect = RequestException("Generic request error")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/test')
        
        self.assertIn('Generic request error', str(context.exception))
    
    @patch('time.sleep')
    @patch('requests.Session.request')
    def test_make_request_retry_with_different_exceptions(self, mock_request, mock_sleep):
        """Test retry logic with different exception types."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        
        # Test with different exception types
        mock_request.side_effect = [
            ConnectionError("Connection failed"),
            Timeout("Request timed out"),
            RequestException("Generic error"),
            mock_response
        ]
        
        result = self.connector.make_request('GET', '/test')
        
        self.assertEqual(result, {'success': True})
        self.assertEqual(mock_request.call_count, 4)  # 3 retries + 1 success
        self.assertEqual(mock_sleep.call_count, 3)
    
    @patch('requests.Session.request')
    def test_make_request_with_max_retries_zero(self, mock_request):
        """Test make_request with max_retries set to zero."""
        config = self.mock_config.copy()
        config['max_retries'] = 0
        connector = GenesisConnector(config)
        
        mock_request.side_effect = ConnectionError("Connection failed")
        
        with self.assertRaises(GenesisConnectionError):
            connector.make_request('GET', '/test')
        
        # Should only try once (no retries)
        self.assertEqual(mock_request.call_count, 1)
    
    @patch('requests.Session.request')
    def test_make_request_with_max_retries_one(self, mock_request):
        """Test make_request with max_retries set to one."""
        config = self.mock_config.copy()
        config['max_retries'] = 1
        connector = GenesisConnector(config)
        
        mock_request.side_effect = [
            ConnectionError("Connection failed"),
            ConnectionError("Connection failed again")
        ]
        
        with self.assertRaises(GenesisConnectionError):
            connector.make_request('GET', '/test')
        
        # Should try twice (1 initial + 1 retry)
        self.assertEqual(mock_request.call_count, 2)
    
    def test_get_model_info_with_none_model_id(self):
        """Test get_model_info with None model ID."""
        with self.assertRaises(ValueError) as context:
            self.connector.get_model_info(None)
        self.assertIn('Model ID cannot be None', str(context.exception))
    
    def test_get_model_info_with_empty_model_id(self):
        """Test get_model_info with empty model ID."""
        with self.assertRaises(ValueError) as context:
            self.connector.get_model_info('')
        self.assertIn('Model ID cannot be empty', str(context.exception))
    
    def test_get_model_info_with_whitespace_model_id(self):
        """Test get_model_info with whitespace-only model ID."""
        with self.assertRaises(ValueError) as context:
            self.connector.get_model_info('   ')
        self.assertIn('Model ID cannot be empty', str(context.exception))
    
    @patch('requests.Session.request')
    def test_get_model_info_with_special_characters_in_id(self, mock_request):
        """Test get_model_info with special characters in model ID."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'id': 'model_with_!@#$%', 'name': 'Special Model'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.get_model_info('model_with_!@#$%')
        
        self.assertEqual(result['id'], 'model_with_!@#$%')
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/models/model_with_!@#$%',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    def test_create_generation_with_none_request(self):
        """Test create_generation with None request."""
        with self.assertRaises(ValueError) as context:
            self.connector.create_generation(None)
        self.assertIn('Generation request cannot be None', str(context.exception))
    
    def test_create_generation_with_empty_request(self):
        """Test create_generation with empty request."""
        with self.assertRaises(ValueError) as context:
            self.connector.create_generation({})
        self.assertIn('Generation request cannot be empty', str(context.exception))
    
    @patch('requests.Session.request')
    def test_create_generation_with_minimal_request(self, mock_request):
        """Test create_generation with minimal valid request."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 'gen_123', 'status': 'pending'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        minimal_request = {'prompt': 'Test'}
        result = self.connector.create_generation(minimal_request)
        
        self.assertEqual(result['id'], 'gen_123')
        mock_request.assert_called_once_with(
            'POST',
            'https://api.genesis.test/generations',
            headers={'Authorization': 'Bearer test_api_key_123'},
            json=minimal_request,
            timeout=30
        )
    
    def test_get_generation_status_with_none_id(self):
        """Test get_generation_status with None generation ID."""
        with self.assertRaises(ValueError) as context:
            self.connector.get_generation_status(None)
        self.assertIn('Generation ID cannot be None', str(context.exception))
    
    def test_get_generation_status_with_empty_id(self):
        """Test get_generation_status with empty generation ID."""
        with self.assertRaises(ValueError) as context:
            self.connector.get_generation_status('')
        self.assertIn('Generation ID cannot be empty', str(context.exception))
    
    def test_cancel_generation_with_none_id(self):
        """Test cancel_generation with None generation ID."""
        with self.assertRaises(ValueError) as context:
            self.connector.cancel_generation(None)
        self.assertIn('Generation ID cannot be None', str(context.exception))
    
    def test_cancel_generation_with_empty_id(self):
        """Test cancel_generation with empty generation ID."""
        with self.assertRaises(ValueError) as context:
            self.connector.cancel_generation('')
        self.assertIn('Generation ID cannot be empty', str(context.exception))
    
    @patch('requests.Session.request')
    def test_list_models_with_none_filters(self, mock_request):
        """Test list_models with None filters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'models': [], 'total': 0}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.list_models(None)
        
        self.assertEqual(result['models'], [])
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/models',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_list_models_with_empty_filters(self, mock_request):
        """Test list_models with empty filters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'models': [], 'total': 0}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.list_models({})
        
        self.assertEqual(result['models'], [])
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/models',
            headers={'Authorization': 'Bearer test_api_key_123'},
            params={},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_list_models_with_complex_filters(self, mock_request):
        """Test list_models with complex filter parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'models': [{'id': 'model_1'}], 'total': 1}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        complex_filters = {
            'status': ['active', 'training'],
            'category': 'text',
            'min_size': 1000,
            'max_size': 10000,
            'tags': ['nlp', 'gpt'],
            'sort': 'created_at',
            'order': 'desc',
            'limit': 50,
            'offset': 100
        }
        
        result = self.connector.list_models(complex_filters)
        
        self.assertEqual(len(result['models']), 1)
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/models',
            headers={'Authorization': 'Bearer test_api_key_123'},
            params=complex_filters,
            timeout=30
        )
    
    def test_build_url_with_multiple_slashes(self):
        """Test URL building with multiple slashes."""
        url = self.connector._build_url('//test//endpoint//')
        self.assertEqual(url, 'https://api.genesis.test//test//endpoint//')
    
    def test_build_url_with_query_parameters(self):
        """Test URL building with query parameters in endpoint."""
        url = self.connector._build_url('/test?param1=value1&param2=value2')
        self.assertEqual(url, 'https://api.genesis.test/test?param1=value1&param2=value2')
    
    def test_build_url_with_fragment(self):
        """Test URL building with fragment in endpoint."""
        url = self.connector._build_url('/test#section')
        self.assertEqual(url, 'https://api.genesis.test/test#section')
    
    def test_build_headers_with_none_custom_headers(self):
        """Test header building with None custom headers."""
        headers = self.connector._build_headers(None)
        expected = {'Authorization': 'Bearer test_api_key_123'}
        self.assertEqual(headers, expected)
    
    def test_build_headers_with_empty_custom_headers(self):
        """Test header building with empty custom headers."""
        headers = self.connector._build_headers({})
        expected = {'Authorization': 'Bearer test_api_key_123'}
        self.assertEqual(headers, expected)
    
    def test_build_headers_case_insensitive_auth_override(self):
        """Test header building with case-insensitive auth override."""
        custom_headers = {'authorization': 'Bearer different_token'}
        headers = self.connector._build_headers(custom_headers)
        
        # Should preserve the case but override the auth
        expected = {'authorization': 'Bearer different_token'}
        self.assertEqual(headers, expected)
    
    def test_build_headers_with_complex_custom_headers(self):
        """Test header building with complex custom headers."""
        custom_headers = {
            'X-Request-ID': 'req-123',
            'X-Client-Version': '1.0.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': 'GenesisConnector/1.0',
            'X-Forwarded-For': '192.168.1.1',
            'X-Real-IP': '10.0.0.1'
        }
        headers = self.connector._build_headers(custom_headers)
        
        expected = {
            'Authorization': 'Bearer test_api_key_123',
            'X-Request-ID': 'req-123',
            'X-Client-Version': '1.0.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': 'GenesisConnector/1.0',
            'X-Forwarded-For': '192.168.1.1',
            'X-Real-IP': '10.0.0.1'
        }
        self.assertEqual(headers, expected)
    
    def test_context_manager_with_exception(self):
        """Test context manager behavior when exception occurs."""
        try:
            with GenesisConnector(self.mock_config) as connector:
                self.assertIsNotNone(connector)
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
    
    def test_context_manager_close_method(self):
        """Test context manager calls close method."""
        connector = GenesisConnector(self.mock_config)
        
        with patch.object(connector, 'close') as mock_close:
            with connector:
                pass
            mock_close.assert_called_once()
    
    def test_repr_with_different_config(self):
        """Test repr with different configuration."""
        config = {
            'api_key': 'very_long_api_key_that_should_be_truncated_in_repr',
            'base_url': 'https://very.long.domain.name.for.testing.com:8443/api/v1',
            'timeout': 60,
            'max_retries': 5
        }
        connector = GenesisConnector(config)
        
        repr_str = repr(connector)
        self.assertIn('GenesisConnector', repr_str)
        self.assertIn('https://very.long.domain.name.for.testing.com:8443/api/v1', repr_str)
        self.assertNotIn('very_long_api_key_that_should_be_truncated_in_repr', repr_str)
    
    def test_str_with_different_config(self):
        """Test str with different configuration."""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://custom.api.com',
            'timeout': 45,
            'max_retries': 2
        }
        connector = GenesisConnector(config)
        
        str_repr = str(connector)
        self.assertIn('GenesisConnector', str_repr)
        self.assertIn('https://custom.api.com', str_repr)
        self.assertNotIn('test_key', str_repr)
    
    def test_equality_comparison(self):
        """Test equality comparison between connectors."""
        config1 = self.mock_config.copy()
        config2 = self.mock_config.copy()
        
        connector1 = GenesisConnector(config1)
        connector2 = GenesisConnector(config2)
        connector3 = GenesisConnector({**config1, 'timeout': 60})
        
        # Should be equal if same config
        self.assertEqual(connector1, connector2)
        
        # Should not be equal if different config
        self.assertNotEqual(connector1, connector3)
        
        # Should not be equal to different types
        self.assertNotEqual(connector1, "not a connector")
        self.assertNotEqual(connector1, None)
    
    def test_hash_consistency(self):
        """Test hash consistency for connector objects."""
        config = self.mock_config.copy()
        connector1 = GenesisConnector(config)
        connector2 = GenesisConnector(config)
        
        # Should have same hash if same config
        self.assertEqual(hash(connector1), hash(connector2))
        
        # Should be usable in sets and dicts
        connector_set = {connector1, connector2}
        self.assertEqual(len(connector_set), 1)
        
        connector_dict = {connector1: 'value1', connector2: 'value2'}
        self.assertEqual(len(connector_dict), 1)
        self.assertEqual(connector_dict[connector1], 'value2')


class TestGenesisConnectorThreadSafety(unittest.TestCase):
    """Test thread safety aspects of GenesisConnector."""
    
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
    def test_concurrent_requests(self, mock_request):
        """Test concurrent requests from multiple threads."""
        import threading
        import time
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        results = []
        errors = []
        
        def make_request(endpoint):
            try:
                result = self.connector.make_request('GET', f'/test_{endpoint}')
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(results), 10)
        self.assertEqual(len(errors), 0)
        self.assertEqual(mock_request.call_count, 10)
    
    @patch('requests.Session.request')
    def test_concurrent_different_methods(self, mock_request):
        """Test concurrent requests with different HTTP methods."""
        import threading
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        results = []
        errors = []
        
        def make_request(method, endpoint):
            try:
                result = self.connector.make_request(method, endpoint)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create threads with different methods
        methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        threads = []
        
        for i, method in enumerate(methods):
            thread = threading.Thread(target=make_request, args=(method, f'/test_{i}'))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)
        self.assertEqual(mock_request.call_count, 5)


class TestGenesisConnectorPerformance(unittest.TestCase):
    """Performance-related tests for GenesisConnector."""
    
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
    def test_request_performance_timing(self, mock_request):
        """Test request performance timing."""
        import time
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Time the request
        start_time = time.time()
        result = self.connector.make_request('GET', '/test')
        end_time = time.time()
        
        # Check that the request completes quickly (under 1 second)
        self.assertLess(end_time - start_time, 1.0)
        self.assertEqual(result, {'success': True})
    
    @patch('requests.Session.request')
    def test_multiple_requests_performance(self, mock_request):
        """Test performance of multiple sequential requests."""
        import time
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Time multiple requests
        start_time = time.time()
        for i in range(100):
            result = self.connector.make_request('GET', f'/test_{i}')
        end_time = time.time()
        
        # Check that 100 requests complete reasonably quickly
        self.assertLess(end_time - start_time, 5.0)
        self.assertEqual(mock_request.call_count, 100)
    
    def test_memory_usage_with_large_objects(self):
        """Test memory usage with large configuration objects."""
        import sys
        
        # Create a large configuration
        large_config = self.mock_config.copy()
        large_config['large_field'] = 'x' * 1000000  # 1MB string
        
        connector = GenesisConnector(large_config)
        
        # Check that the connector doesn't excessively increase memory
        # This is a basic check - in a real scenario, you'd use memory profiling tools
        self.assertIsInstance(connector, GenesisConnector)
    
    def test_object_creation_performance(self):
        """Test performance of creating multiple connector instances."""
        import time
        
        start_time = time.time()
        
        connectors = []
        for i in range(1000):
            config = self.mock_config.copy()
            config['api_key'] = f'test_key_{i}'
            connector = GenesisConnector(config)
            connectors.append(connector)
        
        end_time = time.time()
        
        # Check that creating 1000 connectors is reasonably fast
        self.assertLess(end_time - start_time, 5.0)
        self.assertEqual(len(connectors), 1000)


class TestGenesisConnectorSecurityAspects(unittest.TestCase):
    """Security-related tests for GenesisConnector."""
    
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
        """Test that API key is not exposed in repr."""
        repr_str = repr(self.connector)
        self.assertNotIn('test_api_key_123', repr_str)
        self.assertIn('***', repr_str)  # Should show masked key
    
    def test_api_key_not_logged_in_str(self):
        """Test that API key is not exposed in str."""
        str_repr = str(self.connector)
        self.assertNotIn('test_api_key_123', str_repr)
    
    def test_api_key_sanitization_in_error_messages(self):
        """Test that API key is sanitized in error messages."""
        config = self.mock_config.copy()
        config['api_key'] = 'sensitive_key_123'
        
        # This should not raise an error that contains the API key
        connector = GenesisConnector(config)
        
        # Check that the API key is properly stored but not exposed
        self.assertEqual(connector.api_key, 'sensitive_key_123')
    
    @patch('requests.Session.request')
    def test_sensitive_data_not_in_error_messages(self, mock_request):
        """Test that sensitive data is not exposed in error messages."""
        mock_request.side_effect = ConnectionError("Connection failed")
        
        sensitive_data = {'password': 'secret123', 'token': 'sensitive_token'}
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('POST', '/test', data=sensitive_data)
        
        error_message = str(context.exception)
        self.assertNotIn('secret123', error_message)
        self.assertNotIn('sensitive_token', error_message)
    
    def test_url_validation_prevents_injection(self):
        """Test that URL validation prevents injection attacks."""
        malicious_urls = [
            'javascript:alert("xss")',
            'data:text/html,<script>alert("xss")</script>',
            'file:///etc/passwd',
            'ftp://malicious.com/file.txt'
        ]
        
        for malicious_url in malicious_urls:
            with self.subTest(url=malicious_url):
                config = self.mock_config.copy()
                config['base_url'] = malicious_url
                
                with self.assertRaises(ValueError) as context:
                    GenesisConnector(config)
                
                self.assertIn('Invalid base URL', str(context.exception))
    
    def test_header_injection_prevention(self):
        """Test prevention of header injection attacks."""
        malicious_headers = {
            'X-Test': 'value\r\nX-Injected: malicious',
            'X-Another': 'value\nSet-Cookie: session=hijacked'
        }
        
        # Should not raise an exception but should sanitize headers
        headers = self.connector._build_headers(malicious_headers)
        
        # Check that newlines are handled appropriately
        for key, value in headers.items():
            if key != 'Authorization':  # Skip the auth header
                self.assertNotIn('\r', value)
                self.assertNotIn('\n', value)


if __name__ == '__main__':
    # Run only the new test classes
    unittest.main(verbosity=2)


class TestGenesisConnectorAsyncEnhanced(unittest.TestCase):
    """Enhanced async tests for GenesisConnector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    async def test_async_context_manager(self):
        """Test async context manager functionality."""
        if hasattr(self.connector, '__aenter__'):
            async with self.connector as connector:
                self.assertIsNotNone(connector)
                self.assertEqual(connector.api_key, 'test_api_key_123')
    
    @patch('aiohttp.ClientSession.request')
    async def test_async_make_request_with_data(self, mock_request):
        """Test async request with POST data."""
        mock_response = Mock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={'id': 123, 'status': 'created'})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_request.return_value = mock_response
        
        if hasattr(self.connector, 'async_make_request'):
            test_data = {'name': 'test', 'value': 42}
            result = await self.connector.async_make_request('POST', '/create', data=test_data)
            self.assertEqual(result, {'id': 123, 'status': 'created'})
    
    @patch('aiohttp.ClientSession.request')
    async def test_async_make_request_connection_error(self, mock_request):
        """Test async request connection error handling."""
        mock_request.side_effect = ConnectionError("Async connection failed")
        
        if hasattr(self.connector, 'async_make_request'):
            with self.assertRaises(GenesisConnectionError):
                await self.connector.async_make_request('GET', '/test')
    
    @patch('aiohttp.ClientSession.request')
    async def test_async_make_request_json_error(self, mock_request):
        """Test async request JSON decode error."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        mock_response.text = AsyncMock(return_value="Invalid JSON response")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_request.return_value = mock_response
        
        if hasattr(self.connector, 'async_make_request'):
            with self.assertRaises(GenesisConnectionError):
                await self.connector.async_make_request('GET', '/test')
    
    @patch('aiohttp.ClientSession.request')
    async def test_async_retry_logic(self, mock_request):
        """Test async retry logic."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'success': True})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_request.side_effect = [
            ConnectionError("Connection failed"),
            asyncio.TimeoutError("Request timed out"),
            mock_response
        ]
        
        if hasattr(self.connector, 'async_make_request'):
            result = await self.connector.async_make_request('GET', '/test')
            self.assertEqual(result, {'success': True})
    
    async def test_async_concurrent_requests(self):
        """Test concurrent async requests."""
        if hasattr(self.connector, 'async_make_request'):
            with patch('aiohttp.ClientSession.request') as mock_request:
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={'success': True})
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_request.return_value = mock_response
                
                # Create multiple concurrent requests
                tasks = []
                for i in range(5):
                    task = self.connector.async_make_request('GET', f'/test_{i}')
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
                
                self.assertEqual(len(results), 5)
                for result in results:
                    self.assertEqual(result, {'success': True})
    
    async def test_async_timeout_handling(self):
        """Test async timeout handling."""
        if hasattr(self.connector, 'async_make_request'):
            with patch('aiohttp.ClientSession.request') as mock_request:
                mock_request.side_effect = asyncio.TimeoutError("Request timed out")
                
                with self.assertRaises(GenesisTimeoutError):
                    await self.connector.async_make_request('GET', '/test')


# Run async tests if asyncio is available
if __name__ == '__main__':
    import asyncio
    
    def run_async_tests():
        """Run async tests."""
        async def main():
            test_instance = TestGenesisConnectorAsyncEnhanced()
            test_instance.setUp()
            
            # Run each async test
            await test_instance.test_async_context_manager()
            await test_instance.test_async_make_request_with_data()
            await test_instance.test_async_make_request_connection_error()
            await test_instance.test_async_make_request_json_error()
            await test_instance.test_async_retry_logic()
            await test_instance.test_async_concurrent_requests()
            await test_instance.test_async_timeout_handling()
            
            print("All async tests completed successfully!")
        
        asyncio.run(main())
    
    # Uncomment to run async tests
    # run_async_tests()