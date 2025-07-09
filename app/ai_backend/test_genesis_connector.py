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
    
# Additional comprehensive test cases to enhance coverage

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
    
    def test_initialization_with_non_string_api_key(self):
        """Test initialization with non-string API key."""
        config = self.mock_config.copy()
        config['api_key'] = 12345
        
        with self.assertRaises(TypeError) as context:
            GenesisConnector(config)
        self.assertIn('API key must be a string', str(context.exception))
    
    def test_initialization_with_whitespace_only_api_key(self):
        """Test initialization with whitespace-only API key."""
        config = self.mock_config.copy()
        config['api_key'] = '   '
        
        with self.assertRaises(ValueError) as context:
            GenesisConnector(config)
        self.assertIn('API key cannot be empty', str(context.exception))
    
    def test_initialization_with_very_long_api_key(self):
        """Test initialization with extremely long API key."""
        config = self.mock_config.copy()
        config['api_key'] = 'a' * 10000
        
        connector = GenesisConnector(config)
        self.assertEqual(len(connector.api_key), 10000)
    
    def test_initialization_with_special_characters_in_api_key(self):
        """Test initialization with special characters in API key."""
        config = self.mock_config.copy()
        config['api_key'] = 'test-key_123!@#$%^&*()'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test-key_123!@#$%^&*()')
    
    def test_initialization_with_unicode_in_api_key(self):
        """Test initialization with Unicode characters in API key."""
        config = self.mock_config.copy()
        config['api_key'] = 'test_key_🔑_123'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test_key_🔑_123')
    
    def test_initialization_with_extremely_small_timeout(self):
        """Test initialization with very small timeout."""
        config = self.mock_config.copy()
        config['timeout'] = 0.001
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.timeout, 0.001)
    
    def test_initialization_with_extremely_large_timeout(self):
        """Test initialization with very large timeout."""
        config = self.mock_config.copy()
        config['timeout'] = 86400  # 24 hours
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.timeout, 86400)
    
    def test_initialization_with_float_timeout(self):
        """Test initialization with float timeout."""
        config = self.mock_config.copy()
        config['timeout'] = 30.5
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.timeout, 30.5)
    
    def test_initialization_with_very_large_max_retries(self):
        """Test initialization with very large max_retries."""
        config = self.mock_config.copy()
        config['max_retries'] = 1000
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.max_retries, 1000)
    
    def test_initialization_with_non_integer_max_retries(self):
        """Test initialization with non-integer max_retries."""
        config = self.mock_config.copy()
        config['max_retries'] = 3.5
        
        with self.assertRaises(TypeError) as context:
            GenesisConnector(config)
        self.assertIn('Max retries must be an integer', str(context.exception))
    
    def test_initialization_with_url_with_port(self):
        """Test initialization with URL containing port."""
        config = self.mock_config.copy()
        config['base_url'] = 'https://api.genesis.test:8080'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.base_url, 'https://api.genesis.test:8080')
    
    def test_initialization_with_url_with_path(self):
        """Test initialization with URL containing path."""
        config = self.mock_config.copy()
        config['base_url'] = 'https://api.genesis.test/v1'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.base_url, 'https://api.genesis.test/v1')
    
    def test_initialization_with_http_url(self):
        """Test initialization with HTTP (non-HTTPS) URL."""
        config = self.mock_config.copy()
        config['base_url'] = 'http://api.genesis.test'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.base_url, 'http://api.genesis.test')
    
    def test_initialization_with_localhost_url(self):
        """Test initialization with localhost URL."""
        config = self.mock_config.copy()
        config['base_url'] = 'http://localhost:8000'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.base_url, 'http://localhost:8000')
    
    def test_initialization_with_ip_address_url(self):
        """Test initialization with IP address URL."""
        config = self.mock_config.copy()
        config['base_url'] = 'https://192.168.1.100:8080'
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.base_url, 'https://192.168.1.100:8080')
    
    def test_initialization_with_missing_optional_fields(self):
        """Test initialization with only required fields."""
        minimal_config = {
            'api_key': 'test_key',
            'base_url': 'https://api.genesis.test'
        }
        
        connector = GenesisConnector(minimal_config)
        self.assertEqual(connector.api_key, 'test_key')
        self.assertEqual(connector.base_url, 'https://api.genesis.test')
        # Should use defaults for optional fields
        self.assertEqual(connector.timeout, 30)  # Assuming default
        self.assertEqual(connector.max_retries, 3)  # Assuming default
    
    def test_initialization_with_extra_config_fields(self):
        """Test initialization with extra configuration fields."""
        config = self.mock_config.copy()
        config['extra_field'] = 'extra_value'
        config['another_field'] = 42
        
        connector = GenesisConnector(config)
        self.assertEqual(connector.api_key, 'test_api_key_123')
        # Extra fields should be ignored


class TestGenesisConnectorAdvancedRequests(unittest.TestCase):
    """Advanced request handling tests."""
    
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
    def test_make_request_with_put_method(self, mock_request):
        """Test API request with PUT method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'updated': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('PUT', '/update/123', data={'name': 'updated'})
        
        self.assertEqual(result, {'updated': True})
        mock_request.assert_called_once_with(
            'PUT',
            'https://api.genesis.test/update/123',
            headers={'Authorization': 'Bearer test_api_key_123'},
            json={'name': 'updated'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_patch_method(self, mock_request):
        """Test API request with PATCH method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'patched': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('PATCH', '/patch/123', data={'status': 'active'})
        
        self.assertEqual(result, {'patched': True})
        mock_request.assert_called_once_with(
            'PATCH',
            'https://api.genesis.test/patch/123',
            headers={'Authorization': 'Bearer test_api_key_123'},
            json={'status': 'active'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_head_method(self, mock_request):
        """Test API request with HEAD method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('HEAD', '/exists/123')
        
        self.assertEqual(result, {})
        mock_request.assert_called_once_with(
            'HEAD',
            'https://api.genesis.test/exists/123',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_options_method(self, mock_request):
        """Test API request with OPTIONS method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'allowed_methods': ['GET', 'POST']}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('OPTIONS', '/endpoint')
        
        self.assertEqual(result, {'allowed_methods': ['GET', 'POST']})
        mock_request.assert_called_once_with(
            'OPTIONS',
            'https://api.genesis.test/endpoint',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_query_parameters(self, mock_request):
        """Test API request with query parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'filtered': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        params = {'page': 1, 'limit': 10, 'sort': 'name'}
        result = self.connector.make_request('GET', '/items', params=params)
        
        self.assertEqual(result, {'filtered': True})
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/items',
            headers={'Authorization': 'Bearer test_api_key_123'},
            params=params,
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_large_payload(self, mock_request):
        """Test API request with large JSON payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'processed': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        large_data = {'items': [{'id': i, 'data': 'x' * 1000} for i in range(1000)]}
        result = self.connector.make_request('POST', '/process', data=large_data)
        
        self.assertEqual(result, {'processed': True})
        mock_request.assert_called_once_with(
            'POST',
            'https://api.genesis.test/process',
            headers={'Authorization': 'Bearer test_api_key_123'},
            json=large_data,
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_nested_json_data(self, mock_request):
        """Test API request with complex nested JSON data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'created': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        complex_data = {
            'user': {
                'name': 'John Doe',
                'preferences': {
                    'theme': 'dark',
                    'notifications': {
                        'email': True,
                        'push': False,
                        'schedule': ['09:00', '17:00']
                    }
                }
            },
            'metadata': {
                'created_at': '2023-01-01T00:00:00Z',
                'tags': ['important', 'user-generated']
            }
        }
        
        result = self.connector.make_request('POST', '/users', data=complex_data)
        
        self.assertEqual(result, {'created': True})
        mock_request.assert_called_once_with(
            'POST',
            'https://api.genesis.test/users',
            headers={'Authorization': 'Bearer test_api_key_123'},
            json=complex_data,
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_special_characters_in_url(self, mock_request):
        """Test API request with special characters in URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'found': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('GET', '/search/user name@domain.com')
        
        self.assertEqual(result, {'found': True})
        mock_request.assert_called_once_with(
            'GET',
            'https://api.genesis.test/search/user name@domain.com',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )
    
    @patch('requests.Session.request')
    def test_make_request_with_empty_response(self, mock_request):
        """Test API request with empty JSON response."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = self.connector.make_request('DELETE', '/items/123')
        
        self.assertEqual(result, {})
        mock_request.assert_called_once_with(
            'DELETE',
            'https://api.genesis.test/items/123',
            headers={'Authorization': 'Bearer test_api_key_123'},
            timeout=30
        )


class TestGenesisConnectorErrorHandling(unittest.TestCase):
    """Advanced error handling tests."""
    
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
    def test_make_request_with_401_unauthorized(self, mock_request):
        """Test handling of 401 Unauthorized error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
        mock_response.text = "Invalid API key"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/protected')
        
        self.assertIn('401 Unauthorized', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_403_forbidden(self, mock_request):
        """Test handling of 403 Forbidden error."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        mock_response.text = "Access denied to resource"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/admin')
        
        self.assertIn('403 Forbidden', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_429_rate_limit(self, mock_request):
        """Test handling of 429 Rate Limited error."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        mock_response.text = "Rate limit exceeded"
        mock_response.headers = {'Retry-After': '60'}
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/api/endpoint')
        
        self.assertIn('429 Too Many Requests', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_500_internal_server_error(self, mock_request):
        """Test handling of 500 Internal Server Error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Internal Server Error")
        mock_response.text = "Internal server error occurred"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/api/endpoint')
        
        self.assertIn('500 Internal Server Error', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_502_bad_gateway(self, mock_request):
        """Test handling of 502 Bad Gateway error."""
        mock_response = Mock()
        mock_response.status_code = 502
        mock_response.raise_for_status.side_effect = requests.HTTPError("502 Bad Gateway")
        mock_response.text = "Bad gateway error"
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/api/endpoint')
        
        self.assertIn('502 Bad Gateway', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_503_service_unavailable(self, mock_request):
        """Test handling of 503 Service Unavailable error."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")
        mock_response.text = "Service temporarily unavailable"
        mock_response.headers = {'Retry-After': '120'}
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/api/endpoint')
        
        self.assertIn('503 Service Unavailable', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_ssl_error(self, mock_request):
        """Test handling of SSL certificate errors."""
        import ssl
        mock_request.side_effect = ssl.SSLError("Certificate verification failed")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/api/endpoint')
        
        self.assertIn('Certificate verification failed', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_dns_resolution_error(self, mock_request):
        """Test handling of DNS resolution errors."""
        import socket
        mock_request.side_effect = socket.gaierror("Name or service not known")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/api/endpoint')
        
        self.assertIn('Name or service not known', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_malformed_json_response(self, mock_request):
        """Test handling of malformed JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "response", 0)
        mock_response.text = "This is not valid JSON {"
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/api/endpoint')
        
        self.assertIn('Invalid JSON', str(context.exception))
    
    @patch('requests.Session.request')
    def test_make_request_with_unexpected_content_type(self, mock_request):
        """Test handling of unexpected content type."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "response", 0)
        mock_response.text = "<html><body>This is HTML, not JSON</body></html>"
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('GET', '/api/endpoint')
        
        self.assertIn('Invalid JSON', str(context.exception))


class TestGenesisConnectorConcurrency(unittest.TestCase):
    """Test concurrent usage of GenesisConnector."""
    
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
    def test_concurrent_requests_thread_safety(self, mock_request):
        """Test thread safety of concurrent requests."""
        import threading
        import time
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        results = []
        exceptions = []
        
        def make_request_thread(endpoint):
            try:
                result = self.connector.make_request('GET', f'/test/{endpoint}')
                results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request_thread, args=(f'endpoint_{i}',))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 10)
        self.assertEqual(len(exceptions), 0)
        self.assertEqual(mock_request.call_count, 10)
    
    @patch('requests.Session.request')
    def test_concurrent_requests_with_different_methods(self, mock_request):
        """Test concurrent requests with different HTTP methods."""
        import threading
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        results = []
        
        def make_get_request():
            result = self.connector.make_request('GET', '/test')
            results.append(('GET', result))
        
        def make_post_request():
            result = self.connector.make_request('POST', '/test', data={'test': 'data'})
            results.append(('POST', result))
        
        def make_put_request():
            result = self.connector.make_request('PUT', '/test', data={'test': 'data'})
            results.append(('PUT', result))
        
        # Create threads for different methods
        threads = [
            threading.Thread(target=make_get_request),
            threading.Thread(target=make_post_request),
            threading.Thread(target=make_put_request)
        ]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 3)
        methods = [result[0] for result in results]
        self.assertIn('GET', methods)
        self.assertIn('POST', methods)
        self.assertIn('PUT', methods)


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
    def test_request_performance_under_load(self, mock_request):
        """Test performance of requests under load."""
        import time
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Measure time for multiple requests
        start_time = time.time()
        for i in range(100):
            self.connector.make_request('GET', f'/test/{i}')
        end_time = time.time()
        
        # Performance should be reasonable (less than 1 second for 100 mocked requests)
        self.assertLess(end_time - start_time, 1.0)
        self.assertEqual(mock_request.call_count, 100)
    
    @patch('requests.Session.request')
    def test_memory_usage_with_large_responses(self, mock_request):
        """Test memory usage with large JSON responses."""
        import sys
        
        # Create a large response
        large_response_data = {
            'items': [{'id': i, 'data': 'x' * 1000} for i in range(1000)]
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = large_response_data
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Make request and verify it completes without issues
        result = self.connector.make_request('GET', '/large-data')
        
        self.assertEqual(len(result['items']), 1000)
        self.assertEqual(result['items'][0]['data'], 'x' * 1000)


class TestGenesisConnectorSecurity(unittest.TestCase):
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
        """Test that API key is not exposed in string representation."""
        repr_str = repr(self.connector)
        self.assertNotIn('test_api_key_123', repr_str)
        self.assertIn('***', repr_str)  # Should be masked
    
    def test_api_key_not_logged_in_str(self):
        """Test that API key is not exposed in string conversion."""
        str_repr = str(self.connector)
        self.assertNotIn('test_api_key_123', str_repr)
        self.assertIn('***', str_repr)  # Should be masked
    
    @patch('requests.Session.request')
    def test_sensitive_data_not_logged_on_error(self, mock_request):
        """Test that sensitive data is not logged in error messages."""
        mock_request.side_effect = ConnectionError("Connection failed")
        
        with self.assertRaises(GenesisConnectionError) as context:
            self.connector.make_request('POST', '/test', data={'password': 'secret123'})
        
        error_message = str(context.exception)
        self.assertNotIn('secret123', error_message)
        self.assertNotIn('test_api_key_123', error_message)
    
    @patch('requests.Session.request')
    def test_headers_contain_authorization(self, mock_request):
        """Test that authorization header is properly set."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        self.connector.make_request('GET', '/test')
        
        # Verify authorization header is set
        call_args = mock_request.call_args
        headers = call_args[1]['headers']
        self.assertIn('Authorization', headers)
        self.assertEqual(headers['Authorization'], 'Bearer test_api_key_123')
    
    @patch('requests.Session.request')
    def test_request_uses_https_by_default(self, mock_request):
        """Test that requests use HTTPS by default."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        self.connector.make_request('GET', '/test')
        
        # Verify HTTPS is used
        call_args = mock_request.call_args
        url = call_args[0][1]
        self.assertTrue(url.startswith('https://'))


class TestGenesisConnectorValidation(unittest.TestCase):
    """Input validation tests for GenesisConnector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_make_request_with_invalid_method(self):
        """Test make_request with invalid HTTP method."""
        with self.assertRaises(ValueError) as context:
            self.connector.make_request('INVALID', '/test')
        
        self.assertIn('Invalid HTTP method', str(context.exception))
    
    def test_make_request_with_none_endpoint(self):
        """Test make_request with None endpoint."""
        with self.assertRaises(ValueError) as context:
            self.connector.make_request('GET', None)
        
        self.assertIn('Endpoint cannot be None', str(context.exception))
    
    def test_make_request_with_empty_endpoint(self):
        """Test make_request with empty endpoint."""
        with self.assertRaises(ValueError) as context:
            self.connector.make_request('GET', '')
        
        self.assertIn('Endpoint cannot be empty', str(context.exception))
    
    def test_make_request_with_invalid_data_type(self):
        """Test make_request with invalid data type."""
        with self.assertRaises(TypeError) as context:
            self.connector.make_request('POST', '/test', data='invalid_data')
        
        self.assertIn('Data must be a dictionary', str(context.exception))
    
    def test_make_request_with_invalid_headers_type(self):
        """Test make_request with invalid headers type."""
        with self.assertRaises(TypeError) as context:
            self.connector.make_request('GET', '/test', headers='invalid_headers')
        
        self.assertIn('Headers must be a dictionary', str(context.exception))
    
    def test_make_request_with_invalid_params_type(self):
        """Test make_request with invalid params type."""
        with self.assertRaises(TypeError) as context:
            self.connector.make_request('GET', '/test', params='invalid_params')
        
        self.assertIn('Params must be a dictionary', str(context.exception))
    
    def test_get_model_info_with_invalid_model_id(self):
        """Test get_model_info with invalid model ID."""
        with self.assertRaises(ValueError) as context:
            self.connector.get_model_info('')
        
        self.assertIn('Model ID cannot be empty', str(context.exception))
    
    def test_get_model_info_with_none_model_id(self):
        """Test get_model_info with None model ID."""
        with self.assertRaises(ValueError) as context:
            self.connector.get_model_info(None)
        
        self.assertIn('Model ID cannot be None', str(context.exception))
    
    def test_create_generation_with_invalid_request(self):
        """Test create_generation with invalid request."""
        with self.assertRaises(TypeError) as context:
            self.connector.create_generation('invalid_request')
        
        self.assertIn('Generation request must be a dictionary', str(context.exception))
    
    def test_create_generation_with_none_request(self):
        """Test create_generation with None request."""
        with self.assertRaises(ValueError) as context:
            self.connector.create_generation(None)
        
        self.assertIn('Generation request cannot be None', str(context.exception))
    
    def test_get_generation_status_with_invalid_generation_id(self):
        """Test get_generation_status with invalid generation ID."""
        with self.assertRaises(ValueError) as context:
            self.connector.get_generation_status('')
        
        self.assertIn('Generation ID cannot be empty', str(context.exception))
    
    def test_cancel_generation_with_invalid_generation_id(self):
        """Test cancel_generation with invalid generation ID."""
        with self.assertRaises(ValueError) as context:
            self.connector.cancel_generation(None)
        
        self.assertIn('Generation ID cannot be None', str(context.exception))


if __name__ == '__main__':
    # Run the additional tests
    unittest.main(verbosity=2)