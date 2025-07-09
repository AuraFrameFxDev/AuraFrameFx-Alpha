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
    unittest.main(verbosity=2)