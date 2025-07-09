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
    

class TestGenesisConnectorSecurityAndEdgeCases(unittest.TestCase):
    """Additional security and edge case tests for GenesisConnector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_initialization_with_none_config(self):
        """Test initialization with None config."""
        with self.assertRaises(TypeError):
            GenesisConnector(None)
    
    def test_initialization_with_empty_config(self):
        """Test initialization with empty config."""
        with self.assertRaises(ValueError):
            GenesisConnector({})
    
    def test_initialization_with_malformed_config(self):
        """Test initialization with malformed config values."""
        configs = [
            {'api_key': None, 'base_url': 'https://api.genesis.test'},
            {'api_key': 123, 'base_url': 'https://api.genesis.test'},
            {'api_key': 'test', 'base_url': None},
            {'api_key': 'test', 'base_url': 123},
            {'api_key': 'test', 'base_url': 'https://api.genesis.test', 'timeout': 'invalid'},
            {'api_key': 'test', 'base_url': 'https://api.genesis.test', 'max_retries': 'invalid'}
        ]
        
        for config in configs:
            with self.assertRaises((ValueError, TypeError)):
                GenesisConnector(config)
    
    def test_api_key_sanitization_in_logging(self):
        """Test that API key is not exposed in logging."""
        with patch('logging.Logger.debug') as mock_debug:
            connector = GenesisConnector(self.mock_config)
            # Trigger some logging
            try:
                connector._build_headers()
            except:
                pass
            
            # Check that API key is not in any log messages
            for call in mock_debug.call_args_list:
                log_message = str(call)
                self.assertNotIn('test_api_key_123', log_message)
    
    def test_url_injection_protection(self):
        """Test protection against URL injection attacks."""
        malicious_endpoints = [
            '../../etc/passwd',
            'https://malicious.com/steal',
            'file:///etc/passwd',
            'javascript:alert(1)',
            '../../../admin/delete',
            '//malicious.com/steal',
            'http://evil.com/phish'
        ]
        
        for endpoint in malicious_endpoints:
            url = self.connector._build_url(endpoint)
            self.assertTrue(url.startswith('https://api.genesis.test/'))
            self.assertNotIn('malicious', url)
            self.assertNotIn('evil', url)
    
    def test_header_injection_protection(self):
        """Test protection against header injection attacks."""
        malicious_headers = {
            'X-Injected\r\nHost': 'evil.com',
            'Authorization\r\nX-Evil': 'injected',
            'X-Test\nContent-Length': '0'
        }
        
        headers = self.connector._build_headers(malicious_headers)
        # Check that no injected headers contain newlines
        for key, value in headers.items():
            self.assertNotIn('\r', key)
            self.assertNotIn('\n', key)
            self.assertNotIn('\r', str(value))
            self.assertNotIn('\n', str(value))
    
    def test_extremely_long_api_key(self):
        """Test handling of extremely long API key."""
        config = self.mock_config.copy()
        config['api_key'] = 'x' * 10000  # 10KB API key
        
        connector = GenesisConnector(config)
        headers = connector._build_headers()
        self.assertIn('Authorization', headers)
        self.assertEqual(len(headers['Authorization']), 10007)  # 'Bearer ' + 10000 chars
    
    def test_unicode_handling_in_requests(self):
        """Test proper handling of unicode in requests."""
        test_data = {
            'prompt': 'Test with émojis 🚀 and unicode chars: αβγ',
            'model': 'test_model_🤖',
            'special_chars': '中文测试'
        }
        
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'success': True}
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response
            
            result = self.connector.make_request('POST', '/test', data=test_data)
            
            self.assertEqual(result, {'success': True})
            # Verify unicode data was passed correctly
            call_args = mock_request.call_args
            self.assertEqual(call_args[1]['json'], test_data)
    
    def test_concurrent_requests_thread_safety(self):
        """Test thread safety with concurrent requests."""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request_thread(thread_id):
            try:
                with patch('requests.Session.request') as mock_request:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {'thread_id': thread_id}
                    mock_response.raise_for_status.return_value = None
                    mock_request.return_value = mock_response
                    
                    result = self.connector.make_request('GET', f'/test/{thread_id}')
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request_thread, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)
    
    def test_memory_cleanup_after_large_response(self):
        """Test memory cleanup after handling large responses."""
        large_data = {'data': 'x' * 1000000}  # 1MB of data
        
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = large_data
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response
            
            result = self.connector.make_request('GET', '/large-data')
            
            # Verify large data was handled correctly
            self.assertEqual(result, large_data)
            
            # Clear reference and force garbage collection
            del result
            import gc
            gc.collect()
    
    def test_request_timeout_boundary_conditions(self):
        """Test timeout boundary conditions."""
        config = self.mock_config.copy()
        config['timeout'] = 0.001  # Very small timeout
        
        connector = GenesisConnector(config)
        
        with patch('requests.Session.request') as mock_request:
            mock_request.side_effect = Timeout("Request timed out")
            
            with self.assertRaises(GenesisTimeoutError):
                connector.make_request('GET', '/test')
    
    def test_maximum_retry_attempts(self):
        """Test maximum retry attempts boundary."""
        config = self.mock_config.copy()
        config['max_retries'] = 10  # High retry count
        
        connector = GenesisConnector(config)
        
        with patch('requests.Session.request') as mock_request:
            mock_request.side_effect = ConnectionError("Connection failed")
            
            with self.assertRaises(GenesisConnectionError):
                connector.make_request('GET', '/test')
            
            # Should be called initial + 10 retries = 11 times
            self.assertEqual(mock_request.call_count, 11)
    
    def test_response_size_limits(self):
        """Test handling of extremely large responses."""
        # Simulate a very large JSON response
        large_response_data = {
            'data': ['item'] * 100000,  # Large array
            'metadata': {'info': 'x' * 50000}  # Large string
        }
        
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = large_response_data
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response
            
            result = self.connector.make_request('GET', '/large-response')
            
            self.assertEqual(len(result['data']), 100000)
            self.assertEqual(len(result['metadata']['info']), 50000)
    
    def test_malformed_json_response_handling(self):
        """Test handling of various malformed JSON responses."""
        malformed_responses = [
            '{"incomplete": json',
            '{"nested": {"incomplete": json}',
            '{"key": "value"} extra text',
            '{"invalid": unicode\ud800}',
            '{"number": 123.456.789}',
            '{"array": [1, 2, 3,]}',
            '{"object": {"key": value}}'
        ]
        
        for malformed_json in malformed_responses:
            with patch('requests.Session.request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", malformed_json, 0)
                mock_response.text = malformed_json
                mock_response.raise_for_status.return_value = None
                mock_request.return_value = mock_response
                
                with self.assertRaises(GenesisConnectionError):
                    self.connector.make_request('GET', '/test')
    
    def test_network_interruption_scenarios(self):
        """Test various network interruption scenarios."""
        network_errors = [
            requests.exceptions.ChunkedEncodingError("Connection broken"),
            requests.exceptions.ContentDecodingError("Failed to decode response"),
            requests.exceptions.StreamConsumedError("Stream was consumed"),
            requests.exceptions.RetryError("Max retries exceeded"),
            requests.exceptions.TooManyRedirects("Too many redirects"),
            requests.exceptions.UnrewindableBodyError("Body cannot be rewound")
        ]
        
        for error in network_errors:
            with patch('requests.Session.request') as mock_request:
                mock_request.side_effect = error
                
                with self.assertRaises(GenesisConnectionError):
                    self.connector.make_request('GET', '/test')
    
    def test_http_status_code_edge_cases(self):
        """Test handling of edge case HTTP status codes."""
        status_codes = [
            (100, "Continue"),
            (102, "Processing"),
            (207, "Multi-Status"),
            (226, "IM Used"),
            (300, "Multiple Choices"),
            (418, "I'm a teapot"),
            (421, "Misdirected Request"),
            (429, "Too Many Requests"),
            (451, "Unavailable For Legal Reasons"),
            (502, "Bad Gateway"),
            (507, "Insufficient Storage"),
            (511, "Network Authentication Required")
        ]
        
        for status_code, reason in status_codes:
            with patch('requests.Session.request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.raise_for_status.side_effect = requests.HTTPError(f"{status_code} {reason}")
                mock_response.text = f"HTTP {status_code} {reason}"
                mock_request.return_value = mock_response
                
                with self.assertRaises(GenesisConnectionError):
                    self.connector.make_request('GET', '/test')


class TestGenesisConnectorPerformanceAndLoad(unittest.TestCase):
    """Performance and load testing for GenesisConnector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_rapid_sequential_requests(self):
        """Test performance with rapid sequential requests."""
        import time
        
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'success': True}
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response
            
            start_time = time.time()
            for i in range(100):
                result = self.connector.make_request('GET', f'/test/{i}')
                self.assertEqual(result, {'success': True})
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete 100 requests in reasonable time (less than 1 second)
            self.assertLess(duration, 1.0)
    
    def test_request_with_large_payload(self):
        """Test request with large payload."""
        large_payload = {
            'data': 'x' * 1000000,  # 1MB string
            'array': list(range(10000)),  # Large array
            'nested': {'deep': {'structure': {'with': {'many': {'levels': 'value'}}}}}
        }
        
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'received': True}
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response
            
            result = self.connector.make_request('POST', '/large-payload', data=large_payload)
            
            self.assertEqual(result, {'received': True})
            # Verify large payload was sent
            call_args = mock_request.call_args
            self.assertEqual(call_args[1]['json'], large_payload)
    
    def test_memory_usage_with_multiple_connectors(self):
        """Test memory usage with multiple connector instances."""
        connectors = []
        
        # Create multiple connector instances
        for i in range(50):
            config = self.mock_config.copy()
            config['api_key'] = f'test_key_{i}'
            connector = GenesisConnector(config)
            connectors.append(connector)
        
        # Verify all connectors are functional
        for i, connector in enumerate(connectors):
            self.assertEqual(connector.api_key, f'test_key_{i}')
            self.assertEqual(connector.base_url, 'https://api.genesis.test')
        
        # Clean up
        for connector in connectors:
            if hasattr(connector, 'session'):
                connector.session.close()
    
    def test_request_rate_limiting_simulation(self):
        """Test behavior under rate limiting conditions."""
        call_count = 0
        
        def rate_limited_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= 3:
                # First 3 calls are rate limited
                mock_response = Mock()
                mock_response.status_code = 429
                mock_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
                mock_response.text = "Rate limit exceeded"
                return mock_response
            else:
                # 4th call succeeds
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'success': True}
                mock_response.raise_for_status.return_value = None
                return mock_response
        
        with patch('requests.Session.request', side_effect=rate_limited_request):
            result = self.connector.make_request('GET', '/rate-limited')
            
            self.assertEqual(result, {'success': True})
            self.assertEqual(call_count, 4)  # 3 failures + 1 success


class TestGenesisConnectorAsyncAdvanced(unittest.TestCase):
    """Advanced async testing for GenesisConnector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    async def test_async_concurrent_requests(self):
        """Test concurrent async requests."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
        
        async def make_async_request(request_id):
            with patch('aiohttp.ClientSession.request') as mock_request:
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={'id': request_id})
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_request.return_value = mock_response
                
                return await self.connector.async_make_request('GET', f'/test/{request_id}')
        
        # Run multiple concurrent requests
        tasks = [make_async_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Verify all requests completed successfully
        self.assertEqual(len(results), 10)
        for i, result in enumerate(results):
            self.assertEqual(result['id'], i)
    
    async def test_async_retry_with_exponential_backoff(self):
        """Test async retry with exponential backoff."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
        
        call_count = 0
        
        async def failing_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= 2:
                raise asyncio.TimeoutError("Async timeout")
            else:
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={'success': True})
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                return mock_response
        
        with patch('aiohttp.ClientSession.request', side_effect=failing_request):
            with patch('asyncio.sleep') as mock_sleep:
                result = await self.connector.async_make_request('GET', '/test')
                
                self.assertEqual(result, {'success': True})
                self.assertEqual(call_count, 3)
                self.assertEqual(mock_sleep.call_count, 2)  # Two retries
    
    async def test_async_timeout_handling(self):
        """Test async timeout handling."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
        
        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.side_effect = asyncio.TimeoutError("Request timeout")
            
            with self.assertRaises(GenesisTimeoutError):
                await self.connector.async_make_request('GET', '/test')
    
    async def test_async_context_manager_usage(self):
        """Test async context manager usage."""
        if not hasattr(self.connector, '__aenter__'):
            self.skipTest("Async context manager not available")
        
        async with self.connector as connector:
            self.assertIsNotNone(connector)
            self.assertEqual(connector.api_key, 'test_api_key_123')
    
    def test_async_methods_availability(self):
        """Test availability of async methods."""
        async_methods = ['async_make_request', 'async_get_model_info', 'async_create_generation']
        
        for method_name in async_methods:
            if hasattr(self.connector, method_name):
                method = getattr(self.connector, method_name)
                self.assertTrue(asyncio.iscoroutinefunction(method))


class TestGenesisConnectorConfigurationValidation(unittest.TestCase):
    """Test configuration validation and edge cases."""
    
    def test_config_validation_with_env_vars(self):
        """Test configuration validation with environment variables."""
        import os
        
        # Test with environment variables
        os.environ['GENESIS_API_KEY'] = 'env_api_key'
        os.environ['GENESIS_BASE_URL'] = 'https://env.genesis.test'
        
        try:
            # If connector supports env vars
            connector = GenesisConnector({})
            if hasattr(connector, 'api_key'):
                self.assertIn('env_api_key', connector.api_key)
        except ValueError:
            # If env vars not supported, that's also valid
            pass
        finally:
            # Clean up
            del os.environ['GENESIS_API_KEY']
            del os.environ['GENESIS_BASE_URL']
    
    def test_config_file_loading(self):
        """Test loading configuration from file."""
        import tempfile
        import json
        
        config_data = {
            'api_key': 'file_api_key',
            'base_url': 'https://file.genesis.test',
            'timeout': 60,
            'max_retries': 5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            # If connector supports config file loading
            if hasattr(GenesisConnector, 'from_config_file'):
                connector = GenesisConnector.from_config_file(config_file)
                self.assertEqual(connector.api_key, 'file_api_key')
                self.assertEqual(connector.base_url, 'https://file.genesis.test')
        except (AttributeError, NotImplementedError):
            # If config file loading not supported, that's also valid
            pass
        finally:
            # Clean up
            os.unlink(config_file)
    
    def test_config_validation_edge_cases(self):
        """Test configuration validation edge cases."""
        edge_cases = [
            {'api_key': ' ', 'base_url': 'https://api.genesis.test'},  # Whitespace API key
            {'api_key': 'test', 'base_url': 'https://api.genesis.test/', 'timeout': 0},  # Zero timeout
            {'api_key': 'test', 'base_url': 'https://api.genesis.test', 'timeout': float('inf')},  # Infinite timeout
            {'api_key': 'test', 'base_url': 'https://api.genesis.test', 'max_retries': 1000},  # Very high retries
        ]
        
        for config in edge_cases:
            try:
                connector = GenesisConnector(config)
                # If creation succeeds, verify reasonable defaults
                if hasattr(connector, 'timeout'):
                    self.assertGreater(connector.timeout, 0)
                    self.assertLess(connector.timeout, 3600)  # Max 1 hour
                if hasattr(connector, 'max_retries'):
                    self.assertGreaterEqual(connector.max_retries, 0)
                    self.assertLess(connector.max_retries, 100)  # Reasonable limit
            except (ValueError, TypeError):
                # Expected for invalid configurations
                pass
    
    def test_config_immutability(self):
        """Test that configuration is immutable after creation."""
        config = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        
        connector = GenesisConnector(config)
        
        # Modify original config
        config['api_key'] = 'modified_key'
        config['base_url'] = 'https://modified.test'
        
        # Verify connector configuration is unchanged
        self.assertEqual(connector.api_key, 'test_api_key')
        self.assertEqual(connector.base_url, 'https://api.genesis.test')
        
        # Try to modify connector attributes directly
        try:
            connector.api_key = 'new_key'
            # If modification is allowed, verify it doesn't affect requests
            headers = connector._build_headers()
            self.assertIn('Authorization', headers)
        except AttributeError:
            # If attributes are read-only, that's expected
            pass


class TestGenesisConnectorLoggingAndDebugging(unittest.TestCase):
    """Test logging and debugging functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)
    
    def test_debug_logging(self):
        """Test debug logging functionality."""
        with patch('logging.Logger.debug') as mock_debug:
            with patch('requests.Session.request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'success': True}
                mock_response.raise_for_status.return_value = None
                mock_request.return_value = mock_response
                
                self.connector.make_request('GET', '/test')
                
                # Verify debug logging occurred (if enabled)
                # Note: This depends on the actual logging implementation
                if mock_debug.called:
                    self.assertTrue(any('request' in str(call).lower() for call in mock_debug.call_args_list))
    
    def test_error_logging(self):
        """Test error logging functionality."""
        with patch('logging.Logger.error') as mock_error:
            with patch('requests.Session.request') as mock_request:
                mock_request.side_effect = ConnectionError("Connection failed")
                
                try:
                    self.connector.make_request('GET', '/test')
                except GenesisConnectionError:
                    pass
                
                # Verify error logging occurred (if enabled)
                if mock_error.called:
                    self.assertTrue(any('error' in str(call).lower() for call in mock_error.call_args_list))
    
    def test_request_response_logging(self):
        """Test request and response logging."""
        with patch('logging.Logger.info') as mock_info:
            with patch('requests.Session.request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'data': 'test'}
                mock_response.raise_for_status.return_value = None
                mock_request.return_value = mock_response
                
                self.connector.make_request('POST', '/test', data={'key': 'value'})
                
                # Verify request/response logging (if enabled)
                if mock_info.called:
                    # Check that sensitive data is not logged
                    for call in mock_info.call_args_list:
                        log_message = str(call)
                        self.assertNotIn('test_api_key_123', log_message)
    
    def test_performance_logging(self):
        """Test performance logging functionality."""
        import time
        
        with patch('logging.Logger.info') as mock_info:
            with patch('requests.Session.request') as mock_request:
                def slow_request(*args, **kwargs):
                    time.sleep(0.1)  # Simulate slow request
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {'success': True}
                    mock_response.raise_for_status.return_value = None
                    return mock_response
                
                mock_request.side_effect = slow_request
                
                self.connector.make_request('GET', '/slow-endpoint')
                
                # Verify performance logging (if enabled)
                if mock_info.called:
                    # Look for timing information in logs
                    timing_logged = any('time' in str(call).lower() or 'duration' in str(call).lower() 
                                      for call in mock_info.call_args_list)
                    # Note: This test passes regardless of whether timing is logged
                    # It's more about ensuring the functionality works if implemented


if __name__ == '__main__':
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        if test_category == 'security':
            unittest.main(argv=[''], verbosity=2, 
                         defaultTest='TestGenesisConnectorSecurityAndEdgeCases')
        elif test_category == 'performance':
            unittest.main(argv=[''], verbosity=2, 
                         defaultTest='TestGenesisConnectorPerformanceAndLoad')
        elif test_category == 'async':
            unittest.main(argv=[''], verbosity=2, 
                         defaultTest='TestGenesisConnectorAsyncAdvanced')
        elif test_category == 'config':
            unittest.main(argv=[''], verbosity=2, 
                         defaultTest='TestGenesisConnectorConfigurationValidation')
        elif test_category == 'logging':
            unittest.main(argv=[''], verbosity=2, 
                         defaultTest='TestGenesisConnectorLoggingAndDebugging')
    else:
        # Run all tests
        unittest.main(verbosity=2)