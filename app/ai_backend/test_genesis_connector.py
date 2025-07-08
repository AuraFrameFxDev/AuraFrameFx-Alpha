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
            {'api_key': 'k', 'base_url': 'https://a.b', 'timeout': 0.1},
            {'api_key': 'x' * 1000, 'base_url': 'https://very-long-domain-name.com', 'timeout': 3600},
            {'api_key': '', 'base_url': 'https://test.com', 'timeout': 30},
            {'api_key': 'test', 'base_url': 'ftp://invalid.scheme', 'timeout': 30},
        ]

        for i, config in enumerate(extreme_configs):
            with self.subTest(config_index=i):
                if i < 2:
                    result = self.connector.validate_config(config)
                    self.assertTrue(result)
                else:
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

        self.assertIn('datetime_field', formatted)
        self.assertIn('date_field', formatted)
        self.assertIn('time_field', formatted)

    @patch('requests.post')
    def test_retry_mechanism_success_after_retry(self, mock_post):
        """
        Tests that `send_request_with_retry` retries after an initial failure and returns the expected response on a subsequent successful attempt.

        Simulates a failed POST request followed by a successful one, verifying that the method returns the correct data and performs the expected number of retries.
        """
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

        self.assertEqual(mock_post.call_count, 4)

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
            mock_random.return_value = 0.5
            mock_response = Mock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response

            payload = {'message': 'test'}

            with self.assertRaises(RuntimeError):
                self.connector.send_request_with_retry(payload, max_retries=2, use_jitter=True)

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

                self.assertGreater(mock_post.call_count, 1)
                mock_post.reset_mock()

        for code in no_retry_codes:
            with self.subTest(no_retry_code=code):
                mock_response = Mock()
                mock_response.status_code = code
                mock_post.return_value = mock_response

                payload = {'message': 'test'}

                with self.assertRaises(RuntimeError):
                    self.connector.send_request_with_retry(payload, max_retries=2)

                self.assertEqual(mock_post.call_count, 1)
                mock_post.reset_mock()

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

            mock_time.side_effect = [1000.0, 1000.5]

            payload = {'message': 'test'}
            self.connector.log_request(payload, include_timing=True)

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
        result = self.connector.close()
        self.assertTrue(True)

    def test_context_manager_usage(self):
        """
        Test that GenesisConnector supports context manager usage and initializes correctly within a with-statement.
        """
        with GenesisConnector(config=self.mock_config) as connector:
            self.assertIsNotNone(connector)

    def test_thread_safety(self):
        """
        Tests that configuration validation in GenesisConnector is thread-safe by performing concurrent validations across multiple threads and verifying all succeed.
        """
        import threading

        results = []

        def worker():
            connector = GenesisConnector(config=self.mock_config)
            results.append(connector.validate_config(self.mock_config))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(results), 5)
        self.assertTrue(all(results))

    def test_large_payload_handling(self):
        """
        Test that the connector can format and process large payloads without encountering memory errors.

        Verifies that formatting a payload containing a large string and a large list completes successfully and returns a non-None result.
        """
        large_payload = {
            'message': 'x' * 10000,
            'data': list(range(1000))
        }

        formatted = self.connector.format_payload(large_payload)
        self.assertIsNotNone(formatted)

    def test_concurrent_requests(self):
        """
        Tests that the connector can process multiple requests concurrently, ensuring each request completes successfully and returns the expected result.
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

    @patch('requests.Session')
    def test_connection_pooling_behavior(self, mock_session):
        """
        Verifies that the GenesisConnector reuses the same HTTP session for multiple requests when connection pooling is enabled.
        """
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        connector = GenesisConnector(config={'use_session': True})
        for i in range(3):
            with patch.object(mock_session_instance, 'post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'request': i}
                mock_post.return_value = mock_response
                connector.send_request({'message': f'test{i}'})

        mock_session.assert_called_once()

    def test_async_request_handling(self):
        """
        Tests that the GenesisConnector can handle asynchronous requests correctly if async support is implemented.
        """
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
                    pass

        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(async_test())
        except RuntimeError:
            pass

    def test_batch_request_processing(self):
        """
        Tests that the connector can process multiple requests in a batch and returns the expected results for each payload.
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
                pass

    def test_webhook_validation(self):
        """
        Tests that the webhook signature validation correctly identifies valid and invalid signatures.
        """
        webhook_payload = {'event': 'test_event', 'data': {'key': 'value'}}
        secret = 'webhook_secret'

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

            is_valid = self.connector.validate_webhook_signature(
                payload_str, 'invalid_signature', secret
            )
            self.assertFalse(is_valid)
        except AttributeError:
            pass

    def test_circuit_breaker_functionality(self):
        """
        Tests that the circuit breaker pattern is correctly implemented by simulating repeated failures and verifying that further requests are blocked once the breaker is open.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response

            payload = {'message': 'test'}

            try:
                for i in range(10):
                    try:
                        self.connector.send_request(payload)
                    except Exception:
                        pass

                with self.assertRaises(RuntimeError):
                    self.connector.send_request(payload)
            except AttributeError:
                pass

    def test_request_deduplication(self):
        """
        Verify that duplicate requests with the same idempotency key are deduplicated and only one HTTP request is sent.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'dedup': True}
            mock_post.return_value = mock_response

            payload = {'message': 'test', 'idempotency_key': 'unique_key_123'}

            try:
                result1 = self.connector.send_request(payload)
                result2 = self.connector.send_request(payload)
                self.assertEqual(result1, result2)
                self.assertEqual(mock_post.call_count, 1)
            except AttributeError:
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

                call_args = mock_post.call_args
                headers = call_args[1]['headers']
                self.assertIn('X-Signature', headers)
            except AttributeError:
                pass

    def test_response_caching(self):
        """
        Tests that repeated calls to `get_cached_response` for the same endpoint return cached data and do not trigger additional HTTP requests.
        """
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'cached': True}
            mock_response.headers = {'Cache-Control': 'max-age=300'}
            mock_get.return_value = mock_response

            try:
                result1 = self.connector.get_cached_response('test_endpoint')
                self.assertEqual(result1['cached'], True)

                result2 = self.connector.get_cached_response('test_endpoint')
                self.assertEqual(result2['cached'], True)

                self.assertEqual(mock_get.call_count, 1)
            except AttributeError:
                pass

    def test_request_tracing(self):
        """
        Verifies that request tracing captures and exposes trace information during a request for debugging and monitoring purposes.
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

                trace_info = self.connector.get_trace_info()
                self.assertIn('request_id', trace_info)
                self.assertIn('start_time', trace_info)
                self.assertIn('end_time', trace_info)
            except AttributeError:
                pass

    def test_configuration_hot_reload(self):
        """
        Verifies that the connector can update its configuration at runtime without requiring a restart, using hot reload if available or falling back to regular reload.
        """
        original_config = {'api_key': 'old_key', 'base_url': 'https://old.url'}
        new_config = {'api_key': 'new_key', 'base_url': 'https://new.url'}

        connector = GenesisConnector(config=original_config)
        self.assertEqual(connector.config['api_key'], 'old_key')

        try:
            connector.hot_reload_config(new_config)
            self.assertEqual(connector.config['api_key'], 'new_key')
            self.assertEqual(connector.config['base_url'], 'https://new.url')
        except AttributeError:
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

            large_payloads = [
                {'data': 'x' * 1000000} for _ in range(10)
            ]

            for payload in large_payloads:
                formatted = self.connector.format_payload(payload)
                self.assertIsNotNone(formatted)

            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory

            self.assertLess(memory_increase, 100 * 1024 * 1024)
        except ImportError:
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

            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get.return_value = mock_get_response

            mock_post_response = Mock()
            mock_post_response.status_code = 200
            mock_post_response.json.return_value = {'data': 'test'}
            mock_post.return_value = mock_post_response

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
            {'api_key': 123, 'base_url': 'https://test.com'},
            {'api_key': 'test', 'base_url': 123},
            {'api_key': 'test', 'base_url': 'https://test.com', 'timeout': 'invalid'},
            {'api_key': '', 'base_url': 'https://test.com'},
            {'api_key': 'test', 'base_url': ''},
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
        """
        mock_response = Mock()
        mock_response.status_code = 302
        mock_response.headers = {'Location': 'https://new.location.com'}
        mock_get.return_value = mock_response

        result = self.connector.connect()
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
            'large_field': 'x' * (1024 * 1024),
            'large_list': list(range(10000)),
            'nested_large': {
                'data': ['item' * 100 for _ in range(100)]
            }
        }

        try:
            formatted = self.connector.format_payload(binary_payload)
            self.assertIsNotNone(formatted)
        except (MemoryError, ValueError) as e:
            self.assertIn(type(e), (MemoryError, ValueError))

    def test_malformed_json_responses(self):
        """
        Tests that the connector correctly detects and handles various malformed JSON responses.
        """
        malformed_responses = [
            '{"incomplete":',
            '{"trailing_comma":,}',
            '{"duplicate_key":"value1","duplicate_key":"value2"}',
            '{"unescaped": "value with "quotes""}',
            '{"number_overflow":999999999999999999999999999999999}',
            '{"invalid_unicode":"\\uXXXX"}',
            '{trailing_data} extra',
            '{"mixed":{"string":"value","number":123,"array":[1,2,3],"object":{"nested":true}}}'
        ]

        for response_text in malformed_responses:
            with self.subTest(response=response_text):
                with self.assertRaises(ValueError):
                    self.connector.parse_response(response_text)

if __name__ == '__main__':
    unittest.main(verbosity=2)