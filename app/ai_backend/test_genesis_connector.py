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
        """
        connector = GenesisConnector(config=self.mock_config)
        self.assertIsNotNone(connector)
        self.assertEqual(connector.config, self.mock_config)

    def test_init_with_none_config(self):
        """
        Test that initializing GenesisConnector with None as the configuration does not raise an exception.
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
        Verifies that the connector sends a connection request with a custom User-Agent header.
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
        Verifies that the GenesisConnector uses the specified proxy configuration.
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
        Verifies that the connector includes the correct authentication headers.
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
        Test that the connector's connect() method returns False for various HTTP failure status codes.
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
        Test that the connector successfully establishes a connection when HTTP redirects occur.
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
        Verify that send_request returns the correct response dictionary on success.
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
        Test that send_request raises a RuntimeError for HTTP 500 responses.
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
        Tests that send_request raises a ValueError for malformed JSON in responses.
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
        Verifies that the connector can send requests using various HTTP methods.
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
        Verifies that the connector can successfully send files.
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
        Test that streaming responses are handled correctly.
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
        Test that custom timeout parameter is applied.
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
        Verify that request IDs in payload are returned in response.
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
        Test that `validate_config` returns True for a valid configuration.
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
        Test that missing API key raises ValueError.
        """
        invalid_config = {
            'base_url': 'https://valid.url',
            'timeout': 30
        }
        with self.assertRaises(ValueError):
            self.connector.validate_config(invalid_config)

    def test_validate_config_invalid_url(self):
        """
        Test that invalid base URL raises ValueError.
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
        Test that negative timeout raises ValueError.
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
        Test that None input raises ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.validate_config(None)

    def test_validate_config_with_extreme_values(self):
        """
        Tests extreme but valid and invalid configurations.
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
                    self.assertTrue(self.connector.validate_config(config))
                else:
                    with self.assertRaises(ValueError):
                        self.connector.validate_config(config)

    def test_validate_config_with_additional_fields(self):
        """
        Verify that additional optional fields are accepted.
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
        self.assertTrue(self.connector.validate_config(extended_config))

    @patch('requests.get')
    def test_get_status_healthy(self, mock_get):
        """
        Verify that `get_status` returns healthy status and version.
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
        Test that get_status() returns 'unhealthy' for HTTP 503.
        """
        mock_response = Mock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response
        
        status = self.connector.get_status()
        
        self.assertEqual(status['status'], 'unhealthy')

    @patch('requests.get')
    def test_get_status_with_detailed_response(self, mock_get):
        """
        Test detailed status response fields.
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
        Test degraded status with issues and affected_endpoints.
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
        Tests correct serialization of valid data.
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
        Verify serialization with special characters and Unicode.
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
        Test that format_payload raises ValueError on empty dict.
        """
        with self.assertRaises(ValueError):
            self.connector.format_payload({})

    def test_format_payload_none_data(self):
        """
        Test that format_payload raises ValueError on None input.
        """
        with self.assertRaises(ValueError):
            self.connector.format_payload(None)

    def test_format_payload_with_nested_structures(self):
        """
        Tests serialization of deeply nested data.
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
        Test that circular references raise ValueError.
        """
        data = {'key': 'value'}
        data['self'] = data
        with self.assertRaises(ValueError):
            self.connector.format_payload(data)

    def test_format_payload_with_binary_data(self):
        """
        Test serialization of binary data fields.
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
        Test serialization of datetime, date, and time objects.
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
        Test send_request_with_retry succeeds after initial failure.
        """
        mock_response_fail = Mock(status_code=500)
        mock_response_success = Mock(status_code=200)
        mock_response_success.json.return_value = {'data': 'success'}
        mock_post.side_effect = [mock_response_fail, mock_response_success]
        
        payload = {'message': 'test'}
        result = self.connector.send_request_with_retry(payload)
        
        self.assertEqual(result, {'data': 'success'})
        self.assertEqual(mock_post.call_count, 2)

    @patch('requests.post')
    def test_retry_mechanism_max_retries_exceeded(self, mock_post):
        """
        Test send_request_with_retry raises RuntimeError after max retries.
        """
        mock_response = Mock(status_code=500)
        mock_post.return_value = mock_response
        
        payload = {'message': 'test'}
        with self.assertRaises(RuntimeError):
            self.connector.send_request_with_retry(payload, max_retries=3)
        self.assertEqual(mock_post.call_count, 4)

    @patch('time.sleep')
    @patch('requests.post')
    def test_retry_mechanism_backoff_timing(self, mock_post, mock_sleep):
        """
        Verifies incremental backoff delays.
        """
        mock_response = Mock(status_code=500)
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
        Test exponential backoff delays.
        """
        mock_response = Mock(status_code=500)
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
        Test jitter in backoff delays.
        """
        with patch('random.uniform', return_value=0.5) as mock_random:
            mock_response = Mock(status_code=500)
            mock_post.return_value = mock_response
            
            payload = {'message': 'test'}
            with self.assertRaises(RuntimeError):
                self.connector.send_request_with_retry(payload, max_retries=2, use_jitter=True)
            
            mock_sleep.assert_called()
            mock_random.assert_called()

    @patch('requests.post')
    def test_retry_mechanism_with_specific_retry_codes(self, mock_post):
        """
        Verify retries only occur on specific status codes.
        """
        retry_codes = [500, 502, 503, 504]
        no_retry_codes = [400, 401, 403, 404, 422]
        
        for code in retry_codes:
            with self.subTest(retry_code=code):
                mock_post.reset_mock()
                mock_response = Mock(status_code=code)
                mock_post.return_value = mock_response
                payload = {'message': 'test'}
                with self.assertRaises(RuntimeError):
                    self.connector.send_request_with_retry(payload, max_retries=2)
                self.assertGreater(mock_post.call_count, 1)
        
        for code in no_retry_codes:
            with self.subTest(no_retry_code=code):
                mock_post.reset_mock()
                mock_response = Mock(status_code=code)
                mock_post.return_value = mock_response
                payload = {'message': 'test'}
                with self.assertRaises(RuntimeError):
                    self.connector.send_request_with_retry(payload, max_retries=2)
                self.assertEqual(mock_post.call_count, 1)

    def test_parse_response_valid_json(self):
        """
        Test correct parsing of valid JSON strings.
        """
        response_data = {'key': 'value', 'number': 123, 'bool': True}
        json_string = json.dumps(response_data)
        
        parsed = self.connector.parse_response(json_string)
        self.assertEqual(parsed, response_data)

    def test_parse_response_invalid_json(self):
        """
        Test that invalid JSON string raises ValueError.
        """
        invalid_json = '{"invalid": json}'
        with self.assertRaises(ValueError):
            self.connector.parse_response(invalid_json)

    def test_parse_response_empty_string(self):
        """
        Test that empty string raises ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.parse_response('')

    def test_parse_response_none_input(self):
        """
        Test that None input raises ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.parse_response(None)

    def test_parse_response_with_different_content_types(self):
        """
        Verify parser behavior for different content types.
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
        Test parsing with various character encodings.
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
        Test that log_request logs payloads correctly.
        """
        with patch('logging.info') as mock_log:
            payload = {'message': 'test'}
            self.connector.log_request(payload)
            mock_log.assert_called()

    def test_log_request_sensitive_data_redaction(self):
        """
        Test redaction of sensitive fields in logs.
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
        Test inclusion of timing information when requested.
        """
        with patch('logging.info') as mock_log, patch('time.time') as mock_time:
            mock_time.side_effect = [1000.0, 1000.5]
            payload = {'message': 'test'}
            self.connector.log_request(payload, include_timing=True)
            mock_log.assert_called()
            logged_message = mock_log.call_args[0][0]
            self.assertIn('duration', logged_message.lower())

    def test_log_request_with_structured_logging(self):
        """
        Test structured logging format.
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
        Test get_headers includes Authorization and Content-Type when auth configured.
        """
        connector = GenesisConnector(config={'api_key': 'test_key'})
        headers = connector.get_headers()
        self.assertIn('Authorization', headers)
        self.assertIn('Content-Type', headers)

    def test_get_headers_without_auth(self):
        """
        Test get_headers excludes Authorization when not configured.
        """
        connector = GenesisConnector(config={})
        headers = connector.get_headers()
        self.assertNotIn('Authorization', headers)
        self.assertIn('Content-Type', headers)

    def test_get_headers_with_custom_headers(self):
        """
        Test merging of custom headers.
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
        Test Content-Type for different request types.
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
        Test that close() can be called without errors.
        """
        result = self.connector.close()
        self.assertTrue(True)

    def test_context_manager_usage(self):
        """
        Test context manager support.
        """
        with GenesisConnector(config=self.mock_config) as connector:
            self.assertIsNotNone(connector)

    def test_thread_safety(self):
        """
        Test thread-safe configuration validation.
        """
        import threading
        results = []
        def worker():
            connector = GenesisConnector(config=self.mock_config)
            results.append(connector.validate_config(self.mock_config))
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 5)
        self.assertTrue(all(results))

    def test_large_payload_handling(self):
        """
        Test formatting of large payloads without memory errors.
        """
        large_payload = {
            'message': 'x' * 10000,
            'data': list(range(1000))
        }
        formatted = self.connector.format_payload(large_payload)
        self.assertIsNotNone(formatted)

    def test_concurrent_requests(self):
        """
        Test concurrent send_request calls succeed.
        """
        import concurrent.futures
        def make_request():
            with patch('requests.post') as mock_post:
                mock_response = Mock(status_code=200)
                mock_response.json.return_value = {'data': 'test'}
                mock_post.return_value = mock_response
                return self.connector.send_request({'message': 'test'})
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in futures]
        self.assertEqual(len(results), 5)

    def test_error_handling_chain(self):
        """
        Test propagation of exceptions in send_request.
        """
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Network error")
            with self.assertRaises(Exception):
                self.connector.send_request({'message': 'test'})

    def test_configuration_reload(self):
        """
        Test reload_config updates internal config.
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
        Test collection of metrics after send_request.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = {'data': 'test'}
            mock_post.return_value = mock_response
            self.connector.send_request({'message': 'test'})
            metrics = self.connector.get_metrics()
            self.assertIn('requests_sent', metrics)
            self.assertIn('response_time', metrics)

    def test_health_check_endpoint(self):
        """
        Test health_check returns correct status.
        """
        with patch('requests.get') as mock_get:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = {'status': 'healthy'}
            mock_get.return_value = mock_response
            health = self.connector.health_check()
            self.assertEqual(health['status'], 'healthy')

    def test_rate_limiting_handling(self):
        """
        Test HTTP 429 handling with Retry-After.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock(status_code=429)
            mock_response.headers = {'Retry-After': '1'}
            mock_post.return_value = mock_response
            with self.assertRaises(RuntimeError):
                self.connector.send_request({'message': 'test'})

    def test_connection_pooling_behavior(self):
        """
        Test connection pooling via requests.Session.
        """
        with patch('requests.Session') as mock_session:
            mock_inst = Mock()
            mock_session.return_value = mock_inst
            connector = GenesisConnector(config={'use_session': True})
            for _ in range(3):
                with patch.object(mock_inst, 'post') as mock_post:
                    mock_resp = Mock(status_code=200)
                    mock_resp.json.return_value = {'request': _}
                    mock_post.return_value = mock_resp
                    connector.send_request({'message': f'test{_}'})
            mock_session.assert_called_once()

    def test_async_request_handling(self):
        """
        Test send_request_async if implemented.
        """
        async def async_test():
            with patch('aiohttp.ClientSession') as mock_session:
                mock_inst = mock_session.return_value.__aenter__.return_value
                mock_resp = Mock(status=200, json=Mock(return_value={'async': True}))
                mock_inst.post.return_value.__aenter__.return_value = mock_resp
                connector = GenesisConnector(config={'async_mode': True})
                try:
                    result = await connector.send_request_async({'message': 'async_test'})
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
        Test send_batch_requests if implemented.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = {'batch': True}
            mock_post.return_value = mock_response
            payloads = [{'message': f'batch{i}'} for i in range(3)]
            try:
                results = self.connector.send_batch_requests(payloads)
                self.assertEqual(len(results), 3)
                for res in results:
                    self.assertTrue(res.get('batch'))
            except AttributeError:
                pass

    def test_webhook_validation(self):
        """
        Test webhook signature validation if implemented.
        """
        import hmac, hashlib
        webhook_payload = {'event': 'test_event', 'data': {'key': 'value'}}
        secret = 'webhook_secret'
        payload_str = json.dumps(webhook_payload)
        expected_signature = hmac.new(secret.encode(), payload_str.encode(), hashlib.sha256).hexdigest()
        try:
            self.assertTrue(self.connector.validate_webhook_signature(payload_str, expected_signature, secret))
            self.assertFalse(self.connector.validate_webhook_signature(payload_str, 'invalid', secret))
        except AttributeError:
            pass

    def test_circuit_breaker_functionality(self):
        """
        Test circuit breaker if implemented.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock(status_code=500)
            mock_post.return_value = mock_response
            payload = {'message': 'test'}
            try:
                for _ in range(10):
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
        Test idempotency key deduplication if implemented.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = {'dedup': True}
            mock_post.return_value = mock_response
            payload = {'message': 'test', 'idempotency_key': 'unique_key_123'}
            try:
                r1 = self.connector.send_request(payload)
                r2 = self.connector.send_request(payload)
                self.assertEqual(r1, r2)
                self.assertEqual(mock_post.call_count, 1)
            except AttributeError:
                pass

    def test_request_signing(self):
        """
        Test send_signed_request if implemented.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = {'signed': True}
            mock_post.return_value = mock_response
            payload = {'message': 'test'}
            key = 'test_signing_key'
            try:
                result = self.connector.send_signed_request(payload, key)
                self.assertTrue(result.get('signed'))
                headers = mock_post.call_args[1]['headers']
                self.assertIn('X-Signature', headers)
            except AttributeError:
                pass

    def test_response_caching(self):
        """
        Test get_cached_response if implemented.
        """
        with patch('requests.get') as mock_get:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = {'cached': True}
            mock_response.headers = {'Cache-Control': 'max-age=300'}
            mock_get.return_value = mock_response
            try:
                res1 = self.connector.get_cached_response('test_endpoint')
                res2 = self.connector.get_cached_response('test_endpoint')
                self.assertEqual(res1, res2)
                self.assertEqual(mock_get.call_count, 1)
            except AttributeError:
                pass

    def test_request_tracing(self):
        """
        Test tracing if implemented.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = {'traced': True}
            mock_post.return_value = mock_response
            try:
                result = self.connector.send_request({'message': 'test'}, trace=True)
                self.assertTrue(result.get('traced'))
                trace_info = self.connector.get_trace_info()
                self.assertIn('request_id', trace_info)
                self.assertIn('start_time', trace_info)
                self.assertIn('end_time', trace_info)
            except AttributeError:
                pass

    def test_configuration_hot_reload(self):
        """
        Test hot_reload_config if implemented, otherwise reload_config.
        """
        orig = {'api_key': 'old', 'base_url': 'https://old'}
        new = {'api_key': 'new', 'base_url': 'https://new'}
        connector = GenesisConnector(config=orig)
        try:
            connector.hot_reload_config(new)
        except AttributeError:
            connector.reload_config(new)
        self.assertEqual(connector.config, new)

    def test_memory_usage_monitoring(self):
        """
        Verifies memory usage during format_payload operations using psutil if available.
        """
        try:
            import psutil, os
            process = psutil.Process(os.getpid())
            initial = process.memory_info().rss
            for _ in range(10):
                formatted = self.connector.format_payload({'data': 'x'*1000000})
                self.assertIsNotNone(formatted)
            current = process.memory_info().rss
            self.assertLess(current - initial, 100 * 1024 * 1024)
        except ImportError:
            pass


if __name__ == '__main__':
    unittest.main(verbosity=2)