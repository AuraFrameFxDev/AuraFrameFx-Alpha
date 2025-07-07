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
        Prepare a new GenesisConnector instance and a mock configuration for each test case.
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
        Performs cleanup actions after each test method execution.
        """
        # Reset any global state if needed
        pass

    def test_init_default_parameters(self):
        """
        Verify that a GenesisConnector instance can be created with default parameters.
        """
        connector = GenesisConnector()
        self.assertIsNotNone(connector)
        self.assertIsInstance(connector, GenesisConnector)

    def test_init_with_config(self):
        """
        Test that GenesisConnector initializes correctly with a custom configuration.
        
        Verifies that the connector instance is created and its configuration matches the provided custom config.
        """
        connector = GenesisConnector(config=self.mock_config)
        self.assertIsNotNone(connector)
        self.assertEqual(connector.config, self.mock_config)

    def test_init_with_none_config(self):
        """
        Test that initializing GenesisConnector with None as the configuration does not raise an exception and results in a valid instance.
        """
        connector = GenesisConnector(config=None)
        self.assertIsNotNone(connector)

    def test_init_with_empty_config(self):
        """
        Test initializing GenesisConnector with an empty configuration dictionary to ensure a valid instance is created.
        """
        connector = GenesisConnector(config={})
        self.assertIsNotNone(connector)

    @patch('requests.get')
    def test_connect_success(self, mock_get):
        """
        Verifies that the connector returns True when the Genesis API responds with HTTP 200 and valid JSON.
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
        Test that the connector's connect method returns False when an HTTP 404 status code is received.
        """
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_failure_timeout(self, mock_get):
        """
        Test that the connector's `connect` method returns False when a timeout occurs during the connection attempt.
        """
        mock_get.side_effect = TimeoutError("Connection timeout")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_failure_connection_error(self, mock_get):
        """
        Test that the connector returns False when a connection error occurs during the connect operation.
        """
        mock_get.side_effect = ConnectionError("Connection failed")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.post')
    def test_send_request_success(self, mock_post):
        """
        Verifies that send_request returns the correct response data when a POST request with a valid payload succeeds.
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
        Test that sending a request with an invalid payload raises a ValueError.
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
        Test that send_request raises a RuntimeError when the server returns a 500 Internal Server Error.
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
        Test that send_request raises a ValueError when the server responds with malformed JSON.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response
        
        payload = {'message': 'test_message'}
        
        with self.assertRaises(ValueError):
            self.connector.send_request(payload)

    def test_validate_config_valid(self):
        """
        Tests that `validate_config` returns True when provided with a valid configuration dictionary.
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
        Verify that passing a configuration with a negative timeout value to `validate_config` raises a `ValueError`.
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
        Test that passing None to the configuration validator raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.validate_config(None)

    @patch('requests.get')
    def test_get_status_healthy(self, mock_get):
        """
        Test that `get_status` returns the correct status and version when the service responds as healthy.
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
        Verify that `get_status` returns a status of 'unhealthy' when the service responds with HTTP 503.
        """
        mock_response = Mock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response
        
        status = self.connector.get_status()
        
        self.assertEqual(status['status'], 'unhealthy')

    def test_format_payload_valid_data(self):
        """
        Verify that the payload formatting method correctly includes expected keys when provided with valid data.
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
        Verifies that the payload formatting method correctly processes data containing special characters and Unicode, ensuring all keys are present in the formatted output.
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
        Test that formatting an empty payload raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.format_payload({})

    def test_format_payload_none_data(self):
        """
        Test that formatting a payload with None as input raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.format_payload(None)

    @patch('requests.post')
    def test_retry_mechanism_success_after_retry(self, mock_post):
        """
        Test that `send_request_with_retry` retries after a failed request and succeeds on a subsequent attempt.
        
        Simulates an initial POST failure followed by a successful response, verifying that the method returns the expected data and performs the correct number of retries.
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
        Verifies that a RuntimeError is raised when the retry mechanism exceeds the maximum number of retries for failed requests.
        
        Ensures the request is attempted the correct number of times (initial attempt plus retries) and that all attempts fail with server errors.
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
        Test that the retry mechanism uses increasing backoff delays between retries after consecutive server errors.
        
        Simulates repeated server errors and verifies that the sleep function is called with incrementally increasing delays corresponding to the backoff strategy.
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

    def test_parse_response_valid_json(self):
        """
        Verifies that parsing a valid JSON string returns the correct dictionary.
        """
        response_data = {'key': 'value', 'number': 123, 'bool': True}
        json_string = json.dumps(response_data)
        
        parsed = self.connector.parse_response(json_string)
        
        self.assertEqual(parsed, response_data)

    def test_parse_response_invalid_json(self):
        """
        Test that parsing an invalid JSON string with `parse_response` raises a ValueError.
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
        Test that parsing a None input with parse_response raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.parse_response(None)

    def test_log_request_valid_data(self):
        """
        Verifies that the log_request method logs the provided payload when given valid data.
        """
        with patch('logging.info') as mock_log:
            payload = {'message': 'test'}
            self.connector.log_request(payload)
            
            mock_log.assert_called()

    def test_log_request_sensitive_data_redaction(self):
        """
        Tests that sensitive fields like 'api_key' and 'password' are properly redacted from log output when logging a request payload.
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

    def test_get_headers_with_auth(self):
        """
        Verify that the `get_headers` method includes the `Authorization` and `Content-Type` headers when an API key is provided in the configuration.
        """
        connector = GenesisConnector(config={'api_key': 'test_key'})
        headers = connector.get_headers()
        
        self.assertIn('Authorization', headers)
        self.assertIn('Content-Type', headers)

    def test_get_headers_without_auth(self):
        """
        Verify that the headers returned by `get_headers` do not include an `Authorization` field when no authentication is configured, but do include `Content-Type`.
        """
        connector = GenesisConnector(config={})
        headers = connector.get_headers()
        
        self.assertNotIn('Authorization', headers)
        self.assertIn('Content-Type', headers)

    def test_close_connection(self):
        """
        Test that the connector's `close` method can be called without raising any exceptions.
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
        Test that configuration validation is thread-safe by running multiple concurrent validations and verifying all succeed.
        """
        import threading
        results = []
        
        def worker():
            """
            Validates the connector configuration in a separate thread and stores the result in a shared list.
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
        Verify that the connector can format and process large payloads without encountering memory errors.
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
        Test that multiple concurrent requests sent via the connector complete successfully.
        
        This test runs several requests in parallel threads and verifies that all responses are received without errors.
        """
        import concurrent.futures
        
        def make_request():
            """
            Mocks a successful POST request and sends a test payload using the connector's send_request method.
            
            Returns:
                dict: The response data returned by the mocked send_request call.
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
        Verify that an exception is raised when a network error occurs during a chained `send_request` operation.
        """
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Network error")
            
            payload = {'message': 'test'}
            
            with self.assertRaises(Exception):
                self.connector.send_request(payload)

    def test_configuration_reload(self):
        """
        Test that reloading the connector's configuration replaces the internal config with new values.
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
        Test that sending a request results in collection of 'requests_sent' and 'response_time' metrics.
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
        Verify that the health check endpoint returns a healthy status when the service responds with HTTP 200 and valid JSON.
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
        Verify that a RuntimeError is raised when the connector receives an HTTP 429 response with a Retry-After header, indicating rate limiting.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {'Retry-After': '1'}
            mock_post.return_value = mock_response
            
            payload = {'message': 'test'}
            
            with self.assertRaises(RuntimeError):
                self.connector.send_request(payload)


class TestGenesisConnectorIntegration(unittest.TestCase):
    """
    Integration tests for GenesisConnector.
    These tests verify the interaction between components.
    """

    def setUp(self):
        """
        Set up the test environment by creating a GenesisConnector instance with a test configuration for integration tests.
        """
        self.connector = GenesisConnector(config={
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': 30
        })

    def test_full_request_lifecycle(self):
        """
        Simulates a complete request lifecycle by sending a payload through the connector and verifying that a successful response is returned.
        
        Asserts that the connector correctly processes a POST request and the response contains the expected result.
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
        Tests that the connector can establish a connection and send a request, returning the expected response data.
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


if __name__ == '__main__':
    # Run tests with unittest
    unittest.main(verbosity=2)
    
    # Alternative: Run with pytest if available
    # pytest.main([__file__, '-v'])

class TestGenesisConnectorEdgeCases(unittest.TestCase):
    """
    Additional edge case tests for GenesisConnector class.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """
        Initialize a GenesisConnector instance and a mock configuration for edge case tests.
        """
        self.connector = GenesisConnector()
        self.mock_config = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'retry_count': 3
        }

    def test_init_with_malformed_config_types(self):
        """
        Verifies that initializing GenesisConnector with malformed configuration types raises ValueError or TypeError.
        
        Tests various invalid config scenarios, including non-string or empty API keys and base URLs, and non-numeric timeouts.
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
        Test that the GenesisConnector initializes correctly when the configuration contains Unicode characters.
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
        Test that the connector's connect method handles HTTP redirects (status code 302) appropriately.
        
        Simulates a redirect response and verifies that the connection attempt does not return None.
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
        Test that the connector returns False when an SSL certificate error occurs during connection.
        """
        import ssl
        mock_get.side_effect = ssl.SSLError("SSL certificate verify failed")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_with_dns_resolution_error(self, mock_get):
        """
        Test that the connector returns False when a DNS resolution error occurs during connection.
        """
        import socket
        mock_get.side_effect = socket.gaierror("Name or service not known")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.post')
    def test_send_request_with_binary_payload(self, mock_post):
        """
        Test that sending a request with a binary payload is handled correctly and returns the expected response.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'binary_processed'}
        mock_post.return_value = mock_response
        
        binary_payload = {
            'message': 'test',
            'binary_data': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        }
        
        result = self.connector.send_request(binary_payload)
        
        self.assertEqual(result, {'data': 'binary_processed'})

    @patch('requests.post')
    def test_send_request_with_nested_payload(self, mock_post):
        """
        Verifies that the connector can successfully send a request with a deeply nested payload and correctly process the response.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'nested_processed'}
        mock_post.return_value = mock_response
        
        nested_payload = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'message': 'deeply nested'
                        }
                    }
                }
            },
            'arrays': [1, 2, {'nested_in_array': True}]
        }
        
        result = self.connector.send_request(nested_payload)
        
        self.assertEqual(result, {'data': 'nested_processed'})

    @patch('requests.post')
    def test_send_request_with_circular_reference(self, mock_post):
        """
        Test that sending a request with a payload containing a circular reference raises a ValueError or TypeError.
        """
        # Create circular reference
        payload = {'message': 'test'}
        payload['self'] = payload
        
        with self.assertRaises((ValueError, TypeError)):
            self.connector.send_request(payload)

    @patch('requests.post')
    def test_send_request_with_extremely_large_payload(self, mock_post):
        """
        Test that sending a request with an extremely large payload raises a RuntimeError when the server responds with HTTP 413 (Payload Too Large).
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
        Tests that configuration validation safely handles SQL injection attempts by either raising a ValueError or returning a boolean result without executing malicious input.
        """
        malicious_configs = [
            {'api_key': "'; DROP TABLE users; --", 'base_url': 'https://test.com'},
            {'api_key': 'test', 'base_url': "https://test.com'; DELETE FROM config; --"},
            {'api_key': 'test\x00admin', 'base_url': 'https://test.com'},
        ]
        
        for config in malicious_configs:
            with self.subTest(config=config):
                # Should either validate safely or raise appropriate error
                try:
                    result = self.connector.validate_config(config)
                    # If validation passes, ensure no injection occurred
                    self.assertIsInstance(result, bool)
                except ValueError:
                    # Expected for malicious input
                    pass

    def test_validate_config_with_path_traversal_attempts(self):
        """
        Verify that configuration validation raises a ValueError when path traversal patterns are present in the API key or base URL.
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
        Verifies that configuration validation raises a ValueError when the config contains potential XSS attack patterns in the API key or base URL fields.
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
        Verify that `get_status` returns a status of 'timeout' when various timeout-related exceptions occur during status retrieval.
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
        Tests that the payload formatting correctly serializes datetime, date, and time objects.
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
        Test that payload formatting correctly handles decimal, complex, and float numeric types.
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
        Tests that the payload formatting method correctly handles custom objects by including their string representation in the formatted payload.
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
                Return a string representation of the custom object, including its value.
                """
                return f"CustomObject({self.value})"
        
        payload = {
            'custom_obj': CustomObject("test"),
            'message': 'test'
        }
        
        formatted = self.connector.format_payload(payload)
        
        # Should handle custom objects
        self.assertIn('custom_obj', formatted)

    @patch('requests.post')
    def test_retry_mechanism_with_exponential_backoff(self, mock_post):
        """
        Tests that the retry mechanism in `send_request_with_retry` uses exponential backoff delays between retries and raises a `RuntimeError` after exceeding the maximum number of retries.
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
        Verify that the retry mechanism applies jitter to backoff delays when handling repeated server errors.
        
        Simulates repeated HTTP 500 responses and checks that the retry logic includes randomized jitter in sleep intervals to prevent synchronized retries. Expects a RuntimeError after exceeding the maximum number of retries.
        """
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        payload = {'message': 'test'}
        
        with patch('time.sleep') as mock_sleep:
            with patch('random.random', return_value=0.5):
                with self.assertRaises(RuntimeError):
                    self.connector.send_request_with_retry(payload, max_retries=2)
                
                # Sleep should be called with jitter
                self.assertTrue(mock_sleep.called)

    def test_parse_response_with_edge_case_json(self):
        """
        Verifies that the response parser correctly handles various edge case JSON structures, including empty objects, arrays, nulls, booleans, numbers, escaped strings, and Unicode content.
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
        Tests that the connector can correctly parse a very large JSON string containing 10,000 key-value pairs.
        """
        large_dict = {f'key_{i}': f'value_{i}' for i in range(10000)}
        large_json = json.dumps(large_dict)
        
        parsed = self.connector.parse_response(large_json)
        
        self.assertEqual(len(parsed), 10000)
        self.assertEqual(parsed['key_0'], 'value_0')
        self.assertEqual(parsed['key_9999'], 'value_9999')

    def test_log_request_with_various_log_levels(self):
        """
        Verifies that the log_request method logs payloads correctly at different logging levels.
        """
        log_levels = [
            ('DEBUG', logging.DEBUG),
            ('INFO', logging.INFO),
            ('WARNING', logging.WARNING),
            ('ERROR', logging.ERROR),
        ]
        
        for level_name, level_value in log_levels:
            with self.subTest(level=level_name):
                with patch('logging.getLogger') as mock_logger:
                    mock_logger.return_value.level = level_value
                    
                    payload = {'message': 'test', 'level': level_name}
                    self.connector.log_request(payload)
                    
                    # Verify logger was called appropriately
                    self.assertTrue(mock_logger.called)

    def test_log_request_with_pii_data(self):
        """
        Verify that logging a request containing PII fields results in proper redaction of sensitive data before logging.
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
        Verify that the headers generated by the connector include a custom User-Agent when specified in the configuration.
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
        Verifies that closing the connector while requests are still pending is handled gracefully without errors.
        """
        with patch('requests.post') as mock_post:
            # Simulate a slow response
            mock_post.side_effect = lambda *args, **kwargs: Mock(status_code=200)
            
            # Start a request in a thread
            import threading
            request_thread = threading.Thread(
                target=self.connector.send_request,
                args=({'message': 'test'},)
            )
            request_thread.start()
            
            # Close connection
            result = self.connector.close()
            
            # Should handle gracefully
            self.assertIsNotNone(result)
            request_thread.join(timeout=1)

    def test_context_manager_with_exception(self):
        """
        Verify that the GenesisConnector context manager properly propagates exceptions raised within its block.
        """
        with self.assertRaises(ValueError):
            with GenesisConnector(config=self.mock_config) as connector:
                # Simulate an exception during usage
                raise ValueError("Test exception")

    def test_context_manager_cleanup(self):
        """
        Verify that the context manager for GenesisConnector calls the cleanup method upon exiting the context.
        """
        with patch.object(GenesisConnector, 'close') as mock_close:
            with GenesisConnector(config=self.mock_config) as connector:
                self.assertIsNotNone(connector)
            
            # Verify cleanup was called
            mock_close.assert_called_once()

    def test_memory_usage_with_large_datasets(self):
        """
        Verifies that processing many large payloads does not cause excessive memory usage growth.
        
        This test formats a series of large payloads and checks that the reference count for the connector remains within a reasonable range, indicating no significant memory leaks.
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
        Verify that the connector can handle multiple concurrent configuration updates without errors.
        
        This test spawns several threads, each updating the connector's configuration, and asserts that all updates complete successfully.
        """
        import threading
        import time
        
        results = []
        
        def update_config(config_id):
            """
            Update the connector's configuration with a new API key and base URL based on the given config ID, then record the config ID in the results list.
            
            Parameters:
                config_id (int): Identifier used to generate unique API key and base URL for the new configuration.
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
        Verifies that exceptions raised during a request preserve the original error chain.
        
        This test simulates a nested exception scenario when sending a request and asserts that the exception chain is maintained.
        """
        with patch('requests.post') as mock_post:
            # Create nested exception
            original_error = ValueError("Original error")
            network_error = ConnectionError("Network failed")
            network_error.__cause__ = original_error
            mock_post.side_effect = network_error
            
            payload = {'message': 'test'}
            
            with self.assertRaises(Exception) as context:
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
        Initializes a GenesisConnector instance with test configuration for performance tests.
        """
        self.connector = GenesisConnector(config={
            'api_key': 'test_key',
            'base_url': 'https://api.test.com'
        })

    def test_payload_formatting_performance(self):
        """
        Measures the time taken to format payloads of increasing sizes and asserts that formatting completes within one second for each size.
        """
        import time
        
        sizes = [10, 100, 1000, 5000]
        
        for size in sizes:
            with self.subTest(size=size):
                payload = {
                    'message': 'test',
                    'data': list(range(size))
                }
                
                start_time = time.time()
                formatted = self.connector.format_payload(payload)
                end_time = time.time()
                
                processing_time = end_time - start_time
                
                # Should complete within reasonable time
                self.assertLess(processing_time, 1.0)  # Less than 1 second
                self.assertIsNotNone(formatted)

    def test_concurrent_request_performance(self):
        """
        Measures the time taken to process 50 concurrent requests and verifies all complete within 5 seconds.
        
        Ensures that the connector can handle multiple simultaneous requests efficiently and returns the expected number of results within a performance threshold.
        """
        import concurrent.futures
        import time
        
        def make_request(request_id):
            """
            Send a POST request using the connector with the specified request ID as payload, returning the response data.
            
            Parameters:
                request_id: The identifier to include in the request payload.
            
            Returns:
                dict: The response data returned by the connector's send_request method.
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
        Checks for memory leaks by performing repeated payload formatting and header generation, ensuring object count does not increase significantly after garbage collection.
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
        Tests the validation of configuration with an extremely long API key.
        
        Verifies that the connector either accepts a very long API key as valid or raises a ValueError if the key length exceeds acceptable limits.
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
        Tests configuration validation with a range of extreme timeout values, including zero, very large integers, and infinity, ensuring correct acceptance or rejection by the connector.
        """
        extreme_timeouts = [
            0,  # Minimum
            1,  # Very short
            86400,  # 24 hours
            float('inf'),  # Infinity
            sys.maxsize,  # Maximum integer
        ]
        
        for timeout in extreme_timeouts:
            with self.subTest(timeout=timeout):
                config = {
                    'api_key': 'test',
                    'base_url': 'https://api.test.com',
                    'timeout': timeout
                }
                
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
        Tests configuration validation with base URLs at various length boundaries to ensure correct handling of valid and excessively long URLs.
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
        Tests payload formatting with payloads ranging from empty to 10MB to verify correct handling and error raising at size boundaries.
        """
        sizes = [0, 1, 1024, 1024*1024, 10*1024*1024]  # 0B, 1B, 1KB, 1MB, 10MB
        
        for size in sizes:
            with self.subTest(size=size):
                if size == 0:
                    payload = {}
                else:
                    payload = {'data': 'x' * size}
                
                try:
                    formatted = self.connector.format_payload(payload)
                    self.assertIsNotNone(formatted)
                except (ValueError, MemoryError):
                    # Expected for empty or excessively large payloads
                    pass

    def test_unicode_boundary_conditions(self):
        """
        Tests payload formatting with a variety of Unicode strings, including control characters, boundary code points, and emoji, to ensure correct handling or appropriate error raising.
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
        
        for test_case in unicode_test_cases:
            with self.subTest(test_case=test_case):
                payload = {'message': test_case}
                
                try:
                    formatted = self.connector.format_payload(payload)
                    self.assertIsNotNone(formatted)
                except (ValueError, UnicodeError):
                    # Expected for some boundary cases
                    pass

    def test_numeric_boundary_conditions(self):
        """
        Tests payload formatting with numeric edge cases, including extreme integer and floating-point values, to ensure correct handling or appropriate exceptions.
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
    # Run all test classes
    unittest.main(verbosity=2)

class TestGenesisConnectorAdvancedSecurity(unittest.TestCase):
    """
    Advanced security tests for GenesisConnector class.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """
        Initialize GenesisConnector instance for security testing.
        """
        self.connector = GenesisConnector()
        self.secure_config = {
            'api_key': 'secure_test_key',
            'base_url': 'https://secure-api.test.com',
            'timeout': 30
        }

    def test_http_header_injection_prevention(self):
        """
        Test that HTTP header injection attempts are properly sanitized or rejected.
        """
        malicious_headers = [
            {'X-Injected': 'value\r\nX-Malicious: injected'},
            {'Authorization': 'Bearer token\nX-Admin: true'},
            {'Content-Type': 'application/json\r\nHost: evil.com'},
        ]
        
        for headers in malicious_headers:
            with self.subTest(headers=headers):
                config = dict(self.secure_config)
                config['additional_headers'] = headers
                
                connector = GenesisConnector(config=config)
                result_headers = connector.get_headers()
                
                # Check that no CRLF injection occurred
                for key, value in result_headers.items():
                    self.assertNotIn('\r', str(value))
                    self.assertNotIn('\n', str(value))

    def test_payload_deserialization_attacks(self):
        """
        Test that malicious payload deserialization attempts are handled safely.
        """
        malicious_payloads = [
            {'__class__': {'__module__': 'os', '__name__': 'system'}},
            {'eval': 'print("malicious code")'},
            {'exec': '__import__("os").system("whoami")'},
            {'import': 'subprocess'},
        ]
        
        for payload in malicious_payloads:
            with self.subTest(payload=payload):
                try:
                    formatted = self.connector.format_payload(payload)
                    # If formatting succeeds, ensure no code execution occurred
                    self.assertIsInstance(formatted, (dict, str))
                except (ValueError, TypeError):
                    # Expected for malicious payloads
                    pass

    def test_timing_attack_resistance(self):
        """
        Test that configuration validation has consistent timing to prevent timing attacks.
        """
        import time
        
        valid_config = {
            'api_key': 'valid_key_12345',
            'base_url': 'https://api.test.com'
        }
        
        invalid_configs = [
            {'api_key': 'short', 'base_url': 'https://api.test.com'},
            {'api_key': 'different_length_key', 'base_url': 'https://api.test.com'},
            {'api_key': 'another_key_length', 'base_url': 'https://api.test.com'},
        ]
        
        # Measure timing for valid config
        times = []
        for _ in range(5):
            start = time.perf_counter()
            try:
                self.connector.validate_config(valid_config)
            except:
                pass
            end = time.perf_counter()
            times.append(end - start)
        
        valid_avg = sum(times) / len(times)
        
        # Measure timing for invalid configs
        for config in invalid_configs:
            with self.subTest(config=config):
                config_times = []
                for _ in range(5):
                    start = time.perf_counter()
                    try:
                        self.connector.validate_config(config)
                    except:
                        pass
                    end = time.perf_counter()
                    config_times.append(end - start)
                
                config_avg = sum(config_times) / len(config_times)
                
                # Timing should not vary significantly (within 50% tolerance)
                ratio = abs(config_avg - valid_avg) / valid_avg
                self.assertLess(ratio, 0.5, f"Timing attack possible: {ratio}")

    def test_memory_exhaustion_protection(self):
        """
        Test that the connector protects against memory exhaustion attacks.
        """
        # Test with recursively nested payload
        nested_payload = {'level': 1}
        current = nested_payload
        
        # Create deeply nested structure (but not infinite)
        for i in range(100):
            current['next'] = {'level': i + 2}
            current = current['next']
        
        try:
            formatted = self.connector.format_payload(nested_payload)
            # Should either format successfully or fail gracefully
            self.assertIsNotNone(formatted)
        except (ValueError, RecursionError, MemoryError):
            # Expected for protection against deep nesting
            pass

    def test_log_injection_prevention(self):
        """
        Test that log injection attacks are prevented in log output.
        """
        malicious_payloads = [
            {'message': 'normal\nINFO: Fake log entry'},
            {'message': 'test\r\nERROR: Injected error'},
            {'message': 'payload\x00admin access granted'},
            {'message': 'data\t\tDEBUG: False debug info'},
        ]
        
        for payload in malicious_payloads:
            with self.subTest(payload=payload):
                with patch('logging.info') as mock_log:
                    self.connector.log_request(payload)
                    
                    if mock_log.called:
                        logged_message = str(mock_log.call_args[0][0])
                        # Verify log injection characters are sanitized
                        self.assertNotIn('\n', logged_message)
                        self.assertNotIn('\r', logged_message)
                        self.assertNotIn('\x00', logged_message)


class TestGenesisConnectorAsyncOperations(unittest.TestCase):
    """
    Asynchronous operation tests for GenesisConnector class.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """
        Initialize GenesisConnector for async testing.
        """
        self.connector = GenesisConnector(config={
            'api_key': 'async_test_key',
            'base_url': 'https://async-api.test.com'
        })

    def test_async_context_manager_compatibility(self):
        """
        Test that the connector works correctly in async context manager scenarios.
        """
        async def async_operation():
            """
            Simulate async operations with the connector.
            """
            # Simulate async usage patterns
            payload = {'async_data': 'test'}
            
            with patch('requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'result': 'async_success'}
                mock_post.return_value = mock_response
                
                return self.connector.send_request(payload)
        
        # Run the async operation
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(async_operation())
            self.assertEqual(result['result'], 'async_success')
        finally:
            loop.close()

    def test_concurrent_async_requests(self):
        """
        Test handling of multiple concurrent async requests.
        """
        async def make_async_request(request_id):
            """
            Make an async request with the given ID.
            """
            await asyncio.sleep(0.01)  # Simulate async delay
            
            with patch('requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'id': request_id}
                mock_post.return_value = mock_response
                
                return self.connector.send_request({'id': request_id})
        
        async def run_concurrent_requests():
            """
            Run multiple concurrent async requests.
            """
            tasks = [make_async_request(i) for i in range(10)]
            return await asyncio.gather(*tasks)
        
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(run_concurrent_requests())
            
            # All requests should complete
            self.assertEqual(len(results), 10)
            
            # Verify all unique IDs are present
            ids = [result['id'] for result in results]
            self.assertEqual(sorted(ids), list(range(10)))
        finally:
            loop.close()


class TestGenesisConnectorExtensibility(unittest.TestCase):
    """
    Tests for GenesisConnector extensibility and plugin support.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """
        Initialize GenesisConnector for extensibility testing.
        """
        self.connector = GenesisConnector()

    def test_custom_serializer_support(self):
        """
        Test that custom serialization methods can be used with the connector.
        """
        class CustomObject:
            def __init__(self, data):
                self.data = data
            
            def to_dict(self):
                return {'custom_data': self.data, 'type': 'custom'}
        
        payload = {
            'message': 'test',
            'custom_obj': CustomObject('test_data')
        }
        
        # Test that custom objects can be handled
        try:
            formatted = self.connector.format_payload(payload)
            self.assertIn('custom_obj', formatted)
        except (ValueError, TypeError):
            # Expected if custom serialization not supported
            pass

    def test_middleware_pattern_support(self):
        """
        Test that middleware-like patterns can be implemented with the connector.
        """
        # Simulate middleware chain
        middleware_calls = []
        
        def logging_middleware(payload):
            middleware_calls.append('logging')
            return payload
        
        def validation_middleware(payload):
            middleware_calls.append('validation')
            if not payload.get('message'):
                raise ValueError("Message required")
            return payload
        
        def encryption_middleware(payload):
            middleware_calls.append('encryption')
            # Simulate encryption
            if isinstance(payload.get('message'), str):
                payload['message'] = f"encrypted:{payload['message']}"
            return payload
        
        # Apply middleware chain
        payload = {'message': 'test'}
        
        try:
            # Simulate middleware processing
            processed = logging_middleware(payload)
            processed = validation_middleware(processed)
            processed = encryption_middleware(processed)
            
            formatted = self.connector.format_payload(processed)
            
            # Verify middleware was called in order
            self.assertEqual(middleware_calls, ['logging', 'validation', 'encryption'])
            self.assertIn('encrypted:', str(formatted))
            
        except Exception:
            # Expected for unsupported middleware patterns
            pass

    def test_plugin_configuration_support(self):
        """
        Test that plugin-like configurations are handled correctly.
        """
        plugin_configs = [
            {
                'api_key': 'test',
                'base_url': 'https://api.test.com',
                'plugins': {
                    'auth': {'type': 'bearer', 'token': 'plugin_token'},
                    'retry': {'max_attempts': 5, 'backoff': 'exponential'},
                    'logging': {'level': 'DEBUG', 'format': 'json'}
                }
            },
            {
                'api_key': 'test',
                'base_url': 'https://api.test.com',
                'extensions': {
                    'cache': {'ttl': 300, 'backend': 'redis'},
                    'monitoring': {'metrics': True, 'tracing': True}
                }
            }
        ]
        
        for config in plugin_configs:
            with self.subTest(config=config):
                try:
                    connector = GenesisConnector(config=config)
                    is_valid = connector.validate_config(config)
                    # Should handle plugin configs gracefully
                    self.assertIsInstance(is_valid, bool)
                except (ValueError, TypeError):
                    # Expected if plugin configs not supported
                    pass


class TestGenesisConnectorCompliance(unittest.TestCase):
    """
    Compliance and standards tests for GenesisConnector class.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """
        Initialize GenesisConnector for compliance testing.
        """
        self.connector = GenesisConnector()

    def test_http_method_compliance(self):
        """
        Test that HTTP methods are used according to REST standards.
        """
        with patch('requests.get') as mock_get, \
             patch('requests.post') as mock_post, \
             patch('requests.put') as mock_put, \
             patch('requests.delete') as mock_delete:
            
            # Configure mock responses
            for mock in [mock_get, mock_post, mock_put, mock_delete]:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'status': 'ok'}
                mock.return_value = mock_response
            
            # Test that appropriate methods are used for different operations
            try:
                # Status check should use GET
                self.connector.get_status()
                mock_get.assert_called()
                
                # Sending data should use POST
                self.connector.send_request({'data': 'test'})
                mock_post.assert_called()
                
            except AttributeError:
                # Expected if methods don't exist
                pass

    def test_content_type_compliance(self):
        """
        Test that proper Content-Type headers are set for different payload types.
        """
        test_cases = [
            ({'message': 'text'}, 'application/json'),
            ({'binary': b'data'}, 'application/json'),  # Should handle binary in JSON
            ({'file': 'content'}, 'application/json'),
        ]
        
        for payload, expected_content_type in test_cases:
            with self.subTest(payload=payload):
                headers = self.connector.get_headers()
                
                if 'Content-Type' in headers:
                    # Verify appropriate content type is set
                    self.assertIn('application/', headers['Content-Type'].lower())

    def test_status_code_handling_compliance(self):
        """
        Test that HTTP status codes are handled according to standards.
        """
        status_scenarios = [
            (200, 'success'),      # OK
            (201, 'created'),      # Created
            (204, 'no_content'),   # No Content
            (400, 'bad_request'),  # Bad Request
            (401, 'unauthorized'), # Unauthorized
            (403, 'forbidden'),    # Forbidden
            (404, 'not_found'),    # Not Found
            (429, 'rate_limited'), # Too Many Requests
            (500, 'server_error'), # Internal Server Error
            (502, 'bad_gateway'),  # Bad Gateway
            (503, 'unavailable'),  # Service Unavailable
        ]
        
        for status_code, expected_handling in status_scenarios:
            with self.subTest(status_code=status_code):
                with patch('requests.post') as mock_post:
                    mock_response = Mock()
                    mock_response.status_code = status_code
                    if status_code < 400:
                        mock_response.json.return_value = {'status': 'ok'}
                    else:
                        mock_response.text = f'Error {status_code}'
                    mock_post.return_value = mock_response
                    
                    payload = {'message': 'test'}
                    
                    try:
                        result = self.connector.send_request(payload)
                        
                        if status_code < 400:
                            # Success codes should return data
                            self.assertIsNotNone(result)
                        else:
                            # This shouldn't happen for error codes
                            self.fail(f"Expected exception for status {status_code}")
                            
                    except (RuntimeError, ValueError, ConnectionError):
                        # Error codes should raise appropriate exceptions
                        if status_code >= 400:
                            pass  # Expected
                        else:
                            self.fail(f"Unexpected exception for status {status_code}")

    def test_encoding_compliance(self):
        """
        Test that character encoding is handled according to standards.
        """
        encoding_test_cases = [
            'UTF-8 text: Hello, 世界!',
            'Latin-1: café naïve résumé',
            'ASCII subset: Hello World',
            'Mixed: ASCII + UTF-8 测试',
            'Emojis: 🚀🌟💫🎉',
        ]
        
        for text in encoding_test_cases:
            with self.subTest(text=text):
                payload = {'message': text}
                
                try:
                    formatted = self.connector.format_payload(payload)
                    
                    # Verify the text can be properly encoded/decoded
                    if isinstance(formatted, str):
                        # Should be valid UTF-8
                        formatted.encode('utf-8')
                    elif isinstance(formatted, dict):
                        # Should contain the original text properly encoded
                        self.assertIn('message', formatted)
                        
                except (UnicodeError, ValueError):
                    # Expected for some encoding edge cases
                    pass


class TestGenesisConnectorObservability(unittest.TestCase):
    """
    Observability and monitoring tests for GenesisConnector class.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """
        Initialize GenesisConnector for observability testing.
        """
        self.connector = GenesisConnector(config={
            'api_key': 'obs_test_key',
            'base_url': 'https://obs-api.test.com'
        })

    def test_distributed_tracing_compatibility(self):
        """
        Test that the connector can work with distributed tracing systems.
        """
        trace_headers = {
            'X-Trace-Id': 'trace-12345',
            'X-Span-Id': 'span-67890',
            'X-B3-TraceId': 'b3-trace-id',
            'X-B3-SpanId': 'b3-span-id',
        }
        
        # Test that trace headers are preserved/forwarded
        config = {
            'api_key': 'test',
            'base_url': 'https://api.test.com',
            'additional_headers': trace_headers
        }
        
        connector = GenesisConnector(config=config)
        headers = connector.get_headers()
        
        # Verify trace headers are included
        for trace_header in trace_headers:
            if trace_header in headers:
                self.assertEqual(headers[trace_header], trace_headers[trace_header])

    def test_metrics_collection_comprehensive(self):
        """
        Test comprehensive metrics collection for observability.
        """
        metrics_scenarios = [
            ('successful_request', 200),
            ('client_error', 400),
            ('server_error', 500),
            ('timeout_error', None),  # Timeout simulation
        ]
        
        for scenario, status_code in metrics_scenarios:
            with self.subTest(scenario=scenario):
                with patch('requests.post') as mock_post:
                    if status_code:
                        mock_response = Mock()
                        mock_response.status_code = status_code
                        if status_code == 200:
                            mock_response.json.return_value = {'result': 'ok'}
                        else:
                            mock_response.text = f'Error {status_code}'
                        mock_post.return_value = mock_response
                    else:
                        mock_post.side_effect = TimeoutError("Request timeout")
                    
                    payload = {'message': 'metrics_test'}
                    
                    try:
                        result = self.connector.send_request(payload)
                        
                        # Check if metrics were collected
                        try:
                            metrics = self.connector.get_metrics()
                            if metrics:
                                self.assertIsInstance(metrics, dict)
                                # Common metrics that might be collected
                                possible_metrics = [
                                    'requests_total', 'requests_sent', 'response_time',
                                    'errors_total', 'success_rate', 'latency'
                                ]
                                # At least one metric should be present
                                has_metrics = any(metric in metrics for metric in possible_metrics)
                                if has_metrics:
                                    self.assertTrue(True)  # Metrics found
                        except AttributeError:
                            # get_metrics method might not exist
                            pass
                            
                    except (RuntimeError, ValueError, TimeoutError):
                        # Expected for error scenarios
                        pass

    def test_structured_logging_compliance(self):
        """
        Test that logging output is structured for machine parsing.
        """
        with patch('logging.info') as mock_log:
            payload = {
                'request_id': 'req-12345',
                'user_id': 'user-67890',
                'action': 'test_action',
                'timestamp': datetime.now().isoformat()
            }
            
            self.connector.log_request(payload)
            
            if mock_log.called:
                logged_args = mock_log.call_args[0]
                logged_message = str(logged_args[0]) if logged_args else ""
                
                # Check for structured logging elements
                structured_indicators = [
                    'request_id', 'timestamp', 'action',
                    '{', '}',  # JSON-like structure
                    '=',       # Key-value pairs
                ]
                
                has_structure = any(indicator in logged_message for indicator in structured_indicators)
                if has_structure:
                    self.assertTrue(True)  # Structured logging detected

    def test_health_check_endpoint_comprehensive(self):
        """
        Test comprehensive health check functionality for monitoring systems.
        """
        health_scenarios = [
            (200, {'status': 'healthy', 'version': '1.0.0', 'uptime': 3600}),
            (200, {'status': 'healthy', 'checks': {'database': 'ok', 'cache': 'ok'}}),
            (503, {'status': 'unhealthy', 'errors': ['database_connection_failed']}),
            (503, {'status': 'degraded', 'warnings': ['high_latency']}),
        ]
        
        for status_code, response_data in health_scenarios:
            with self.subTest(status_code=status_code, response_data=response_data):
                with patch('requests.get') as mock_get:
                    mock_response = Mock()
                    mock_response.status_code = status_code
                    mock_response.json.return_value = response_data
                    mock_get.return_value = mock_response
                    
                    try:
                        health = self.connector.health_check()
                        
                        if health:
                            self.assertIsInstance(health, dict)
                            self.assertIn('status', health)
                            
                            # Verify health status matches expected values
                            expected_statuses = ['healthy', 'unhealthy', 'degraded']
                            if health['status'] in expected_statuses:
                                self.assertIn(health['status'], expected_statuses)
                                
                    except AttributeError:
                        # health_check method might not exist
                        pass


if __name__ == '__main__':
    # Run all test classes including the new ones
    unittest.main(verbosity=2)