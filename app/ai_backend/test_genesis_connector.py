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
        Initializes a GenesisConnector instance and a mock configuration dictionary before each test.
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
        Performs cleanup after each test method. Override to reset state or resources if necessary.
        """
        # Reset any global state if needed
        pass

    def test_init_default_parameters(self):
        """Test GenesisConnector initialization with default parameters."""
        connector = GenesisConnector()
        self.assertIsNotNone(connector)
        self.assertIsInstance(connector, GenesisConnector)

    def test_init_with_config(self):
        """
        Test that GenesisConnector initializes correctly with a provided custom configuration.
        """
        connector = GenesisConnector(config=self.mock_config)
        self.assertIsNotNone(connector)
        self.assertEqual(connector.config, self.mock_config)

    def test_init_with_none_config(self):
        """
        Test that GenesisConnector initializes successfully when given a None configuration.
        """
        connector = GenesisConnector(config=None)
        self.assertIsNotNone(connector)

    def test_init_with_empty_config(self):
        """
        Test that GenesisConnector initializes successfully when provided with an empty configuration dictionary.
        """
        connector = GenesisConnector(config={})
        self.assertIsNotNone(connector)

    @patch('requests.get')
    def test_connect_success(self, mock_get):
        """
        Test that the connector successfully establishes a connection when the Genesis API responds with a 200 status and a valid JSON payload.
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
        Test that `connect()` returns False when the server responds with a 404 status code.
        """
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_failure_timeout(self, mock_get):
        """
        Test that `connect()` returns False when a timeout occurs during the connection attempt.
        """
        mock_get.side_effect = TimeoutError("Connection timeout")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_failure_connection_error(self, mock_get):
        """
        Test that `connect()` returns False when a connection error occurs during the HTTP request.
        """
        mock_get.side_effect = ConnectionError("Connection failed")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.post')
    def test_send_request_success(self, mock_post):
        """
        Test that send_request returns the expected response when the POST request succeeds.
        
        Asserts that a valid payload results in the correct JSON response and that the POST request is called once.
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
        Test that send_request raises RuntimeError when the server responds with an HTTP 500 error.
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
        Test that send_request raises ValueError when the server returns a malformed JSON response.
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
        Test that `validate_config` returns True for a valid configuration dictionary.
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
        Test that configuration validation raises a ValueError when the API key is missing from the config dictionary.
        """
        invalid_config = {
            'base_url': 'https://valid.url',
            'timeout': 30
        }
        
        with self.assertRaises(ValueError):
            self.connector.validate_config(invalid_config)

    def test_validate_config_invalid_url(self):
        """
        Test that configuration validation raises a ValueError when provided with an invalid base URL.
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
        Test that configuration validation raises a ValueError when the timeout is negative.
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
        Test that validating a None configuration input raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.validate_config(None)

    @patch('requests.get')
    def test_get_status_healthy(self, mock_get):
        """
        Test that get_status() returns the correct status and version when the service responds as healthy.
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
        Test that `get_status()` returns 'unhealthy' when the service responds with HTTP 503.
        """
        mock_response = Mock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response
        
        status = self.connector.get_status()
        
        self.assertEqual(status['status'], 'unhealthy')

    def test_format_payload_valid_data(self):
        """
        Tests that the payload is correctly formatted when valid data containing message, timestamp, and metadata is provided.
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
        Tests that the payload formatting correctly handles strings containing special characters and unicode.
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
        Verify that formatting an empty payload raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.format_payload({})

    def test_format_payload_none_data(self):
        """
        Verify that formatting a payload with None data raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.connector.format_payload(None)

    @patch('requests.post')
    def test_retry_mechanism_success_after_retry(self, mock_post):
        """
        Verify that the retry mechanism in `send_request_with_retry` succeeds after an initial failure by simulating a failed request followed by a successful one.
        
        Parameters:
            mock_post: Mocked `requests.post` method to control HTTP response behavior.
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
        Test that the retry mechanism raises a RuntimeError after exceeding the maximum number of retries.
        
        Verifies that `send_request_with_retry` attempts the correct number of retries when the server consistently returns an error, and that a `RuntimeError` is raised after the maximum retries are exhausted.
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
        Tests that the retry mechanism in `send_request_with_retry` applies increasing backoff delays between retries when requests fail.
        
        Asserts that `time.sleep` is called with incrementally increasing delays after each failed attempt and that a `RuntimeError` is raised after exceeding the maximum number of retries.
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
        Tests that parsing a valid JSON string with `parse_response()` returns the expected dictionary.
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
        Verify that logging a request with valid data triggers a logging call.
        """
        with patch('logging.info') as mock_log:
            payload = {'message': 'test'}
            self.connector.log_request(payload)
            
            mock_log.assert_called()

    def test_log_request_sensitive_data_redaction(self):
        """
        Verify that logging a request payload redacts sensitive data such as API keys and passwords before outputting to logs.
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
        Test that `get_headers()` returns headers including `Authorization` and `Content-Type` when an API key is provided in the configuration.
        """
        connector = GenesisConnector(config={'api_key': 'test_key'})
        headers = connector.get_headers()
        
        self.assertIn('Authorization', headers)
        self.assertIn('Content-Type', headers)

    def test_get_headers_without_auth(self):
        """
        Verify that the headers returned by `get_headers()` do not include an Authorization field when no API key is provided, but always include Content-Type.
        """
        connector = GenesisConnector(config={})
        headers = connector.get_headers()
        
        self.assertNotIn('Authorization', headers)
        self.assertIn('Content-Type', headers)

    def test_close_connection(self):
        """
        Verifies that closing the connector does not raise any exceptions.
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
        Verifies that multiple threads can concurrently validate the GenesisConnector configuration without errors or data races.
        """
        import threading
        results = []
        
        def worker():
            """
            Creates a `GenesisConnector` instance with the mock configuration and appends the result of its configuration validation to the `results` list.
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
        Test that the connector can format and handle large payloads without raising memory errors.
        
        Verifies that formatting a payload containing a large string and a large list completes successfully.
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
        Verifies that the connector can handle multiple concurrent requests successfully.
        
        Submits several requests in parallel threads and asserts that all complete without errors.
        """
        import concurrent.futures
        
        def make_request():
            """
            Send a test request using the connector with a mocked HTTP POST response.
            
            Returns:
                dict: The JSON response returned by the connector's send_request method.
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
        Tests that exceptions raised during a chained request operation are properly propagated by the connector.
        
        Asserts that when a network error occurs during a request, the exception is not suppressed and is raised to the caller.
        """
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Network error")
            
            payload = {'message': 'test'}
            
            with self.assertRaises(Exception):
                self.connector.send_request(payload)

    def test_configuration_reload(self):
        """
        Test that reloading the connector's configuration updates its internal config dictionary.
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
        Test that metrics are collected during a request operation.
        
        Verifies that after sending a request, the connector's metrics include both 'requests_sent' and 'response_time' keys.
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
        Tests that the health check endpoint returns a healthy status when the service responds with HTTP 200 and a healthy status payload.
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
        Test that the connector raises a RuntimeError when a rate-limited (HTTP 429) response is received from the server.
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
        Initializes the GenesisConnector instance with test configuration before each integration test.
        """
        self.connector = GenesisConnector(config={
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': 30
        })

    def test_full_request_lifecycle(self):
        """
        Tests the complete lifecycle of sending a request and receiving a response using the connector.
        
        Verifies that a request can be sent and a successful response is returned, with the HTTP POST method called exactly once.
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
        Tests that the connector can successfully establish a connection and send a request, verifying the expected response is returned.
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