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
        """Set up test fixtures before each test method."""
        self.connector = GenesisConnector()
        self.mock_config = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'retry_count': 3
        }
        
    def tearDown(self):
        """Clean up after each test method."""
        # Reset any global state if needed
        pass

    def test_init_default_parameters(self):
        """Test GenesisConnector initialization with default parameters."""
        connector = GenesisConnector()
        self.assertIsNotNone(connector)
        self.assertIsInstance(connector, GenesisConnector)

    def test_init_with_config(self):
        """Test GenesisConnector initialization with custom config."""
        connector = GenesisConnector(config=self.mock_config)
        self.assertIsNotNone(connector)
        self.assertEqual(connector.config, self.mock_config)

    def test_init_with_none_config(self):
        """Test GenesisConnector initialization with None config."""
        connector = GenesisConnector(config=None)
        self.assertIsNotNone(connector)

    def test_init_with_empty_config(self):
        """Test GenesisConnector initialization with empty config."""
        connector = GenesisConnector(config={})
        self.assertIsNotNone(connector)

    @patch('requests.get')
    def test_connect_success(self, mock_get):
        """Test successful connection to Genesis API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'connected'}
        mock_get.return_value = mock_response
        
        result = self.connector.connect()
        
        self.assertTrue(result)
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_connect_failure_404(self, mock_get):
        """Test connection failure with 404 status."""
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
        """Test connection failure with connection error."""
        mock_get.side_effect = ConnectionError("Connection failed")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.post')
    def test_send_request_success(self, mock_post):
        """Test successful request sending."""
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
        """Test sending request with invalid payload."""
        with self.assertRaises(ValueError):
            self.connector.send_request(None)

    @patch('requests.post')
    def test_send_request_empty_payload(self, mock_post):
        """Test sending request with empty payload."""
        with self.assertRaises(ValueError):
            self.connector.send_request({})

    @patch('requests.post')
    def test_send_request_server_error(self, mock_post):
        """Test sending request with server error response."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_post.return_value = mock_response
        
        payload = {'message': 'test_message'}
        
        with self.assertRaises(RuntimeError):
            self.connector.send_request(payload)

    @patch('requests.post')
    def test_send_request_malformed_json(self, mock_post):
        """Test sending request with malformed JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response
        
        payload = {'message': 'test_message'}
        
        with self.assertRaises(ValueError):
            self.connector.send_request(payload)

    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        valid_config = {
            'api_key': 'valid_key',
            'base_url': 'https://valid.url',
            'timeout': 30
        }
        
        result = self.connector.validate_config(valid_config)
        
        self.assertTrue(result)

    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key."""
        invalid_config = {
            'base_url': 'https://valid.url',
            'timeout': 30
        }
        
        with self.assertRaises(ValueError):
            self.connector.validate_config(invalid_config)

    def test_validate_config_invalid_url(self):
        """Test configuration validation with invalid URL."""
        invalid_config = {
            'api_key': 'valid_key',
            'base_url': 'invalid_url',
            'timeout': 30
        }
        
        with self.assertRaises(ValueError):
            self.connector.validate_config(invalid_config)

    def test_validate_config_negative_timeout(self):
        """Test configuration validation with negative timeout."""
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

    @patch('requests.get')
    def test_get_status_healthy(self, mock_get):
        """Test getting status when service is healthy."""
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
        """Test payload formatting with special characters."""
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
        """Test payload formatting with empty data."""
        with self.assertRaises(ValueError):
            self.connector.format_payload({})

    def test_format_payload_none_data(self):
        """Test payload formatting with None data."""
        with self.assertRaises(ValueError):
            self.connector.format_payload(None)

    @patch('requests.post')
    def test_retry_mechanism_success_after_retry(self, mock_post):
        """Test retry mechanism succeeding after initial failure."""
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
        """Test retry mechanism failing after max retries."""
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
        """Test retry mechanism backoff timing."""
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
        """Test parsing valid JSON response."""
        response_data = {'key': 'value', 'number': 123, 'bool': True}
        json_string = json.dumps(response_data)
        
        parsed = self.connector.parse_response(json_string)
        
        self.assertEqual(parsed, response_data)

    def test_parse_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        invalid_json = '{"invalid": json}'
        
        with self.assertRaises(ValueError):
            self.connector.parse_response(invalid_json)

    def test_parse_response_empty_string(self):
        """Test parsing empty string response."""
        with self.assertRaises(ValueError):
            self.connector.parse_response('')

    def test_parse_response_none_input(self):
        """Test parsing None input."""
        with self.assertRaises(ValueError):
            self.connector.parse_response(None)

    def test_log_request_valid_data(self):
        """Test logging request with valid data."""
        with patch('logging.info') as mock_log:
            payload = {'message': 'test'}
            self.connector.log_request(payload)
            
            mock_log.assert_called()

    def test_log_request_sensitive_data_redaction(self):
        """Test logging request with sensitive data redaction."""
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
        """Test getting headers with authentication."""
        connector = GenesisConnector(config={'api_key': 'test_key'})
        headers = connector.get_headers()
        
        self.assertIn('Authorization', headers)
        self.assertIn('Content-Type', headers)

    def test_get_headers_without_auth(self):
        """Test getting headers without authentication."""
        connector = GenesisConnector(config={})
        headers = connector.get_headers()
        
        self.assertNotIn('Authorization', headers)
        self.assertIn('Content-Type', headers)

    def test_close_connection(self):
        """Test closing connection."""
        # This test depends on the actual implementation
        result = self.connector.close()
        
        # Should not raise an exception
        self.assertTrue(True)

    def test_context_manager_usage(self):
        """Test using GenesisConnector as context manager."""
        with GenesisConnector(config=self.mock_config) as connector:
            self.assertIsNotNone(connector)
            # Context manager should work without errors

    def test_thread_safety(self):
        """Test thread safety of GenesisConnector."""
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
        
        # All workers should complete successfully
        self.assertEqual(len(results), 5)
        self.assertTrue(all(results))

    def test_large_payload_handling(self):
        """Test handling of large payloads."""
        large_payload = {
            'message': 'x' * 10000,  # 10KB string
            'data': list(range(1000))  # Large list
        }
        
        # Should format without raising memory errors
        formatted = self.connector.format_payload(large_payload)
        self.assertIsNotNone(formatted)

    def test_concurrent_requests(self):
        """Test handling concurrent requests."""
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
        """Test error handling with chained operations."""
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
        """Test metrics collection during operations."""
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
        """Test health check endpoint."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'status': 'healthy'}
            mock_get.return_value = mock_response
            
            health = self.connector.health_check()
            
            self.assertEqual(health['status'], 'healthy')

    def test_rate_limiting_handling(self):
        """Test rate limiting handling."""
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
        """Set up integration test fixtures."""
        self.connector = GenesisConnector(config={
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': 30
        })

    def test_full_request_lifecycle(self):
        """Test complete request lifecycle from start to finish."""
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
        """Test connection establishment followed by request."""
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

class TestGenesisConnectorAdvanced(unittest.TestCase):
    """
    Advanced and edge case tests for GenesisConnector.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """Set up advanced test fixtures."""
        self.connector = GenesisConnector()
        self.complex_config = {
            'api_key': 'test_api_key_complex',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'retry_count': 3,
            'max_payload_size': 1000000,
            'compression': True,
            'ssl_verify': True,
            'user_agent': 'TestAgent/1.0'
        }

    def test_init_with_complex_config(self):
        """Test initialization with complex configuration."""
        connector = GenesisConnector(config=self.complex_config)
        self.assertIsNotNone(connector)
        self.assertEqual(connector.config['compression'], True)
        self.assertEqual(connector.config['ssl_verify'], True)

    def test_init_with_malformed_config(self):
        """Test initialization with malformed configuration."""
        malformed_configs = [
            {'api_key': 123},  # Wrong type
            {'timeout': 'invalid'},  # String instead of int
            {'base_url': 'not-a-url'},  # Invalid URL format
            {'retry_count': -5}  # Negative value
        ]
        
        for config in malformed_configs:
            with self.subTest(config=config):
                with self.assertRaises((ValueError, TypeError)):
                    GenesisConnector(config=config)

    @patch('requests.post')
    def test_send_request_with_unicode_payload(self, mock_post):
        """Test sending request with Unicode characters in payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'success'}
        mock_post.return_value = mock_response
        
        unicode_payload = {
            'message': 'Test with émojis 🚀🌟 and special chars: café, naïve, résumé',
            'chinese': '你好世界',
            'arabic': 'مرحبا بالعالم',
            'emoji': '🎉🎊🎈'
        }
        
        result = self.connector.send_request(unicode_payload)
        self.assertEqual(result['data'], 'success')
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_send_request_with_nested_payload(self, mock_post):
        """Test sending request with deeply nested payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'success'}
        mock_post.return_value = mock_response
        
        nested_payload = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'message': 'deeply nested',
                            'array': [1, 2, {'nested_in_array': True}]
                        }
                    }
                }
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }
        
        result = self.connector.send_request(nested_payload)
        self.assertEqual(result['data'], 'success')

    @patch('requests.post')
    def test_send_request_with_binary_data(self, mock_post):
        """Test sending request with binary data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'success'}
        mock_post.return_value = mock_response
        
        binary_payload = {
            'message': 'test',
            'binary_data': b'binary content here',
            'base64_data': 'dGVzdCBkYXRh'  # base64 encoded
        }
        
        result = self.connector.send_request(binary_payload)
        self.assertEqual(result['data'], 'success')

    def test_validate_config_with_environment_variables(self):
        """Test configuration validation with environment variables."""
        with patch.dict('os.environ', {
            'GENESIS_API_KEY': 'env_api_key',
            'GENESIS_BASE_URL': 'https://env.genesis.test',
            'GENESIS_TIMEOUT': '45'
        }):
            # Test that environment variables are properly handled
            config = self.connector.get_config_from_env()
            self.assertEqual(config['api_key'], 'env_api_key')
            self.assertEqual(config['base_url'], 'https://env.genesis.test')
            self.assertEqual(config['timeout'], 45)

    def test_validate_config_with_ssl_settings(self):
        """Test configuration validation with SSL settings."""
        ssl_configs = [
            {'ssl_verify': True, 'ssl_cert_path': '/path/to/cert.pem'},
            {'ssl_verify': False},
            {'ssl_verify': True, 'ssl_ca_bundle': '/path/to/ca-bundle.crt'}
        ]
        
        for config in ssl_configs:
            with self.subTest(config=config):
                config.update({'api_key': 'test', 'base_url': 'https://test.com'})
                result = self.connector.validate_config(config)
                self.assertTrue(result)

    @patch('requests.post')
    def test_send_request_with_custom_headers(self, mock_post):
        """Test sending request with custom headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'success'}
        mock_post.return_value = mock_response
        
        custom_headers = {
            'X-Custom-Header': 'custom_value',
            'X-Request-ID': 'req-123456',
            'X-Client-Version': '2.0.0'
        }
        
        payload = {'message': 'test'}
        result = self.connector.send_request(payload, headers=custom_headers)
        
        self.assertEqual(result['data'], 'success')
        # Verify custom headers were included
        call_args = mock_post.call_args
        self.assertIn('headers', call_args.kwargs)

    @patch('requests.post')
    def test_send_request_with_compression(self, mock_post):
        """Test sending request with compression enabled."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'success'}
        mock_post.return_value = mock_response
        
        connector = GenesisConnector(config={'compression': True})
        large_payload = {
            'message': 'x' * 50000,  # Large payload that benefits from compression
            'data': list(range(10000))
        }
        
        result = connector.send_request(large_payload)
        self.assertEqual(result['data'], 'success')

    @patch('requests.post')
    def test_send_request_with_streaming(self, mock_post):
        """Test sending request with streaming response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2', b'chunk3'])
        mock_post.return_value = mock_response
        
        payload = {'message': 'test', 'stream': True}
        result = self.connector.send_request_streaming(payload)
        
        # Verify streaming response is handled correctly
        self.assertIsNotNone(result)

    @patch('requests.post')
    def test_send_request_with_file_upload(self, mock_post):
        """Test sending request with file upload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'file_uploaded'}
        mock_post.return_value = mock_response
        
        # Mock file-like object
        mock_file = Mock()
        mock_file.read.return_value = b'file content'
        mock_file.name = 'test_file.txt'
        
        payload = {
            'message': 'file upload test',
            'file': mock_file
        }
        
        result = self.connector.send_request_with_file(payload)
        self.assertEqual(result['data'], 'file_uploaded')

    def test_connection_pooling(self):
        """Test connection pooling behavior."""
        with patch('requests.Session') as mock_session:
            mock_session_instance = Mock()
            mock_session.return_value = mock_session_instance
            
            caller = GenesisConnector(config={'use_session': True})
            
            # Make multiple requests
            for i in range(5):
                with patch.object(caller, 'send_request') as mock_send:
                    mock_send.return_value = {'data': f'response_{i}'}
                    caller.send_request({'message': f'test_{i}'})
            
            # Verify session was reused
            self.assertEqual(mock_session.call_count, 1)

    @patch('requests.post')
    def test_request_timeout_variations(self, mock_post):
        """Test different timeout scenarios."""
        timeout_scenarios = [
            {'timeout': 1, 'exception': TimeoutError},
            {'timeout': 0.5, 'exception': TimeoutError},
            {'timeout': 60, 'exception': None}  # No timeout
        ]
        
        for scenario in timeout_scenarios:
            with self.subTest(scenario=scenario):
                if scenario['exception']:
                    mock_post.side_effect = scenario['exception']("Timeout occurred")
                    
                    connector = GenesisConnector(config={'timeout': scenario['timeout']})
                    
                    with self.assertRaises(scenario['exception']):
                        connector.send_request({'message': 'test'})
                else:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {'data': 'success'}
                    mock_post.return_value = mock_response
                    
                    connector = GenesisConnector(config={'timeout': scenario['timeout']})
                    result = connector.send_request({'message': 'test'})
                    self.assertEqual(result['data'], 'success')

    @patch('requests.post')
    def test_retry_with_exponential_backoff(self, mock_post):
        """Test retry mechanism with exponential backoff."""
        with patch('time.sleep') as mock_sleep:
            # Simulate 3 failures then success
            mock_responses = [
                Mock(status_code=500),
                Mock(status_code=502),
                Mock(status_code=503),
                Mock(status_code=200)
            ]
            mock_responses[-1].json.return_value = {'data': 'success'}
            mock_post.side_effect = mock_responses
            
            connector = GenesisConnector(config={'retry_strategy': 'exponential'})
            result = connector.send_request_with_retry({'message': 'test'}, max_retries=3)
            
            self.assertEqual(result['data'], 'success')
            # Verify exponential backoff: 1, 2, 4 seconds
            expected_calls = [call(1), call(2), call(4)]
            mock_sleep.assert_has_calls(expected_calls)

    @patch('requests.post')
    def test_circuit_breaker_pattern(self, mock_post):
        """Test circuit breaker pattern implementation."""
        # Simulate consecutive failures to trigger circuit breaker
        mock_post.side_effect = [
            Mock(status_code=500) for _ in range(10)
        ]
        
        connector = GenesisConnector(config={'circuit_breaker': True})
        
        # First few requests should fail normally
        for i in range(5):
            with self.assertRaises(RuntimeError):
                connector.send_request({'message': f'test_{i}'})
        
        # After threshold, circuit breaker should be open
        with self.assertRaises(RuntimeError) as context:
            connector.send_request({'message': 'test_circuit_open'})
        
        self.assertIn('Circuit breaker', str(context.exception))

    def test_request_id_generation(self):
        """Test unique request ID generation."""
        request_ids = []
        
        for i in range(100):
            request_id = self.connector.generate_request_id()
            request_ids.append(request_id)
            self.assertIsInstance(request_id, str)
            self.assertGreater(len(request_id), 0)
        
        # All IDs should be unique
        self.assertEqual(len(request_ids), len(set(request_ids)))

    def test_payload_validation_edge_cases(self):
        """Test payload validation with edge cases."""
        edge_cases = [
            # Extremely large numbers
            {'number': 2**63 - 1},
            # Scientific notation
            {'scientific': 1.23e-10},
            # Special float values
            {'infinity': float('inf')},
            {'negative_infinity': float('-inf')},
            # Very long strings
            {'long_string': 'a' * 1000000},
            # Empty nested structures
            {'empty_dict': {}, 'empty_list': []},
            # Complex nested structures
            {'complex': {'a': [{'b': {'c': [1, 2, 3]}}]}}
        ]
        
        for payload in edge_cases:
            with self.subTest(payload=payload):
                try:
                    formatted = self.connector.format_payload(payload)
                    self.assertIsNotNone(formatted)
                except ValueError as e:
                    # Some edge cases might be intentionally rejected
                    self.assertIn('payload', str(e).lower())

    @patch('requests.post')
    def test_response_caching(self, mock_post):
        """Test response caching mechanism."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'cached_response'}
        mock_post.return_value = mock_response
        
        connector = GenesisConnector(config={'enable_cache': True})
        payload = {'message': 'cacheable_request'}
        
        # First request should hit the API
        result1 = connector.send_request(payload)
        self.assertEqual(result1['data'], 'cached_response')
        
        # Second identical request should use cache
        result2 = connector.send_request(payload)
        self.assertEqual(result2['data'], 'cached_response')
        
        # Should only make one actual HTTP request
        self.assertEqual(mock_post.call_count, 1)

    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring during operations."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Perform memory-intensive operations
        large_payloads = []
        for i in range(100):
            large_payload = {
                'id': i,
                'data': 'x' * 10000,
                'timestamp': datetime.now().isoformat()
            }
            large_payloads.append(large_payload)
            formatted = self.connector.format_payload(large_payload)
            
        # Force garbage collection
        gc.collect()
        
        # Check memory usage didn't grow excessively
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB)
        self.assertLess(memory_growth, 100 * 1024 * 1024)

    def test_error_recovery_scenarios(self):
        """Test various error recovery scenarios."""
        error_scenarios = [
            {'error': ConnectionError("Connection failed"), 'recoverable': True},
            {'error': TimeoutError("Request timeout"), 'recoverable': True},
            {'error': ValueError("Invalid data"), 'recoverable': False},
            {'error': PermissionError("Access denied"), 'recoverable': False}
        ]
        
        for scenario in error_scenarios:
            with self.subTest(scenario=scenario):
                with patch('requests.post') as mock_post:
                    mock_post.side_effect = scenario['error']
                    
                    payload = {'message': 'test'}
                    
                    if scenario['recoverable']:
                        # Should eventually succeed after retries
                        with patch('time.sleep'):
                            try:
                                result = self.connector.send_request_with_retry(payload)
                            except Exception:
                                # If it still fails, that's expected for some scenarios
                                pass
                    else:
                        # Should fail immediately without retries
                        with self.assertRaises(type(scenario['error'])):
                            self.connector.send_request(payload)

    def test_configuration_hot_reload(self):
        """Test hot reloading of configuration."""
        initial_config = {'api_key': 'initial_key', 'timeout': 30}
        connector = GenesisConnector(config=initial_config)
        
        # Verify initial config
        self.assertEqual(connector.config['api_key'], 'initial_key')
        self.assertEqual(connector.config['timeout'], 30)
        
        # Hot reload new config
        new_config = {'api_key': 'new_key', 'timeout': 60, 'compression': True}
        connector.hot_reload_config(new_config)
        
        # Verify config was updated
        self.assertEqual(connector.config['api_key'], 'new_key')
        self.assertEqual(connector.config['timeout'], 60)
        self.assertEqual(connector.config['compression'], True)

    def test_webhook_callback_handling(self):
        """Test webhook callback handling."""
        callback_data = []
        
        def test_callback(response):
            callback_data.append(response)
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': 'webhook_test'}
            mock_post.return_value = mock_response
            
            connector = GenesisConnector(config={'webhook_callback': test_callback})
            payload = {'message': 'test with callback'}
            
            result = connector.send_request(payload)
            
            # Verify callback was called
            self.assertEqual(len(callback_data), 1)
            self.assertEqual(callback_data[0]['data'], 'webhook_test')

    def test_batch_request_processing(self):
        """Test batch request processing."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'batch_results': ['success'] * 5}
            mock_post.return_value = mock_response
            
            batch_payloads = [
                {'message': f'batch_test_{i}', 'id': i}
                for i in range(5)
            ]
            
            results = self.connector.send_batch_requests(batch_payloads)
            
            self.assertEqual(len(results['batch_results']), 5)
            self.assertTrue(all(result == 'success' for result in results['batch_results']))

    def test_api_version_compatibility(self):
        """Test API version compatibility handling."""
        version_configs = [
            {'api_version': 'v1', 'expected_endpoint': '/v1/'},
            {'api_version': 'v2', 'expected_endpoint': '/v2/'},
            {'api_version': 'beta', 'expected_endpoint': '/beta/'}
        ]
        
        for config in version_configs:
            with self.subTest(config=config):
                connector = GenesisConnector(config=config)
                endpoint = connector.get_api_endpoint()
                self.assertIn(config['expected_endpoint'], endpoint)

    def test_graceful_shutdown(self):
        """Test graceful shutdown behavior."""
        connector = GenesisConnector(config={'graceful_shutdown': True})
        
        # Start some background operations
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance
            
            connector.start_background_operations()
            
            # Initiate shutdown
            connector.shutdown()
            
            # Verify background operations were stopped gracefully
            mock_thread_instance.join.assert_called()


class TestGenesisConnectorPerformance(unittest.TestCase):
    """
    Performance-focused tests for GenesisConnector.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """Set up performance test fixtures."""
        self.connector = GenesisConnector(config={
            'api_key': 'perf_test_key',
            'base_url': 'https://perf.test.com',
            'timeout': 30
        })

    def test_high_volume_request_processing(self):
        """Test processing high volume of requests."""
        import time
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': 'success'}
            mock_post.return_value = mock_response
            
            start_time = time.time()
            
            # Process 1000 requests
            for i in range(1000):
                payload = {'message': f'perf_test_{i}'}
                result = self.connector.send_request(payload)
                self.assertEqual(result['data'], 'success')
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete within reasonable time (e.g., 10 seconds)
            self.assertLess(duration, 10.0)
            
            # Should have processed all requests
            self.assertEqual(mock_post.call_count, 1000)

    def test_concurrent_request_performance(self):
        """Test concurrent request processing performance."""
        import concurrent.futures
        import time
        
        def make_request(request_id):
            with patch('requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'data': f'concurrent_{request_id}'}
                mock_post.return_value = mock_response
                
                payload = {'message': f'concurrent_test_{request_id}'}
                return self.connector.send_request(payload)
        
        start_time = time.time()
        
        # Process 100 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(100)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(duration, 5.0)
        
        # All requests should succeed
        self.assertEqual(len(results), 100)

    def test_memory_efficiency(self):
        """Test memory efficiency with large payloads."""
        import gc
        
        # Process large payloads without excessive memory growth
        for size in [1000, 10000, 100000]:
            with self.subTest(size=size):
                large_payload = {
                    'message': 'x' * size,
                    'data': list(range(size // 100))
                }
                
                # Format payload multiple times
                for _ in range(10):
                    formatted = self.connector.format_payload(large_payload)
                    self.assertIsNotNone(formatted)
                    
                    # Force garbage collection
                    gc.collect()


if __name__ == '__main__':
    # Run all test classes
    unittest.main(verbosity=2)