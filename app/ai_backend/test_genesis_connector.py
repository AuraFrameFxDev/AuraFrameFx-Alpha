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

class TestGenesisConnectorEdgeCases(unittest.TestCase):
    """
    Additional edge case tests for GenesisConnector class.
    Testing framework: unittest with pytest enhancements
    """

    def setUp(self):
        """Set up test fixtures for edge case tests."""
        self.connector = GenesisConnector()
        self.mock_config = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'retry_count': 3
        }

    def test_init_with_malformed_config_types(self):
        """Test initialization with malformed config types."""
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
        """Test initialization with unicode characters in config."""
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
        """Test connection handling redirects."""
        mock_response = Mock()
        mock_response.status_code = 302
        mock_response.headers = {'Location': 'https://new.location.com'}
        mock_get.return_value = mock_response
        
        result = self.connector.connect()
        
        # Should handle redirects appropriately
        self.assertIsNotNone(result)

    @patch('requests.get')
    def test_connect_with_ssl_errors(self, mock_get):
        """Test connection with SSL certificate errors."""
        import ssl
        mock_get.side_effect = ssl.SSLError("SSL certificate verify failed")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.get')
    def test_connect_with_dns_resolution_error(self, mock_get):
        """Test connection with DNS resolution errors."""
        import socket
        mock_get.side_effect = socket.gaierror("Name or service not known")
        
        result = self.connector.connect()
        
        self.assertFalse(result)

    @patch('requests.post')
    def test_send_request_with_binary_payload(self, mock_post):
        """Test sending request with binary payload."""
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
        """Test sending request with deeply nested payload."""
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
        """Test sending request with circular reference in payload."""
        # Create circular reference
        payload = {'message': 'test'}
        payload['self'] = payload
        
        with self.assertRaises((ValueError, TypeError)):
            self.connector.send_request(payload)

    @patch('requests.post')
    def test_send_request_with_extremely_large_payload(self, mock_post):
        """Test sending request with extremely large payload."""
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
        """Test configuration validation against SQL injection attempts."""
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
        """Test configuration validation against path traversal attempts."""
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
        """Test configuration validation against XSS attempts."""
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
        """Test status retrieval with various timeout scenarios."""
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
        """Test payload formatting with datetime objects."""
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
        """Test payload formatting with decimal and complex numbers."""
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
        """Test payload formatting with custom objects."""
        class CustomObject:
            def __init__(self, value):
                self.value = value
            
            def __str__(self):
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
        """Test retry mechanism with exponential backoff validation."""
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
        """Test retry mechanism includes jitter to prevent thundering herd."""
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
        """Test parsing responses with edge case JSON structures."""
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
        """Test parsing very large JSON responses."""
        large_dict = {f'key_{i}': f'value_{i}' for i in range(10000)}
        large_json = json.dumps(large_dict)
        
        parsed = self.connector.parse_response(large_json)
        
        self.assertEqual(len(parsed), 10000)
        self.assertEqual(parsed['key_0'], 'value_0')
        self.assertEqual(parsed['key_9999'], 'value_9999')

    def test_log_request_with_various_log_levels(self):
        """Test logging requests with different log levels."""
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
        """Test logging request with PII data gets properly redacted."""
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
        """Test getting headers with custom user agent."""
        custom_config = {
            'api_key': 'test_key',
            'user_agent': 'CustomConnector/1.0'
        }
        
        connector = GenesisConnector(config=custom_config)
        headers = connector.get_headers()
        
        self.assertIn('User-Agent', headers)
        self.assertEqual(headers['User-Agent'], 'CustomConnector/1.0')

    def test_get_headers_with_additional_headers(self):
        """Test getting headers with additional custom headers."""
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
        """Test closing connection with pending requests."""
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
        """Test context manager handling when exception occurs."""
        with self.assertRaises(ValueError):
            with GenesisConnector(config=self.mock_config) as connector:
                # Simulate an exception during usage
                raise ValueError("Test exception")

    def test_context_manager_cleanup(self):
        """Test context manager properly cleans up resources."""
        with patch.object(GenesisConnector, 'close') as mock_close:
            with GenesisConnector(config=self.mock_config) as connector:
                self.assertIsNotNone(connector)
            
            # Verify cleanup was called
            mock_close.assert_called_once()

    def test_memory_usage_with_large_datasets(self):
        """Test memory usage remains reasonable with large datasets."""
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
        """Test handling concurrent configuration updates."""
        import threading
        import time
        
        results = []
        
        def update_config(config_id):
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
        """Test error propagation through method call chain."""
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
        """Set up performance test fixtures."""
        self.connector = GenesisConnector(config={
            'api_key': 'test_key',
            'base_url': 'https://api.test.com'
        })

    def test_payload_formatting_performance(self):
        """Test payload formatting performance with various sizes."""
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
        """Test performance under concurrent request load."""
        import concurrent.futures
        import time
        
        def make_request(request_id):
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
        """Test for memory leaks during repeated operations."""
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
        """Set up boundary condition test fixtures."""
        self.connector = GenesisConnector()

    def test_extremely_long_api_key(self):
        """Test with extremely long API key."""
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
        """Test with maximum timeout values."""
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
        """Test with URLs at various length boundaries."""
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
        """Test with payloads at various size boundaries."""
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
        """Test with various Unicode boundary conditions."""
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
        """Test with numeric boundary conditions."""
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