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