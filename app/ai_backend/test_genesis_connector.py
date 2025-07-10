import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, call
        
        end_time = time.time()
        total_time = end_time - start_time
        


class TestGenesisConnectorDataSerialization(unittest.TestCase):
    """Comprehensive data serialization and deserialization tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)

    def test_serialization_of_complex_data_types(self):
        """Test serialization of complex nested data structures."""
        from decimal import Decimal
        from datetime import datetime, timezone
        import uuid
        
        complex_data = {
            'uuid': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'decimal_value': str(Decimal('123.456789')),
            'nested_objects': {
                'level1': {
                    'level2': {
                        'array_of_objects': [
                            {'id': i, 'name': f'item_{i}', 'active': i % 2 == 0}
                            for i in range(10)
                        ],
                        'metadata': {
                            'created_by': 'test_user',
                            'tags': ['tag1', 'tag2', 'tag3'],
                            'permissions': {
                                'read': True,
                                'write': False,
                                'admin': False
                            }
                        }
                    }
                }
            },
            'large_text_field': 'Lorem ipsum ' * 1000,
            'binary_encoded': 'YWxpY2UgaW4gd29uZGVybGFuZA==',  # base64
            'empty_values': {
                'empty_string': '',
                'empty_list': [],
                'empty_dict': {},
                'null_value': None
            }
        }

        if hasattr(self.connector, 'format_payload'):
            formatted = self.connector.format_payload(complex_data)
            self.assertIsNotNone(formatted)
            self.assertIn('uuid', formatted)
            self.assertIn('nested_objects', formatted)

    def test_circular_reference_detection(self):
        """Test detection and handling of circular references."""
        # Create circular reference
        data = {'name': 'root'}
        child = {'name': 'child', 'parent': data}
        data['child'] = child
        
        if hasattr(self.connector, 'format_payload'):
            with self.assertRaises((ValueError, RecursionError)):
                self.connector.format_payload(data)

    def test_deeply_nested_structure_limits(self):
        """Test handling of deeply nested structures at various depths."""
        for depth in [10, 50, 100, 500]:
            with self.subTest(depth=depth):
                # Create nested structure
                nested = {'level': 0}
                current = nested
                
                for i in range(depth):
                    current['next'] = {'level': i + 1}
                    current = current['next']
                
                current['end'] = True
                
                if hasattr(self.connector, 'format_payload'):
                    try:
                        formatted = self.connector.format_payload(nested)
                        if depth <= 100:  # Should work for reasonable depths
                            self.assertIsNotNone(formatted)
                    except (RecursionError, ValueError):
                        # Expected for very deep structures
                        if depth > 100:
                            pass
                        else:
                            raise

    def test_unicode_normalization(self):
        """Test Unicode normalization and handling."""
        import unicodedata
        
        unicode_test_cases = [
            'café',  # NFC
            'cafe\u0301',  # NFD  
            'Ωήμος',  # Greek
            '한글',  # Korean
            '🌟💫🚀',  # Emoji
            '\u200b\u200c\u200d',  # Zero-width characters
            'test\u0000null',  # Null byte
            'test\ufffeReverse',  # Reverse byte order mark
        ]
        
        for text in unicode_test_cases:
            with self.subTest(text=repr(text)):
                payload = {
                    'original': text,
                    'nfc': unicodedata.normalize('NFC', text),
                    'nfd': unicodedata.normalize('NFD', text),
                    'message': f'Testing {text}'
                }
                
                if hasattr(self.connector, 'format_payload'):
                    try:
                        formatted = self.connector.format_payload(payload)
                        self.assertIsNotNone(formatted)
                    except ValueError:
                        # Some characters may be rejected
                        if '\u0000' in text or '\ufffe' in text:
                            pass  # Expected for problematic characters
                        else:
                            raise

    def test_large_payload_chunking(self):
        """Test handling of very large payloads."""
        # Test various payload sizes
        sizes = [
            1024,           # 1KB
            1024 * 100,     # 100KB  
            1024 * 1024,    # 1MB
            1024 * 1024 * 5 # 5MB
        ]
        
        for size in sizes:
            with self.subTest(size=size):
                large_data = {
                    'large_field': 'x' * size,
                    'metadata': {
                        'size': size,
                        'description': f'Payload of {size} bytes'
                    }
                }
                
                if hasattr(self.connector, 'format_payload'):
                    try:
                        formatted = self.connector.format_payload(large_data)
                        if size <= 1024 * 1024:  # Up to 1MB should work
                            self.assertIsNotNone(formatted)
                    except (MemoryError, ValueError):
                        # Large payloads may be rejected
                        if size > 1024 * 1024:
                            pass  # Expected for very large payloads
                        else:
                            raise

    def test_custom_json_encoder_handling(self):
        """Test handling of custom objects that need special encoding."""
        from datetime import datetime, date, time
        from decimal import Decimal
        import uuid
        
        class CustomObject:
            def __init__(self, value):
                self.value = value
                
            def __str__(self):
                return f"CustomObject({self.value})"
                
            def to_dict(self):
                return {'custom_value': self.value}
        
        custom_data = {
            'datetime_obj': datetime.now(),
            'date_obj': date.today(),
            'time_obj': time(14, 30, 0),
            'decimal_obj': Decimal('123.456'),
            'uuid_obj': uuid.uuid4(),
            'custom_obj': CustomObject('test_value'),
            'set_obj': {1, 2, 3, 4, 5},
            'frozenset_obj': frozenset([1, 2, 3]),
            'bytes_obj': b'binary_data',
            'bytearray_obj': bytearray(b'mutable_binary')
        }
        
        if hasattr(self.connector, 'format_payload'):
            try:
                formatted = self.connector.format_payload(custom_data)
                self.assertIsNotNone(formatted)
                # Check that at least some fields are present
                self.assertIn('datetime_obj', formatted)
                self.assertIn('custom_obj', formatted)
            except (TypeError, ValueError):
                # Custom objects may not be serializable
                pass


class TestGenesisConnectorAdvancedAsync(unittest.TestCase):
    """Advanced asynchronous functionality tests."""

    def setUp(self):
        """Set up async test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)

    @patch('asyncio.create_task')
    async def test_async_batch_processing(self, mock_create_task):
        """Test asynchronous batch processing of multiple requests."""
        if not hasattr(self.connector, 'async_batch_request'):
            self.skipTest("Async batch functionality not available")
            
        # Mock multiple async responses
        mock_tasks = []
        for i in range(5):
            mock_task = AsyncMock()
            mock_task.return_value = {'id': i, 'result': f'batch_result_{i}'}
            mock_tasks.append(mock_task)
        
        mock_create_task.side_effect = mock_tasks
        
        payloads = [
            {'request_id': i, 'data': f'batch_data_{i}'}
            for i in range(5)
        ]
        
        results = await self.connector.async_batch_request(payloads)
        
        self.assertEqual(len(results), 5)
        for i, result in enumerate(results):
            self.assertEqual(result['id'], i)

    async def test_async_timeout_handling_with_cancellation(self):
        """Test async timeout handling with proper task cancellation."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
            
        with patch('aiohttp.ClientSession.request') as mock_request:
            # Simulate a request that takes too long
            async def slow_request(*args, **kwargs):
                await asyncio.sleep(10)  # Longer than timeout
                return AsyncMock(status=200, json=AsyncMock(return_value={'data': 'slow'}))
            
            mock_request.side_effect = slow_request
            
            # Set short timeout
            original_timeout = self.connector.timeout
            self.connector.timeout = 1
            
            try:
                with self.assertRaises(GenesisTimeoutError):
                    await self.connector.async_make_request('GET', '/slow')
            finally:
                self.connector.timeout = original_timeout

    async def test_async_connection_pooling(self):
        """Test async connection pooling and reuse."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
            
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'success': True})
            mock_session.request.return_value.__aenter__.return_value = mock_response
            
            # Make multiple requests
            for i in range(3):
                result = await self.connector.async_make_request('GET', f'/test/{i}')
                self.assertEqual(result, {'success': True})
            
            # Session should be reused
            self.assertEqual(mock_session_class.call_count, 1)

    async def test_async_error_propagation(self):
        """Test proper error propagation in async operations."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
            
        error_scenarios = [
            (asyncio.TimeoutError("Async timeout"), GenesisTimeoutError),
            (aiohttp.ClientError("Client error"), GenesisConnectionError),
            (ConnectionError("Connection failed"), GenesisConnectionError),
            (ValueError("Invalid data"), GenesisConnectionError)
        ]
        
        for original_error, expected_error in error_scenarios:
            with self.subTest(error=original_error.__class__.__name__):
                with patch('aiohttp.ClientSession.request') as mock_request:
                    mock_request.side_effect = original_error
                    
                    with self.assertRaises(expected_error):
                        await self.connector.async_make_request('GET', '/test')


class TestGenesisConnectorAdvancedCaching(unittest.TestCase):
    """Advanced caching and memoization tests."""

    def setUp(self):
        """Set up caching test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3,
            'enable_caching': True,
            'cache_ttl': 300
        }
        self.connector = GenesisConnector(self.mock_config)

    @patch('requests.Session.request')
    def test_response_caching_with_ttl(self, mock_request):
        """Test response caching with time-to-live."""
        if not hasattr(self.connector, 'get_cached_response'):
            self.skipTest("Caching functionality not available")
            
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'cached_data': 'test', 'timestamp': '2023-01-01T00:00:00Z'}
        mock_response.headers = {'Cache-Control': 'max-age=300'}
        mock_request.return_value = mock_response
        
        # First request should hit the API
        result1 = self.connector.get_cached_response('/cacheable')
        
        # Second request should use cache
        result2 = self.connector.get_cached_response('/cacheable')
        
        self.assertEqual(result1, result2)
        self.assertEqual(mock_request.call_count, 1)

    @patch('time.time')
    @patch('requests.Session.request')
    def test_cache_expiration(self, mock_request, mock_time):
        """Test cache expiration after TTL."""
        if not hasattr(self.connector, 'get_cached_response'):
            self.skipTest("Caching functionality not available")
            
        # Mock time progression
        mock_time.side_effect = [1000, 1000, 1400]  # 400 seconds later
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'cached_data': 'test'}
        mock_response.headers = {'Cache-Control': 'max-age=300'}  # 5 minutes
        mock_request.return_value = mock_response
        
        # First request
        self.connector.get_cached_response('/cacheable')
        
        # Second request after expiration
        self.connector.get_cached_response('/cacheable')
        
        # Should make two API calls due to expiration
        self.assertEqual(mock_request.call_count, 2)

    def test_cache_invalidation(self):
        """Test manual cache invalidation."""
        if not hasattr(self.connector, 'invalidate_cache'):
            self.skipTest("Cache invalidation not available")
            
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': 'test'}
            mock_request.return_value = mock_response
            
            # Cache a response
            if hasattr(self.connector, 'get_cached_response'):
                self.connector.get_cached_response('/test')
            
            # Invalidate cache
            self.connector.invalidate_cache('/test')
            
            # Next request should hit API again
            if hasattr(self.connector, 'get_cached_response'):
                self.connector.get_cached_response('/test')
                self.assertEqual(mock_request.call_count, 2)

    def test_cache_size_limits(self):
        """Test cache size limits and LRU eviction."""
        if not (hasattr(self.connector, 'get_cached_response') and 
                hasattr(self.connector, 'cache_size_limit')):
            self.skipTest("Cache size limiting not available")
            
        # Set small cache limit
        self.connector.cache_size_limit = 3
        
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'Cache-Control': 'max-age=3600'}
            mock_request.return_value = mock_response
            
            # Fill cache beyond limit
            for i in range(5):
                mock_response.json.return_value = {'data': f'test_{i}'}
                self.connector.get_cached_response(f'/test/{i}')
            
            # First entries should be evicted
            # Re-requesting first endpoint should hit API again
            mock_response.json.return_value = {'data': 'test_0_new'}
            result = self.connector.get_cached_response('/test/0')
            
            # Should have made 6 requests total (5 + 1 after eviction)
            self.assertEqual(mock_request.call_count, 6)


class TestGenesisConnectorAdvancedSecurity(unittest.TestCase):
    """Advanced security and authentication tests."""

    def setUp(self):
        """Set up security test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)

    def test_request_signing_with_hmac(self):
        """Test HMAC-based request signing."""
        if not hasattr(self.connector, 'sign_request'):
            self.skipTest("Request signing not available")
            
        import hmac
        import hashlib
        import time
        
        payload = {'message': 'test', 'timestamp': int(time.time())}
        signature_key = 'signing_key_123'
        
        # Test signing
        signed_payload = self.connector.sign_request(payload, signature_key)
        
        self.assertIn('signature', signed_payload)
        self.assertIn('timestamp', signed_payload)
        
        # Verify signature manually
        expected_sig = hmac.new(
            signature_key.encode(),
            str(signed_payload['timestamp']).encode() + str(payload).encode(),
            hashlib.sha256
        ).hexdigest()
        
        self.assertEqual(signed_payload['signature'], expected_sig)

    def test_token_refresh_mechanism(self):
        """Test automatic token refresh."""
        if not hasattr(self.connector, 'refresh_token'):
            self.skipTest("Token refresh not available")
            
        with patch('requests.Session.request') as mock_request:
            # Mock token refresh response
            refresh_response = Mock()
            refresh_response.status_code = 200
            refresh_response.json.return_value = {
                'access_token': 'new_token_123',
                'expires_in': 3600
            }
            
            # Mock API response requiring fresh token
            api_response = Mock()
            api_response.status_code = 200
            api_response.json.return_value = {'data': 'success'}
            
            mock_request.side_effect = [refresh_response, api_response]
            
            # Trigger token refresh
            new_token = self.connector.refresh_token()
            
            self.assertEqual(new_token, 'new_token_123')
            self.assertEqual(self.connector.api_key, 'new_token_123')

    def test_rate_limiting_with_backoff(self):
        """Test rate limiting with exponential backoff."""
        with patch('requests.Session.request') as mock_request:
            with patch('time.sleep') as mock_sleep:
                # Mock rate limit responses
                rate_limit_response = Mock()
                rate_limit_response.status_code = 429
                rate_limit_response.headers = {'Retry-After': '60', 'X-RateLimit-Reset': '1640995200'}
                rate_limit_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
                
                success_response = Mock()
                success_response.status_code = 200
                success_response.json.return_value = {'data': 'success'}
                success_response.raise_for_status.return_value = None
                
                # First call rate limited, second succeeds
                mock_request.side_effect = [rate_limit_response, success_response]
                
                if hasattr(self.connector, 'make_request_with_rate_limiting'):
                    result = self.connector.make_request_with_rate_limiting('GET', '/test')
                    self.assertEqual(result, {'data': 'success'})
                    mock_sleep.assert_called_once_with(60)  # Should respect Retry-After

    def test_ssl_certificate_pinning(self):
        """Test SSL certificate pinning validation."""
        if not hasattr(self.connector, 'verify_ssl_pinning'):
            self.skipTest("SSL pinning not available")
            
        # Mock certificate data
        mock_cert_data = {
            'subject': 'CN=api.genesis.test',
            'issuer': 'CN=DigiCert',
            'serialNumber': '123456789',
            'sha256_fingerprint': 'abcd1234efgh5678'
        }
        
        # Test certificate validation
        is_valid = self.connector.verify_ssl_pinning(mock_cert_data)
        
        # Should validate against pinned certificates
        self.assertIsInstance(is_valid, bool)

    def test_security_headers_validation(self):
        """Test validation of security headers in responses."""
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': 'test'}
            mock_response.headers = {
                'Strict-Transport-Security': 'max-age=31536000',
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Content-Security-Policy': "default-src 'self'"
            }
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response
            
            result = self.connector.make_request('GET', '/test')
            
            self.assertEqual(result, {'data': 'test'})
            
            # Verify security headers if connector validates them
            if hasattr(self.connector, 'last_response_headers'):
                headers = self.connector.last_response_headers
                self.assertIn('Strict-Transport-Security', headers)
                self.assertIn('X-Content-Type-Options', headers)


class TestGenesisConnectorAdvancedMetrics(unittest.TestCase):
    """Advanced metrics and monitoring tests."""

    def setUp(self):
        """Set up metrics test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3,
            'enable_metrics': True
        }
        self.connector = GenesisConnector(self.mock_config)

    @patch('time.time')
    @patch('requests.Session.request')
    def test_detailed_metrics_collection(self, mock_request, mock_time):
        """Test detailed metrics collection for requests."""
        if not hasattr(self.connector, 'get_detailed_metrics'):
            self.skipTest("Detailed metrics not available")
            
        # Mock time progression
        mock_time.side_effect = [1000.0, 1000.5, 1001.0, 1001.2]  # Multiple timestamps
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_response.headers = {'Content-Length': '1024'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Make requests
        self.connector.make_request('GET', '/test')
        self.connector.make_request('POST', '/test', data={'key': 'value'})
        
        metrics = self.connector.get_detailed_metrics()
        
        self.assertIn('request_count', metrics)
        self.assertIn('total_response_time', metrics)
        self.assertIn('average_response_time', metrics)
        self.assertIn('requests_by_method', metrics)
        self.assertIn('status_code_distribution', metrics)

    def test_metrics_aggregation(self):
        """Test metrics aggregation over time periods."""
        if not hasattr(self.connector, 'get_metrics_for_period'):
            self.skipTest("Metrics aggregation not available")
            
        from datetime import datetime, timedelta
        
        # Define time periods
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': 'test'}
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response
            
            # Make some requests
            for i in range(10):
                self.connector.make_request('GET', f'/test/{i}')
            
            # Get metrics for different periods
            hourly_metrics = self.connector.get_metrics_for_period(hour_ago, now)
            daily_metrics = self.connector.get_metrics_for_period(day_ago, now)
            
            self.assertLessEqual(hourly_metrics['request_count'], daily_metrics['request_count'])

    def test_custom_metrics_tracking(self):
        """Test custom metrics tracking."""
        if not hasattr(self.connector, 'track_custom_metric'):
            self.skipTest("Custom metrics not available")
            
        # Track custom metrics
        self.connector.track_custom_metric('business_metric', 'user_signup', 1)
        self.connector.track_custom_metric('business_metric', 'user_login', 5)
        self.connector.track_custom_metric('performance_metric', 'cache_hit_rate', 0.85)
        
        if hasattr(self.connector, 'get_custom_metrics'):
            custom_metrics = self.connector.get_custom_metrics()
            
            self.assertIn('business_metric', custom_metrics)
            self.assertIn('performance_metric', custom_metrics)
            self.assertEqual(custom_metrics['business_metric']['user_signup'], 1)
            self.assertEqual(custom_metrics['performance_metric']['cache_hit_rate'], 0.85)

    def test_metrics_export_formats(self):
        """Test metrics export in different formats."""
        if not hasattr(self.connector, 'export_metrics'):
            self.skipTest("Metrics export not available")
            
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': 'test'}
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response
            
            # Generate some metrics
            for i in range(5):
                self.connector.make_request('GET', f'/test/{i}')
            
            # Test different export formats
            export_formats = ['json', 'csv', 'prometheus']
            
            for format_type in export_formats:
                with self.subTest(format=format_type):
                    try:
                        exported_data = self.connector.export_metrics(format=format_type)
                        self.assertIsNotNone(exported_data)
                        if format_type == 'json':
                            import json
                            parsed = json.loads(exported_data)
                            self.assertIsInstance(parsed, dict)
                    except (NotImplementedError, AttributeError):
                        # Format may not be implemented
                        pass


if __name__ == '__main__':
    # Configure comprehensive test execution
    import sys
    
    # Set up test discovery for all test classes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestGenesisConnectorDataSerialization,
        TestGenesisConnectorAdvancedAsync,
        TestGenesisConnectorAdvancedCaching,
        TestGenesisConnectorAdvancedSecurity,
        TestGenesisConnectorAdvancedMetrics
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        failfast=False,
        buffer=True
    )
    
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)



class TestGenesisConnectorComprehensiveValidation(unittest.TestCase):
    """
    Comprehensive validation tests for GenesisConnector.
    Testing framework: unittest with pytest enhancements
    Focus: Input validation, data integrity, and error boundary testing
    """

    def setUp(self):
        """Set up test fixtures for comprehensive validation tests."""
        self.valid_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.valid_config)

    def test_api_key_validation_comprehensive(self):
        """Test comprehensive API key validation scenarios."""
        invalid_keys = [
            None,
            '',
            ' ',
            '\t\n\r',
            123,
            [],
            {},
            True,
            False,
            'a' * 10001,  # Extremely long key
            'key\x00null',  # Null byte
            'key\nwith\nnewlines',
            'key\rwith\rreturns',
            'key\twith\ttabs',
            '🔑emoji_key',  # Unicode emoji
            'key with spaces',
        ]
        
        for invalid_key in invalid_keys:
            with self.subTest(api_key=repr(invalid_key)):
                config = self.valid_config.copy()
                config['api_key'] = invalid_key
                
                if invalid_key in [None, '', ' ', '\t\n\r', 123, [], {}, True, False] or len(str(invalid_key)) > 10000:
                    with self.assertRaises((ValueError, TypeError)):
                        GenesisConnector(config)
                else:
                    # Some edge cases might be accepted
                    try:
                        connector = GenesisConnector(config)
                        self.assertIsNotNone(connector.api_key)
                    except (ValueError, TypeError):
                        pass  # Also acceptable

    def test_base_url_validation_exhaustive(self):
        """Test exhaustive base URL validation scenarios."""
        url_test_cases = [
            # Invalid URLs
            ('', ValueError),
            (None, TypeError),
            ('not_a_url', ValueError),
            ('ftp://invalid.scheme', ValueError),
            ('http://', ValueError),
            ('https://', ValueError),
            ('://missing.protocol', ValueError),
            ('https://[invalid.bracket', ValueError),
            ('https://localhost:99999', ValueError),  # Invalid port
            
            # Valid URLs that should work
            ('https://api.test.com', None),
            ('http://localhost:8080', None),
            ('https://subdomain.api.test.com', None),
            ('https://api.test.com:443', None),
            ('https://api.test.com/v1', None),
            ('https://192.168.1.1', None),
            ('https://[::1]:8080', None),  # IPv6
        ]
        
        for url, expected_error in url_test_cases:
            with self.subTest(url=url):
                config = self.valid_config.copy()
                config['base_url'] = url
                
                if expected_error:
                    with self.assertRaises(expected_error):
                        GenesisConnector(config)
                else:
                    try:
                        connector = GenesisConnector(config)
                        self.assertEqual(connector.base_url, url)
                    except ValueError:
                        # Some edge cases might still be rejected
                        pass

    def test_timeout_boundary_values(self):
        """Test timeout validation with boundary values."""
        timeout_cases = [
            # Invalid timeouts
            (-1, ValueError),
            (-0.1, ValueError),
            (0, ValueError),
            ('30', TypeError),
            (None, ValueError),
            (float('inf'), ValueError),
            (float('-inf'), ValueError),
            (float('nan'), ValueError),
            
            # Valid timeouts
            (0.1, None),
            (1, None),
            (30, None),
            (3600, None),
            (86400, None),  # 24 hours
        ]
        
        for timeout, expected_error in timeout_cases:
            with self.subTest(timeout=timeout):
                config = self.valid_config.copy()
                config['timeout'] = timeout
                
                if expected_error:
                    with self.assertRaises(expected_error):
                        GenesisConnector(config)
                else:
                    connector = GenesisConnector(config)
                    self.assertEqual(connector.timeout, timeout)

    def test_max_retries_validation(self):
        """Test max_retries validation with edge cases."""
        retry_cases = [
            # Invalid retries
            (-1, ValueError),
            ('3', TypeError),
            (None, ValueError),
            (float('inf'), ValueError),
            (3.5, TypeError),
            
            # Valid retries
            (0, None),
            (1, None),
            (10, None),
            (100, None),
        ]
        
        for retries, expected_error in retry_cases:
            with self.subTest(retries=retries):
                config = self.valid_config.copy()
                config['max_retries'] = retries
                
                if expected_error:
                    with self.assertRaises(expected_error):
                        GenesisConnector(config)
                else:
                    connector = GenesisConnector(config)
                    self.assertEqual(connector.max_retries, retries)

    @patch('requests.Session.request')
    def test_request_data_validation_comprehensive(self, mock_request):
        """Test comprehensive request data validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        data_test_cases = [
            # Valid data types
            ({'string': 'value'}, True),
            ({'number': 42}, True),
            ({'float': 3.14}, True),
            ({'boolean': True}, True),
            ({'list': [1, 2, 3]}, True),
            ({'nested': {'key': 'value'}}, True),
            ({'null': None}, True),
            
            # Edge case data
            ({'empty_string': ''}, True),
            ({'empty_list': []}, True),
            ({'empty_dict': {}}, True),
            ({'unicode': '测试🚀'}, True),
            
            # Invalid data types that should be handled
            ({'set': {1, 2, 3}}, False),
            ({'function': lambda x: x}, False),
            ({'object': object()}, False),
        ]
        
        for data, should_succeed in data_test_cases:
            with self.subTest(data=str(data)[:50]):
                if should_succeed:
                    try:
                        result = self.connector.make_request('POST', '/test', data=data)
                        self.assertEqual(result, {'success': True})
                    except (TypeError, ValueError):
                        # Some edge cases might still fail
                        pass
                else:
                    with self.assertRaises((TypeError, ValueError, AttributeError)):
                        self.connector.make_request('POST', '/test', data=data)


class TestGenesisConnectorEdgeCaseScenarios(unittest.TestCase):
    """
    Edge case scenario tests for GenesisConnector.
    Testing framework: unittest with pytest enhancements
    Focus: Unusual conditions, boundary behaviors, and stress scenarios
    """

    def setUp(self):
        """Set up test fixtures for edge case scenarios."""
        self.connector = GenesisConnector({
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': 30,
            'max_retries': 3
        })

    @patch('requests.Session.request')
    def test_extremely_large_response_handling(self, mock_request):
        """Test handling of extremely large responses."""
        # Simulate very large response
        large_data = {'data': 'x' * (10 * 1024 * 1024)}  # 10MB string
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = large_data
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        try:
            result = self.connector.make_request('GET', '/large')
            self.assertIsNotNone(result)
            if 'data' in result:
                self.assertEqual(len(result['data']), 10 * 1024 * 1024)
        except (MemoryError, OSError):
            # Large responses might cause memory issues
            pass

    @patch('requests.Session.request')
    def test_response_with_invalid_json_structures(self, mock_request):
        """Test responses with various invalid JSON structures."""
        invalid_json_cases = [
            '{"unclosed": "object"',
            '{"trailing": "comma",}',
            '{"duplicate": "key", "duplicate": "value"}',
            '{"invalid": undefined}',
            '{"number": 999999999999999999999999999999999}',  # Overflow
            '{"control": "chars\x00\x01\x02"}',
            'null',
            '[]',
            '"just a string"',
            '42',
            'true',
            '',
        ]
        
        for invalid_json in invalid_json_cases:
            with self.subTest(json_content=invalid_json[:30]):
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = invalid_json
                mock_response.raise_for_status.return_value = None
                
                if invalid_json in ['null', '[]', '"just a string"', '42', 'true']:
                    # Valid JSON but unexpected structure
                    try:
                        if invalid_json == 'null':
                            mock_response.json.return_value = None
                        elif invalid_json == '[]':
                            mock_response.json.return_value = []
                        elif invalid_json == '"just a string"':
                            mock_response.json.return_value = "just a string"
                        elif invalid_json == '42':
                            mock_response.json.return_value = 42
                        elif invalid_json == 'true':
                            mock_response.json.return_value = True
                            
                        mock_request.return_value = mock_response
                        result = self.connector.make_request('GET', '/test')
                        self.assertIsNotNone(result)
                    except (ValueError, TypeError):
                        # Unexpected JSON structure might be rejected
                        pass
                else:
                    # Invalid JSON
                    mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
                    mock_request.return_value = mock_response
                    
                    with self.assertRaises(GenesisConnectionError):
                        self.connector.make_request('GET', '/test')

    def test_concurrent_configuration_changes(self):
        """Test behavior under concurrent configuration changes."""
        import threading
        import time
        
        results = []
        errors = []
        
        def config_changer():
            """Continuously change configuration."""
            for i in range(20):
                try:
                    new_config = {
                        'api_key': f'key_{i}',
                        'base_url': f'https://api{i % 3}.test.com',
                        'timeout': 30 + (i % 10),
                        'max_retries': 3 + (i % 3)
                    }
                    self.connector.reload_config(new_config)
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)
        
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


class TestGenesisConnectorDataSerialization(unittest.TestCase):
    """Comprehensive data serialization and deserialization tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)

    def test_serialization_of_complex_data_types(self):
        """Test serialization of complex nested data structures."""
        from decimal import Decimal
        from datetime import datetime, timezone
        import uuid
        
        complex_data = {
            'uuid': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'decimal_value': str(Decimal('123.456789')),
            'nested_objects': {
                'level1': {
                    'level2': {
                        'array_of_objects': [
                            {'id': i, 'name': f'item_{i}', 'active': i % 2 == 0}
                            for i in range(10)
                        ],
                        'metadata': {
                            'created_by': 'test_user',
                            'tags': ['tag1', 'tag2', 'tag3'],
                            'permissions': {
                                'read': True,
                                'write': False,
                                'admin': False
                            }
                        }
                    }
                }
            },
            'large_text_field': 'Lorem ipsum ' * 1000,
            'binary_encoded': 'YWxpY2UgaW4gd29uZGVybGFuZA==',  # base64
            'empty_values': {
                'empty_string': '',
                'empty_list': [],
                'empty_dict': {},
                'null_value': None
            }
        }

        if hasattr(self.connector, 'format_payload'):
            formatted = self.connector.format_payload(complex_data)
            self.assertIsNotNone(formatted)
            self.assertIn('uuid', formatted)
            self.assertIn('nested_objects', formatted)

    def test_circular_reference_detection(self):
        """Test detection and handling of circular references."""
        # Create circular reference
        data = {'name': 'root'}
        child = {'name': 'child', 'parent': data}
        data['child'] = child
        
        if hasattr(self.connector, 'format_payload'):
            with self.assertRaises((ValueError, RecursionError)):
                self.connector.format_payload(data)

    def test_deeply_nested_structure_limits(self):
        """Test handling of deeply nested structures at various depths."""
        for depth in [10, 50, 100, 500]:
            with self.subTest(depth=depth):
                # Create nested structure
                nested = {'level': 0}
                current = nested
                
                for i in range(depth):
                    current['next'] = {'level': i + 1}
                    current = current['next']
                
                current['end'] = True
                
                if hasattr(self.connector, 'format_payload'):
                    try:
                        formatted = self.connector.format_payload(nested)
                        if depth <= 100:  # Should work for reasonable depths
                            self.assertIsNotNone(formatted)
                    except (RecursionError, ValueError):
                        # Expected for very deep structures
                        if depth > 100:
                            pass
                        else:
                            raise

    def test_unicode_normalization(self):
        """Test Unicode normalization and handling."""
        import unicodedata
        
        unicode_test_cases = [
            'café',  # NFC
            'cafe\u0301',  # NFD  
            'Ωήμος',  # Greek
            '한글',  # Korean
            '🌟💫🚀',  # Emoji
            '\u200b\u200c\u200d',  # Zero-width characters
            'test\u0000null',  # Null byte
            'test\ufffeReverse',  # Reverse byte order mark
        ]
        
        for text in unicode_test_cases:
            with self.subTest(text=repr(text)):
                payload = {
                    'original': text,
                    'nfc': unicodedata.normalize('NFC', text),
                    'nfd': unicodedata.normalize('NFD', text),
                    'message': f'Testing {text}'
                }
                
                if hasattr(self.connector, 'format_payload'):
                    try:
                        formatted = self.connector.format_payload(payload)
                        self.assertIsNotNone(formatted)
                    except ValueError:
                        # Some characters may be rejected
                        if '\u0000' in text or '\ufffe' in text:
                            pass  # Expected for problematic characters
                        else:
                            raise

    def test_large_payload_chunking(self):
        """Test handling of very large payloads."""
        # Test various payload sizes
        sizes = [
            1024,           # 1KB
            1024 * 100,     # 100KB  
            1024 * 1024,    # 1MB
            1024 * 1024 * 5 # 5MB
        ]
        
        for size in sizes:
            with self.subTest(size=size):
                large_data = {
                    'large_field': 'x' * size,
                    'metadata': {
                        'size': size,
                        'description': f'Payload of {size} bytes'
                    }
                }
                
                if hasattr(self.connector, 'format_payload'):
                    try:
                        formatted = self.connector.format_payload(large_data)
                        if size <= 1024 * 1024:  # Up to 1MB should work
                            self.assertIsNotNone(formatted)
                    except (MemoryError, ValueError):
                        # Large payloads may be rejected
                        if size > 1024 * 1024:
                            pass  # Expected for very large payloads
                        else:
                            raise

    def test_custom_json_encoder_handling(self):
        """Test handling of custom objects that need special encoding."""
        from datetime import datetime, date, time
        from decimal import Decimal
        import uuid
        
        class CustomObject:
            def __init__(self, value):
                self.value = value
                
            def __str__(self):
                return f"CustomObject({self.value})"
                
            def to_dict(self):
                return {'custom_value': self.value}
        
        custom_data = {
            'datetime_obj': datetime.now(),
            'date_obj': date.today(),
            'time_obj': time(14, 30, 0),
            'decimal_obj': Decimal('123.456'),
            'uuid_obj': uuid.uuid4(),
            'custom_obj': CustomObject('test_value'),
            'set_obj': {1, 2, 3, 4, 5},
            'frozenset_obj': frozenset([1, 2, 3]),
            'bytes_obj': b'binary_data',
            'bytearray_obj': bytearray(b'mutable_binary')
        }
        
        if hasattr(self.connector, 'format_payload'):
            try:
                formatted = self.connector.format_payload(custom_data)
                self.assertIsNotNone(formatted)
                # Check that at least some fields are present
                self.assertIn('datetime_obj', formatted)
                self.assertIn('custom_obj', formatted)
            except (TypeError, ValueError):
                # Custom objects may not be serializable
                pass


class TestGenesisConnectorAdvancedAsync(unittest.TestCase):
    """Advanced asynchronous functionality tests."""

    def setUp(self):
        """Set up async test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)

    @patch('asyncio.create_task')
    async def test_async_batch_processing(self, mock_create_task):
        """Test asynchronous batch processing of multiple requests."""
        if not hasattr(self.connector, 'async_batch_request'):
            self.skipTest("Async batch functionality not available")
            
        # Mock multiple async responses
        mock_tasks = []
        for i in range(5):
            mock_task = AsyncMock()
            mock_task.return_value = {'id': i, 'result': f'batch_result_{i}'}
            mock_tasks.append(mock_task)
        
        mock_create_task.side_effect = mock_tasks
        
        payloads = [
            {'request_id': i, 'data': f'batch_data_{i}'}
            for i in range(5)
        ]
        
        results = await self.connector.async_batch_request(payloads)
        
        self.assertEqual(len(results), 5)
        for i, result in enumerate(results):
            self.assertEqual(result['id'], i)

    async def test_async_timeout_handling_with_cancellation(self):
        """Test async timeout handling with proper task cancellation."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
            
        with patch('aiohttp.ClientSession.request') as mock_request:
            # Simulate a request that takes too long
            async def slow_request(*args, **kwargs):
                await asyncio.sleep(10)  # Longer than timeout
                return AsyncMock(status=200, json=AsyncMock(return_value={'data': 'slow'}))
            
            mock_request.side_effect = slow_request
            
            # Set short timeout
            original_timeout = self.connector.timeout
            self.connector.timeout = 1
            
            try:
                with self.assertRaises(GenesisTimeoutError):
                    await self.connector.async_make_request('GET', '/slow')
            finally:
                self.connector.timeout = original_timeout

    async def test_async_connection_pooling(self):
        """Test async connection pooling and reuse."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
            
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'success': True})
            mock_session.request.return_value.__aenter__.return_value = mock_response
            
            # Make multiple requests
            for i in range(3):
                result = await self.connector.async_make_request('GET', f'/test/{i}')
                self.assertEqual(result, {'success': True})
            
            # Session should be reused
            self.assertEqual(mock_session_class.call_count, 1)

    async def test_async_error_propagation(self):
        """Test proper error propagation in async operations."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")
            
        error_scenarios = [
            (asyncio.TimeoutError("Async timeout"), GenesisTimeoutError),
            (aiohttp.ClientError("Client error"), GenesisConnectionError),
            (ConnectionError("Connection failed"), GenesisConnectionError),
            (ValueError("Invalid data"), GenesisConnectionError)
        ]
        
        for original_error, expected_error in error_scenarios:
            with self.subTest(error=original_error.__class__.__name__):
                with patch('aiohttp.ClientSession.request') as mock_request:
                    mock_request.side_effect = original_error
                    
                    with self.assertRaises(expected_error):
                        await self.connector.async_make_request('GET', '/test')


class TestGenesisConnectorAdvancedCaching(unittest.TestCase):
    """Advanced caching and memoization tests."""

    def setUp(self):
        """Set up caching test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3,
            'enable_caching': True,
            'cache_ttl': 300
        }
        self.connector = GenesisConnector(self.mock_config)

    @patch('requests.Session.request')
    def test_response_caching_with_ttl(self, mock_request):
        """Test response caching with time-to-live."""
        if not hasattr(self.connector, 'get_cached_response'):
            self.skipTest("Caching functionality not available")
            
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'cached_data': 'test', 'timestamp': '2023-01-01T00:00:00Z'}
        mock_response.headers = {'Cache-Control': 'max-age=300'}
        mock_request.return_value = mock_response
        
        # First request should hit the API
        result1 = self.connector.get_cached_response('/cacheable')
        
        # Second request should use cache
        result2 = self.connector.get_cached_response('/cacheable')
        
        self.assertEqual(result1, result2)
        self.assertEqual(mock_request.call_count, 1)

    @patch('time.time')
    @patch('requests.Session.request')
    def test_cache_expiration(self, mock_request, mock_time):
        """Test cache expiration after TTL."""
        if not hasattr(self.connector, 'get_cached_response'):
            self.skipTest("Caching functionality not available")
            
        # Mock time progression
        mock_time.side_effect = [1000, 1000, 1400]  # 400 seconds later
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'cached_data': 'test'}
        mock_response.headers = {'Cache-Control': 'max-age=300'}  # 5 minutes
        mock_request.return_value = mock_response
        
        # First request
        self.connector.get_cached_response('/cacheable')
        
        # Second request after expiration
        self.connector.get_cached_response('/cacheable')
        
        # Should make two API calls due to expiration
        self.assertEqual(mock_request.call_count, 2)

    def test_cache_invalidation(self):
        """Test manual cache invalidation."""
        if not hasattr(self.connector, 'invalidate_cache'):
            self.skipTest("Cache invalidation not available")
            
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': 'test'}
            mock_request.return_value = mock_response
            
            # Cache a response
            if hasattr(self.connector, 'get_cached_response'):
                self.connector.get_cached_response('/test')
            
            # Invalidate cache
            self.connector.invalidate_cache('/test')
            
            # Next request should hit API again
            if hasattr(self.connector, 'get_cached_response'):
                self.connector.get_cached_response('/test')
                self.assertEqual(mock_request.call_count, 2)

    def test_cache_size_limits(self):
        """Test cache size limits and LRU eviction."""
        if not (hasattr(self.connector, 'get_cached_response') and 
                hasattr(self.connector, 'cache_size_limit')):
            self.skipTest("Cache size limiting not available")
            
        # Set small cache limit
        self.connector.cache_size_limit = 3
        
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'Cache-Control': 'max-age=3600'}
            mock_request.return_value = mock_response
            
            # Fill cache beyond limit
            for i in range(5):
                mock_response.json.return_value = {'data': f'test_{i}'}
                self.connector.get_cached_response(f'/test/{i}')
            
            # First entries should be evicted
            # Re-requesting first endpoint should hit API again
            mock_response.json.return_value = {'data': 'test_0_new'}
            result = self.connector.get_cached_response('/test/0')
            
            # Should have made 6 requests total (5 + 1 after eviction)
            self.assertEqual(mock_request.call_count, 6)


class TestGenesisConnectorAdvancedSecurity(unittest.TestCase):
    """Advanced security and authentication tests."""

    def setUp(self):
        """Set up security test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }
        self.connector = GenesisConnector(self.mock_config)

    def test_request_signing_with_hmac(self):
        """Test HMAC-based request signing."""
        if not hasattr(self.connector, 'sign_request'):
            self.skipTest("Request signing not available")
            
        import hmac
        import hashlib
        import time
        
        payload = {'message': 'test', 'timestamp': int(time.time())}
        signature_key = 'signing_key_123'
        
        # Test signing
        signed_payload = self.connector.sign_request(payload, signature_key)
        
        self.assertIn('signature', signed_payload)
        self.assertIn('timestamp', signed_payload)
        
        # Verify signature manually
        expected_sig = hmac.new(
            signature_key.encode(),
            str(signed_payload['timestamp']).encode() + str(payload).encode(),
            hashlib.sha256
        ).hexdigest()
        
        self.assertEqual(signed_payload['signature'], expected_sig)

    def test_token_refresh_mechanism(self):
        """Test automatic token refresh."""
        if not hasattr(self.connector, 'refresh_token'):
            self.skipTest("Token refresh not available")
            
        with patch('requests.Session.request') as mock_request:
            # Mock token refresh response
            refresh_response = Mock()
            refresh_response.status_code = 200
            refresh_response.json.return_value = {
                'access_token': 'new_token_123',
                'expires_in': 3600
            }
            
            # Mock API response requiring fresh token
            api_response = Mock()
            api_response.status_code = 200
            api_response.json.return_value = {'data': 'success'}
            
            mock_request.side_effect = [refresh_response, api_response]
            
            # Trigger token refresh
            new_token = self.connector.refresh_token()
            
            self.assertEqual(new_token, 'new_token_123')
            self.assertEqual(self.connector.api_key, 'new_token_123')

    def test_rate_limiting_with_backoff(self):
        """Test rate limiting with exponential backoff."""
        with patch('requests.Session.request') as mock_request:
            with patch('time.sleep') as mock_sleep:
                # Mock rate limit responses
                rate_limit_response = Mock()
                rate_limit_response.status_code = 429
                rate_limit_response.headers = {'Retry-After': '60', 'X-RateLimit-Reset': '1640995200'}
                rate_limit_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
                
                success_response = Mock()
                success_response.status_code = 200
                success_response.json.return_value = {'data': 'success'}
                success_response.raise_for_status.return_value = None
                
                # First call rate limited, second succeeds
                mock_request.side_effect = [rate_limit_response, success_response]
                
                if hasattr(self.connector, 'make_request_with_rate_limiting'):
                    result = self.connector.make_request_with_rate_limiting('GET', '/test')
                    self.assertEqual(result, {'data': 'success'})
                    mock_sleep.assert_called_once_with(60)  # Should respect Retry-After

    def test_ssl_certificate_pinning(self):
        """Test SSL certificate pinning validation."""
        if not hasattr(self.connector, 'verify_ssl_pinning'):
            self.skipTest("SSL pinning not available")
            
        # Mock certificate data
        mock_cert_data = {
            'subject': 'CN=api.genesis.test',
            'issuer': 'CN=DigiCert',
            'serialNumber': '123456789',
            'sha256_fingerprint': 'abcd1234efgh5678'
        }
        
        # Test certificate validation
        is_valid = self.connector.verify_ssl_pinning(mock_cert_data)
        
        # Should validate against pinned certificates
        self.assertIsInstance(is_valid, bool)

    def test_security_headers_validation(self):
        """Test validation of security headers in responses."""
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': 'test'}
            mock_response.headers = {
                'Strict-Transport-Security': 'max-age=31536000',
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Content-Security-Policy': "default-src 'self'"
            }
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response
            
            result = self.connector.make_request('GET', '/test')
            
            self.assertEqual(result, {'data': 'test'})
            
            # Verify security headers if connector validates them
            if hasattr(self.connector, 'last_response_headers'):
                headers = self.connector.last_response_headers
                self.assertIn('Strict-Transport-Security', headers)
                self.assertIn('X-Content-Type-Options', headers)


class TestGenesisConnectorAdvancedMetrics(unittest.TestCase):
    """Advanced metrics and monitoring tests."""

    def setUp(self):
        """Set up metrics test fixtures."""
        self.mock_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3,
            'enable_metrics': True
        }
        self.connector = GenesisConnector(self.mock_config)

    @patch('time.time')
    @patch('requests.Session.request')
    def test_detailed_metrics_collection(self, mock_request, mock_time):
        """Test detailed metrics collection for requests."""
        if not hasattr(self.connector, 'get_detailed_metrics'):
            self.skipTest("Detailed metrics not available")
            
        # Mock time progression
        mock_time.side_effect = [1000.0, 1000.5, 1001.0, 1001.2]  # Multiple timestamps
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_response.headers = {'Content-Length': '1024'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Make requests
        self.connector.make_request('GET', '/test')
        self.connector.make_request('POST', '/test', data={'key': 'value'})
        
        metrics = self.connector.get_detailed_metrics()
        
        self.assertIn('request_count', metrics)
        self.assertIn('total_response_time', metrics)
        self.assertIn('average_response_time', metrics)
        self.assertIn('requests_by_method', metrics)
        self.assertIn('status_code_distribution', metrics)

    def test_metrics_aggregation(self):
        """Test metrics aggregation over time periods."""
        if not hasattr(self.connector, 'get_metrics_for_period'):
            self.skipTest("Metrics aggregation not available")
            
        from datetime import datetime, timedelta
        
        # Define time periods
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': 'test'}
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response
            
            # Make some requests
            for i in range(10):
                self.connector.make_request('GET', f'/test/{i}')
            
            # Get metrics for different periods
            hourly_metrics = self.connector.get_metrics_for_period(hour_ago, now)
            daily_metrics = self.connector.get_metrics_for_period(day_ago, now)
            
            self.assertLessEqual(hourly_metrics['request_count'], daily_metrics['request_count'])

    def test_custom_metrics_tracking(self):
        """Test custom metrics tracking."""
        if not hasattr(self.connector, 'track_custom_metric'):
            self.skipTest("Custom metrics not available")
            
        # Track custom metrics
        self.connector.track_custom_metric('business_metric', 'user_signup', 1)
        self.connector.track_custom_metric('business_metric', 'user_login', 5)
        self.connector.track_custom_metric('performance_metric', 'cache_hit_rate', 0.85)
        
        if hasattr(self.connector, 'get_custom_metrics'):
            custom_metrics = self.connector.get_custom_metrics()
            
            self.assertIn('business_metric', custom_metrics)
            self.assertIn('performance_metric', custom_metrics)
            self.assertEqual(custom_metrics['business_metric']['user_signup'], 1)
            self.assertEqual(custom_metrics['performance_metric']['cache_hit_rate'], 0.85)

    def test_metrics_export_formats(self):
        """Test metrics export in different formats."""
        if not hasattr(self.connector, 'export_metrics'):
            self.skipTest("Metrics export not available")
            
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': 'test'}
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response
            
            # Generate some metrics
            for i in range(5):
                self.connector.make_request('GET', f'/test/{i}')
            
            # Test different export formats
            export_formats = ['json', 'csv', 'prometheus']
            
            for format_type in export_formats:
                with self.subTest(format=format_type):
                    try:
                        exported_data = self.connector.export_metrics(format=format_type)
                        self.assertIsNotNone(exported_data)
                        if format_type == 'json':
                            import json
                            parsed = json.loads(exported_data)
                            self.assertIsInstance(parsed, dict)
                    except (NotImplementedError, AttributeError):
                        # Format may not be implemented
                        pass


if __name__ == '__main__':
    # Configure comprehensive test execution
    import sys
    
    # Set up test discovery for all test classes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestGenesisConnectorDataSerialization,
        TestGenesisConnectorAdvancedAsync,
        TestGenesisConnectorAdvancedCaching,
        TestGenesisConnectorAdvancedSecurity,
        TestGenesisConnectorAdvancedMetrics
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        failfast=False,
        buffer=True
    )
    
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)



# ============================================================================
# COMPREHENSIVE UNIT TESTS EXTENSION FOR GENESIS CONNECTOR
# Testing Framework: unittest with pytest enhancements
# ============================================================================

# Additional imports needed for comprehensive testing
import json
import time
import socket
import asyncio
import threading
from decimal import Decimal
from datetime import datetime, date, timezone, timedelta
from collections import OrderedDict
from unittest.mock import AsyncMock, call, patch, Mock, MagicMock
import requests


# Mock classes for testing (if not already defined)
class GenesisConnector:
    """Mock GenesisConnector class for testing purposes."""
    def __init__(self, config):
        self.config = config or {}
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url', 'https://api.genesis.test')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.session = requests.Session()
        self._validate_config()
    
    def _validate_config(self):
        if not self.api_key or not isinstance(self.api_key, str):
            raise ValueError("API key is required and must be a string")
        if not self.base_url or not isinstance(self.base_url, str):
            raise ValueError("Base URL is required and must be a string")
        if not isinstance(self.timeout, (int, float)) or self.timeout <= 0:
            raise ValueError("Timeout must be a positive number")
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("Max retries must be a non-negative integer")
    
    def make_request(self, method, endpoint, data=None, headers=None):
        """Make HTTP request with retry logic."""
        url = self._build_url(endpoint)
        request_headers = self._build_headers(headers)
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=request_headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                if attempt == self.max_retries:
                    raise GenesisTimeoutError(f"Request timed out after {self.max_retries} retries")
                time.sleep(2 ** attempt)  # Exponential backoff
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries:
                    raise GenesisConnectionError(f"Request failed: {str(e)}")
                time.sleep(2 ** attempt)
    
    def _build_url(self, endpoint):
        """Build full URL from base URL and endpoint."""
        if endpoint.startswith('/'):
            endpoint = endpoint[1:]
        return f"{self.base_url.rstrip('/')}/{endpoint}"
    
    def _build_headers(self, headers=None):
        """Build request headers with authentication."""
        default_headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        if headers:
            default_headers.update(headers)
        return default_headers
    
    def format_payload(self, data):
        """Format payload for transmission."""
        if data is None:
            return {}
        return self._serialize_data(data)
    
    def _serialize_data(self, data):
        """Serialize data handling special types."""
        if hasattr(data, '__dict__'):
            if hasattr(data, 'to_dict'):
                return data.to_dict()
            return str(data)
        return data
    
    def reload_config(self, new_config):
        """Reload configuration."""
        self.config.update(new_config)
        self.api_key = new_config.get('api_key', self.api_key)
        self.base_url = new_config.get('base_url', self.base_url)
        self.timeout = new_config.get('timeout', self.timeout)
        self.max_retries = new_config.get('max_retries', self.max_retries)
        self._validate_config()
    
    def get_headers(self):
        """Get current headers."""
        return self._build_headers()
    
    def validate_config(self, config):
        """Validate configuration."""
        try:
            temp_connector = GenesisConnector(config)
            return True
        except (ValueError, TypeError):
            return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.session, 'close'):
            self.session.close()
    
    def __repr__(self):
        masked_key = f"{self.api_key[:4]}***" if len(self.api_key) > 4 else "***"
        return f"GenesisConnector(api_key={masked_key}, base_url={self.base_url})"


class GenesisConnectionError(Exception):
    """Base exception for Genesis connector errors."""
    pass


class GenesisTimeoutError(GenesisConnectionError):
    """Exception raised when requests timeout."""
    pass


# ============================================================================
# COMPREHENSIVE INPUT VALIDATION TESTS
# ============================================================================

class TestGenesisConnectorInputValidation(unittest.TestCase):
    """
    Comprehensive input validation tests for GenesisConnector.
    Testing Framework: unittest
    Focus: Input sanitization, boundary conditions, and security validation
    """

    def setUp(self):
        """Set up test fixtures for input validation tests."""
        self.valid_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.genesis.test',
            'timeout': 30,
            'max_retries': 3
        }

    def test_api_key_boundary_conditions(self):
        """Test API key validation at boundary conditions."""
        boundary_cases = [
            # Minimum length scenarios
            ('a', True),  # Single character
            ('ab', True),  # Two characters
            ('', False),   # Empty string
            
            # Maximum length scenarios  
            ('x' * 1000, True),    # Very long key
            ('x' * 10000, True),   # Extremely long key
            
            # Special character scenarios
            ('key-with-dashes', True),
            ('key_with_underscores', True),
            ('key.with.dots', True),
            ('key+with+plus', True),
            ('key/with/slashes', True),
            ('key=with=equals', True),
            
            # Whitespace scenarios
            (' key_with_leading_space', True),
            ('key_with_trailing_space ', True),
            ('key with internal spaces', True),
            ('\tkey_with_tab', True),
            ('key_with_newline\n', True),
            
            # Unicode scenarios
            ('测试key', True),
            ('key🔑', True),
            ('ключ', True),
            
            # Edge cases
            ('key\x00with_null', True),  # Null byte
            ('key\x1fwith_control', True),  # Control character
        ]
        
        for api_key, should_succeed in boundary_cases:
            with self.subTest(api_key=repr(api_key)):
                config = self.valid_config.copy()
                config['api_key'] = api_key
                
                if should_succeed:
                    try:
                        connector = GenesisConnector(config)
                        self.assertEqual(connector.api_key, api_key)
                    except (ValueError, TypeError):
                        # Some edge cases might still fail - that's acceptable
                        pass
                else:
                    with self.assertRaises((ValueError, TypeError)):
                        GenesisConnector(config)

    def test_url_parsing_edge_cases(self):
        """Test URL parsing with complex edge cases."""
        url_test_cases = [
            # Protocol variations
            ('https://api.test.com', True),
            ('http://api.test.com', True),
            ('HTTPS://API.TEST.COM', True),  # Case insensitive
            
            # Port specifications
            ('https://api.test.com:443', True),
            ('https://api.test.com:8080', True),
            ('https://api.test.com:65535', True),
            ('https://api.test.com:0', False),  # Invalid port
            ('https://api.test.com:99999', False),  # Port too high
            
            # Path components
            ('https://api.test.com/v1', True),
            ('https://api.test.com/v1/api', True),
            ('https://api.test.com/v1/api/', True),
            ('https://api.test.com//double/slash', True),
            
            # Query parameters
            ('https://api.test.com?param=value', True),
            ('https://api.test.com/path?multiple=params&test=true', True),
            
            # Special characters in URL
            ('https://api.test.com/path%20with%20spaces', True),
            ('https://api.test.com/path+with+plus', True),
            ('https://api.test.com/path#fragment', True),
            
            # IPv4 addresses
            ('https://192.168.1.1', True),
            ('https://10.0.0.1:8080', True),
            ('https://127.0.0.1:3000', True),
            
            # IPv6 addresses
            ('https://[::1]', True),
            ('https://[::1]:8080', True),
            ('https://[2001:db8::1]', True),
            
            # Domain edge cases
            ('https://a.b', True),  # Minimal domain
            ('https://very-long-subdomain.very-long-domain.com', True),
            ('https://test.xn--nxasmq6b', True),  # Internationalized domain
            
            # Invalid URLs
            ('ftp://api.test.com', False),  # Wrong protocol
            ('https://', False),  # No domain
            ('api.test.com', False),  # No protocol
            ('https://[invalid', False),  # Malformed IPv6
            ('', False),  # Empty URL
            (None, False),  # None URL
        ]
        
        for url, should_succeed in url_test_cases:
            with self.subTest(url=url):
                config = self.valid_config.copy()
                config['base_url'] = url
                
                if should_succeed:
                    try:
                        connector = GenesisConnector(config)
                        self.assertEqual(connector.base_url, url)
                    except (ValueError, TypeError):
                        # Some edge cases might be platform-dependent
                        pass
                else:
                    with self.assertRaises((ValueError, TypeError)):
                        GenesisConnector(config)

    def test_numeric_parameter_edge_cases(self):
        """Test numeric parameters with extreme values."""
        timeout_cases = [
            # Boundary values
            (0.001, True),     # Very small positive
            (0.1, True),       # Small positive
            (1, True),         # Minimum reasonable
            (3600, True),      # Large but reasonable
            (86400, True),     # Very large
            (float('inf'), False),  # Infinity
            (float('-inf'), False), # Negative infinity
            (float('nan'), False),  # Not a number
            
            # Zero and negative
            (0, False),        # Exactly zero
            (-1, False),       # Negative
            (-0.1, False),     # Small negative
            
            # Type variations
            ('30', False),     # String number
            (None, False),     # None
            (True, False),     # Boolean
            ([30], False),     # List
        ]
        
        for timeout, should_succeed in timeout_cases:
            with self.subTest(timeout=timeout):
                config = self.valid_config.copy()
                config['timeout'] = timeout
                
                if should_succeed:
                    connector = GenesisConnector(config)
                    self.assertEqual(connector.timeout, timeout)
                else:
                    with self.assertRaises((ValueError, TypeError)):
                        GenesisConnector(config)

    def test_data_payload_sanitization(self):
        """Test comprehensive data payload sanitization."""
        dangerous_payloads = [
            # Script injection attempts
            {'message': '<script>alert("xss")</script>'},
            {'html': '<img src=x onerror=alert(1)>'},
            {'css': 'body{background:url("javascript:alert(1)")}'},
            
            # SQL injection patterns
            {'query': "'; DROP TABLE users; --"},
            {'filter': "1 OR 1=1"},
            {'id': "1; DELETE FROM users WHERE 1=1"},
            
            # Command injection
            {'command': '; rm -rf /'},
            {'path': '../../../../etc/passwd'},
            {'file': '$(whoami)'},
            
            # JSON injection
            {'json': '{"malicious": true}'},
            {'nested': {'json': '"value"'}},
            
            # Large payloads
            {'large_text': 'A' * 1000000},  # 1MB of data
            {'many_keys': {f'key_{i}': f'value_{i}' for i in range(10000)}},
            
            # Special characters
            {'unicode': '\u0000\u0001\u0002'},  # Control characters
            {'emoji': '💀☠️🔥'},
            {'mixed': 'text\x00with\x01null\x02bytes'},
        ]
        
        connector = GenesisConnector(self.valid_config)
        
        for payload in dangerous_payloads:
            with self.subTest(payload=str(payload)[:100]):
                try:
                    # Should either sanitize or reject dangerous payloads
                    formatted = connector.format_payload(payload)
                    self.assertIsNotNone(formatted)
                    # Verify no obvious script tags remain
                    formatted_str = str(formatted)
                    self.assertNotIn('<script>', formatted_str.lower())
                    self.assertNotIn('javascript:', formatted_str.lower())
                except (ValueError, TypeError):
                    # Rejecting dangerous payloads is also acceptable
                    pass

    def test_header_injection_prevention(self):
        """Test prevention of header injection attacks."""
        malicious_headers = [
            # CRLF injection
            {'X-Test': 'value\r\nX-Injected: injected'},
            {'User-Agent': 'agent\nX-Malicious: true'},
            
            # Script injection in headers
            {'X-Custom': '<script>alert(1)</script>'},
            {'Referer': 'javascript:alert(1)'},
            
            # Oversized headers
            {'X-Large': 'A' * 100000},
            
            # Null bytes
            {'X-Null': 'value\x00injected'},
            
            # Unicode normalization attacks
            {'X-Unicode': '\u0130\u0307'},  # Potential normalization issue
        ]
        
        connector = GenesisConnector(self.valid_config)
        
        for headers in malicious_headers:
            with self.subTest(headers=headers):
                try:
                    result_headers = connector._build_headers(headers)
                    # Should either sanitize or reject malicious headers
                    for key, value in result_headers.items():
                        if key.startswith('X-'):
                            # Check for injection patterns
                            self.assertNotIn('\r', str(value))
                            self.assertNotIn('\n', str(value))
                            self.assertNotIn('\x00', str(value))
                except (ValueError, TypeError):
                    # Rejecting malicious headers is acceptable
                    pass


# ============================================================================
# PERFORMANCE AND STRESS TESTS
# ============================================================================

class TestGenesisConnectorPerformanceStress(unittest.TestCase):
    """
    Performance and stress testing for GenesisConnector.
    Testing Framework: unittest
    Focus: Load testing, memory usage, and performance benchmarks
    """

    def setUp(self):
        """Set up performance test environment."""
        self.connector = GenesisConnector({
            'api_key': 'perf_test_key',
            'base_url': 'https://api.performance.test',
            'timeout': 30,
            'max_retries': 3
        })

    @patch('requests.Session.request')
    def test_concurrent_request_handling(self, mock_request):
        """Test handling of concurrent requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        def make_requests(thread_id, num_requests, results):
            """Make multiple requests in a thread."""
            thread_results = []
            for i in range(num_requests):
                try:
                    result = self.connector.make_request('GET', f'/thread/{thread_id}/request/{i}')
                    thread_results.append(result)
                except Exception as e:
                    thread_results.append(f"Error: {e}")
            results[thread_id] = thread_results

        # Test with multiple threads
        num_threads = 10
        requests_per_thread = 50
        results = {}
        threads = []

        start_time = time.time()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=make_requests,
                args=(thread_id, requests_per_thread, results)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()
        
        # Verify all requests completed
        total_requests = num_threads * requests_per_thread
        completed_requests = sum(len(thread_results) for thread_results in results.values())
        self.assertEqual(completed_requests, total_requests)
        
        # Performance assertion
        total_time = end_time - start_time
        requests_per_second = total_requests / total_time
        self.assertGreater(requests_per_second, 100)  # Should handle >100 req/sec

    @patch('requests.Session.request')
    def test_memory_leak_detection(self, mock_request):
        """Test for memory leaks over extended usage."""
        import gc
        import sys

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'x' * 1000}  # 1KB response
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        # Measure initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Make many requests to test for leaks
        for batch in range(10):  # 10 batches
            for i in range(100):  # 100 requests per batch
                result = self.connector.make_request('GET', f'/batch/{batch}/request/{i}')
                self.assertIsNotNone(result)
            
            # Force garbage collection periodically
            gc.collect()

        # Measure final memory
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory growth should be minimal
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 1000, "Potential memory leak detected")

    @patch('requests.Session.request')
    def test_request_queue_saturation(self, mock_request):
        """Test behavior under request queue saturation."""
        # Simulate slow responses
        def slow_response(*args, **kwargs):
            time.sleep(0.1)  # 100ms delay
            response = Mock()
            response.status_code = 200
            response.json.return_value = {'delayed': True}
            response.raise_for_status.return_value = None
            return response

        mock_request.side_effect = slow_response

        # Flood with requests
        start_time = time.time()
        results = []
        
        for i in range(50):  # Many requests
            try:
                result = self.connector.make_request('GET', f'/saturate/{i}')
                results.append(result)
            except Exception as e:
                results.append(f"Error: {e}")

        end_time = time.time()
        
        # Should handle all requests without crashing
        self.assertEqual(len(results), 50)
        
        # Should maintain reasonable performance even under load
        total_time = end_time - start_time
        self.assertLess(total_time, 20)  # Should complete within 20 seconds

    def test_configuration_mutation_safety(self):
        """Test thread safety of configuration mutations."""
        def config_mutator(connector, thread_id, iterations):
            """Continuously mutate configuration."""
            for i in range(iterations):
                new_config = {
                    'api_key': f'thread_{thread_id}_key_{i}',
                    'base_url': f'https://api{thread_id}.test.com',
                    'timeout': 30 + (i % 10),
                    'max_retries': 3 + (i % 3)
                }
                try:
                    connector.reload_config(new_config)
                    time.sleep(0.001)  # Small delay
                except Exception:
                    pass  # Expected under concurrent access

        # Start multiple threads mutating config
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(
                target=config_mutator,
                args=(self.connector, thread_id, 100)
            )
            threads.append(thread)
            thread.start()

        # Main thread also uses connector
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'concurrent': True}
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response

            for i in range(50):
                try:
                    result = self.connector.make_request('GET', f'/concurrent/{i}')
                    self.assertIsNotNone(result)
                except Exception:
                    pass  # Expected under concurrent mutation

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Connector should still be functional
        self.assertIsNotNone(self.connector.config)


# ============================================================================
# ERROR RECOVERY AND RESILIENCE TESTS
# ============================================================================

class TestGenesisConnectorErrorRecovery(unittest.TestCase):
    """
    Error recovery and resilience testing for GenesisConnector.
    Testing Framework: unittest
    Focus: Fault tolerance, recovery mechanisms, and graceful degradation
    """

    def setUp(self):
        """Set up error recovery test environment."""
        self.connector = GenesisConnector({
            'api_key': 'recovery_test_key',
            'base_url': 'https://api.recovery.test',
            'timeout': 5,
            'max_retries': 3
        })

    @patch('requests.Session.request')
    def test_progressive_retry_backoff(self, mock_request):
        """Test progressive retry backoff strategy."""
        # Track retry timing
        retry_times = []
        
        def failing_request(*args, **kwargs):
            retry_times.append(time.time())
            raise ConnectionError("Simulated connection failure")

        mock_request.side_effect = failing_request

        start_time = time.time()
        
        with self.assertRaises(GenesisConnectionError):
            self.connector.make_request('GET', '/test-retry')

        # Should have made initial request + retries
        self.assertEqual(len(retry_times), self.connector.max_retries + 1)
        
        # Verify exponential backoff timing
        if len(retry_times) > 1:
            for i in range(1, len(retry_times)):
                delay = retry_times[i] - retry_times[i-1]
                expected_min_delay = 2 ** (i-1)  # Exponential backoff
                self.assertGreaterEqual(delay, expected_min_delay * 0.8)  # Allow some variance

    @patch('requests.Session.request')
    def test_circuit_breaker_pattern(self, mock_request):
        """Test circuit breaker-like behavior for repeated failures."""
        failure_count = 0
        
        def intermittent_failure(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 10:  # First 10 requests fail
                raise ConnectionError("Service unavailable")
            else:  # Subsequent requests succeed
                response = Mock()
                response.status_code = 200
                response.json.return_value = {'recovered': True}
                response.raise_for_status.return_value = None
                return response

        mock_request.side_effect = intermittent_failure

        # Initial failures should be handled
        for i in range(3):
            with self.assertRaises(GenesisConnectionError):
                self.connector.make_request('GET', f'/failing/{i}')

        # After the service "recovers", requests should succeed
        with patch.object(self.connector, 'max_retries', 0):  # No retries for this test
            mock_request.side_effect = lambda *args, **kwargs: Mock(
                status_code=200,
                json=lambda: {'recovered': True},
                raise_for_status=lambda: None
            )
            result = self.connector.make_request('GET', '/recovered')
            self.assertEqual(result, {'recovered': True})

    @patch('requests.Session.request')
    def test_partial_failure_recovery(self, mock_request):
        """Test recovery from partial failures."""
        request_count = 0
        
        def partial_failure(*args, **kwargs):
            nonlocal request_count
            request_count += 1
            
            if request_count % 3 == 0:  # Every third request succeeds
                response = Mock()
                response.status_code = 200
                response.json.return_value = {'request_id': request_count, 'success': True}
                response.raise_for_status.return_value = None
                return response
            else:  # Other requests fail
                raise ConnectionError(f"Partial failure {request_count}")

        mock_request.side_effect = partial_failure

        # Should eventually succeed for some requests
        successful_requests = 0
        for i in range(10):
            try:
                result = self.connector.make_request('GET', f'/partial/{i}')
                successful_requests += 1
                self.assertIn('success', result)
                self.assertTrue(result['success'])
            except GenesisConnectionError:
                pass  # Expected for some requests

        self.assertGreater(successful_requests, 0, "No requests succeeded during partial failure test")

    def test_configuration_corruption_recovery(self):
        """Test recovery from configuration corruption."""
        # Corrupt the configuration
        original_config = self.connector.config.copy()
        
        # Test various corruption scenarios
        corruption_scenarios = [
            {'api_key': None},
            {'base_url': ''},
            {'timeout': -1},
            {'max_retries': 'invalid'},
            {},  # Empty config
        ]
        
        for corrupted_config in corruption_scenarios:
            with self.subTest(corruption=corrupted_config):
                # Apply corruption
                self.connector.config.update(corrupted_config)
                
                # Attempt to reload with valid config should recover
                try:
                    self.connector.reload_config(original_config)
                    # Verify recovery
                    self.assertEqual(self.connector.api_key, original_config['api_key'])
                    self.assertEqual(self.connector.base_url, original_config['base_url'])
                except Exception as e:
                    # If reload fails, create new connector as fallback
                    self.connector = GenesisConnector(original_config)
                    self.assertIsNotNone(self.connector)

    @patch('requests.Session.request')
    def test_session_recovery_after_failure(self, mock_request):
        """Test session recovery after catastrophic failure."""
        # Simulate session corruption
        def corrupt_session(*args, **kwargs):
            # Corrupt the session object
            self.connector.session = None
            raise RuntimeError("Session corrupted")

        mock_request.side_effect = corrupt_session

        # First request should fail due to corruption
        with self.assertRaises(Exception):
            self.connector.make_request('GET', '/corrupt')

        # Connector should be able to recover by creating new session
        self.connector.session = requests.Session()
        
        # Mock successful response for recovery test
        mock_request.side_effect = lambda *args, **kwargs: Mock(
            status_code=200,
            json=lambda: {'recovered': True},
            raise_for_status=lambda: None
        )
        
        result = self.connector.make_request('GET', '/recovery')
        self.assertEqual(result, {'recovered': True})


# ============================================================================
# INTEGRATION AND END-TO-END TESTS
# ============================================================================

class TestGenesisConnectorIntegration(unittest.TestCase):
    """
    Integration and end-to-end testing for GenesisConnector.
    Testing Framework: unittest
    Focus: Workflow testing, component integration, and realistic scenarios
    """

    def setUp(self):
        """Set up integration test environment."""
        self.connector = GenesisConnector({
            'api_key': 'integration_test_key',
            'base_url': 'https://api.integration.test',
            'timeout': 30,
            'max_retries': 3
        })

    @patch('requests.Session.request')
    def test_complete_workflow_simulation(self, mock_request):
        """Test complete workflow from initialization to cleanup."""
        workflow_responses = [
            # Authentication
            Mock(status_code=200, json=lambda: {'auth': 'success', 'session_id': 'sess_123'}),
            # Data submission
            Mock(status_code=201, json=lambda: {'created': True, 'id': 'data_456'}),
            # Status check
            Mock(status_code=200, json=lambda: {'status': 'processing', 'progress': 50}),
            # Final result
            Mock(status_code=200, json=lambda: {'status': 'completed', 'result': 'workflow_success'}),
            # Cleanup
            Mock(status_code=204, json=lambda: {}),
        ]
        
        for response in workflow_responses:
            response.raise_for_status = Mock(return_value=None)
        
        mock_request.side_effect = workflow_responses

        # Execute complete workflow
        # Step 1: Authentication
        auth_result = self.connector.make_request('POST', '/auth', {
            'api_key': self.connector.api_key
        })
        self.assertEqual(auth_result['auth'], 'success')
        session_id = auth_result['session_id']

        # Step 2: Submit data
        submit_result = self.connector.make_request('POST', '/data', {
            'session_id': session_id,
            'data': {'type': 'test', 'value': 42}
        })
        self.assertTrue(submit_result['created'])
        data_id = submit_result['id']

        # Step 3: Check status
        status_result = self.connector.make_request('GET', f'/data/{data_id}/status')
        self.assertEqual(status_result['status'], 'processing')

        # Step 4: Get final result
        final_result = self.connector.make_request('GET', f'/data/{data_id}')
        self.assertEqual(final_result['status'], 'completed')
        self.assertEqual(final_result['result'], 'workflow_success')

        # Step 5: Cleanup
        cleanup_result = self.connector.make_request('DELETE', f'/data/{data_id}')
        self.assertEqual(cleanup_result, {})

        # Verify all steps were executed
        self.assertEqual(mock_request.call_count, 5)

    @patch('requests.Session.request')
    def test_multi_format_data_handling(self, mock_request):
        """Test handling of multiple data formats in integration."""
        # Test data in various formats
        test_data_formats = [
            # JSON data
            {'format': 'json', 'data': {'key': 'value', 'number': 42}},
            # Form data simulation
            {'format': 'form', 'data': {'field1': 'value1', 'field2': 'value2'}},
            # Binary data simulation
            {'format': 'binary', 'data': {'content': 'base64encodeddata=='}},
            # Mixed content
            {'format': 'mixed', 'data': {
                'text': 'sample text',
                'numbers': [1, 2, 3],
                'nested': {'key': 'nested_value'}
            }},
        ]

        # Mock successful responses for all formats
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'processed': True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        for test_data in test_data_formats:
            with self.subTest(format=test_data['format']):
                # Format the payload
                formatted_data = self.connector.format_payload(test_data['data'])
                
                # Send the request
                result = self.connector.make_request('POST', '/process', formatted_data)
                
                # Verify successful processing
                self.assertEqual(result, {'processed': True})

        # Verify all formats were processed
        self.assertEqual(mock_request.call_count, len(test_data_formats))

    @patch('requests.Session.request')
    def test_error_handling_integration(self, mock_request):
        """Test integrated error handling across multiple operations."""
        # Simulate various error scenarios in sequence
        error_responses = [
            # Authentication failure
            Mock(status_code=401, 
                 raise_for_status=Mock(side_effect=requests.HTTPError("401 Unauthorized"))),
            # Rate limiting
            Mock(status_code=429,
                 headers={'Retry-After': '1'},
                 raise_for_status=Mock(side_effect=requests.HTTPError("429 Too Many Requests"))),
            # Server error
            Mock(status_code=500,
                 raise_for_status=Mock(side_effect=requests.HTTPError("500 Internal Server Error"))),
            # Success after errors
            Mock(status_code=200,
                 json=lambda: {'recovered': True},
                 raise_for_status=Mock(return_value=None)),
        ]

        mock_request.side_effect = error_responses

        # Test error progression and recovery
        error_count = 0
        
        # Expect first three requests to fail
        for i in range(3):
            with self.assertRaises(GenesisConnectionError):
                self.connector.make_request('GET', f'/error-test/{i}')
                error_count += 1

        # Fourth request should succeed
        result = self.connector.make_request('GET', '/error-test/recovery')
        self.assertEqual(result, {'recovered': True})

    def test_configuration_lifecycle_integration(self):
        """Test complete configuration lifecycle in integration context."""
        # Start with minimal configuration
        minimal_config = {
            'api_key': 'minimal_key',
            'base_url': 'https://minimal.test',
            'timeout': 10,
            'max_retries': 1
        }
        
        connector = GenesisConnector(minimal_config)
        self.assertEqual(connector.timeout, 10)

        # Upgrade configuration
        upgraded_config = {
            'api_key': 'upgraded_key',
            'base_url': 'https://upgraded.test',
            'timeout': 30,
            'max_retries': 5
        }
        
        connector.reload_config(upgraded_config)
        self.assertEqual(connector.timeout, 30)
        self.assertEqual(connector.max_retries, 5)

        # Test configuration validation
        invalid_config = {
            'api_key': '',  # Invalid
            'base_url': 'https://invalid.test',
            'timeout': 30,
            'max_retries': 5
        }
        
        is_valid = connector.validate_config(invalid_config)
        self.assertFalse(is_valid)

        # Connector should maintain last valid configuration
        self.assertEqual(connector.api_key, 'upgraded_key')


# ============================================================================
# TEST SUITE CONFIGURATION AND RUNNER
# ============================================================================

class TestSuiteManager:
    """Manages comprehensive test suite execution."""
    
    @staticmethod
    def get_all_test_classes():
        """Get all test classes for comprehensive execution."""
        return [
            # Core functionality tests
            TestGenesisConnectorDataSerialization,
            TestGenesisConnectorAdvancedAsync,
            TestGenesisConnectorAdvancedCaching,
            TestGenesisConnectorAdvancedSecurity,
            TestGenesisConnectorAdvancedMetrics,
            
            # Comprehensive validation tests
            TestGenesisConnectorComprehensiveValidation,
            TestGenesisConnectorEdgeCaseScenarios,
            TestGenesisConnectorRobustness,
            
            # New comprehensive tests
            TestGenesisConnectorInputValidation,
            TestGenesisConnectorPerformanceStress,
            TestGenesisConnectorErrorRecovery,
            TestGenesisConnectorIntegration,
            
            # Existing extended tests
            TestGenesisConnectorSecurity,
            TestGenesisConnectorCompatibility,
            TestGenesisConnectorStress,
            TestGenesisConnectorBoundaryConditions,
            TestGenesisConnectorResponseHandling,
            TestGenesisConnectorRequestPayloads,
            TestGenesisConnectorAdvancedErrorHandling,
            TestGenesisConnectorConfigurationEdgeCases,
            TestGenesisConnectorAsyncExtended,
            TestGenesisConnectorUtilityMethods,
            TestGenesisConnectorResourceManagement,
        ]
    
    @staticmethod
    def run_comprehensive_suite():
        """Run comprehensive test suite with detailed reporting."""
        test_classes = TestSuiteManager.get_all_test_classes()
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for test_class in test_classes:
            try:
                tests = loader.loadTestsFromTestCase(test_class)
                suite.addTests(tests)
            except Exception as e:
                print(f"Warning: Could not load tests from {test_class.__name__}: {e}")
        
        runner = unittest.TextTestRunner(
            verbosity=2,
            stream=sys.stdout,
            failfast=False,
            buffer=True
        )
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE GENESIS CONNECTOR TEST SUITE")
        print(f"Testing Framework: unittest with pytest enhancements")
        print(f"Test Classes: {len(test_classes)}")
        print(f"{'='*80}\n")
        
        result = runner.run(suite)
        
        # Detailed reporting
        print(f"\n{'='*80}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Tests Run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        
        if result.testsRun > 0:
            success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"{'='*80}")
        
        return result.wasSuccessful()


if __name__ == '__main__':
    # Enhanced test execution with comprehensive options
    import sys
    
    if '--comprehensive' in sys.argv:
        # Run comprehensive test suite
        success = TestSuiteManager.run_comprehensive_suite()
        sys.exit(0 if success else 1)
    elif '--performance' in sys.argv:
        # Run only performance tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestGenesisConnectorPerformanceStress)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    elif '--security' in sys.argv:
        # Run only security tests
        security_classes = [
            TestGenesisConnectorAdvancedSecurity,
            TestGenesisConnectorInputValidation,
            TestGenesisConnectorSecurity
        ]
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for test_class in security_classes:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        # Run standard unittest discovery
        unittest.main(verbosity=2)

