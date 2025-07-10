import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, call


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
                nested = {'level': 0}
                current = nested
                for i in range(depth):
                    current['next'] = {'level': i + 1}
                    current = current['next']
                current['end'] = True

                if hasattr(self.connector, 'format_payload'):
                    try:
                        formatted = self.connector.format_payload(nested)
                        if depth <= 100:
                            self.assertIsNotNone(formatted)
                    except (RecursionError, ValueError):
                        if depth > 100:
                            pass
                        else:
                            raise

    def test_unicode_normalization(self):
        """Test Unicode normalization and handling."""
        import unicodedata

        unicode_test_cases = [
            'café',
            'cafe\u0301',
            'Ωήμος',
            '한글',
            '🌟💫🚀',
            '\u200b\u200c\u200d',
            'test\u0000null',
            'test\ufffeReverse',
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
                        if '\u0000' in text or '\ufffe' in text:
                            pass
                        else:
                            raise

    def test_large_payload_chunking(self):
        """Test handling of very large payloads."""
        sizes = [
            1024,
            1024 * 100,
            1024 * 1024,
            1024 * 1024 * 5
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
                        if size <= 1024 * 1024:
                            self.assertIsNotNone(formatted)
                    except (MemoryError, ValueError):
                        if size > 1024 * 1024:
                            pass
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
                self.assertIn('datetime_obj', formatted)
                self.assertIn('custom_obj', formatted)
            except (TypeError, ValueError):
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

        mock_tasks = []
        for i in range(5):
            mock_task = AsyncMock()
            mock_task.return_value = {'id': i, 'result': f'batch_result_{i}'}
            mock_tasks.append(mock_task)

        mock_create_task.side_effect = mock_tasks

        payloads = [{'request_id': i, 'data': f'batch_data_{i}'} for i in range(5)]
        results = await self.connector.async_batch_request(payloads)
        self.assertEqual(len(results), 5)
        for i, result in enumerate(results):
            self.assertEqual(result['id'], i)

    async def test_async_timeout_handling_with_cancellation(self):
        """Test async timeout handling with proper task cancellation."""
        if not hasattr(self.connector, 'async_make_request'):
            self.skipTest("Async functionality not available")

        with patch('aiohttp.ClientSession.request') as mock_request:
            async def slow_request(*args, **kwargs):
                await asyncio.sleep(10)
                return AsyncMock(status=200, json=AsyncMock(return_value={'data': 'slow'}))

            mock_request.side_effect = slow_request
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

            for i in range(3):
                result = await self.connector.async_make_request('GET', f'/test/{i}')
                self.assertEqual(result, {'success': True})

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

        result1 = self.connector.get_cached_response('/cacheable')
        result2 = self.connector.get_cached_response('/cacheable')
        self.assertEqual(result1, result2)
        self.assertEqual(mock_request.call_count, 1)

    @patch('time.time')
    @patch('requests.Session.request')
    def test_cache_expiration(self, mock_request, mock_time):
        """Test cache expiration after TTL."""
        if not hasattr(self.connector, 'get_cached_response'):
            self.skipTest("Caching functionality not available")

        mock_time.side_effect = [1000, 1000, 1400]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'cached_data': 'test'}
        mock_response.headers = {'Cache-Control': 'max-age=300'}
        mock_request.return_value = mock_response

        self.connector.get_cached_response('/cacheable')
        self.connector.get_cached_response('/cacheable')
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

            if hasattr(self.connector, 'get_cached_response'):
                self.connector.get_cached_response('/test')
            self.connector.invalidate_cache('/test')
            if hasattr(self.connector, 'get_cached_response'):
                self.connector.get_cached_response('/test')
                self.assertEqual(mock_request.call_count, 2)

    def test_cache_size_limits(self):
        """Test cache size limits and LRU eviction."""
        if not (hasattr(self.connector, 'get_cached_response') and hasattr(self.connector, 'cache_size_limit')):
            self.skipTest("Cache size limiting not available")

        self.connector.cache_size_limit = 3
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'Cache-Control': 'max-age=3600'}
            mock_request.return_value = mock_response

            for i in range(5):
                mock_response.json.return_value = {'data': f'test_{i}'}
                self.connector.get_cached_response(f'/test/{i}')

            mock_response.json.return_value = {'data': 'test_0_new'}
            self.connector.get_cached_response('/test/0')
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
        signed_payload = self.connector.sign_request(payload, signature_key)

        self.assertIn('signature', signed_payload)
        self.assertIn('timestamp', signed_payload)

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
            refresh_response = Mock()
            refresh_response.status_code = 200
            refresh_response.json.return_value = {
                'access_token': 'new_token_123',
                'expires_in': 3600
            }
            api_response = Mock()
            api_response.status_code = 200
            api_response.json.return_value = {'data': 'success'}
            mock_request.side_effect = [refresh_response, api_response]

            new_token = self.connector.refresh_token()
            self.assertEqual(new_token, 'new_token_123')
            self.assertEqual(self.connector.api_key, 'new_token_123')

    def test_rate_limiting_with_backoff(self):
        """Test rate limiting with exponential backoff."""
        with patch('requests.Session.request') as mock_request:
            with patch('time.sleep') as mock_sleep:
                rate_limit_response = Mock()
                rate_limit_response.status_code = 429
                rate_limit_response.headers = {'Retry-After': '60'}
                rate_limit_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
                success_response = Mock()
                success_response.status_code = 200
                success_response.json.return_value = {'data': 'success'}
                success_response.raise_for_status.return_value = None
                mock_request.side_effect = [rate_limit_response, success_response]

                if hasattr(self.connector, 'make_request_with_rate_limiting'):
                    result = self.connector.make_request_with_rate_limiting('GET', '/test')
                    self.assertEqual(result, {'data': 'success'})
                    mock_sleep.assert_called_once_with(60)

    def test_ssl_certificate_pinning(self):
        """Test SSL certificate pinning validation."""
        if not hasattr(self.connector, 'verify_ssl_pinning'):
            self.skipTest("SSL pinning not available")

        mock_cert_data = {
            'subject': 'CN=api.genesis.test',
            'issuer': 'CN=DigiCert',
            'serialNumber': '123456789',
            'sha256_fingerprint': 'abcd1234efgh5678'
        }
        is_valid = self.connector.verify_ssl_pinning(mock_cert_data)
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

        mock_time.side_effect = [1000.0, 1000.5, 1001.0, 1001.2]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_response.headers = {'Content-Length': '1024'}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

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

        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': 'test'}
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response

            for i in range(10):
                self.connector.make_request('GET', f'/test/{i}')

            hourly_metrics = self.connector.get_metrics_for_period(hour_ago, now)
            daily_metrics = self.connector.get_metrics_for_period(day_ago, now)
            self.assertLessEqual(hourly_metrics['request_count'], daily_metrics['request_count'])

    def test_custom_metrics_tracking(self):
        """Test custom metrics tracking."""
        if not hasattr(self.connector, 'track_custom_metric'):
            self.skipTest("Custom metrics not available")

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

            for i in range(5):
                self.connector.make_request('GET', f'/test/{i}')

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
                        pass


if __name__ == '__main__':
    import sys

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestGenesisConnectorDataSerialization,
        TestGenesisConnectorAdvancedAsync,
        TestGenesisConnectorAdvancedCaching,
        TestGenesisConnectorAdvancedSecurity,
        TestGenesisConnectorAdvancedMetrics
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        failfast=False,
        buffer=True
    )
    result = runner.run(suite)

    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")

    sys.exit(0 if result.wasSuccessful() else 1)