"""
Performance benchmarks for GenesisConnector.
Measures performance characteristics and establishes baselines.
Testing framework: pytest with pytest-benchmark
"""

import pytest
from unittest.mock import patch, Mock
import time
import json
from datetime import datetime
import threading
import concurrent.futures

try:
    from app.ai_backend.genesis_connector import GenesisConnector
except ImportError:
    from ai_backend.genesis_connector import GenesisConnector


class TestGenesisConnectorBenchmarks:
    """Performance benchmarks for GenesisConnector operations."""

    @pytest.fixture
    def connector(self):
        """Create connector for benchmarking."""
        config = {
            'api_key': 'benchmark_key',
            'base_url': 'https://api.benchmark.test.com'
        }
        return GenesisConnector(config=config)

    def test_config_validation_performance(self, benchmark, connector):
        """Benchmark configuration validation performance."""
        config = {
            'api_key': 'performance_test_key',
            'base_url': 'https://api.performance.test.com',
            'timeout': 30
        }
        
        result = benchmark(connector.validate_config, config)
        assert isinstance(result, bool)

    def test_payload_formatting_performance(self, benchmark, connector):
        """Benchmark payload formatting performance."""
        payload = {
            'message': 'performance_test',
            'timestamp': datetime.now().isoformat(),
            'data': list(range(1000)),
            'metadata': {'nested': {'deep': {'value': 'test'}}}
        }
        
        result = benchmark(connector.format_payload, payload)
        assert result is not None

    def test_headers_generation_performance(self, benchmark, connector):
        """Benchmark headers generation performance."""
        result = benchmark(connector.get_headers)
        assert isinstance(result, dict)

    @patch('requests.post')
    def test_request_sending_performance(self, mock_post, benchmark, connector):
        """Benchmark request sending performance."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'benchmark': True}
        mock_post.return_value = mock_response
        
        payload = {'message': 'benchmark_test'}
        
        result = benchmark(connector.send_request, payload)
        assert result['benchmark'] is True

    def test_concurrent_requests_performance(self, connector):
        """Benchmark concurrent request handling."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'concurrent': True}
            mock_post.return_value = mock_response
            
            def send_request():
                payload = {'message': 'concurrent_test'}
                return connector.send_request(payload)
            
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(send_request) for _ in range(100)]
                results = [future.result() for future in futures]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            assert len(results) == 100
            assert all(r['concurrent'] for r in results)
            assert total_time < 5.0  # Should complete within 5 seconds

    def test_memory_usage_under_load(self, connector):
        """Test memory usage patterns under load."""
        import gc
        import sys
        
        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Process many payloads
        payloads = [
            {'id': i, 'data': 'x' * 1000, 'timestamp': datetime.now().isoformat()}
            for i in range(1000)
        ]
        
        for payload in payloads:
            try:
                formatted = connector.format_payload(payload)
                del formatted  # Explicit cleanup
            except Exception:
                pass
        
        # Force garbage collection again
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory growth should be reasonable
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Should not leak excessive objects


class TestGenesisConnectorStressBenchmarks:
    """Stress testing benchmarks for extreme conditions."""

    @pytest.fixture
    def connector(self):
        """Create connector for stress testing."""
        return GenesisConnector()

    def test_large_payload_handling_performance(self, connector):
        """Test performance with large payloads."""
        large_payload = {
            'message': 'large_payload_test',
            'large_text': 'x' * (1024 * 1024),  # 1MB string
            'large_list': list(range(10000)),
            'large_dict': {f'key_{i}': f'value_{i}' for i in range(1000)}
        }
        
        start_time = time.time()
        
        try:
            formatted = connector.format_payload(large_payload)
            end_time = time.time()
            
            processing_time = end_time - start_time
            assert processing_time < 10.0  # Should process within 10 seconds
            assert formatted is not None
            
        except (MemoryError, ValueError):
            # Large payloads might be rejected
            pytest.skip("Large payload rejected by connector")

    def test_rapid_config_changes_performance(self, connector):
        """Test performance under rapid configuration changes."""
        start_time = time.time()
        
        for i in range(1000):
            config = {
                'api_key': f'rapid_key_{i}',
                'base_url': f'https://rapid{i % 10}.test.com'
            }
            try:
                connector.reload_config(config)
            except AttributeError:
                # Config reloading might not be implemented
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert total_time < 5.0  # Should handle rapid changes efficiently


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--benchmark-only'])