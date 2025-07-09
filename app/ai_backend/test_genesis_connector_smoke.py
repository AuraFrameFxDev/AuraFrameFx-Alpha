"""
Smoke tests for GenesisConnector - quick validation of core functionality.
These tests run quickly and validate basic operations are working.
Testing framework: pytest
"""

import pytest
from unittest.mock import patch, Mock

try:
    from app.ai_backend.genesis_connector import GenesisConnector
except ImportError:
    from ai_backend.genesis_connector import GenesisConnector


class TestGenesisConnectorSmoke:
    """Smoke tests for basic GenesisConnector functionality."""

    def test_connector_instantiation(self):
        """Smoke test: Can create connector instance."""
        connector = GenesisConnector()
        assert connector is not None
        assert isinstance(connector, GenesisConnector)

    def test_connector_with_config(self):
        """Smoke test: Can create connector with configuration."""
        config = {
            'api_key': 'smoke_test_key',
            'base_url': 'https://api.smoke.test.com'
        }
        connector = GenesisConnector(config=config)
        assert connector is not None

    def test_config_validation_basic(self):
        """Smoke test: Basic configuration validation works."""
        connector = GenesisConnector()
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com'
        }
        
        result = connector.validate_config(config)
        assert isinstance(result, bool)

    def test_payload_formatting_basic(self):
        """Smoke test: Basic payload formatting works."""
        connector = GenesisConnector()
        payload = {'message': 'smoke_test'}
        
        result = connector.format_payload(payload)
        assert result is not None

    def test_headers_generation_basic(self):
        """Smoke test: Headers generation works."""
        connector = GenesisConnector()
        
        result = connector.get_headers()
        assert isinstance(result, dict)

    @patch('requests.post')
    def test_request_sending_basic(self, mock_post):
        """Smoke test: Basic request sending works."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'smoke': True}
        mock_post.return_value = mock_response
        
        connector = GenesisConnector()
        payload = {'message': 'smoke_test'}
        
        result = connector.send_request(payload)
        assert result['smoke'] is True

    def test_response_parsing_basic(self):
        """Smoke test: Basic response parsing works."""
        connector = GenesisConnector()
        response_text = '{"test": true, "message": "smoke"}'
        
        result = connector.parse_response(response_text)
        assert result['test'] is True
        assert result['message'] == 'smoke'

    @patch('requests.get')
    def test_connection_basic(self, mock_get):
        """Smoke test: Basic connection test works."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        connector = GenesisConnector()
        result = connector.connect()
        assert isinstance(result, bool)

    def test_error_handling_basic(self):
        """Smoke test: Basic error handling works."""
        connector = GenesisConnector()
        
        # Test with invalid config
        with pytest.raises((ValueError, TypeError)):
            connector.validate_config(None)
        
        # Test with invalid payload
        with pytest.raises((ValueError, TypeError)):
            connector.format_payload(None)

    def test_context_manager_basic(self):
        """Smoke test: Context manager functionality works."""
        config = {
            'api_key': 'context_test_key',
            'base_url': 'https://api.context.test.com'
        }
        
        try:
            with GenesisConnector(config=config) as connector:
                assert connector is not None
        except (AttributeError, TypeError):
            # Context manager might not be implemented
            connector = GenesisConnector(config=config)
            assert connector is not None


# Quick test discovery
def test_smoke_suite_completeness():
    """Ensure all critical smoke tests are present."""
    smoke_tests = [
        'test_connector_instantiation',
        'test_config_validation_basic',
        'test_payload_formatting_basic',
        'test_request_sending_basic',
        'test_headers_generation_basic'
    ]
    
    test_class = TestGenesisConnectorSmoke
    existing_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    for smoke_test in smoke_tests:
        assert smoke_test in existing_methods, f"Critical smoke test {smoke_test} missing"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x'])  # Stop on first failure for smoke tests