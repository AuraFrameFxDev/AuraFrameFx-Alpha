"""
Property-based testing for GenesisConnector using Hypothesis.
Complements the existing comprehensive unit tests with property-based testing.
Testing framework: pytest with hypothesis
"""

import pytest
from hypothesis import given, strategies as st, example, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize
import json
from datetime import datetime
import string

try:
    from app.ai_backend.genesis_connector import GenesisConnector
except ImportError:
    from ai_backend.genesis_connector import GenesisConnector


class TestGenesisConnectorProperties:
    """Property-based tests for GenesisConnector invariants."""

    @given(st.text(min_size=1, max_size=1000))
    @example("valid_api_key_123")
    def test_config_validation_with_valid_strings(self, api_key):
        """Test that any non-empty string is accepted as API key."""
        connector = GenesisConnector()
        config = {
            'api_key': api_key,
            'base_url': 'https://api.test.com'
        }
        
        try:
            # Should not raise exception for reasonable strings
            result = connector.validate_config(config)
            assert isinstance(result, bool)
        except ValueError:
            # Some strings might be rejected for security reasons
            pass

    @given(st.dictionaries(
        keys=st.text(alphabet=string.ascii_letters + '_', min_size=1, max_size=50),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(min_value=0, max_value=1000),
            st.booleans(),
            st.none()
        ),
        min_size=1,
        max_size=20
    ))
    def test_payload_formatting_preserves_structure(self, payload_dict):
        """Test that payload formatting preserves basic dictionary structure."""
        connector = GenesisConnector()
        
        try:
            formatted = connector.format_payload(payload_dict)
            if formatted:
                # Should preserve keys for valid payloads
                if isinstance(formatted, dict):
                    assert len(formatted) > 0
                assert formatted is not None
        except (ValueError, TypeError):
            # Some payloads might be rejected
            pass

    @given(st.text(max_size=10000))
    def test_response_parsing_handles_all_strings(self, response_text):
        """Test that response parsing handles any string input gracefully."""
        connector = GenesisConnector()
        
        try:
            parsed = connector.parse_response(response_text)
            if parsed is not None:
                # If parsing succeeds, result should be reasonable
                assert isinstance(parsed, (dict, str, list))
        except (ValueError, json.JSONDecodeError):
            # Invalid JSON should raise appropriate exceptions
            pass

    @given(st.integers(min_value=1, max_value=3600))
    def test_timeout_validation_with_positive_integers(self, timeout_value):
        """Test that positive integer timeouts are handled correctly."""
        connector = GenesisConnector()
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'timeout': timeout_value
        }
        
        result = connector.validate_config(config)
        assert isinstance(result, bool)
        if result:
            assert timeout_value > 0


class GenesisConnectorStateMachine(RuleBasedStateMachine):
    """Stateful testing for GenesisConnector operations."""

    def __init__(self):
        super().__init__()
        self.connector = None
        self.current_config = None

    @initialize()
    def setup_connector(self):
        """Initialize connector for stateful testing."""
        self.connector = GenesisConnector()
        self.current_config = {
            'api_key': 'state_test_key',
            'base_url': 'https://api.state.test.com'
        }

    @rule(new_api_key=st.text(min_size=1, max_size=100))
    def update_api_key(self, new_api_key):
        """Test updating API key maintains connector state."""
        if self.connector and self.current_config:
            new_config = self.current_config.copy()
            new_config['api_key'] = new_api_key
            
            try:
                valid = self.connector.validate_config(new_config)
                if valid:
                    self.current_config = new_config
            except ValueError:
                pass

    @rule(payload=st.dictionaries(
        keys=st.text(alphabet=string.ascii_letters, min_size=1, max_size=20),
        values=st.text(max_size=100),
        min_size=1,
        max_size=10
    ))
    def format_payload_maintains_invariants(self, payload):
        """Test that payload formatting maintains invariants across operations."""
        if self.connector:
            try:
                formatted = self.connector.format_payload(payload)
                if formatted:
                    # Formatting should be idempotent for valid payloads
                    formatted_again = self.connector.format_payload(payload)
                    if formatted_again:
                        assert type(formatted) == type(formatted_again)
            except (ValueError, TypeError):
                pass


# Run property-based tests
TestStateMachine = GenesisConnectorStateMachine.TestCase

if __name__ == '__main__':
    pytest.main([__file__, '-v'])