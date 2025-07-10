"""
Pytest configuration and fixtures for GenesisConnector tests.
"""
import pytest
import asyncio
from unittest.mock import Mock

@pytest.fixture
def mock_genesis_config():
    """Fixture providing a mock GenesisConnector configuration."""
    return {
        'api_key': 'test_api_key_123',
        'base_url': 'https://api.genesis.test',
        'timeout': 30,
        'max_retries': 3
    }

@pytest.fixture
def genesis_connector(mock_genesis_config):
    """Fixture providing a GenesisConnector instance or mock."""
    try:
        from app.ai_backend.genesis_connector import GenesisConnector
        return GenesisConnector(mock_genesis_config)
    except ImportError:
        # Return a mock if the actual class isn't available
        mock = Mock()
        mock.config = mock_genesis_config
        return mock

@pytest.fixture
def event_loop():
    """Fixture providing an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Fixture to ensure cleanup after each test."""
    yield
    # Cleanup logic here if needed
    import gc
    gc.collect()

# Pytest markers
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )
    config.addinivalue_line(
        "markers", "async_test: mark test as async test"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add appropriate markers."""
    for item in items:
        # Mark tests based on their class or method names
        if "Integration" in item.cls.__name__ if item.cls else False:
            item.add_marker(pytest.mark.integration)
        if "Performance" in item.cls.__name__ if item.cls else False:
            item.add_marker(pytest.mark.performance)
        if "Security" in item.cls.__name__ if item.cls else False:
            item.add_marker(pytest.mark.security)
        if "Async" in item.cls.__name__ if item.cls else False:
            item.add_marker(pytest.mark.async_test)