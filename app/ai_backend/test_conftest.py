"""
Test configuration module for AI Backend testing.

This module provides pytest fixtures and configuration specifically
for testing the AI backend components. It complements the main conftest.py
by providing additional mock objects and test utilities.

Testing Framework: pytest with pytest-asyncio
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import Dict, Any, Generator, List
import tempfile
import os
import json
from pathlib import Path
from datetime import datetime, timedelta


# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_ai_config():
    """Provide mock AI configuration for testing."""
    return {
        "model_name": "test-model",
        "api_key": "test-key-123",
        "endpoint": "https://test-api.example.com",
        "timeout": 30,
        "max_retries": 3,
        "temperature": 0.7,
        "max_tokens": 1024,
        "api_version": "v1"
    }


@pytest.fixture
def mock_genesis_core():
    """Provide mock Genesis core instance for testing."""
    mock_core = MagicMock()
    mock_core.initialize = AsyncMock(return_value=True)
    mock_core.shutdown = AsyncMock(return_value=True)
    mock_core.is_active = True
    mock_core.version = "1.0.0"
    mock_core.config = {"model": "test", "temperature": 0.7}
    mock_core.get_status = Mock(return_value="active")
    return mock_core


@pytest.fixture
def mock_consciousness_matrix():
    """Provide mock consciousness matrix for testing."""
    mock_matrix = MagicMock()
    mock_matrix.activate = AsyncMock(return_value=True)
    mock_matrix.deactivate = AsyncMock(return_value=True)
    mock_matrix.process = AsyncMock(return_value={"status": "processed", "confidence": 0.95})
    mock_matrix.state = "active"
    mock_matrix.dimension = 512
    return mock_matrix


@pytest.fixture
def mock_ethical_governor():
    """Provide mock ethical governor for testing."""
    mock_governor = MagicMock()
    mock_governor.evaluate = AsyncMock(return_value={"ethical_score": 0.85, "approved": True})
    mock_governor.set_constraints = Mock()
    mock_governor.is_enabled = True
    mock_governor.policies = ["privacy", "safety", "fairness"]
    return mock_governor


@pytest.fixture
def mock_evolutionary_conduit():
    """Provide mock evolutionary conduit for testing."""
    mock_conduit = MagicMock()
    mock_conduit.evolve = AsyncMock(return_value={"evolution_score": 0.92, "mutations": 3})
    mock_conduit.adapt = AsyncMock(return_value=True)
    mock_conduit.generation = 1
    mock_conduit.fitness_score = 0.88
    return mock_conduit


@pytest.fixture
def temp_directory():
    """Provide temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_test_data():
    """Provide sample test data for various testing scenarios."""
    return {
        "user_input": "Hello, how are you?",
        "expected_output": "I'm doing well, thank you for asking!",
        "metadata": {
            "timestamp": "2024-01-01T12:00:00Z",
            "user_id": "test-user-123",
            "session_id": "test-session-456",
            "request_id": "req-789"
        },
        "context": {
            "conversation_history": [],
            "user_preferences": {"language": "en", "tone": "friendly"},
            "system_state": "active"
        }
    }


@pytest.fixture
def mock_api_response():
    """Provide mock API response for testing external API calls."""
    return {
        "status_code": 200,
        "headers": {"Content-Type": "application/json", "X-Request-ID": "test-123"},
        "json": {
            "success": True,
            "data": {
                "response": "Mock API response",
                "confidence": 0.95,
                "processing_time": 0.123,
                "model_version": "1.0.0"
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "tokens_used": 50
            }
        }
    }


@pytest.fixture
def mock_database_connection():
    """Provide mock database connection for testing."""
    mock_db = MagicMock()
    mock_db.connect = AsyncMock(return_value=True)
    mock_db.disconnect = AsyncMock(return_value=True)
    mock_db.execute = AsyncMock(return_value={"rows_affected": 1})
    mock_db.fetch = AsyncMock(return_value=[{"id": 1, "data": "test", "created_at": datetime.now()}])
    mock_db.fetch_one = AsyncMock(return_value={"id": 1, "data": "test"})
    mock_db.is_connected = True
    mock_db.transaction = AsyncMock()
    return mock_db


@pytest.fixture
def mock_genesis_profile():
    """Provide mock Genesis profile for testing."""
    mock_profile = MagicMock()
    mock_profile.load_profile = AsyncMock(return_value=True)
    mock_profile.save_profile = AsyncMock(return_value=True)
    mock_profile.get_capabilities = Mock(return_value=["reasoning", "creativity", "analysis"])
    mock_profile.personality_traits = {"creativity": 0.8, "logic": 0.9, "empathy": 0.7}
    mock_profile.name = "TestProfile"
    return mock_profile


@pytest.fixture
def mock_genesis_api():
    """Provide mock Genesis API for testing."""
    mock_api = MagicMock()
    mock_api.process_request = AsyncMock(return_value={"response": "Test response", "status": "success"})
    mock_api.validate_input = Mock(return_value=True)
    mock_api.format_response = Mock(return_value={"formatted": True})
    mock_api.is_available = True
    mock_api.rate_limit = {"requests_per_minute": 100, "current_usage": 10}
    return mock_api


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set up test environment for each test."""
    # Set test environment variables
    original_env = os.environ.copy()
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["GENESIS_ENV"] = "test"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_logger():
    """Provide mock logger for testing logging functionality."""
    mock_log = MagicMock()
    mock_log.info = Mock()
    mock_log.warning = Mock()
    mock_log.error = Mock()
    mock_log.debug = Mock()
    mock_log.critical = Mock()
    mock_log.exception = Mock()
    return mock_log


@pytest.fixture
def sample_ethical_data():
    """Provide sample ethical evaluation data."""
    return {
        "input_text": "Should I share personal information?",
        "ethical_concerns": ["privacy", "consent"],
        "risk_level": "medium",
        "recommendations": [
            "Ensure explicit consent",
            "Minimize data collection",
            "Implement data protection"
        ],
        "compliance_status": {
            "gdpr": True,
            "ccpa": True,
            "internal_policies": True
        }
    }


@pytest.fixture
def mock_file_handler():
    """Provide mock file handler for testing file operations."""
    mock_handler = MagicMock()
    mock_handler.read_file = AsyncMock(return_value="test content")
    mock_handler.write_file = AsyncMock(return_value=True)
    mock_handler.delete_file = AsyncMock(return_value=True)
    mock_handler.exists = Mock(return_value=True)
    mock_handler.get_size = Mock(return_value=1024)
    return mock_handler


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks tests as asyncio tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add asyncio marker to async test functions
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # Add unit marker to test functions by default
        if not any(marker.name in ['integration', 'slow', 'performance'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# Utility functions for testing
def create_mock_response(status_code=200, json_data=None, headers=None):
    """Create a mock HTTP response for testing."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = headers or {}
    mock_response.json.return_value = json_data or {}
    return mock_response


def assert_async_mock_called_with(async_mock, *args, **kwargs):
    """Assert that an async mock was called with specific arguments."""
    async_mock.assert_called_with(*args, **kwargs)


def assert_dict_contains_keys(dictionary, required_keys):
    """Assert that a dictionary contains all required keys."""
    for key in required_keys:
        assert key in dictionary, f"Missing required key: {key}"