import pytest
import os
import sys
from unittest.mock import MagicMock

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

@pytest.fixture
def mock_api_key():
    """
    Provides a static mock API key string for use in Genesis API integration tests.
    
    Returns:
        str: A fixed mock API key value.
    """
    return "test_api_key_12345"

@pytest.fixture
def mock_base_url():
    """
    Provides a fixed mock base URL representing the Genesis API endpoint for use in integration tests.
    
    Returns:
        str: Mock Genesis API endpoint URL.
    """
    return "https://api.genesis.test"

@pytest.fixture
def sample_api_response():
    """
    Provides a mock dictionary representing a successful Genesis API chat completion response.
    
    The returned dictionary includes metadata, a list of choices with an assistant message and finish reason, and token usage statistics. Useful for tests that require a realistic Genesis API response structure.
    
    Returns:
        dict: Simulated Genesis API chat completion response.
    """
    return {
        "id": "test_response_id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "genesis-1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from Genesis API"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 50,
            "total_tokens": 75
        }
    }

@pytest.fixture
def sample_error_response():
    """
    Provides a mock error response dictionary emulating a Genesis API error for use in tests.
    
    Returns:
        dict: Dictionary with an `error` object containing type, message, parameter, and code fields.
    """
    return {
        "error": {
            "type": "invalid_request_error",
            "message": "Invalid request parameters",
            "param": "model",
            "code": "invalid_model"
        }
    }

@pytest.fixture(autouse=True)
def mock_environment():
    """
    Sets Genesis API environment variables to test values before each test and removes them afterward to ensure test isolation.
    
    This autouse pytest fixture configures `GENESIS_API_KEY` and `GENESIS_BASE_URL` for each test run, then cleans them up to prevent cross-test contamination.
    """
    os.environ["GENESIS_API_KEY"] = "test_env_key"
    os.environ["GENESIS_BASE_URL"] = "https://api.genesis.test"
    yield
    # Cleanup
    if "GENESIS_API_KEY" in os.environ:
        del os.environ["GENESIS_API_KEY"]
    if "GENESIS_BASE_URL" in os.environ:
        del os.environ["GENESIS_BASE_URL"]