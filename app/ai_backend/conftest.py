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
        str: The mock API key value.
    """
    return "test_api_key_12345"

@pytest.fixture
def mock_base_url():
    """
    Provides a static mock base URL for the Genesis API for use in integration tests.
    
    Returns:
        str: The mock Genesis API endpoint URL.
    """
    return "https://api.genesis.test"

@pytest.fixture
def sample_api_response():
    """
    Return a dictionary that mimics a successful Genesis API chat completion response.
    
    The mock response includes metadata, a list of choices with an assistant message and finish reason, and token usage statistics. Intended for use in tests that require a realistic Genesis API response format.
    
    Returns:
        dict: A simulated Genesis API chat completion response.
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
    Return a dictionary simulating an error response from the Genesis API.
    
    The returned dictionary contains an `error` object with fields for error type, message, parameter, and code, useful for testing error handling scenarios in Genesis API integrations.
    
    Returns:
        dict: Mock Genesis API error response.
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
    Automatically sets up and cleans up Genesis API environment variables for each test.
    
    This autouse pytest fixture assigns test values to the `GENESIS_API_KEY` and `GENESIS_BASE_URL` environment variables before each test runs, and removes them after the test completes to maintain test isolation.
    """
    os.environ["GENESIS_API_KEY"] = "test_env_key"
    os.environ["GENESIS_BASE_URL"] = "https://api.genesis.test"
    yield
    # Cleanup
    if "GENESIS_API_KEY" in os.environ:
        del os.environ["GENESIS_API_KEY"]
    if "GENESIS_BASE_URL" in os.environ:
        del os.environ["GENESIS_BASE_URL"]