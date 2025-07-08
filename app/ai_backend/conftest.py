import pytest
import os
import sys
from unittest.mock import MagicMock

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

@pytest.fixture
def mock_api_key():
    """
    Return a fixed mock API key string for use in tests.
    """
    return "test_api_key_12345"

@pytest.fixture
def mock_base_url():
    """
    Provides a mock base URL string for use in API client tests.
    
    Returns:
        str: The mock base URL.
    """
    return "https://api.genesis.test"

@pytest.fixture
def sample_api_response():
    """
    Return a dictionary simulating a successful Genesis API chat completion response for use in tests.
    
    Returns:
        dict: A sample API response containing metadata, message content, and token usage details.
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
    Return a dictionary representing a sample error response from the API for use in tests.
    
    Returns:
        dict: A mock error response containing error type, message, parameter, and code fields.
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
    Automatically sets and cleans up environment variables required for Genesis API testing.
    
    This fixture sets `GENESIS_API_KEY` and `GENESIS_BASE_URL` to test values before each test and removes them afterward to ensure isolated test environments.
    """
    os.environ["GENESIS_API_KEY"] = "test_env_key"
    os.environ["GENESIS_BASE_URL"] = "https://api.genesis.test"
    yield
    # Cleanup
    if "GENESIS_API_KEY" in os.environ:
        del os.environ["GENESIS_API_KEY"]
    if "GENESIS_BASE_URL" in os.environ:
        del os.environ["GENESIS_BASE_URL"]