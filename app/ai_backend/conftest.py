import pytest
import os
import sys
from unittest.mock import MagicMock

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

@pytest.fixture
def mock_api_key():
    """
    Provides a mock API key string for use in Genesis API integration tests.
    
    Returns:
        str: A static mock API key.
    """
    return "test_api_key_12345"

@pytest.fixture
def mock_base_url():
    """
    Return a mock Genesis API base URL string for use in tests.
    """
    return "https://api.genesis.test"

@pytest.fixture
def sample_api_response():
    """
    Return a mock dictionary representing a successful Genesis API chat completion response.
    
    The response includes metadata, an assistant message, and token usage details for use in tests.
    
    Returns:
        dict: Simulated API response data.
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
    Return a mock error response dictionary simulating a failed Genesis API request.
    
    Returns:
        dict: A dictionary with error details including type, message, parameter, and code.
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
    Temporarily sets the `GENESIS_API_KEY` and `GENESIS_BASE_URL` environment variables for the duration of a test, then removes them after the test completes.
    """
    os.environ["GENESIS_API_KEY"] = "test_env_key"
    os.environ["GENESIS_BASE_URL"] = "https://api.genesis.test"
    yield
    # Cleanup
    if "GENESIS_API_KEY" in os.environ:
        del os.environ["GENESIS_API_KEY"]
    if "GENESIS_BASE_URL" in os.environ:
        del os.environ["GENESIS_BASE_URL"]