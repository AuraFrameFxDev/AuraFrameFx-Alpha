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
    Return a mock Genesis API base URL string for use in tests.
    """
    return "https://api.genesis.test"

@pytest.fixture
def sample_api_response():
    """
    Return a mock dictionary simulating a successful Genesis API chat completion response.
    
    Returns:
        dict: Contains response metadata, an assistant message, finish reason, and token usage statistics.
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
    Return a mock dictionary simulating a typical error response from the Genesis API.
    
    Returns:
        dict: A dictionary containing error details, including error type, message, parameter, and code.
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
    Autouse fixture that sets test values for Genesis API environment variables before each test and removes them afterward to ensure test isolation.
    """
    os.environ["GENESIS_API_KEY"] = "test_env_key"
    os.environ["GENESIS_BASE_URL"] = "https://api.genesis.test"
    yield
    # Cleanup
    if "GENESIS_API_KEY" in os.environ:
        del os.environ["GENESIS_API_KEY"]
    if "GENESIS_BASE_URL" in os.environ:
        del os.environ["GENESIS_BASE_URL"]