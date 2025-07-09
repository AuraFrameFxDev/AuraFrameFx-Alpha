import pytest
import os
import sys
from unittest.mock import MagicMock

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

@pytest.fixture
def mock_api_key():
    """
    Return a static mock API key string for use in Genesis API integration tests.
    
    Returns:
        str: The mock API key.
    """
    return "test_api_key_12345"

@pytest.fixture
def mock_base_url():
    """
    Provides a mock base URL for the Genesis API to be used in integration tests.
    
    Returns:
        str: A simulated Genesis API endpoint URL for testing purposes.
    """
    return "https://api.genesis.test"

@pytest.fixture
def sample_api_response():
    """
    Provides a mock dictionary simulating a successful Genesis API chat completion response.
    
    The returned dictionary includes metadata such as response ID, object type, creation timestamp, model name, a list of choices with an assistant message and finish reason, and token usage statistics. Useful for integration tests requiring a representative Genesis API response.
    
    Returns:
        dict: Mocked Genesis API chat completion response.
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
    Return a mock dictionary representing an error response from the Genesis API.
    
    The response includes error details such as type, message, parameter, and code, suitable for testing error handling in API integrations.
    
    Returns:
        dict: Simulated error response with detailed error information.
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
    Pytest autouse fixture that configures and cleans up environment variables for Genesis API integration tests.
    
    Sets the `GENESIS_API_KEY` and `GENESIS_BASE_URL` environment variables to test values before each test, and removes them after the test completes to ensure test isolation.
    """
    os.environ["GENESIS_API_KEY"] = "test_env_key"
    os.environ["GENESIS_BASE_URL"] = "https://api.genesis.test"
    yield
    # Cleanup
    if "GENESIS_API_KEY" in os.environ:
        del os.environ["GENESIS_API_KEY"]
    if "GENESIS_BASE_URL" in os.environ:
        del os.environ["GENESIS_BASE_URL"]