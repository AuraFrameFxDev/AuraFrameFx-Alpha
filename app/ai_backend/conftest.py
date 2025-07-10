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
    Provides a fixed mock base URL for the Genesis API endpoint used in integration tests.
    
    Returns:
        str: The mock Genesis API endpoint URL.
    """
    return "https://api.genesis.test"

@pytest.fixture
def sample_api_response():
    """
    Provides a dictionary that simulates a successful Genesis API chat completion response.
    
    The dictionary includes metadata, a list of choices with an assistant message and finish reason, and token usage statistics. Useful for tests requiring a realistic Genesis API response structure.
    
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
    Provides a dictionary simulating an error response from the Genesis API for testing purposes.
    
    Returns:
        dict: A dictionary containing an `error` object with type, message, parameter, and code fields.
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
    Ensures Genesis API environment variables are set to test values before each test and removed afterward for isolation.
    
    This autouse pytest fixture sets `GENESIS_API_KEY` and `GENESIS_BASE_URL` to predefined test values at the start of each test, then deletes them after the test completes to prevent cross-test contamination.
    """
    os.environ["GENESIS_API_KEY"] = "test_env_key"
    os.environ["GENESIS_BASE_URL"] = "https://api.genesis.test"
    yield
    # Cleanup
    if "GENESIS_API_KEY" in os.environ:
        del os.environ["GENESIS_API_KEY"]
    if "GENESIS_BASE_URL" in os.environ:
        del os.environ["GENESIS_BASE_URL"]