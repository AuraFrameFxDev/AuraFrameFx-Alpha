import pytest
import os
import sys
from unittest.mock import MagicMock

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

@pytest.fixture
def mock_api_key():
    """
    Returns a static mock API key string for use in Genesis API integration tests.
    
    Returns:
        str: The mock API key.
    """
    return "test_api_key_12345"

@pytest.fixture
def mock_base_url():
    """
    Return a mock base URL string for the Genesis API used in testing.
    
    Returns:
        str: The mock Genesis API endpoint URL.
    """
    return "https://api.genesis.test"

@pytest.fixture
def sample_api_response():
    """
    Return a mock dictionary simulating a successful Genesis API chat completion response.
    
    The response includes metadata fields (`id`, `object`, `created`, `model`), a list of choices with an assistant message and finish reason, and token usage statistics. Useful for testing code that processes Genesis API responses.
    
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
    Return a mock dictionary simulating an error response from the Genesis API.
    
    The dictionary contains an `error` key with details including error type, message, parameter, and code, suitable for testing error handling in Genesis API integrations.
    
    Returns:
        dict: Mock error response with detailed error information.
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
    Autouse pytest fixture that sets and cleans up Genesis API environment variables for each test.
    
    Before each test, assigns test values to `GENESIS_API_KEY` and `GENESIS_BASE_URL`. After the test, removes these variables to ensure test isolation.
    """
    os.environ["GENESIS_API_KEY"] = "test_env_key"
    os.environ["GENESIS_BASE_URL"] = "https://api.genesis.test"
    yield
    # Cleanup
    if "GENESIS_API_KEY" in os.environ:
        del os.environ["GENESIS_API_KEY"]
    if "GENESIS_BASE_URL" in os.environ:
        del os.environ["GENESIS_BASE_URL"]