import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock, AsyncMock
import json
import asyncio
from datetime import datetime, timedelta
import sys
import os

# Add the app directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from app.ai_backend.genesis_api import (
    GenesisAPI,
    GenesisAPIError,
    GenesisAPITimeoutError,
    GenesisAPIAuthenticationError,
    GenesisAPIRateLimitError,
    parse_genesis_response,
    validate_genesis_request,
    format_genesis_prompt,
    handle_genesis_error
)


class TestGenesisAPI:
    """Test suite for GenesisAPI class"""
    
    @pytest.fixture
    def genesis_api(self):
        """
        Creates and returns a GenesisAPI instance configured for testing with preset API key, base URL, and timeout.
        """
        return GenesisAPI(
            api_key="test_api_key",
            base_url="https://api.genesis.test",
            timeout=30
        )
    
    @pytest.fixture
    def mock_response(self):
        """
        Create a mocked HTTP response object with a predefined JSON payload for testing purposes.
        
        Returns:
            MagicMock: A mock response object simulating a successful HTTP response with expected JSON structure.
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "test_id_123",
            "choices": [
                {
                    "message": {
                        "content": "Test response content"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75
            }
        }
        return mock_resp
    
    def test_genesis_api_initialization(self):
        """Test GenesisAPI initialization with valid parameters"""
        api = GenesisAPI(
            api_key="test_key",
            base_url="https://api.genesis.test",
            timeout=30
        )
        assert api.api_key == "test_key"
        assert api.base_url == "https://api.genesis.test"
        assert api.timeout == 30
        assert api.session is not None
    
    def test_genesis_api_initialization_with_defaults(self):
        """Test GenesisAPI initialization with default parameters"""
        api = GenesisAPI(api_key="test_key")
        assert api.api_key == "test_key"
        assert api.base_url == "https://api.genesis.ai"
        assert api.timeout == 60
    
    def test_genesis_api_initialization_missing_api_key(self):
        """Test GenesisAPI initialization fails without API key"""
        with pytest.raises(ValueError, match="API key is required"):
            GenesisAPI(api_key="")
    
    def test_genesis_api_initialization_invalid_url(self):
        """
        Test that initializing GenesisAPI with an invalid base URL raises a ValueError.
        """
        with pytest.raises(ValueError, match="Invalid base URL"):
            GenesisAPI(api_key="test_key", base_url="invalid_url")
    
    @patch('requests.Session.post')
    def test_make_request_success(self, mock_post, genesis_api, mock_response):
        """
        Tests that a successful API request returns the expected response structure and content.
        """
        mock_post.return_value = mock_response
        
        result = genesis_api.make_request("test prompt")
        
        assert result["id"] == "test_id_123"
        assert result["choices"][0]["message"]["content"] == "Test response content"
        mock_post.assert_called_once()
    
    @patch('requests.Session.post')
    def test_make_request_authentication_error(self, mock_post, genesis_api):
        """
        Test that `make_request` raises `GenesisAPIAuthenticationError` when the API responds with a 401 authentication error.
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": "Invalid API key"}
        mock_post.return_value = mock_resp
        
        with pytest.raises(GenesisAPIAuthenticationError):
            genesis_api.make_request("test prompt")
    
    @patch('requests.Session.post')
    def test_make_request_rate_limit_error(self, mock_post, genesis_api):
        """
        Test that `make_request` raises `GenesisAPIRateLimitError` when the API responds with a 429 status code.
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"error": "Rate limit exceeded"}
        mock_post.return_value = mock_resp
        
        with pytest.raises(GenesisAPIRateLimitError):
            genesis_api.make_request("test prompt")
    
    @patch('requests.Session.post')
    def test_make_request_timeout_error(self, mock_post, genesis_api):
        """
        Test that a timeout during an API request raises a GenesisAPITimeoutError.
        """
        mock_post.side_effect = TimeoutError("Request timed out")
        
        with pytest.raises(GenesisAPITimeoutError):
            genesis_api.make_request("test prompt")
    
    @patch('requests.Session.post')
    def test_make_request_server_error(self, mock_post, genesis_api):
        """
        Test that a server error response (HTTP 500) from the API raises a GenesisAPIError.
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "Internal server error"}
        mock_post.return_value = mock_resp
        
        with pytest.raises(GenesisAPIError):
            genesis_api.make_request("test prompt")
    
    @patch('requests.Session.post')
    def test_make_request_malformed_response(self, mock_post, genesis_api):
        """
        Test that `make_request` raises `GenesisAPIError` when the API returns a malformed JSON response.
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_resp
        
        with pytest.raises(GenesisAPIError):
            genesis_api.make_request("test prompt")
    
    def test_generate_headers(self, genesis_api):
        """
        Test that the GenesisAPI client generates correct HTTP headers for API requests.
        
        Verifies that the Authorization, Content-Type, and User-Agent headers are set as expected.
        """
        headers = genesis_api._generate_headers()
        
        assert headers["Authorization"] == "Bearer test_api_key"
        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"].startswith("GenesisAPI")
    
    def test_validate_prompt_valid(self, genesis_api):
        """
        Test that a valid prompt passes the GenesisAPI prompt validation.
        """
        valid_prompt = "This is a valid prompt"
        assert genesis_api._validate_prompt(valid_prompt) is True
    
    def test_validate_prompt_empty(self, genesis_api):
        """
        Test that validating an empty prompt raises a ValueError with the expected message.
        """
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            genesis_api._validate_prompt("")
    
    def test_validate_prompt_too_long(self, genesis_api):
        """
        Test that prompt validation raises a ValueError when the input prompt exceeds the maximum allowed length.
        """
        long_prompt = "x" * 10001
        with pytest.raises(ValueError, match="Prompt is too long"):
            genesis_api._validate_prompt(long_prompt)
    
    def test_validate_prompt_invalid_characters(self, genesis_api):
        """
        Test that prompt validation raises a ValueError when the prompt contains invalid characters.
        """
        invalid_prompt = "Test prompt with \x00 null character"
        with pytest.raises(ValueError, match="Invalid characters"):
            genesis_api._validate_prompt(invalid_prompt)


class TestGenesisAPIAsync:
    """Test suite for async GenesisAPI methods"""
    
    @pytest.fixture
    def genesis_api(self):
        """
        Creates and returns a GenesisAPI instance configured with a test API key for use in tests.
        """
        return GenesisAPI(api_key="test_api_key")
    
    @pytest.mark.asyncio
    async def test_make_request_async_success(self, genesis_api):
        """
        Test that a successful asynchronous API request returns the expected response structure.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "id": "async_test_id",
                "choices": [{"message": {"content": "Async response"}}]
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await genesis_api.make_request_async("test prompt")
            
            assert result["id"] == "async_test_id"
            assert result["choices"][0]["message"]["content"] == "Async response"
    
    @pytest.mark.asyncio
    async def test_make_request_async_timeout(self, genesis_api):
        """
        Test that `make_request_async` raises `GenesisAPITimeoutError` when an asynchronous API request times out.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError()
            
            with pytest.raises(GenesisAPITimeoutError):
                await genesis_api.make_request_async("test prompt")


class TestGenesisAPIUtilities:
    """Test suite for utility functions"""
    
    def test_parse_genesis_response_valid(self):
        """
        Test that a valid Genesis API response is correctly parsed to extract content and token usage.
        """
        response = {
            "id": "test_id",
            "choices": [
                {
                    "message": {
                        "content": "Test content"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        parsed = parse_genesis_response(response)
        
        assert parsed["content"] == "Test content"
        assert parsed["token_usage"]["total"] == 15
    
    def test_parse_genesis_response_missing_content(self):
        """
        Test that `parse_genesis_response` raises a ValueError when the response lacks content.
        """
        response = {
            "id": "test_id",
            "choices": [],
            "usage": {"total_tokens": 0}
        }
        
        with pytest.raises(ValueError, match="No content found"):
            parse_genesis_response(response)
    
    def test_parse_genesis_response_malformed(self):
        """
        Test that parsing a malformed Genesis API response raises a ValueError with the expected message.
        """
        response = {"invalid": "structure"}
        
        with pytest.raises(ValueError, match="Invalid response structure"):
            parse_genesis_response(response)
    
    def test_validate_genesis_request_valid(self):
        """
        Test that a valid Genesis API request passes validation without errors.
        """
        request = {
            "model": "genesis-1",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        assert validate_genesis_request(request) is True
    
    def test_validate_genesis_request_missing_model(self):
        """
        Test that `validate_genesis_request` raises a ValueError when the model field is missing from the request.
        """
        request = {
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 100
        }
        
        with pytest.raises(ValueError, match="Model is required"):
            validate_genesis_request(request)
    
    def test_validate_genesis_request_invalid_temperature(self):
        """
        Test that `validate_genesis_request` raises a ValueError when the temperature parameter is outside the allowed range (0 to 2).
        """
        request = {
            "model": "genesis-1",
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 2.5
        }
        
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            validate_genesis_request(request)
    
    def test_validate_genesis_request_invalid_max_tokens(self):
        """
        Test that `validate_genesis_request` raises a ValueError when `max_tokens` is not positive.
        """
        request = {
            "model": "genesis-1",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": -1
        }
        
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            validate_genesis_request(request)
    
    def test_format_genesis_prompt_simple(self):
        """
        Test that a simple prompt is correctly formatted into the expected message structure for the Genesis API.
        """
        prompt = "Hello, world!"
        formatted = format_genesis_prompt(prompt)
        
        assert formatted["messages"][0]["content"] == "Hello, world!"
        assert formatted["messages"][0]["role"] == "user"
    
    def test_format_genesis_prompt_with_system_message(self):
        """
        Test that `format_genesis_prompt` correctly includes a system message when provided, placing it before the user prompt in the formatted message structure.
        """
        prompt = "Hello, world!"
        system_message = "You are a helpful assistant."
        formatted = format_genesis_prompt(prompt, system_message=system_message)
        
        assert len(formatted["messages"]) == 2
        assert formatted["messages"][0]["role"] == "system"
        assert formatted["messages"][0]["content"] == system_message
        assert formatted["messages"][1]["role"] == "user"
        assert formatted["messages"][1]["content"] == "Hello, world!"
    
    def test_format_genesis_prompt_with_parameters(self):
        """
        Test that `format_genesis_prompt` correctly includes additional parameters such as temperature, max_tokens, and top_p in the formatted prompt output.
        """
        prompt = "Hello, world!"
        formatted = format_genesis_prompt(
            prompt,
            temperature=0.5,
            max_tokens=150,
            top_p=0.9
        )
        
        assert formatted["temperature"] == 0.5
        assert formatted["max_tokens"] == 150
        assert formatted["top_p"] == 0.9
    
    def test_handle_genesis_error_authentication(self):
        """
        Test that `handle_genesis_error` raises `GenesisAPIAuthenticationError` when an authentication error response with status code 401 is received.
        """
        error_response = {
            "error": {
                "type": "authentication_error",
                "message": "Invalid API key"
            }
        }
        
        with pytest.raises(GenesisAPIAuthenticationError):
            handle_genesis_error(error_response, status_code=401)
    
    def test_handle_genesis_error_rate_limit(self):
        """
        Test that `handle_genesis_error` raises `GenesisAPIRateLimitError` when a rate limit error response with status code 429 is received.
        """
        error_response = {
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded"
            }
        }
        
        with pytest.raises(GenesisAPIRateLimitError):
            handle_genesis_error(error_response, status_code=429)
    
    def test_handle_genesis_error_generic(self):
        """
        Test that `handle_genesis_error` raises a generic `GenesisAPIError` for unknown error types.
        """
        error_response = {
            "error": {
                "type": "unknown_error",
                "message": "Something went wrong"
            }
        }
        
        with pytest.raises(GenesisAPIError):
            handle_genesis_error(error_response, status_code=500)


class TestGenesisAPIEdgeCases:
    """Test suite for edge cases and boundary conditions"""
    
    @pytest.fixture
    def genesis_api(self):
        """
        Provides a GenesisAPI instance initialized with a test API key.
        """
        return GenesisAPI(api_key="test_key")
    
    def test_concurrent_requests(self, genesis_api):
        """
        Test that the GenesisAPI client can handle multiple concurrent requests successfully.
        
        Simulates several threads making simultaneous API calls and verifies that each receives a valid response.
        """
        with patch('requests.Session.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "concurrent_test",
                "choices": [{"message": {"content": "Concurrent response"}}]
            }
            mock_post.return_value = mock_response
            
            # Simulate concurrent requests
            import threading
            results = []
            
            def make_request():
                """
                Calls the `make_request` method of the `genesis_api` instance with a test prompt and appends the result or any exception raised to the `results` list.
                """
                try:
                    result = genesis_api.make_request("test prompt")
                    results.append(result)
                except Exception as e:
                    results.append(e)
            
            threads = [threading.Thread(target=make_request) for _ in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            assert len(results) == 5
            assert all(isinstance(r, dict) and "id" in r for r in results)
    
    def test_large_prompt_handling(self, genesis_api):
        """
        Tests that the API correctly processes and returns a response for a large but valid prompt input.
        """
        large_prompt = "x" * 5000  # Large but valid prompt
        
        with patch('requests.Session.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "large_prompt_test",
                "choices": [{"message": {"content": "Large prompt response"}}]
            }
            mock_post.return_value = mock_response
            
            result = genesis_api.make_request(large_prompt)
            assert result["id"] == "large_prompt_test"
    
    def test_unicode_prompt_handling(self, genesis_api):
        """
        Test that the API correctly processes prompts containing Unicode characters.
        
        Verifies that prompts with non-ASCII characters are accepted and that the response is handled as expected.
        """
        unicode_prompt = "Hello 世界 🌍 café naïve résumé"
        
        with patch('requests.Session.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "unicode_test",
                "choices": [{"message": {"content": "Unicode response"}}]
            }
            mock_post.return_value = mock_response
            
            result = genesis_api.make_request(unicode_prompt)
            assert result["id"] == "unicode_test"
    
    def test_retry_mechanism(self, genesis_api):
        """
        Test that the API client's retry mechanism correctly retries on transient failures and succeeds after multiple attempts.
        
        Simulates consecutive timeout errors followed by a successful response, verifying that the request is retried up to the configured maximum and ultimately returns the expected result.
        """
        with patch('requests.Session.post') as mock_post:
            # First two calls fail, third succeeds
            mock_post.side_effect = [
                TimeoutError("Timeout 1"),
                TimeoutError("Timeout 2"),
                MagicMock(status_code=200, json=lambda: {"id": "retry_success"})
            ]
            
            # Enable retry mechanism
            genesis_api.max_retries = 3
            result = genesis_api.make_request_with_retry("test prompt")
            
            assert result["id"] == "retry_success"
            assert mock_post.call_count == 3
    
    def test_response_caching(self, genesis_api):
        """
        Test that the GenesisAPI response caching mechanism returns cached results for identical requests and avoids redundant HTTP calls.
        """
        with patch('requests.Session.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "cached_response",
                "choices": [{"message": {"content": "Cached content"}}]
            }
            mock_post.return_value = mock_response
            
            # Enable caching
            genesis_api.enable_caching = True
            
            # First request
            result1 = genesis_api.make_request("test prompt")
            # Second identical request (should be cached)
            result2 = genesis_api.make_request("test prompt")
            
            assert result1 == result2
            assert mock_post.call_count == 1  # Only called once due to caching


class TestGenesisAPIExceptions:
    """Test suite for custom exceptions"""
    
    def test_genesis_api_error_creation(self):
        """Test GenesisAPIError creation"""
        error = GenesisAPIError("Test error message")
        assert str(error) == "Test error message"
    
    def test_genesis_api_authentication_error_creation(self):
        """Test GenesisAPIAuthenticationError creation"""
        error = GenesisAPIAuthenticationError("Auth failed")
        assert str(error) == "Auth failed"
        assert isinstance(error, GenesisAPIError)
    
    def test_genesis_api_rate_limit_error_creation(self):
        """Test GenesisAPIRateLimitError creation"""
        error = GenesisAPIRateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, GenesisAPIError)
    
    def test_genesis_api_timeout_error_creation(self):
        """Test GenesisAPITimeoutError creation"""
        error = GenesisAPITimeoutError("Request timed out")
        assert str(error) == "Request timed out"
        assert isinstance(error, GenesisAPIError)


class TestGenesisAPIIntegration:
    """Integration tests for GenesisAPI"""
    
    @pytest.fixture
    def genesis_api(self):
        """
        Creates and returns a GenesisAPI instance configured with a test integration API key.
        """
        return GenesisAPI(api_key="test_integration_key")
    
    @patch('requests.Session.post')
    def test_full_conversation_flow(self, mock_post, genesis_api):
        """
        Simulates a multi-turn conversation by making sequential API requests and verifies that each response matches the expected message ID and content.
        """
        responses = [
            {
                "id": "msg1",
                "choices": [{"message": {"content": "Hello! How can I help?"}}]
            },
            {
                "id": "msg2", 
                "choices": [{"message": {"content": "I can help with that."}}]
            },
            {
                "id": "msg3",
                "choices": [{"message": {"content": "Is there anything else?"}}]
            }
        ]
        
        mock_responses = []
        for response in responses:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = response
            mock_responses.append(mock_resp)
        
        mock_post.side_effect = mock_responses
        
        # Simulate conversation
        result1 = genesis_api.make_request("Hello")
        result2 = genesis_api.make_request("I need help with something")
        result3 = genesis_api.make_request("Thank you")
        
        assert result1["id"] == "msg1"
        assert result2["id"] == "msg2"
        assert result3["id"] == "msg3"
        assert mock_post.call_count == 3
    
    def test_api_key_rotation(self, genesis_api):
        """
        Test that rotating the API key updates the stored key and the Authorization header.
        
        Verifies that after calling `rotate_api_key`, the `GenesisAPI` instance uses the new key for authentication.
        """
        original_key = genesis_api.api_key
        new_key = "new_test_key"
        
        genesis_api.rotate_api_key(new_key)
        
        assert genesis_api.api_key == new_key
        assert genesis_api.api_key != original_key
        
        # Verify headers are updated
        headers = genesis_api._generate_headers()
        assert headers["Authorization"] == f"Bearer {new_key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])