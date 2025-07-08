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
        """Create a GenesisAPI instance for testing"""
        return GenesisAPI(
            api_key="test_api_key",
            base_url="https://api.genesis.test",
            timeout=30
        )
    
    @pytest.fixture
    def mock_response(self):
        """Create a mock HTTP response"""
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
        """Test GenesisAPI initialization with invalid URL"""
        with pytest.raises(ValueError, match="Invalid base URL"):
            GenesisAPI(api_key="test_key", base_url="invalid_url")
    
    @patch('requests.Session.post')
    def test_make_request_success(self, mock_post, genesis_api, mock_response):
        """Test successful API request"""
        mock_post.return_value = mock_response
        
        result = genesis_api.make_request("test prompt")
        
        assert result["id"] == "test_id_123"
        assert result["choices"][0]["message"]["content"] == "Test response content"
        mock_post.assert_called_once()
    
    @patch('requests.Session.post')
    def test_make_request_authentication_error(self, mock_post, genesis_api):
        """Test API request with authentication error"""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": "Invalid API key"}
        mock_post.return_value = mock_resp
        
        with pytest.raises(GenesisAPIAuthenticationError):
            genesis_api.make_request("test prompt")
    
    @patch('requests.Session.post')
    def test_make_request_rate_limit_error(self, mock_post, genesis_api):
        """Test API request with rate limit error"""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"error": "Rate limit exceeded"}
        mock_post.return_value = mock_resp
        
        with pytest.raises(GenesisAPIRateLimitError):
            genesis_api.make_request("test prompt")
    
    @patch('requests.Session.post')
    def test_make_request_timeout_error(self, mock_post, genesis_api):
        """Test API request with timeout error"""
        mock_post.side_effect = TimeoutError("Request timed out")
        
        with pytest.raises(GenesisAPITimeoutError):
            genesis_api.make_request("test prompt")
    
    @patch('requests.Session.post')
    def test_make_request_server_error(self, mock_post, genesis_api):
        """Test API request with server error"""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "Internal server error"}
        mock_post.return_value = mock_resp
        
        with pytest.raises(GenesisAPIError):
            genesis_api.make_request("test prompt")
    
    @patch('requests.Session.post')
    def test_make_request_malformed_response(self, mock_post, genesis_api):
        """Test API request with malformed JSON response"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_resp
        
        with pytest.raises(GenesisAPIError):
            genesis_api.make_request("test prompt")
    
    def test_generate_headers(self, genesis_api):
        """Test header generation for API requests"""
        headers = genesis_api._generate_headers()
        
        assert headers["Authorization"] == "Bearer test_api_key"
        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"].startswith("GenesisAPI")
    
    def test_validate_prompt_valid(self, genesis_api):
        """Test prompt validation with valid input"""
        valid_prompt = "This is a valid prompt"
        assert genesis_api._validate_prompt(valid_prompt) is True
    
    def test_validate_prompt_empty(self, genesis_api):
        """Test prompt validation with empty input"""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            genesis_api._validate_prompt("")
    
    def test_validate_prompt_too_long(self, genesis_api):
        """Test prompt validation with input that's too long"""
        long_prompt = "x" * 10001
        with pytest.raises(ValueError, match="Prompt is too long"):
            genesis_api._validate_prompt(long_prompt)
    
    def test_validate_prompt_invalid_characters(self, genesis_api):
        """Test prompt validation with invalid characters"""
        invalid_prompt = "Test prompt with \x00 null character"
        with pytest.raises(ValueError, match="Invalid characters"):
            genesis_api._validate_prompt(invalid_prompt)


class TestGenesisAPIAsync:
    """Test suite for async GenesisAPI methods"""
    
    @pytest.fixture
    def genesis_api(self):
        """Create a GenesisAPI instance for testing"""
        return GenesisAPI(api_key="test_api_key")
    
    @pytest.mark.asyncio
    async def test_make_request_async_success(self, genesis_api):
        """Test successful async API request"""
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
        """Test async API request with timeout"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError()
            
            with pytest.raises(GenesisAPITimeoutError):
                await genesis_api.make_request_async("test prompt")


class TestGenesisAPIUtilities:
    """Test suite for utility functions"""
    
    def test_parse_genesis_response_valid(self):
        """Test parsing valid Genesis API response"""
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
        """Test parsing Genesis API response with missing content"""
        response = {
            "id": "test_id",
            "choices": [],
            "usage": {"total_tokens": 0}
        }
        
        with pytest.raises(ValueError, match="No content found"):
            parse_genesis_response(response)
    
    def test_parse_genesis_response_malformed(self):
        """Test parsing malformed Genesis API response"""
        response = {"invalid": "structure"}
        
        with pytest.raises(ValueError, match="Invalid response structure"):
            parse_genesis_response(response)
    
    def test_validate_genesis_request_valid(self):
        """Test validating valid Genesis API request"""
        request = {
            "model": "genesis-1",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        assert validate_genesis_request(request) is True
    
    def test_validate_genesis_request_missing_model(self):
        """Test validating request with missing model"""
        request = {
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 100
        }
        
        with pytest.raises(ValueError, match="Model is required"):
            validate_genesis_request(request)
    
    def test_validate_genesis_request_invalid_temperature(self):
        """Test validating request with invalid temperature"""
        request = {
            "model": "genesis-1",
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 2.5
        }
        
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            validate_genesis_request(request)
    
    def test_validate_genesis_request_invalid_max_tokens(self):
        """Test validating request with invalid max_tokens"""
        request = {
            "model": "genesis-1",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": -1
        }
        
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            validate_genesis_request(request)
    
    def test_format_genesis_prompt_simple(self):
        """Test formatting simple prompt"""
        prompt = "Hello, world!"
        formatted = format_genesis_prompt(prompt)
        
        assert formatted["messages"][0]["content"] == "Hello, world!"
        assert formatted["messages"][0]["role"] == "user"
    
    def test_format_genesis_prompt_with_system_message(self):
        """Test formatting prompt with system message"""
        prompt = "Hello, world!"
        system_message = "You are a helpful assistant."
        formatted = format_genesis_prompt(prompt, system_message=system_message)
        
        assert len(formatted["messages"]) == 2
        assert formatted["messages"][0]["role"] == "system"
        assert formatted["messages"][0]["content"] == system_message
        assert formatted["messages"][1]["role"] == "user"
        assert formatted["messages"][1]["content"] == "Hello, world!"
    
    def test_format_genesis_prompt_with_parameters(self):
        """Test formatting prompt with additional parameters"""
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
        """Test handling authentication errors"""
        error_response = {
            "error": {
                "type": "authentication_error",
                "message": "Invalid API key"
            }
        }
        
        with pytest.raises(GenesisAPIAuthenticationError):
            handle_genesis_error(error_response, status_code=401)
    
    def test_handle_genesis_error_rate_limit(self):
        """Test handling rate limit errors"""
        error_response = {
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded"
            }
        }
        
        with pytest.raises(GenesisAPIRateLimitError):
            handle_genesis_error(error_response, status_code=429)
    
    def test_handle_genesis_error_generic(self):
        """Test handling generic errors"""
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
        return GenesisAPI(api_key="test_key")
    
    def test_concurrent_requests(self, genesis_api):
        """Test handling multiple concurrent requests"""
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
        """Test handling of large prompts"""
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
        """Test handling of unicode characters in prompts"""
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
        """Test automatic retry mechanism on transient failures"""
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
        """Test response caching mechanism"""
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
        return GenesisAPI(api_key="test_integration_key")
    
    @patch('requests.Session.post')
    def test_full_conversation_flow(self, mock_post, genesis_api):
        """Test full conversation flow with multiple exchanges"""
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
        """Test API key rotation functionality"""
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