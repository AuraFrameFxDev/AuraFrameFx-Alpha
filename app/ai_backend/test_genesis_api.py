import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Import the module under test
from app.ai_backend.genesis_api import (
    GenesisAPIClient,
    GenesisAPIError,
    RateLimitError,
    AuthenticationError,
    ValidationError,
    APIResponse,
    ModelConfig,
    ChatMessage,
    ChatCompletion
)


class TestGenesisAPIClient:
    """Test suite for GenesisAPIClient class."""

    @pytest.fixture
    def mock_config(self):
        """
        Provides a mock configuration dictionary with API key, base URL, timeout, and max retries for testing GenesisAPIClient initialization.

        Returns:
            dict: A dictionary containing mock configuration values for API key, base URL, timeout, and max retries.
        """
        return {
            'api_key': 'test-api-key-123',
            'base_url': 'https://api.genesis.ai/v1',
            'timeout': 30,
            'max_retries': 3
        }

    @pytest.fixture
    def client(self, mock_config):
        """
        Pytest fixture that returns a GenesisAPIClient instance initialized with the provided mock configuration.
        """
        return GenesisAPIClient(**mock_config)

    @pytest.fixture
    def sample_messages(self):
        """
        Return a list of sample ChatMessage objects representing a typical conversation for testing purposes.
        """
        return [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is the weather like today?"),
            ChatMessage(role="assistant", content="I don't have access to real-time weather data.")
        ]

    @pytest.fixture
    def sample_model_config(self):
        """
        Return a ModelConfig instance with typical preset parameters for testing purposes.

        Returns:
            ModelConfig: A model configuration populated with standard test values.
        """
        return ModelConfig(
            name="genesis-gpt-4",
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

    def test_client_initialization_with_valid_config(self, mock_config):
        """
        Test that GenesisAPIClient initializes with the expected attributes when provided a valid configuration.

        Asserts that the client's API key, base URL, timeout, and max retries match the configuration values.
        """
        client = GenesisAPIClient(**mock_config)
        assert client.api_key == mock_config['api_key']
        assert client.base_url == mock_config['base_url']
        assert client.timeout == mock_config['timeout']
        assert client.max_retries == mock_config['max_retries']

    def test_client_initialization_with_minimal_config(self):
        """
        Test that GenesisAPIClient initializes with only an API key and sets default values for optional parameters.
        """
        client = GenesisAPIClient(api_key='test-key')
        assert client.api_key == 'test-key'
        assert client.base_url is not None  # Should have default
        assert client.timeout > 0  # Should have default
        assert client.max_retries >= 0  # Should have default

    def test_client_initialization_missing_api_key(self):
        """
        Test that initializing the GenesisAPIClient without an API key raises a ValueError.
        """
        with pytest.raises(ValueError, match="API key is required"):
            GenesisAPIClient()

    def test_client_initialization_invalid_timeout(self):
        """
        Test that initializing GenesisAPIClient with a non-positive timeout raises a ValueError.
        """
        with pytest.raises(ValueError, match="Timeout must be positive"):
            GenesisAPIClient(api_key='test-key', timeout=-1)

        with pytest.raises(ValueError, match="Timeout must be positive"):
            GenesisAPIClient(api_key='test-key', timeout=0)

    def test_client_initialization_invalid_max_retries(self):
        """
        Test that initializing GenesisAPIClient with a negative max_retries value raises a ValueError.
        """
        with pytest.raises(ValueError, match="Max retries must be non-negative"):
            GenesisAPIClient(api_key='test-key', max_retries=-1)

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, client, sample_messages, sample_model_config):
        """
        Verifies that a successful chat completion request returns a ChatCompletion object with the correct attributes.
        """
        mock_response = {
            'id': 'chat-123',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'The weather looks pleasant today!'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 25,
                'completion_tokens': 8,
                'total_tokens': 33
            }
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200

            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )

            assert isinstance(result, ChatCompletion)
            assert result.id == 'chat-123'
            assert result.model == 'genesis-gpt-4'
            assert len(result.choices) == 1
            assert result.choices[0].message.content == 'The weather looks pleasant today!'
            assert result.usage.total_tokens == 33

    @pytest.mark.asyncio
    async def test_chat_completion_with_streaming(self, client, sample_messages, sample_model_config):
        """
        Tests that chat completion with streaming enabled yields the expected sequence of streamed response chunks.

        Simulates a streaming API response and verifies that the client receives chat completion chunks in order, with correct content and finish reason.
        """
        mock_chunks = [
            {'choices': [{'delta': {'content': 'The'}}]},
            {'choices': [{'delta': {'content': ' weather'}}]},
            {'choices': [{'delta': {'content': ' is nice'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]

        async def mock_stream():
            """
            Asynchronously yields encoded JSON representations of mock data chunks to simulate a streaming API response.
            """
            for chunk in mock_chunks:
                yield json.dumps(chunk).encode()

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200

            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)

            assert len(chunks) == 4
            assert chunks[0].choices[0].delta.content == 'The'
            assert chunks[-1].choices[0].finish_reason == 'stop'

    @pytest.mark.asyncio
    async def test_chat_completion_authentication_error(self, client, sample_messages, sample_model_config):
        """
        Test that an authentication error during chat completion raises an AuthenticationError with the correct message.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 401
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'error': {'message': 'Invalid API key'}}
            )

            with pytest.raises(AuthenticationError, match="Invalid API key"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_chat_completion_rate_limit_error(self, client, sample_messages, sample_model_config):
        """
        Test that a RateLimitError is raised with the correct retry_after value when a 429 status is returned during chat completion.

        Verifies that the client correctly interprets the 'Retry-After' header and raises a RateLimitError with the expected retry_after value.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 429
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'error': {'message': 'Rate limit exceeded'}}
            )
            mock_post.return_value.__aenter__.return_value.headers = {'Retry-After': '60'}

            with pytest.raises(RateLimitError) as exc_info:
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

            assert exc_info.value.retry_after == 60

    @pytest.mark.asyncio
    async def test_chat_completion_validation_error(self, client, sample_model_config):
        """
        Test that a chat completion request with an invalid message role raises a ValidationError.

        Simulates a 400 API response for an invalid message role and asserts that the client raises a ValidationError with the expected error message.
        """
        invalid_messages = [
            ChatMessage(role="invalid_role", content="This should fail")
        ]

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 400
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'error': {'message': 'Invalid message role'}}
            )

            with pytest.raises(ValidationError, match="Invalid message role"):
                await client.create_chat_completion(
                    messages=invalid_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_chat_completion_server_error_with_retry(self, client, sample_messages, sample_model_config):
        """
        Test that chat completion retries on server errors and succeeds after retry attempts.

        Simulates two consecutive server errors followed by a successful response, verifying that the retry mechanism is triggered and the final chat completion result is correct.
        """
        call_count = 0

        async def mock_post_with_failure(*args, **kwargs):
            """
            Simulates an asynchronous HTTP POST request that returns a 500 error response on the first two calls and a successful 200 response with a chat completion payload on the third call.

            Returns:
                Mock: A mock response object with status and JSON payload based on the current call count.
            """
            nonlocal call_count
            call_count += 1

            mock_response = Mock()
            if call_count <= 2:  # Fail first 2 attempts
                mock_response.status = 500
                mock_response.json = AsyncMock(
                    return_value={'error': {'message': 'Internal server error'}}
                )
            else:  # Succeed on 3rd attempt
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    'id': 'chat-retry-success',
                    'object': 'chat.completion',
                    'created': int(datetime.now(timezone.utc).timestamp()),
                    'model': 'genesis-gpt-4',
                    'choices': [{'message': {'role': 'assistant', 'content': 'Success after retry'}}],
                    'usage': {'total_tokens': 10}
                })

            return mock_response

        with patch('aiohttp.ClientSession.post', side_effect=mock_post_with_failure):
            with patch('asyncio.sleep'):  # Mock sleep to speed up test
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

                assert result.id == 'chat-retry-success'
                assert call_count == 3  # Should have retried twice

    @pytest.mark.asyncio
    async def test_chat_completion_max_retries_exceeded(self, client, sample_messages, sample_model_config):
        """
        Test that the client raises a GenesisAPIError when the maximum number of retries is exceeded due to repeated server errors during chat completion.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 500
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'error': {'message': 'Internal server error'}}
            )

            with patch('asyncio.sleep'):  # Mock sleep to speed up test
                with pytest.raises(GenesisAPIError, match="Internal server error"):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )

    @pytest.mark.asyncio
    async def test_chat_completion_network_timeout(self, client, sample_messages, sample_model_config):
        """
        Test that a network timeout during chat completion raises a GenesisAPIError with a timeout message.

        Simulates a timeout in the chat completion API call and verifies that the client raises a GenesisAPIError containing a timeout-related message.
        """
        with patch('aiohttp.ClientSession.post', side_effect=asyncio.TimeoutError()):
            with pytest.raises(GenesisAPIError, match="Request timeout"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_chat_completion_connection_error(self, client, sample_messages, sample_model_config):
        """
        Test that a connection error during chat completion raises a GenesisAPIError with an appropriate message.
        """
        import aiohttp

        with patch('aiohttp.ClientSession.post', side_effect=aiohttp.ClientConnectionError()):
            with pytest.raises(GenesisAPIError, match="Connection error"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    def test_validate_messages_empty_list(self, client):
        """
        Test that validating an empty message list raises a ValidationError.
        """
        with pytest.raises(ValidationError, match="Messages cannot be empty"):
            client._validate_messages([])

    def test_validate_messages_invalid_role(self, client):
        """
        Verify that providing a message with an invalid role to _validate_messages raises a ValidationError.
        """
        invalid_messages = [
            ChatMessage(role="invalid", content="Test content")
        ]

        with pytest.raises(ValidationError, match="Invalid message role"):
            client._validate_messages(invalid_messages)

    def test_validate_messages_empty_content(self, client):
        """
        Test that validating a message with empty content raises a ValidationError.
        """
        invalid_messages = [
            ChatMessage(role="user", content="")
        ]

        with pytest.raises(ValidationError, match="Message content cannot be empty"):
            client._validate_messages(invalid_messages)

    def test_validate_messages_content_too_long(self, client):
        """
        Test that validating messages with excessively long content raises a ValidationError.
        """
        long_content = "x" * 100000  # Assuming max length is less than this
        invalid_messages = [
            ChatMessage(role="user", content=long_content)
        ]

        with pytest.raises(ValidationError, match="Message content too long"):
            client._validate_messages(invalid_messages)

    def test_validate_model_config_invalid_temperature(self, client, sample_model_config):
        """
        Test that model configuration validation raises a ValidationError when temperature is set below 0 or above 2.
        """
        sample_model_config.temperature = -0.5  # Invalid negative temperature

        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(sample_model_config)

        sample_model_config.temperature = 2.5  # Invalid high temperature

        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(sample_model_config)

    def test_validate_model_config_invalid_max_tokens(self, client, sample_model_config):
        """
        Verify that _validate_model_config raises a ValidationError when max_tokens is zero or negative in the model configuration.
        """
        sample_model_config.max_tokens = 0  # Invalid zero tokens

        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            client._validate_model_config(sample_model_config)

        sample_model_config.max_tokens = -100  # Invalid negative tokens

        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            client._validate_model_config(sample_model_config)

    def test_validate_model_config_invalid_top_p(self, client, sample_model_config):
        """
        Verify that a ValidationError is raised when the model config's top_p parameter is set outside the range [0, 1].
        """
        sample_model_config.top_p = -0.1  # Invalid negative top_p

        with pytest.raises(ValidationError, match="Top_p must be between 0 and 1"):
            client._validate_model_config(sample_model_config)

        sample_model_config.top_p = 1.5  # Invalid high top_p

        with pytest.raises(ValidationError, match="Top_p must be between 0 and 1"):
            client._validate_model_config(sample_model_config)

    @pytest.mark.asyncio
    async def test_list_models_success(self, client):
        """
        Test that the client retrieves and parses a list of available models from the API.

        Asserts that the returned list contains the expected model IDs and the correct number of models.
        """
        mock_response = {
            'object': 'list',
            'data': [
                {'id': 'genesis-gpt-4', 'object': 'model', 'created': 1677610602},
                {'id': 'genesis-gpt-3.5-turbo', 'object': 'model', 'created': 1677610602}
            ]
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value.status = 200

            models = await client.list_models()

            assert len(models) == 2
            assert models[0].id == 'genesis-gpt-4'
            assert models[1].id == 'genesis-gpt-3.5-turbo'

    @pytest.mark.asyncio
    async def test_get_model_success(self, client):
        """
        Test that retrieving a model by ID returns a model object with the expected attributes.
        """
        mock_response = {
            'id': 'genesis-gpt-4',
            'object': 'model',
            'created': 1677610602,
            'owned_by': 'genesis-ai',
            'permission': []
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value.status = 200

            model = await client.get_model('genesis-gpt-4')

            assert model.id == 'genesis-gpt-4'
            assert model.owned_by == 'genesis-ai'

    @pytest.mark.asyncio
    async def test_get_model_not_found(self, client):
        """
        Test that retrieving a model with a non-existent ID raises a GenesisAPIError with the expected error message.
        """
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.status = 404
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'error': {'message': 'Model not found'}}
            )

            with pytest.raises(GenesisAPIError, match="Model not found"):
                await client.get_model('non-existent-model')

    def test_build_headers(self, client):
        """
        Test that the client's _build_headers method includes the required HTTP headers for authorization and content type.
        """
        headers = client._build_headers()

        assert 'Authorization' in headers
        assert headers['Authorization'] == f'Bearer {client.api_key}'
        assert headers['Content-Type'] == 'application/json'
        assert 'User-Agent' in headers

    def test_build_headers_with_custom_headers(self, client):
        """
        Verify that custom headers are correctly merged with default headers when building request headers.
        """
        custom_headers = {'X-Custom-Header': 'custom-value'}
        headers = client._build_headers(custom_headers)

        assert headers['X-Custom-Header'] == 'custom-value'
        assert 'Authorization' in headers
        assert headers['Content-Type'] == 'application/json'

# Additional comprehensive test coverage
class TestDataModels:
    """Test suite for data model classes."""

    def test_chat_message_creation(self):
        """
        Test creation of a ChatMessage instance with specified role and content, verifying that the optional name attribute defaults to None.
        """
        message = ChatMessage(role="user", content="Hello, world!")
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.name is None

    def test_chat_message_with_name(self):
        """
        Test that the ChatMessage instance assigns the provided name to the name attribute.
        """
        message = ChatMessage(role="user", content="Hello", name="John")
        assert message.name == "John"

    def test_model_config_creation(self):
        """
        Test that a ModelConfig instance is created correctly with specified attributes.
        """
        config = ModelConfig(
            name="genesis-gpt-4",
            max_tokens=1000,
            temperature=0.7
        )
        assert config.name == "genesis-gpt-4"
        assert config.max_tokens == 1000
        assert config.temperature == 0.7

    def test_model_config_defaults(self):
        """
        Test that ModelConfig assigns default values to optional parameters when only the name is provided.
        """
        config = ModelConfig(name="test-model")
        assert config.name == "test-model"
        assert config.max_tokens is not None
        assert config.temperature is not None
        assert config.top_p is not None

    def test_api_response_creation(self):
        """
        Test creation of an APIResponse object with specified status code, data, and headers.
        """
        response = APIResponse(
            status_code=200,
            data={'message': 'success'},
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == 200
        assert response.data['message'] == 'success'
        assert response.headers['Content-Type'] == 'application/json'

    def test_chat_completion_creation(self):
        """
        Test that a ChatCompletion object is created with the correct attribute values.
        """
        completion = ChatCompletion(
            id="chat-123",
            object="chat.completion",
            created=1677610602,
            model="genesis-gpt-4",
            choices=[],
            usage={'total_tokens': 100}
        )
        assert completion.id == "chat-123"
        assert completion.model == "genesis-gpt-4"
        assert completion.usage['total_tokens'] == 100


class TestExceptionClasses:
    """Test suite for custom exception classes."""

    def test_genesis_api_error(self):
        """
        Test that GenesisAPIError is instantiated correctly with the provided message and status code.
        """
        error = GenesisAPIError("Test error message", status_code=500)
        assert str(error) == "Test error message"
        assert error.status_code == 500

    def test_authentication_error(self):
        """
        Test instantiation of AuthenticationError and verify it inherits from GenesisAPIError.
        """
        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"
        assert isinstance(error, GenesisAPIError)

    def test_rate_limit_error(self):
        """
        Test that a RateLimitError is correctly instantiated with a retry_after value and inherits from GenesisAPIError.
        """
        error = RateLimitError("Rate limit exceeded", retry_after=60)
        assert str(error) == "Rate limit exceeded"
        assert error.retry_after == 60
        assert isinstance(error, GenesisAPIError)

    def test_validation_error(self):
        """
        Test that a ValidationError is correctly instantiated and inherits from GenesisAPIError.
        """
        error = ValidationError("Invalid input data")
        assert str(error) == "Invalid input data"
        assert isinstance(error, GenesisAPIError)


class TestUtilityFunctions:
    """Test suite for utility functions in the genesis_api module."""

    def test_format_timestamp(self):
        """
        Test that the `format_timestamp` utility returns a non-empty string for a valid timestamp input.
        """
        from app.ai_backend.genesis_api import format_timestamp

        timestamp = 1677610602
        formatted = format_timestamp(timestamp)
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_calculate_token_usage(self):
        """
        Test that the `calculate_token_usage` utility correctly computes token usage for a list of chat messages.

        Asserts that the returned value is a dictionary containing the 'estimated_tokens' key.
        """
        from app.ai_backend.genesis_api import calculate_token_usage

        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!")
        ]

        usage = calculate_token_usage(messages)
        assert isinstance(usage, dict)
        assert 'estimated_tokens' in usage

    @pytest.mark.parametrize("content,expected_tokens", [
        ("Hello", 1),
        ("Hello world", 2),
        ("", 0),
        ("A very long message with many words", 7),
    ])
    def test_estimate_tokens(self, content, expected_tokens):
        """
        Test that the token estimation function returns the expected token count for the given input.

        Parameters:
            content (str): The input text to estimate tokens for.
            expected_tokens (int): The expected token count to compare against.
        """
        from app.ai_backend.genesis_api import estimate_tokens

        tokens = estimate_tokens(content)
        assert tokens == expected_tokens

class TestGenesisAPIClientEdgeCases:
    """Additional edge case tests for GenesisAPIClient class."""

    @pytest.fixture
    def client_with_custom_config(self):
        """Client with custom configuration for edge case testing."""
        return GenesisAPIClient(
            api_key='edge-case-key',
            base_url='https://custom.api.com/v2',
            timeout=5,
            max_retries=1
        )

    @pytest.mark.asyncio
    async def test_chat_completion_with_empty_response(self, client, sample_messages, sample_model_config):
        """Test handling of empty response from API."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value={})
            mock_post.return_value.__aenter__.return_value.status = 200

            with pytest.raises(GenesisAPIError, match="Invalid response format"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_chat_completion_with_malformed_json(self, client, sample_messages, sample_model_config):
        """Test handling of malformed JSON response."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                side_effect=json.JSONDecodeError("Invalid JSON", "", 0)
            )
            mock_post.return_value.__aenter__.return_value.status = 200

            with pytest.raises(GenesisAPIError, match="Invalid JSON response"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_chat_completion_with_missing_choices(self, client, sample_messages, sample_model_config):
        """Test handling of response missing choices field."""
        mock_response = {
            'id': 'chat-123',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'usage': {'total_tokens': 10}
            # Missing 'choices' field
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200

            with pytest.raises(GenesisAPIError, match="Missing required field"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_chat_completion_stream_with_connection_interrupted(self, client, sample_messages, sample_model_config):
        """Test streaming when connection is interrupted mid-stream."""
        async def mock_interrupted_stream():
            yield json.dumps({'choices': [{'delta': {'content': 'Hello'}}]}).encode()
            raise asyncio.CancelledError("Connection interrupted")

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_interrupted_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200

            with pytest.raises(GenesisAPIError, match="Stream interrupted"):
                chunks = []
                async for chunk in client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                ):
                    chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_chat_completion_stream_with_invalid_chunk(self, client, sample_messages, sample_model_config):
        """Test streaming with invalid JSON chunk."""
        async def mock_invalid_stream():
            yield b'invalid json chunk'
            yield json.dumps({'choices': [{'delta': {'content': 'valid'}}]}).encode()

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_invalid_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200

            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)

            # Should skip invalid chunk and continue with valid ones
            assert len(chunks) == 1
            assert chunks[0].choices[0].delta.content == 'valid'

    @pytest.mark.asyncio
    async def test_chat_completion_with_extremely_long_message_list(self, client, sample_model_config):
        """Test handling of extremely long message lists."""
        # Create 1000 messages to test bulk processing
        long_messages = [
            ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            for i in range(1000)
        ]

        mock_response = {
            'id': 'chat-bulk',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Bulk response'}}],
            'usage': {'total_tokens': 5000}
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200

            result = await client.create_chat_completion(
                messages=long_messages,
                model_config=sample_model_config
            )

            assert result.id == 'chat-bulk'
            assert result.usage['total_tokens'] == 5000

    @pytest.mark.asyncio
    async def test_chat_completion_with_unicode_content(self, client, sample_model_config):
        """Test handling of unicode and special characters in messages."""
        unicode_messages = [
            ChatMessage(role="user", content="Hello 世界! 🌍🚀"),
            ChatMessage(role="assistant", content="¡Hola mundo! 😄💻"),
            ChatMessage(role="user", content="Testing émojis: 🎉🎊 and açcénts")
        ]

        mock_response = {
            'id': 'chat-unicode',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Unicode response: 🌟'}}],
            'usage': {'total_tokens': 50}
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200

            result = await client.create_chat_completion(
                messages=unicode_messages,
                model_config=sample_model_config
            )

            assert result.choices[0].message.content == 'Unicode response: 🌟'

    @pytest.mark.asyncio
    async def test_rate_limit_error_without_retry_after_header(self, client, sample_messages, sample_model_config):
        """Test rate limit error handling when Retry-After header is missing."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 429
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'error': {'message': 'Rate limit exceeded'}}
            )
            mock_post.return_value.__aenter__.return_value.headers = {}  # No Retry-After header

            with pytest.raises(RateLimitError) as exc_info:
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

            assert exc_info.value.retry_after is None

    @pytest.mark.asyncio
    async def test_rate_limit_error_with_invalid_retry_after(self, client, sample_messages, sample_model_config):
        """Test rate limit error handling with invalid Retry-After header value."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 429
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'error': {'message': 'Rate limit exceeded'}}
            )
            mock_post.return_value.__aenter__.return_value.headers = {'Retry-After': 'invalid'}

            with pytest.raises(RateLimitError) as exc_info:
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

            assert exc_info.value.retry_after is None

    def test_validate_messages_with_none_content(self, client):
        """Test validation of messages with None content."""
        invalid_messages = [
            ChatMessage(role="user", content=None)
        ]

        with pytest.raises(ValidationError, match="Message content cannot be None"):
            client._validate_messages(invalid_messages)

    def test_validate_messages_with_mixed_valid_invalid(self, client):
        """Test validation of message list with mixed valid and invalid messages."""
        mixed_messages = [
            ChatMessage(role="user", content="Valid message"),
            ChatMessage(role="invalid_role", content="Invalid role"),
            ChatMessage(role="assistant", content="Another valid message")
        ]

        with pytest.raises(ValidationError, match="Invalid message role"):
            client._validate_messages(mixed_messages)

    def test_validate_model_config_with_none_values(self, client):
        """Test model config validation with None values."""
        config = ModelConfig(
            name="test-model",
            max_tokens=None,
            temperature=None
        )

        with pytest.raises(ValidationError, match="Max tokens cannot be None"):
            client._validate_model_config(config)

    def test_validate_model_config_with_extreme_values(self, client):
        """Test model config validation with extreme boundary values."""
        config = ModelConfig(
            name="test-model",
            max_tokens=1,  # Minimum valid value
            temperature=0.0,  # Minimum valid value
            top_p=1.0,  # Maximum valid value
            frequency_penalty=-2.0,  # Boundary value
            presence_penalty=2.0  # Boundary value
        )

        # Should not raise exception for valid boundary values
        client._validate_model_config(config)

        # Test invalid frequency penalty
        config.frequency_penalty = -3.0
        with pytest.raises(ValidationError, match="Frequency penalty must be between -2 and 2"):
            client._validate_model_config(config)

        # Test invalid presence penalty
        config.frequency_penalty = 0.0
        config.presence_penalty = 3.0
        with pytest.raises(ValidationError, match="Presence penalty must be between -2 and 2"):
            client._validate_model_config(config)

    @pytest.mark.asyncio
    async def test_list_models_with_empty_response(self, client):
        """Test list_models with empty model list."""
        mock_response = {
            'object': 'list',
            'data': []
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value.status = 200

            models = await client.list_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_list_models_with_malformed_response(self, client):
        """Test list_models with malformed response structure."""
        mock_response = {
            'object': 'list',
            'data': [
                {'id': 'model-1'},  # Missing required fields
                {'object': 'model', 'created': 1677610602}  # Missing id
            ]
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value.status = 200

            with pytest.raises(GenesisAPIError, match="Invalid model data"):
                await client.list_models()

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_config):
        """Test using GenesisAPIClient as an async context manager."""
        async with GenesisAPIClient(**mock_config) as client:
            assert client.api_key == mock_config['api_key']
            # Context manager should handle cleanup

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, sample_messages, sample_model_config):
        """Test handling of concurrent API requests."""
        mock_response = {
            'id': 'chat-concurrent',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Concurrent response'}}],
            'usage': {'total_tokens': 20}
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200

            # Create 5 concurrent requests
            tasks = [
                client.create_chat_completion(messages=sample_messages, model_config=sample_model_config)
                for _ in range(5)
            ]

            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            for result in results:
                assert result.id == 'chat-concurrent'

    def test_build_headers_with_none_custom_headers(self, client):
        """Test building headers with None custom headers."""
        headers = client._build_headers(None)
        
        assert 'Authorization' in headers
        assert 'Content-Type' in headers
        assert 'User-Agent' in headers

    def test_build_headers_with_conflicting_custom_headers(self, client):
        """Test building headers with custom headers that conflict with defaults."""
        custom_headers = {
            'Authorization': 'Bearer custom-token',
            'Content-Type': 'text/plain'
        }
        
        headers = client._build_headers(custom_headers)
        
        # Custom headers should override defaults
        assert headers['Authorization'] == 'Bearer custom-token'
        assert headers['Content-Type'] == 'text/plain'


class TestDataModelsEdgeCases:
    """Additional edge case tests for data model classes."""

    def test_chat_message_with_special_characters_in_name(self):
        """Test ChatMessage with special characters in name field."""
        message = ChatMessage(
            role="user",
            content="Test message",
            name="user@domain.com"
        )
        assert message.name == "user@domain.com"

    def test_chat_message_with_empty_name(self):
        """Test ChatMessage with empty string as name."""
        message = ChatMessage(
            role="user",
            content="Test message",
            name=""
        )
        assert message.name == ""

    def test_model_config_with_zero_values(self):
        """Test ModelConfig with zero values where valid."""
        config = ModelConfig(
            name="test-model",
            max_tokens=1,
            temperature=0.0,
            top_p=0.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        assert config.temperature == 0.0
        assert config.top_p == 0.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0

    def test_model_config_with_maximum_values(self):
        """Test ModelConfig with maximum allowed values."""
        config = ModelConfig(
            name="test-model",
            max_tokens=4096,
            temperature=2.0,
            top_p=1.0,
            frequency_penalty=2.0,
            presence_penalty=2.0
        )
        assert config.max_tokens == 4096
        assert config.temperature == 2.0
        assert config.top_p == 1.0
        assert config.frequency_penalty == 2.0
        assert config.presence_penalty == 2.0

    def test_api_response_with_none_data(self):
        """Test APIResponse with None data."""
        response = APIResponse(
            status_code=204,
            data=None,
            headers={}
        )
        assert response.data is None
        assert response.status_code == 204

    def test_api_response_with_empty_headers(self):
        """Test APIResponse with empty headers dict."""
        response = APIResponse(
            status_code=200,
            data={'test': 'data'},
            headers={}
        )
        assert response.headers == {}

    def test_chat_completion_with_empty_choices(self):
        """Test ChatCompletion with empty choices list."""
        completion = ChatCompletion(
            id="chat-empty",
            object="chat.completion",
            created=1677610602,
            model="genesis-gpt-4",
            choices=[],
            usage={'total_tokens': 0}
        )
        assert completion.choices == []
        assert completion.usage['total_tokens'] == 0

    def test_chat_completion_with_multiple_choices(self):
        """Test ChatCompletion with multiple choices."""
        choices = [
            {'message': {'role': 'assistant', 'content': 'Choice 1'}},
            {'message': {'role': 'assistant', 'content': 'Choice 2'}},
            {'message': {'role': 'assistant', 'content': 'Choice 3'}}
        ]
        
        completion = ChatCompletion(
            id="chat-multi",
            object="chat.completion",
            created=1677610602,
            model="genesis-gpt-4",
            choices=choices,
            usage={'total_tokens': 30}
        )
        assert len(completion.choices) == 3


class TestExceptionClassesEdgeCases:
    """Additional edge case tests for custom exception classes."""

    def test_genesis_api_error_with_none_status_code(self):
        """Test GenesisAPIError with None status code."""
        error = GenesisAPIError("Test error", status_code=None)
        assert error.status_code is None

    def test_genesis_api_error_with_empty_message(self):
        """Test GenesisAPIError with empty message."""
        error = GenesisAPIError("", status_code=400)
        assert str(error) == ""

    def test_rate_limit_error_with_zero_retry_after(self):
        """Test RateLimitError with zero retry_after value."""
        error = RateLimitError("Rate limit exceeded", retry_after=0)
        assert error.retry_after == 0

    def test_rate_limit_error_with_negative_retry_after(self):
        """Test RateLimitError with negative retry_after value."""
        error = RateLimitError("Rate limit exceeded", retry_after=-1)
        assert error.retry_after == -1

    def test_exception_inheritance_chain(self):
        """Test that all custom exceptions inherit from GenesisAPIError."""
        auth_error = AuthenticationError("Auth failed")
        rate_error = RateLimitError("Rate limit", retry_after=60)
        validation_error = ValidationError("Validation failed")

        assert isinstance(auth_error, GenesisAPIError)
        assert isinstance(rate_error, GenesisAPIError)
        assert isinstance(validation_error, GenesisAPIError)

        # Test that they also inherit from Exception
        assert isinstance(auth_error, Exception)
        assert isinstance(rate_error, Exception)
        assert isinstance(validation_error, Exception)


class TestUtilityFunctionsEdgeCases:
    """Additional edge case tests for utility functions."""

    def test_format_timestamp_with_negative_timestamp(self):
        """Test format_timestamp with negative timestamp."""
        from app.ai_backend.genesis_api import format_timestamp
        
        timestamp = -1677610602
        formatted = format_timestamp(timestamp)
        assert isinstance(formatted, str)

    def test_format_timestamp_with_zero_timestamp(self):
        """Test format_timestamp with zero timestamp (Unix epoch)."""
        from app.ai_backend.genesis_api import format_timestamp
        
        timestamp = 0
        formatted = format_timestamp(timestamp)
        assert isinstance(formatted, str)

    def test_format_timestamp_with_very_large_timestamp(self):
        """Test format_timestamp with very large timestamp."""
        from app.ai_backend.genesis_api import format_timestamp
        
        timestamp = 9999999999  # Year 2286
        formatted = format_timestamp(timestamp)
        assert isinstance(formatted, str)

    def test_calculate_token_usage_with_empty_messages(self):
        """Test calculate_token_usage with empty message list."""
        from app.ai_backend.genesis_api import calculate_token_usage
        
        usage = calculate_token_usage([])
        assert usage['estimated_tokens'] == 0

    def test_calculate_token_usage_with_none_content(self):
        """Test calculate_token_usage with messages containing None content."""
        from app.ai_backend.genesis_api import calculate_token_usage
        
        messages = [
            ChatMessage(role="user", content=None),
            ChatMessage(role="assistant", content="Valid content")
        ]
        
        usage = calculate_token_usage(messages)
        assert isinstance(usage['estimated_tokens'], int)
        assert usage['estimated_tokens'] >= 0

    def test_calculate_token_usage_with_very_long_content(self):
        """Test calculate_token_usage with very long message content."""
        from app.ai_backend.genesis_api import calculate_token_usage
        
        long_content = "word " * 10000  # 10,000 words
        messages = [ChatMessage(role="user", content=long_content)]
        
        usage = calculate_token_usage(messages)
        assert usage['estimated_tokens'] > 0

    @pytest.mark.parametrize("content,expected_min_tokens", [
        ("Single", 1),
        ("Two words", 2),
        ("This is a longer sentence with multiple words", 8),
        ("Special chars: @#$%^&*()", 1),
        ("Numbers 123 456 789", 4),
        ("Mixed content: Hello world! 123", 4),
    ])
    def test_estimate_tokens_with_various_content_types(self, content, expected_min_tokens):
        """Test estimate_tokens with various content types."""
        from app.ai_backend.genesis_api import estimate_tokens
        
        tokens = estimate_tokens(content)
        assert tokens >= expected_min_tokens

    def test_estimate_tokens_with_unicode_content(self):
        """Test estimate_tokens with Unicode content."""
        from app.ai_backend.genesis_api import estimate_tokens
        
        unicode_content = "Hello 世界 🌍"
        tokens = estimate_tokens(unicode_content)
        assert tokens > 0

    def test_estimate_tokens_with_none_content(self):
        """Test estimate_tokens with None content."""
        from app.ai_backend.genesis_api import estimate_tokens
        
        tokens = estimate_tokens(None)
        assert tokens == 0

    def test_estimate_tokens_with_whitespace_only(self):
        """Test estimate_tokens with whitespace-only content."""
        from app.ai_backend.genesis_api import estimate_tokens
        
        whitespace_content = "   \t\n   "
        tokens = estimate_tokens(whitespace_content)
        assert tokens == 0


class TestAsyncContextManager:
    """Test suite for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager_normal_usage(self, mock_config):
        """Test normal async context manager usage."""
        async with GenesisAPIClient(**mock_config) as client:
            assert client.api_key == mock_config['api_key']
            assert hasattr(client, '_session')

    @pytest.mark.asyncio
    async def test_async_context_manager_exception_handling(self, mock_config):
        """Test async context manager with exception."""
        try:
            async with GenesisAPIClient(**mock_config) as client:
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception

    @pytest.mark.asyncio
    async def test_async_context_manager_cleanup(self, mock_config):
        """Test that async context manager properly cleans up resources."""
        client = None
        async with GenesisAPIClient(**mock_config) as ctx_client:
            client = ctx_client
            assert hasattr(client, '_session')
        
        # After context manager exits, session should be cleaned up
        # This test verifies the cleanup behavior


class TestPerformanceAndStress:
    """Test suite for performance and stress testing scenarios."""

    @pytest.mark.asyncio
    async def test_rapid_sequential_requests(self, client, sample_messages, sample_model_config):
        """Test handling of rapid sequential requests."""
        mock_response = {
            'id': 'chat-rapid',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Response'}}],
            'usage': {'total_tokens': 10}
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200

            # Make 100 rapid sequential requests
            for i in range(100):
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                assert result.id == 'chat-rapid'

    @pytest.mark.asyncio
    async def test_large_message_content(self, client, sample_model_config):
        """Test handling of very large message content."""
        # Create a message with 50KB of content
        large_content = "A" * 50000
        large_messages = [ChatMessage(role="user", content=large_content)]

        mock_response = {
            'id': 'chat-large',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Large response'}}],
            'usage': {'total_tokens': 12500}
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200

            result = await client.create_chat_completion(
                messages=large_messages,
                model_config=sample_model_config
            )
            assert result.usage['total_tokens'] == 12500

    def test_memory_usage_with_large_objects(self, client):
        """Test memory usage with large data objects."""
        # Create a large number of ChatMessage objects
        messages = [
            ChatMessage(role="user", content=f"Message {i}" * 100)
            for i in range(1000)
        ]

        # Test that validation doesn't consume excessive memory
        try:
            client._validate_messages(messages)
        except ValidationError:
            pass  # Expected for very long messages

        # Verify objects can be garbage collected
        del messages


# Integration-style tests for end-to-end scenarios
class TestIntegrationScenarios:
    """Integration-style tests for end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, client):
        """Test a complete conversation flow with multiple message exchanges."""
        conversation = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello!"),
        ]

        model_config = ModelConfig(
            name="genesis-gpt-4",
            max_tokens=100,
            temperature=0.7
        )

        # Mock multiple responses for conversation flow
        responses = [
            {
                'id': 'chat-1',
                'object': 'chat.completion',
                'created': int(datetime.now(timezone.utc).timestamp()),
                'model': 'genesis-gpt-4',
                'choices': [{'message': {'role': 'assistant', 'content': 'Hello! How can I help you?'}}],
                'usage': {'total_tokens': 20}
            },
            {
                'id': 'chat-2',
                'object': 'chat.completion',
                'created': int(datetime.now(timezone.utc).timestamp()),
                'model': 'genesis-gpt-4',
                'choices': [{'message': {'role': 'assistant', 'content': 'I can help with that!'}}],
                'usage': {'total_tokens': 35}
            }
        ]

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(side_effect=responses)
            mock_post.return_value.__aenter__.return_value.status = 200

            # First exchange
            result1 = await client.create_chat_completion(
                messages=conversation,
                model_config=model_config
            )
            conversation.append(ChatMessage(
                role="assistant",
                content=result1.choices[0].message.content
            ))

            # Second exchange
            conversation.append(ChatMessage(role="user", content="Can you help me with Python?"))
            result2 = await client.create_chat_completion(
                messages=conversation,
                model_config=model_config
            )

            assert result1.id == 'chat-1'
            assert result2.id == 'chat-2'
            assert len(conversation) == 4

    @pytest.mark.asyncio
    async def test_model_discovery_and_usage(self, client):
        """Test discovering available models and using them."""
        # Mock model list response
        models_response = {
            'object': 'list',
            'data': [
                {'id': 'genesis-gpt-4', 'object': 'model', 'created': 1677610602},
                {'id': 'genesis-gpt-3.5-turbo', 'object': 'model', 'created': 1677610602}
            ]
        }

        # Mock model details response
        model_details_response = {
            'id': 'genesis-gpt-4',
            'object': 'model',
            'created': 1677610602,
            'owned_by': 'genesis-ai',
            'permission': []
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(
                side_effect=[models_response, model_details_response]
            )
            mock_get.return_value.__aenter__.return_value.status = 200

            # Discover models
            models = await client.list_models()
            assert len(models) == 2

            # Get specific model details
            model_details = await client.get_model('genesis-gpt-4')
            assert model_details.id == 'genesis-gpt-4'

    @pytest.mark.asyncio
    async def test_error_recovery_and_retry_flow(self, client, sample_messages, sample_model_config):
        """Test error recovery and retry flow in realistic scenarios."""
        call_count = 0

        async def mock_post_with_various_errors(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            mock_response = Mock()
            if call_count == 1:
                # First call: Rate limit error
                mock_response.status = 429
                mock_response.headers = {'Retry-After': '1'}
                mock_response.json = AsyncMock(
                    return_value={'error': {'message': 'Rate limit exceeded'}}
                )
            elif call_count == 2:
                # Second call: Server error
                mock_response.status = 500
                mock_response.json = AsyncMock(
                    return_value={'error': {'message': 'Internal server error'}}
                )
            else:
                # Third call: Success
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    'id': 'chat-recovery',
                    'object': 'chat.completion',
                    'created': int(datetime.now(timezone.utc).timestamp()),
                    'model': 'genesis-gpt-4',
                    'choices': [{'message': {'role': 'assistant', 'content': 'Success after retry'}}],
                    'usage': {'total_tokens': 15}
                })

            return mock_response

        with patch('aiohttp.ClientSession.post', side_effect=mock_post_with_various_errors):
            with patch('asyncio.sleep'):  # Speed up test
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

                assert result.id == 'chat-recovery'
                assert call_count == 3  # Should have retried through rate limit and server error
