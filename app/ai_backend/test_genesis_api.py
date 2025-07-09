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
        Return a mock configuration dictionary for initializing GenesisAPIClient in tests.
        
        Returns:
            dict: Mock values for API key, base URL, timeout, and max retries.
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
        Pytest fixture that provides a GenesisAPIClient instance initialized with the given mock configuration.
        """
        return GenesisAPIClient(**mock_config)

    @pytest.fixture
    def sample_messages(self):
        """
        Provides a list of sample ChatMessage objects simulating a typical conversation for use in tests.
        
        Returns:
            List[ChatMessage]: Sample conversation messages with system, user, and assistant roles.
        """
        return [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is the weather like today?"),
            ChatMessage(role="assistant", content="I don't have access to real-time weather data.")
        ]

    @pytest.fixture
    def sample_model_config(self):
        """
        Return a ModelConfig instance with standard parameters for use in tests.
        
        Returns:
            ModelConfig: A model configuration populated with typical test values.
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
        Verify that GenesisAPIClient initializes with correct attributes when given a valid configuration.
        
        Asserts that the client's API key, base URL, timeout, and max retries match the provided configuration.
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
        Test that initializing GenesisAPIClient with a zero or negative timeout raises a ValueError.
        """
        with pytest.raises(ValueError, match="Timeout must be positive"):
            GenesisAPIClient(api_key='test-key', timeout=-1)

        with pytest.raises(ValueError, match="Timeout must be positive"):
            GenesisAPIClient(api_key='test-key', timeout=0)

    def test_client_initialization_invalid_max_retries(self):
        """
        Test that initializing GenesisAPIClient with a negative max_retries raises a ValueError.
        """
        with pytest.raises(ValueError, match="Max retries must be non-negative"):
            GenesisAPIClient(api_key='test-key', max_retries=-1)

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, client, sample_messages, sample_model_config):
        """
        Test that a successful chat completion request returns a ChatCompletion object with expected attributes.
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
        Test that chat completion with streaming enabled yields the expected sequence of streamed response chunks.
        
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
            Asynchronously yields encoded JSON data chunks to simulate a streaming API response.
            
            Yields:
                bytes: Each yielded value is a JSON-encoded mock data chunk as bytes.
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
        Verify that an authentication error (HTTP 401) during chat completion raises an AuthenticationError with the expected message.
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
        Test that a RateLimitError is raised with the correct retry_after value when the API returns a 429 status during chat completion.
        
        Ensures the client interprets the 'Retry-After' header and raises a RateLimitError with the expected retry_after value.
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
        
        Simulates a 400 Bad Request response from the API when an invalid message role is provided and asserts that a ValidationError is raised with the expected error message.
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
        Test that the chat completion method retries on server errors and succeeds after retry attempts.
        
        Simulates two consecutive server errors followed by a successful response, verifying that the retry mechanism is triggered and the final chat completion result is correct.
        """
        call_count = 0

        async def mock_post_with_failure(*args, **kwargs):
            """
            Simulates an asynchronous HTTP POST request that returns a 500 error response for the first two calls and a successful 200 response with a chat completion payload on the third call.
            
            Returns:
                Mock: A mock response object with a status code and JSON payload reflecting the simulated server behavior.
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
        Test that exceeding the maximum number of retries for server errors during chat completion raises a GenesisAPIError.
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
        Test that a connection error during chat completion raises a GenesisAPIError with a connection error message.
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
        Test that validating an empty message list with the client raises a ValidationError.
        """
        with pytest.raises(ValidationError, match="Messages cannot be empty"):
            client._validate_messages([])

    def test_validate_messages_invalid_role(self, client):
        """
        Test that _validate_messages raises a ValidationError when a message has an invalid role.
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
        Test that validating a message with content exceeding the maximum allowed length raises a ValidationError.
        """
        long_content = "x" * 100000  # Assuming max length is less than this
        invalid_messages = [
            ChatMessage(role="user", content=long_content)
        ]

        with pytest.raises(ValidationError, match="Message content too long"):
            client._validate_messages(invalid_messages)

    def test_validate_model_config_invalid_temperature(self, client, sample_model_config):
        """
        Verify that model configuration validation raises a ValidationError when the temperature parameter is set outside the valid range [0, 2].
        """
        sample_model_config.temperature = -0.5  # Invalid negative temperature

        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(sample_model_config)

        sample_model_config.temperature = 2.5  # Invalid high temperature

        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(sample_model_config)

    def test_validate_model_config_invalid_max_tokens(self, client, sample_model_config):
        """
        Test that _validate_model_config raises ValidationError when max_tokens is zero or negative.
        """
        sample_model_config.max_tokens = 0  # Invalid zero tokens

        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            client._validate_model_config(sample_model_config)

        sample_model_config.max_tokens = -100  # Invalid negative tokens

        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            client._validate_model_config(sample_model_config)

    def test_validate_model_config_invalid_top_p(self, client, sample_model_config):
        """
        Test that a ValidationError is raised if the model config's top_p parameter is set below 0 or above 1.
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
        Test that the client successfully retrieves and parses a list of available models from the API.
        
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
        Verifies that retrieving a model by ID using the client returns a model object with the correct attributes.
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
        Test that attempting to retrieve a model with a non-existent ID raises a GenesisAPIError with the message "Model not found".
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
        Verify that the client's _build_headers method returns HTTP headers containing Authorization, Content-Type, and User-Agent fields.
        """
        headers = client._build_headers()

        assert 'Authorization' in headers
        assert headers['Authorization'] == f'Bearer {client.api_key}'
        assert headers['Content-Type'] == 'application/json'
        assert 'User-Agent' in headers

    def test_build_headers_with_custom_headers(self, client):
        """
        Test that custom headers are merged with default headers when building request headers.
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
        Test that a ChatMessage instance is created with the correct role and content, and that the name attribute defaults to None.
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
        Verify that ModelConfig sets default values for optional parameters when only the name is specified.
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
        Test creation of a ChatCompletion object with specified attributes and verify their values.
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
        Test that a GenesisAPIError instance is created with the correct message and status code.
        """
        error = GenesisAPIError("Test error message", status_code=500)
        assert str(error) == "Test error message"
        assert error.status_code == 500

    def test_authentication_error(self):
        """
        Test that AuthenticationError is instantiated correctly and inherits from GenesisAPIError.
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
        Test that `calculate_token_usage` returns a dictionary with an 'estimated_tokens' key for a list of chat messages.
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
        Verify that the token estimation utility returns the expected token count for a given input string.
        
        Parameters:
            content (str): The input text to estimate tokens for.
            expected_tokens (int): The expected number of tokens for the input.
        """
        from app.ai_backend.genesis_api import estimate_tokens

        tokens = estimate_tokens(content)
        assert tokens == expected_tokens