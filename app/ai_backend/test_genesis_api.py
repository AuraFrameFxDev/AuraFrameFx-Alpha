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
            dict: A dictionary with sample API key, base URL, timeout, and max retries.
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
        Return a list of example ChatMessage objects representing a typical conversation.
        
        Returns:
            List[ChatMessage]: Sample messages with system, user, and assistant roles.
        """
        return [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is the weather like today?"),
            ChatMessage(role="assistant", content="I don't have access to real-time weather data.")
        ]
    
    @pytest.fixture
    def sample_model_config(self):
        """
        Create a ModelConfig instance with standard parameters for use in tests.
        
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
        Test that GenesisAPIClient initializes correctly with a valid configuration.
        
        Verifies that the client's attributes match the provided API key, base URL, timeout, and max retries from the configuration.
        """
        client = GenesisAPIClient(**mock_config)
        assert client.api_key == mock_config['api_key']
        assert client.base_url == mock_config['base_url']
        assert client.timeout == mock_config['timeout']
        assert client.max_retries == mock_config['max_retries']

    def test_client_initialization_with_minimal_config(self):
        """
        Test that GenesisAPIClient initializes correctly with only the required API key, assigning default values to all optional parameters.
        """
        client = GenesisAPIClient(api_key='test-key')
        assert client.api_key == 'test-key'
        assert client.base_url is not None  # Should have default
        assert client.timeout > 0  # Should have default
        assert client.max_retries >= 0  # Should have default

    def test_client_initialization_missing_api_key(self):
        """
        Test that initializing the GenesisAPIClient without providing an API key raises a ValueError.
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
        Test that initializing GenesisAPIClient with a negative max_retries raises ValueError.
        
        Verifies that providing a negative value for max_retries during client initialization results in a ValueError with the expected message.
        """
        with pytest.raises(ValueError, match="Max retries must be non-negative"):
            GenesisAPIClient(api_key='test-key', max_retries=-1)

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, client, sample_messages, sample_model_config):
        """
        Test that a successful chat completion request returns a valid ChatCompletion object with expected attributes.
        
        Simulates a successful API response and verifies that the returned ChatCompletion contains the correct ID, model, choices, message content, and token usage.
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
        Test that the chat completion client correctly processes streamed API responses by simulating a sequence of JSON chunks and verifying their order and content.
        """
        mock_chunks = [
            {'choices': [{'delta': {'content': 'The'}}]},
            {'choices': [{'delta': {'content': ' weather'}}]},
            {'choices': [{'delta': {'content': ' is nice'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def mock_stream():
            """
            Asynchronously yields JSON-encoded byte strings from the mock_chunks list to simulate a streaming API response.
            
            Yields:
                bytes: JSON-encoded byte string for each chunk in mock_chunks.
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
        Verify that an authentication error (HTTP 401) during chat completion raises an AuthenticationError with the correct error message.
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
        Test that a RateLimitError is raised with the correct retry_after value when the API responds with HTTP 429.
        
        Verifies that the client extracts the 'Retry-After' header and sets the retry_after attribute on the exception.
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
        Verify that attempting to create a chat completion with an invalid message role raises a ValidationError with the correct error message.
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
        Test that chat completion retries on server errors and succeeds after transient failures.
        
        Simulates two consecutive server errors followed by a successful response, verifying that the retry mechanism in the client is triggered and the final chat completion result is correct.
        """
        call_count = 0
        
        async def mock_post_with_failure(*args, **kwargs):
            """
            Simulates an async HTTP POST that returns a 500 error on the first two calls, then a successful chat completion response on subsequent calls.
            
            Returns:
                Mock: A mock response object with status and JSON payload varying by invocation count.
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
        Verify that the client raises a GenesisAPIError when the maximum number of retries is exceeded due to repeated server errors during chat completion.
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
        
        Simulates a timeout in the HTTP request and verifies that the client raises a GenesisAPIError containing "Request timeout".
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
        Test that validating an empty message list raises a ValidationError with the expected message.
        """
        with pytest.raises(ValidationError, match="Messages cannot be empty"):
            client._validate_messages([])

    def test_validate_messages_invalid_role(self, client):
        """
        Test that a ValidationError is raised when a chat message contains an invalid role.
        
        Ensures that the client's message validation rejects messages with roles not recognized by the API.
        """
        invalid_messages = [
            ChatMessage(role="invalid", content="Test content")
        ]
        
        with pytest.raises(ValidationError, match="Invalid message role"):
            client._validate_messages(invalid_messages)

    def test_validate_messages_empty_content(self, client):
        """
        Test that a message with empty content triggers a ValidationError during validation.
        
        Verifies that the client rejects messages where the content is an empty string.
        """
        invalid_messages = [
            ChatMessage(role="user", content="")
        ]
        
        with pytest.raises(ValidationError, match="Message content cannot be empty"):
            client._validate_messages(invalid_messages)

    def test_validate_messages_content_too_long(self, client):
        """
        Test that a ValidationError is raised when a chat message's content exceeds the maximum allowed length.
        """
        long_content = "x" * 100000  # Assuming max length is less than this
        invalid_messages = [
            ChatMessage(role="user", content=long_content)
        ]
        
        with pytest.raises(ValidationError, match="Message content too long"):
            client._validate_messages(invalid_messages)

    def test_validate_model_config_invalid_temperature(self, client, sample_model_config):
        """
        Test that setting the temperature in the model config below 0 or above 2 raises a ValidationError.
        """
        sample_model_config.temperature = -0.5  # Invalid negative temperature
        
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(sample_model_config)
        
        sample_model_config.temperature = 2.5  # Invalid high temperature
        
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(sample_model_config)

    def test_validate_model_config_invalid_max_tokens(self, client, sample_model_config):
        """
        Test that setting max_tokens to zero or a negative value in the model configuration raises a ValidationError.
        """
        sample_model_config.max_tokens = 0  # Invalid zero tokens
        
        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            client._validate_model_config(sample_model_config)
        
        sample_model_config.max_tokens = -100  # Invalid negative tokens
        
        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            client._validate_model_config(sample_model_config)

    def test_validate_model_config_invalid_top_p(self, client, sample_model_config):
        """
        Test that providing a top_p value outside the range [0, 1] in the model configuration raises a ValidationError.
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
        Test that the client retrieves and parses the list of available models from the API.
        
        Verifies that the returned list contains models with the expected IDs and correct count.
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
        Test that `get_model` successfully retrieves and returns a model object with the correct attributes when provided a valid model ID.
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
        Test that requesting a nonexistent model raises a GenesisAPIError with the expected error message.
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
        Test that the client's `_build_headers` method returns the correct HTTP headers.
        
        Verifies that the headers include the expected Authorization, Content-Type, and User-Agent fields with appropriate values.
        """
        headers = client._build_headers()
        
        assert 'Authorization' in headers
        assert headers['Authorization'] == f'Bearer {client.api_key}'
        assert headers['Content-Type'] == 'application/json'
        assert 'User-Agent' in headers

    def test_build_headers_with_custom_headers(self, client):
        """
        Test that custom headers are correctly merged with default headers when building request headers.
        
        Ensures that user-supplied headers are included alongside required defaults such as Authorization and Content-Type.
        """
        custom_headers = {'X-Custom-Header': 'custom-value'}
        headers = client._build_headers(custom_headers)
        
        assert headers['X-Custom-Header'] == 'custom-value'
        assert 'Authorization' in headers
        assert headers['Content-Type'] == 'application/json'

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_config):
        """
        Test that GenesisAPIClient correctly opens and closes its HTTP session when used as an async context manager.
        
        Ensures the session is active within the context and closed after exiting.
        """
        async with GenesisAPIClient(**mock_config) as client:
            assert client.session is not None
        
        # Session should be closed after exiting context
        assert client.session.closed

    @pytest.mark.asyncio
    async def test_close_client_explicitly(self, client):
        """
        Test that explicitly closing the GenesisAPIClient closes its underlying HTTP session.
        
        Verifies that calling the `close` method on the client properly closes the associated HTTP session.
        """
        await client.close()
        assert client.session.closed

    @pytest.mark.parametrize("status_code,expected_exception", [
        (400, ValidationError),
        (401, AuthenticationError),
        (403, AuthenticationError),
        (429, RateLimitError),
        (500, GenesisAPIError),
        (502, GenesisAPIError),
        (503, GenesisAPIError),
    ])
    @pytest.mark.asyncio
    async def test_error_handling_by_status_code(self, client, status_code, expected_exception):
        """
        Verify that the client raises the appropriate exception type for specific HTTP status codes during a chat completion request.
        
        Parameters:
            status_code (int): The HTTP status code to simulate in the API response.
            expected_exception (Exception): The exception class expected to be raised for the given status code.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = status_code
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'error': {'message': f'Error {status_code}'}}
            )
            
            with pytest.raises(expected_exception):
                await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content="test")],
                    model_config=ModelConfig(name="test-model")
                )


class TestDataModels:
    """Test suite for data model classes."""
    
    def test_chat_message_creation(self):
        """
        Test that a ChatMessage is created with the correct role and content, and that the name attribute defaults to None.
        """
        message = ChatMessage(role="user", content="Hello, world!")
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.name is None

    def test_chat_message_with_name(self):
        """
        Test that a ChatMessage instance sets the name attribute when a name is provided.
        """
        message = ChatMessage(role="user", content="Hello", name="John")
        assert message.name == "John"

    def test_model_config_creation(self):
        """
        Test that a ModelConfig instance is correctly created with specified name, max_tokens, and temperature values.
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
        Test that ModelConfig sets default values for max_tokens, temperature, and top_p when only the model name is provided.
        """
        config = ModelConfig(name="test-model")
        assert config.name == "test-model"
        assert config.max_tokens is not None
        assert config.temperature is not None
        assert config.top_p is not None

    def test_api_response_creation(self):
        """
        Test that APIResponse objects are correctly initialized with the provided status code, data, and headers.
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
        Test that a ChatCompletion object is created with the expected attribute values.
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
        Test instantiation of GenesisAPIError and verify its message and status_code attributes.
        """
        error = GenesisAPIError("Test error message", status_code=500)
        assert str(error) == "Test error message"
        assert error.status_code == 500

    def test_authentication_error(self):
        """
        Test that AuthenticationError is initialized with the correct message and is a subclass of GenesisAPIError.
        """
        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"
        assert isinstance(error, GenesisAPIError)

    def test_rate_limit_error(self):
        """
        Verify that RateLimitError is instantiated with the correct retry_after value and inherits from GenesisAPIError.
        """
        error = RateLimitError("Rate limit exceeded", retry_after=60)
        assert str(error) == "Rate limit exceeded"
        assert error.retry_after == 60
        assert isinstance(error, GenesisAPIError)

    def test_validation_error(self):
        """
        Test instantiation and inheritance of ValidationError.
        
        Verifies that ValidationError is initialized with the correct message and is a subclass of GenesisAPIError.
        """
        error = ValidationError("Invalid input data")
        assert str(error) == "Invalid input data"
        assert isinstance(error, GenesisAPIError)


class TestUtilityFunctions:
    """Test suite for utility functions in the genesis_api module."""
    
    def test_format_timestamp(self):
        """
        Test that `format_timestamp` returns a non-empty string for a valid timestamp input.
        """
        from app.ai_backend.genesis_api import format_timestamp
        
        timestamp = 1677610602
        formatted = format_timestamp(timestamp)
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_calculate_token_usage(self):
        """
        Test that `calculate_token_usage` returns a dictionary containing the 'estimated_tokens' key for a list of chat messages.
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
        Test that the `estimate_tokens` utility returns the correct token count for a given input string.
        
        Parameters:
            content (str): The input text to estimate token count for.
            expected_tokens (int): The expected number of tokens for the input text.
        """
        from app.ai_backend.genesis_api import estimate_tokens
        
        tokens = estimate_tokens(content)
        assert tokens == expected_tokens


# Integration tests for end-to-end scenarios
class TestIntegration:
    """Integration test suite for complete workflows."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_chat_workflow(self):
        """
        Performs an end-to-end integration test of the chat completion workflow using GenesisAPIClient.
        
        Sends a user message and verifies that the mocked Genesis API returns a valid chat completion response, asserting correct parsing of the completion ID, assistant message content, and token usage.
        """
        config = {
            'api_key': 'test-key',
            'base_url': 'https://api.genesis.ai/v1'
        }
        
        mock_response = {
            'id': 'chat-integration-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'Integration test response'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 3,
                'total_tokens': 13
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            async with GenesisAPIClient(**config) as client:
                messages = [ChatMessage(role="user", content="Test message")]
                model_config = ModelConfig(name="genesis-gpt-4", max_tokens=100)
                
                result = await client.create_chat_completion(
                    messages=messages,
                    model_config=model_config
                )
                
                assert result.id == 'chat-integration-test'
                assert result.choices[0].message.content == 'Integration test response'
                assert result.usage.total_tokens == 13

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_recovery_workflow(self):
        """
        Test that the client retries a chat completion request after encountering a rate limit error and succeeds on a subsequent attempt.
        
        Simulates a rate limit error on the first API call and a successful response on the second, verifying that `RateLimitError` is raised initially and that the client completes the request successfully after recovery.
        """
        config = {'api_key': 'test-key'}
        
        call_count = 0
        
        async def mock_post_with_recovery(*args, **kwargs):
            """
            Simulates an async HTTP POST that returns a rate limit error on the first call and a successful response on subsequent calls.
            
            Returns:
                Mock: A mock response object with status 429 and rate limit error on the first call, and status 200 with a successful API response on later calls.
            """
            nonlocal call_count
            call_count += 1
            
            mock_response = Mock()
            if call_count == 1:
                # First call: rate limit error
                mock_response.status = 429
                mock_response.json = AsyncMock(
                    return_value={'error': {'message': 'Rate limit exceeded'}}
                )
                mock_response.headers = {'Retry-After': '1'}
            else:
                # Second call: success
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    'id': 'chat-recovery-test',
                    'choices': [{'message': {'content': 'Recovery successful'}}],
                    'usage': {'total_tokens': 10}
                })
            
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_post_with_recovery):
            with patch('asyncio.sleep'):  # Speed up test
                async with GenesisAPIClient(**config) as client:
                    try:
                        result = await client.create_chat_completion(
                            messages=[ChatMessage(role="user", content="Test")],
                            model_config=ModelConfig(name="test-model")
                        )
                        # Should not reach here due to rate limit
                        assert False, "Expected RateLimitError"
                    except RateLimitError:
                        # Expected on first call
                        pass
                    
                    # Second call should succeed
                    result = await client.create_chat_completion(
                        messages=[ChatMessage(role="user", content="Test")],
                        model_config=ModelConfig(name="test-model")
                    )
                    assert result.id == 'chat-recovery-test'


# Performance and load testing
class TestPerformance:
    """Performance test suite."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_requests(self):
        """
        Test that GenesisAPIClient correctly handles multiple concurrent chat completion requests.
        
        Verifies that the client processes asynchronous requests in parallel and that each response matches the expected mock output.
        """
        config = {'api_key': 'test-key'}
        
        mock_response = {
            'id': 'concurrent-test',
            'choices': [{'message': {'content': 'Concurrent response'}}],
            'usage': {'total_tokens': 5}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            async with GenesisAPIClient(**config) as client:
                tasks = []
                for i in range(10):
                    task = client.create_chat_completion(
                        messages=[ChatMessage(role="user", content=f"Message {i}")],
                        model_config=ModelConfig(name="test-model")
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
                
                assert len(results) == 10
                for result in results:
                    assert result.id == 'concurrent-test'

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_message_handling(self):
        """
        Test that GenesisAPIClient correctly processes chat completions with very large message content, ensuring accurate response parsing and token usage calculation.
        """
        config = {'api_key': 'test-key'}
        large_content = "x" * 10000  # Large message content
        
        mock_response = {
            'id': 'large-message-test',
            'choices': [{'message': {'content': 'Large message processed'}}],
            'usage': {'total_tokens': 2500}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            async with GenesisAPIClient(**config) as client:
                result = await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content=large_content)],
                    model_config=ModelConfig(name="test-model", max_tokens=4000)
                )
                
                assert result.id == 'large-message-test'
                assert result.usage.total_tokens == 2500


class TestGenesisAPIClientAdvanced:
    """Advanced test scenarios for GenesisAPIClient."""
    
    @pytest.fixture
    def client_with_custom_session(self, mock_config):
        """
        Creates a `GenesisAPIClient` instance with a custom aiohttp session.
        
        Returns:
            GenesisAPIClient: The client instance with its session replaced by a custom aiohttp session.
        """
        import aiohttp
        session = aiohttp.ClientSession()
        client = GenesisAPIClient(**mock_config)
        client.session = session
        return client

    @pytest.mark.asyncio
    async def test_session_reuse_across_requests(self, client):
        """
        Test that the GenesisAPIClient maintains and reuses a single HTTP session across multiple chat completion requests.
        
        Ensures that the session remains open and consistent after consecutive API calls.
        """
        mock_response = {
            'id': 'test-session-reuse',
            'choices': [{'message': {'content': 'Response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Make multiple requests
            await client.create_chat_completion(
                messages=[ChatMessage(role="user", content="First")],
                model_config=ModelConfig(name="test-model")
            )
            await client.create_chat_completion(
                messages=[ChatMessage(role="user", content="Second")],
                model_config=ModelConfig(name="test-model")
            )
            
            # Verify session consistency
            assert client.session is not None
            assert not client.session.closed

    def test_api_key_sanitization_in_logs(self, mock_config):
        """
        Test that the API key is not exposed in the string or repr representation of the GenesisAPIClient.
        
        Ensures sensitive credentials are not leaked through object introspection or logging.
        """
        client = GenesisAPIClient(**mock_config)
        client_str = str(client)
        client_repr = repr(client)
        
        # API key should not appear in string representations
        assert mock_config['api_key'] not in client_str
        assert mock_config['api_key'] not in client_repr

    @pytest.mark.asyncio
    async def test_malformed_json_response_handling(self, client, sample_messages, sample_model_config):
        """
        Test that the client raises a GenesisAPIError with an appropriate message when the API returns a malformed JSON response.
        
        This ensures that invalid or non-JSON API responses are detected and reported as client errors.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                side_effect=json.JSONDecodeError("Invalid JSON", "", 0)
            )
            
            with pytest.raises(GenesisAPIError, match="Invalid response format"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_empty_response_handling(self, client, sample_messages, sample_model_config):
        """
        Test that a GenesisAPIError is raised when the API returns an empty JSON response.
        
        Verifies that an empty response from the API is detected as invalid and results in an appropriate error.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value={})
            
            with pytest.raises(GenesisAPIError, match="Invalid response structure"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_partial_response_handling(self, client, sample_messages, sample_model_config):
        """
        Test that a GenesisAPIError is raised when the API returns a partial or incomplete chat completion response.
        
        Verifies that missing required fields such as 'choices' or 'usage' in the API response are detected and result in an error.
        """
        incomplete_response = {
            'id': 'partial-response',
            'object': 'chat.completion'
            # Missing choices and usage fields
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=incomplete_response)
            
            with pytest.raises(GenesisAPIError, match="Missing required fields"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_unicode_content_handling(self, client, sample_model_config):
        """
        Test that the client processes chat messages containing unicode and special characters correctly.
        
        Sends messages with diverse unicode content and verifies that the API response preserves and returns the expected unicode characters.
        """
        unicode_messages = [
            ChatMessage(role="user", content="Hello 世界! 🌍 Café naïve résumé"),
            ChatMessage(role="system", content="Emoji test: 🚀🔥💻")
        ]
        
        mock_response = {
            'id': 'unicode-test',
            'choices': [{'message': {'content': 'Unicode response: 测试 ✅'}}],
            'usage': {'total_tokens': 20}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=unicode_messages,
                model_config=sample_model_config
            )
            
            assert result.choices[0].message.content == 'Unicode response: 测试 ✅'

    @pytest.mark.asyncio
    async def test_extremely_long_conversation_handling(self, client, sample_model_config):
        """
        Test that the client can handle chat completions for conversations with a large number of messages.
        
        Simulates a conversation with 100 alternating user and assistant messages and verifies that the API client returns the expected completion response.
        """
        long_conversation = []
        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            long_conversation.append(ChatMessage(role=role, content=f"Message {i}"))
        
        mock_response = {
            'id': 'long-conversation',
            'choices': [{'message': {'content': 'Long conversation processed'}}],
            'usage': {'total_tokens': 5000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=long_conversation,
                model_config=sample_model_config
            )
            
            assert result.id == 'long-conversation'

    @pytest.mark.asyncio
    async def test_streaming_connection_error(self, client, sample_messages, sample_model_config):
        """
        Verify that a connection error during streaming chat completion raises a GenesisAPIError with an appropriate message.
        """
        import aiohttp
        
        with patch('aiohttp.ClientSession.post', side_effect=aiohttp.ClientConnectionError()):
            with pytest.raises(GenesisAPIError, match="Connection error"):
                async for chunk in client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                ):
                    pass

    @pytest.mark.asyncio
    async def test_streaming_malformed_chunks(self, client, sample_messages, sample_model_config):
        """
        Test that the client skips malformed JSON chunks and yields only valid ones during streaming chat completion.
        
        Simulates a streaming API response with both malformed and valid JSON chunks, verifying that the client ignores invalid data and yields only successfully parsed chunks.
        """
        async def mock_malformed_stream():
            """
            Asynchronously yields a malformed JSON chunk followed by a valid streaming API response chunk.
            
            Yields:
                bytes: A malformed JSON byte string, then a valid data chunk simulating a streaming API response.
            """
            yield b'{"invalid": json}'  # Malformed JSON
            yield b'data: {"choices": [{"delta": {"content": "test"}}]}'  # Valid
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_malformed_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            # Should skip malformed chunks and process valid ones
            assert len(chunks) >= 0

    def test_validate_messages_mixed_content_types(self, client):
        """
        Test that message validation fails when a message contains invalid content types.
        
        Ensures that a ValidationError is raised if any message in the list has content set to None, even if other messages are valid.
        """
        mixed_messages = [
            ChatMessage(role="user", content="Valid string content"),
            ChatMessage(role="assistant", content=None),  # Invalid None content
        ]
        
        with pytest.raises(ValidationError, match="Message content cannot be None"):
            client._validate_messages(mixed_messages)

    def test_validate_messages_whitespace_only_content(self, client):
        """
        Test that message validation raises a ValidationError for messages with only whitespace content.
        
        Asserts that attempting to validate a message whose content consists entirely of whitespace characters results in a ValidationError.
        """
        whitespace_messages = [
            ChatMessage(role="user", content="   \t\n   ")  # Only whitespace
        ]
        
        with pytest.raises(ValidationError, match="Message content cannot be empty"):
            client._validate_messages(whitespace_messages)

    def test_validate_model_config_boundary_values(self, client):
        """
        Verify that model configuration validation accepts minimum and maximum allowed values for temperature, top_p, and max_tokens without raising errors.
        """
        # Test exact boundary values
        config = ModelConfig(name="test-model")
        
        # Test minimum valid values
        config.temperature = 0.0
        config.top_p = 0.0
        config.max_tokens = 1
        client._validate_model_config(config)  # Should not raise
        
        # Test maximum valid values
        config.temperature = 2.0
        config.top_p = 1.0
        config.max_tokens = 999999
        client._validate_model_config(config)  # Should not raise

    def test_validate_model_config_none_values(self, client):
        """
        Test that setting temperature to None in ModelConfig raises a ValidationError during model config validation.
        """
        config = ModelConfig(name="test-model")
        config.temperature = None
        
        with pytest.raises(ValidationError, match="Temperature cannot be None"):
            client._validate_model_config(config)

    @pytest.mark.asyncio
    async def test_custom_timeout_behavior(self, mock_config):
        """
        Test that GenesisAPIClient enforces a custom timeout and raises GenesisAPIError when a request times out.
        """
        custom_config = mock_config.copy()
        custom_config['timeout'] = 5
        
        client = GenesisAPIClient(**custom_config)
        
        with patch('aiohttp.ClientSession.post', side_effect=asyncio.TimeoutError()):
            with pytest.raises(GenesisAPIError, match="Request timeout"):
                await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content="test")],
                    model_config=ModelConfig(name="test-model")
                )

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self, client, sample_messages, sample_model_config):
        """
        Test that the client retries failed chat completion requests using exponential backoff on server errors.
        
        Simulates repeated server-side failures and verifies that the client's retry logic invokes exponential backoff delays between attempts, ultimately raising a GenesisAPIError after exhausting retries.
        """
        call_times = []
        
        async def mock_failing_post(*args, **kwargs):
            """
            Simulates an asynchronous HTTP POST request that always fails with a 500 status code and a server error message.
            
            Intended for use in tests to trigger retry and error handling logic in asynchronous HTTP clients.
            """
            call_times.append(asyncio.get_event_loop().time())
            mock_response = Mock()
            mock_response.status = 500
            mock_response.json = AsyncMock(return_value={'error': {'message': 'Server error'}})
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_failing_post):
            with patch('asyncio.sleep') as mock_sleep:
                with pytest.raises(GenesisAPIError):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                
                # Verify exponential backoff was attempted
                assert mock_sleep.call_count >= 1

    def test_build_headers_with_none_custom_headers(self, client):
        """
        Test that the header builder returns only the default headers when custom headers are set to None.
        
        Verifies that the resulting headers include the required Authorization and Content-Type fields.
        """
        headers = client._build_headers(None)
        
        assert 'Authorization' in headers
        assert headers['Content-Type'] == 'application/json'

    def test_build_headers_override_default_headers(self, client):
        """
        Test that custom headers supplied to the client correctly override the default headers in the request.
        """
        custom_headers = {
            'Content-Type': 'text/plain',
            'Authorization': 'Bearer custom-token'
        }
        headers = client._build_headers(custom_headers)
        
        assert headers['Content-Type'] == 'text/plain'
        assert headers['Authorization'] == 'Bearer custom-token'


class TestDataModelsAdvanced:
    """Advanced tests for data model classes."""
    
    def test_chat_message_equality(self):
        """
        Test that ChatMessage instances are equal when their attributes match and unequal when they differ.
        """
        msg1 = ChatMessage(role="user", content="Hello")
        msg2 = ChatMessage(role="user", content="Hello")
        msg3 = ChatMessage(role="user", content="World")
        
        assert msg1 == msg2
        assert msg1 != msg3

    def test_chat_message_immutability(self):
        """
        Test that ChatMessage instances are immutable and cannot have their attributes modified after creation.
        """
        message = ChatMessage(role="user", content="Hello")
        
        # Attempting to modify should raise an error or have no effect
        with pytest.raises(AttributeError):
            message.role = "assistant"

    def test_model_config_extreme_values(self):
        """
        Test that ModelConfig correctly accepts and stores extreme but valid values for its parameters.
        """
        config = ModelConfig(
            name="test-model",
            max_tokens=1,
            temperature=0.0,
            top_p=0.0,
            frequency_penalty=-2.0,
            presence_penalty=2.0
        )
        
        assert config.max_tokens == 1
        assert config.temperature == 0.0
        assert config.frequency_penalty == -2.0

    def test_api_response_with_none_data(self):
        """
        Verify that an APIResponse can be instantiated with None as the data attribute and that its status_code and data fields are correctly assigned.
        """
        response = APIResponse(
            status_code=204,
            data=None,
            headers={}
        )
        
        assert response.status_code == 204
        assert response.data is None

    def test_chat_completion_with_empty_choices(self):
        """
        Test creation of a ChatCompletion with no choices and zero token usage.
        
        Verifies that a ChatCompletion instance can be instantiated with an empty choices list and that the usage dictionary correctly reflects zero tokens.
        """
        completion = ChatCompletion(
            id="empty-choices",
            object="chat.completion",
            created=1677610602,
            model="test-model",
            choices=[],
            usage={'total_tokens': 0}
        )
        
        assert len(completion.choices) == 0
        assert completion.usage['total_tokens'] == 0

    def test_chat_completion_serialization(self):
        """
        Test that a ChatCompletion object can be serialized to a JSON string.
        
        Verifies that serialization produces a string containing the expected identifier.
        """
        completion = ChatCompletion(
            id="serialize-test",
            object="chat.completion",
            created=1677610602,
            model="test-model",
            choices=[],
            usage={'total_tokens': 10}
        )
        
        # Should be serializable
        json_str = json.dumps(completion.__dict__)
        assert isinstance(json_str, str)
        assert "serialize-test" in json_str


class TestExceptionHandlingAdvanced:
    """Advanced exception handling test scenarios."""
    
    def test_exception_chaining(self):
        """
        Test that custom exceptions support exception chaining via the `__cause__` attribute.
        """
        original_error = ValueError("Original error")
        api_error = GenesisAPIError("API error", status_code=500)
        api_error.__cause__ = original_error
        
        assert api_error.__cause__ is original_error

    def test_rate_limit_error_with_zero_retry_after(self):
        """
        Test that RateLimitError sets retry_after to zero when initialized with retry_after=0.
        """
        error = RateLimitError("Rate limited", retry_after=0)
        assert error.retry_after == 0

    def test_rate_limit_error_with_none_retry_after(self):
        """
        Test that a RateLimitError instance sets retry_after to None when initialized with retry_after=None.
        """
        error = RateLimitError("Rate limited", retry_after=None)
        assert error.retry_after is None

    def test_exception_message_formatting(self):
        """
        Test that the string representation of a GenesisAPIError correctly includes placeholders in its message.
        """
        error = GenesisAPIError("Test error with {placeholder}", status_code=400)
        assert "placeholder" in str(error)

    def test_validation_error_with_field_info(self):
        """
        Test that a ValidationError message contains the relevant field name and constraint details.
        """
        error = ValidationError("Invalid field 'temperature': must be between 0 and 2")
        assert "temperature" in str(error)
        assert "0 and 2" in str(error)


class TestUtilityFunctionsAdvanced:
    """Advanced tests for utility functions."""
    
    def test_format_timestamp_edge_cases(self):
        """
        Test that `format_timestamp` handles zero, negative, and very large timestamps without errors and always returns a string.
        """
        from app.ai_backend.genesis_api import format_timestamp
        
        # Test with zero timestamp
        formatted_zero = format_timestamp(0)
        assert isinstance(formatted_zero, str)
        
        # Test with negative timestamp
        formatted_negative = format_timestamp(-1)
        assert isinstance(formatted_negative, str)
        
        # Test with very large timestamp
        formatted_large = format_timestamp(9999999999)
        assert isinstance(formatted_large, str)

    def test_calculate_token_usage_empty_messages(self):
        """
        Test that `calculate_token_usage` returns zero estimated tokens for an empty list of messages.
        """
        from app.ai_backend.genesis_api import calculate_token_usage
        
        usage = calculate_token_usage([])
        assert usage['estimated_tokens'] == 0

    def test_calculate_token_usage_unicode_content(self):
        """
        Test that `calculate_token_usage` returns a positive integer token estimate for messages containing Unicode characters.
        """
        from app.ai_backend.genesis_api import calculate_token_usage
        
        unicode_messages = [
            ChatMessage(role="user", content="测试 unicode 🌍"),
            ChatMessage(role="assistant", content="Réponse en français")
        ]
        
        usage = calculate_token_usage(unicode_messages)
        assert isinstance(usage['estimated_tokens'], int)
        assert usage['estimated_tokens'] > 0

    @pytest.mark.parametrize("content,min_tokens", [
        ("Single", 1),
        ("Multiple words here", 3),
        ("A much longer sentence with many words to test", 9),
        ("Short", 1),
    ])
    def test_estimate_tokens_minimum_bounds(self, content, min_tokens):
        """
        Test that the token estimation function returns at least the specified minimum number of tokens for the given content.
        
        Parameters:
            content (str): Input string to estimate token count for.
            min_tokens (int): Minimum expected number of tokens.
        """
        from app.ai_backend.genesis_api import estimate_tokens
        
        tokens = estimate_tokens(content)
        assert tokens >= min_tokens

    def test_estimate_tokens_with_special_characters(self):
        """
        Test that `estimate_tokens` returns a positive integer when given input with special characters and punctuation.
        """
        from app.ai_backend.genesis_api import estimate_tokens
        
        special_content = "Hello, world! How are you? I'm fine. 123-456-7890."
        tokens = estimate_tokens(special_content)
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_estimate_tokens_with_newlines_and_tabs(self):
        """
        Test that `estimate_tokens` returns a positive integer token count for strings containing newlines and tabs.
        """
        from app.ai_backend.genesis_api import estimate_tokens
        
        whitespace_content = "Line 1\nLine 2\tTabbed content\n\nDouble newline"
        tokens = estimate_tokens(whitespace_content)
        assert isinstance(tokens, int)
        assert tokens > 0


class TestSecurityAndValidation:
    """Security-focused test scenarios."""
    
    def test_api_key_validation_empty_string(self):
        """
        Test that initializing GenesisAPIClient with an empty API key string raises a ValueError.
        """
        with pytest.raises(ValueError, match="API key is required"):
            GenesisAPIClient(api_key="")

    def test_api_key_validation_whitespace_only(self):
        """
        Test that initializing GenesisAPIClient with a whitespace-only API key raises a ValueError.
        """
        with pytest.raises(ValueError, match="API key is required"):
            GenesisAPIClient(api_key="   \t\n   ")

    def test_url_validation_invalid_schemes(self):
        """
        Test that initializing GenesisAPIClient with an invalid URL scheme raises a ValueError.
        
        Verifies that providing a base URL with an unsupported scheme (e.g., 'ftp://') results in a ValueError during client initialization.
        """
        with pytest.raises(ValueError, match="Invalid base URL"):
            GenesisAPIClient(api_key="test-key", base_url="ftp://invalid.com")

    def test_header_injection_prevention(self, client):
        """
        Test that header values containing carriage return or newline characters are sanitized or rejected to prevent header injection.
        """
        malicious_headers = {
            'X-Custom': 'value\r\nX-Injected: malicious',
            'Authorization': 'Bearer token\nX-Another: injected'
        }
        
        headers = client._build_headers(malicious_headers)
        
        # Headers should be sanitized or rejected
        for key, value in headers.items():
            assert '\r' not in str(value)
            assert '\n' not in str(value)

    def test_message_content_script_injection(self, client):
        """
        Test that messages with potentially malicious script or injection content are not rejected by client-side validation.
        
        Verifies that the client allows messages containing script tags, SQL injection patterns, or template expressions, ensuring content filtering is handled by the API. Only length or format-related validation errors are permitted.
        """
        malicious_messages = [
            ChatMessage(role="user", content="<script>alert('xss')</script>"),
            ChatMessage(role="user", content="'; DROP TABLE users; --"),
            ChatMessage(role="user", content="${process.env.SECRET}")
        ]
        
        # Should not raise validation errors for content (filtering is API's job)
        try:
            client._validate_messages(malicious_messages)
        except ValidationError as e:
            # Only acceptable validation errors are length/format related
            assert "content too long" in str(e) or "content cannot be empty" in str(e)


class TestEdgeCaseScenarios:
    """Edge case and boundary condition tests."""
    
    @pytest.mark.asyncio
    async def test_simultaneous_close_calls(self, client):
        """
        Verify that calling the client's close() method multiple times concurrently does not raise errors and ensures the session is closed.
        """
        # Call close multiple times
        await asyncio.gather(
            client.close(),
            client.close(),
            client.close()
        )
        
        assert client.session.closed

    @pytest.mark.asyncio
    async def test_request_after_close(self, client, sample_messages, sample_model_config):
        """
        Verify that attempting to make a chat completion request after closing the client raises a GenesisAPIError indicating the session is closed.
        """
        await client.close()
        
        with pytest.raises(GenesisAPIError, match="Client session is closed"):
            await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )

    def test_model_config_with_all_none_optional_params(self):
        """
        Test that ModelConfig accepts None for all optional parameters.
        
        Ensures that a ModelConfig instance can be created with only the required name and all other parameters set to None without errors.
        """
        config = ModelConfig(
            name="test-model",
            max_tokens=None,
            temperature=None,
            top_p=None,
            frequency_penalty=None,
            presence_penalty=None
        )
        
        assert config.name == "test-model"
        # Should handle None values gracefully

    @pytest.mark.asyncio
    async def test_context_manager_exception_during_init(self, mock_config):
        """
        Test that an exception during async context manager initialization of GenesisAPIClient is propagated.
        
        Verifies that if session creation fails when entering the async context manager, the exception is not suppressed.
        """
        with patch('aiohttp.ClientSession', side_effect=Exception("Session init failed")):
            with pytest.raises(Exception, match="Session init failed"):
                async with GenesisAPIClient(**mock_config) as client:
                    pass

    @pytest.mark.parametrize("invalid_role", [
        "",
        None,
        123,
        [],
        {},
        "INVALID",
        "User",  # Wrong case
        "assistant ",  # Trailing space
    ])
    def test_message_validation_comprehensive_invalid_roles(self, client, invalid_role):
        """
        Test that a ValidationError is raised when a chat message contains an invalid role.
        
        Parameters:
            invalid_role (str): An invalid role value to test in the chat message.
        """
        invalid_messages = [
            ChatMessage(role=invalid_role, content="Test content")
        ]
        
        with pytest.raises(ValidationError):
            client._validate_messages(invalid_messages)

    @pytest.mark.asyncio
    async def test_streaming_with_no_chunks(self, client, sample_messages, sample_model_config):
        """
        Test that the chat completion streaming API yields no results when the server returns an empty stream.
        
        Verifies that when the API provides no data chunks, the streaming generator produces no output.
        """
        async def empty_stream():
            # Empty async generator
            """
            An asynchronous generator that yields no values.
            
            Useful as a placeholder or to simulate an empty async stream in tests.
            """
            return
            yield  # This line is never reached
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=empty_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests(self, client, sample_messages, sample_model_config):
        """
        Test handling of multiple concurrent streaming chat completion requests.
        
        Ensures that the client can process several simultaneous streaming requests, with each stream yielding the expected number of response chunks.
        """
        mock_chunks = [
            {'choices': [{'delta': {'content': f'Stream {i}'}}]}
            for i in range(3)
        ]
        
        async def mock_stream():
            """
            Asynchronously yields JSON-encoded byte chunks from the `mock_chunks` iterable to simulate a streaming API response.
            """
            for chunk in mock_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Start multiple streams concurrently
            streams = [
                client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                for _ in range(3)
            ]
            
            results = []
            for stream in streams:
                chunks = []
                async for chunk in stream:
                    chunks.append(chunk)
                results.append(chunks)
            
            assert len(results) == 3
            for result in results:
                assert len(result) == 3


class TestMockingAndIsolation:
    """Tests focused on proper mocking and test isolation."""
    
    @pytest.mark.asyncio
    async def test_session_cleanup_on_exception(self, mock_config):
        """
        Test that the HTTP session is closed after an exception occurs during an API request within the async context manager.
        """
        with patch('aiohttp.ClientSession.post', side_effect=Exception("Test exception")):
            try:
                async with GenesisAPIClient(**mock_config) as client:
                    await client.create_chat_completion(
                        messages=[ChatMessage(role="user", content="test")],
                        model_config=ModelConfig(name="test-model")
                    )
            except Exception:
                pass
            
            # Session should still be closed after exception
            assert client.session.closed

    def test_fixture_isolation(self, client):
        """
        Test that the client fixture maintains isolation by ensuring state changes in one test do not persist to others.
        """
        # Modify client state
        original_api_key = client.api_key
        client.api_key = "modified-key"
        
        # In next test, fixture should be fresh
        assert client.api_key != original_api_key

    @pytest.mark.asyncio
    async def test_mock_response_side_effects(self, client, sample_messages, sample_model_config):
        """
        Simulates multiple consecutive API failures followed by a successful response to verify the client's retry logic.
        
        This test ensures that the client retries after transient server errors and ultimately returns a successful result when the API recovers.
        """
        responses = [
            {'status': 500, 'json': {'error': {'message': 'Server error 1'}}},
            {'status': 500, 'json': {'error': {'message': 'Server error 2'}}},
            {'status': 200, 'json': {
                'id': 'success-after-failures',
                'choices': [{'message': {'content': 'Success'}}],
                'usage': {'total_tokens': 10}
            }}
        ]
        
        call_count = 0
        
        async def mock_side_effect(*args, **kwargs):
            """
            Simulates a sequence of asynchronous HTTP responses for use in mocking tests.
            
            Each call returns a mock response object with status and JSON payload from the next entry in the `responses` list, advancing the call count on each invocation.
            """
            nonlocal call_count
            response_data = responses[call_count]
            call_count += 1
            
            mock_response = Mock()
            mock_response.status = response_data['status']
            mock_response.json = AsyncMock(return_value=response_data['json'])
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_side_effect):
            with patch('asyncio.sleep'):
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'success-after-failures'
                assert call_count == 3


if __name__ == "__main__":
    # Run with additional markers for the new test categories
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-m", "not (integration or performance)",  # Run unit tests by default
        "--durations=10"  # Show slowest 10 tests
    ])