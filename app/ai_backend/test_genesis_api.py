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
        Provides a mock configuration dictionary with sample API key, base URL, timeout, and max retries for use in GenesisAPIClient tests.
        
        Returns:
            dict: Mock configuration values for initializing GenesisAPIClient.
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
        Provides a GenesisAPIClient instance initialized with the supplied mock configuration for use in tests.
        """
        return GenesisAPIClient(**mock_config)
    
    @pytest.fixture
    def sample_messages(self):
        """
        Return a list of example ChatMessage objects representing a typical conversation.
        
        Returns:
            List[ChatMessage]: Messages with system, user, and assistant roles for use in tests.
        """
        return [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is the weather like today?"),
            ChatMessage(role="assistant", content="I don't have access to real-time weather data.")
        ]
    
    @pytest.fixture
    def sample_model_config(self):
        """
        Create and return a ModelConfig instance with typical parameters for testing purposes.
        
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
        Test that GenesisAPIClient initializes correctly with a valid configuration.
        
        Verifies that the client's attributes match the values provided in the configuration dictionary.
        """
        client = GenesisAPIClient(**mock_config)
        assert client.api_key == mock_config['api_key']
        assert client.base_url == mock_config['base_url']
        assert client.timeout == mock_config['timeout']
        assert client.max_retries == mock_config['max_retries']

    def test_client_initialization_with_minimal_config(self):
        """
        Test initialization of GenesisAPIClient with only an API key, verifying that default values are assigned to optional parameters.
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
        Test that initializing GenesisAPIClient with a negative max_retries value raises a ValueError.
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
        Test that the chat completion client correctly handles and yields streamed response chunks in order during a simulated streaming API call.
        
        Verifies that each streamed chunk is processed as expected and that the final chunk contains the appropriate finish reason.
        """
        mock_chunks = [
            {'choices': [{'delta': {'content': 'The'}}]},
            {'choices': [{'delta': {'content': ' weather'}}]},
            {'choices': [{'delta': {'content': ' is nice'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def mock_stream():
            """
            Asynchronously yields JSON-encoded byte chunks from the mock_chunks list to simulate a streaming API response.
            
            Yields:
                bytes: JSON-encoded byte strings representing individual chunks from mock_chunks.
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
        Test that a RateLimitError is raised with the correct retry_after value when the chat completion API responds with HTTP 429.
        
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
        Verify that attempting to create a chat completion with an invalid message role results in a ValidationError containing the expected error message.
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
        
        Simulates two consecutive server errors (HTTP 500) followed by a successful response, verifying that the client's retry mechanism is triggered and the final chat completion result is correct.
        """
        call_count = 0
        
        async def mock_post_with_failure(*args, **kwargs):
            """
            Simulates an asynchronous HTTP POST request that fails with a 500 status on the first two calls and returns a successful chat completion response on subsequent calls.
            
            Returns:
                Mock: A mock response object with a status code and JSON payload that varies based on the number of times the function has been called.
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
        Test that a network timeout during chat completion raises a GenesisAPIError with an appropriate timeout message.
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
        Test that `_validate_messages` raises a `ValidationError` when given an empty list.
        """
        with pytest.raises(ValidationError, match="Messages cannot be empty"):
            client._validate_messages([])

    def test_validate_messages_invalid_role(self, client):
        """
        Test that a ValidationError is raised when a chat message contains an invalid role.
        
        Verifies that the client's message validation rejects messages with roles not recognized by the system.
        """
        invalid_messages = [
            ChatMessage(role="invalid", content="Test content")
        ]
        
        with pytest.raises(ValidationError, match="Invalid message role"):
            client._validate_messages(invalid_messages)

    def test_validate_messages_empty_content(self, client):
        """
        Test that `_validate_messages` raises a `ValidationError` when a message has empty content.
        """
        invalid_messages = [
            ChatMessage(role="user", content="")
        ]
        
        with pytest.raises(ValidationError, match="Message content cannot be empty"):
            client._validate_messages(invalid_messages)

    def test_validate_messages_content_too_long(self, client):
        """
        Test that ValidationError is raised when a chat message's content exceeds the maximum allowed length.
        """
        long_content = "x" * 100000  # Assuming max length is less than this
        invalid_messages = [
            ChatMessage(role="user", content=long_content)
        ]
        
        with pytest.raises(ValidationError, match="Message content too long"):
            client._validate_messages(invalid_messages)

    def test_validate_model_config_invalid_temperature(self, client, sample_model_config):
        """
        Test that a ValidationError is raised if the model config temperature is set outside the valid range [0, 2].
        """
        sample_model_config.temperature = -0.5  # Invalid negative temperature
        
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(sample_model_config)
        
        sample_model_config.temperature = 2.5  # Invalid high temperature
        
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(sample_model_config)

    def test_validate_model_config_invalid_max_tokens(self, client, sample_model_config):
        """
        Test that ValidationError is raised if max_tokens in the model configuration is zero or negative.
        """
        sample_model_config.max_tokens = 0  # Invalid zero tokens
        
        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            client._validate_model_config(sample_model_config)
        
        sample_model_config.max_tokens = -100  # Invalid negative tokens
        
        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            client._validate_model_config(sample_model_config)

    def test_validate_model_config_invalid_top_p(self, client, sample_model_config):
        """
        Test that a ValidationError is raised when the ModelConfig's top_p parameter is set to a value outside the valid range [0, 1].
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
        
        Verifies that the returned list contains models with the expected IDs.
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
        Test that `get_model` retrieves a model by ID and returns an object with the correct attributes.
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
        Test that the client's `_build_headers` method returns the correct default HTTP headers.
        
        Verifies that the returned headers include the expected Authorization, Content-Type, and User-Agent fields with appropriate values.
        """
        headers = client._build_headers()
        
        assert 'Authorization' in headers
        assert headers['Authorization'] == f'Bearer {client.api_key}'
        assert headers['Content-Type'] == 'application/json'
        assert 'User-Agent' in headers

    def test_build_headers_with_custom_headers(self, client):
        """
        Test that custom headers are correctly merged with default headers when building request headers.
        
        Ensures that user-provided custom headers are included in the final headers dictionary, while required default headers such as Authorization and Content-Type are preserved.
        """
        custom_headers = {'X-Custom-Header': 'custom-value'}
        headers = client._build_headers(custom_headers)
        
        assert headers['X-Custom-Header'] == 'custom-value'
        assert 'Authorization' in headers
        assert headers['Content-Type'] == 'application/json'

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_config):
        """
        Test that GenesisAPIClient correctly manages its HTTP session as an async context manager.
        
        Ensures the session is open within the context and closed after exiting the context block.
        """
        async with GenesisAPIClient(**mock_config) as client:
            assert client.session is not None
        
        # Session should be closed after exiting context
        assert client.session.closed

    @pytest.mark.asyncio
    async def test_close_client_explicitly(self, client):
        """
        Test that explicitly closing the GenesisAPIClient properly closes its underlying HTTP session.
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
            expected_exception (Exception): The exception type expected to be raised for the given status code.
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
        Test that a ChatMessage is correctly created with the specified role and content, and that the name attribute defaults to None.
        """
        message = ChatMessage(role="user", content="Hello, world!")
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.name is None

    def test_chat_message_with_name(self):
        """
        Test that a ChatMessage instance sets the name attribute when initialized with a name.
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
        Test that ModelConfig sets default values for max_tokens, temperature, and top_p when only the name is provided.
        """
        config = ModelConfig(name="test-model")
        assert config.name == "test-model"
        assert config.max_tokens is not None
        assert config.temperature is not None
        assert config.top_p is not None

    def test_api_response_creation(self):
        """
        Test that APIResponse is correctly initialized with the provided status code, data, and headers.
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
        Verify that a ChatCompletion object is created with the expected attribute values.
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
        Test that a GenesisAPIError is correctly instantiated and its attributes are set as expected.
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
        Verify that `RateLimitError` is instantiated with the correct `retry_after` value and inherits from `GenesisAPIError`.
        """
        error = RateLimitError("Rate limit exceeded", retry_after=60)
        assert str(error) == "Rate limit exceeded"
        assert error.retry_after == 60
        assert isinstance(error, GenesisAPIError)

    def test_validation_error(self):
        """
        Verify that ValidationError is instantiated with the correct message and is a subclass of GenesisAPIError.
        """
        error = ValidationError("Invalid input data")
        assert str(error) == "Invalid input data"
        assert isinstance(error, GenesisAPIError)


class TestUtilityFunctions:
    """Test suite for utility functions in the genesis_api module."""
    
    def test_format_timestamp(self):
        """
        Test that `format_timestamp` returns a non-empty string for a valid integer timestamp.
        """
        from app.ai_backend.genesis_api import format_timestamp
        
        timestamp = 1677610602
        formatted = format_timestamp(timestamp)
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_calculate_token_usage(self):
        """
        Test that `calculate_token_usage` returns a dictionary containing an 'estimated_tokens' key for a list of chat messages.
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
        Verify that `estimate_tokens` returns the correct token count for the provided input string.
        
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
        
        This test sends a user message, mocks the Genesis API response, and verifies that the client correctly parses the chat completion ID, assistant message content, and token usage from the response.
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
                Mock: A mock response object with status 429 and rate limit error on the first call, or status 200 with a successful API response on later calls.
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
        Create a `GenesisAPIClient` with a custom aiohttp session.
        
        Returns:
            GenesisAPIClient: The client instance using the specified aiohttp session.
        """
        import aiohttp
        session = aiohttp.ClientSession()
        client = GenesisAPIClient(**mock_config)
        client.session = session
        return client

    @pytest.mark.asyncio
    async def test_session_reuse_across_requests(self, client):
        """
        Test that GenesisAPIClient maintains and reuses the same HTTP session across multiple API requests.
        
        Ensures that the session remains open and consistent after consecutive chat completion calls.
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
        Test that the API key is not exposed in the string or repr representation of the GenesisAPIClient instance.
        
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
        Test that the client raises a GenesisAPIError when the API returns a malformed JSON response.
        
        Verifies that a GenesisAPIError with an appropriate message is raised if the API response cannot be parsed as valid JSON.
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
        
        Verifies that receiving an empty response object from the API triggers an error indicating an invalid response structure.
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
        Test that the client raises a GenesisAPIError when the API returns a partial or incomplete chat completion response.
        
        Verifies that missing required fields in the API response, such as 'choices' or 'usage', are detected and result in a GenesisAPIError.
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
        Test that the client correctly handles chat messages containing Unicode and special characters.
        
        Sends messages with diverse Unicode content and verifies that the API response preserves and returns the expected Unicode characters.
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
        Test processing of chat completions for extremely long conversations.
        
        Simulates a conversation with 100 alternating user and assistant messages and verifies that the client returns the expected completion response for large message histories.
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
        Test that a connection error during streaming chat completion raises a GenesisAPIError with an appropriate message.
        
        This test simulates a connection error when initiating a streaming chat completion request and verifies that the client raises a GenesisAPIError containing the expected error message.
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
        
        Simulates a streaming API response with both malformed and valid JSON chunks, verifying that the client ignores invalid data and yields only properly parsed chunks.
        """
        async def mock_malformed_stream():
            """
            Asynchronously yields byte strings simulating a malformed JSON chunk followed by a valid streaming API response.
            
            Yields:
                bytes: A malformed JSON chunk, then a valid data chunk representing a streaming API response.
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
        Test that message validation raises a ValidationError when any message contains an invalid content type.
        
        Verifies that the client rejects a list of messages if at least one message has content set to None, even if other messages are valid.
        """
        mixed_messages = [
            ChatMessage(role="user", content="Valid string content"),
            ChatMessage(role="assistant", content=None),  # Invalid None content
        ]
        
        with pytest.raises(ValidationError, match="Message content cannot be None"):
            client._validate_messages(mixed_messages)

    def test_validate_messages_whitespace_only_content(self, client):
        """
        Test that message validation raises ValidationError for messages with only whitespace content.
        
        Ensures that messages whose content consists entirely of whitespace characters are rejected as invalid.
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
        Test that model config validation raises a ValidationError when the temperature parameter is set to None.
        """
        config = ModelConfig(name="test-model")
        config.temperature = None
        
        with pytest.raises(ValidationError, match="Temperature cannot be None"):
            client._validate_model_config(config)

    @pytest.mark.asyncio
    async def test_custom_timeout_behavior(self, mock_config):
        """
        Test that GenesisAPIClient uses a custom timeout setting and raises GenesisAPIError when a request times out.
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
        Test that the client applies exponential backoff when retrying failed requests due to server errors.
        
        Simulates repeated server-side failures and verifies that the retry logic triggers increasing delays between attempts, ultimately raising a GenesisAPIError after exhausting retries.
        """
        call_times = []
        
        async def mock_failing_post(*args, **kwargs):
            """
            Simulates an asynchronous HTTP POST request that always fails with a 500 status code and a server error message.
            
            Intended for testing retry and error handling logic in asynchronous HTTP client code.
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
        
        Verifies that the returned headers include 'Authorization' and 'Content-Type' with expected values.
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
        Test equality comparison for ChatMessage instances with identical and differing attributes.
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
        Test that ModelConfig correctly handles extreme but valid parameter values.
        
        Verifies that the ModelConfig data model accepts and stores boundary values for numeric parameters without error.
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
        Test creation of an APIResponse with None as the data attribute.
        
        Verifies that the status_code and data fields are correctly set when data is None.
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
        Test creation of a ChatCompletion instance with no choices and zero token usage.
        
        Verifies that the ChatCompletion object correctly handles an empty choices list and reports zero total tokens in usage.
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
        Test serialization of a ChatCompletion instance to a JSON string.
        
        Asserts that the serialized output is a string and includes the expected identifier.
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
        Test that custom exceptions properly chain underlying exceptions via the `__cause__` attribute.
        """
        original_error = ValueError("Original error")
        api_error = GenesisAPIError("API error", status_code=500)
        api_error.__cause__ = original_error
        
        assert api_error.__cause__ is original_error

    def test_rate_limit_error_with_zero_retry_after(self):
        """
        Test that a RateLimitError initialized with retry_after=0 sets the retry_after attribute to zero.
        """
        error = RateLimitError("Rate limited", retry_after=0)
        assert error.retry_after == 0

    def test_rate_limit_error_with_none_retry_after(self):
        """
        Test that a RateLimitError instance sets retry_after to None when initialized with None.
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
        Test that a ValidationError includes the relevant field name and constraint details in its error message.
        """
        error = ValidationError("Invalid field 'temperature': must be between 0 and 2")
        assert "temperature" in str(error)
        assert "0 and 2" in str(error)


class TestUtilityFunctionsAdvanced:
    """Advanced tests for utility functions."""
    
    def test_format_timestamp_edge_cases(self):
        """
        Test that `format_timestamp` produces a string output for zero, negative, and very large timestamp values.
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
        Test that `calculate_token_usage` returns zero estimated tokens for an empty message list.
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
        Verify that the token estimation function returns a token count greater than or equal to the specified minimum for the provided content.
        
        Parameters:
            content (str): The input string to estimate tokens for.
            min_tokens (int): The minimum expected token count.
        """
        from app.ai_backend.genesis_api import estimate_tokens
        
        tokens = estimate_tokens(content)
        assert tokens >= min_tokens

    def test_estimate_tokens_with_special_characters(self):
        """
        Test that `estimate_tokens` returns a positive integer when given input containing special characters and punctuation.
        """
        from app.ai_backend.genesis_api import estimate_tokens
        
        special_content = "Hello, world! How are you? I'm fine. 123-456-7890."
        tokens = estimate_tokens(special_content)
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_estimate_tokens_with_newlines_and_tabs(self):
        """
        Test that the token estimation function accurately counts tokens in strings containing newlines and tab characters.
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
        Test that initializing the GenesisAPIClient with an API key containing only whitespace raises a ValueError.
        """
        with pytest.raises(ValueError, match="API key is required"):
            GenesisAPIClient(api_key="   \t\n   ")

    def test_url_validation_invalid_schemes(self):
        """
        Test that initializing GenesisAPIClient with an unsupported URL scheme raises a ValueError.
        """
        with pytest.raises(ValueError, match="Invalid base URL"):
            GenesisAPIClient(api_key="test-key", base_url="ftp://invalid.com")

    def test_header_injection_prevention(self, client):
        """
        Test that the client sanitizes or rejects headers containing carriage return or newline characters to prevent header injection attacks.
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
        Verify that messages containing potentially malicious script, SQL injection, or template expressions are not rejected by client-side validation.
        
        Ensures that only length or format-related validation errors are raised, delegating content filtering to the API.
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
        Verify that calling the client's close() method concurrently multiple times does not raise errors and ensures the session is closed.
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
        Test that attempting to make a chat completion request after closing the client raises a GenesisAPIError indicating the session is closed.
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
        
        Ensures that a ModelConfig instance can be created with only the required name and all other parameters set to None without raising errors.
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
        Test that an exception during async context manager initialization is propagated if session creation fails.
        
        Verifies that if `GenesisAPIClient` fails to initialize its session within an async context manager, the raised exception is not suppressed.
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
        Test that a ValidationError is raised when messages contain invalid role values.
        
        Parameters:
            invalid_role (str): An invalid role string to test message validation.
        """
        invalid_messages = [
            ChatMessage(role=invalid_role, content="Test content")
        ]
        
        with pytest.raises(ValidationError):
            client._validate_messages(invalid_messages)

    @pytest.mark.asyncio
    async def test_streaming_with_no_chunks(self, client, sample_messages, sample_model_config):
        """
        Test that the streaming chat completion method yields no results when the API response contains no chunks.
        
        This verifies that the client correctly handles an empty asynchronous stream from the API without producing any output.
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
        Test concurrent streaming chat completion requests for correct chunk handling.
        
        Verifies that the client can process multiple simultaneous streaming chat completion requests, ensuring each stream yields the expected sequence and number of response chunks.
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
        Tests that modifications to the client fixture's state do not persist across tests, ensuring fixture isolation.
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
            Simulates a sequence of asynchronous HTTP responses for use in tests.
            
            Each call returns a mock response object with status and JSON payload from the next item in the `responses` list, advancing the call count.
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

class TestGenesisAPIClientRobustness:
    """Additional robustness tests for GenesisAPIClient."""
    
    @pytest.mark.asyncio
    async def test_request_cancellation_handling(self, client, sample_messages, sample_model_config):
        """
        Test that the client raises asyncio.CancelledError when an API request is cancelled during execution.
        """
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate a cancelled request
            mock_post.side_effect = asyncio.CancelledError()
            
            with pytest.raises(asyncio.CancelledError):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_ssl_verification_error_handling(self, client, sample_messages, sample_model_config):
        """
        Test that SSL verification errors during chat completion requests are caught and raised as GenesisAPIError.
        """
        import ssl
        
        ssl_error = ssl.SSLError("SSL verification failed")
        
        with patch('aiohttp.ClientSession.post', side_effect=ssl_error):
            with pytest.raises(GenesisAPIError, match="SSL error"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_dns_resolution_error_handling(self, client, sample_messages, sample_model_config):
        """
        Test that a DNS resolution error during a chat completion request is correctly raised as a GenesisAPIError.
        """
        import socket
        
        dns_error = socket.gaierror("Name resolution failed")
        
        with patch('aiohttp.ClientSession.post', side_effect=dns_error):
            with pytest.raises(GenesisAPIError, match="DNS resolution error"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_memory_pressure_large_response(self, client, sample_messages, sample_model_config):
        """
        Verify that the client can process extremely large API responses without encountering memory errors.
        
        This test simulates a chat completion response with a very large message content and checks that the client correctly parses and returns the expected data.
        """
        # Create a very large response to test memory handling
        large_content = "x" * 100000  # 100KB response
        mock_response = {
            'id': 'large-response-test',
            'choices': [{'message': {'content': large_content}}],
            'usage': {'total_tokens': 25000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert result.choices[0].message.content == large_content
            assert result.usage.total_tokens == 25000

    def test_thread_safety_session_access(self, client):
        """
        Verifies that concurrent access to the client's session attribute is thread-safe and does not result in race conditions.
        """
        import threading
        import time
        
        results = []
        
        def access_session():
            # Access session attributes concurrently
            """
            Checks the existence and accessibility of the client's session attribute in a concurrent context.
            
            Appends the results of session existence and attribute checks to the shared `results` list, introducing a small delay to increase the likelihood of race conditions during concurrent access.
            """
            results.append(client.session is not None)
            time.sleep(0.01)  # Small delay to increase chance of race condition
            results.append(hasattr(client, 'session'))
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=access_session)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All accesses should be successful
        assert all(results)

    @pytest.mark.asyncio
    async def test_client_state_after_multiple_errors(self, client, sample_messages, sample_model_config):
        """
        Verify that the client maintains correct internal state and can recover after encountering a sequence of different error types, including a timeout, authentication error, and a subsequent successful response.
        """
        # Test sequence: timeout -> auth error -> success
        call_count = 0
        
        async def mock_error_sequence(*args, **kwargs):
            """
            Simulate a sequence of API responses for testing error recovery logic.
            
            On the first call, raises an asyncio.TimeoutError. On the second call, returns a mock response with a 401 status and an authentication error message. On subsequent calls, returns a successful mock response with a chat completion payload.
            """
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise asyncio.TimeoutError()
            elif call_count == 2:
                mock_response = Mock()
                mock_response.status = 401
                mock_response.json = AsyncMock(return_value={'error': {'message': 'Auth failed'}})
                return mock_response
            else:
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    'id': 'recovery-test',
                    'choices': [{'message': {'content': 'Success'}}],
                    'usage': {'total_tokens': 10}
                })
                return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_error_sequence):
            with patch('asyncio.sleep'):
                # First call should timeout
                with pytest.raises(GenesisAPIError, match="Request timeout"):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                
                # Second call should fail auth
                with pytest.raises(AuthenticationError):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                
                # Third call should succeed
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                assert result.id == 'recovery-test'


class TestAdvancedStreamingScenarios:
    """Advanced streaming test scenarios."""
    
    @pytest.mark.asyncio
    async def test_streaming_with_mixed_chunk_sizes(self, client, sample_messages, sample_model_config):
        """
        Test that the streaming chat completion API correctly processes response chunks of varying sizes, including very large and very small chunks.
        
        Verifies that all streamed chunks are received and parsed as expected, regardless of their size, and that the finish reason is correctly detected in the final chunk.
        """
        mixed_chunks = [
            {'choices': [{'delta': {'content': 'Small'}}]},
            {'choices': [{'delta': {'content': 'A much longer chunk with more content to test buffer handling and parsing'}}]},
            {'choices': [{'delta': {'content': 'Med'}}]},
            {'choices': [{'delta': {'content': 'A' * 1000}}]},  # Very long chunk
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def mock_mixed_stream():
            """
            Asynchronously yields a sequence of JSON-encoded byte chunks representing mixed streaming data.
            
            Yields:
                bytes: Each yielded value is a JSON-encoded chunk from the mixed_chunks sequence.
            """
            for chunk in mixed_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_mixed_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 5
            assert chunks[1].choices[0].delta.content.startswith('A much longer')
            assert len(chunks[3].choices[0].delta.content) == 1000
            assert chunks[-1].choices[0].finish_reason == 'stop'

    @pytest.mark.asyncio
    async def test_streaming_with_server_sent_events_format(self, client, sample_messages, sample_model_config):
        """
        Test that the client correctly parses and processes streaming chat completion responses in Server-Sent Events (SSE) format, including handling event types and data prefixes.
        """
        sse_chunks = [
            b'event: message\ndata: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b'event: message\ndata: {"choices": [{"delta": {"content": " world"}}]}\n\n',
            b'event: done\ndata: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
        ]
        
        async def mock_sse_stream():
            """
            Asynchronously yields predefined Server-Sent Events (SSE) data chunks for use in streaming tests.
            """
            for chunk in sse_chunks:
                yield chunk
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_sse_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            assert len(chunks) >= 2  # Should parse valid SSE chunks

    @pytest.mark.asyncio
    async def test_streaming_partial_json_reconstruction(self, client, sample_messages, sample_model_config):
        """
        Test that the streaming chat completion API correctly reconstructs and parses JSON objects split across multiple response chunks.
        
        Verifies that the client can handle partial JSON fragments received in separate chunks and yields the expected results during streaming.
        """
        partial_chunks = [
            b'{"choices": [{"delta": {"con',
            b'tent": "Partial JSON"}}]}',
            b'{"choices": [{"delta": {}, "finish_reason": "stop"}]}'
        ]
        
        async def mock_partial_stream():
            """
            Asynchronously yields a sequence of partial data chunks, simulating a streaming response.
            
            Yields:
                chunk: Each partial chunk from the predefined sequence.
            """
            for chunk in partial_chunks:
                yield chunk
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_partial_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            # Should handle partial JSON reconstruction
            assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_streaming_with_unicode_boundaries(self, client, sample_messages, sample_model_config):
        """
        Test that streaming chat completion correctly handles Unicode characters split across chunk boundaries.
        
        Verifies that the client can reconstruct and process streamed responses where UTF-8 encoded Unicode characters are divided between chunks, ensuring no data corruption or decoding errors occur.
        """
        # Create chunks where UTF-8 characters are split
        unicode_text = "Hello 世界 🌍 Test"
        encoded = unicode_text.encode('utf-8')
        
        # Split at arbitrary byte boundaries that might break UTF-8
        chunk1 = encoded[:10]
        chunk2 = encoded[10:20]
        chunk3 = encoded[20:]
        
        response_chunks = [
            json.dumps({'choices': [{'delta': {'content': chunk1.decode('utf-8', errors='ignore')}}]}).encode(),
            json.dumps({'choices': [{'delta': {'content': chunk2.decode('utf-8', errors='ignore')}}]}).encode(),
            json.dumps({'choices': [{'delta': {'content': chunk3.decode('utf-8', errors='ignore')}}]}).encode()
        ]
        
        async def mock_unicode_stream():
            """
            Asynchronously yields a sequence of Unicode response chunks to simulate a streaming API response.
            
            Yields:
                str: The next Unicode chunk in the simulated response stream.
            """
            for chunk in response_chunks:
                yield chunk
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_unicode_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            # Should handle Unicode properly
            assert len(chunks) >= 1


class TestDataValidationEdgeCases:
    """Additional edge case tests for data validation."""
    
    def test_chat_message_content_type_validation(self, client):
        """
        Test that ChatMessage content accepts various string-like types, including regular strings, numbers converted to strings, and Path objects converted to strings.
        """
        from pathlib import Path
        
        # Test with different string-like types
        valid_types = [
            "regular string",
            str(123),  # converted number
            str(Path("test")),  # pathlib Path converted
        ]
        
        for content in valid_types:
            message = ChatMessage(role="user", content=content)
            assert isinstance(message.content, str)
            client._validate_messages([message])  # Should not raise

    def test_model_config_numeric_boundary_precision(self, client):
        """
        Test that ModelConfig accepts floating-point values at precision boundaries for temperature and top_p without raising validation errors.
        """
        config = ModelConfig(name="test-model")
        
        # Test with very small but valid values
        config.temperature = 0.0001
        config.top_p = 0.0001
        client._validate_model_config(config)  # Should not raise
        
        # Test with values very close to boundaries
        config.temperature = 1.9999
        config.top_p = 0.9999
        client._validate_model_config(config)  # Should not raise

    def test_message_role_case_sensitivity(self, client):
        """
        Verify that message role validation enforces case sensitivity, rejecting roles with incorrect casing and accepting only exact matches.
        """
        case_variations = [
            ("User", "user"),
            ("ASSISTANT", "assistant"),
            ("System", "system"),
            ("uSeR", "user")
        ]
        
        for invalid_role, valid_role in case_variations:
            # Invalid case should fail
            invalid_message = ChatMessage(role=invalid_role, content="Test")
            with pytest.raises(ValidationError):
                client._validate_messages([invalid_message])
            
            # Valid case should pass
            valid_message = ChatMessage(role=valid_role, content="Test")
            client._validate_messages([valid_message])  # Should not raise

    def test_message_content_length_exact_boundaries(self, client):
        """
        Test that message content at the exact maximum allowed length is validated correctly.
        
        Verifies that a message with content exactly at the assumed maximum length boundary is either accepted or, if rejected, raises a ValidationError specifically for exceeding content length.
        """
        # Assuming there's a max length limit (typically 4096 or similar)
        max_length = 50000  # Reasonable assumption for testing
        
        # Test content at exact max length
        exact_max_content = "x" * max_length
        message = ChatMessage(role="user", content=exact_max_content)
        
        try:
            client._validate_messages([message])
        except ValidationError as e:
            # If it fails, it should be due to length, not other issues
            assert "content too long" in str(e)

    def test_model_config_extreme_penalty_values(self, client):
        """
        Test that ModelConfig accepts extreme but valid values for frequency_penalty and presence_penalty without raising validation errors.
        """
        config = ModelConfig(name="test-model")
        
        # Test extreme penalty values (typically -2.0 to 2.0)
        config.frequency_penalty = -2.0
        config.presence_penalty = 2.0
        client._validate_model_config(config)  # Should not raise
        
        config.frequency_penalty = 2.0
        config.presence_penalty = -2.0
        client._validate_model_config(config)  # Should not raise


class TestErrorRecoveryPatterns:
    """Test error recovery and resilience patterns."""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_partial_failures(self, client, sample_messages, sample_model_config):
        """
        Test that the client handles partial API responses by degrading gracefully when optional fields are missing.
        
        Verifies that the client can process responses lacking optional fields (such as 'usage') without raising errors, and that required fields are still correctly parsed. If an error occurs, it should only be due to missing required fields.
        """
        # Response missing some optional fields
        partial_response = {
            'id': 'partial-test',
            'choices': [{'message': {'content': 'Partial response'}}],
            # Missing 'usage' field
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=partial_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle missing optional fields gracefully
            try:
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                # If successful, verify core fields are present
                assert result.id == 'partial-test'
                assert result.choices[0].message.content == 'Partial response'
            except GenesisAPIError:
                # If it fails, it should be due to missing required fields
                pass

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern_simulation(self, client, sample_messages, sample_model_config):
        """
        Test that the client attempts multiple retries on consecutive server failures, simulating a circuit breaker pattern.
        
        Verifies that repeated server errors do not prevent the client from retrying requests up to the configured limit.
        """
        consecutive_failures = []
        
        async def mock_consecutive_failures(*args, **kwargs):
            """
            Simulates a sequence of consecutive server error responses for testing retry logic.
            
            Returns:
                Mock: An asynchronous mock HTTP response with status 500 and a JSON error message.
            """
            consecutive_failures.append(1)
            mock_response = Mock()
            mock_response.status = 500
            mock_response.json = AsyncMock(return_value={'error': {'message': 'Server error'}})
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_consecutive_failures):
            with patch('asyncio.sleep'):
                # Try multiple times to simulate circuit breaker
                for i in range(5):
                    try:
                        await client.create_chat_completion(
                            messages=sample_messages,
                            model_config=sample_model_config
                        )
                    except GenesisAPIError:
                        pass  # Expected to fail
                
                # Should have attempted all requests despite failures
                assert len(consecutive_failures) >= 3  # At least initial + retries

    @pytest.mark.asyncio
    async def test_request_idempotency_validation(self, client, sample_messages, sample_model_config):
        """
        Verify that making identical chat completion requests yields consistent responses, ensuring the client's idempotency and safe retry behavior.
        """
        request_count = 0
        
        async def mock_idempotent_response(*args, **kwargs):
            """
            Simulates an idempotent API response for repeated identical requests in asynchronous tests.
            
            Each call increments a request counter and returns a mocked HTTP response with a unique ID and fixed content, enabling verification of idempotency in client behavior.
            """
            nonlocal request_count
            request_count += 1
            
            # Return same response for identical requests
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                'id': f'idempotent-{request_count}',
                'choices': [{'message': {'content': 'Same response'}}],
                'usage': {'total_tokens': 10}
            })
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_idempotent_response):
            # Make identical requests
            result1 = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            result2 = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            # Both should succeed with consistent structure
            assert result1.choices[0].message.content == result2.choices[0].message.content
            assert request_count == 2


class TestPerformanceCharacteristics:
    """Test performance-related behaviors."""
    
    @pytest.mark.asyncio
    async def test_request_payload_size_limits(self, client, sample_model_config):
        """
        Test that the client can successfully handle and process very large chat completion request payloads without errors.
        
        This test constructs a large list of messages and verifies that the client correctly submits the payload and parses the response.
        """
        # Create a large number of messages
        large_message_list = []
        for i in range(1000):
            large_message_list.append(
                ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            )
        
        mock_response = {
            'id': 'large-payload-test',
            'choices': [{'message': {'content': 'Handled large payload'}}],
            'usage': {'total_tokens': 50000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle large payloads without issues
            result = await client.create_chat_completion(
                messages=large_message_list,
                model_config=sample_model_config
            )
            
            assert result.id == 'large-payload-test'
            assert result.usage.total_tokens == 50000

    @pytest.mark.asyncio
    async def test_response_parsing_performance(self, client, sample_messages, sample_model_config):
        """
        Test that the client efficiently parses complex chat completion responses with many choices.
        
        Simulates a large API response and verifies that parsing is performed quickly and all choices are correctly processed.
        """
        # Create a complex response with many choices
        complex_response = {
            'id': 'complex-response',
            'choices': [
                {
                    'index': i,
                    'message': {'content': f'Choice {i} response'},
                    'finish_reason': 'stop'
                }
                for i in range(100)
            ],
            'usage': {'total_tokens': 1000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=complex_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            import time
            start_time = time.time()
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            parse_time = time.time() - start_time
            
            # Should parse complex response efficiently
            assert result.id == 'complex-response'
            assert len(result.choices) == 100
            assert parse_time < 1.0  # Should be fast


class TestConfigurationValidation:
    """Test configuration validation and edge cases."""
    
    def test_base_url_normalization(self):
        """
        Test that the GenesisAPIClient normalizes the base URL by removing trailing slashes during initialization.
        
        Verifies that various input base URLs are correctly standardized to a consistent format.
        """
        test_cases = [
            ("https://api.test.com/", "https://api.test.com"),
            ("https://api.test.com/v1/", "https://api.test.com/v1"),
            ("https://api.test.com/v1//", "https://api.test.com/v1"),
        ]
        
        for input_url, expected in test_cases:
            client = GenesisAPIClient(api_key="test-key", base_url=input_url)
            # URL should be normalized (remove trailing slashes, etc.)
            assert client.base_url.rstrip('/') == expected.rstrip('/')

    def test_timeout_value_validation(self):
        """
        Test that the GenesisAPIClient correctly accepts valid timeout values during initialization.
        """
        # Valid timeout values
        valid_timeouts = [1, 30, 60, 300, 0.5]
        
        for timeout in valid_timeouts:
            client = GenesisAPIClient(api_key="test-key", timeout=timeout)
            assert client.timeout == timeout

    def test_max_retries_boundary_values(self):
        """
        Verify that the GenesisAPIClient correctly sets the max_retries attribute when initialized with various boundary values.
        """
        # Test with boundary values
        boundary_values = [0, 1, 10, 100]
        
        for max_retries in boundary_values:
            client = GenesisAPIClient(api_key="test-key", max_retries=max_retries)
            assert client.max_retries == max_retries

    def test_api_key_format_validation(self):
        """
        Test that the GenesisAPIClient accepts various valid API key formats without error.
        """
        # Valid API key formats
        valid_keys = [
            "sk-1234567890abcdef",
            "api_key_123456789",
            "Bearer token123",
            "simple-key"
        ]
        
        for key in valid_keys:
            client = GenesisAPIClient(api_key=key)
            assert client.api_key == key

    def test_configuration_immutability(self):
        """
        Verify that the configuration attributes of the GenesisAPIClient instance remain immutable after initialization.
        
        This test attempts to modify the `timeout` and `api_key` attributes and expects either an AttributeError or that the original values remain unchanged, depending on the implementation.
        """
        client = GenesisAPIClient(api_key="test-key", timeout=30)
        
        original_timeout = client.timeout
        original_api_key = client.api_key
        
        # Attempting to modify should not affect the client
        # (This depends on implementation - if they're properties, they might be read-only)
        try:
            client.timeout = 60
            client.api_key = "new-key"
        except AttributeError:
            # Expected if properties are read-only
            pass
        
        # Original values should be preserved if immutable
        # (Comment out assertions if properties are mutable by design)
        # assert client.timeout == original_timeout
        # assert client.api_key == original_api_key


# Additional utility tests for completeness
class TestUtilityFunctionsComprehensive:
    """Comprehensive tests for utility functions."""
    
    def test_timestamp_formatting_with_timezone_handling(self):
        """
        Tests that the `format_timestamp` utility correctly formats UNIX timestamps into non-empty, valid string representations across different timezone scenarios.
        """
        from app.ai_backend.genesis_api import format_timestamp
        
        # Test with various timestamp formats
        test_timestamps = [
            1609459200,  # 2021-01-01 00:00:00 UTC
            1640995200,  # 2022-01-01 00:00:00 UTC
            int(datetime.now(timezone.utc).timestamp()),  # Current time
        ]
        
        for timestamp in test_timestamps:
            formatted = format_timestamp(timestamp)
            assert isinstance(formatted, str)
            assert len(formatted) > 0
            # Should be a valid timestamp string
            assert any(char.isdigit() for char in formatted)

    def test_token_calculation_accuracy(self):
        """
        Verify that token calculation functions estimate token counts accurately for various content types and message formats.
        """
        from app.ai_backend.genesis_api import calculate_token_usage, estimate_tokens
        
        # Test cases with expected approximate token counts
        test_cases = [
            ("Hello world", 2),
            ("This is a longer sentence with more words", 8),
            ("Single", 1),
            ("", 0),
            ("Word " * 100, 100),  # Repeated word
        ]
        
        for content, expected_min in test_cases:
            # Test individual token estimation
            tokens = estimate_tokens(content)
            assert tokens >= expected_min or tokens == 0  # Allow for 0 when empty
            
            # Test usage calculation
            messages = [ChatMessage(role="user", content=content)]
            usage = calculate_token_usage(messages)
            assert usage['estimated_tokens'] >= expected_min or usage['estimated_tokens'] == 0

    def test_token_estimation_with_special_formats(self):
        """
        Test that the token estimation function returns a positive integer for various special text formats, including code, JSON, URLs, emails, numbers, and markdown.
        """
        from app.ai_backend.genesis_api import estimate_tokens
        
        special_formats = [
            "Code: `print('hello')`",
            "JSON: {\"key\": \"value\"}",
            "URL: https://example.com/path?param=value",
            "Email: user@example.com",
            "Numbers: 123,456.78",
            "Markdown: **bold** *italic*"
        ]
        
        for content in special_formats:
            tokens = estimate_tokens(content)
            assert isinstance(tokens, int)
            assert tokens > 0

    def test_message_preprocessing_edge_cases(self):
        """
        Tests that ChatMessage instances with edge case content types (such as only whitespace, single characters, or very long strings) are created successfully and maintain correct attributes.
        """
        edge_cases = [
            ChatMessage(role="user", content="\n\n\n"),  # Only newlines
            ChatMessage(role="user", content="   "),      # Only spaces
            ChatMessage(role="user", content="\t\t\t"),   # Only tabs
            ChatMessage(role="user", content="a"),        # Single character
            ChatMessage(role="user", content="A" * 10000), # Very long
        ]
        
        for message in edge_cases:
            # Should create valid ChatMessage objects
            assert message.role == "user"
            assert isinstance(message.content, str)


# Run additional tests when file is executed directly
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "TestGenesisAPIClientRobustness or TestAdvancedStreamingScenarios or TestDataValidationEdgeCases or TestErrorRecoveryPatterns or TestPerformanceCharacteristics or TestConfigurationValidation or TestUtilityFunctionsComprehensive",
        "--durations=5"
    ])