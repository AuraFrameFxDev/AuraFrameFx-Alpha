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
        Provides a mock configuration dictionary for initializing the GenesisAPIClient in tests.
        
        Returns:
            dict: A dictionary containing mock API key, base URL, timeout, and max retries.
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
        Fixture that returns a GenesisAPIClient instance configured with the provided mock configuration.
        """
        return GenesisAPIClient(**mock_config)
    
    @pytest.fixture
    def sample_messages(self):
        """
        Provides a fixture that returns a list of sample chat messages representing a typical conversation.
        """
        return [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is the weather like today?"),
            ChatMessage(role="assistant", content="I don't have access to real-time weather data.")
        ]
    
    @pytest.fixture
    def sample_model_config(self):
        """
        Provides a sample ModelConfig instance with typical parameters for testing purposes.
        
        Returns:
            ModelConfig: A sample model configuration with preset values.
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
        Test that the GenesisAPIClient initializes correctly with a valid configuration.
        
        Asserts that all client attributes match the provided configuration values.
        """
        client = GenesisAPIClient(**mock_config)
        assert client.api_key == mock_config['api_key']
        assert client.base_url == mock_config['base_url']
        assert client.timeout == mock_config['timeout']
        assert client.max_retries == mock_config['max_retries']

    def test_client_initialization_with_minimal_config(self):
        """
        Test that the GenesisAPIClient initializes correctly with only the required API key, and that default values are set for optional parameters.
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
        Test that initializing the client with a non-positive timeout value raises a ValueError.
        """
        with pytest.raises(ValueError, match="Timeout must be positive"):
            GenesisAPIClient(api_key='test-key', timeout=-1)
        
        with pytest.raises(ValueError, match="Timeout must be positive"):
            GenesisAPIClient(api_key='test-key', timeout=0)

    def test_client_initialization_invalid_max_retries(self):
        """
        Test that initializing the client with a negative max_retries value raises a ValueError.
        """
        with pytest.raises(ValueError, match="Max retries must be non-negative"):
            GenesisAPIClient(api_key='test-key', max_retries=-1)

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, client, sample_messages, sample_model_config):
        """
        Test that a successful chat completion request returns a valid ChatCompletion object with expected fields.
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
        
        Asserts that the streamed chunks are received in order and contain the correct content and finish reason.
        """
        mock_chunks = [
            {'choices': [{'delta': {'content': 'The'}}]},
            {'choices': [{'delta': {'content': ' weather'}}]},
            {'choices': [{'delta': {'content': ' is nice'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def mock_stream():
            """
            Asynchronously yields encoded JSON chunks from the `mock_chunks` iterable, simulating a streaming response.
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
        Test that an authentication error during chat completion raises AuthenticationError with the correct message.
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
        Test that the client raises a RateLimitError with the correct retry_after value when the API responds with a 429 status during chat completion.
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
        Test that creating a chat completion with invalid message roles raises a ValidationError.
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
        Test that the chat completion method retries on server errors and succeeds after retries.
        
        Simulates server errors on the first two attempts and a successful response on the third, verifying that the retry mechanism is triggered and the final result is correct.
        """
        call_count = 0
        
        async def mock_post_with_failure(*args, **kwargs):
            """
            Simulates an asynchronous HTTP POST request that fails with a 500 status on the first two calls and succeeds with a 200 status on the third call.
            
            Returns:
                Mock: A mock response object with appropriate status and JSON payload based on the call count.
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
        Test that chat completion raises a GenesisAPIError after exceeding the maximum number of retries due to repeated server errors.
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
        Test that a connection error during chat completion raises a GenesisAPIError with the appropriate message.
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
        Test that message validation raises ValidationError when a message has an invalid role.
        """
        invalid_messages = [
            ChatMessage(role="invalid", content="Test content")
        ]
        
        with pytest.raises(ValidationError, match="Invalid message role"):
            client._validate_messages(invalid_messages)

    def test_validate_messages_empty_content(self, client):
        """
        Test that message validation raises ValidationError when a message has empty content.
        """
        invalid_messages = [
            ChatMessage(role="user", content="")
        ]
        
        with pytest.raises(ValidationError, match="Message content cannot be empty"):
            client._validate_messages(invalid_messages)

    def test_validate_messages_content_too_long(self, client):
        """
        Test that message validation raises ValidationError when message content exceeds the maximum allowed length.
        """
        long_content = "x" * 100000  # Assuming max length is less than this
        invalid_messages = [
            ChatMessage(role="user", content=long_content)
        ]
        
        with pytest.raises(ValidationError, match="Message content too long"):
            client._validate_messages(invalid_messages)

    def test_validate_model_config_invalid_temperature(self, client, sample_model_config):
        """
        Test that model configuration validation raises a ValidationError for temperatures outside the valid range [0, 2].
        """
        sample_model_config.temperature = -0.5  # Invalid negative temperature
        
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(sample_model_config)
        
        sample_model_config.temperature = 2.5  # Invalid high temperature
        
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(sample_model_config)

    def test_validate_model_config_invalid_max_tokens(self, client, sample_model_config):
        """
        Test that model config validation raises ValidationError for non-positive max_tokens values.
        """
        sample_model_config.max_tokens = 0  # Invalid zero tokens
        
        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            client._validate_model_config(sample_model_config)
        
        sample_model_config.max_tokens = -100  # Invalid negative tokens
        
        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            client._validate_model_config(sample_model_config)

    def test_validate_model_config_invalid_top_p(self, client, sample_model_config):
        """
        Test that model config validation raises ValidationError for top_p values outside the range [0, 1].
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
        Test that the client successfully retrieves a model by ID and parses the response correctly.
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
        Test that retrieving a non-existent model raises a GenesisAPIError with the appropriate error message.
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
        Test that the client's _build_headers method returns headers containing Authorization, Content-Type, and User-Agent fields with correct values.
        """
        headers = client._build_headers()
        
        assert 'Authorization' in headers
        assert headers['Authorization'] == f'Bearer {client.api_key}'
        assert headers['Content-Type'] == 'application/json'
        assert 'User-Agent' in headers

    def test_build_headers_with_custom_headers(self, client):
        """
        Test that custom headers are correctly merged with default headers when building request headers.
        """
        custom_headers = {'X-Custom-Header': 'custom-value'}
        headers = client._build_headers(custom_headers)
        
        assert headers['X-Custom-Header'] == 'custom-value'
        assert 'Authorization' in headers
        assert headers['Content-Type'] == 'application/json'

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_config):
        """
        Test that the GenesisAPIClient correctly manages its session when used as an async context manager.
        
        Ensures the session is open within the context and properly closed after exiting.
        """
        async with GenesisAPIClient(**mock_config) as client:
            assert client.session is not None
        
        # Session should be closed after exiting context
        assert client.session.closed

    @pytest.mark.asyncio
    async def test_close_client_explicitly(self, client):
        """
        Test that explicitly closing the client properly closes its session.
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
        Test that the client raises the correct exception type for various HTTP status codes during chat completion.
        
        Parameters:
            status_code (int): The HTTP status code to simulate in the response.
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
        Test that a ChatMessage instance is correctly created with the specified role and content, and that the optional name attribute defaults to None.
        """
        message = ChatMessage(role="user", content="Hello, world!")
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.name is None

    def test_chat_message_with_name(self):
        """
        Test that a ChatMessage instance correctly sets the optional name field.
        """
        message = ChatMessage(role="user", content="Hello", name="John")
        assert message.name == "John"

    def test_model_config_creation(self):
        """Test ModelConfig creation with valid data."""
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
        Test that ModelConfig initializes with default values for optional parameters when only the name is provided.
        """
        config = ModelConfig(name="test-model")
        assert config.name == "test-model"
        assert config.max_tokens is not None
        assert config.temperature is not None
        assert config.top_p is not None

    def test_api_response_creation(self):
        """
        Test that an APIResponse object is correctly created with the provided status code, data, and headers.
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
        Test that a ChatCompletion object is correctly created with valid data and its attributes are set as expected.
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
        Test that an AuthenticationError is correctly instantiated and inherits from GenesisAPIError.
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
        """Test timestamp formatting utility."""
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
        Test that the token estimation function returns the expected token count for given content.
        
        Parameters:
            content (str): The input text whose tokens are to be estimated.
            expected_tokens (int): The expected number of tokens for the input content.
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
        Test the end-to-end chat completion workflow, including client creation, sending a chat message, and validating the response from the API.
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
        Test the client's ability to recover from a rate limit error during a chat completion workflow.
        
        Simulates a rate limit error on the first API call and verifies that the client raises a `RateLimitError`. On a subsequent call, simulates a successful response and asserts that the chat completion result is returned as expected.
        """
        config = {'api_key': 'test-key'}
        
        call_count = 0
        
        async def mock_post_with_recovery(*args, **kwargs):
            """
            Simulates an HTTP POST request that first returns a rate limit error, then a successful response on subsequent calls.
            
            Returns:
                Mock: A mock response object with appropriate status, headers, and JSON payload based on the call count.
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
        Test that the GenesisAPIClient can handle multiple concurrent chat completion requests and returns correct results for each.
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
        Test that the GenesisAPIClient correctly processes chat completions with very large message content.
        
        Simulates a successful API response for a large input message and verifies that the returned ChatCompletion object contains the expected ID and token usage.
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

# Additional comprehensive test coverage
class TestAdvancedGenesisAPIClient:
    """Extended test suite for advanced scenarios and edge cases."""
    
    @pytest.fixture
    def client_with_custom_session(self, mock_config):
        """Fixture that creates a client with a custom aiohttp session."""
        import aiohttp
        custom_session = aiohttp.ClientSession()
        client = GenesisAPIClient(**mock_config)
        client.session = custom_session
        return client
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_very_long_conversation(self, client):
        """Test chat completion with an extremely long conversation history."""
        long_messages = []
        for i in range(100):  # Create a very long conversation
            long_messages.append(ChatMessage(role="user", content=f"Message {i}: " + "x" * 100))
            long_messages.append(ChatMessage(role="assistant", content=f"Response {i}: " + "y" * 100))
        
        mock_response = {
            'id': 'long-conversation-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'Long conversation processed'},
                'finish_reason': 'stop'
            }],
            'usage': {'prompt_tokens': 20000, 'completion_tokens': 10, 'total_tokens': 20010}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=long_messages,
                model_config=ModelConfig(name="genesis-gpt-4", max_tokens=4000)
            )
            
            assert result.id == 'long-conversation-test'
            assert result.usage.total_tokens == 20010

    @pytest.mark.asyncio
    async def test_chat_completion_with_special_characters(self, client):
        """Test chat completion with special Unicode characters and emojis."""
        special_messages = [
            ChatMessage(role="user", content="Hello 🌍! How are you? 你好世界 🚀"),
            ChatMessage(role="system", content="Special chars: àáâãäåæçèéêë ñøü ™©® 💖"),
            ChatMessage(role="user", content="Math: ∑∞≠≤≥±÷×√∆π∅∈∉⊂⊃∪∩")
        ]
        
        mock_response = {
            'id': 'special-chars-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'Unicode response: ✅ 成功 🎉'},
                'finish_reason': 'stop'
            }],
            'usage': {'prompt_tokens': 50, 'completion_tokens': 15, 'total_tokens': 65}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=special_messages,
                model_config=ModelConfig(name="genesis-gpt-4")
            )
            
            assert result.choices[0].message.content == 'Unicode response: ✅ 成功 🎉'

    @pytest.mark.asyncio
    async def test_chat_completion_with_malformed_json_response(self, client, sample_messages, sample_model_config):
        """Test handling of malformed JSON responses from the API."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                side_effect=json.JSONDecodeError("Invalid JSON", "", 0)
            )
            
            with pytest.raises(GenesisAPIError, match="Invalid JSON response"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_chat_completion_with_partial_response(self, client, sample_messages, sample_model_config):
        """Test handling of incomplete API responses missing required fields."""
        incomplete_response = {
            'id': 'incomplete-test'
            # Missing required fields like 'choices', 'usage', etc.
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=incomplete_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            with pytest.raises((KeyError, GenesisAPIError)):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_streaming_with_connection_drop(self, client, sample_messages, sample_model_config):
        """Test streaming chat completion when connection drops mid-stream."""
        async def mock_stream_with_error():
            yield json.dumps({'choices': [{'delta': {'content': 'Start'}}]}).encode()
            raise ConnectionError("Connection dropped")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_stream_with_error()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            with pytest.raises(GenesisAPIError, match="Connection error"):
                chunks = []
                async for chunk in client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                ):
                    chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_streaming_with_malformed_chunk(self, client, sample_messages, sample_model_config):
        """Test streaming when receiving malformed JSON chunks."""
        async def mock_stream_with_bad_json():
            yield json.dumps({'choices': [{'delta': {'content': 'Good'}}]}).encode()
            yield b'invalid json chunk'
            yield json.dumps({'choices': [{'delta': {'content': 'End'}}]}).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_stream_with_bad_json()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            try:
                async for chunk in client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                ):
                    chunks.append(chunk)
            except GenesisAPIError:
                pass  # Expected to handle malformed JSON gracefully
            
            # Should have processed at least the first valid chunk
            assert len(chunks) >= 1

    def test_validate_messages_with_mixed_invalid_roles(self, client):
        """Test message validation with a mix of valid and invalid roles."""
        mixed_messages = [
            ChatMessage(role="user", content="Valid message"),
            ChatMessage(role="invalid_role", content="Invalid role"),
            ChatMessage(role="assistant", content="Another valid message"),
            ChatMessage(role="another_invalid", content="Another invalid")
        ]
        
        with pytest.raises(ValidationError, match="Invalid message role"):
            client._validate_messages(mixed_messages)

    def test_validate_messages_with_whitespace_only_content(self, client):
        """Test message validation with whitespace-only content."""
        whitespace_messages = [
            ChatMessage(role="user", content="   "),  # Only spaces
            ChatMessage(role="user", content="\t\n"),  # Only tabs and newlines
        ]
        
        with pytest.raises(ValidationError, match="Message content cannot be empty"):
            client._validate_messages(whitespace_messages)

    def test_validate_model_config_edge_values(self, client):
        """Test model config validation with boundary values."""
        config = ModelConfig(name="test-model")
        
        # Test exact boundary values
        config.temperature = 0.0  # Minimum valid
        client._validate_model_config(config)
        
        config.temperature = 2.0  # Maximum valid
        client._validate_model_config(config)
        
        config.top_p = 0.0  # Minimum valid
        client._validate_model_config(config)
        
        config.top_p = 1.0  # Maximum valid
        client._validate_model_config(config)
        
        config.max_tokens = 1  # Minimum valid
        client._validate_model_config(config)

    def test_validate_model_config_float_precision(self, client):
        """Test model config validation with high precision float values."""
        config = ModelConfig(name="test-model")
        
        # Test values very close to boundaries
        config.temperature = 0.000001  # Very small but valid
        client._validate_model_config(config)
        
        config.temperature = 1.999999  # Very close to maximum but valid
        client._validate_model_config(config)

    @pytest.mark.asyncio
    async def test_list_models_empty_response(self, client):
        """Test list_models when API returns empty model list."""
        mock_response = {
            'object': 'list',
            'data': []
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value.status = 200
            
            models = await client.list_models()
            assert len(models) == 0
            assert isinstance(models, list)

    @pytest.mark.asyncio
    async def test_list_models_with_pagination(self, client):
        """Test list_models with paginated response."""
        mock_response = {
            'object': 'list',
            'data': [
                {'id': f'model-{i}', 'object': 'model', 'created': 1677610602}
                for i in range(50)  # Large number of models
            ],
            'has_more': False
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value.status = 200
            
            models = await client.list_models()
            assert len(models) == 50

    def test_build_headers_preserves_authorization(self, client):
        """Test that custom headers don't override the Authorization header."""
        malicious_headers = {
            'Authorization': 'Bearer fake-key',
            'X-Custom': 'value'
        }
        
        headers = client._build_headers(malicious_headers)
        
        # Should preserve the client's API key, not the custom one
        assert headers['Authorization'] == f'Bearer {client.api_key}'
        assert headers['X-Custom'] == 'value'

    def test_build_headers_case_sensitivity(self, client):
        """Test header building with case-sensitive header names."""
        custom_headers = {
            'content-type': 'text/plain',  # lowercase
            'AUTHORIZATION': 'Bearer fake',  # uppercase
            'X-Custom-Header': 'value'
        }
        
        headers = client._build_headers(custom_headers)
        
        # Should have both the default Content-Type and the custom one
        assert 'Content-Type' in headers
        assert 'content-type' in headers
        assert headers['Content-Type'] == 'application/json'  # Default
        assert headers['content-type'] == 'text/plain'  # Custom

    @pytest.mark.asyncio
    async def test_session_reuse_across_requests(self, client):
        """Test that the same session is reused across multiple requests."""
        mock_response = {
            'id': 'test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Test'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            session_before = client.session
            
            await client.create_chat_completion(
                messages=[ChatMessage(role="user", content="Test 1")],
                model_config=ModelConfig(name="test-model")
            )
            
            session_after_first = client.session
            
            await client.create_chat_completion(
                messages=[ChatMessage(role="user", content="Test 2")],
                model_config=ModelConfig(name="test-model")
            )
            
            session_after_second = client.session
            
            # Session should be the same across requests
            assert session_before is session_after_first
            assert session_after_first is session_after_second

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self, client, sample_messages, sample_model_config):
        """Test that retry delays follow exponential backoff pattern."""
        delays = []
        
        async def mock_sleep(delay):
            delays.append(delay)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 500
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'error': {'message': 'Server error'}}
            )
            
            with patch('asyncio.sleep', side_effect=mock_sleep):
                with pytest.raises(GenesisAPIError):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                
                # Verify exponential backoff pattern
                assert len(delays) == client.max_retries
                for i in range(1, len(delays)):
                    assert delays[i] > delays[i-1]  # Each delay should be longer


class TestAdvancedDataModels:
    """Extended tests for data model classes with edge cases."""
    
    def test_chat_message_with_none_values(self):
        """Test ChatMessage creation with None values where allowed."""
        message = ChatMessage(role="user", content="Test", name=None)
        assert message.name is None
        
        # Test that None content should be rejected if validation exists
        with pytest.raises((ValueError, TypeError)):
            ChatMessage(role="user", content=None)

    def test_chat_message_serialization(self):
        """Test ChatMessage can be serialized to dict for API requests."""
        message = ChatMessage(role="user", content="Hello", name="John")
        
        # Assuming there's a to_dict method or similar
        if hasattr(message, 'to_dict'):
            data = message.to_dict()
            assert data['role'] == 'user'
            assert data['content'] == 'Hello'
            assert data['name'] == 'John'
        elif hasattr(message, '__dict__'):
            # Fallback to checking attributes
            assert hasattr(message, 'role')
            assert hasattr(message, 'content')
            assert hasattr(message, 'name')

    def test_model_config_with_extreme_values(self):
        """Test ModelConfig with extreme but valid values."""
        config = ModelConfig(
            name="extreme-test",
            max_tokens=1000000,  # Very large
            temperature=0.0,     # Minimum
            top_p=1.0,          # Maximum
            frequency_penalty=2.0,  # Maximum (if supported)
            presence_penalty=-2.0   # Minimum (if supported)
        )
        
        assert config.max_tokens == 1000000
        assert config.temperature == 0.0
        assert config.top_p == 1.0

    def test_api_response_with_empty_data(self):
        """Test APIResponse creation with empty data."""
        response = APIResponse(
            status_code=204,  # No Content
            data={},
            headers={}
        )
        
        assert response.status_code == 204
        assert response.data == {}
        assert response.headers == {}

    def test_chat_completion_with_minimal_data(self):
        """Test ChatCompletion creation with minimal required data."""
        completion = ChatCompletion(
            id="minimal-test",
            object="chat.completion",
            created=0,
            model="test-model",
            choices=[],
            usage={}
        )
        
        assert completion.id == "minimal-test"
        assert len(completion.choices) == 0
        assert completion.usage == {}


class TestAdvancedExceptionHandling:
    """Extended tests for exception handling and error conditions."""
    
    def test_genesis_api_error_with_none_status_code(self):
        """Test GenesisAPIError when status_code is None."""
        error = GenesisAPIError("Error without status code", status_code=None)
        assert str(error) == "Error without status code"
        assert error.status_code is None

    def test_rate_limit_error_with_invalid_retry_after(self):
        """Test RateLimitError with invalid retry_after values."""
        # Test with negative retry_after
        error = RateLimitError("Rate limited", retry_after=-1)
        assert error.retry_after == -1  # Should store as-is
        
        # Test with very large retry_after
        error = RateLimitError("Rate limited", retry_after=999999)
        assert error.retry_after == 999999

    def test_exception_inheritance_chain(self):
        """Test that all custom exceptions inherit correctly."""
        auth_error = AuthenticationError("Auth failed")
        rate_error = RateLimitError("Rate limited")
        validation_error = ValidationError("Validation failed")
        
        # All should inherit from GenesisAPIError
        assert isinstance(auth_error, GenesisAPIError)
        assert isinstance(rate_error, GenesisAPIError)
        assert isinstance(validation_error, GenesisAPIError)
        
        # And ultimately from Exception
        assert isinstance(auth_error, Exception)
        assert isinstance(rate_error, Exception)
        assert isinstance(validation_error, Exception)


class TestAdvancedUtilityFunctions:
    """Extended tests for utility functions with edge cases."""
    
    def test_format_timestamp_edge_cases(self):
        """Test timestamp formatting with edge case values."""
        from app.ai_backend.genesis_api import format_timestamp
        
        # Test with zero timestamp
        formatted = format_timestamp(0)
        assert isinstance(formatted, str)
        
        # Test with very large timestamp
        large_timestamp = 9999999999
        formatted = format_timestamp(large_timestamp)
        assert isinstance(formatted, str)
        
        # Test with negative timestamp (if supported)
        try:
            formatted = format_timestamp(-1)
            assert isinstance(formatted, str)
        except (ValueError, OSError):
            pass  # May not be supported on all systems

    def test_calculate_token_usage_empty_messages(self):
        """Test token usage calculation with empty message list."""
        from app.ai_backend.genesis_api import calculate_token_usage
        
        usage = calculate_token_usage([])
        assert isinstance(usage, dict)
        assert usage.get('estimated_tokens', 0) == 0

    def test_calculate_token_usage_large_messages(self):
        """Test token usage calculation with very large messages."""
        from app.ai_backend.genesis_api import calculate_token_usage
        
        large_messages = [
            ChatMessage(role="user", content="x" * 10000),
            ChatMessage(role="assistant", content="y" * 5000)
        ]
        
        usage = calculate_token_usage(large_messages)
        assert isinstance(usage, dict)
        assert 'estimated_tokens' in usage
        assert usage['estimated_tokens'] > 0

    @pytest.mark.parametrize("content,min_tokens,max_tokens", [
        ("", 0, 1),
        ("x" * 1000, 100, 1500),  # Range for very long text
        ("Hello world! 123 @#$", 2, 10),  # Mixed content
        ("🚀💖🌍", 1, 10),  # Emojis
    ])
    def test_estimate_tokens_ranges(self, content, min_tokens, max_tokens):
        """Test token estimation returns reasonable ranges for various content types."""
        from app.ai_backend.genesis_api import estimate_tokens
        
        tokens = estimate_tokens(content)
        assert min_tokens <= tokens <= max_tokens
        assert isinstance(tokens, int)

    def test_estimate_tokens_consistency(self):
        """Test that token estimation is consistent for the same input."""
        from app.ai_backend.genesis_api import estimate_tokens
        
        content = "This is a test message for consistency checking."
        
        # Call multiple times and ensure consistent results
        results = [estimate_tokens(content) for _ in range(5)]
        assert all(r == results[0] for r in results)


class TestAdvancedIntegration:
    """Extended integration tests for complex workflows."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_conversation_simulation(self):
        """Test a complete multi-turn conversation simulation."""
        config = {'api_key': 'test-key'}
        
        conversation_responses = [
            {
                'id': 'conv-1',
                'choices': [{'message': {'role': 'assistant', 'content': 'Hello! How can I help?'}}],
                'usage': {'total_tokens': 15}
            },
            {
                'id': 'conv-2', 
                'choices': [{'message': {'role': 'assistant', 'content': 'I can help with that!'}}],
                'usage': {'total_tokens': 25}
            },
            {
                'id': 'conv-3',
                'choices': [{'message': {'role': 'assistant', 'content': 'Anything else?'}}],
                'usage': {'total_tokens': 35}
            }
        ]
        
        call_count = 0
        
        async def mock_post_conversation(*args, **kwargs):
            nonlocal call_count
            response = Mock()
            response.status = 200
            response.json = AsyncMock(return_value=conversation_responses[call_count])
            call_count += 1
            return response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_post_conversation):
            async with GenesisAPIClient(**config) as client:
                conversation = []
                
                # Turn 1
                conversation.append(ChatMessage(role="user", content="Hello"))
                result1 = await client.create_chat_completion(
                    messages=conversation.copy(),
                    model_config=ModelConfig(name="test-model")
                )
                conversation.append(ChatMessage(
                    role="assistant", 
                    content=result1.choices[0].message.content
                ))
                
                # Turn 2
                conversation.append(ChatMessage(role="user", content="I need help"))
                result2 = await client.create_chat_completion(
                    messages=conversation.copy(),
                    model_config=ModelConfig(name="test-model")
                )
                conversation.append(ChatMessage(
                    role="assistant",
                    content=result2.choices[0].message.content
                ))
                
                # Turn 3
                conversation.append(ChatMessage(role="user", content="Thanks"))
                result3 = await client.create_chat_completion(
                    messages=conversation.copy(),
                    model_config=ModelConfig(name="test-model")
                )
                
                assert len(conversation) == 5  # 3 user + 2 assistant messages
                assert result1.id == 'conv-1'
                assert result2.id == 'conv-2'
                assert result3.id == 'conv-3'

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_streaming_conversation_workflow(self):
        """Test a streaming conversation with multiple chunks per response."""
        config = {'api_key': 'test-key'}
        
        # Simulate streaming chunks for a complete response
        mock_chunks = [
            {'choices': [{'delta': {'role': 'assistant'}}]},
            {'choices': [{'delta': {'content': 'I'}}]},
            {'choices': [{'delta': {'content': ' can'}}]},
            {'choices': [{'delta': {'content': ' help'}}]},
            {'choices': [{'delta': {'content': ' you'}}]},
            {'choices': [{'delta': {'content': ' with'}}]},
            {'choices': [{'delta': {'content': ' that!'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def mock_stream():
            for chunk in mock_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            async with GenesisAPIClient(**config) as client:
                full_content = ""
                chunk_count = 0
                
                async for chunk in client.create_chat_completion_stream(
                    messages=[ChatMessage(role="user", content="Help me")],
                    model_config=ModelConfig(name="test-model")
                ):
                    chunk_count += 1
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        if chunk.choices[0].delta.content:
                            full_content += chunk.choices[0].delta.content
                
                assert chunk_count == len(mock_chunks)
                assert full_content == "I can help you with that!"


class TestAdvancedPerformance:
    """Extended performance and stress tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage_with_large_responses(self):
        """Test memory efficiency with very large API responses."""
        config = {'api_key': 'test-key'}
        
        # Create a very large mock response
        large_content = "x" * 100000  # 100KB of content
        mock_response = {
            'id': 'large-response-test',
            'choices': [{'message': {'content': large_content}}],
            'usage': {'total_tokens': 25000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            async with GenesisAPIClient(**config) as client:
                result = await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content="Generate large text")],
                    model_config=ModelConfig(name="test-model", max_tokens=30000)
                )
                
                assert len(result.choices[0].message.content) == 100000
                assert result.usage.total_tokens == 25000

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_rapid_sequential_requests(self):
        """Test rapid sequential API requests without concurrency."""
        config = {'api_key': 'test-key'}
        
        mock_response = {
            'id': 'sequential-test',
            'choices': [{'message': {'content': 'Sequential response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            async with GenesisAPIClient(**config) as client:
                start_time = datetime.now()
                
                results = []
                for i in range(20):  # 20 rapid sequential requests
                    result = await client.create_chat_completion(
                        messages=[ChatMessage(role="user", content=f"Request {i}")],
                        model_config=ModelConfig(name="test-model")
                    )
                    results.append(result)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                assert len(results) == 20
                assert all(r.id == 'sequential-test' for r in results)
                # Verify it completed in reasonable time (adjust threshold as needed)
                assert duration < 10  # Should complete in under 10 seconds

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_streaming_performance_with_many_chunks(self):
        """Test streaming performance with a large number of small chunks."""
        config = {'api_key': 'test-key'}
        
        # Generate many small chunks
        mock_chunks = []
        for i in range(1000):  # 1000 small chunks
            mock_chunks.append({'choices': [{'delta': {'content': f'{i}'}}]})
        mock_chunks.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
        
        async def mock_stream():
            for chunk in mock_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            async with GenesisAPIClient(**config) as client:
                start_time = datetime.now()
                
                chunk_count = 0
                async for chunk in client.create_chat_completion_stream(
                    messages=[ChatMessage(role="user", content="Generate many chunks")],
                    model_config=ModelConfig(name="test-model")
                ):
                    chunk_count += 1
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                assert chunk_count == 1001  # 1000 content chunks + 1 finish chunk
                # Verify reasonable processing time
                assert duration < 5  # Should process 1000 chunks in under 5 seconds


# Test configuration and framework validation
class TestFrameworkConfiguration:
    """Tests to validate the testing framework and configuration."""
    
    def test_pytest_framework_available(self):
        """Verify that pytest is the testing framework in use."""
        import pytest
        assert pytest.__version__ is not None
        # This test confirms we're using pytest as the testing framework
    
    def test_asyncio_support_available(self):
        """Verify that asyncio testing support is properly configured."""
        import asyncio
        
        # Test that we can create and run async functions
        async def dummy_async():
            await asyncio.sleep(0.001)
            return True
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(dummy_async())
            assert result is True
        finally:
            loop.close()
    
    def test_mock_libraries_available(self):
        """Verify that all required mocking libraries are available."""
        from unittest.mock import Mock, patch, AsyncMock, MagicMock
        
        # Test that we can create different types of mocks
        mock = Mock()
        async_mock = AsyncMock()
        magic_mock = MagicMock()
        
        assert mock is not None
        assert async_mock is not None
        assert magic_mock is not None
        
        # Test basic patching functionality
        with patch('builtins.print') as mock_print:
            print("test")
            mock_print.assert_called_once_with("test")


if __name__ == "__main__":
    # Extended test runner configuration
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "--maxfail=5",  # Stop after 5 failures
        "--disable-warnings",  # Reduce noise in output
        "-x"  # Stop on first failure for debugging
    ])