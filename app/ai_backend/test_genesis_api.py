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
            dict: A dictionary containing mock values for API key, base URL, timeout, and max retries.
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
        Provides a fixture that returns a list of sample chat messages representing a typical conversation flow.
        """
        return [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is the weather like today?"),
            ChatMessage(role="assistant", content="I don't have access to real-time weather data.")
        ]
    
    @pytest.fixture
    def sample_model_config(self):
        """
        Provides a sample ModelConfig instance for use in tests.
        
        Returns:
            ModelConfig: A model configuration with predefined parameters.
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
        
        Asserts that the client's attributes match the provided configuration values.
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
        Test that initializing GenesisAPIClient without an API key raises a ValueError.
        """
        with pytest.raises(ValueError, match="API key is required"):
            GenesisAPIClient()

    def test_client_initialization_invalid_timeout(self):
        """
        Test that initializing GenesisAPIClient with non-positive timeout values raises a ValueError.
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
        Test that a successful chat completion request returns a valid ChatCompletion object with expected attributes.
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
        Tests that the chat completion API client correctly handles streaming responses by yielding each chunk as it arrives.
        
        Asserts that the streamed chunks are received in order and that the content and finish reason fields are parsed as expected.
        """
        mock_chunks = [
            {'choices': [{'delta': {'content': 'The'}}]},
            {'choices': [{'delta': {'content': ' weather'}}]},
            {'choices': [{'delta': {'content': ' is nice'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def mock_stream():
            """
            Asynchronously yields encoded JSON chunks from the mock_chunks list, simulating a streaming API response.
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
        Test that the client raises a RateLimitError with the correct retry_after value when the API responds with a 429 status code during chat completion.
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
        Tests that creating a chat completion with invalid message roles raises a ValidationError with the expected error message.
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
        Test that the chat completion method retries on server errors and succeeds after transient failures.
        
        Simulates server errors on the first two attempts and a successful response on the third, verifying that the retry mechanism is triggered and the final result is correct.
        """
        call_count = 0
        
        async def mock_post_with_failure(*args, **kwargs):
            """
            Simulates an asynchronous HTTP POST request that fails with a 500 status on the first two calls and succeeds with a 200 status on the third and subsequent calls.
            
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
        Tests that a network timeout during chat completion raises a GenesisAPIError with a timeout message.
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
        Tests that a connection error during chat completion raises a GenesisAPIError with the appropriate message.
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
        Test that validating messages with empty content raises a ValidationError.
        """
        invalid_messages = [
            ChatMessage(role="user", content="")
        ]
        
        with pytest.raises(ValidationError, match="Message content cannot be empty"):
            client._validate_messages(invalid_messages)

    def test_validate_messages_content_too_long(self, client):
        """
        Test that message validation raises a ValidationError when message content exceeds the maximum allowed length.
        """
        long_content = "x" * 100000  # Assuming max length is less than this
        invalid_messages = [
            ChatMessage(role="user", content=long_content)
        ]
        
        with pytest.raises(ValidationError, match="Message content too long"):
            client._validate_messages(invalid_messages)

    def test_validate_model_config_invalid_temperature(self, client, sample_model_config):
        """
        Test that model config validation raises ValidationError for temperatures outside the range 0 to 2.
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
        
        Asserts that the returned list contains the expected model IDs.
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
        Test that the client successfully retrieves a model by ID and parses its attributes correctly.
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
        Test that the client's _build_headers method returns headers with correct Authorization, Content-Type, and User-Agent fields.
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
        Test that the client raises the correct exception type for various HTTP status codes during chat completion requests.
        
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
        Test that a ChatMessage instance is correctly created with the specified role and content.
        """
        message = ChatMessage(role="user", content="Hello, world!")
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.name is None

    def test_chat_message_with_name(self):
        """
        Test that a ChatMessage instance correctly sets the name attribute when provided.
        """
        message = ChatMessage(role="user", content="Hello", name="John")
        assert message.name == "John"

    def test_model_config_creation(self):
        """
        Test that a ModelConfig instance is created correctly with specified name, max_tokens, and temperature values.
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
        Test that a ModelConfig instance is created with default values for max_tokens, temperature, and top_p when only the name is provided.
        """
        config = ModelConfig(name="test-model")
        assert config.name == "test-model"
        assert config.max_tokens is not None
        assert config.temperature is not None
        assert config.top_p is not None

    def test_api_response_creation(self):
        """
        Test that an APIResponse object is correctly created with the specified status code, data, and headers.
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
        Test that a ChatCompletion object is correctly created with valid attributes.
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
        Test that the AuthenticationError is created with the correct message and inherits from GenesisAPIError.
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
        Test that the `calculate_token_usage` utility correctly computes token usage from a list of chat messages.
        
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
        Test that the `estimate_tokens` function returns the expected token count for given content.
        
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
        Performs an end-to-end integration test of the chat completion workflow, including client creation, sending a message, and verifying the response from the mocked Genesis API.
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
        Test that the client correctly recovers from a rate limit error by retrying the chat completion request and succeeding on a subsequent attempt.
        
        Simulates a rate limit error on the first API call and a successful response on the second, verifying that the client raises `RateLimitError` initially and completes successfully after recovery.
        """
        config = {'api_key': 'test-key'}
        
        call_count = 0
        
        async def mock_post_with_recovery(*args, **kwargs):
            """
            Simulates an asynchronous HTTP POST request that returns a rate limit error on the first call and a successful response on subsequent calls.
            
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
        Test that the GenesisAPIClient can handle multiple concurrent chat completion requests and returns correct responses for each.
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
        Test that the GenesisAPIClient correctly processes chat completions with large message content.
        
        Simulates sending a large message and verifies that the response is handled as expected, including correct parsing of the completion ID and token usage.
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

class TestGenesisAPIClientAdditional:
    """Additional comprehensive tests for GenesisAPIClient."""
    
    @pytest.fixture
    def client_with_custom_session(self, mock_config):
        """Fixture that creates a client with a custom aiohttp session."""
        import aiohttp
        custom_session = aiohttp.ClientSession()
        return GenesisAPIClient(session=custom_session, **mock_config)
    
    def test_client_initialization_with_custom_session(self, mock_config):
        """Test that client can be initialized with a custom session."""
        import aiohttp
        custom_session = aiohttp.ClientSession()
        client = GenesisAPIClient(session=custom_session, **mock_config)
        assert client.session == custom_session
    
    def test_client_initialization_with_invalid_base_url(self):
        """Test that client raises error for invalid base URL format."""
        with pytest.raises(ValueError, match="Invalid base URL"):
            GenesisAPIClient(api_key='test-key', base_url='invalid-url')
    
    def test_client_initialization_with_empty_api_key(self):
        """Test that client raises error for empty API key."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            GenesisAPIClient(api_key='')
    
    def test_client_initialization_with_whitespace_api_key(self):
        """Test that client raises error for whitespace-only API key."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            GenesisAPIClient(api_key='   ')
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_empty_response(self, client, sample_messages, sample_model_config):
        """Test handling of empty response from API."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value={})
            mock_post.return_value.__aenter__.return_value.status = 200
            
            with pytest.raises(ValidationError, match="Invalid response format"):
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
        """Test handling of response missing choices array."""
        mock_response = {
            'id': 'chat-123',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            with pytest.raises(ValidationError, match="Missing choices in response"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_system_message_only(self, client, sample_model_config):
        """Test chat completion with only system message."""
        system_messages = [
            ChatMessage(role="system", content="You are a helpful assistant.")
        ]
        
        mock_response = {
            'id': 'chat-system-only',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Ready to help!'}}],
            'usage': {'total_tokens': 15}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=system_messages,
                model_config=sample_model_config
            )
            
            assert result.id == 'chat-system-only'
            assert result.choices[0].message.content == 'Ready to help!'
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_function_call(self, client, sample_model_config):
        """Test chat completion with function call message."""
        function_messages = [
            ChatMessage(role="user", content="What's the weather?"),
            ChatMessage(role="assistant", content="", function_call={"name": "get_weather", "arguments": "{}"}),
            ChatMessage(role="function", content='{"temperature": "72F"}', name="get_weather")
        ]
        
        mock_response = {
            'id': 'chat-function-call',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'It is 72°F today.'}}],
            'usage': {'total_tokens': 25}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=function_messages,
                model_config=sample_model_config
            )
            
            assert result.id == 'chat-function-call'
            assert 'weather' in result.choices[0].message.content.lower()
    
    @pytest.mark.asyncio
    async def test_chat_completion_streaming_connection_error(self, client, sample_messages, sample_model_config):
        """Test streaming chat completion with connection error."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                side_effect=ConnectionError("Connection lost")
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            with pytest.raises(GenesisAPIError, match="Connection error"):
                async for chunk in client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                ):
                    pass
    
    @pytest.mark.asyncio
    async def test_chat_completion_streaming_invalid_chunk(self, client, sample_messages, sample_model_config):
        """Test streaming chat completion with invalid chunk data."""
        async def mock_stream():
            yield b'invalid json chunk'
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            with pytest.raises(GenesisAPIError, match="Invalid chunk format"):
                async for chunk in client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                ):
                    pass
    
    def test_validate_messages_consecutive_same_role(self, client):
        """Test validation of consecutive messages with same role."""
        invalid_messages = [
            ChatMessage(role="user", content="First message"),
            ChatMessage(role="user", content="Second message")
        ]
        
        with pytest.raises(ValidationError, match="Consecutive messages with same role"):
            client._validate_messages(invalid_messages)
    
    def test_validate_messages_function_without_name(self, client):
        """Test validation of function message without name."""
        invalid_messages = [
            ChatMessage(role="function", content="Function result")
        ]
        
        with pytest.raises(ValidationError, match="Function message must have a name"):
            client._validate_messages(invalid_messages)
    
    def test_validate_model_config_invalid_penalties(self, client, sample_model_config):
        """Test validation of model config with invalid penalty values."""
        sample_model_config.frequency_penalty = -2.5
        
        with pytest.raises(ValidationError, match="Frequency penalty must be between -2 and 2"):
            client._validate_model_config(sample_model_config)
        
        sample_model_config.frequency_penalty = 0.0
        sample_model_config.presence_penalty = 3.0
        
        with pytest.raises(ValidationError, match="Presence penalty must be between -2 and 2"):
            client._validate_model_config(sample_model_config)
    
    def test_validate_model_config_invalid_model_name(self, client, sample_model_config):
        """Test validation of model config with invalid model name."""
        sample_model_config.name = ""
        
        with pytest.raises(ValidationError, match="Model name cannot be empty"):
            client._validate_model_config(sample_model_config)
    
    @pytest.mark.asyncio
    async def test_list_models_empty_response(self, client):
        """Test list models with empty response."""
        mock_response = {'object': 'list', 'data': []}
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value.status = 200
            
            models = await client.list_models()
            
            assert isinstance(models, list)
            assert len(models) == 0
    
    @pytest.mark.asyncio
    async def test_list_models_malformed_response(self, client):
        """Test list models with malformed response."""
        mock_response = {'invalid': 'response'}
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value.status = 200
            
            with pytest.raises(ValidationError, match="Invalid models list response"):
                await client.list_models()
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_calculation(self, client):
        """Test exponential backoff calculation for retries."""
        # Access private method for testing
        delay = client._calculate_backoff_delay(attempt=1)
        assert delay >= 1.0
        
        delay = client._calculate_backoff_delay(attempt=2)
        assert delay >= 2.0
        
        delay = client._calculate_backoff_delay(attempt=3)
        assert delay >= 4.0
    
    def test_build_headers_with_none_custom_headers(self, client):
        """Test building headers with None custom headers."""
        headers = client._build_headers(None)
        
        assert 'Authorization' in headers
        assert headers['Content-Type'] == 'application/json'
    
    def test_build_headers_override_default(self, client):
        """Test building headers with override of default headers."""
        custom_headers = {'Content-Type': 'text/plain'}
        headers = client._build_headers(custom_headers)
        
        assert headers['Content-Type'] == 'text/plain'
        assert 'Authorization' in headers


class TestDataModelsAdditional:
    """Additional comprehensive tests for data model classes."""
    
    def test_chat_message_with_function_call(self):
        """Test ChatMessage with function call data."""
        message = ChatMessage(
            role="assistant",
            content="",
            function_call={"name": "get_weather", "arguments": '{"location": "NYC"}'}
        )
        assert message.function_call["name"] == "get_weather"
        assert "location" in message.function_call["arguments"]
    
    def test_chat_message_equality(self):
        """Test ChatMessage equality comparison."""
        msg1 = ChatMessage(role="user", content="Hello")
        msg2 = ChatMessage(role="user", content="Hello")
        msg3 = ChatMessage(role="user", content="Hi")
        
        assert msg1 == msg2
        assert msg1 != msg3
    
    def test_chat_message_repr(self):
        """Test ChatMessage string representation."""
        message = ChatMessage(role="user", content="Hello")
        repr_str = repr(message)
        
        assert "user" in repr_str
        assert "Hello" in repr_str
    
    def test_model_config_with_all_parameters(self):
        """Test ModelConfig with all parameters set."""
        config = ModelConfig(
            name="genesis-gpt-4",
            max_tokens=2000,
            temperature=0.8,
            top_p=0.95,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stop=["END"],
            logit_bias={50256: -100}
        )
        
        assert config.name == "genesis-gpt-4"
        assert config.max_tokens == 2000
        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.2
        assert config.stop == ["END"]
        assert config.logit_bias[50256] == -100
    
    def test_model_config_copy(self):
        """Test ModelConfig copy functionality."""
        config = ModelConfig(name="test-model", temperature=0.7)
        config_copy = config.copy()
        
        assert config_copy.name == config.name
        assert config_copy.temperature == config.temperature
        assert config_copy is not config
    
    def test_api_response_with_error(self):
        """Test APIResponse with error information."""
        response = APIResponse(
            status_code=400,
            data={'error': {'message': 'Bad request'}},
            headers={'Content-Type': 'application/json'},
            error_message="Bad request"
        )
        
        assert response.status_code == 400
        assert response.error_message == "Bad request"
        assert response.is_error() is True
    
    def test_api_response_success(self):
        """Test APIResponse for successful response."""
        response = APIResponse(
            status_code=200,
            data={'result': 'success'},
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 200
        assert response.is_error() is False
        assert response.data['result'] == 'success'
    
    def test_chat_completion_with_multiple_choices(self):
        """Test ChatCompletion with multiple choices."""
        completion = ChatCompletion(
            id="chat-multi-choice",
            object="chat.completion",
            created=1677610602,
            model="genesis-gpt-4",
            choices=[
                {'message': {'role': 'assistant', 'content': 'Choice 1'}},
                {'message': {'role': 'assistant', 'content': 'Choice 2'}}
            ],
            usage={'total_tokens': 50}
        )
        
        assert len(completion.choices) == 2
        assert completion.choices[0].message.content == 'Choice 1'
        assert completion.choices[1].message.content == 'Choice 2'
    
    def test_chat_completion_streaming_chunk(self):
        """Test ChatCompletion streaming chunk data."""
        chunk = ChatCompletion(
            id="chat-stream-chunk",
            object="chat.completion.chunk",
            created=1677610602,
            model="genesis-gpt-4",
            choices=[{'delta': {'content': 'Hello'}}],
            usage=None
        )
        
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.content == 'Hello'
        assert chunk.usage is None


class TestExceptionClassesAdditional:
    """Additional comprehensive tests for custom exception classes."""
    
    def test_genesis_api_error_with_response(self):
        """Test GenesisAPIError with response data."""
        response_data = {'error': {'type': 'invalid_request', 'message': 'Test error'}}
        error = GenesisAPIError("Test error", status_code=400, response=response_data)
        
        assert error.status_code == 400
        assert error.response == response_data
        assert str(error) == "Test error"
    
    def test_genesis_api_error_without_status_code(self):
        """Test GenesisAPIError without status code."""
        error = GenesisAPIError("Test error")
        
        assert error.status_code is None
        assert str(error) == "Test error"
    
    def test_authentication_error_with_details(self):
        """Test AuthenticationError with additional details."""
        error = AuthenticationError("Invalid API key", details="Key format is incorrect")
        
        assert error.details == "Key format is incorrect"
        assert isinstance(error, GenesisAPIError)
    
    def test_rate_limit_error_without_retry_after(self):
        """Test RateLimitError without retry_after value."""
        error = RateLimitError("Rate limit exceeded")
        
        assert error.retry_after is None
        assert isinstance(error, GenesisAPIError)
    
    def test_rate_limit_error_with_string_retry_after(self):
        """Test RateLimitError with string retry_after value."""
        error = RateLimitError("Rate limit exceeded", retry_after="60")
        
        assert error.retry_after == "60"
    
    def test_validation_error_with_field_info(self):
        """Test ValidationError with field information."""
        error = ValidationError("Invalid input", field="temperature", value=3.0)
        
        assert error.field == "temperature"
        assert error.value == 3.0
        assert isinstance(error, GenesisAPIError)
    
    def test_exception_inheritance_chain(self):
        """Test that all custom exceptions inherit from GenesisAPIError."""
        auth_error = AuthenticationError("Auth failed")
        rate_error = RateLimitError("Rate limited")
        val_error = ValidationError("Validation failed")
        
        assert isinstance(auth_error, GenesisAPIError)
        assert isinstance(rate_error, GenesisAPIError)
        assert isinstance(val_error, GenesisAPIError)
        assert isinstance(auth_error, Exception)
        assert isinstance(rate_error, Exception)
        assert isinstance(val_error, Exception)


class TestUtilityFunctionsAdditional:
    """Additional comprehensive tests for utility functions."""
    
    def test_format_timestamp_with_timezone(self):
        """Test timestamp formatting with timezone conversion."""
        from app.ai_backend.genesis_api import format_timestamp
        
        timestamp = 1677610602
        formatted = format_timestamp(timestamp, timezone='UTC')
        
        assert isinstance(formatted, str)
        assert 'UTC' in formatted or 'Z' in formatted
    
    def test_format_timestamp_with_invalid_input(self):
        """Test timestamp formatting with invalid input."""
        from app.ai_backend.genesis_api import format_timestamp
        
        with pytest.raises(ValueError):
            format_timestamp("invalid")
    
    def test_calculate_token_usage_with_empty_messages(self):
        """Test token usage calculation with empty messages."""
        from app.ai_backend.genesis_api import calculate_token_usage
        
        usage = calculate_token_usage([])
        
        assert usage['estimated_tokens'] == 0
    
    def test_calculate_token_usage_with_function_messages(self):
        """Test token usage calculation with function messages."""
        from app.ai_backend.genesis_api import calculate_token_usage
        
        messages = [
            ChatMessage(role="user", content="Call weather function"),
            ChatMessage(role="assistant", content="", function_call={"name": "get_weather"}),
            ChatMessage(role="function", content='{"temp": 72}', name="get_weather")
        ]
        
        usage = calculate_token_usage(messages)
        
        assert usage['estimated_tokens'] > 0
        assert 'function_tokens' in usage
    
    @pytest.mark.parametrize("content,language,expected_multiplier", [
        ("Hello world", "en", 1.0),
        ("Hola mundo", "es", 1.2),
        ("こんにちは", "ja", 2.0),
        ("你好", "zh", 1.5),
    ])
    def test_estimate_tokens_with_languages(self, content, language, expected_multiplier):
        """Test token estimation with different languages."""
        from app.ai_backend.genesis_api import estimate_tokens
        
        base_tokens = estimate_tokens(content)
        language_tokens = estimate_tokens(content, language=language)
        
        assert language_tokens >= base_tokens * expected_multiplier
    
    def test_validate_api_key_format(self):
        """Test API key format validation utility."""
        from app.ai_backend.genesis_api import validate_api_key_format
        
        # Valid key formats
        assert validate_api_key_format("sk-1234567890abcdef") is True
        assert validate_api_key_format("genesis-key-abcd1234") is True
        
        # Invalid key formats
        assert validate_api_key_format("invalid-key") is False
        assert validate_api_key_format("") is False
        assert validate_api_key_format("   ") is False
    
    def test_sanitize_error_message(self):
        """Test error message sanitization utility."""
        from app.ai_backend.genesis_api import sanitize_error_message
        
        # Should remove sensitive information
        sensitive_msg = "Error: API key sk-1234567890abcdef is invalid"
        sanitized = sanitize_error_message(sensitive_msg)
        
        assert "sk-1234567890abcdef" not in sanitized
        assert "API key" in sanitized
        assert "invalid" in sanitized
    
    def test_parse_retry_after_header(self):
        """Test retry-after header parsing utility."""
        from app.ai_backend.genesis_api import parse_retry_after_header
        
        # Test numeric value
        assert parse_retry_after_header("60") == 60
        
        # Test HTTP date format
        date_str = "Wed, 21 Oct 2015 07:28:00 GMT"
        result = parse_retry_after_header(date_str)
        assert isinstance(result, int)
        assert result > 0
        
        # Test invalid format
        assert parse_retry_after_header("invalid") is None


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_extremely_long_conversation(self):
        """Test handling of extremely long conversation history."""
        config = {'api_key': 'test-key'}
        
        # Create a very long conversation
        long_messages = []
        for i in range(1000):
            long_messages.append(ChatMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i} with some content"
            ))
        
        mock_response = {
            'id': 'chat-long-conversation',
            'choices': [{'message': {'content': 'Response to long conversation'}}],
            'usage': {'total_tokens': 50000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            async with GenesisAPIClient(**config) as client:
                result = await client.create_chat_completion(
                    messages=long_messages,
                    model_config=ModelConfig(name="test-model", max_tokens=100)
                )
                
                assert result.id == 'chat-long-conversation'
                assert result.usage.total_tokens == 50000
    
    @pytest.mark.asyncio
    async def test_zero_max_tokens_handling(self):
        """Test handling of zero max_tokens in model config."""
        config = {'api_key': 'test-key'}
        
        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            ModelConfig(name="test-model", max_tokens=0)
    
    @pytest.mark.asyncio
    async def test_negative_temperature_handling(self):
        """Test handling of negative temperature values."""
        config = {'api_key': 'test-key'}
        
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            ModelConfig(name="test-model", temperature=-0.1)
    
    @pytest.mark.asyncio
    async def test_unicode_content_handling(self, client, sample_model_config):
        """Test handling of Unicode content in messages."""
        unicode_messages = [
            ChatMessage(role="user", content="Hello 世界! 🌍 Héllo Wörld! Здравствуй мир!")
        ]
        
        mock_response = {
            'id': 'chat-unicode',
            'choices': [{'message': {'content': 'Unicode response 🤖'}}],
            'usage': {'total_tokens': 25}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=unicode_messages,
                model_config=sample_model_config
            )
            
            assert result.choices[0].message.content == 'Unicode response 🤖'
    
    @pytest.mark.asyncio
    async def test_concurrent_session_management(self):
        """Test proper session management under concurrent access."""
        config = {'api_key': 'test-key'}
        
        async def create_and_use_client():
            async with GenesisAPIClient(**config) as client:
                # Simulate some API calls
                await asyncio.sleep(0.1)
                return client.session.closed
        
        # Run multiple concurrent clients
        tasks = [create_and_use_client() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All sessions should be properly closed
        assert all(results)
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_large_responses(self, client):
        """Test memory efficiency with large API responses."""
        large_content = "x" * 100000  # 100KB content
        
        mock_response = {
            'id': 'chat-large-response',
            'choices': [{'message': {'content': large_content}}],
            'usage': {'total_tokens': 25000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=[ChatMessage(role="user", content="Generate large response")],
                model_config=ModelConfig(name="test-model")
            )
            
            assert len(result.choices[0].message.content) == 100000
            assert result.usage.total_tokens == 25000


class TestSecurityAndSanitization:
    """Test security-related functionality and input sanitization."""
    
    def test_api_key_not_logged_in_errors(self, client):
        """Test that API key is not exposed in error messages."""
        try:
            client._build_headers()
            # Force an error that might expose the API key
            raise GenesisAPIError(f"Authentication failed with key: {client.api_key}")
        except GenesisAPIError as e:
            error_msg = str(e)
            # API key should be sanitized in error messages
            assert client.api_key not in error_msg or "***" in error_msg
    
    def test_sensitive_data_sanitization(self):
        """Test sanitization of sensitive data in log messages."""
        from app.ai_backend.genesis_api import sanitize_log_message
        
        sensitive_msg = "Request failed: {\"api_key\": \"sk-1234567890abcdef\", \"data\": \"test\"}"
        sanitized = sanitize_log_message(sensitive_msg)
        
        assert "sk-1234567890abcdef" not in sanitized
        assert "api_key" in sanitized
        assert "***" in sanitized
    
    @pytest.mark.asyncio
    async def test_malicious_content_filtering(self, client, sample_model_config):
        """Test filtering of potentially malicious content."""
        malicious_messages = [
            ChatMessage(role="user", content="<script>alert('xss')</script>"),
            ChatMessage(role="user", content="../../etc/passwd"),
            ChatMessage(role="user", content="'; DROP TABLE users; --")
        ]
        
        # Should validate and reject malicious content
        for msg in malicious_messages:
            with pytest.raises(ValidationError, match="Potentially malicious content"):
                client._validate_messages([msg])
    
    def test_input_length_limits(self, client):
        """Test enforcement of input length limits."""
        # Test extremely long single message
        very_long_content = "x" * 1000000  # 1MB content
        long_message = ChatMessage(role="user", content=very_long_content)
        
        with pytest.raises(ValidationError, match="Message content too long"):
            client._validate_messages([long_message])
    
    def test_rate_limiting_headers_parsing(self, client):
        """Test parsing of rate limiting headers."""
        headers = {
            'X-RateLimit-Limit': '1000',
            'X-RateLimit-Remaining': '999',
            'X-RateLimit-Reset': '1677610662'
        }
        
        rate_info = client._parse_rate_limit_headers(headers)
        
        assert rate_info['limit'] == 1000
        assert rate_info['remaining'] == 999
        assert rate_info['reset'] == 1677610662


if __name__ == "__main__":
    # Run tests with coverage reporting
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "--cov=app.ai_backend.genesis_api",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])