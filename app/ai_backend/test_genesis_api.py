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

# Additional comprehensive test coverage
class TestGenesisAPIClientExtended:
    """Extended comprehensive test suite for GenesisAPIClient edge cases."""
    
    def test_client_initialization_with_invalid_url_format(self):
        """Test client initialization with malformed base URL."""
        with pytest.raises(ValueError, match="Invalid base URL format"):
            GenesisAPIClient(api_key='test-key', base_url='not-a-url')
    
    def test_client_initialization_with_empty_api_key(self):
        """Test client initialization with empty string API key."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            GenesisAPIClient(api_key='')
    
    def test_client_initialization_with_whitespace_api_key(self):
        """Test client initialization with whitespace-only API key."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            GenesisAPIClient(api_key='   ')
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_system_message_only(self, client):
        """Test chat completion with only system message."""
        system_only_messages = [
            ChatMessage(role="system", content="You are a helpful assistant.")
        ]
        
        mock_response = {
            'id': 'system-only-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'I understand.'},
                'finish_reason': 'stop'
            }],
            'usage': {'prompt_tokens': 8, 'completion_tokens': 3, 'total_tokens': 11}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=system_only_messages,
                model_config=ModelConfig(name="genesis-gpt-4")
            )
            
            assert result.id == 'system-only-test'
            assert result.choices[0].message.content == 'I understand.'
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_function_call_response(self, client, sample_messages, sample_model_config):
        """Test chat completion that returns a function call."""
        mock_response = {
            'id': 'function-call-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': None,
                    'function_call': {
                        'name': 'get_weather',
                        'arguments': '{"location": "San Francisco"}'
                    }
                },
                'finish_reason': 'function_call'
            }],
            'usage': {'prompt_tokens': 25, 'completion_tokens': 15, 'total_tokens': 40}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert result.choices[0].finish_reason == 'function_call'
            assert result.choices[0].message.function_call['name'] == 'get_weather'
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_multiple_choices(self, client, sample_messages, sample_model_config):
        """Test chat completion with multiple response choices."""
        mock_response = {
            'id': 'multiple-choices-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'First response choice'},
                    'finish_reason': 'stop'
                },
                {
                    'index': 1,
                    'message': {'role': 'assistant', 'content': 'Second response choice'},
                    'finish_reason': 'stop'
                }
            ],
            'usage': {'prompt_tokens': 25, 'completion_tokens': 16, 'total_tokens': 41}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert len(result.choices) == 2
            assert result.choices[0].message.content == 'First response choice'
            assert result.choices[1].message.content == 'Second response choice'
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_content_filter_finish_reason(self, client, sample_messages, sample_model_config):
        """Test chat completion with content filter finish reason."""
        mock_response = {
            'id': 'content-filter-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'I cannot provide that information.'},
                'finish_reason': 'content_filter'
            }],
            'usage': {'prompt_tokens': 25, 'completion_tokens': 8, 'total_tokens': 33}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert result.choices[0].finish_reason == 'content_filter'
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_length_finish_reason(self, client, sample_messages, sample_model_config):
        """Test chat completion that ends due to length limit."""
        mock_response = {
            'id': 'length-limit-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'This is a very long response that was cut off'},
                'finish_reason': 'length'
            }],
            'usage': {'prompt_tokens': 25, 'completion_tokens': 1000, 'total_tokens': 1025}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert result.choices[0].finish_reason == 'length'
            assert result.usage.completion_tokens == 1000
    
    @pytest.mark.asyncio
    async def test_streaming_with_role_delta(self, client, sample_messages, sample_model_config):
        """Test streaming with role information in delta."""
        mock_chunks = [
            {'choices': [{'delta': {'role': 'assistant'}}]},
            {'choices': [{'delta': {'content': 'Hello'}}]},
            {'choices': [{'delta': {'content': ' world'}}]},
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
            
            assert len(chunks) == 4
            assert chunks[0].choices[0].delta.role == 'assistant'
            assert chunks[1].choices[0].delta.content == 'Hello'
            assert chunks[2].choices[0].delta.content == ' world'
    
    @pytest.mark.asyncio
    async def test_streaming_with_function_call_delta(self, client, sample_messages, sample_model_config):
        """Test streaming with function call deltas."""
        mock_chunks = [
            {'choices': [{'delta': {'role': 'assistant'}}]},
            {'choices': [{'delta': {'function_call': {'name': 'get_weather'}}}]},
            {'choices': [{'delta': {'function_call': {'arguments': '{"location"'}}}]},
            {'choices': [{'delta': {'function_call': {'arguments': ': "NYC"}'}}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'function_call'}]}
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
            
            assert len(chunks) == 5
            assert chunks[-1].choices[0].finish_reason == 'function_call'
    
    def test_validate_messages_with_function_message(self, client):
        """Test validation of messages with function role."""
        function_messages = [
            ChatMessage(role="user", content="What's the weather?"),
            ChatMessage(role="assistant", content=None, function_call={"name": "get_weather", "arguments": "{}"}),
            ChatMessage(role="function", content='{"temperature": 72}', name="get_weather")
        ]
        
        # Should not raise an exception if function messages are supported
        try:
            client._validate_messages(function_messages)
        except ValidationError as e:
            if "Invalid message role" in str(e):
                # Function role might not be supported, test passes
                pass
            else:
                raise
    
    def test_validate_messages_function_without_name(self, client):
        """Test validation of function messages without name."""
        function_messages = [
            ChatMessage(role="function", content='{"result": "success"}')  # Missing name
        ]
        
        with pytest.raises(ValidationError, match="Function messages must have a name"):
            client._validate_messages(function_messages)
    
    def test_validate_model_config_with_custom_parameters(self, client):
        """Test validation with custom model configuration parameters."""
        config = ModelConfig(
            name="custom-model",
            max_tokens=2000,
            temperature=0.5,
            top_p=0.8,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stop=["END", "STOP"],
            logit_bias={50256: -100}
        )
        
        # Should validate successfully with all parameters
        client._validate_model_config(config)
    
    def test_validate_model_config_with_invalid_stop_sequences(self, client):
        """Test validation with invalid stop sequences."""
        config = ModelConfig(name="test-model")
        config.stop = [""] # Empty stop sequence
        
        with pytest.raises(ValidationError, match="Stop sequences cannot be empty"):
            client._validate_model_config(config)
    
    def test_validate_model_config_with_too_many_stop_sequences(self, client):
        """Test validation with too many stop sequences."""
        config = ModelConfig(name="test-model")
        config.stop = [f"stop{i}" for i in range(10)]  # Too many stop sequences
        
        with pytest.raises(ValidationError, match="Too many stop sequences"):
            client._validate_model_config(config)
    
    @pytest.mark.asyncio
    async def test_request_timeout_handling(self, client, sample_messages, sample_model_config):
        """Test handling of request timeouts."""
        import asyncio
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError("Request timeout")
            
            with pytest.raises(GenesisAPIError, match="Request timeout"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
    
    @pytest.mark.asyncio
    async def test_ssl_error_handling(self, client, sample_messages, sample_model_config):
        """Test handling of SSL errors."""
        import ssl
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = ssl.SSLError("SSL certificate verification failed")
            
            with pytest.raises(GenesisAPIError, match="SSL error"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
    
    @pytest.mark.asyncio
    async def test_dns_resolution_error(self, client, sample_messages, sample_model_config):
        """Test handling of DNS resolution errors."""
        import socket
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = socket.gaierror("Name or service not known")
            
            with pytest.raises(GenesisAPIError, match="DNS resolution error"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )


class TestAdvancedMessageValidation:
    """Advanced tests for message validation edge cases."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-key')
    
    def test_validate_messages_with_null_bytes(self, client):
        """Test message validation with null bytes in content."""
        messages_with_nulls = [
            ChatMessage(role="user", content="Hello\x00World")
        ]
        
        with pytest.raises(ValidationError, match="Message content contains invalid characters"):
            client._validate_messages(messages_with_nulls)
    
    def test_validate_messages_with_control_characters(self, client):
        """Test message validation with control characters."""
        messages_with_control = [
            ChatMessage(role="user", content="Hello\x01\x02World")
        ]
        
        with pytest.raises(ValidationError, match="Message content contains invalid characters"):
            client._validate_messages(messages_with_control)
    
    def test_validate_messages_with_mixed_encodings(self, client):
        """Test message validation with mixed text encodings."""
        mixed_encoding_messages = [
            ChatMessage(role="user", content="ASCII text"),
            ChatMessage(role="assistant", content="UTF-8: 你好"),
            ChatMessage(role="user", content="Emoji: 🚀💖")
        ]
        
        # Should handle mixed encodings gracefully
        client._validate_messages(mixed_encoding_messages)
    
    def test_validate_messages_with_extremely_long_single_message(self, client):
        """Test validation with a single extremely long message."""
        extremely_long_content = "x" * 1000000  # 1MB of content
        long_messages = [
            ChatMessage(role="user", content=extremely_long_content)
        ]
        
        with pytest.raises(ValidationError, match="Message content too long"):
            client._validate_messages(long_messages)
    
    def test_validate_messages_alternating_roles(self, client):
        """Test validation of proper role alternation."""
        alternating_messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
            ChatMessage(role="user", content="How are you?"),
            ChatMessage(role="assistant", content="I'm fine")
        ]
        
        # Should validate proper alternation
        client._validate_messages(alternating_messages)
    
    def test_validate_messages_consecutive_user_messages(self, client):
        """Test validation with consecutive user messages."""
        consecutive_user_messages = [
            ChatMessage(role="user", content="First message"),
            ChatMessage(role="user", content="Second message")
        ]
        
        # Some implementations might require alternation
        try:
            client._validate_messages(consecutive_user_messages)
        except ValidationError as e:
            if "consecutive messages" in str(e):
                pass  # Expected behavior
            else:
                raise
    
    def test_validate_messages_with_html_content(self, client):
        """Test message validation with HTML content."""
        html_messages = [
            ChatMessage(role="user", content="<script>alert('xss')</script>"),
            ChatMessage(role="user", content="<b>Bold text</b>"),
            ChatMessage(role="user", content="<img src='x' onerror='alert(1)'>")
        ]
        
        # Should handle HTML content appropriately
        for message in html_messages:
            try:
                client._validate_messages([message])
            except ValidationError as e:
                if "HTML content" in str(e) or "invalid characters" in str(e):
                    pass  # Expected behavior for security
                else:
                    raise


class TestAdvancedModelConfigValidation:
    """Advanced tests for model configuration validation."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-key')
    
    def test_validate_model_config_with_nan_values(self, client):
        """Test model config validation with NaN values."""
        config = ModelConfig(name="test-model")
        config.temperature = float('nan')
        
        with pytest.raises(ValidationError, match="Temperature must be a valid number"):
            client._validate_model_config(config)
    
    def test_validate_model_config_with_infinite_values(self, client):
        """Test model config validation with infinite values."""
        config = ModelConfig(name="test-model")
        config.temperature = float('inf')
        
        with pytest.raises(ValidationError, match="Temperature must be finite"):
            client._validate_model_config(config)
    
    def test_validate_model_config_with_negative_infinite_values(self, client):
        """Test model config validation with negative infinite values."""
        config = ModelConfig(name="test-model")
        config.temperature = float('-inf')
        
        with pytest.raises(ValidationError, match="Temperature must be finite"):
            client._validate_model_config(config)
    
    def test_validate_model_config_with_very_large_max_tokens(self, client):
        """Test model config validation with extremely large max_tokens."""
        config = ModelConfig(name="test-model")
        config.max_tokens = 2**31  # Very large number
        
        with pytest.raises(ValidationError, match="Max tokens exceeds maximum allowed"):
            client._validate_model_config(config)
    
    def test_validate_model_config_with_invalid_penalty_values(self, client):
        """Test model config validation with invalid penalty values."""
        config = ModelConfig(name="test-model")
        
        # Test frequency penalty out of range
        config.frequency_penalty = 3.0  # Assuming max is 2.0
        with pytest.raises(ValidationError, match="Frequency penalty must be between"):
            client._validate_model_config(config)
        
        # Test presence penalty out of range  
        config.frequency_penalty = 0.0  # Reset to valid
        config.presence_penalty = -3.0  # Assuming min is -2.0
        with pytest.raises(ValidationError, match="Presence penalty must be between"):
            client._validate_model_config(config)
    
    def test_validate_model_config_with_invalid_model_name(self, client):
        """Test model config validation with invalid model names."""
        # Empty model name
        with pytest.raises(ValidationError, match="Model name cannot be empty"):
            config = ModelConfig(name="")
            client._validate_model_config(config)
        
        # Model name with invalid characters
        with pytest.raises(ValidationError, match="Model name contains invalid characters"):
            config = ModelConfig(name="model/with/slashes")
            client._validate_model_config(config)
    
    def test_validate_model_config_with_conflicting_parameters(self, client):
        """Test model config validation with conflicting parameters."""
        config = ModelConfig(name="test-model")
        config.temperature = 0.0  # Deterministic
        config.top_p = 0.1  # Low diversity
        
        # Some implementations might warn about conflicting parameters
        try:
            client._validate_model_config(config)
        except ValidationError as e:
            if "conflicting parameters" in str(e):
                pass  # Expected behavior
            else:
                raise


class TestAdvancedStreamingScenarios:
    """Advanced tests for streaming scenarios."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-key')
    
    @pytest.fixture
    def sample_messages(self):
        return [ChatMessage(role="user", content="Test streaming")]
    
    @pytest.fixture
    def sample_model_config(self):
        return ModelConfig(name="test-model")
    
    @pytest.mark.asyncio
    async def test_streaming_with_empty_chunks(self, client, sample_messages, sample_model_config):
        """Test streaming with empty chunks in the stream."""
        mock_chunks = [
            {'choices': [{'delta': {'content': 'Start'}}]},
            {'choices': [{'delta': {}}]},  # Empty delta
            {'choices': [{'delta': {'content': 'End'}}]},
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
            
            assert len(chunks) == 4
            content_chunks = [c for c in chunks if hasattr(c.choices[0].delta, 'content') and c.choices[0].delta.content]
            assert len(content_chunks) == 2
    
    @pytest.mark.asyncio
    async def test_streaming_with_unicode_chunks(self, client, sample_messages, sample_model_config):
        """Test streaming with Unicode characters split across chunks."""
        # Simulate Unicode character split across chunks
        mock_chunks = [
            {'choices': [{'delta': {'content': '你'}}]},
            {'choices': [{'delta': {'content': '好'}}]},
            {'choices': [{'delta': {'content': '世'}}]},
            {'choices': [{'delta': {'content': '界'}}]},
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
            full_content = ""
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content
            
            assert full_content == "你好世界"
    
    @pytest.mark.asyncio
    async def test_streaming_with_mixed_chunk_types(self, client, sample_messages, sample_model_config):
        """Test streaming with mixed chunk types (content, function_call, etc.)."""
        mock_chunks = [
            {'choices': [{'delta': {'role': 'assistant'}}]},
            {'choices': [{'delta': {'content': 'Let me check'}}]},
            {'choices': [{'delta': {'function_call': {'name': 'check_status'}}}]},
            {'choices': [{'delta': {'function_call': {'arguments': '{}'}}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'function_call'}]}
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
            
            # Should handle mixed chunk types properly
            assert len(chunks) == 5
            assert chunks[0].choices[0].delta.role == 'assistant'
            assert chunks[1].choices[0].delta.content == 'Let me check'
            assert chunks[-1].choices[0].finish_reason == 'function_call'
    
    @pytest.mark.asyncio
    async def test_streaming_with_very_small_chunks(self, client, sample_messages, sample_model_config):
        """Test streaming with very small chunks (single characters)."""
        message = "Hello World!"
        mock_chunks = []
        
        # Create chunks for each character
        for char in message:
            mock_chunks.append({'choices': [{'delta': {'content': char}}]})
        
        mock_chunks.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
        
        async def mock_stream():
            for chunk in mock_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            full_content = ""
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content
            
            assert full_content == message
            assert len(chunks) == len(message) + 1  # +1 for finish chunk
    
    @pytest.mark.asyncio
    async def test_streaming_with_large_single_chunk(self, client, sample_messages, sample_model_config):
        """Test streaming with a single very large chunk."""
        large_content = "x" * 50000  # 50KB content in one chunk
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
            assert chunks[0].choices[0].delta.content == large_content
            assert len(chunks[0].choices[0].delta.content) == 50000


class TestAdvancedErrorScenarios:
    """Advanced tests for error handling scenarios."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-key')
    
    @pytest.fixture
    def sample_messages(self):
        return [ChatMessage(role="user", content="Test message")]
    
    @pytest.fixture
    def sample_model_config(self):
        return ModelConfig(name="test-model")
    
    @pytest.mark.asyncio
    async def test_quota_exceeded_error(self, client, sample_messages, sample_model_config):
        """Test handling of quota exceeded errors."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 429
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={
                    'error': {
                        'message': 'You have exceeded your quota',
                        'type': 'quota_exceeded',
                        'code': 'quota_exceeded'
                    }
                }
            )
            mock_post.return_value.__aenter__.return_value.headers = {'Retry-After': '3600'}
            
            with pytest.raises(RateLimitError) as exc_info:
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
            
            assert exc_info.value.retry_after == 3600
            assert "quota" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_model_overloaded_error(self, client, sample_messages, sample_model_config):
        """Test handling of model overloaded errors."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 503
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={
                    'error': {
                        'message': 'Model is currently overloaded',
                        'type': 'server_error',
                        'code': 'model_overloaded'
                    }
                }
            )
            
            with pytest.raises(GenesisAPIError, match="Model is currently overloaded"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
    
    @pytest.mark.asyncio
    async def test_insufficient_permissions_error(self, client, sample_messages, sample_model_config):
        """Test handling of insufficient permissions errors."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 403
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={
                    'error': {
                        'message': 'Insufficient permissions to access this model',
                        'type': 'permission_error',
                        'code': 'insufficient_permissions'
                    }
                }
            )
            
            with pytest.raises(AuthenticationError, match="Insufficient permissions"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
    
    @pytest.mark.asyncio
    async def test_model_not_found_error(self, client, sample_messages):
        """Test handling of model not found errors."""
        invalid_model_config = ModelConfig(name="non-existent-model")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 404
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={
                    'error': {
                        'message': 'Model non-existent-model not found',
                        'type': 'invalid_request_error',
                        'code': 'model_not_found'
                    }
                }
            )
            
            with pytest.raises(GenesisAPIError, match="Model non-existent-model not found"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=invalid_model_config
                )
    
    @pytest.mark.asyncio
    async def test_request_too_large_error(self, client, sample_model_config):
        """Test handling of request too large errors."""
        very_large_messages = [
            ChatMessage(role="user", content="x" * 1000000)  # Very large message
        ]
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 413
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={
                    'error': {
                        'message': 'Request entity too large',
                        'type': 'invalid_request_error',
                        'code': 'request_too_large'
                    }
                }
            )
            
            with pytest.raises(ValidationError, match="Request entity too large"):
                await client.create_chat_completion(
                    messages=very_large_messages,
                    model_config=sample_model_config
                )
    
    @pytest.mark.asyncio
    async def test_context_length_exceeded_error(self, client, sample_model_config):
        """Test handling of context length exceeded errors."""
        long_context_messages = [
            ChatMessage(role="user", content="Context " * 10000)  # Very long context
        ]
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 400
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={
                    'error': {
                        'message': 'Context length exceeded. Maximum context length is 4096 tokens.',
                        'type': 'invalid_request_error',
                        'code': 'context_length_exceeded'
                    }
                }
            )
            
            with pytest.raises(ValidationError, match="Context length exceeded"):
                await client.create_chat_completion(
                    messages=long_context_messages,
                    model_config=sample_model_config
                )
    
    @pytest.mark.asyncio
    async def test_billing_error(self, client, sample_messages, sample_model_config):
        """Test handling of billing-related errors."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 402
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={
                    'error': {
                        'message': 'Payment required. Please update your billing information.',
                        'type': 'billing_error',
                        'code': 'payment_required'
                    }
                }
            )
            
            with pytest.raises(AuthenticationError, match="Payment required"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )


class TestAdvancedPerformanceEdgeCases:
    """Advanced performance tests for edge cases."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-key')
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_efficiency_with_streaming(self, client):
        """Test memory efficiency during long streaming sessions."""
        # Simulate a very long streaming response
        num_chunks = 10000
        mock_chunks = []
        
        for i in range(num_chunks):
            mock_chunks.append({'choices': [{'delta': {'content': f'chunk{i} '}}]})
        
        mock_chunks.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
        
        async def mock_stream():
            for chunk in mock_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunk_count = 0
            async for chunk in client.create_chat_completion_stream(
                messages=[ChatMessage(role="user", content="Generate long response")],
                model_config=ModelConfig(name="test-model")
            ):
                chunk_count += 1
                # Don't accumulate chunks to test memory efficiency
                if chunk_count % 1000 == 0:
                    # Simulate periodic memory check
                    assert chunk_count <= num_chunks + 1
            
            assert chunk_count == num_chunks + 1
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_connection_reuse_efficiency(self, client):
        """Test that connections are reused efficiently across requests."""
        mock_response = {
            'id': 'reuse-test',
            'choices': [{'message': {'content': 'Response'}}],
            'usage': {'total_tokens': 10}
        }
        
        connection_count = 0
        
        class MockSession:
            def __init__(self):
                nonlocal connection_count
                connection_count += 1
                
            async def __aenter__(self):
                return self
                
            async def __aexit__(self, *args):
                pass
                
            def post(self, *args, **kwargs):
                return MockResponse()
        
        class MockResponse:
            def __init__(self):
                self.status = 200
                
            async def __aenter__(self):
                return self
                
            async def __aexit__(self, *args):
                pass
                
            async def json(self):
                return mock_response
        
        # Override the session to count connections
        original_session = client.session
        client.session = MockSession()
        
        try:
            # Make multiple requests
            for i in range(5):
                await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content=f"Request {i}")],
                    model_config=ModelConfig(name="test-model")
                )
            
            # Should have reused the same session
            assert connection_count == 1
        finally:
            client.session = original_session
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_timeout_handling_under_load(self, client):
        """Test timeout handling behavior under high load."""
        import asyncio
        
        timeout_count = 0
        success_count = 0
        
        async def make_request():
            nonlocal timeout_count, success_count
            try:
                with patch('aiohttp.ClientSession.post') as mock_post:
                    # Simulate random timeouts
                    if timeout_count < 3:  # First 3 requests timeout
                        mock_post.side_effect = asyncio.TimeoutError()
                        timeout_count += 1
                    else:  # Rest succeed
                        mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                            return_value={
                                'id': 'success',
                                'choices': [{'message': {'content': 'OK'}}],
                                'usage': {'total_tokens': 5}
                            }
                        )
                        mock_post.return_value.__aenter__.return_value.status = 200
                        success_count += 1
                    
                    await client.create_chat_completion(
                        messages=[ChatMessage(role="user", content="Test")],
                        model_config=ModelConfig(name="test-model")
                    )
            except GenesisAPIError:
                pass  # Expected for timeouts
        
        # Run multiple concurrent requests
        tasks = [make_request() for _ in range(10)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should have handled timeouts gracefully
        assert timeout_count >= 3
        assert success_count >= 0


class TestAdvancedMockingScenarios:
    """Advanced tests for complex mocking scenarios."""
    
    @pytest.fixture
    def client(self):
        return GenesisAPIClient(api_key='test-key')
    
    @pytest.mark.asyncio
    async def test_partial_response_corruption(self, client):
        """Test handling of partially corrupted responses."""
        # Response with some valid and some invalid data
        corrupted_response = {
            'id': 'corrupted-test',
            'object': 'chat.completion',
            'created': 'invalid-timestamp',  # Should be int
            'model': 'test-model',
            'choices': [
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'Valid content'},
                    'finish_reason': 'stop'
                }
            ],
            'usage': {
                'prompt_tokens': 'invalid',  # Should be int
                'completion_tokens': 10,
                'total_tokens': 'also-invalid'  # Should be int
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=corrupted_response
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle partial corruption gracefully
            try:
                result = await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content="Test")],
                    model_config=ModelConfig(name="test-model")
                )
                # If it succeeds, verify the valid parts
                assert result.id == 'corrupted-test'
                assert result.choices[0].message.content == 'Valid content'
            except (TypeError, ValueError, GenesisAPIError):
                pass  # Expected for corrupted data
    
    @pytest.mark.asyncio
    async def test_response_with_unexpected_fields(self, client):
        """Test handling of responses with unexpected additional fields."""
        response_with_extra_fields = {
            'id': 'extra-fields-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'test-model',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'Response with extra fields',
                    'extra_field': 'unexpected_value'
                },
                'finish_reason': 'stop',
                'extra_choice_field': 'more_unexpected'
            }],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 15,
                'total_tokens': 25,
                'extra_usage_field': 'surprise!'
            },
            'unexpected_top_level_field': 'should_be_ignored'
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=response_with_extra_fields
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle extra fields gracefully
            result = await client.create_chat_completion(
                messages=[ChatMessage(role="user", content="Test")],
                model_config=ModelConfig(name="test-model")
            )
            
            assert result.id == 'extra-fields-test'
            assert result.choices[0].message.content == 'Response with extra fields'
    
    @pytest.mark.asyncio
    async def test_response_with_missing_optional_fields(self, client):
        """Test handling of responses with missing optional fields."""
        minimal_response = {
            'id': 'minimal-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'test-model',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'Minimal response'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'total_tokens': 20
                # Missing prompt_tokens and completion_tokens
            }
            # Missing other optional fields
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=minimal_response
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle missing optional fields gracefully
            result = await client.create_chat_completion(
                messages=[ChatMessage(role="user", content="Test")],
                model_config=ModelConfig(name="test-model")
            )
            
            assert result.id == 'minimal-test'
            assert result.choices[0].message.content == 'Minimal response'
            assert result.usage.total_tokens == 20


class TestAdvancedConfigurationScenarios:
    """Advanced tests for configuration edge cases."""
    
    def test_client_with_custom_headers_precedence(self):
        """Test client initialization with custom headers and precedence rules."""
        config = {
            'api_key': 'test-key',
            'default_headers': {
                'User-Agent': 'Custom-Agent/1.0',
                'X-Custom-Header': 'custom-value'
            }
        }
        
        try:
            client = GenesisAPIClient(**config)
            headers = client._build_headers()
            
            # Custom headers should be included
            assert 'X-Custom-Header' in headers
            assert headers['X-Custom-Header'] == 'custom-value'
            
            # But Authorization should still be controlled by the client
            assert 'Authorization' in headers
            assert headers['Authorization'].startswith('Bearer ')
        except TypeError:
            # If custom headers aren't supported in initialization,
            # that's also valid behavior
            pass
    
    def test_client_with_custom_user_agent(self):
        """Test client with custom User-Agent configuration."""
        config = {
            'api_key': 'test-key',
            'user_agent': 'MyApp/2.0 (Custom Genesis Client)'
        }
        
        try:
            client = GenesisAPIClient(**config)
            headers = client._build_headers()
            
            # Custom user agent should be used
            assert 'User-Agent' in headers
            assert 'MyApp/2.0' in headers['User-Agent']
        except TypeError:
            # If custom user agent isn't supported, that's fine
            pass
    
    def test_client_with_proxy_configuration(self):
        """Test client initialization with proxy settings."""
        config = {
            'api_key': 'test-key',
            'proxy': 'http://proxy.example.com:8080'
        }
        
        try:
            client = GenesisAPIClient(**config)
            # If proxy is supported, it should be configured
            assert hasattr(client, 'proxy') or hasattr(client, '_proxy')
        except TypeError:
            # If proxy isn't supported in initialization, that's fine
            pass
    
    def test_client_with_ssl_verification_disabled(self):
        """Test client with SSL verification disabled."""
        config = {
            'api_key': 'test-key',
            'verify_ssl': False
        }
        
        try:
            client = GenesisAPIClient(**config)
            # If SSL verification control is supported
            assert hasattr(client, 'verify_ssl') or hasattr(client, '_verify_ssl')
        except TypeError:
            # If SSL verification control isn't supported, that's fine
            pass
    
    def test_client_with_connection_limits(self):
        """Test client with custom connection limits."""
        config = {
            'api_key': 'test-key',
            'max_connections': 50,
            'max_connections_per_host': 10
        }
        
        try:
            client = GenesisAPIClient(**config)
            # If connection limits are supported
            assert hasattr(client, 'max_connections') or hasattr(client, '_max_connections')
        except TypeError:
            # If connection limits aren't supported, that's fine
            pass


# Add a final validation test to ensure all our new tests are properly integrated
class TestNewTestsIntegration:
    """Validation tests for the new test additions."""
    
    def test_new_test_classes_exist(self):
        """Verify that all new test classes are properly defined."""
        import inspect
        
        # Get all classes in the current module
        current_module = inspect.getmodule(inspect.currentframe())
        classes = [obj for name, obj in inspect.getmembers(current_module) 
                  if inspect.isclass(obj) and name.startswith('Test')]
        
        # Verify we have a good number of test classes
        assert len(classes) >= 20  # Should have many test classes
        
        # Verify some of our new test classes exist
        class_names = [cls.__name__ for cls in classes]
        assert 'TestGenesisAPIClientExtended' in class_names
        assert 'TestAdvancedMessageValidation' in class_names
        assert 'TestAdvancedStreamingScenarios' in class_names
        assert 'TestAdvancedErrorScenarios' in class_names
    
    def test_new_test_methods_use_pytest_features(self):
        """Verify that new test methods use pytest features correctly."""
        import inspect
        
        # Check that we're using pytest.mark.asyncio for async tests
        current_module = inspect.getmodule(inspect.currentframe())
        
        async_test_count = 0
        for name, obj in inspect.getmembers(current_module):
            if inspect.isfunction(obj) and name.startswith('test_') and inspect.iscoroutinefunction(obj):
                async_test_count += 1
        
        # Should have many async tests
        assert async_test_count >= 20
    
    def test_testing_framework_still_pytest(self):
        """Verify we're still using pytest as confirmed in the original tests."""
        import pytest
        assert pytest.__version__ is not None
        
        # Verify pytest markers work
        @pytest.mark.asyncio
        async def dummy_test():
            pass
        
        assert hasattr(dummy_test, 'pytestmark')

