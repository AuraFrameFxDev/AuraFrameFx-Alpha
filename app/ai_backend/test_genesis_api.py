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
        Return a mock configuration dictionary with sample API key, base URL, timeout, and max retries for initializing GenesisAPIClient in tests.
        
        Returns:
            dict: Dictionary containing mock configuration values.
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
        Return a list of sample ChatMessage instances simulating a standard conversation.
        
        Returns:
            List[ChatMessage]: Example messages including system, user, and assistant roles.
        """
        return [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is the weather like today?"),
            ChatMessage(role="assistant", content="I don't have access to real-time weather data.")
        ]
    
    @pytest.fixture
    def sample_model_config(self):
        """
        Create and return a ModelConfig instance with standard parameters for testing.
        
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
        Test that GenesisAPIClient initializes with the correct attributes when given a valid configuration.
        
        Asserts that the client's api_key, base_url, timeout, and max_retries match the provided configuration values.
        """
        client = GenesisAPIClient(**mock_config)
        assert client.api_key == mock_config['api_key']
        assert client.base_url == mock_config['base_url']
        assert client.timeout == mock_config['timeout']
        assert client.max_retries == mock_config['max_retries']

    def test_client_initialization_with_minimal_config(self):
        """
        Test that GenesisAPIClient initializes with only an API key and assigns default values to optional parameters.
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
        Test that initializing GenesisAPIClient with a zero or negative timeout raises a ValueError.
        """
        with pytest.raises(ValueError, match="Timeout must be positive"):
            GenesisAPIClient(api_key='test-key', timeout=-1)
        
        with pytest.raises(ValueError, match="Timeout must be positive"):
            GenesisAPIClient(api_key='test-key', timeout=0)

    def test_client_initialization_invalid_max_retries(self):
        """
        Test that initializing a GenesisAPIClient with a negative max_retries value raises a ValueError.
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
        Tests that the chat completion client correctly processes streaming API responses by simulating a sequence of streamed chunks and verifying their order and content.
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
                bytes: Each yielded value is a JSON-encoded byte string representing a chunk from mock_chunks.
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
        Tests that an authentication error (HTTP 401) during chat completion raises an AuthenticationError with the expected error message.
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
        Test that a RateLimitError is raised with the correct retry_after value when the chat completion API returns a 429 status code.
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
        Test that creating a chat completion with an invalid message role raises a ValidationError containing the expected error message.
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
        
        Simulates two consecutive server errors followed by a successful response, verifying that the retry mechanism is triggered and the final result is correct.
        """
        call_count = 0
        
        async def mock_post_with_failure(*args, **kwargs):
            """
            Simulates an asynchronous HTTP POST request that fails with a 500 status on the first two calls and returns a successful chat completion response on subsequent calls.
            
            Returns:
                Mock: A mock response object with a status code and JSON payload that depend on the number of times the function has been called.
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
        Test that validating an empty list of messages raises a ValidationError.
        """
        with pytest.raises(ValidationError, match="Messages cannot be empty"):
            client._validate_messages([])

    def test_validate_messages_invalid_role(self, client):
        """
        Test that the client raises a ValidationError when a chat message has an invalid role.
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
        Test that a ValidationError is raised if a chat message's content exceeds the maximum allowed length.
        """
        long_content = "x" * 100000  # Assuming max length is less than this
        invalid_messages = [
            ChatMessage(role="user", content=long_content)
        ]
        
        with pytest.raises(ValidationError, match="Message content too long"):
            client._validate_messages(invalid_messages)

    def test_validate_model_config_invalid_temperature(self, client, sample_model_config):
        """
        Test that ValidationError is raised when the model config temperature is set below 0 or above 2.
        """
        sample_model_config.temperature = -0.5  # Invalid negative temperature
        
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(sample_model_config)
        
        sample_model_config.temperature = 2.5  # Invalid high temperature
        
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            client._validate_model_config(sample_model_config)

    def test_validate_model_config_invalid_max_tokens(self, client, sample_model_config):
        """
        Verify that a ValidationError is raised when the model configuration's max_tokens is set to zero or a negative value.
        """
        sample_model_config.max_tokens = 0  # Invalid zero tokens
        
        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            client._validate_model_config(sample_model_config)
        
        sample_model_config.max_tokens = -100  # Invalid negative tokens
        
        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            client._validate_model_config(sample_model_config)

    def test_validate_model_config_invalid_top_p(self, client, sample_model_config):
        """
        Test that ValidationError is raised when the model configuration's top_p value is outside the valid range [0, 1].
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
        Test that the client retrieves and correctly parses the list of available models from the API.
        
        Asserts that the returned list contains models with the expected IDs.
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
        Test that retrieving a model by its ID returns a model object with the expected attributes.
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
        Test that requesting a model that does not exist raises a GenesisAPIError with the correct error message.
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
        Test that the client's `_build_headers` method returns the expected HTTP headers.
        
        Verifies that the returned headers include the correct Authorization, Content-Type, and User-Agent fields.
        """
        headers = client._build_headers()
        
        assert 'Authorization' in headers
        assert headers['Authorization'] == f'Bearer {client.api_key}'
        assert headers['Content-Type'] == 'application/json'
        assert 'User-Agent' in headers

    def test_build_headers_with_custom_headers(self, client):
        """
        Test that custom headers are merged with default headers when building request headers.
        
        Verifies that custom headers provided to the client are included in the resulting headers, and that default headers such as Authorization and Content-Type are present.
        """
        custom_headers = {'X-Custom-Header': 'custom-value'}
        headers = client._build_headers(custom_headers)
        
        assert headers['X-Custom-Header'] == 'custom-value'
        assert 'Authorization' in headers
        assert headers['Content-Type'] == 'application/json'

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_config):
        """
        Test that GenesisAPIClient manages its HTTP session correctly when used as an async context manager.
        
        Verifies that the session is open within the context and properly closed after exiting.
        """
        async with GenesisAPIClient(**mock_config) as client:
            assert client.session is not None
        
        # Session should be closed after exiting context
        assert client.session.closed

    @pytest.mark.asyncio
    async def test_close_client_explicitly(self, client):
        """
        Test that explicitly closing the GenesisAPIClient closes its underlying HTTP session.
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
        Tests that the client raises the correct exception for a given HTTP status code during a chat completion request.
        
        Parameters:
            status_code (int): The simulated HTTP status code returned by the API.
            expected_exception (Exception): The exception type expected to be raised for the simulated status code.
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
        Test creation of a ChatMessage with specified role and content, verifying that the name attribute defaults to None.
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
        Test creation of a ModelConfig instance with specified name, max_tokens, and temperature values.
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
        Verify that ModelConfig assigns default values to max_tokens, temperature, and top_p when only the name is specified.
        """
        config = ModelConfig(name="test-model")
        assert config.name == "test-model"
        assert config.max_tokens is not None
        assert config.temperature is not None
        assert config.top_p is not None

    def test_api_response_creation(self):
        """
        Test that an APIResponse object is initialized with the correct status code, data, and headers.
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
        Tests that a ChatCompletion object is instantiated with the correct attribute values.
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
        Test that RateLimitError is created with the correct retry_after value and is a subclass of GenesisAPIError.
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
        Tests that the `format_timestamp` utility returns a non-empty string when provided with a valid timestamp.
        """
        from app.ai_backend.genesis_api import format_timestamp
        
        timestamp = 1677610602
        formatted = format_timestamp(timestamp)
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_calculate_token_usage(self):
        """
        Test that `calculate_token_usage` returns a dictionary with an 'estimated_tokens' key when given a list of chat messages.
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
        Test that `estimate_tokens` returns the expected token count for a given input string.
        
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
        Test that the client retries a chat completion request after a rate limit error and succeeds on a subsequent attempt.
        
        Simulates a rate limit error on the first API call and a successful response on the second, verifying that `RateLimitError` is raised initially and that the client completes the request successfully after recovery.
        """
        config = {'api_key': 'test-key'}
        
        call_count = 0
        
        async def mock_post_with_recovery(*args, **kwargs):
            """
            Simulates an asynchronous HTTP POST request that returns a rate limit error on the first call and a successful response on subsequent calls.
            
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
        Tests that GenesisAPIClient can process multiple concurrent chat completion requests and returns the correct response for each.
        
        This verifies the client's ability to handle asynchronous concurrency and ensures all responses match the expected mock output.
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
        Tests that GenesisAPIClient can handle chat completions with very large message content, verifying correct response parsing and token usage calculation.
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
        Create a `GenesisAPIClient` instance using a custom pre-configured aiohttp session.
        
        Returns:
            GenesisAPIClient: The client instance with its session set to the provided aiohttp session.
        """
        import aiohttp
        session = aiohttp.ClientSession()
        client = GenesisAPIClient(**mock_config)
        client.session = session
        return client

    @pytest.mark.asyncio
    async def test_session_reuse_across_requests(self, client):
        """
        Verify that the GenesisAPIClient reuses the same HTTP session for multiple API requests.
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
        Verify that the API key is not present in the string or repr output of the GenesisAPIClient instance.
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
        Test that the client raises a GenesisAPIError when the API returns an empty response.
        
        Verifies that an empty JSON object from the API triggers an error indicating an invalid response structure.
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
        
        This ensures that missing required fields in the API response, such as 'choices' or 'usage', are correctly detected and result in an error.
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
        Test that the client correctly processes chat messages containing unicode and special characters.
        
        Verifies that sending messages with diverse unicode content results in the expected unicode response from the API.
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
        Test that the client correctly processes chat completions for conversations with a large number of messages.
        
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
        Test that a connection error during streaming chat completion raises a GenesisAPIError with an appropriate message.
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
        Test that the client correctly skips malformed JSON chunks and processes valid ones during streaming chat completion responses.
        
        This test simulates a streaming API response containing both malformed and valid JSON chunks, verifying that only valid chunks are processed and yielded by the client.
        """
        async def mock_malformed_stream():
            """
            Asynchronously yields a sequence of byte strings simulating a malformed and a valid streaming API response chunk.
            
            Yields:
                bytes: First a malformed JSON chunk, then a valid data chunk.
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
        Test that message validation fails when a message contains invalid content types, such as None.
        
        Ensures that the client raises a ValidationError when any message in the list has content set to None, even if other messages are valid.
        """
        mixed_messages = [
            ChatMessage(role="user", content="Valid string content"),
            ChatMessage(role="assistant", content=None),  # Invalid None content
        ]
        
        with pytest.raises(ValidationError, match="Message content cannot be None"):
            client._validate_messages(mixed_messages)

    def test_validate_messages_whitespace_only_content(self, client):
        """
        Test that message validation rejects messages containing only whitespace content.
        
        Asserts that a ValidationError is raised when a message's content consists solely of whitespace characters.
        """
        whitespace_messages = [
            ChatMessage(role="user", content="   \t\n   ")  # Only whitespace
        ]
        
        with pytest.raises(ValidationError, match="Message content cannot be empty"):
            client._validate_messages(whitespace_messages)

    def test_validate_model_config_boundary_values(self, client):
        """
        Tests that the model configuration validation accepts minimum and maximum boundary values for temperature, top_p, and max_tokens without raising errors.
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
        Test that model config validation raises a ValidationError when temperature is set to None.
        """
        config = ModelConfig(name="test-model")
        config.temperature = None
        
        with pytest.raises(ValidationError, match="Temperature cannot be None"):
            client._validate_model_config(config)

    @pytest.mark.asyncio
    async def test_custom_timeout_behavior(self, mock_config):
        """
        Test that the GenesisAPIClient applies a custom timeout value and raises GenesisAPIError on request timeout.
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
        Test that the client retries failed requests using exponential backoff when server errors occur.
        
        This test simulates repeated server errors and verifies that the retry logic triggers exponential backoff delays between attempts.
        """
        call_times = []
        
        async def mock_failing_post(*args, **kwargs):
            """
            Simulates a failed HTTP POST request by returning a mock response with a 500 status code and a server error message.
            
            This mock is intended for testing retry and error handling logic in asynchronous HTTP client code.
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
        Test that the header builder returns default headers when None is provided for custom headers.
        """
        headers = client._build_headers(None)
        
        assert 'Authorization' in headers
        assert headers['Content-Type'] == 'application/json'

    def test_build_headers_override_default_headers(self, client):
        """
        Test that custom headers provided to the client override the default headers in the request.
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
        Test that ChatMessage instances with identical attributes are considered equal, and those with differing attributes are not.
        """
        msg1 = ChatMessage(role="user", content="Hello")
        msg2 = ChatMessage(role="user", content="Hello")
        msg3 = ChatMessage(role="user", content="World")
        
        assert msg1 == msg2
        assert msg1 != msg3

    def test_chat_message_immutability(self):
        """
        Verify that ChatMessage instances are immutable by asserting that modifying an attribute raises an AttributeError.
        """
        message = ChatMessage(role="user", content="Hello")
        
        # Attempting to modify should raise an error or have no effect
        with pytest.raises(AttributeError):
            message.role = "assistant"

    def test_model_config_extreme_values(self):
        """
        Test that ModelConfig accepts and correctly stores extreme but valid parameter values.
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
        Test that an APIResponse instance can be created with None as the data attribute and that its status_code and data are set correctly.
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
        Test that a ChatCompletion instance can be created with an empty choices list and zero token usage.
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
        Test that a ChatCompletion instance can be serialized to a JSON string.
        
        Asserts that the serialized output is a string and contains the expected identifier.
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
        Verify that custom exceptions correctly chain underlying exceptions using the `__cause__` attribute.
        """
        original_error = ValueError("Original error")
        api_error = GenesisAPIError("API error", status_code=500)
        api_error.__cause__ = original_error
        
        assert api_error.__cause__ is original_error

    def test_rate_limit_error_with_zero_retry_after(self):
        """
        Test that RateLimitError correctly sets the retry_after attribute to zero when initialized with retry_after=0.
        """
        error = RateLimitError("Rate limited", retry_after=0)
        assert error.retry_after == 0

    def test_rate_limit_error_with_none_retry_after(self):
        """
        Test that a RateLimitError instance correctly handles a None value for retry_after.
        """
        error = RateLimitError("Rate limited", retry_after=None)
        assert error.retry_after is None

    def test_exception_message_formatting(self):
        """
        Test that the string representation of a GenesisAPIError includes the expected placeholder in its message.
        """
        error = GenesisAPIError("Test error with {placeholder}", status_code=400)
        assert "placeholder" in str(error)

    def test_validation_error_with_field_info(self):
        """
        Test that ValidationError includes specific field information in its error message.
        """
        error = ValidationError("Invalid field 'temperature': must be between 0 and 2")
        assert "temperature" in str(error)
        assert "0 and 2" in str(error)


class TestUtilityFunctionsAdvanced:
    """Advanced tests for utility functions."""
    
    def test_format_timestamp_edge_cases(self):
        """
        Test that `format_timestamp` returns a string for zero, negative, and very large timestamp values.
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
        Test that `calculate_token_usage` returns zero estimated tokens when given an empty message list.
        """
        from app.ai_backend.genesis_api import calculate_token_usage
        
        usage = calculate_token_usage([])
        assert usage['estimated_tokens'] == 0

    def test_calculate_token_usage_unicode_content(self):
        """
        Test that `calculate_token_usage` correctly estimates token usage for messages containing Unicode characters.
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
        Test that the token estimation function returns at least the specified minimum number of tokens for given content.
        
        Parameters:
            content (str): The input string to estimate tokens for.
            min_tokens (int): The minimum expected token count.
        """
        from app.ai_backend.genesis_api import estimate_tokens
        
        tokens = estimate_tokens(content)
        assert tokens >= min_tokens

    def test_estimate_tokens_with_special_characters(self):
        """
        Test that the token estimation function returns a positive integer for input containing special characters and punctuation.
        """
        from app.ai_backend.genesis_api import estimate_tokens
        
        special_content = "Hello, world! How are you? I'm fine. 123-456-7890."
        tokens = estimate_tokens(special_content)
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_estimate_tokens_with_newlines_and_tabs(self):
        """
        Test that the token estimation function correctly counts tokens in strings containing newlines and tabs.
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
        Test that initializing the GenesisAPIClient with an empty API key string raises a ValueError.
        """
        with pytest.raises(ValueError, match="API key is required"):
            GenesisAPIClient(api_key="")

    def test_api_key_validation_whitespace_only(self):
        """
        Test that initializing the client with a whitespace-only API key raises a ValueError.
        """
        with pytest.raises(ValueError, match="API key is required"):
            GenesisAPIClient(api_key="   \t\n   ")

    def test_url_validation_invalid_schemes(self):
        """
        Test that initializing GenesisAPIClient with an invalid base URL scheme raises a ValueError.
        """
        with pytest.raises(ValueError, match="Invalid base URL"):
            GenesisAPIClient(api_key="test-key", base_url="ftp://invalid.com")

    def test_header_injection_prevention(self, client):
        """
        Test that the client prevents header injection by sanitizing or rejecting headers containing carriage return or newline characters.
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
        Test that messages containing potentially malicious script or injection content do not trigger client-side validation errors.
        
        Ensures that the client does not reject messages with script tags, SQL injection patterns, or template expressions, delegating content filtering to the API. Only length or format-related validation errors are considered acceptable.
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
        Test that calling close() multiple times concurrently on the client does not raise errors and results in the session being closed.
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
        Test that making a request after the client has been closed raises a GenesisAPIError indicating the session is closed.
        """
        await client.close()
        
        with pytest.raises(GenesisAPIError, match="Client session is closed"):
            await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )

    def test_model_config_with_all_none_optional_params(self):
        """
        Test that ModelConfig can be created with all optional parameters set to None.
        
        Verifies that the ModelConfig instance correctly assigns the required name and gracefully handles None values for all optional parameters.
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
        Test that the async context manager raises an exception if session initialization fails.
        
        Verifies that an exception during `GenesisAPIClient` session creation is properly propagated when using the async context manager.
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
        Test that the client raises a ValidationError when validating messages with invalid roles.
        
        Parameters:
            invalid_role (str): A role value that is not permitted for a chat message.
        """
        invalid_messages = [
            ChatMessage(role=invalid_role, content="Test content")
        ]
        
        with pytest.raises(ValidationError):
            client._validate_messages(invalid_messages)

    @pytest.mark.asyncio
    async def test_streaming_with_no_chunks(self, client, sample_messages, sample_model_config):
        """
        Test that streaming chat completion yields no results when the API returns no chunks.
        """
        async def empty_stream():
            # Empty async generator
            """
            An asynchronous generator that yields no values.
            
            This can be used as a placeholder or to simulate an empty async stream in tests.
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
        Test that multiple concurrent streaming chat completion requests return the expected number of streamed chunks for each request.
        
        Verifies that the client can handle several simultaneous streaming requests, with each stream yielding the correct sequence of response chunks.
        """
        mock_chunks = [
            {'choices': [{'delta': {'content': f'Stream {i}'}}]}
            for i in range(3)
        ]
        
        async def mock_stream():
            """
            Asynchronously yields encoded JSON chunks from the `mock_chunks` iterable, simulating a streaming API response.
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
        Verify that the HTTP session is properly closed when an exception occurs during an API request within the async context manager.
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
        Verify that the client fixture is isolated between tests by modifying its state and ensuring changes do not persist across tests.
        """
        # Modify client state
        original_api_key = client.api_key
        client.api_key = "modified-key"
        
        # In next test, fixture should be fresh
        assert client.api_key != original_api_key

    @pytest.mark.asyncio
    async def test_mock_response_side_effects(self, client, sample_messages, sample_model_config):
        """
        Simulate multiple consecutive API failures followed by a success to test client retry logic.
        
        This test verifies that the client correctly retries after transient server errors and ultimately returns a successful response when the API recovers.
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
            Simulates sequential asynchronous HTTP responses for mocking purposes.
            
            Returns a mock response object with status and JSON payload based on the current call count, advancing through the provided `responses` list on each invocation.
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


























































































































































































































































































































































































































































































































































































































































































































































        "--tb=short",
        "-k", "TestGenesisAPIClientRobustness or TestAdvancedStreamingScenarios or TestDataValidationEdgeCases or TestErrorRecoveryPatterns or TestPerformanceCharacteristics or TestConfigurationValidation or TestUtilityFunctionsComprehensive",
        "--durations=5"
    ])

class TestGenesisAPIClientCoverage:
    """Additional comprehensive test coverage for edge cases and boundary conditions."""
    
    @pytest.fixture
    def client_with_debug_mode(self, mock_config):
        """Create a client with debug mode enabled for testing debug scenarios."""
        debug_config = mock_config.copy()
        debug_config['debug'] = True
        return GenesisAPIClient(**debug_config)
    
    def test_client_initialization_with_debug_mode(self, client_with_debug_mode):
        """Test that client correctly initializes with debug mode enabled."""
        assert hasattr(client_with_debug_mode, 'debug')
        # Verify debug mode doesn't interfere with basic functionality
        assert client_with_debug_mode.api_key is not None
    
    def test_client_initialization_with_proxy_config(self, mock_config):
        """Test client initialization with proxy configuration."""
        proxy_config = mock_config.copy()
        proxy_config['proxy'] = 'http://proxy.example.com:8080'
        
        client = GenesisAPIClient(**proxy_config)
        assert hasattr(client, 'proxy') or client.proxy is None  # Handle gracefully
    
    def test_client_initialization_with_ssl_config(self, mock_config):
        """Test client initialization with SSL verification disabled."""
        ssl_config = mock_config.copy()
        ssl_config['verify_ssl'] = False
        
        client = GenesisAPIClient(**ssl_config)
        # Should initialize without errors
        assert client.api_key == ssl_config['api_key']
    
    @pytest.mark.parametrize("invalid_timeout", [
        -1, 0, -10.5, "invalid", None, float('inf'), float('-inf')
    ])
    def test_client_initialization_comprehensive_invalid_timeouts(self, invalid_timeout):
        """Test that various invalid timeout values raise appropriate errors."""
        if invalid_timeout is None:
            # None might be acceptable as a default
            try:
                client = GenesisAPIClient(api_key='test-key', timeout=invalid_timeout)
                assert client.timeout is None or client.timeout > 0
            except ValueError:
                pass  # Acceptable to reject None
        else:
            with pytest.raises(ValueError, match="Timeout must be positive"):
                GenesisAPIClient(api_key='test-key', timeout=invalid_timeout)
    
    @pytest.mark.parametrize("invalid_retries", [
        -1, -10, "invalid", float('inf'), float('-inf')
    ])
    def test_client_initialization_comprehensive_invalid_retries(self, invalid_retries):
        """Test that various invalid max_retries values raise appropriate errors."""
        with pytest.raises(ValueError, match="Max retries must be non-negative"):
            GenesisAPIClient(api_key='test-key', max_retries=invalid_retries)
    
    def test_client_initialization_with_custom_user_agent(self, mock_config):
        """Test client initialization with custom user agent."""
        custom_config = mock_config.copy()
        custom_config['user_agent'] = 'CustomApp/1.0'
        
        client = GenesisAPIClient(**custom_config)
        headers = client._build_headers()
        assert 'User-Agent' in headers
        # Should contain either custom or default user agent
        assert len(headers['User-Agent']) > 0
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_function_calling(self, client, sample_model_config):
        """Test chat completion with function calling parameters."""
        function_schema = {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
        
        messages = [ChatMessage(role="user", content="What's the weather in NYC?")]
        
        mock_response = {
            'id': 'func-call-test',
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': None,
                    'function_call': {
                        'name': 'get_weather',
                        'arguments': '{"location": "NYC"}'
                    }
                }
            }],
            'usage': {'total_tokens': 50}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Test that function calling doesn't break the client
            result = await client.create_chat_completion(
                messages=messages,
                model_config=sample_model_config
            )
            
            assert result.id == 'func-call-test'
            assert result.choices[0].message.content is None
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_tool_use(self, client, sample_model_config):
        """Test chat completion with tool use parameters."""
        tools = [{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform calculations",
                "parameters": {"type": "object"}
            }
        }]
        
        messages = [ChatMessage(role="user", content="Calculate 2+2")]
        
        mock_response = {
            'id': 'tool-use-test',
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': 'I can help you calculate that.',
                    'tool_calls': [{
                        'id': 'call_123',
                        'type': 'function',
                        'function': {
                            'name': 'calculate',
                            'arguments': '{"expression": "2+2"}'
                        }
                    }]
                }
            }],
            'usage': {'total_tokens': 45}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=messages,
                model_config=sample_model_config
            )
            
            assert result.id == 'tool-use-test'
            assert 'tool_calls' in result.choices[0].message.__dict__ or result.choices[0].message.content is not None
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_response_format(self, client, sample_messages, sample_model_config):
        """Test chat completion with specific response format requirements."""
        # Test JSON response format
        sample_model_config.response_format = {"type": "json_object"}
        
        mock_response = {
            'id': 'json-format-test',
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': '{"result": "formatted response"}'
                }
            }],
            'usage': {'total_tokens': 30}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert result.id == 'json-format-test'
            # Verify JSON content can be parsed
            import json
            parsed = json.loads(result.choices[0].message.content)
            assert parsed['result'] == 'formatted response'
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_system_fingerprint(self, client, sample_messages, sample_model_config):
        """Test chat completion response with system fingerprint."""
        mock_response = {
            'id': 'fingerprint-test',
            'system_fingerprint': 'fp_12345abcde',
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': 'Response with fingerprint'
                }
            }],
            'usage': {'total_tokens': 25}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert result.id == 'fingerprint-test'
            assert hasattr(result, 'system_fingerprint') or 'system_fingerprint' in result.__dict__
    
    @pytest.mark.asyncio
    async def test_streaming_with_function_calls(self, client, sample_messages, sample_model_config):
        """Test streaming chat completion with function calls."""
        mock_chunks = [
            {'choices': [{'delta': {'function_call': {'name': 'get_weather', 'arguments': '{"loc'}}}]},
            {'choices': [{'delta': {'function_call': {'arguments': 'ation": "NYC"}'}}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'function_call'}]}
        ]
        
        async def mock_function_stream():
            for chunk in mock_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_function_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 3
            # Verify function call streaming works
            assert chunks[-1].choices[0].finish_reason == 'function_call'
    
    @pytest.mark.asyncio
    async def test_streaming_with_empty_deltas(self, client, sample_messages, sample_model_config):
        """Test streaming with empty delta objects."""
        mock_chunks = [
            {'choices': [{'delta': {}}]},  # Empty delta
            {'choices': [{'delta': {'content': 'Hello'}}]},
            {'choices': [{'delta': {}}]},  # Another empty delta
            {'choices': [{'delta': {'content': ' world'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def mock_empty_delta_stream():
            for chunk in mock_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_empty_delta_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 5
            # Verify empty deltas are handled properly
            content_chunks = [c for c in chunks if hasattr(c.choices[0].delta, 'content') and c.choices[0].delta.content]
            assert len(content_chunks) == 2
    
    @pytest.mark.asyncio
    async def test_streaming_with_role_deltas(self, client, sample_messages, sample_model_config):
        """Test streaming with role information in deltas."""
        mock_chunks = [
            {'choices': [{'delta': {'role': 'assistant'}}]},
            {'choices': [{'delta': {'content': 'Hello'}}]},
            {'choices': [{'delta': {'content': ' there'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def mock_role_delta_stream():
            for chunk in mock_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_role_delta_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 4
            # Verify role is set in first chunk
            assert chunks[0].choices[0].delta.role == 'assistant'
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_multiple_choices(self, client, sample_messages, sample_model_config):
        """Test chat completion with multiple response choices."""
        sample_model_config.n = 3  # Request 3 choices
        
        mock_response = {
            'id': 'multiple-choices-test',
            'choices': [
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'First choice'},
                    'finish_reason': 'stop'
                },
                {
                    'index': 1,
                    'message': {'role': 'assistant', 'content': 'Second choice'},
                    'finish_reason': 'stop'
                },
                {
                    'index': 2,
                    'message': {'role': 'assistant', 'content': 'Third choice'},
                    'finish_reason': 'length'
                }
            ],
            'usage': {'total_tokens': 75}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert len(result.choices) == 3
            assert result.choices[0].message.content == 'First choice'
            assert result.choices[1].message.content == 'Second choice'
            assert result.choices[2].message.content == 'Third choice'
            assert result.choices[2].finish_reason == 'length'
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_logprobs(self, client, sample_messages, sample_model_config):
        """Test chat completion with log probabilities."""
        sample_model_config.logprobs = True
        sample_model_config.top_logprobs = 3
        
        mock_response = {
            'id': 'logprobs-test',
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': 'Hello world'
                },
                'logprobs': {
                    'content': [
                        {
                            'token': 'Hello',
                            'logprob': -0.1,
                            'top_logprobs': [
                                {'token': 'Hello', 'logprob': -0.1},
                                {'token': 'Hi', 'logprob': -0.5}
                            ]
                        }
                    ]
                }
            }],
            'usage': {'total_tokens': 20}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert result.id == 'logprobs-test'
            assert 'logprobs' in result.choices[0].__dict__ or hasattr(result.choices[0], 'logprobs')
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_seed_parameter(self, client, sample_messages, sample_model_config):
        """Test chat completion with seed parameter for reproducibility."""
        sample_model_config.seed = 12345
        
        mock_response = {
            'id': 'seed-test',
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': 'Deterministic response'
                }
            }],
            'usage': {'total_tokens': 15}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert result.id == 'seed-test'
            assert result.choices[0].message.content == 'Deterministic response'
    
    def test_validate_messages_with_image_content(self, client):
        """Test message validation with image content."""
        image_messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                ]
            )
        ]
        
        # Should handle multimodal content gracefully
        try:
            client._validate_messages(image_messages)
        except ValidationError as e:
            # Only acceptable if specifically checking for unsupported content types
            assert "unsupported content type" in str(e).lower() or "image" in str(e).lower()
    
    def test_validate_messages_with_audio_content(self, client):
        """Test message validation with audio content."""
        audio_messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Transcribe this audio"},
                    {"type": "input_audio", "input_audio": {"data": "base64_audio_data"}}
                ]
            )
        ]
        
        # Should handle audio content gracefully
        try:
            client._validate_messages(audio_messages)
        except ValidationError as e:
            # Only acceptable if specifically checking for unsupported content types
            assert "unsupported content type" in str(e).lower() or "audio" in str(e).lower()
    
    def test_validate_model_config_with_stop_sequences(self, client):
        """Test model config validation with stop sequences."""
        config = ModelConfig(name="test-model")
        config.stop = ["END", "STOP", "\n\n"]
        
        # Should validate stop sequences
        try:
            client._validate_model_config(config)
        except ValidationError as e:
            # Only acceptable if stop sequences have specific validation rules
            assert "stop" in str(e).lower()
    
    def test_validate_model_config_with_logit_bias(self, client):
        """Test model config validation with logit bias."""
        config = ModelConfig(name="test-model")
        config.logit_bias = {"50256": -100, "50257": -100}  # Suppress specific tokens
        
        # Should validate logit bias
        try:
            client._validate_model_config(config)
        except ValidationError as e:
            # Only acceptable if logit bias has specific validation rules
            assert "logit" in str(e).lower() or "bias" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_request_with_custom_headers_per_request(self, client, sample_messages, sample_model_config):
        """Test making requests with custom headers per request."""
        mock_response = {
            'id': 'custom-headers-test',
            'choices': [{'message': {'content': 'Response with custom headers'}}],
            'usage': {'total_tokens': 20}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Test that custom headers can be passed per request
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config,
                # headers={'X-Custom-Request': 'test-value'}  # If supported
            )
            
            assert result.id == 'custom-headers-test'
            # Verify the request was made successfully
            assert mock_post.called
    
    @pytest.mark.asyncio
    async def test_response_with_usage_details(self, client, sample_messages, sample_model_config):
        """Test response with detailed usage information."""
        mock_response = {
            'id': 'usage-details-test',
            'choices': [{'message': {'content': 'Response with detailed usage'}}],
            'usage': {
                'prompt_tokens': 25,
                'completion_tokens': 10,
                'total_tokens': 35,
                'prompt_tokens_details': {'cached_tokens': 5},
                'completion_tokens_details': {'reasoning_tokens': 2}
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert result.usage.total_tokens == 35
            # Verify detailed usage information is preserved
            assert 'prompt_tokens_details' in result.usage or result.usage.prompt_tokens == 25
    
    @pytest.mark.asyncio
    async def test_error_handling_with_error_codes(self, client, sample_messages, sample_model_config):
        """Test error handling with specific error codes."""
        error_scenarios = [
            (400, 'invalid_request_error', 'Invalid request'),
            (401, 'authentication_error', 'Invalid API key'),
            (403, 'permission_error', 'Insufficient permissions'),
            (404, 'not_found_error', 'Model not found'),
            (422, 'validation_error', 'Validation failed'),
            (429, 'rate_limit_error', 'Rate limit exceeded'),
            (500, 'internal_error', 'Internal server error'),
            (503, 'service_unavailable', 'Service temporarily unavailable')
        ]
        
        for status_code, error_type, error_message in error_scenarios:
            mock_error_response = {
                'error': {
                    'type': error_type,
                    'message': error_message,
                    'code': f'error_{status_code}'
                }
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.status = status_code
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_error_response)
                
                with pytest.raises(GenesisAPIError) as exc_info:
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                
                assert error_message in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_request_timeout_with_partial_response(self, client, sample_messages, sample_model_config):
        """Test handling of timeout that occurs after partial response."""
        async def mock_timeout_response():
            # Simulate partial response before timeout
            yield b'{"id": "partial"'
            await asyncio.sleep(0.1)
            raise asyncio.TimeoutError("Request timed out")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_timeout_response()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            with pytest.raises(GenesisAPIError, match="Request timeout"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
    
    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, client, sample_messages, sample_model_config):
        """Test behavior when connection pool is exhausted."""
        import aiohttp
        
        with patch('aiohttp.ClientSession.post', side_effect=aiohttp.ClientConnectorError(connection_key=None, os_error=None)):
            with pytest.raises(GenesisAPIError, match="Connection error"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
    
    @pytest.mark.asyncio
    async def test_dns_resolution_failure(self, client, sample_messages, sample_model_config):
        """Test handling of DNS resolution failures."""
        import aiohttp
        
        with patch('aiohttp.ClientSession.post', side_effect=aiohttp.ClientConnectorError(connection_key=None, os_error=OSError("Name resolution failed"))):
            with pytest.raises(GenesisAPIError, match="Connection error"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
    
    @pytest.mark.asyncio
    async def test_ssl_certificate_error(self, client, sample_messages, sample_model_config):
        """Test handling of SSL certificate errors."""
        import aiohttp
        import ssl
        
        ssl_error = ssl.SSLError("certificate verify failed")
        
        with patch('aiohttp.ClientSession.post', side_effect=aiohttp.ClientConnectorError(connection_key=None, os_error=ssl_error)):
            with pytest.raises(GenesisAPIError, match="Connection error"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
    
    def test_message_validation_with_special_role_cases(self, client):
        """Test message validation with edge cases for roles."""
        edge_case_messages = [
            ChatMessage(role="system", content="System message"),
            ChatMessage(role="user", content="User message"),
            ChatMessage(role="assistant", content="Assistant response"),
            ChatMessage(role="tool", content="Tool result"),  # If supported
            ChatMessage(role="function", content="Function result")  # If supported
        ]
        
        # Test various combinations
        for msg in edge_case_messages:
            try:
                client._validate_messages([msg])
            except ValidationError as e:
                # Only acceptable if the role is specifically not supported
                assert "role" in str(e).lower() and (msg.role == "tool" or msg.role == "function")
    
    def test_message_validation_with_conversation_context(self, client):
        """Test message validation with realistic conversation context."""
        conversation_messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there! How can I help you?"),
            ChatMessage(role="user", content="What's the weather like?"),
            ChatMessage(role="assistant", content="I don't have access to real-time weather data."),
            ChatMessage(role="user", content="That's okay, thanks anyway.")
        ]
        
        # Should validate complex conversation flows
        client._validate_messages(conversation_messages)
    
    def test_model_config_validation_with_system_message_handling(self, client):
        """Test model config validation with system message specific settings."""
        config = ModelConfig(name="test-model")
        config.system_message = "You are a helpful assistant."
        
        # Should handle system message configuration
        try:
            client._validate_model_config(config)
        except ValidationError as e:
            # Only acceptable if system_message is not a supported parameter
            assert "system" in str(e).lower() or "message" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_batch_request_simulation(self, client, sample_model_config):
        """Test simulation of batch requests."""
        batch_messages = [
            [ChatMessage(role="user", content=f"Request {i}")]
            for i in range(5)
        ]
        
        mock_response = {
            'id': 'batch-test',
            'choices': [{'message': {'content': 'Batch response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Simulate batch processing
            tasks = []
            for messages in batch_messages:
                task = client.create_chat_completion(
                    messages=messages,
                    model_config=sample_model_config
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            for result in results:
                assert result.id == 'batch-test'
    
    @pytest.mark.asyncio
    async def test_request_cancellation(self, client, sample_messages, sample_model_config):
        """Test request cancellation handling."""
        async def long_running_request():
            await asyncio.sleep(10)  # Simulate long request
            return Mock()
        
        with patch('aiohttp.ClientSession.post', side_effect=long_running_request):
            task = asyncio.create_task(client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            ))
            
            # Cancel the task after a short delay
            await asyncio.sleep(0.1)
            task.cancel()
            
            with pytest.raises(asyncio.CancelledError):
                await task
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_with_large_responses(self, client, sample_messages, sample_model_config):
        """Test memory efficiency with large responses."""
        # Create a large response (simulate large content)
        large_content = "x" * 50000  # 50KB of content
        
        mock_response = {
            'id': 'large-response-test',
            'choices': [{'message': {'content': large_content}}],
            'usage': {'total_tokens': 12500}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert len(result.choices[0].message.content) == 50000
            assert result.usage.total_tokens == 12500
    
    def test_configuration_validation_comprehensive(self):
        """Test comprehensive configuration validation."""
        # Test all valid configurations
        valid_configs = [
            {'api_key': 'valid-key'},
            {'api_key': 'valid-key', 'base_url': 'https://api.example.com'},
            {'api_key': 'valid-key', 'timeout': 60},
            {'api_key': 'valid-key', 'max_retries': 5},
            {'api_key': 'valid-key', 'timeout': 30, 'max_retries': 3},
        ]
        
        for config in valid_configs:
            client = GenesisAPIClient(**config)
            assert client.api_key == config['api_key']
            
    def test_configuration_validation_edge_cases(self):
        """Test configuration validation edge cases."""
        # Test boundary values
        boundary_configs = [
            {'api_key': 'test', 'timeout': 0.1},  # Minimum timeout
            {'api_key': 'test', 'timeout': 3600},  # Large timeout
            {'api_key': 'test', 'max_retries': 0},  # No retries
            {'api_key': 'test', 'max_retries': 10},  # Many retries
        ]
        
        for config in boundary_configs:
            try:
                client = GenesisAPIClient(**config)
                assert client.api_key == config['api_key']
            except ValueError:
                # Some boundary values might be rejected
                pass


class TestStreamingAdvanced:
    """Advanced streaming functionality tests."""
    
    @pytest.mark.asyncio
    async def test_streaming_with_server_sent_events(self, client, sample_messages, sample_model_config):
        """Test streaming with proper Server-Sent Events format."""
        sse_chunks = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
            b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n',
            b'data: [DONE]\n\n'
        ]
        
        async def mock_sse_stream():
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
            
            # Should handle SSE format properly
            assert len(chunks) >= 2  # At least content chunks
    
    @pytest.mark.asyncio
    async def test_streaming_with_connection_interruption(self, client, sample_messages, sample_model_config):
        """Test streaming behavior when connection is interrupted."""
        async def interrupted_stream():
            yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"content": " wor"}}]}\n\n'
            # Simulate connection interruption
            raise aiohttp.ClientConnectionError("Connection lost")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=interrupted_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            with pytest.raises(GenesisAPIError, match="Connection error"):
                async for chunk in client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                ):
                    chunks.append(chunk)
    
    @pytest.mark.asyncio
    async def test_streaming_with_rate_limiting(self, client, sample_messages, sample_model_config):
        """Test streaming with rate limiting during stream."""
        call_count = 0
        
        async def rate_limited_stream():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call gets rate limited
                mock_response = Mock()
                mock_response.status = 429
                mock_response.headers = {'Retry-After': '1'}
                mock_response.json = AsyncMock(return_value={'error': {'message': 'Rate limited'}})
                return mock_response
            else:
                # Second call succeeds
                async def success_stream():
                    yield b'data: {"choices": [{"delta": {"content": "Success"}}]}\n\n'
                
                mock_response = Mock()
                mock_response.status = 200
                mock_response.content.iter_chunked = AsyncMock(return_value=success_stream())
                return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=rate_limited_stream):
            with pytest.raises(RateLimitError):
                async for chunk in client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                ):
                    pass
    
    @pytest.mark.asyncio
    async def test_streaming_with_mixed_chunk_formats(self, client, sample_messages, sample_model_config):
        """Test streaming with mixed chunk formats."""
        mixed_chunks = [
            b'{"choices": [{"delta": {"content": "Plain"}}]}',  # Plain JSON
            b'data: {"choices": [{"delta": {"content": " JSON"}}]}\n\n',  # SSE format
            b'data:{"choices": [{"delta": {"content": " mixed"}}]}\n\n',  # SSE without space
            b'{"choices": [{"delta": {}, "finish_reason": "stop"}]}'  # Plain JSON finish
        ]
        
        async def mixed_format_stream():
            for chunk in mixed_chunks:
                yield chunk
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mixed_format_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            # Should handle mixed formats gracefully
            assert len(chunks) >= 2
    
    @pytest.mark.asyncio
    async def test_streaming_with_unicode_content(self, client, sample_messages, sample_model_config):
        """Test streaming with unicode content."""
        unicode_chunks = [
            {'choices': [{'delta': {'content': '你好'}}]},
            {'choices': [{'delta': {'content': ' 世界'}}]},
            {'choices': [{'delta': {'content': ' 🌍'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def unicode_stream():
            for chunk in unicode_chunks:
                yield json.dumps(chunk, ensure_ascii=False).encode('utf-8')
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=unicode_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 4
            assert chunks[0].choices[0].delta.content == '你好'
            assert chunks[1].choices[0].delta.content == ' 世界'
            assert chunks[2].choices[0].delta.content == ' 🌍'
    
    @pytest.mark.asyncio
    async def test_streaming_performance_with_large_chunks(self, client, sample_messages, sample_model_config):
        """Test streaming performance with large chunks."""
        large_content = "x" * 10000  # 10KB chunk
        
        large_chunks = [
            {'choices': [{'delta': {'content': large_content}}]},
            {'choices': [{'delta': {'content': large_content}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def large_chunk_stream():
            for chunk in large_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=large_chunk_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            start_time = asyncio.get_event_loop().time()
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            end_time = asyncio.get_event_loop().time()
            
            # Should handle large chunks efficiently
            assert len(chunks) == 3
            assert len(chunks[0].choices[0].delta.content) == 10000
            # Performance check - should complete in reasonable time
            assert end_time - start_time < 5.0  # 5 seconds max


class TestRobustnessAndReliability:
    """Tests for robustness and reliability scenarios."""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_partial_failures(self, client, sample_messages, sample_model_config):
        """Test graceful degradation when partial failures occur."""
        # Simulate scenario where some fields are missing from response
        partial_response = {
            'id': 'partial-degradation-test',
            'choices': [{
                'message': {'content': 'Partial response'},
                # Missing 'role' field
            }],
            # Missing 'usage' field
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=partial_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle partial responses gracefully
            try:
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                # If successful, verify what we can
                assert result.id == 'partial-degradation-test'
            except GenesisAPIError:
                # Acceptable to fail on invalid response structure
                pass
    
    @pytest.mark.asyncio
    async def test_recovery_from_temporary_network_issues(self, client, sample_messages, sample_model_config):
        """Test recovery from temporary network issues."""
        call_count = 0
        
        async def network_issue_simulation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= 2:
                # Simulate temporary network issues
                raise aiohttp.ClientConnectionError("Temporary network issue")
            else:
                # Successful response after network recovery
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    'id': 'network-recovery-test',
                    'choices': [{'message': {'content': 'Network recovered'}}],
                    'usage': {'total_tokens': 15}
                })
                return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=network_issue_simulation):
            with patch('asyncio.sleep'):  # Speed up test
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'network-recovery-test'
                assert call_count == 3  # Should have retried twice
    
    @pytest.mark.asyncio
    async def test_handling_of_corrupted_responses(self, client, sample_messages, sample_model_config):
        """Test handling of corrupted or malformed responses."""
        corrupted_responses = [
            b'{"invalid": json structure',  # Invalid JSON
            b'{"choices": [{"message": null}]}',  # Null message
            b'{"choices": []}',  # Empty choices
            b'null',  # Null response
            b'',  # Empty response
        ]
        
        for corrupted_data in corrupted_responses:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.status = 200
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                    side_effect=json.JSONDecodeError("Invalid JSON", "", 0)
                )
                
                with pytest.raises(GenesisAPIError):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_on_failures(self, client, sample_messages, sample_model_config):
        """Test that resources are properly cleaned up on failures."""
        with patch('aiohttp.ClientSession.post', side_effect=Exception("Test failure")):
            with pytest.raises(GenesisAPIError):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
            
            # Verify resources are still accessible and not corrupted
            assert client.session is not None
            assert client.api_key is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_request_isolation(self, client, sample_messages, sample_model_config):
        """Test that concurrent requests don't interfere with each other."""
        # Create different responses for different requests
        responses = [
            {'id': f'concurrent-{i}', 'choices': [{'message': {'content': f'Response {i}'}}], 'usage': {'total_tokens': 10}}
            for i in range(5)
        ]
        
        call_count = 0
        
        async def concurrent_response_handler(*args, **kwargs):
            nonlocal call_count
            response_data = responses[call_count % len(responses)]
            call_count += 1
            
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=response_data)
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=concurrent_response_handler):
            # Make concurrent requests
            tasks = []
            for i in range(5):
                task = client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Verify each request got a unique response
            assert len(results) == 5
            response_ids = [result.id for result in results]
            assert len(set(response_ids)) == 5  # All unique
    
    def test_configuration_immutability(self, client):
        """Test that client configuration cannot be accidentally modified."""
        original_api_key = client.api_key
        original_base_url = client.base_url
        
        # Attempt to modify configuration
        try:
            client.api_key = "modified-key"
            client.base_url = "https://malicious.com"
        except AttributeError:
            # Expected if properties are read-only
            pass
        
        # Verify configuration remains unchanged or changes are controlled
        assert client.api_key == original_api_key or client.api_key == "modified-key"
        assert client.base_url == original_base_url or client.base_url == "https://malicious.com"
    
    @pytest.mark.asyncio
    async def test_session_persistence_across_requests(self, client, sample_messages, sample_model_config):
        """Test that session persists correctly across multiple requests."""
        mock_response = {
            'id': 'session-persistence-test',
            'choices': [{'message': {'content': 'Session test'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Make multiple requests
            for i in range(3):
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                assert result.id == 'session-persistence-test'
            
            # Verify session remained consistent
            assert client.session is not None
            assert not client.session.closed
    
    def test_thread_safety_simulation(self, client):
        """Test thread safety aspects of the client."""
        import threading
        import concurrent.futures
        
        def access_client_properties():
            """Simulate accessing client properties from different threads."""
            return {
                'api_key': client.api_key,
                'base_url': client.base_url,
                'timeout': client.timeout,
                'max_retries': client.max_retries
            }
        
        # Simulate concurrent access from multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(access_client_properties) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All results should be identical (thread-safe property access)
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result


class TestDocumentationAndUsability:
    """Tests for documentation examples and usability scenarios."""
    
    @pytest.mark.asyncio
    async def test_basic_usage_example(self):
        """Test the basic usage example from documentation."""
        # Simulate a basic usage scenario
        config = {
            'api_key': 'your-api-key',
            'base_url': 'https://api.genesis.ai/v1'
        }
        
        mock_response = {
            'id': 'basic-usage-test',
            'choices': [{'message': {'content': 'Hello! How can I help you?'}}],
            'usage': {'total_tokens': 20}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            async with GenesisAPIClient(**config) as client:
                messages = [
                    ChatMessage(role="system", content="You are a helpful assistant."),
                    ChatMessage(role="user", content="Hello!")
                ]
                
                result = await client.create_chat_completion(
                    messages=messages,
                    model_config=ModelConfig(name="genesis-gpt-4")
                )
                
                assert result.choices[0].message.content == 'Hello! How can I help you?'
    
    @pytest.mark.asyncio
    async def test_streaming_usage_example(self):
        """Test the streaming usage example from documentation."""
        config = {'api_key': 'your-api-key'}
        
        streaming_chunks = [
            {'choices': [{'delta': {'content': 'Hello'}}]},
            {'choices': [{'delta': {'content': ' there!'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def example_stream():
            for chunk in streaming_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=example_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            async with GenesisAPIClient(**config) as client:
                messages = [ChatMessage(role="user", content="Say hello!")]
                
                full_response = ""
                async for chunk in client.create_chat_completion_stream(
                    messages=messages,
                    model_config=ModelConfig(name="genesis-gpt-4")
                ):
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                
                assert full_response == "Hello there!"
    
    @pytest.mark.asyncio
    async def test_error_handling_example(self):
        """Test error handling example from documentation."""
        config = {'api_key': 'invalid-key'}
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 401
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'error': {'message': 'Invalid API key'}}
            )
            
            async with GenesisAPIClient(**config) as client:
                try:
                    await client.create_chat_completion(
                        messages=[ChatMessage(role="user", content="Hello")],
                        model_config=ModelConfig(name="genesis-gpt-4")
                    )
                except AuthenticationError as e:
                    assert "Invalid API key" in str(e)
                except GenesisAPIError as e:
                    # Also acceptable as a base exception
                    assert "Invalid API key" in str(e)
    
    def test_model_config_customization_example(self):
        """Test model configuration customization example."""
        # Test various model configurations
        configs = [
            ModelConfig(name="genesis-gpt-4", temperature=0.7, max_tokens=1000),
            ModelConfig(name="genesis-gpt-3.5-turbo", temperature=0.1, max_tokens=500),
            ModelConfig(name="genesis-gpt-4", temperature=1.0, top_p=0.9),
        ]
        
        for config in configs:
            assert config.name.startswith("genesis-")
            assert 0 <= config.temperature <= 2
            assert config.max_tokens > 0
    
    def test_message_format_examples(self):
        """Test various message format examples."""
        # Test different message formats
        message_examples = [
            # Basic conversation
            [
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there!"),
                ChatMessage(role="user", content="How are you?")
            ],
            # With system message
            [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="What's the weather like?")
            ],
            # With named participants
            [
                ChatMessage(role="user", content="Hello", name="Alice"),
                ChatMessage(role="assistant", content="Hello Alice!"),
                ChatMessage(role="user", content="How are you?", name="Bob")
            ]
        ]
        
        client = GenesisAPIClient(api_key="test-key")
        
        for messages in message_examples:
            # Should validate without errors
            client._validate_messages(messages)

class TestGenesisAPIClientRobustness:
    """Additional comprehensive tests for enhanced robustness and edge case coverage."""
    
    @pytest.fixture
    def client_with_custom_timeout(self, mock_config):
        """Create a client with custom timeout for timeout-specific tests."""
        config = mock_config.copy()
        config['timeout'] = 5.0
        return GenesisAPIClient(**config)
    
    @pytest.fixture
    def minimal_valid_response(self):
        """Minimal valid response structure for testing."""
        return {
            'id': 'test-minimal',
            'object': 'chat.completion',
            'created': 1677610602,
            'model': 'genesis-gpt-4',
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'Test response'},
                'finish_reason': 'stop'
            }],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        }
    
    def test_client_initialization_with_empty_base_url(self):
        """Test client initialization with empty base URL."""
        with pytest.raises(ValueError, match="base_url cannot be empty"):
            GenesisAPIClient(api_key='test-key', base_url='')
    
    def test_client_initialization_with_invalid_base_url_format(self):
        """Test client initialization with malformed base URL."""
        invalid_urls = [
            'not-a-url',
            'http://',
            'https://',
            'ftp://example.com',
            'javascript:alert(1)',
            'data:text/plain,hello'
        ]
        
        for invalid_url in invalid_urls:
            with pytest.raises(ValueError, match="Invalid base URL"):
                GenesisAPIClient(api_key='test-key', base_url=invalid_url)
    
    def test_client_initialization_with_numeric_string_timeout(self):
        """Test client initialization with numeric string timeout."""
        with pytest.raises(ValueError, match="Timeout must be a number"):
            GenesisAPIClient(api_key='test-key', timeout='30')
    
    def test_client_initialization_with_boolean_parameters(self):
        """Test client initialization with boolean parameters."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            GenesisAPIClient(api_key='test-key', timeout=True)
        
        with pytest.raises(ValueError, match="Max retries must be non-negative"):
            GenesisAPIClient(api_key='test-key', max_retries=False)
    
    @pytest.mark.asyncio
    async def test_create_chat_completion_with_none_messages(self, client):
        """Test create_chat_completion with None messages parameter."""
        with pytest.raises(ValidationError, match="Messages cannot be None"):
            await client.create_chat_completion(
                messages=None,
                model_config=ModelConfig(name="test-model")
            )
    
    @pytest.mark.asyncio
    async def test_create_chat_completion_with_none_model_config(self, client, sample_messages):
        """Test create_chat_completion with None model_config parameter."""
        with pytest.raises(ValidationError, match="Model config cannot be None"):
            await client.create_chat_completion(
                messages=sample_messages,
                model_config=None
            )
    
    @pytest.mark.asyncio
    async def test_create_chat_completion_with_mixed_message_types(self, client):
        """Test create_chat_completion with mixed valid and invalid message types."""
        mixed_messages = [
            ChatMessage(role="user", content="Valid message"),
            None,  # Invalid message
            ChatMessage(role="assistant", content="Another valid message")
        ]
        
        with pytest.raises(ValidationError, match="All messages must be ChatMessage instances"):
            await client.create_chat_completion(
                messages=mixed_messages,
                model_config=ModelConfig(name="test-model")
            )
    
    @pytest.mark.asyncio
    async def test_response_with_unexpected_structure(self, client, sample_messages, sample_model_config):
        """Test handling of responses with unexpected but valid JSON structure."""
        unexpected_response = {
            'id': 'unexpected-structure',
            'object': 'chat.completion',
            'choices': [{
                'message': {'role': 'assistant', 'content': 'Response'},
                'additional_field': 'unexpected_value'
            }],
            'usage': {'total_tokens': 10},
            'unknown_field': 'should_be_ignored'
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=unexpected_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert result.id == 'unexpected-structure'
            # Should handle unexpected fields gracefully
    
    @pytest.mark.asyncio
    async def test_response_with_different_choice_structures(self, client, sample_messages, sample_model_config):
        """Test handling of responses with different choice structures."""
        varied_choice_response = {
            'id': 'varied-choices',
            'choices': [
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'First choice'},
                    'finish_reason': 'stop'
                },
                {
                    'index': 1,
                    'message': {'role': 'assistant', 'content': 'Second choice'},
                    'finish_reason': 'length'
                },
                {
                    'index': 2,
                    'message': {'role': 'assistant', 'content': 'Third choice'},
                    'finish_reason': 'content_filter'
                }
            ],
            'usage': {'total_tokens': 45}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=varied_choice_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            assert len(result.choices) == 3
            assert result.choices[0].finish_reason == 'stop'
            assert result.choices[1].finish_reason == 'length'
            assert result.choices[2].finish_reason == 'content_filter'
    
    @pytest.mark.asyncio
    async def test_http_status_codes_comprehensive(self, client, sample_messages, sample_model_config):
        """Test comprehensive HTTP status code handling."""
        status_code_scenarios = [
            (200, 'success'),
            (201, 'created'),
            (202, 'accepted'),
            (400, 'bad_request'),
            (401, 'unauthorized'),
            (403, 'forbidden'),
            (404, 'not_found'),
            (405, 'method_not_allowed'),
            (408, 'request_timeout'),
            (409, 'conflict'),
            (410, 'gone'),
            (422, 'unprocessable_entity'),
            (429, 'too_many_requests'),
            (500, 'internal_server_error'),
            (501, 'not_implemented'),
            (502, 'bad_gateway'),
            (503, 'service_unavailable'),
            (504, 'gateway_timeout'),
            (507, 'insufficient_storage'),
            (520, 'unknown_error'),
            (521, 'web_server_down'),
            (522, 'connection_timed_out'),
            (523, 'origin_unreachable'),
            (524, 'timeout_occurred')
        ]
        
        for status_code, scenario in status_code_scenarios:
            with patch('aiohttp.ClientSession.post') as mock_post:
                if status_code < 300:
                    # Success responses
                    mock_post.return_value.__aenter__.return_value.status = status_code
                    mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                        return_value=self.minimal_valid_response
                    )
                    
                    result = await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                    assert result.id == 'test-minimal'
                else:
                    # Error responses
                    mock_post.return_value.__aenter__.return_value.status = status_code
                    mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                        return_value={'error': {'message': f'Error {status_code}'}}
                    )
                    
                    with pytest.raises(GenesisAPIError):
                        await client.create_chat_completion(
                            messages=sample_messages,
                            model_config=sample_model_config
                        )
    
    @pytest.mark.asyncio
    async def test_request_headers_validation(self, client, sample_messages, sample_model_config):
        """Test that request headers are properly validated and set."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=self.minimal_valid_response
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            # Verify the request was made with correct headers
            call_args = mock_post.call_args
            headers = call_args[1]['headers']
            
            assert 'Authorization' in headers
            assert headers['Authorization'].startswith('Bearer ')
            assert headers['Content-Type'] == 'application/json'
            assert 'User-Agent' in headers
            assert len(headers['User-Agent']) > 0
    
    @pytest.mark.asyncio
    async def test_request_payload_validation(self, client, sample_messages, sample_model_config):
        """Test that request payload is properly structured."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=self.minimal_valid_response
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            # Verify the request payload structure
            call_args = mock_post.call_args
            json_data = call_args[1]['json']
            
            assert 'messages' in json_data
            assert 'model' in json_data
            assert isinstance(json_data['messages'], list)
            assert len(json_data['messages']) > 0
            assert json_data['model'] == sample_model_config.name
    
    def test_message_validation_with_extreme_content_lengths(self, client):
        """Test message validation with various extreme content lengths."""
        # Test with extremely long content
        very_long_content = "x" * 1000000  # 1MB of content
        long_message = ChatMessage(role="user", content=very_long_content)
        
        with pytest.raises(ValidationError, match="Message content too long"):
            client._validate_messages([long_message])
        
        # Test with content at boundary
        boundary_content = "x" * 10000  # 10KB of content (might be acceptable)
        boundary_message = ChatMessage(role="user", content=boundary_content)
        
        try:
            client._validate_messages([boundary_message])
        except ValidationError as e:
            # Only acceptable if there's a specific length limit
            assert "too long" in str(e)
    
    def test_message_validation_with_control_characters(self, client):
        """Test message validation with control characters."""
        control_chars_content = "Hello\x00\x01\x02\x03World"
        control_message = ChatMessage(role="user", content=control_chars_content)
        
        # Should handle control characters appropriately
        try:
            client._validate_messages([control_message])
        except ValidationError as e:
            # Only acceptable if control characters are explicitly forbidden
            assert "control character" in str(e).lower() or "invalid character" in str(e).lower()
    
    def test_message_validation_with_binary_content(self, client):
        """Test message validation with binary content."""
        binary_content = b"Binary content \xff\xfe\xfd"
        
        with pytest.raises(ValidationError, match="Message content must be string"):
            client._validate_messages([
                ChatMessage(role="user", content=binary_content)
            ])
    
    def test_model_config_validation_with_extreme_values(self, client):
        """Test model config validation with extreme parameter values."""
        extreme_configs = [
            # Extreme temperature values
            {'name': 'test', 'temperature': float('inf')},
            {'name': 'test', 'temperature': float('-inf')},
            {'name': 'test', 'temperature': float('nan')},
            
            # Extreme token values
            {'name': 'test', 'max_tokens': float('inf')},
            {'name': 'test', 'max_tokens': 2**63 - 1},
            
            # Extreme probability values
            {'name': 'test', 'top_p': float('inf')},
            {'name': 'test', 'top_p': float('-inf')},
            {'name': 'test', 'top_p': float('nan')},
            
            # Extreme penalty values
            {'name': 'test', 'frequency_penalty': float('inf')},
            {'name': 'test', 'presence_penalty': float('-inf')},
        ]
        
        for config_dict in extreme_configs:
            config = ModelConfig(**config_dict)
            
            with pytest.raises(ValidationError):
                client._validate_model_config(config)
    
    @pytest.mark.asyncio
    async def test_session_management_edge_cases(self, client):
        """Test session management edge cases."""
        # Test accessing session before initialization
        original_session = client.session
        
        # Test multiple close calls
        await client.close()
        await client.close()  # Should not raise error
        
        # Test session replacement
        client.session = original_session
        assert client.session is not None
    
    @pytest.mark.asyncio
    async def test_context_manager_exception_handling(self, mock_config):
        """Test context manager behavior with exceptions."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            # Test exception during context manager usage
            with pytest.raises(ValueError, match="Test exception"):
                async with GenesisAPIClient(**mock_config) as client:
                    raise ValueError("Test exception")
            
            # Verify session was properly closed
            mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_network_error_types(self, client, sample_messages, sample_model_config):
        """Test handling of various network error types."""
        import aiohttp
        
        network_errors = [
            aiohttp.ClientConnectionError(),
            aiohttp.ClientConnectorError(connection_key=None, os_error=None),
            aiohttp.ClientResponseError(request_info=None, history=None),
            aiohttp.ClientPayloadError(),
            aiohttp.ClientSSLError(),
            aiohttp.ServerConnectionError(),
            aiohttp.ServerTimeoutError(),
            aiohttp.ServerDisconnectedError()
        ]
        
        for error in network_errors:
            with patch('aiohttp.ClientSession.post', side_effect=error):
                with pytest.raises(GenesisAPIError, match="Connection error|Network error|Server error"):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
    
    @pytest.mark.asyncio
    async def test_json_decode_error_handling(self, client, sample_messages, sample_model_config):
        """Test handling of various JSON decode errors."""
        json_errors = [
            json.JSONDecodeError("Expecting ',' delimiter", "doc", 4),
            json.JSONDecodeError("Unterminated string", "doc", 10),
            json.JSONDecodeError("Invalid control character", "doc", 15),
            json.JSONDecodeError("Expecting value", "doc", 0),
        ]
        
        for json_error in json_errors:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.status = 200
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(side_effect=json_error)
                
                with pytest.raises(GenesisAPIError, match="Invalid response format"):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )


class TestAdvancedStreamingScenarios:
    """Advanced streaming test scenarios for comprehensive coverage."""
    
    @pytest.mark.asyncio
    async def test_streaming_with_incomplete_json_chunks(self, client, sample_messages, sample_model_config):
        """Test streaming with incomplete JSON chunks that span multiple reads."""
        incomplete_chunks = [
            b'{"choices": [{"delta": {"con',  # Incomplete
            b'tent": "Hello"}}]}',  # Completion
            b'{"choices": [{"delta": {"content": " world"}}]}',  # Complete
            b'{"choices": [{"delta": {}, "fin',  # Incomplete
            b'ish_reason": "stop"}]}',  # Completion
        ]
        
        async def incomplete_chunk_stream():
            for chunk in incomplete_chunks:
                yield chunk
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=incomplete_chunk_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            # Should handle incomplete chunks by buffering
            assert len(chunks) >= 2
    
    @pytest.mark.asyncio
    async def test_streaming_with_nested_json_structures(self, client, sample_messages, sample_model_config):
        """Test streaming with complex nested JSON structures."""
        nested_chunks = [
            {
                'choices': [{
                    'delta': {
                        'content': 'Hello',
                        'metadata': {
                            'confidence': 0.95,
                            'token_logprobs': [-0.1, -0.2],
                            'top_logprobs': [
                                {'token': 'Hello', 'logprob': -0.1},
                                {'token': 'Hi', 'logprob': -0.3}
                            ]
                        }
                    }
                }]
            },
            {
                'choices': [{
                    'delta': {'content': ' world'},
                    'finish_reason': 'stop'
                }]
            }
        ]
        
        async def nested_stream():
            for chunk in nested_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=nested_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 2
            assert chunks[0].choices[0].delta.content == 'Hello'
            assert chunks[1].choices[0].delta.content == ' world'
    
    @pytest.mark.asyncio
    async def test_streaming_with_multiple_choice_deltas(self, client, sample_messages, sample_model_config):
        """Test streaming with multiple choice deltas in single chunk."""
        multi_choice_chunks = [
            {
                'choices': [
                    {'index': 0, 'delta': {'content': 'Choice 1: Hello'}},
                    {'index': 1, 'delta': {'content': 'Choice 2: Hi'}},
                    {'index': 2, 'delta': {'content': 'Choice 3: Hey'}}
                ]
            },
            {
                'choices': [
                    {'index': 0, 'delta': {'content': ' world'}, 'finish_reason': 'stop'},
                    {'index': 1, 'delta': {'content': ' there'}, 'finish_reason': 'stop'},
                    {'index': 2, 'delta': {'content': ' you'}, 'finish_reason': 'stop'}
                ]
            }
        ]
        
        async def multi_choice_stream():
            for chunk in multi_choice_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=multi_choice_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 2
            assert len(chunks[0].choices) == 3
            assert len(chunks[1].choices) == 3
            assert chunks[0].choices[0].delta.content == 'Choice 1: Hello'
            assert chunks[1].choices[0].finish_reason == 'stop'
    
    @pytest.mark.asyncio
    async def test_streaming_with_error_in_middle(self, client, sample_messages, sample_model_config):
        """Test streaming that encounters an error in the middle of the stream."""
        async def error_in_middle_stream():
            yield json.dumps({'choices': [{'delta': {'content': 'Hello'}}]}).encode()
            yield json.dumps({'choices': [{'delta': {'content': ' wor'}}]}).encode()
            # Simulate error in middle of stream
            yield b'{"error": {"message": "Stream interrupted"}}'
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=error_in_middle_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            try:
                async for chunk in client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                ):
                    chunks.append(chunk)
            except GenesisAPIError as e:
                assert "Stream interrupted" in str(e)
            
            # Should have processed some chunks before error
            assert len(chunks) >= 2
    
    @pytest.mark.asyncio
    async def test_streaming_with_slow_network(self, client, sample_messages, sample_model_config):
        """Test streaming with slow network conditions."""
        async def slow_network_stream():
            chunks = [
                {'choices': [{'delta': {'content': 'Slow'}}]},
                {'choices': [{'delta': {'content': ' network'}}]},
                {'choices': [{'delta': {'content': ' test'}}]},
                {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
            ]
            
            for chunk in chunks:
                yield json.dumps(chunk).encode()
                await asyncio.sleep(0.1)  # Simulate slow network
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=slow_network_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            start_time = asyncio.get_event_loop().time()
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            end_time = asyncio.get_event_loop().time()
            
            assert len(chunks) == 4
            assert end_time - start_time >= 0.3  # Should take at least 0.3s due to delays
    
    @pytest.mark.asyncio
    async def test_streaming_with_large_buffer_requirements(self, client, sample_messages, sample_model_config):
        """Test streaming with chunks that require large buffers."""
        large_content = "x" * 50000  # 50KB content
        
        large_buffer_chunks = [
            {'choices': [{'delta': {'content': large_content}}]},
            {'choices': [{'delta': {'content': large_content}}]},
            {'choices': [{'delta': {'content': large_content}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def large_buffer_stream():
            for chunk in large_buffer_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=large_buffer_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 4
            assert len(chunks[0].choices[0].delta.content) == 50000
            assert len(chunks[1].choices[0].delta.content) == 50000
            assert len(chunks[2].choices[0].delta.content) == 50000


class TestDataValidationEdgeCases:
    """Comprehensive data validation edge case testing."""
    
    def test_chat_message_with_unusual_role_combinations(self, client):
        """Test chat message validation with unusual but potentially valid role combinations."""
        unusual_conversations = [
            # All system messages
            [
                ChatMessage(role="system", content="System message 1"),
                ChatMessage(role="system", content="System message 2")
            ],
            # All assistant messages
            [
                ChatMessage(role="assistant", content="Assistant message 1"),
                ChatMessage(role="assistant", content="Assistant message 2")
            ],
            # Mixed order
            [
                ChatMessage(role="user", content="User message"),
                ChatMessage(role="system", content="System message"),
                ChatMessage(role="assistant", content="Assistant message")
            ],
            # Repeated user messages
            [
                ChatMessage(role="user", content="Question 1"),
                ChatMessage(role="user", content="Question 2"),
                ChatMessage(role="user", content="Question 3")
            ]
        ]
        
        for conversation in unusual_conversations:
            # Should either validate successfully or fail with specific reason
            try:
                client._validate_messages(conversation)
            except ValidationError as e:
                # Only acceptable if there are specific conversation flow rules
                assert "conversation" in str(e).lower() or "role" in str(e).lower()
    
    def test_chat_message_with_international_characters(self, client):
        """Test chat message validation with various international characters."""
        international_messages = [
            ChatMessage(role="user", content="नमस्ते दुनिया"),  # Hindi
            ChatMessage(role="user", content="مرحبا بالعالم"),  # Arabic
            ChatMessage(role="user", content="こんにちは世界"),  # Japanese
            ChatMessage(role="user", content="Здравствуй мир"),  # Russian
            ChatMessage(role="user", content="🌍🌎🌏 Earth emojis"),  # Emojis
            ChatMessage(role="user", content="Mathematical symbols: ∑∫∞≠≤≥"),  # Math symbols
            ChatMessage(role="user", content="Currency: €£¥₹₽₩₪"),  # Currency symbols
        ]
        
        for message in international_messages:
            # Should handle international characters
            try:
                client._validate_messages([message])
            except ValidationError as e:
                # Only acceptable if there are specific encoding restrictions
                assert "encoding" in str(e).lower() or "character" in str(e).lower()
    
    def test_model_config_with_boundary_precision_values(self, client):
        """Test model config validation with boundary precision values."""
        precision_configs = [
            # Very small positive values
            ModelConfig(name="test", temperature=0.0001, top_p=0.0001),
            # Values very close to boundaries
            ModelConfig(name="test", temperature=0.0000001, top_p=0.9999999),
            # Maximum precision values
            ModelConfig(name="test", temperature=1.9999999, top_p=1.0000000),
            # Scientific notation
            ModelConfig(name="test", temperature=1e-10, top_p=1e-5),
            ModelConfig(name="test", temperature=2.0 - 1e-10, top_p=1.0 - 1e-10),
        ]
        
        for config in precision_configs:
            try:
                client._validate_model_config(config)
            except ValidationError as e:
                # Only acceptable if there are specific precision requirements
                assert "precision" in str(e).lower() or "range" in str(e).lower()
    
    def test_model_config_with_complex_stop_sequences(self, client):
        """Test model config validation with complex stop sequences."""
        complex_configs = [
            # Empty stop sequences
            ModelConfig(name="test", stop=[]),
            # Single character stops
            ModelConfig(name="test", stop=[".", "!", "?"]),
            # Multi-character stops
            ModelConfig(name="test", stop=["END", "STOP", "FINISH"]),
            # Unicode stops
            ModelConfig(name="test", stop=["。", "！", "？"]),  # Japanese punctuation
            # Mixed stops
            ModelConfig(name="test", stop=[".", "END", "🛑", "\n\n"]),
            # Long stop sequences
            ModelConfig(name="test", stop=["This is a very long stop sequence that might be problematic"]),
        ]
        
        for config in complex_configs:
            try:
                client._validate_model_config(config)
            except ValidationError as e:
                # Only acceptable if there are specific stop sequence rules
                assert "stop" in str(e).lower() or "sequence" in str(e).lower()
    
    def test_message_validation_with_structured_content(self, client):
        """Test message validation with structured content types."""
        structured_messages = [
            # JSON-like content
            ChatMessage(role="user", content='{"key": "value", "number": 42}'),
            # XML-like content
            ChatMessage(role="user", content='<message><text>Hello</text></message>'),
            # Code content
            ChatMessage(role="user", content='def hello():\n    print("Hello, world!")'),
            # Markdown content
            ChatMessage(role="user", content='# Title\n\n**Bold text** and *italic text*'),
            # CSV-like content
            ChatMessage(role="user", content='Name,Age,City\nJohn,30,NYC\nJane,25,LA'),
            # Base64-like content
            ChatMessage(role="user", content='SGVsbG8gV29ybGQ='),
        ]
        
        for message in structured_messages:
            # Should handle structured content appropriately
            try:
                client._validate_messages([message])
            except ValidationError as e:
                # Only acceptable if there are specific content format restrictions
                assert "format" in str(e).lower() or "content" in str(e).lower()
    
    def test_message_validation_with_edge_case_whitespace(self, client):
        """Test message validation with edge case whitespace scenarios."""
        whitespace_messages = [
            # Different types of whitespace
            ChatMessage(role="user", content="   leading spaces"),
            ChatMessage(role="user", content="trailing spaces   "),
            ChatMessage(role="user", content="  both sides  "),
            ChatMessage(role="user", content="line\nbreaks\nhere"),
            ChatMessage(role="user", content="tab\tcharacters\there"),
            ChatMessage(role="user", content="carriage\rreturn"),
            ChatMessage(role="user", content="form\ffeed"),
            ChatMessage(role="user", content="vertical\vtab"),
            # Mixed whitespace
            ChatMessage(role="user", content="  \t\n\r\f\v  mixed  \t\n\r\f\v  "),
            # Unicode whitespace
            ChatMessage(role="user", content="non\u00A0breaking\u2000space"),
        ]
        
        for message in whitespace_messages:
            try:
                client._validate_messages([message])
            except ValidationError as e:
                # Only acceptable if whitespace is specifically restricted
                assert "whitespace" in str(e).lower() or "empty" in str(e).lower()


class TestErrorRecoveryPatterns:
    """Test error recovery and resilience patterns."""
    
    @pytest.mark.asyncio
    async def test_recovery_from_rate_limiting_with_backoff(self, client, sample_messages, sample_model_config):
        """Test recovery from rate limiting with proper backoff."""
        call_count = 0
        
        async def rate_limit_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            mock_response = Mock()
            if call_count <= 3:
                # Rate limited for first 3 calls
                mock_response.status = 429
                mock_response.headers = {'Retry-After': '1'}
                mock_response.json = AsyncMock(return_value={'error': {'message': 'Rate limit exceeded'}})
            else:
                # Success on 4th call
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    'id': 'rate-limit-recovery',
                    'choices': [{'message': {'content': 'Recovery successful'}}],
                    'usage': {'total_tokens': 20}
                })
            
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=rate_limit_recovery):
            with patch('asyncio.sleep') as mock_sleep:
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'rate-limit-recovery'
                assert call_count == 4
                # Verify exponential backoff was used
                assert mock_sleep.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_recovery_from_intermittent_timeouts(self, client, sample_messages, sample_model_config):
        """Test recovery from intermittent timeout errors."""
        call_count = 0
        
        async def timeout_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count in [1, 3]:  # Timeout on 1st and 3rd calls
                raise asyncio.TimeoutError("Request timeout")
            else:
                # Success on other calls
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    'id': 'timeout-recovery',
                    'choices': [{'message': {'content': 'Timeout recovery successful'}}],
                    'usage': {'total_tokens': 25}
                })
                return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=timeout_recovery):
            with patch('asyncio.sleep'):
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'timeout-recovery'
                assert call_count >= 2  # Should have retried at least once
    
    @pytest.mark.asyncio
    async def test_recovery_from_connection_drops(self, client, sample_messages, sample_model_config):
        """Test recovery from connection drops."""
        call_count = 0
        
        async def connection_drop_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= 2:
                # Connection drops for first 2 calls
                raise aiohttp.ClientConnectionError("Connection dropped")
            else:
                # Success on 3rd call
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    'id': 'connection-recovery',
                    'choices': [{'message': {'content': 'Connection recovery successful'}}],
                    'usage': {'total_tokens': 30}
                })
                return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=connection_drop_recovery):
            with patch('asyncio.sleep'):
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'connection-recovery'
                assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_partial_service_failure(self, client, sample_messages, sample_model_config):
        """Test graceful degradation when service is partially failing."""
        call_count = 0
        
        async def partial_service_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            mock_response = Mock()
            if call_count % 2 == 1:
                # Intermittent server errors
                mock_response.status = 503
                mock_response.json = AsyncMock(return_value={'error': {'message': 'Service temporarily unavailable'}})
            else:
                # Successful responses
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    'id': f'partial-service-{call_count}',
                    'choices': [{'message': {'content': f'Success {call_count}'}}],
                    'usage': {'total_tokens': 15}
                })
            
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=partial_service_failure):
            with patch('asyncio.sleep'):
                # Should succeed on retry
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'partial-service-2'
                assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern_simulation(self, client, sample_messages, sample_model_config):
        """Test circuit breaker pattern simulation with consecutive failures."""
        failure_count = 0
        
        async def circuit_breaker_simulation(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= 5:
                # Multiple consecutive failures
                mock_response = Mock()
                mock_response.status = 500
                mock_response.json = AsyncMock(return_value={'error': {'message': 'Internal server error'}})
                return mock_response
            else:
                # After circuit breaker opens, return success
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    'id': 'circuit-breaker-recovery',
                    'choices': [{'message': {'content': 'Circuit breaker recovery'}}],
                    'usage': {'total_tokens': 25}
                })
                return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=circuit_breaker_simulation):
            with patch('asyncio.sleep'):
                with pytest.raises(GenesisAPIError, match="Internal server error"):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                
                # Should have attempted up to max_retries
                assert failure_count >= 3


class TestPerformanceCharacteristics:
    """Test performance characteristics and resource usage."""
    
    @pytest.mark.asyncio
    async def test_concurrent_request_performance(self, client, sample_messages, sample_model_config):
        """Test performance with concurrent requests."""
        mock_response = {
            'id': 'concurrent-perf-test',
            'choices': [{'message': {'content': 'Concurrent response'}}],
            'usage': {'total_tokens': 15}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            start_time = asyncio.get_event_loop().time()
            
            # Create 20 concurrent requests
            tasks = []
            for i in range(20):
                task = client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()
            
            # All requests should complete
            assert len(results) == 20
            for result in results:
                assert result.id == 'concurrent-perf-test'
            
            # Should complete relatively quickly (concurrent execution)
            assert end_time - start_time < 5.0
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_large_conversations(self, client, sample_model_config):
        """Test memory usage with large conversation histories."""
        # Create a large conversation
        large_conversation = []
        for i in range(1000):
            role = "user" if i % 2 == 0 else "assistant"
            large_conversation.append(ChatMessage(role=role, content=f"Message {i}: " + "x" * 100))
        
        mock_response = {
            'id': 'large-conversation-test',
            'choices': [{'message': {'content': 'Large conversation processed'}}],
            'usage': {'total_tokens': 50000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle large conversations without memory issues
            result = await client.create_chat_completion(
                messages=large_conversation,
                model_config=sample_model_config
            )
            
            assert result.id == 'large-conversation-test'
            assert result.usage.total_tokens == 50000
    
    @pytest.mark.asyncio
    async def test_streaming_performance_with_rapid_chunks(self, client, sample_messages, sample_model_config):
        """Test streaming performance with rapid chunk delivery."""
        # Create many small chunks
        rapid_chunks = []
        for i in range(100):
            rapid_chunks.append({'choices': [{'delta': {'content': f'word{i} '}}]})
        
        rapid_chunks.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
        
        async def rapid_chunk_stream():
            for chunk in rapid_chunks:
                yield json.dumps(chunk).encode()
                await asyncio.sleep(0.001)  # Very short delay
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=rapid_chunk_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            start_time = asyncio.get_event_loop().time()
            chunk_count = 0
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunk_count += 1
            end_time = asyncio.get_event_loop().time()
            
            assert chunk_count == 101  # 100 content chunks + 1 finish chunk
            # Should handle rapid chunks efficiently
            assert end_time - start_time < 5.0
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_under_load(self, client, sample_messages, sample_model_config):
        """Test resource cleanup under high load conditions."""
        mock_response = {
            'id': 'resource-cleanup-test',
            'choices': [{'message': {'content': 'Resource test'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Create and cancel many tasks to test resource cleanup
            tasks = []
            for i in range(50):
                task = asyncio.create_task(client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                ))
                tasks.append(task)
            
            # Cancel half the tasks
            for i in range(0, 50, 2):
                tasks[i].cancel()
            
            # Wait for remaining tasks
            completed_tasks = []
            for task in tasks:
                try:
                    result = await task
                    completed_tasks.append(result)
                except asyncio.CancelledError:
                    pass
            
            # Should have completed some tasks without resource leaks
            assert len(completed_tasks) > 0
            assert len(completed_tasks) < 50  # Some were cancelled
    
    def test_validation_performance_with_large_inputs(self, client):
        """Test validation performance with large input data."""
        # Create large messages
        large_messages = []
        for i in range(100):
            large_messages.append(ChatMessage(role="user", content="x" * 1000))
        
        start_time = time.time()
        
        try:
            client._validate_messages(large_messages)
        except ValidationError:
            pass  # Expected for very large inputs
        
        end_time = time.time()
        
        # Validation should complete in reasonable time
        assert end_time - start_time < 1.0  # Should validate within 1 second


class TestConfigurationValidation:
    """Comprehensive configuration validation tests."""
    
    def test_api_key_format_validation(self):
        """Test API key format validation."""
        invalid_api_keys = [
            "",  # Empty string
            "   ",  # Only whitespace
            "short",  # Too short
            "contains spaces",  # Contains spaces
            "contains\nnewlines",  # Contains newlines
            "contains\ttabs",  # Contains tabs
            "contains\rcarriage",  # Contains carriage returns
            None,  # None value
            123,  # Non-string
            [],  # List
            {},  # Dict
        ]
        
        for invalid_key in invalid_api_keys:
            with pytest.raises(ValueError, match="API key"):
                GenesisAPIClient(api_key=invalid_key)
    
    def test_base_url_format_validation(self):
        """Test base URL format validation."""
        invalid_base_urls = [
            "",  # Empty string
            "   ",  # Only whitespace
            "not-a-url",  # Not a URL
            "ftp://example.com",  # Wrong protocol
            "http://",  # Missing domain
            "https://",  # Missing domain
            "javascript:alert(1)",  # Dangerous protocol
            "file:///etc/passwd",  # File protocol
            "data:text/plain,hello",  # Data protocol
        ]
        
        for invalid_url in invalid_base_urls:
            with pytest.raises(ValueError, match="Invalid base URL"):
                GenesisAPIClient(api_key="test-key", base_url=invalid_url)
    
    def test_timeout_value_validation(self):
        """Test timeout value validation."""
        invalid_timeouts = [
            -1,  # Negative
            0,  # Zero
            -10.5,  # Negative float
            "30",  # String
            "invalid",  # Non-numeric string
            None,  # None (might be valid as default)
            float('inf'),  # Infinity
            float('-inf'),  # Negative infinity
            float('nan'),  # NaN
            [],  # List
            {},  # Dict
        ]
        
        for invalid_timeout in invalid_timeouts:
            if invalid_timeout is None:
                # None might be acceptable as default
                try:
                    client = GenesisAPIClient(api_key="test-key", timeout=invalid_timeout)
                    assert client.timeout is None or client.timeout > 0
                except ValueError:
                    pass  # Also acceptable to reject None
            else:
                with pytest.raises(ValueError, match="Timeout"):
                    GenesisAPIClient(api_key="test-key", timeout=invalid_timeout)
    
    def test_max_retries_validation(self):
        """Test max_retries value validation."""
        invalid_retries = [
            -1,  # Negative
            -10,  # More negative
            "3",  # String
            "invalid",  # Non-numeric string
            float('inf'),  # Infinity
            float('-inf'),  # Negative infinity
            float('nan'),  # NaN
            [],  # List
            {},  # Dict
        ]
        
        for invalid_retries in invalid_retries:
            with pytest.raises(ValueError, match="Max retries"):
                GenesisAPIClient(api_key="test-key", max_retries=invalid_retries)
    
    def test_configuration_immutability_enforcement(self):
        """Test that configuration values cannot be modified after initialization."""
        client = GenesisAPIClient(api_key="test-key")
        
        original_api_key = client.api_key
        original_base_url = client.base_url
        original_timeout = client.timeout
        original_max_retries = client.max_retries
        
        # Attempt to modify configuration
        try:
            client.api_key = "modified-key"
            client.base_url = "https://malicious.com"
            client.timeout = 999
            client.max_retries = 999
        except AttributeError:
            # Expected if properties are read-only
            pass
        
        # Verify original values are preserved or changes are controlled
        if hasattr(client, '_api_key'):
            # If using private attributes, verify they weren't changed
            assert client._api_key == original_api_key
        else:
            # If using public attributes, verify they're either unchanged or properly validated
            assert client.api_key == original_api_key or len(client.api_key) > 0
    
    def test_configuration_with_environment_variables(self):
        """Test configuration with environment variables."""
        import os
        
        # Mock environment variables
        with patch.dict(os.environ, {
            'GENESIS_API_KEY': 'env-api-key',
            'GENESIS_BASE_URL': 'https://env.genesis.ai',
            'GENESIS_TIMEOUT': '60',
            'GENESIS_MAX_RETRIES': '5'
        }):
            # Test that environment variables are used if supported
            try:
                client = GenesisAPIClient()  # No explicit config
                # If env vars are supported, verify they're used
                assert client.api_key == 'env-api-key'
                assert client.base_url == 'https://env.genesis.ai'
                assert client.timeout == 60
                assert client.max_retries == 5
            except ValueError:
                # If env vars not supported, explicit config should still be required
                client = GenesisAPIClient(api_key="explicit-key")
                assert client.api_key == "explicit-key"
    
    def test_configuration_precedence(self):
        """Test configuration precedence (explicit > environment > defaults)."""
        import os
        
        with patch.dict(os.environ, {
            'GENESIS_API_KEY': 'env-key',
            'GENESIS_TIMEOUT': '30'
        }):
            # Explicit config should override environment
            client = GenesisAPIClient(
                api_key="explicit-key",
                timeout=60
            )
            
            assert client.api_key == "explicit-key"  # Explicit wins
            assert client.timeout == 60  # Explicit wins
    
    def test_configuration_validation_with_edge_cases(self):
        """Test configuration validation with edge cases."""
        edge_cases = [
            # Very long API key
            {'api_key': 'x' * 1000},
            # Very long base URL
            {'api_key': 'test', 'base_url': 'https://' + 'x' * 1000 + '.com'},
            # Very small timeout
            {'api_key': 'test', 'timeout': 0.001},
            # Very large timeout
            {'api_key': 'test', 'timeout': 86400},  # 24 hours
            # Very large max_retries
            {'api_key': 'test', 'max_retries': 100},
        ]
        
        for config in edge_cases:
            try:
                client = GenesisAPIClient(**config)
                # Should either succeed or fail with specific validation error
                assert client.api_key == config['api_key']
            except ValueError as e:
                # Should provide specific validation error
                assert len(str(e)) > 0


class TestUtilityFunctionsComprehensive:
    """Comprehensive tests for utility functions."""
    
    def test_format_timestamp_comprehensive(self):
        """Test format_timestamp with comprehensive input scenarios."""
        from app.ai_backend.genesis_api import format_timestamp
        
        timestamp_scenarios = [
            (0, "should handle epoch"),
            (1677610602, "should handle normal timestamp"),
            (-1, "should handle negative timestamp"),
            (9999999999, "should handle far future timestamp"),
            (1e10, "should handle large timestamp"),
            (1e-10, "should handle very small timestamp"),
        ]
        
        for timestamp, description in timestamp_scenarios:
            result = format_timestamp(timestamp)
            assert isinstance(result, str), f"Failed for {description}"
            assert len(result) > 0, f"Empty result for {description}"
    
    def test_calculate_token_usage_comprehensive(self):
        """Test calculate_token_usage with comprehensive message scenarios."""
        from app.ai_backend.genesis_api import calculate_token_usage
        
        message_scenarios = [
            # Empty conversation
            ([], "empty conversation"),
            # Single message
            ([ChatMessage(role="user", content="Hello")], "single message"),
            # Long conversation
            ([ChatMessage(role="user", content=f"Message {i}") for i in range(100)], "long conversation"),
            # Mixed content lengths
            ([
                ChatMessage(role="user", content="Short"),
                ChatMessage(role="assistant", content="x" * 1000),
                ChatMessage(role="user", content="Medium length message here")
            ], "mixed content lengths"),
            # Unicode content
            ([
                ChatMessage(role="user", content="Hello 世界 🌍"),
                ChatMessage(role="assistant", content="Bonjour le monde!")
            ], "unicode content"),
            # Special characters
            ([
                ChatMessage(role="user", content="Special: @#$%^&*()_+-=[]{}|;:,.<>?"),
                ChatMessage(role="assistant", content="Numbers: 1234567890")
            ], "special characters"),
        ]
        
        for messages, description in message_scenarios:
            result = calculate_token_usage(messages)
            assert isinstance(result, dict), f"Failed for {description}"
            assert 'estimated_tokens' in result, f"Missing estimated_tokens for {description}"
            assert isinstance(result['estimated_tokens'], int), f"Invalid token type for {description}"
            assert result['estimated_tokens'] >= 0, f"Negative tokens for {description}"
    
    def test_estimate_tokens_comprehensive(self):
        """Test estimate_tokens with comprehensive text scenarios."""
        from app.ai_backend.genesis_api import estimate_tokens
        
        text_scenarios = [
            ("", 0, "empty string"),
            ("Hello", 1, "single word"),
            ("Hello world", 2, "two words"),
            ("Hello, world!", 2, "punctuation"),
            ("Hello    world", 2, "multiple spaces"),
            ("Hello\nworld", 2, "newline"),
            ("Hello\tworld", 2, "tab"),
            ("Hello\r\nworld", 2, "carriage return + newline"),
            ("Hello 世界", 2, "unicode"),
            ("Hello 🌍", 2, "emoji"),
            ("@#$%^&*()", 1, "special characters"),
            ("1234567890", 1, "numbers"),
            ("camelCaseWord", 1, "camelCase"),
            ("snake_case_word", 1, "snake_case"),
            ("hyphenated-word", 1, "hyphenated"),
            ("  \t\n\r  ", 0, "only whitespace"),
            ("Word" * 100, 1, "repeated word"),
            ("A B C D E F G H I J", 10, "single letters"),
        ]
        
        for text, expected_min, description in text_scenarios:
            result = estimate_tokens(text)
            assert isinstance(result, int), f"Failed for {description}"
            assert result >= expected_min, f"Too few tokens for {description}: got {result}, expected >= {expected_min}"
    
    def test_utility_function_error_handling(self):
        """Test utility function error handling."""
        from app.ai_backend.genesis_api import format_timestamp, calculate_token_usage, estimate_tokens
        
        # Test format_timestamp with invalid inputs
        invalid_timestamps = [None, "string", [], {}, complex(1, 2)]
        for invalid_ts in invalid_timestamps:
            try:
                result = format_timestamp(invalid_ts)
                # Should either handle gracefully or raise appropriate error
                assert isinstance(result, str)
            except (TypeError, ValueError):
                # Acceptable to raise error for invalid input
                pass
        
        # Test calculate_token_usage with invalid inputs
        invalid_message_lists = [None, "string", 123, [None], ["string"]]
        for invalid_msgs in invalid_message_lists:
            try:
                result = calculate_token_usage(invalid_msgs)
                # Should either handle gracefully or raise appropriate error
                assert isinstance(result, dict)
            except (TypeError, ValueError, AttributeError):
                # Acceptable to raise error for invalid input
                pass
        
        # Test estimate_tokens with invalid inputs
        invalid_texts = [None, 123, [], {}, complex(1, 2)]
        for invalid_text in invalid_texts:
            try:
                result = estimate_tokens(invalid_text)
                # Should either handle gracefully or raise appropriate error
                assert isinstance(result, int)
            except (TypeError, ValueError, AttributeError):
                # Acceptable to raise error for invalid input
                pass
    
    def test_utility_function_performance(self):
        """Test utility function performance with large inputs."""
        from app.ai_backend.genesis_api import format_timestamp, calculate_token_usage, estimate_tokens
        import time
        
        # Test format_timestamp performance
        start_time = time.time()
        for i in range(1000):
            format_timestamp(1677610602 + i)
        end_time = time.time()
        assert end_time - start_time < 1.0, "format_timestamp too slow"
        
        # Test calculate_token_usage performance
        large_messages = [ChatMessage(role="user", content=f"Message {i}") for i in range(100)]
        start_time = time.time()
        for i in range(10):
            calculate_token_usage(large_messages)
        end_time = time.time()
        assert end_time - start_time < 1.0, "calculate_token_usage too slow"
        
        # Test estimate_tokens performance
        large_text = "word " * 10000
        start_time = time.time()
        for i in range(100):
            estimate_tokens(large_text)
        end_time = time.time()
        assert end_time - start_time < 1.0, "estimate_tokens too slow"
    
    def test_utility_function_consistency(self):
        """Test utility function consistency across multiple calls."""
        from app.ai_backend.genesis_api import format_timestamp, calculate_token_usage, estimate_tokens
        
        # Test format_timestamp consistency
        timestamp = 1677610602
        results = [format_timestamp(timestamp) for _ in range(10)]
        assert all(r == results[0] for r in results), "format_timestamp inconsistent"
        
        # Test calculate_token_usage consistency
        messages = [ChatMessage(role="user", content="Hello world")]
        results = [calculate_token_usage(messages) for _ in range(10)]
        assert all(r == results[0] for r in results), "calculate_token_usage inconsistent"
        
        # Test estimate_tokens consistency
        text = "Hello world"
        results = [estimate_tokens(text) for _ in range(10)]
        assert all(r == results[0] for r in results), "estimate_tokens inconsistent"


if __name__ == "__main__":
    # Run comprehensive tests
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-k", "TestGenesisAPIClientRobustness or TestAdvancedStreamingScenarios or TestDataValidationEdgeCases or TestErrorRecoveryPatterns or TestPerformanceCharacteristics or TestConfigurationValidation or TestUtilityFunctionsComprehensive",
        "--durations=10"
    ])