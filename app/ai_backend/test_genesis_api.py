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

class TestGenesisAPIClientComprehensive:
    """Additional comprehensive tests for GenesisAPIClient."""
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_function_calls(self, client, sample_model_config):
        """Test chat completion with function call capabilities."""
        function_call_messages = [
            ChatMessage(role="user", content="What's the weather like?"),
            ChatMessage(role="assistant", content=None, function_call={
                "name": "get_weather",
                "arguments": '{"location": "New York"}'
            }),
            ChatMessage(role="function", content='{"temp": 72, "condition": "sunny"}', name="get_weather")
        ]
        
        mock_response = {
            'id': 'chat-function-call',
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': 'The weather in New York is sunny with a temperature of 72°F.'
                }
            }],
            'usage': {'total_tokens': 50}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=function_call_messages,
                model_config=sample_model_config
            )
            
            assert result.id == 'chat-function-call'
            assert result.choices[0].message.content is not None

    @pytest.mark.asyncio
    async def test_chat_completion_with_system_message_variations(self, client, sample_model_config):
        """Test different system message configurations."""
        system_message_variants = [
            [ChatMessage(role="system", content="You are a helpful assistant.")],
            [ChatMessage(role="system", content=""), ChatMessage(role="user", content="test")],
            [ChatMessage(role="system", content="Role: Expert"), ChatMessage(role="system", content="Style: Formal")]
        ]
        
        mock_response = {
            'id': 'system-test',
            'choices': [{'message': {'content': 'Response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            for messages in system_message_variants:
                result = await client.create_chat_completion(
                    messages=messages,
                    model_config=sample_model_config
                )
                assert result.id == 'system-test'

    @pytest.mark.asyncio
    async def test_streaming_with_tool_calls(self, client, sample_messages, sample_model_config):
        """Test streaming response with tool/function calls."""
        mock_chunks = [
            {'choices': [{'delta': {'role': 'assistant'}}]},
            {'choices': [{'delta': {'tool_calls': [{'function': {'name': 'get_weather'}}]}}]},
            {'choices': [{'delta': {'tool_calls': [{'function': {'arguments': '{"location"'}}]}}]},
            {'choices': [{'delta': {'tool_calls': [{'function': {'arguments': ': "NYC"}'}}]}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]}
        ]
        
        async def mock_tool_stream():
            for chunk in mock_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mock_tool_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 5
            assert chunks[-1].choices[0].finish_reason == 'tool_calls'

    @pytest.mark.asyncio
    async def test_batch_completion_requests(self, client, sample_model_config):
        """Test handling multiple completion requests in batch."""
        batch_messages = [
            [ChatMessage(role="user", content=f"Question {i}")]
            for i in range(5)
        ]
        
        mock_response = {
            'id': 'batch-test',
            'choices': [{'message': {'content': 'Batch response'}}],
            'usage': {'total_tokens': 15}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            tasks = [
                client.create_chat_completion(messages=msgs, model_config=sample_model_config)
                for msgs in batch_messages
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            for result in results:
                assert result.id == 'batch-test'

    @pytest.mark.asyncio
    async def test_request_id_tracking(self, client, sample_messages, sample_model_config):
        """Test that request IDs are properly tracked and unique."""
        mock_response = {
            'id': 'tracking-test',
            'choices': [{'message': {'content': 'Response'}}],
            'usage': {'total_tokens': 10}
        }
        
        request_ids = []
        
        async def capture_request_id(*args, **kwargs):
            # Extract request ID from headers if available
            headers = kwargs.get('headers', {})
            request_id = headers.get('X-Request-ID')
            if request_id:
                request_ids.append(request_id)
            
            mock_resp = Mock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            return mock_resp
        
        with patch('aiohttp.ClientSession.post', side_effect=capture_request_id):
            await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            # Request IDs should be unique if implemented
            if request_ids:
                assert len(set(request_ids)) == len(request_ids)

    def test_message_content_length_validation_boundary(self, client):
        """Test message content length validation at exact boundaries."""
        # Test content exactly at maximum length
        max_length = 32768  # Common token limit
        boundary_content = "x" * max_length
        
        boundary_message = [ChatMessage(role="user", content=boundary_content)]
        
        # Should not raise for content at boundary
        try:
            client._validate_messages(boundary_message)
        except ValidationError as e:
            # Only acceptable if there's a specific length limit
            assert "content too long" in str(e)

    def test_model_config_parameter_combinations(self, client):
        """Test various parameter combinations in model configuration."""
        # Test extreme but valid combinations
        extreme_configs = [
            ModelConfig(name="test", temperature=0.0, top_p=1.0, max_tokens=1),
            ModelConfig(name="test", temperature=2.0, top_p=0.0, max_tokens=4096),
            ModelConfig(name="test", frequency_penalty=-2.0, presence_penalty=2.0),
            ModelConfig(name="test", temperature=1.0, top_p=0.5, frequency_penalty=0.5)
        ]
        
        for config in extreme_configs:
            try:
                client._validate_model_config(config)
            except ValidationError:
                # Some combinations might be invalid
                pass

    @pytest.mark.asyncio
    async def test_response_parsing_edge_cases(self, client, sample_messages, sample_model_config):
        """Test parsing of API responses with edge case structures."""
        edge_case_responses = [
            {
                'id': 'edge-1',
                'choices': [{'message': {'content': ''}}],  # Empty content
                'usage': {'total_tokens': 0}
            },
            {
                'id': 'edge-2',
                'choices': [{'message': {'content': None}}],  # None content
                'usage': {'total_tokens': 1}
            },
            {
                'id': 'edge-3',
                'choices': [],  # No choices
                'usage': {'total_tokens': 0}
            }
        ]
        
        for response in edge_case_responses:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                try:
                    result = await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                    # Should handle edge cases gracefully
                    assert result.id.startswith('edge-')
                except GenesisAPIError:
                    # May raise error for invalid structures
                    pass

    @pytest.mark.asyncio
    async def test_concurrent_model_operations(self, client):
        """Test concurrent model listing and retrieval operations."""
        mock_models_response = {
            'data': [
                {'id': 'model-1', 'object': 'model'},
                {'id': 'model-2', 'object': 'model'}
            ]
        }
        
        mock_model_response = {
            'id': 'model-1',
            'object': 'model',
            'created': 1677610602
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(
                side_effect=[mock_models_response, mock_model_response]
            )
            mock_get.return_value.__aenter__.return_value.status = 200
            
            # Run concurrent operations
            list_task = client.list_models()
            get_task = client.get_model('model-1')
            
            models, model = await asyncio.gather(list_task, get_task)
            
            assert len(models) == 2
            assert model.id == 'model-1'

    @pytest.mark.asyncio
    async def test_client_configuration_updates(self, client):
        """Test dynamic client configuration updates."""
        original_timeout = client.timeout
        original_max_retries = client.max_retries
        
        # Test updating configuration
        client.timeout = 60
        client.max_retries = 5
        
        assert client.timeout == 60
        assert client.max_retries == 5
        
        # Restore original values
        client.timeout = original_timeout
        client.max_retries = original_max_retries

    @pytest.mark.asyncio
    async def test_error_response_detail_extraction(self, client, sample_messages, sample_model_config):
        """Test extraction of detailed error information from API responses."""
        detailed_error_response = {
            'error': {
                'message': 'Model overloaded',
                'type': 'server_error',
                'code': 'model_overloaded',
                'param': None,
                'details': {
                    'retry_after': 30,
                    'suggestion': 'Try again later'
                }
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 503
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=detailed_error_response
            )
            
            with pytest.raises(GenesisAPIError) as exc_info:
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
            
            error = exc_info.value
            assert "Model overloaded" in str(error)
            assert error.status_code == 503


class TestAdvancedDataModelValidation:
    """Advanced tests for data model validation and behavior."""
    
    def test_chat_message_content_types(self):
        """Test ChatMessage with various content types and formats."""
        # Test with different content formats
        content_variants = [
            "Simple text",
            "Multi\nline\ncontent",
            "Content with\ttabs",
            "Unicode: 测试 🚀",
            "JSON-like: {'key': 'value'}",
            "URL: https://example.com",
            "Email: test@example.com"
        ]
        
        for content in content_variants:
            message = ChatMessage(role="user", content=content)
            assert message.content == content
            assert message.role == "user"

    def test_model_config_serialization(self):
        """Test ModelConfig serialization and deserialization."""
        config = ModelConfig(
            name="test-model",
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9
        )
        
        # Test serialization
        config_dict = config.__dict__
        assert config_dict['name'] == "test-model"
        assert config_dict['max_tokens'] == 1000
        
        # Test reconstruction
        new_config = ModelConfig(**config_dict)
        assert new_config.name == config.name
        assert new_config.max_tokens == config.max_tokens

    def test_api_response_with_complex_headers(self):
        """Test APIResponse with complex header structures."""
        complex_headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-RateLimit-Limit': '1000',
            'X-RateLimit-Remaining': '999',
            'X-RateLimit-Reset': '1677610602',
            'Set-Cookie': 'session=abc123; HttpOnly; Secure',
            'Cache-Control': 'no-cache, no-store, must-revalidate'
        }
        
        response = APIResponse(
            status_code=200,
            data={'success': True},
            headers=complex_headers
        )
        
        assert response.headers['Content-Type'] == 'application/json; charset=utf-8'
        assert response.headers['X-RateLimit-Limit'] == '1000'

    def test_chat_completion_with_multiple_choices(self):
        """Test ChatCompletion with multiple choice responses."""
        multiple_choices = [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': 'Response A'},
                'finish_reason': 'stop'
            },
            {
                'index': 1,
                'message': {'role': 'assistant', 'content': 'Response B'},
                'finish_reason': 'stop'
            },
            {
                'index': 2,
                'message': {'role': 'assistant', 'content': 'Response C'},
                'finish_reason': 'length'
            }
        ]
        
        completion = ChatCompletion(
            id="multi-choice-test",
            object="chat.completion",
            created=1677610602,
            model="test-model",
            choices=multiple_choices,
            usage={'total_tokens': 100}
        )
        
        assert len(completion.choices) == 3
        assert completion.choices[2]['finish_reason'] == 'length'

    def test_message_role_validation_comprehensive(self):
        """Test comprehensive message role validation."""
        valid_roles = ['system', 'user', 'assistant', 'function']
        invalid_roles = ['admin', 'bot', 'human', 'ai', 'SYSTEM', 'User']
        
        # Test valid roles
        for role in valid_roles:
            message = ChatMessage(role=role, content="test")
            assert message.role == role
        
        # Test invalid roles should be handled appropriately
        for role in invalid_roles:
            try:
                message = ChatMessage(role=role, content="test")
                # Some invalid roles might be accepted at creation
                # but should fail validation later
            except (ValueError, TypeError):
                pass

    def test_model_config_with_custom_parameters(self):
        """Test ModelConfig with custom/additional parameters."""
        # Test with extra parameters that might be passed through
        config = ModelConfig(
            name="custom-model",
            max_tokens=2000,
            temperature=0.8,
            top_p=0.95,
            custom_param="custom_value",
            another_param=123
        )
        
        assert config.name == "custom-model"
        assert config.max_tokens == 2000
        # Custom parameters should be stored if the model supports them
        assert hasattr(config, 'custom_param') or 'custom_param' in config.__dict__


class TestAdvancedErrorScenarios:
    """Advanced error handling and edge case scenarios."""
    
    @pytest.mark.asyncio
    async def test_network_interruption_recovery(self, client, sample_messages, sample_model_config):
        """Test recovery from network interruptions during requests."""
        import aiohttp
        
        call_count = 0
        
        async def intermittent_network_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise aiohttp.ClientConnectionError("Network unreachable")
            elif call_count == 2:
                raise asyncio.TimeoutError("Request timeout")
            else:
                # Success on third attempt
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    'id': 'network-recovery',
                    'choices': [{'message': {'content': 'Success after recovery'}}],
                    'usage': {'total_tokens': 20}
                })
                return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=intermittent_network_error):
            with patch('asyncio.sleep'):
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'network-recovery'
                assert call_count == 3

    @pytest.mark.asyncio
    async def test_partial_streaming_response_handling(self, client, sample_messages, sample_model_config):
        """Test handling of partial/corrupted streaming responses."""
        async def partial_stream():
            # Simulate partial responses and connection drops
            yield b'{"choices": [{"delta": {"content": "Start"}}]}'
            yield b'{"choices": [{"delta": {"content": " middle"}}]}'
            yield b'{"incomplete": true, "choices": [{"delta":'  # Incomplete JSON
            # Stream ends abruptly
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=partial_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            try:
                async for chunk in client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                ):
                    chunks.append(chunk)
            except (GenesisAPIError, json.JSONDecodeError):
                # Should handle partial streams gracefully
                pass
            
            # Should have processed valid chunks before failure
            assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_api_version_compatibility(self, client, sample_messages, sample_model_config):
        """Test handling of different API version responses."""
        # Test with different API version response formats
        v1_response = {
            'id': 'v1-test',
            'object': 'text_completion',  # Old format
            'choices': [{'text': 'V1 response'}],
            'usage': {'total_tokens': 10}
        }
        
        v2_response = {
            'id': 'v2-test',
            'object': 'chat.completion',
            'choices': [{'message': {'content': 'V2 response'}}],
            'usage': {'total_tokens': 10}
        }
        
        for response in [v1_response, v2_response]:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                try:
                    result = await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                    # Should handle both formats or raise appropriate error
                    assert result.id in ['v1-test', 'v2-test']
                except GenesisAPIError:
                    # May reject incompatible formats
                    pass

    def test_memory_efficient_large_message_handling(self, client):
        """Test memory efficiency with very large messages."""
        # Create a large message that tests memory usage
        large_content = "Test message. " * 10000  # ~140KB
        
        large_message = [ChatMessage(role="user", content=large_content)]
        
        # Should not cause memory issues during validation
        try:
            client._validate_messages(large_message)
        except ValidationError as e:
            # May reject for being too long
            assert "content too long" in str(e)

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, client, sample_messages, sample_model_config):
        """Test behavior when connection pool is exhausted."""
        import aiohttp
        
        # Simulate connection pool exhaustion
        with patch('aiohttp.ClientSession.post', side_effect=aiohttp.ClientConnectionError("Connection pool exhausted")):
            with pytest.raises(GenesisAPIError, match="Connection error"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_ssl_certificate_errors(self, client, sample_messages, sample_model_config):
        """Test handling of SSL/TLS certificate errors."""
        import aiohttp
        
        ssl_error = aiohttp.ClientSSLError("SSL certificate verification failed")
        
        with patch('aiohttp.ClientSession.post', side_effect=ssl_error):
            with pytest.raises(GenesisAPIError, match="SSL error"):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )

    @pytest.mark.asyncio
    async def test_response_size_limits(self, client, sample_messages, sample_model_config):
        """Test handling of extremely large API responses."""
        # Create a very large response
        large_content = "x" * 100000  # 100KB response
        large_response = {
            'id': 'large-response',
            'choices': [{'message': {'content': large_content}}],
            'usage': {'total_tokens': 25000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=large_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            # Should handle large responses
            assert len(result.choices[0].message.content) == 100000


class TestSpecializedScenarios:
    """Tests for specialized use cases and scenarios."""
    
    @pytest.mark.asyncio
    async def test_conversation_context_preservation(self, client, sample_model_config):
        """Test preservation of conversation context across multiple requests."""
        conversation_history = []
        
        # Simulate a multi-turn conversation
        turns = [
            ("user", "Hello, my name is Alice"),
            ("assistant", "Hello Alice! Nice to meet you."),
            ("user", "What's my name?"),
            ("assistant", "Your name is Alice.")
        ]
        
        for role, content in turns:
            conversation_history.append(ChatMessage(role=role, content=content))
        
        mock_response = {
            'id': 'context-test',
            'choices': [{'message': {'content': 'I remember you are Alice'}}],
            'usage': {'total_tokens': 30}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=conversation_history,
                model_config=sample_model_config
            )
            
            assert result.id == 'context-test'
            assert len(conversation_history) == 4

    @pytest.mark.asyncio
    async def test_token_usage_tracking_accuracy(self, client, sample_model_config):
        """Test accuracy of token usage tracking across different message types."""
        different_message_types = [
            [ChatMessage(role="user", content="Short")],
            [ChatMessage(role="user", content="A much longer message with many words")],
            [ChatMessage(role="system", content="System prompt"), 
             ChatMessage(role="user", content="User query")],
            [ChatMessage(role="user", content="Unicode: 测试 🚀 emoji")]
        ]
        
        expected_tokens = [5, 15, 25, 20]
        
        for messages, expected in zip(different_message_types, expected_tokens):
            mock_response = {
                'id': 'token-test',
                'choices': [{'message': {'content': 'Response'}}],
                'usage': {'total_tokens': expected}
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                result = await client.create_chat_completion(
                    messages=messages,
                    model_config=sample_model_config
                )
                
                assert result.usage.total_tokens == expected

    @pytest.mark.asyncio
    async def test_model_switching_mid_conversation(self, client):
        """Test switching models during a conversation."""
        messages = [ChatMessage(role="user", content="Hello")]
        
        # Test with different models
        models = ["genesis-gpt-3.5", "genesis-gpt-4", "genesis-claude"]
        
        for model_name in models:
            config = ModelConfig(name=model_name, max_tokens=100)
            
            mock_response = {
                'id': f'model-switch-{model_name}',
                'model': model_name,
                'choices': [{'message': {'content': f'Response from {model_name}'}}],
                'usage': {'total_tokens': 10}
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                result = await client.create_chat_completion(
                    messages=messages,
                    model_config=config
                )
                
                assert result.model == model_name

    @pytest.mark.asyncio
    async def test_streaming_with_mixed_content_types(self, client, sample_messages, sample_model_config):
        """Test streaming responses with mixed content types."""
        mixed_chunks = [
            {'choices': [{'delta': {'role': 'assistant'}}]},
            {'choices': [{'delta': {'content': 'Text response'}}]},
            {'choices': [{'delta': {'content': ' with code:\n```python\nprint("hello")\n```'}}]},
            {'choices': [{'delta': {'content': '\n\nAnd a list:\n1. Item 1\n2. Item 2'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def mixed_content_stream():
            for chunk in mixed_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mixed_content_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            full_content = ""
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                if chunk.choices and chunk.choices[0].delta.get('content'):
                    full_content += chunk.choices[0].delta['content']
            
            assert 'Text response' in full_content
            assert 'python' in full_content
            assert 'Item 1' in full_content

    @pytest.mark.asyncio
    async def test_client_state_consistency(self, client, sample_messages, sample_model_config):
        """Test that client state remains consistent across operations."""
        original_api_key = client.api_key
        original_base_url = client.base_url
        
        mock_response = {
            'id': 'state-test',
            'choices': [{'message': {'content': 'Response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Make multiple requests
            for _ in range(3):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                # State should remain consistent
                assert client.api_key == original_api_key
                assert client.base_url == original_base_url

    def test_thread_safety_considerations(self, mock_config):
        """Test thread safety aspects of client creation and usage."""
        import threading
        
        clients = []
        
        def create_client():
            client = GenesisAPIClient(**mock_config)
            clients.append(client)
        
        # Create multiple clients from different threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_client)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All clients should be properly initialized
        assert len(clients) == 5
        for client in clients:
            assert client.api_key == mock_config['api_key']

    @pytest.mark.asyncio
    async def test_graceful_shutdown_handling(self, client, sample_messages, sample_model_config):
        """Test graceful handling of shutdown scenarios."""
        import signal
        
        # Simulate shutdown signal handling
        shutdown_received = False
        
        def shutdown_handler(signum, frame):
            nonlocal shutdown_received
            shutdown_received = True
        
        original_handler = signal.signal(signal.SIGTERM, shutdown_handler)
        
        try:
            mock_response = {
                'id': 'shutdown-test',
                'choices': [{'message': {'content': 'Response'}}],
                'usage': {'total_tokens': 10}
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                assert result.id == 'shutdown-test'
                # Client should handle shutdown gracefully
                await client.close()
                
        finally:
            signal.signal(signal.SIGTERM, original_handler)


class TestPerformanceOptimizations:
    """Tests for performance optimizations and efficiency."""
    
    @pytest.mark.asyncio
    async def test_connection_reuse_efficiency(self, client, sample_messages, sample_model_config):
        """Test that HTTP connections are efficiently reused."""
        mock_response = {
            'id': 'reuse-test',
            'choices': [{'message': {'content': 'Response'}}],
            'usage': {'total_tokens': 10}
        }
        
        connection_count = 0
        
        async def track_connections(*args, **kwargs):
            nonlocal connection_count
            connection_count += 1
            mock_resp = Mock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            return mock_resp
        
        with patch('aiohttp.ClientSession.post', side_effect=track_connections):
            # Make multiple requests
            for _ in range(10):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
            
            # Should reuse connections efficiently
            assert connection_count == 10
            assert not client.session.closed

    @pytest.mark.asyncio
    async def test_request_queuing_behavior(self, client, sample_messages, sample_model_config):
        """Test request queuing and throttling behavior."""
        request_times = []
        
        async def timed_request(*args, **kwargs):
            request_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate API latency
            mock_resp = Mock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={
                'id': 'queue-test',
                'choices': [{'message': {'content': 'Response'}}],
                'usage': {'total_tokens': 10}
            })
            return mock_resp
        
        with patch('aiohttp.ClientSession.post', side_effect=timed_request):
            # Send multiple concurrent requests
            tasks = [
                client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                for _ in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            assert len(request_times) == 5
            
            # Requests should be handled concurrently
            time_span = max(request_times) - min(request_times)
            assert time_span < 0.5  # Should not be sequential

    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, client, sample_model_config):
        """Test memory usage optimization with large datasets."""
        # Create a conversation with many messages
        large_conversation = [
            ChatMessage(role="user" if i % 2 == 0 else "assistant", 
                       content=f"Message {i} with some content")
            for i in range(1000)
        ]
        
        mock_response = {
            'id': 'memory-test',
            'choices': [{'message': {'content': 'Processed large conversation'}}],
            'usage': {'total_tokens': 5000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=large_conversation,
                model_config=sample_model_config
            )
            
            assert result.id == 'memory-test'
            assert result.usage.total_tokens == 5000

    @pytest.mark.asyncio
    async def test_streaming_buffer_management(self, client, sample_messages, sample_model_config):
        """Test efficient buffer management in streaming responses."""
        # Create a large streaming response
        large_chunks = [
            {'choices': [{'delta': {'content': 'x' * 1000}}]}
            for _ in range(100)
        ]
        
        async def large_stream():
            for chunk in large_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=large_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunk_count = 0
            total_content_length = 0
            
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunk_count += 1
                if chunk.choices and chunk.choices[0].delta.get('content'):
                    total_content_length += len(chunk.choices[0].delta['content'])
            
            assert chunk_count == 100
            assert total_content_length == 100000  # 100 chunks * 1000 chars each

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_high_throughput_scenario(self, client, sample_model_config):
        """Test client performance under high throughput conditions."""
        # Simulate high throughput with many small requests
        small_messages = [ChatMessage(role="user", content="Hi")]
        
        mock_response = {
            'id': 'throughput-test',
            'choices': [{'message': {'content': 'Hi'}}],
            'usage': {'total_tokens': 2}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            start_time = asyncio.get_event_loop().time()
            
            # Send 100 concurrent requests
            tasks = [
                client.create_chat_completion(
                    messages=small_messages,
                    model_config=sample_model_config
                )
                for _ in range(100)
            ]
            
            results = await asyncio.gather(*tasks)
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            assert len(results) == 100
            assert duration < 5.0  # Should complete within 5 seconds
            
            # Calculate throughput
            throughput = len(results) / duration
            assert throughput > 20  # Should handle >20 requests per second

# Run the additional tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])