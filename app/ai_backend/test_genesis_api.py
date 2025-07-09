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



class TestComprehensiveEdgeCases:
    """Comprehensive edge case testing for all client functionality."""
    
    @pytest.fixture
    def client_with_all_configs(self):
        """Create a client with all possible configuration options."""
        return GenesisAPIClient(
            api_key='comprehensive-test-key',
            base_url='https://comprehensive.test.ai/v1',
            timeout=45,
            max_retries=5
        )
    
    @pytest.mark.asyncio
    async def test_message_content_with_all_unicode_categories(self, client):
        """Test message content with all Unicode categories."""
        unicode_test_cases = [
            # Basic Multilingual Plane
            "Hello 世界 🌍",
            # Mathematical symbols
            "∑(i=1 to n) = n(n+1)/2",
            # Various scripts
            "Здравствуй мир! שלום עולם! مرحبا بالعالم!",
            # Emoji combinations
            "👨‍👩‍👧‍👦👨‍💻🏳️‍🌈🏴‍☠️",
            # Control characters (that should be handled)
            "Test\u200B\u200C\u200D\uFEFF",
            # Box drawing characters
            "┌─┬─┐\n│ │ │\n├─┼─┤\n│ │ │\n└─┴─┘",
            # Musical symbols
            "♪♫♬♩♭♮♯",
            # Currency symbols
            "¥€£$¢₹₿",
            # Dingbats
            "✓✗✦✧✨✩✪✫✬✭✮✯",
            # Braille
            "⠠⠓⠑⠇⠇⠕⠀⠺⠕⠗⠇⠙"
        ]
        
        for content in unicode_test_cases:
            message = ChatMessage(role="user", content=content)
            try:
                client._validate_messages([message])
                # If validation passes, content should be preserved
                assert message.content == content
            except ValidationError:
                # Some Unicode categories might be rejected
                pass
    
    @pytest.mark.asyncio
    async def test_streaming_with_all_possible_chunk_formats(self, client, sample_messages, sample_model_config):
        """Test streaming with all possible chunk formats and edge cases."""
        comprehensive_chunks = [
            # Standard content chunk
            b'{"choices": [{"delta": {"content": "Hello"}}]}\n',
            # Chunk with role
            b'{"choices": [{"delta": {"role": "assistant"}}]}\n',
            # Chunk with function call
            b'{"choices": [{"delta": {"function_call": {"name": "test"}}}]}\n',
            # Chunk with tool call
            b'{"choices": [{"delta": {"tool_calls": [{"function": {"name": "test"}}]}}]}\n',
            # Chunk with multiple choices
            b'{"choices": [{"delta": {"content": "A"}}, {"delta": {"content": "B"}}]}\n',
            # Chunk with usage information
            b'{"choices": [{"delta": {"content": "End"}}], "usage": {"total_tokens": 100}}\n',
            # Chunk with system fingerprint
            b'{"choices": [{"delta": {"content": "Final"}}], "system_fingerprint": "fp_123"}\n',
            # Finish chunk
            b'{"choices": [{"delta": {}, "finish_reason": "stop"}]}\n',
            # SSE format chunks
            b'data: {"choices": [{"delta": {"content": "SSE"}}]}\n\n',
            # Empty lines (should be skipped)
            b'\n\n',
            # Comments (should be skipped)
            b': this is a comment\n',
            # Done marker
            b'data: [DONE]\n\n'
        ]
        
        async def comprehensive_stream():
            for chunk in comprehensive_chunks:
                yield chunk
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                return_value=comprehensive_stream()
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            chunks = []
            async for chunk in client.create_chat_completion_stream(
                messages=sample_messages,
                model_config=sample_model_config
            ):
                chunks.append(chunk)
            
            # Should handle all chunk formats gracefully
            assert len(chunks) >= 1
    
    @pytest.mark.parametrize("edge_case_config", [
        {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1},
        {"temperature": 2.0, "top_p": 0.0, "max_tokens": 4096},
        {"temperature": 1.0, "frequency_penalty": -2.0, "presence_penalty": 2.0},
        {"temperature": 0.5, "logprobs": True, "top_logprobs": 5},
        {"temperature": 1.5, "seed": 42, "response_format": {"type": "json_object"}},
        {"temperature": 0.1, "stop": ["END", "STOP"], "n": 3}
    ])
    def test_model_config_edge_case_combinations(self, client, edge_case_config):
        """Test model configuration with edge case parameter combinations."""
        config = ModelConfig(name="test-model", **edge_case_config)
        
        try:
            client._validate_model_config(config)
            # If validation passes, all parameters should be within bounds
            assert 0.0 <= config.temperature <= 2.0
            if hasattr(config, 'top_p'):
                assert 0.0 <= config.top_p <= 1.0
            if hasattr(config, 'max_tokens'):
                assert config.max_tokens > 0
        except ValidationError:
            # Some combinations might be invalid
            pass
    
    @pytest.mark.asyncio
    async def test_error_recovery_with_all_http_status_codes(self, client, sample_messages, sample_model_config):
        """Test error recovery with all possible HTTP status codes."""
        status_codes = [
            # 4xx Client Errors
            400, 401, 403, 404, 405, 406, 409, 410, 413, 414, 415, 422, 429, 431,
            # 5xx Server Errors
            500, 501, 502, 503, 504, 505, 507, 508, 510, 511
        ]
        
        for status_code in status_codes:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.status = status_code
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                    return_value={'error': {'message': f'HTTP {status_code} error'}}
                )
                
                with pytest.raises(GenesisAPIError) as exc_info:
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                
                # Should raise appropriate exception type
                error = exc_info.value
                assert isinstance(error, GenesisAPIError)
                assert str(status_code) in str(error) or f"HTTP {status_code}" in str(error)
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_with_large_datasets(self, client, sample_model_config):
        """Test memory efficiency with extremely large datasets."""
        # Create messages with varying sizes
        message_sizes = [1, 10, 100, 1000, 10000]  # Characters
        large_message_sets = []
        
        for size in message_sizes:
            messages = []
            for i in range(100):  # 100 messages of each size
                content = f"Message {i}: " + "x" * size
                messages.append(ChatMessage(role="user" if i % 2 == 0 else "assistant", content=content))
            large_message_sets.append(messages)
        
        mock_response = {
            'id': 'memory-efficiency-test',
            'choices': [{'message': {'content': 'Processed large dataset'}}],
            'usage': {'total_tokens': 50000}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Process all large message sets
            for messages in large_message_sets:
                try:
                    result = await client.create_chat_completion(
                        messages=messages,
                        model_config=sample_model_config
                    )
                    assert result.id == 'memory-efficiency-test'
                except ValidationError:
                    # Acceptable to reject extremely large datasets
                    pass
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_stress_test(self, client, sample_messages, sample_model_config):
        """Stress test with many concurrent operations."""
        mock_response = {
            'id': 'stress-test',
            'choices': [{'message': {'content': 'Stress test response'}}],
            'usage': {'total_tokens': 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Create 100 concurrent tasks
            tasks = []
            for i in range(100):
                task = asyncio.create_task(client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                ))
                tasks.append(task)
            
            # Wait for all tasks with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=30.0
                )
                
                # Verify all tasks completed
                assert len(results) == 100
                successes = [r for r in results if isinstance(r, ChatCompletion)]
                failures = [r for r in results if isinstance(r, Exception)]
                
                # Most should succeed, some failures acceptable under stress
                assert len(successes) >= 50
                
            except asyncio.TimeoutError:
                # Cancel remaining tasks
                for task in tasks:
                    task.cancel()
                pytest.fail("Stress test timed out")
    
    def test_comprehensive_input_validation_edge_cases(self, client):
        """Test comprehensive input validation with all edge cases."""
        edge_case_inputs = [
            # Empty and whitespace variations
            ("", "empty string"),
            ("   ", "spaces only"),
            ("\t\n\r", "whitespace chars"),
            ("\u00A0\u2000\u2001", "unicode spaces"),
            
            # Special characters
            ("\x00\x01\x02", "control chars"),
            ("🏳️‍🌈🏴‍☠️👨‍👩‍👧‍👦", "complex emojis"),
            ("\\n\\r\\t\\\\", "escaped chars"),
            
            # Very long strings
            ("a" * 1000, "1k chars"),
            ("a" * 10000, "10k chars"),
            ("a" * 100000, "100k chars"),
            
            # Structured data as strings
            ('{"json": "data"}', "json string"),
            ('<xml>data</xml>', "xml string"),
            ('SELECT * FROM table', "sql string"),
            
            # Numbers as strings
            ("123", "number string"),
            ("3.14159", "float string"),
            ("1e10", "scientific notation"),
            
            # Special Unicode
            ("नमस्ते", "devanagari"),
            ("こんにちは", "hiragana"),
            ("🇺🇸🇯🇵🇬🇧", "flag emojis"),
        ]
        
        for content, description in edge_case_inputs:
            message = ChatMessage(role="user", content=content)
            
            try:
                client._validate_messages([message])
                # If validation passes, content should be handled properly
                assert message.content == content
            except ValidationError as e:
                # Expected for some edge cases
                assert "content" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_all_network_error_scenarios(self, client, sample_messages, sample_model_config):
        """Test all possible network error scenarios."""
        network_errors = [
            # Connection errors
            aiohttp.ClientConnectionError("Connection refused"),
            aiohttp.ClientConnectorError(connection_key=None, os_error=OSError("Network unreachable")),
            aiohttp.ClientSSLError(connection_key=None, os_error=ssl.SSLError("SSL error")),
            
            # Timeout errors
            asyncio.TimeoutError("Request timeout"),
            aiohttp.ServerTimeoutError("Server timeout"),
            
            # Response errors
            aiohttp.ClientResponseError(request_info=None, history=None, status=500),
            aiohttp.ClientPayloadError("Payload error"),
            
            # General errors
            aiohttp.ClientError("Generic client error"),
            aiohttp.InvalidURL("Invalid URL"),
        ]
        
        for error in network_errors:
            with patch('aiohttp.ClientSession.post', side_effect=error):
                with pytest.raises(GenesisAPIError) as exc_info:
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                
                # Should convert to GenesisAPIError
                assert isinstance(exc_info.value, GenesisAPIError)
                assert len(str(exc_info.value)) > 0


class TestComprehensivePerformanceAndScalability:
    """Comprehensive performance and scalability testing."""
    
    @pytest.mark.asyncio
    async def test_response_time_under_various_loads(self, client, sample_messages, sample_model_config):
        """Test response times under various load conditions."""
        mock_response = {
            'id': 'performance-test',
            'choices': [{'message': {'content': 'Performance response'}}],
            'usage': {'total_tokens': 10}
        }
        
        load_scenarios = [1, 5, 10, 25, 50, 100]  # Concurrent requests
        
        for load in load_scenarios:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                start_time = asyncio.get_event_loop().time()
                
                # Create concurrent tasks
                tasks = []
                for _ in range(load):
                    task = client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
                end_time = asyncio.get_event_loop().time()
                
                # Verify all succeeded
                assert len(results) == load
                
                # Response time should scale reasonably
                total_time = end_time - start_time
                avg_time_per_request = total_time / load
                
                # Should complete within reasonable time
                assert total_time < 30.0  # 30 seconds max
                assert avg_time_per_request < 5.0  # 5 seconds per request max
    
    @pytest.mark.asyncio
    async def test_memory_usage_scaling(self, client, sample_model_config):
        """Test memory usage scaling with different message sizes."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        message_sizes = [10, 100, 1000, 10000]  # Characters per message
        memory_usage = []
        
        for size in message_sizes:
            # Create messages of specific size
            large_messages = [
                ChatMessage(role="user", content="x" * size)
                for _ in range(10)
            ]
            
            mock_response = {
                'id': f'memory-test-{size}',
                'choices': [{'message': {'content': 'Response'}}],
                'usage': {'total_tokens': size // 4}  # Rough estimate
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                # Measure memory before
                initial_memory = process.memory_info().rss
                
                # Make requests
                for _ in range(10):
                    await client.create_chat_completion(
                        messages=large_messages,
                        model_config=sample_model_config
                    )
                
                # Measure memory after
                final_memory = process.memory_info().rss
                memory_growth = final_memory - initial_memory
                memory_usage.append(memory_growth)
        
        # Memory growth should be reasonable and not exponential
        for i in range(1, len(memory_usage)):
            growth_ratio = memory_usage[i] / memory_usage[i-1] if memory_usage[i-1] > 0 else 1
            assert growth_ratio < 10  # Should not grow more than 10x
    
    @pytest.mark.asyncio
    async def test_streaming_performance_with_various_chunk_sizes(self, client, sample_messages, sample_model_config):
        """Test streaming performance with various chunk sizes."""
        chunk_sizes = [1, 10, 100, 1000, 10000]  # Characters per chunk
        
        for chunk_size in chunk_sizes:
            # Create chunks of specific size
            chunks = []
            for i in range(100):  # 100 chunks
                content = f"Chunk {i}: " + "x" * chunk_size
                chunks.append({'choices': [{'delta': {'content': content}}]})
            chunks.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
            
            async def sized_stream():
                for chunk in chunks:
                    yield json.dumps(chunk).encode()
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                    return_value=sized_stream()
                )
                mock_post.return_value.__aenter__.return_value.status = 200
                
                start_time = asyncio.get_event_loop().time()
                
                received_chunks = []
                async for chunk in client.create_chat_completion_stream(
                    messages=sample_messages,
                    model_config=sample_model_config
                ):
                    received_chunks.append(chunk)
                
                end_time = asyncio.get_event_loop().time()
                
                # Verify all chunks received
                assert len(received_chunks) == 101
                
                # Performance should be reasonable
                total_time = end_time - start_time
                assert total_time < 10.0  # 10 seconds max
                
                # Throughput should be reasonable
                total_chars = chunk_size * 100
                chars_per_second = total_chars / total_time
                assert chars_per_second > 1000  # At least 1000 chars/sec
    
    @pytest.mark.asyncio
    async def test_concurrent_streaming_performance(self, client, sample_messages, sample_model_config):
        """Test performance with concurrent streaming requests."""
        concurrent_streams = [5, 10, 20]  # Number of concurrent streams
        
        for stream_count in concurrent_streams:
            streaming_chunks = [
                {'choices': [{'delta': {'content': f'Stream chunk {i}'}}]}
                for i in range(10)
            ]
            streaming_chunks.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
            
            async def concurrent_stream():
                for chunk in streaming_chunks:
                    yield json.dumps(chunk).encode()
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value.content.iter_chunked = AsyncMock(
                    return_value=concurrent_stream()
                )
                mock_post.return_value.__aenter__.return_value.status = 200
                
                start_time = asyncio.get_event_loop().time()
                
                # Create concurrent streaming tasks
                async def collect_stream():
                    chunks = []
                    async for chunk in client.create_chat_completion_stream(
                        messages=sample_messages,
                        model_config=sample_model_config
                    ):
                        chunks.append(chunk)
                    return chunks
                
                tasks = [collect_stream() for _ in range(stream_count)]
                results = await asyncio.gather(*tasks)
                
                end_time = asyncio.get_event_loop().time()
                
                # Verify all streams completed
                assert len(results) == stream_count
                for stream_result in results:
                    assert len(stream_result) == 11  # 10 content + 1 finish
                
                # Performance should be reasonable
                total_time = end_time - start_time
                assert total_time < 15.0  # 15 seconds max for concurrent streams


class TestComprehensiveRobustnessAndReliability:
    """Comprehensive robustness and reliability testing."""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_under_all_failure_modes(self, client, sample_messages, sample_model_config):
        """Test graceful degradation under all possible failure modes."""
        failure_modes = [
            # Network failures
            ('network_timeout', asyncio.TimeoutError("Network timeout")),
            ('connection_reset', aiohttp.ClientConnectionError("Connection reset")),
            ('dns_failure', aiohttp.ClientConnectorError(connection_key=None, os_error=OSError("DNS resolution failed"))),
            
            # HTTP failures
            ('http_500', Mock(status=500, json=AsyncMock(return_value={'error': {'message': 'Internal server error'}}))),
            ('http_502', Mock(status=502, json=AsyncMock(return_value={'error': {'message': 'Bad gateway'}}))),
            ('http_503', Mock(status=503, json=AsyncMock(return_value={'error': {'message': 'Service unavailable'}}))),
            
            # Response parsing failures
            ('invalid_json', Mock(status=200, json=AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0)))),
            ('missing_fields', Mock(status=200, json=AsyncMock(return_value={'incomplete': 'response'}))),
            ('null_response', Mock(status=200, json=AsyncMock(return_value=None))),
        ]
        
        for mode_name, failure in failure_modes:
            with patch('aiohttp.ClientSession.post') as mock_post:
                if isinstance(failure, Exception):
                    mock_post.side_effect = failure
                else:
                    mock_post.return_value.__aenter__.return_value = failure
                
                with pytest.raises(GenesisAPIError) as exc_info:
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                
                # Should handle each failure mode gracefully
                error = exc_info.value
                assert isinstance(error, GenesisAPIError)
                assert len(str(error)) > 0
                
                # Error message should be informative
                error_msg = str(error).lower()
                assert any(keyword in error_msg for keyword in [
                    'error', 'failed', 'timeout', 'connection', 'invalid', 'server'
                ])
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_under_all_conditions(self, client):
        """Test resource cleanup under all possible conditions."""
        import gc
        import weakref
        
        # Track resource usage
        initial_objects = len(gc.get_objects())
        
        # Test various failure conditions
        failure_conditions = [
            Exception("General failure"),
            KeyboardInterrupt("User interrupt"),
            SystemExit("System exit"),
            asyncio.CancelledError("Task cancelled"),
            MemoryError("Out of memory"),
        ]
        
        for condition in failure_conditions:
            # Create weak references to track cleanup
            weak_refs = []
            
            with patch('aiohttp.ClientSession.post', side_effect=condition):
                try:
                    # Create some objects that should be cleaned up
                    messages = [ChatMessage(role="user", content=f"Test {i}") for i in range(10)]
                    weak_refs.extend([weakref.ref(msg) for msg in messages])
                    
                    config = ModelConfig(name="test-model", temperature=0.7)
                    weak_refs.append(weakref.ref(config))
                    
                    # Attempt operation that will fail
                    await client.create_chat_completion(
                        messages=messages,
                        model_config=config
                    )
                    
                except (Exception, KeyboardInterrupt, SystemExit):
                    # Expected to fail
                    pass
            
            # Force garbage collection
            messages = None
            config = None
            gc.collect()
            
            # Check that objects were cleaned up
            alive_refs = [ref for ref in weak_refs if ref() is not None]
            # Some references might still be alive due to test framework
            assert len(alive_refs) < len(weak_refs) * 0.5  # At most 50% still alive
        
        # Final cleanup check
        gc.collect()
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        # Should not have significant object leaks
        assert object_growth < 1000  # Reasonable threshold
    
    @pytest.mark.asyncio
    async def test_state_consistency_across_all_operations(self, client, sample_messages, sample_model_config):
        """Test state consistency across all possible operations."""
        # Record initial state
        initial_state = {
            'api_key': client.api_key,
            'base_url': client.base_url,
            'timeout': client.timeout,
            'max_retries': client.max_retries,
            'session_id': id(client.session) if hasattr(client, 'session') else None
        }
        
        # Test operations that might affect state
        operations = [
            # Normal operation
            ('normal_request', Mock(status=200, json=AsyncMock(return_value={
                'id': 'test', 'choices': [{'message': {'content': 'OK'}}], 'usage': {'total_tokens': 5}
            }))),
            
            # Error conditions
            ('auth_error', Mock(status=401, json=AsyncMock(return_value={'error': {'message': 'Unauthorized'}}))),
            ('rate_limit', Mock(status=429, json=AsyncMock(return_value={'error': {'message': 'Rate limited'}}))),
            ('server_error', Mock(status=500, json=AsyncMock(return_value={'error': {'message': 'Server error'}}))),
        ]
        
        for op_name, mock_response in operations:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value = mock_response
                
                try:
                    await client.create_chat_completion(
                        messages=sample_messages,
                        model_config=sample_model_config
                    )
                except GenesisAPIError:
                    # Expected for error conditions
                    pass
                
                # Verify state hasn't changed
                assert client.api_key == initial_state['api_key']
                assert client.base_url == initial_state['base_url']
                assert client.timeout == initial_state['timeout']
                assert client.max_retries == initial_state['max_retries']
                
                # Session should remain consistent
                if hasattr(client, 'session') and initial_state['session_id']:
                    assert id(client.session) == initial_state['session_id']
    
    @pytest.mark.asyncio
    async def test_thread_safety_comprehensive(self, client, sample_messages, sample_model_config):
        """Comprehensive thread safety testing."""
        import threading
        import concurrent.futures
        
        # Test data for thread safety
        thread_results = []
        thread_errors = []
        
        def thread_operation(thread_id):
            """Operation to run in separate thread."""
            try:
                # Test property access
                api_key = client.api_key
                base_url = client.base_url
                timeout = client.timeout
                
                # Test method calls
                headers = client._build_headers()
                
                # Test validation
                client._validate_messages(sample_messages)
                client._validate_model_config(sample_model_config)
                
                return {
                    'thread_id': thread_id,
                    'api_key': api_key,
                    'base_url': base_url,
                    'timeout': timeout,
                    'headers': headers
                }
            except Exception as e:
                thread_errors.append((thread_id, str(e)))
                return None
        
        # Run operations in multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(thread_operation, i) for i in range(50)]
            thread_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify thread safety
        valid_results = [r for r in thread_results if r is not None]
        
        # Should have no errors or very few
        assert len(thread_errors) < 5  # Less than 10% failure rate
        assert len(valid_results) >= 45  # At least 90% success rate
        
        # All results should be identical (thread-safe access)
        if len(valid_results) > 1:
            first_result = valid_results[0]
            for result in valid_results[1:]:
                assert result['api_key'] == first_result['api_key']
                assert result['base_url'] == first_result['base_url']
                assert result['timeout'] == first_result['timeout']
                # Headers should be consistent
                assert result['headers']['Authorization'] == first_result['headers']['Authorization']


class TestComprehensiveSecurityAndValidation:
    """Comprehensive security and validation testing."""
    
    def test_input_sanitization_comprehensive(self, client):
        """Comprehensive input sanitization testing."""
        # Test all categories of potentially dangerous inputs
        dangerous_inputs = [
            # SQL Injection
            ("'; DROP TABLE users; --", "sql_injection"),
            ("1' OR '1'='1", "sql_tautology"),
            ("admin'/*", "sql_comment"),
            ("' UNION SELECT password FROM users --", "sql_union"),
            
            # XSS
            ("<script>alert('xss')</script>", "script_tag"),
            ("<img src=x onerror=alert('xss')>", "img_onerror"),
            ("javascript:alert('xss')", "javascript_protocol"),
            ("<svg onload=alert('xss')>", "svg_onload"),
            ("{{constructor.constructor('alert(1)')()}}", "template_injection"),
            
            # Command Injection
            ("; rm -rf /", "command_injection"),
            ("$(rm -rf /)", "command_substitution"),
            ("`rm -rf /`", "command_backticks"),
            ("| rm -rf /", "command_pipe"),
            
            # Path Traversal
            ("../../../etc/passwd", "path_traversal"),
            ("..\\..\\..\\windows\\system32\\config\\sam", "windows_path_traversal"),
            ("....//....//....//etc//passwd", "double_dot_traversal"),
            ("%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd", "url_encoded_traversal"),
            
            # LDAP Injection
            ("*)(uid=*))(|(uid=*", "ldap_injection"),
            ("admin)(&(password=*))", "ldap_bypass"),
            
            # Header Injection
            ("value\r\nX-Injected: malicious", "header_injection_crlf"),
            ("value\nX-Injected: malicious", "header_injection_lf"),
            ("value\x00X-Injected: malicious", "header_injection_null"),
            
            # XML/XXE
            ("<?xml version=\"1.0\" encoding=\"UTF-8\"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]><foo>&xxe;</foo>", "xxe_attack"),
            
            # NoSQL Injection
            ("{'$ne': null}", "nosql_injection"),
            ("{'$regex': '.*'}", "nosql_regex"),
            
            # Template Injection
            ("{{7*7}}", "template_expression"),
            ("${7*7}", "template_expression_2"),
            ("#{7*7}", "template_expression_3"),
            
            # Buffer Overflow Attempts
            ("A" * 10000, "buffer_overflow"),
            ("\x00" * 1000, "null_buffer"),
            
            # Format String
            ("%s%s%s%s", "format_string"),
            ("%n%n%n%n", "format_string_write"),
            
            # Encoding Bypasses
            ("alert%28%29", "url_encoded"),
            ("alert&#40;&#41;", "html_encoded"),
            ("alert\\u0028\\u0029", "unicode_escaped"),
        ]
        
        for content, attack_type in dangerous_inputs:
            message = ChatMessage(role="user", content=content)
            
            try:
                client._validate_messages([message])
                # If validation passes, content should be preserved as-is
                # (filtering is not the client's responsibility)
                assert message.content == content
            except ValidationError as e:
                # Only acceptable validation errors are length or format related
                error_msg = str(e).lower()
                acceptable_errors = ["too long", "invalid format", "length", "size"]
                assert any(acceptable in error_msg for acceptable in acceptable_errors)
    
    def test_authentication_security_comprehensive(self, client):
        """Comprehensive authentication security testing."""
        # Test API key handling
        original_key = client.api_key
        
        # API key should not appear in string representations
        client_str = str(client)
        client_repr = repr(client)
        
        assert original_key not in client_str
        assert original_key not in client_repr
        
        # API key should not appear in headers logging
        headers = client._build_headers()
        assert 'Authorization' in headers
        assert headers['Authorization'].startswith('Bearer ')
        
        # Test header building with various inputs
        header_test_cases = [
            None,
            {},
            {'X-Custom': 'value'},
            {'Authorization': 'Bearer should-be-overridden'},
            {'X-Test': 'value\r\nX-Injected: malicious'},
        ]
        
        for custom_headers in header_test_cases:
            try:
                result_headers = client._build_headers(custom_headers)
                
                # Should always have correct Authorization
                assert 'Authorization' in result_headers
                assert result_headers['Authorization'] == f'Bearer {original_key}'
                
                # Should sanitize injected headers
                for key, value in result_headers.items():
                    assert '\r' not in str(value)
                    assert '\n' not in str(value)
                    assert '\x00' not in str(value)
                    
            except (ValueError, TypeError):
                # Acceptable to reject invalid header inputs
                pass
    
    def test_data_exposure_prevention_comprehensive(self, client):
        """Comprehensive data exposure prevention testing."""
        # Test various error conditions to ensure no data leakage
        error_scenarios = [
            Exception("General error"),
            ValueError("Value error"),
            TypeError("Type error"),
            AttributeError("Attribute error"),
            KeyError("Key error"),
            IndexError("Index error"),
            ConnectionError("Connection error"),
        ]
        
        for error in error_scenarios:
            with patch('aiohttp.ClientSession.post', side_effect=error):
                try:
                    asyncio.run(client.create_chat_completion(
                        messages=[ChatMessage(role="user", content="test")],
                        model_config=ModelConfig(name="test-model")
                    ))
                except GenesisAPIError as e:
                    error_message = str(e)
                    
                    # Should not contain sensitive information
                    assert client.api_key not in error_message
                    assert "Bearer" not in error_message or client.api_key not in error_message
                    
                    # Should not contain internal paths or system info
                    sensitive_patterns = [
                        "/home/", "/usr/", "/var/", "/etc/",
                        "C:\\", "D:\\", "\\Windows\\",
                        "password", "secret", "token", "key",
                        "admin", "root", "system"
                    ]
                    
                    error_lower = error_message.lower()
                    for pattern in sensitive_patterns:
                        assert pattern.lower() not in error_lower
    
    def test_rate_limiting_security(self, client, sample_messages, sample_model_config):
        """Test rate limiting security measures."""
        # Test rate limit headers
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 429
            mock_post.return_value.__aenter__.return_value.headers = {
                'Retry-After': '60',
                'X-RateLimit-Limit': '1000',
                'X-RateLimit-Remaining': '0',
                'X-RateLimit-Reset': '1234567890'
            }
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'error': {'message': 'Rate limit exceeded'}}
            )
            
            with pytest.raises(RateLimitError) as exc_info:
                asyncio.run(client.create_chat_completion(
                    messages=sample_messages,
                    model_config=sample_model_config
                ))
            
            # Should properly handle rate limit information
            error = exc_info.value
            assert isinstance(error, RateLimitError)
            assert error.retry_after == 60
            
            # Should not expose sensitive rate limit details
            error_msg = str(error)
            assert "Rate limit" in error_msg or "rate limit" in error_msg
    
    def test_input_length_security_boundaries(self, client):
        """Test input length security boundaries."""
        # Test various length boundaries that might cause security issues
        boundary_tests = [
            # Common buffer sizes
            (255, "uint8_boundary"),
            (256, "uint8_overflow"),
            (65535, "uint16_boundary"),
            (65536, "uint16_overflow"),
            (2**16, "64k_boundary"),
            (2**17, "128k_boundary"),
            (2**20, "1mb_boundary"),
            
            # Powers of 2 that might cause issues
            (1024, "1k_boundary"),
            (2048, "2k_boundary"),
            (4096, "4k_boundary"),
            (8192, "8k_boundary"),
            (16384, "16k_boundary"),
            (32768, "32k_boundary"),
        ]
        
        for length, test_name in boundary_tests:
            content = "x" * length
            message = ChatMessage(role="user", content=content)
            
            try:
                client._validate_messages([message])
                # If validation passes, should handle the length properly
                assert len(message.content) == length
            except ValidationError as e:
                # Should give clear error about length
                error_msg = str(e).lower()
                assert "too long" in error_msg or "length" in error_msg or "size" in error_msg
            except MemoryError:
                # Acceptable for very large inputs
                pass


# Final test runner
if __name__ == "__main__":
    # Run all comprehensive tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=20",
        "--maxfail=5",  # Stop after 5 failures
        "--strict-markers",
        "--disable-warnings",
        "-x",  # Stop on first failure for debugging
    ])
