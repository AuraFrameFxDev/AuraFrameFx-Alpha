import unittest
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add the app directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.ai_backend.genesis_core import (
    GenesisCore,
    GenesisConfig,
    GenesisException,
    ModelInitializationError,
    InferenceError,
    ConfigurationError
)


class TestGenesisConfig(unittest.TestCase):
    """Test cases for GenesisConfig class."""

    def setUp(self):
        """
        Prepare a valid configuration dictionary for use in test cases.
        """
        self.valid_config = {
            'model_name': 'test-model',
            'model_version': '1.0.0',
            'max_tokens': 1000,
            'temperature': 0.7,
            'top_p': 0.9,
            'api_key': 'test-key',
            'endpoint': 'https://api.test.com'
        }

    def test_genesis_config_init_with_valid_config(self):
        """Test GenesisConfig initialization with valid configuration."""
        config = GenesisConfig(self.valid_config)
        self.assertEqual(config.model_name, 'test-model')
        self.assertEqual(config.model_version, '1.0.0')
        self.assertEqual(config.max_tokens, 1000)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)

    def test_genesis_config_init_with_minimal_config(self):
        """Test GenesisConfig initialization with minimal required configuration."""
        minimal_config = {
            'model_name': 'minimal-model',
            'api_key': 'test-key'
        }
        config = GenesisConfig(minimal_config)
        self.assertEqual(config.model_name, 'minimal-model')
        self.assertEqual(config.api_key, 'test-key')
        # Test default values
        self.assertEqual(config.max_tokens, 2048)
        self.assertEqual(config.temperature, 0.8)

    def test_genesis_config_init_with_empty_config(self):
        """Test GenesisConfig initialization with empty configuration raises error."""
        with self.assertRaises(ConfigurationError):
            GenesisConfig({})

    def test_genesis_config_init_with_invalid_temperature(self):
        """Test GenesisConfig initialization with invalid temperature value."""
        invalid_config = self.valid_config.copy()
        invalid_config['temperature'] = 2.5  # Invalid temperature > 2.0
        with self.assertRaises(ConfigurationError):
            GenesisConfig(invalid_config)

    def test_genesis_config_init_with_negative_max_tokens(self):
        """Test GenesisConfig initialization with negative max_tokens."""
        invalid_config = self.valid_config.copy()
        invalid_config['max_tokens'] = -100
        with self.assertRaises(ConfigurationError):
            GenesisConfig(invalid_config)

    def test_genesis_config_to_dict(self):
        """Test GenesisConfig to_dict method."""
        config = GenesisConfig(self.valid_config)
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['model_name'], 'test-model')
        self.assertEqual(config_dict['max_tokens'], 1000)

    def test_genesis_config_from_file_valid(self):
        """
        Tests that GenesisConfig.from_file correctly loads a valid configuration from a JSON file.
        """
        # Create temporary config file
        import tempfile
        import json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.valid_config, f)
            temp_file = f.name

        try:
            config = GenesisConfig.from_file(temp_file)
            self.assertEqual(config.model_name, 'test-model')
        finally:
            os.unlink(temp_file)

    def test_genesis_config_from_file_invalid_json(self):
        """Test GenesisConfig from_file method with invalid JSON file."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json content')
            temp_file = f.name

        try:
            with self.assertRaises(ConfigurationError):
                GenesisConfig.from_file(temp_file)
        finally:
            os.unlink(temp_file)

    def test_genesis_config_from_file_nonexistent(self):
        """Test GenesisConfig from_file method with nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            GenesisConfig.from_file('/path/to/nonexistent/file.json')


class TestGenesisCore(unittest.TestCase):
    """Test cases for GenesisCore class."""

    def setUp(self):
        """
        Prepare a valid configuration dictionary and initialize a GenesisConfig instance for use in tests.
        """
        self.valid_config = {
            'model_name': 'test-model',
            'model_version': '1.0.0',
            'max_tokens': 1000,
            'temperature': 0.7,
            'api_key': 'test-key',
            'endpoint': 'https://api.test.com'
        }
        self.genesis_config = GenesisConfig(self.valid_config)

    def test_genesis_core_init_with_valid_config(self):
        """Test GenesisCore initialization with valid configuration."""
        core = GenesisCore(self.genesis_config)
        self.assertIsNotNone(core.config)
        self.assertEqual(core.config.model_name, 'test-model')
        self.assertFalse(core.is_initialized)

    def test_genesis_core_init_with_none_config(self):
        """Test GenesisCore initialization with None configuration."""
        with self.assertRaises(ConfigurationError):
            GenesisCore(None)

    @patch('app.ai_backend.genesis_core.requests.post')
    def test_genesis_core_initialize_success(self, mock_post):
        """
        Tests that GenesisCore initializes successfully when the API returns a successful response.
        
        Verifies that the initialize method returns True, sets the is_initialized flag, and assigns the correct model_id.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'success',
            'model_id': 'test-model-123',
            'initialized': True
        }
        mock_post.return_value = mock_response

        core = GenesisCore(self.genesis_config)
        result = core.initialize()
        
        self.assertTrue(result)
        self.assertTrue(core.is_initialized)
        self.assertEqual(core.model_id, 'test-model-123')

    @patch('app.ai_backend.genesis_core.requests.post')
    def test_genesis_core_initialize_failure(self, mock_post):
        """Test failed GenesisCore initialization."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_post.return_value = mock_response

        core = GenesisCore(self.genesis_config)
        
        with self.assertRaises(ModelInitializationError):
            core.initialize()

    @patch('app.ai_backend.genesis_core.requests.post')
    def test_genesis_core_initialize_timeout(self, mock_post):
        """Test GenesisCore initialization with timeout."""
        mock_post.side_effect = requests.exceptions.Timeout()

        core = GenesisCore(self.genesis_config)
        
        with self.assertRaises(ModelInitializationError):
            core.initialize()

    @patch('app.ai_backend.genesis_core.requests.post')
    def test_genesis_core_generate_success(self, mock_post):
        """
        Test that GenesisCore successfully generates text after initialization, returning the expected response fields.
        """
        # Mock initialization
        mock_init_response = Mock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {
            'status': 'success',
            'model_id': 'test-model-123',
            'initialized': True
        }
        
        # Mock generation
        mock_gen_response = Mock()
        mock_gen_response.status_code = 200
        mock_gen_response.json.return_value = {
            'text': 'Generated text response',
            'tokens_used': 150,
            'finish_reason': 'stop'
        }
        
        mock_post.side_effect = [mock_init_response, mock_gen_response]

        core = GenesisCore(self.genesis_config)
        core.initialize()
        
        result = core.generate('Test prompt')
        
        self.assertEqual(result['text'], 'Generated text response')
        self.assertEqual(result['tokens_used'], 150)
        self.assertEqual(result['finish_reason'], 'stop')

    def test_genesis_core_generate_without_initialization(self):
        """Test text generation without initialization raises error."""
        core = GenesisCore(self.genesis_config)
        
        with self.assertRaises(ModelInitializationError):
            core.generate('Test prompt')

    @patch('app.ai_backend.genesis_core.requests.post')
    def test_genesis_core_generate_empty_prompt(self, mock_post):
        """
        Test that generating text with an empty prompt raises a ValueError.
        """
        # Mock initialization
        mock_init_response = Mock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {
            'status': 'success',
            'model_id': 'test-model-123',
            'initialized': True
        }
        mock_post.return_value = mock_init_response

        core = GenesisCore(self.genesis_config)
        core.initialize()
        
        with self.assertRaises(ValueError):
            core.generate('')

    @patch('app.ai_backend.genesis_core.requests.post')
    def test_genesis_core_generate_none_prompt(self, mock_post):
        """
        Test that generating text with a None prompt raises a ValueError.
        """
        # Mock initialization
        mock_init_response = Mock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {
            'status': 'success',
            'model_id': 'test-model-123',
            'initialized': True
        }
        mock_post.return_value = mock_init_response

        core = GenesisCore(self.genesis_config)
        core.initialize()
        
        with self.assertRaises(ValueError):
            core.generate(None)

    @patch('app.ai_backend.genesis_core.requests.post')
    def test_genesis_core_generate_inference_error(self, mock_post):
        """
        Test that `GenesisCore.generate` raises `InferenceError` when the model API returns an inference error during text generation.
        """
        # Mock initialization
        mock_init_response = Mock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {
            'status': 'success',
            'model_id': 'test-model-123',
            'initialized': True
        }
        
        # Mock generation error
        mock_gen_response = Mock()
        mock_gen_response.status_code = 400
        mock_gen_response.text = 'Bad Request'
        
        mock_post.side_effect = [mock_init_response, mock_gen_response]

        core = GenesisCore(self.genesis_config)
        core.initialize()
        
        with self.assertRaises(InferenceError):
            core.generate('Test prompt')

    @patch('app.ai_backend.genesis_core.requests.post')
    def test_genesis_core_generate_with_parameters(self, mock_post):
        """
        Test that GenesisCore generates text using custom parameters and sends them in the request payload.
        
        Verifies that the generated result reflects the custom `temperature` and `max_tokens` values, and that these parameters are included in the API request.
        """
        # Mock initialization and generation
        mock_init_response = Mock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {
            'status': 'success',
            'model_id': 'test-model-123',
            'initialized': True
        }
        
        mock_gen_response = Mock()
        mock_gen_response.status_code = 200
        mock_gen_response.json.return_value = {
            'text': 'Generated text with custom params',
            'tokens_used': 200,
            'finish_reason': 'stop'
        }
        
        mock_post.side_effect = [mock_init_response, mock_gen_response]

        core = GenesisCore(self.genesis_config)
        core.initialize()
        
        result = core.generate('Test prompt', temperature=0.9, max_tokens=500)
        
        self.assertEqual(result['text'], 'Generated text with custom params')
        self.assertEqual(result['tokens_used'], 200)
        
        # Verify the request was made with custom parameters
        call_args = mock_post.call_args_list[1]
        request_data = call_args[1]['json']
        self.assertEqual(request_data['temperature'], 0.9)
        self.assertEqual(request_data['max_tokens'], 500)

    def test_genesis_core_get_model_info_without_initialization(self):
        """Test get_model_info without initialization."""
        core = GenesisCore(self.genesis_config)
        
        with self.assertRaises(ModelInitializationError):
            core.get_model_info()

    @patch('app.ai_backend.genesis_core.requests.get')
    def test_genesis_core_get_model_info_success(self, mock_get):
        """
        Tests that `GenesisCore.get_model_info` successfully retrieves and returns model information after initialization.
        
        Verifies that the returned dictionary contains expected fields such as model name, version, parameter count, and context length.
        """
        # Mock initialization
        with patch('app.ai_backend.genesis_core.requests.post') as mock_post:
            mock_init_response = Mock()
            mock_init_response.status_code = 200
            mock_init_response.json.return_value = {
                'status': 'success',
                'model_id': 'test-model-123',
                'initialized': True
            }
            mock_post.return_value = mock_init_response

            core = GenesisCore(self.genesis_config)
            core.initialize()

        # Mock model info retrieval
        mock_info_response = Mock()
        mock_info_response.status_code = 200
        mock_info_response.json.return_value = {
            'name': 'test-model',
            'version': '1.0.0',
            'parameters': 175000000000,
            'context_length': 4096
        }
        mock_get.return_value = mock_info_response

        result = core.get_model_info()
        
        self.assertEqual(result['name'], 'test-model')
        self.assertEqual(result['version'], '1.0.0')
        self.assertEqual(result['parameters'], 175000000000)

    def test_genesis_core_reset(self):
        """
        Tests that the GenesisCore reset method clears the initialization state and model ID.
        """
        core = GenesisCore(self.genesis_config)
        
        # Manually set some state
        core.is_initialized = True
        core.model_id = 'test-model-123'
        
        core.reset()
        
        self.assertFalse(core.is_initialized)
        self.assertIsNone(core.model_id)

    @patch('app.ai_backend.genesis_core.requests.post')
    def test_genesis_core_batch_generate_success(self, mock_post):
        """
        Test that GenesisCore.batch_generate successfully generates responses for multiple prompts.
        
        Verifies that batch generation returns the expected number of results with correct content when the API responds successfully.
        """
        # Mock initialization
        mock_init_response = Mock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {
            'status': 'success',
            'model_id': 'test-model-123',
            'initialized': True
        }
        
        # Mock batch generation
        mock_batch_response = Mock()
        mock_batch_response.status_code = 200
        mock_batch_response.json.return_value = {
            'results': [
                {'text': 'Response 1', 'tokens_used': 100},
                {'text': 'Response 2', 'tokens_used': 120}
            ]
        }
        
        mock_post.side_effect = [mock_init_response, mock_batch_response]

        core = GenesisCore(self.genesis_config)
        core.initialize()
        
        prompts = ['Prompt 1', 'Prompt 2']
        results = core.batch_generate(prompts)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['text'], 'Response 1')
        self.assertEqual(results[1]['text'], 'Response 2')

    def test_genesis_core_batch_generate_empty_prompts(self):
        """Test batch generation with empty prompts list."""
        core = GenesisCore(self.genesis_config)
        
        with self.assertRaises(ValueError):
            core.batch_generate([])

    @patch('app.ai_backend.genesis_core.requests.post')
    def test_genesis_core_streaming_generate_success(self, mock_post):
        """
        Test that `GenesisCore.streaming_generate` yields correct text chunks during successful streaming generation.
        
        Verifies that the method returns the expected sequence of text chunks when the streaming API responds with multiple data events.
        """
        # Mock initialization
        mock_init_response = Mock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {
            'status': 'success',
            'model_id': 'test-model-123',
            'initialized': True
        }
        
        # Mock streaming response
        mock_stream_response = Mock()
        mock_stream_response.status_code = 200
        mock_stream_response.iter_lines.return_value = [
            b'data: {"text": "Hello", "delta": "Hello"}',
            b'data: {"text": "Hello world", "delta": " world"}',
            b'data: [DONE]'
        ]
        
        mock_post.side_effect = [mock_init_response, mock_stream_response]

        core = GenesisCore(self.genesis_config)
        core.initialize()
        
        chunks = list(core.streaming_generate('Test prompt'))
        
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]['text'], 'Hello')
        self.assertEqual(chunks[1]['text'], 'Hello world')

    def test_genesis_core_context_manager(self):
        """Test GenesisCore as context manager."""
        with patch('app.ai_backend.genesis_core.requests.post') as mock_post:
            mock_init_response = Mock()
            mock_init_response.status_code = 200
            mock_init_response.json.return_value = {
                'status': 'success',
                'model_id': 'test-model-123',
                'initialized': True
            }
            mock_post.return_value = mock_init_response

            with GenesisCore(self.genesis_config) as core:
                self.assertTrue(core.is_initialized)

    def test_genesis_core_repr(self):
        """Test GenesisCore string representation."""
        core = GenesisCore(self.genesis_config)
        repr_str = repr(core)
        self.assertIn('GenesisCore', repr_str)
        self.assertIn('test-model', repr_str)

    def test_genesis_core_str(self):
        """Test GenesisCore string representation."""
        core = GenesisCore(self.genesis_config)
        str_repr = str(core)
        self.assertIn('GenesisCore', str_repr)
        self.assertIn('test-model', str_repr)


class TestGenesisExceptions(unittest.TestCase):
    """Test cases for GenesisCore custom exceptions."""

    def test_genesis_exception_creation(self):
        """Test GenesisException creation."""
        exc = GenesisException('Test error message')
        self.assertEqual(str(exc), 'Test error message')
        self.assertIsInstance(exc, Exception)

    def test_model_initialization_error_creation(self):
        """
        Test that a ModelInitializationError is correctly instantiated and inherits from GenesisException.
        """
        exc = ModelInitializationError('Model init failed')
        self.assertEqual(str(exc), 'Model init failed')
        self.assertIsInstance(exc, GenesisException)

    def test_inference_error_creation(self):
        """
        Tests that an `InferenceError` can be created with a message and is an instance of `GenesisException`.
        """
        exc = InferenceError('Inference failed')
        self.assertEqual(str(exc), 'Inference failed')
        self.assertIsInstance(exc, GenesisException)

    def test_configuration_error_creation(self):
        """
        Test that a ConfigurationError is correctly instantiated and inherits from GenesisException.
        """
        exc = ConfigurationError('Config error')
        self.assertEqual(str(exc), 'Config error')
        self.assertIsInstance(exc, GenesisException)

    def test_exception_with_additional_info(self):
        """
        Verify that an InferenceError can be instantiated with additional attributes and that these attributes are accessible.
        """
        exc = InferenceError('Inference failed', error_code=500, details={'retry': True})
        self.assertEqual(str(exc), 'Inference failed')
        self.assertEqual(exc.error_code, 500)
        self.assertEqual(exc.details['retry'], True)


class TestGenesisUtilities(unittest.TestCase):
    """Test cases for GenesisCore utility functions."""

    def test_validate_prompt_valid(self):
        """
        Tests that the prompt validation utility returns True for a valid prompt string.
        """
        from app.ai_backend.genesis_core import validate_prompt
        
        result = validate_prompt('This is a valid prompt')
        self.assertTrue(result)

    def test_validate_prompt_empty(self):
        """
        Test that validating an empty prompt raises a ValueError.
        """
        from app.ai_backend.genesis_core import validate_prompt
        
        with self.assertRaises(ValueError):
            validate_prompt('')

    def test_validate_prompt_none(self):
        """
        Test that passing None to the prompt validator raises a ValueError.
        """
        from app.ai_backend.genesis_core import validate_prompt
        
        with self.assertRaises(ValueError):
            validate_prompt(None)

    def test_validate_prompt_too_long(self):
        """
        Test that validating an excessively long prompt raises a ValueError.
        """
        from app.ai_backend.genesis_core import validate_prompt
        
        long_prompt = 'x' * 100000  # Very long prompt
        with self.assertRaises(ValueError):
            validate_prompt(long_prompt)

    def test_sanitize_config_valid(self):
        """
        Test that the sanitize_config utility redacts sensitive fields in a valid configuration dictionary.
        """
        from app.ai_backend.genesis_core import sanitize_config
        
        config = {
            'model_name': 'test-model',
            'temperature': 0.7,
            'max_tokens': 1000,
            'api_key': 'secret-key'
        }
        
        sanitized = sanitize_config(config)
        self.assertEqual(sanitized['model_name'], 'test-model')
        self.assertEqual(sanitized['temperature'], 0.7)
        self.assertEqual(sanitized['api_key'], '***')  # Should be redacted

    def test_format_model_response_valid(self):
        """
        Tests that the model response formatting utility correctly processes a valid raw response, preserving key fields and adding a timestamp.
        """
        from app.ai_backend.genesis_core import format_model_response
        
        raw_response = {
            'text': 'Generated text',
            'tokens_used': 150,
            'finish_reason': 'stop',
            'model_id': 'test-model-123'
        }
        
        formatted = format_model_response(raw_response)
        self.assertEqual(formatted['text'], 'Generated text')
        self.assertEqual(formatted['tokens_used'], 150)
        self.assertIn('timestamp', formatted)


class TestGenesisIntegration(unittest.TestCase):
    """Integration test cases for GenesisCore."""

    def setUp(self):
        """
        Prepare integration test configuration with predefined model parameters.
        """
        self.config = {
            'model_name': 'integration-test-model',
            'model_version': '1.0.0',
            'max_tokens': 1000,
            'temperature': 0.7,
            'api_key': 'integration-test-key',
            'endpoint': 'https://api.test.com'
        }

    @patch('app.ai_backend.genesis_core.requests.post')
    def test_full_workflow_success(self, mock_post):
        """
        Tests the complete GenesisCore workflow, including initialization, text generation, and reset, using mocked API responses to verify correct state transitions and outputs.
        """
        # Mock initialization
        mock_init_response = Mock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {
            'status': 'success',
            'model_id': 'integration-test-model-123',
            'initialized': True
        }
        
        # Mock generation
        mock_gen_response = Mock()
        mock_gen_response.status_code = 200
        mock_gen_response.json.return_value = {
            'text': 'Integration test response',
            'tokens_used': 250,
            'finish_reason': 'stop'
        }
        
        mock_post.side_effect = [mock_init_response, mock_gen_response]

        genesis_config = GenesisConfig(self.config)
        core = GenesisCore(genesis_config)
        
        # Test initialization
        init_result = core.initialize()
        self.assertTrue(init_result)
        self.assertTrue(core.is_initialized)
        
        # Test generation
        gen_result = core.generate('Integration test prompt')
        self.assertEqual(gen_result['text'], 'Integration test response')
        self.assertEqual(gen_result['tokens_used'], 250)
        
        # Test reset
        core.reset()
        self.assertFalse(core.is_initialized)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)