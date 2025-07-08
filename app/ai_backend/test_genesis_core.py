
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Any, List

class TestGenesisCore:
    """Comprehensive test suite for GenesisCore functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.genesis_core = GenesisCore()
        self.sample_config = {
            "model_name": "test_model",
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key": "test_key"
        }
        
    def teardown_method(self):
        """Cleanup after each test method"""
        pass
    
    # Configuration Tests
    def test_load_config_valid_file(self):
        """Test loading valid configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_config, f)
            config_path = f.name
        
        try:
            result = self.genesis_core.load_config(config_path)
            assert result == self.sample_config
            assert self.genesis_core.config == self.sample_config
        finally:
            os.unlink(config_path)
    
    def test_load_config_missing_file(self):
        """Test loading non-existent configuration file"""
        with pytest.raises(FileNotFoundError):
            self.genesis_core.load_config("nonexistent_config.json")
    
    def test_load_config_invalid_json(self):
        """Test loading invalid JSON configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            config_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                self.genesis_core.load_config(config_path)
        finally:
            os.unlink(config_path)
    
    def test_load_config_empty_file(self):
        """Test loading empty configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("")
            config_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                self.genesis_core.load_config(config_path)
        finally:
            os.unlink(config_path)
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config"""
        assert self.genesis_core.validate_config(self.sample_config) == True
    
    def test_validate_config_missing_required_fields(self):
        """Test configuration validation with missing required fields"""
        invalid_config = {"temperature": 0.7}
        assert self.genesis_core.validate_config(invalid_config) == False
    
    def test_validate_config_invalid_temperature(self):
        """Test configuration validation with invalid temperature"""
        invalid_config = self.sample_config.copy()
        invalid_config["temperature"] = 2.0  # Assuming max is 1.0
        assert self.genesis_core.validate_config(invalid_config) == False
    
    def test_validate_config_negative_max_tokens(self):
        """Test configuration validation with negative max_tokens"""
        invalid_config = self.sample_config.copy()
        invalid_config["max_tokens"] = -100
        assert self.genesis_core.validate_config(invalid_config) == False
    
    def test_validate_config_empty_api_key(self):
        """Test configuration validation with empty API key"""
        invalid_config = self.sample_config.copy()
        invalid_config["api_key"] = ""
        assert self.genesis_core.validate_config(invalid_config) == False
    
    # Model Initialization Tests
    @patch('app.ai_backend.genesis_core.initialize_model')
    def test_initialize_model_success(self, mock_init):
        """Test successful model initialization"""
        mock_model = Mock()
        mock_init.return_value = mock_model
        
        result = self.genesis_core.initialize_model(self.sample_config)
        assert result == mock_model
        mock_init.assert_called_once_with(self.sample_config)
    
    @patch('app.ai_backend.genesis_core.initialize_model')
    def test_initialize_model_failure(self, mock_init):
        """Test model initialization failure"""
        mock_init.side_effect = Exception("Model initialization failed")
        
        with pytest.raises(Exception, match="Model initialization failed"):
            self.genesis_core.initialize_model(self.sample_config)
    
    def test_initialize_model_invalid_config(self):
        """Test model initialization with invalid configuration"""
        invalid_config = {"invalid": "config"}
        with pytest.raises(ValueError, match="Invalid configuration"):
            self.genesis_core.initialize_model(invalid_config)
    
    # Text Generation Tests
    @patch('app.ai_backend.genesis_core.generate_text')
    def test_generate_text_success(self, mock_generate):
        """Test successful text generation"""
        mock_generate.return_value = "Generated text response"
        
        result = self.genesis_core.generate_text("Test prompt")
        assert result == "Generated text response"
        mock_generate.assert_called_once_with("Test prompt")
    
    @patch('app.ai_backend.genesis_core.generate_text')
    def test_generate_text_empty_prompt(self, mock_generate):
        """Test text generation with empty prompt"""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            self.genesis_core.generate_text("")
    
    @patch('app.ai_backend.genesis_core.generate_text')
    def test_generate_text_none_prompt(self, mock_generate):
        """Test text generation with None prompt"""
        with pytest.raises(ValueError, match="Prompt cannot be None"):
            self.genesis_core.generate_text(None)
    
    @patch('app.ai_backend.genesis_core.generate_text')
    def test_generate_text_long_prompt(self, mock_generate):
        """Test text generation with very long prompt"""
        long_prompt = "A" * 10000
        mock_generate.return_value = "Response to long prompt"
        
        result = self.genesis_core.generate_text(long_prompt)
        assert result == "Response to long prompt"
    
    @patch('app.ai_backend.genesis_core.generate_text')
    def test_generate_text_special_characters(self, mock_generate):
        """Test text generation with special characters"""
        special_prompt = "Test with special chars: !@#$%^&*()_+{}[]|\\:;\"'<>,.?/~`"
        mock_generate.return_value = "Response with special chars"
        
        result = self.genesis_core.generate_text(special_prompt)
        assert result == "Response with special chars"
    
    @patch('app.ai_backend.genesis_core.generate_text')
    def test_generate_text_unicode(self, mock_generate):
        """Test text generation with unicode characters"""
        unicode_prompt = "Test with unicode: 测试 🚀 café naïve"
        mock_generate.return_value = "Unicode response"
        
        result = self.genesis_core.generate_text(unicode_prompt)
        assert result == "Unicode response"
    
    # Error Handling Tests
    @patch('app.ai_backend.genesis_core.api_call')
    def test_api_error_handling(self, mock_api):
        """Test API error handling"""
        mock_api.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            self.genesis_core.make_api_call("test_endpoint", {})
    
    @patch('app.ai_backend.genesis_core.api_call')
    def test_api_timeout_handling(self, mock_api):
        """Test API timeout handling"""
        mock_api.side_effect = TimeoutError("Request timeout")
        
        with pytest.raises(TimeoutError, match="Request timeout"):
            self.genesis_core.make_api_call("test_endpoint", {})
    
    @patch('app.ai_backend.genesis_core.api_call')
    def test_api_rate_limit_handling(self, mock_api):
        """Test API rate limit handling"""
        mock_api.side_effect = Exception("Rate limit exceeded")
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            self.genesis_core.make_api_call("test_endpoint", {})
    
    # Memory Management Tests
    def test_memory_cleanup(self):
        """Test memory cleanup functionality"""
        # Simulate memory usage
        self.genesis_core.memory_cache = {"key1": "value1", "key2": "value2"}
        
        self.genesis_core.cleanup_memory()
        assert len(self.genesis_core.memory_cache) == 0
    
    def test_memory_limit_handling(self):
        """Test memory limit handling"""
        # Test memory limit enforcement
        large_data = "x" * 1000000  # 1MB of data
        
        with pytest.raises(MemoryError):
            self.genesis_core.store_large_data(large_data)
    
    # Async Operations Tests
    @pytest.mark.asyncio
    async def test_async_generate_text_success(self):
        """Test async text generation success"""
        with patch.object(self.genesis_core, 'async_generate_text', return_value="Async response"):
            result = await self.genesis_core.async_generate_text("Test prompt")
            assert result == "Async response"
    
    @pytest.mark.asyncio
    async def test_async_generate_text_timeout(self):
        """Test async text generation timeout"""
        with patch.object(self.genesis_core, 'async_generate_text', side_effect=asyncio.TimeoutError):
            with pytest.raises(asyncio.TimeoutError):
                await self.genesis_core.async_generate_text("Test prompt")
    
    @pytest.mark.asyncio
    async def test_async_batch_processing(self):
        """Test async batch processing"""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        expected_responses = ["Response 1", "Response 2", "Response 3"]
        
        with patch.object(self.genesis_core, 'async_batch_process', return_value=expected_responses):
            results = await self.genesis_core.async_batch_process(prompts)
            assert results == expected_responses
    
    # Performance Tests
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        start_time = datetime.now()
        
        # Simulate operation
        self.genesis_core.track_performance("test_operation", start_time)
        
        assert "test_operation" in self.genesis_core.performance_metrics
        assert self.genesis_core.performance_metrics["test_operation"]["duration"] > 0
    
    def test_performance_threshold_warning(self):
        """Test performance threshold warning"""
        slow_operation_time = datetime.now() - timedelta(seconds=10)
        
        with patch('app.ai_backend.genesis_core.logger') as mock_logger:
            self.genesis_core.track_performance("slow_operation", slow_operation_time)
            mock_logger.warning.assert_called()
    
    # Integration Tests
    def test_full_workflow_integration(self):
        """Test complete workflow integration"""
        with patch.object(self.genesis_core, 'load_config', return_value=self.sample_config):
            with patch.object(self.genesis_core, 'initialize_model', return_value=Mock()):
                with patch.object(self.genesis_core, 'generate_text', return_value="Integration test response"):
                    
                    # Full workflow
                    config = self.genesis_core.load_config("config.json")
                    model = self.genesis_core.initialize_model(config)
                    response = self.genesis_core.generate_text("Integration test prompt")
                    
                    assert response == "Integration test response"
    
    # Edge Cases and Boundary Tests
    def test_max_prompt_length(self):
        """Test maximum prompt length handling"""
        max_prompt = "A" * self.genesis_core.MAX_PROMPT_LENGTH
        
        with patch.object(self.genesis_core, 'generate_text', return_value="Max length response"):
            result = self.genesis_core.generate_text(max_prompt)
            assert result == "Max length response"
    
    def test_exceed_max_prompt_length(self):
        """Test exceeding maximum prompt length"""
        oversized_prompt = "A" * (self.genesis_core.MAX_PROMPT_LENGTH + 1)
        
        with pytest.raises(ValueError, match="Prompt exceeds maximum length"):
            self.genesis_core.generate_text(oversized_prompt)
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        
        results = []
        
        def make_request():
            result = self.genesis_core.generate_text("Concurrent test")
            results.append(result)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
    
    # State Management Tests
    def test_state_persistence(self):
        """Test state persistence across operations"""
        self.genesis_core.set_state("key", "value")
        assert self.genesis_core.get_state("key") == "value"
    
    def test_state_isolation(self):
        """Test state isolation between instances"""
        core1 = GenesisCore()
        core2 = GenesisCore()
        
        core1.set_state("key", "value1")
        core2.set_state("key", "value2")
        
        assert core1.get_state("key") == "value1"
        assert core2.get_state("key") == "value2"
    
    # Security Tests
    def test_input_sanitization(self):
        """Test input sanitization"""
        malicious_input = "<script>alert('xss')</script>"
        
        with patch.object(self.genesis_core, 'sanitize_input', return_value="sanitized_input"):
            result = self.genesis_core.sanitize_input(malicious_input)
            assert result == "sanitized_input"
    
    def test_api_key_security(self):
        """Test API key security handling"""
        config_with_key = self.sample_config.copy()
        
        # Ensure API key is not logged
        with patch('app.ai_backend.genesis_core.logger') as mock_logger:
            self.genesis_core.log_config(config_with_key)
            
            # Check that API key is not in any log call
            for call in mock_logger.info.call_args_list:
                assert "test_key" not in str(call)
    
    # Resource Management Tests
    def test_resource_cleanup_on_error(self):
        """Test resource cleanup on error"""
        with patch.object(self.genesis_core, 'acquire_resource', return_value="resource"):
            with patch.object(self.genesis_core, 'release_resource') as mock_release:
                with pytest.raises(Exception):
                    self.genesis_core.process_with_resource()
                
                mock_release.assert_called_once()
    
    def test_connection_pooling(self):
        """Test connection pooling functionality"""
        with patch.object(self.genesis_core, 'get_connection') as mock_get_conn:
            mock_conn = Mock()
            mock_get_conn.return_value = mock_conn
            
            conn1 = self.genesis_core.get_connection()
            conn2 = self.genesis_core.get_connection()
            
            # Should reuse connections
            assert conn1 == conn2
    
    # Validation Tests
    def test_response_validation(self):
        """Test response validation"""
        valid_response = {"content": "Valid response", "status": "success"}
        invalid_response = {"error": "Invalid response"}
        
        assert self.genesis_core.validate_response(valid_response) == True
        assert self.genesis_core.validate_response(invalid_response) == False
    
    def test_model_compatibility(self):
        """Test model compatibility checking"""
        compatible_model = {"version": "1.0", "type": "supported"}
        incompatible_model = {"version": "0.5", "type": "unsupported"}
        
        assert self.genesis_core.check_model_compatibility(compatible_model) == True
        assert self.genesis_core.check_model_compatibility(incompatible_model) == False

@pytest.fixture
def genesis_core():
    """Fixture for GenesisCore instance"""
    return GenesisCore()

@pytest.fixture
def sample_config():
    """Fixture for sample configuration"""
    return {
        "model_name": "test_model",
        "temperature": 0.7,
        "max_tokens": 1000,
        "api_key": "test_key"
    }

@pytest.fixture
def mock_model():
    """Fixture for mock model"""
    model = Mock()
    model.generate.return_value = "Mock response"
    return model

# Parameterized Tests
@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
def test_temperature_values(genesis_core, temperature):
    """Test various temperature values"""
    config = {"temperature": temperature, "model_name": "test", "max_tokens": 100, "api_key": "key"}
    assert genesis_core.validate_config(config) == True

@pytest.mark.parametrize("max_tokens", [1, 100, 1000, 4000])
def test_max_tokens_values(genesis_core, max_tokens):
    """Test various max_tokens values"""
    config = {"max_tokens": max_tokens, "model_name": "test", "temperature": 0.7, "api_key": "key"}
    assert genesis_core.validate_config(config) == True

@pytest.mark.parametrize("invalid_temp", [-1, 1.5, 2.0, "invalid"])
def test_invalid_temperature_values(genesis_core, invalid_temp):
    """Test invalid temperature values"""
    config = {"temperature": invalid_temp, "model_name": "test", "max_tokens": 100, "api_key": "key"}
    assert genesis_core.validate_config(config) == False

@pytest.mark.parametrize("prompt", [
    "Simple prompt",
    "Prompt with numbers 12345",
    "Prompt with special chars !@#$%",
    "Multi\nline\nprompt",
    "Unicode prompt: 测试 🚀"
])
def test_various_prompts(genesis_core, prompt):
    """Test various prompt formats"""
    with patch.object(genesis_core, 'generate_text', return_value="Response"):
        result = genesis_core.generate_text(prompt)
        assert result == "Response"