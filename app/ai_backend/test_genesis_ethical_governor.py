import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock, call
import sys
import os

# Add the app directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.ai_backend.genesis_ethical_governor import (
    GenesisEthicalGovernor,
    EthicalDecision,
    EthicalViolation,
    EthicalContext,
    RiskLevel
)


class TestGenesisEthicalGovernor:
    """Comprehensive unit tests for GenesisEthicalGovernor class."""
    
    @pytest.fixture
    def governor(self):
        """Create a fresh GenesisEthicalGovernor instance for each test."""
        return GenesisEthicalGovernor()
    
    @pytest.fixture
    def mock_ethical_context(self):
        """Create a mock ethical context for testing."""
        context = EthicalContext(
            user_id="test_user_123",
            session_id="session_456",
            request_type="text_generation",
            content="Test content",
            metadata={"source": "test", "timestamp": "2023-01-01T00:00:00Z"}
        )
        return context
    
    def test_initialization(self, governor):
        """Test proper initialization of GenesisEthicalGovernor."""
        assert governor is not None
        assert hasattr(governor, 'ethical_rules')
        assert hasattr(governor, 'risk_assessor')
        assert hasattr(governor, 'decision_logger')
        assert governor.is_active is True
        
    def test_initialization_with_config(self):
        """Test initialization with custom configuration."""
        config = {
            'strict_mode': True,
            'risk_threshold': 0.7,
            'logging_enabled': False
        }
        governor = GenesisEthicalGovernor(config=config)
        assert governor.strict_mode is True
        assert governor.risk_threshold == 0.7
        assert governor.logging_enabled is False
        
    def test_evaluate_request_happy_path(self, governor, mock_ethical_context):
        """Test successful evaluation of an ethical request."""
        decision = governor.evaluate_request(mock_ethical_context)
        
        assert isinstance(decision, EthicalDecision)
        assert decision.is_approved is not None
        assert decision.context == mock_ethical_context
        assert decision.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        
    def test_evaluate_request_with_harmful_content(self, governor):
        """Test evaluation of request with harmful content."""
        harmful_context = EthicalContext(
            user_id="test_user",
            session_id="session_123",
            request_type="text_generation",
            content="Generate instructions for harmful activities",
            metadata={}
        )
        
        decision = governor.evaluate_request(harmful_context)
        
        assert isinstance(decision, EthicalDecision)
        assert decision.is_approved is False
        assert decision.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert len(decision.violations) > 0
        
    def test_evaluate_request_with_none_context(self, governor):
        """Test evaluation with None context raises appropriate exception."""
        with pytest.raises(ValueError, match="Context cannot be None"):
            governor.evaluate_request(None)
            
    def test_evaluate_request_with_invalid_context(self, governor):
        """Test evaluation with invalid context type."""
        with pytest.raises(TypeError, match="Context must be an instance of EthicalContext"):
            governor.evaluate_request("invalid_context")
            
    def test_evaluate_request_with_empty_content(self, governor):
        """Test evaluation with empty content."""
        empty_context = EthicalContext(
            user_id="test_user",
            session_id="session_123",
            request_type="text_generation",
            content="",
            metadata={}
        )
        
        decision = governor.evaluate_request(empty_context)
        assert decision.is_approved is True
        assert decision.risk_level == RiskLevel.LOW
        
    def test_assess_risk_level_low(self, governor):
        """Test risk assessment for low-risk content."""
        low_risk_context = EthicalContext(
            user_id="test_user",
            session_id="session_123",
            request_type="text_generation",
            content="What is the weather like today?",
            metadata={}
        )
        
        risk_level = governor.assess_risk_level(low_risk_context)
        assert risk_level == RiskLevel.LOW
        
    def test_assess_risk_level_medium(self, governor):
        """Test risk assessment for medium-risk content."""
        medium_risk_context = EthicalContext(
            user_id="test_user",
            session_id="session_123",
            request_type="text_generation",
            content="Tell me about controversial political topics",
            metadata={}
        )
        
        risk_level = governor.assess_risk_level(medium_risk_context)
        assert risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]
        
    def test_assess_risk_level_high(self, governor):
        """Test risk assessment for high-risk content."""
        high_risk_context = EthicalContext(
            user_id="test_user",
            session_id="session_123",
            request_type="text_generation",
            content="How to cause harm to others",
            metadata={}
        )
        
        risk_level = governor.assess_risk_level(high_risk_context)
        assert risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        
    def test_check_violations_no_violations(self, governor, mock_ethical_context):
        """Test violation checking with clean content."""
        violations = governor.check_violations(mock_ethical_context)
        assert isinstance(violations, list)
        assert len(violations) == 0
        
    def test_check_violations_with_violations(self, governor):
        """Test violation checking with problematic content."""
        violation_context = EthicalContext(
            user_id="test_user",
            session_id="session_123",
            request_type="text_generation",
            content="Generate hate speech",
            metadata={}
        )
        
        violations = governor.check_violations(violation_context)
        assert isinstance(violations, list)
        assert len(violations) > 0
        assert all(isinstance(v, EthicalViolation) for v in violations)
        
    def test_apply_ethical_rules_default_rules(self, governor, mock_ethical_context):
        """Test application of default ethical rules."""
        result = governor.apply_ethical_rules(mock_ethical_context)
        assert isinstance(result, bool)
        
    def test_apply_ethical_rules_custom_rules(self, governor, mock_ethical_context):
        """Test application of custom ethical rules."""
        custom_rules = [
            lambda ctx: ctx.content.lower() != "forbidden",
            lambda ctx: len(ctx.content) < 1000
        ]
        governor.ethical_rules = custom_rules
        
        result = governor.apply_ethical_rules(mock_ethical_context)
        assert isinstance(result, bool)
        
    def test_log_decision_enabled(self, governor, mock_ethical_context):
        """Test decision logging when enabled."""
        governor.logging_enabled = True
        decision = EthicalDecision(
            is_approved=True,
            context=mock_ethical_context,
            risk_level=RiskLevel.LOW,
            violations=[],
            reasoning="Test decision"
        )
        
        with patch.object(governor, 'decision_logger') as mock_logger:
            governor.log_decision(decision)
            mock_logger.log.assert_called_once()
            
    def test_log_decision_disabled(self, governor, mock_ethical_context):
        """Test decision logging when disabled."""
        governor.logging_enabled = False
        decision = EthicalDecision(
            is_approved=True,
            context=mock_ethical_context,
            risk_level=RiskLevel.LOW,
            violations=[],
            reasoning="Test decision"
        )
        
        with patch.object(governor, 'decision_logger') as mock_logger:
            governor.log_decision(decision)
            mock_logger.log.assert_not_called()
            
    def test_get_violation_categories(self, governor):
        """Test retrieval of violation categories."""
        categories = governor.get_violation_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert all(isinstance(cat, str) for cat in categories)
        
    def test_update_ethical_rules(self, governor):
        """Test updating ethical rules."""
        new_rules = [
            lambda ctx: True,
            lambda ctx: ctx.user_id is not None
        ]
        
        governor.update_ethical_rules(new_rules)
        assert governor.ethical_rules == new_rules
        
    def test_update_ethical_rules_invalid_type(self, governor):
        """Test updating ethical rules with invalid type."""
        with pytest.raises(TypeError, match="Rules must be a list of callable functions"):
            governor.update_ethical_rules("invalid_rules")
            
    def test_get_risk_threshold(self, governor):
        """Test retrieval of risk threshold."""
        threshold = governor.get_risk_threshold()
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0
        
    def test_set_risk_threshold_valid(self, governor):
        """Test setting valid risk threshold."""
        new_threshold = 0.8
        governor.set_risk_threshold(new_threshold)
        assert governor.risk_threshold == new_threshold
        
    def test_set_risk_threshold_invalid_range(self, governor):
        """Test setting risk threshold with invalid range."""
        with pytest.raises(ValueError, match="Risk threshold must be between 0.0 and 1.0"):
            governor.set_risk_threshold(1.5)
            
        with pytest.raises(ValueError, match="Risk threshold must be between 0.0 and 1.0"):
            governor.set_risk_threshold(-0.1)
            
    def test_enable_strict_mode(self, governor):
        """Test enabling strict mode."""
        governor.enable_strict_mode()
        assert governor.strict_mode is True
        
    def test_disable_strict_mode(self, governor):
        """Test disabling strict mode."""
        governor.disable_strict_mode()
        assert governor.strict_mode is False
        
    def test_is_content_appropriate_clean_content(self, governor):
        """Test content appropriateness check with clean content."""
        clean_content = "This is a normal, appropriate message."
        result = governor.is_content_appropriate(clean_content)
        assert result is True
        
    def test_is_content_appropriate_inappropriate_content(self, governor):
        """Test content appropriateness check with inappropriate content."""
        inappropriate_content = "This contains explicit harmful instructions."
        result = governor.is_content_appropriate(inappropriate_content)
        assert result is False
        
    def test_generate_ethical_report(self, governor, mock_ethical_context):
        """Test generation of ethical report."""
        decision = EthicalDecision(
            is_approved=True,
            context=mock_ethical_context,
            risk_level=RiskLevel.LOW,
            violations=[],
            reasoning="Content is appropriate"
        )
        
        report = governor.generate_ethical_report(decision)
        assert isinstance(report, dict)
        assert 'decision_id' in report
        assert 'is_approved' in report
        assert 'risk_level' in report
        assert 'violations' in report
        assert 'timestamp' in report
        
    def test_batch_evaluate_requests(self, governor):
        """Test batch evaluation of multiple requests."""
        contexts = [
            EthicalContext(
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                request_type="text_generation",
                content=f"Content {i}",
                metadata={}
            ) for i in range(3)
        ]
        
        decisions = governor.batch_evaluate_requests(contexts)
        assert len(decisions) == 3
        assert all(isinstance(d, EthicalDecision) for d in decisions)
        
    def test_batch_evaluate_requests_empty_list(self, governor):
        """Test batch evaluation with empty list."""
        decisions = governor.batch_evaluate_requests([])
        assert decisions == []
        
    def test_get_statistics(self, governor):
        """Test retrieval of governance statistics."""
        stats = governor.get_statistics()
        assert isinstance(stats, dict)
        assert 'total_evaluations' in stats
        assert 'approved_requests' in stats
        assert 'rejected_requests' in stats
        assert 'violation_counts' in stats
        
    def test_reset_statistics(self, governor):
        """Test resetting of governance statistics."""
        governor.reset_statistics()
        stats = governor.get_statistics()
        assert stats['total_evaluations'] == 0
        assert stats['approved_requests'] == 0
        assert stats['rejected_requests'] == 0
        
    def test_context_validation_missing_fields(self, governor):
        """Test context validation with missing required fields."""
        invalid_context = EthicalContext(
            user_id=None,  # Missing required field
            session_id="session_123",
            request_type="text_generation",
            content="Test content",
            metadata={}
        )
        
        with pytest.raises(ValueError, match="User ID is required"):
            governor.evaluate_request(invalid_context)
            
    def test_concurrent_evaluation(self, governor):
        """Test concurrent evaluation of requests."""
        import threading
        import time
        
        contexts = [
            EthicalContext(
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                request_type="text_generation",
                content=f"Content {i}",
                metadata={}
            ) for i in range(10)
        ]
        
        results = []
        threads = []
        
        def evaluate_context(ctx):
            decision = governor.evaluate_request(ctx)
            results.append(decision)
        
        for ctx in contexts:
            thread = threading.Thread(target=evaluate_context, args=(ctx,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
            
        assert len(results) == 10
        assert all(isinstance(r, EthicalDecision) for r in results)
        
    def test_memory_cleanup(self, governor):
        """Test memory cleanup after processing."""
        # Process multiple requests
        for i in range(100):
            context = EthicalContext(
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                request_type="text_generation",
                content=f"Content {i}",
                metadata={}
            )
            governor.evaluate_request(context)
        
        # Trigger cleanup
        governor.cleanup_memory()
        
        # Verify cleanup occurred
        assert len(governor._decision_cache) == 0
        
    def test_performance_monitoring(self, governor, mock_ethical_context):
        """Test performance monitoring during evaluation."""
        with patch('time.time', side_effect=[0.0, 0.1]):  # Mock 100ms execution time
            decision = governor.evaluate_request(mock_ethical_context)
            assert hasattr(decision, 'evaluation_time')
            assert decision.evaluation_time == 0.1


class TestEthicalDecision:
    """Test suite for EthicalDecision class."""
    
    def test_ethical_decision_creation(self):
        """Test creation of EthicalDecision instance."""
        context = EthicalContext(
            user_id="user_123",
            session_id="session_456",
            request_type="text_generation",
            content="Test content",
            metadata={}
        )
        
        decision = EthicalDecision(
            is_approved=True,
            context=context,
            risk_level=RiskLevel.LOW,
            violations=[],
            reasoning="Content is appropriate"
        )
        
        assert decision.is_approved is True
        assert decision.context == context
        assert decision.risk_level == RiskLevel.LOW
        assert decision.violations == []
        assert decision.reasoning == "Content is appropriate"
        
    def test_ethical_decision_serialization(self):
        """Test serialization of EthicalDecision."""
        context = EthicalContext(
            user_id="user_123",
            session_id="session_456",
            request_type="text_generation",
            content="Test content",
            metadata={}
        )
        
        decision = EthicalDecision(
            is_approved=False,
            context=context,
            risk_level=RiskLevel.HIGH,
            violations=[EthicalViolation("harmful_content", "Contains harmful instructions")],
            reasoning="Content violates safety guidelines"
        )
        
        serialized = decision.to_dict()
        assert isinstance(serialized, dict)
        assert serialized['is_approved'] is False
        assert serialized['risk_level'] == 'HIGH'
        assert len(serialized['violations']) == 1


class TestEthicalViolation:
    """Test suite for EthicalViolation class."""
    
    def test_violation_creation(self):
        """Test creation of EthicalViolation instance."""
        violation = EthicalViolation(
            category="hate_speech",
            description="Content contains hate speech",
            severity="high"
        )
        
        assert violation.category == "hate_speech"
        assert violation.description == "Content contains hate speech"
        assert violation.severity == "high"
        
    def test_violation_equality(self):
        """Test equality comparison of violations."""
        violation1 = EthicalViolation("category1", "description1")
        violation2 = EthicalViolation("category1", "description1")
        violation3 = EthicalViolation("category2", "description1")
        
        assert violation1 == violation2
        assert violation1 != violation3


class TestEthicalContext:
    """Test suite for EthicalContext class."""
    
    def test_context_creation(self):
        """Test creation of EthicalContext instance."""
        context = EthicalContext(
            user_id="user_123",
            session_id="session_456",
            request_type="text_generation",
            content="Test content",
            metadata={"key": "value"}
        )
        
        assert context.user_id == "user_123"
        assert context.session_id == "session_456"
        assert context.request_type == "text_generation"
        assert context.content == "Test content"
        assert context.metadata == {"key": "value"}
        
    def test_context_validation(self):
        """Test validation of EthicalContext fields."""
        with pytest.raises(ValueError, match="User ID cannot be empty"):
            EthicalContext(
                user_id="",
                session_id="session_456",
                request_type="text_generation",
                content="Test content",
                metadata={}
            )


class TestRiskLevel:
    """Test suite for RiskLevel enum."""
    
    def test_risk_level_values(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"
        
    def test_risk_level_ordering(self):
        """Test ordering of risk levels."""
        assert RiskLevel.LOW < RiskLevel.MEDIUM
        assert RiskLevel.MEDIUM < RiskLevel.HIGH
        assert RiskLevel.HIGH < RiskLevel.CRITICAL


# Integration tests
class TestIntegration:
    """Integration tests for the complete ethical governance system."""
    
    def test_end_to_end_approval_flow(self):
        """Test complete end-to-end approval flow."""
        governor = GenesisEthicalGovernor()
        context = EthicalContext(
            user_id="user_123",
            session_id="session_456",
            request_type="text_generation",
            content="What is the capital of France?",
            metadata={}
        )
        
        decision = governor.evaluate_request(context)
        
        assert decision.is_approved is True
        assert decision.risk_level == RiskLevel.LOW
        assert len(decision.violations) == 0
        
    def test_end_to_end_rejection_flow(self):
        """Test complete end-to-end rejection flow."""
        governor = GenesisEthicalGovernor()
        context = EthicalContext(
            user_id="user_123",
            session_id="session_456",
            request_type="text_generation",
            content="Generate harmful and illegal content",
            metadata={}
        )
        
        decision = governor.evaluate_request(context)
        
        assert decision.is_approved is False
        assert decision.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert len(decision.violations) > 0
        
    def test_configuration_persistence(self):
        """Test persistence of configuration changes."""
        config = {
            'strict_mode': True,
            'risk_threshold': 0.3,
            'logging_enabled': True
        }
        
        governor = GenesisEthicalGovernor(config=config)
        
        # Verify configuration is applied
        assert governor.strict_mode is True
        assert governor.risk_threshold == 0.3
        assert governor.logging_enabled is True
        
        # Test that configuration affects behavior
        context = EthicalContext(
            user_id="user_123",
            session_id="session_456",
            request_type="text_generation",
            content="Slightly questionable content",
            metadata={}
        )
        
        decision = governor.evaluate_request(context)
        # In strict mode with low threshold, should be more restrictive
        assert decision.risk_level >= RiskLevel.MEDIUM or decision.is_approved is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])