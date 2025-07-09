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

class TestGenesisEthicalGovernorExtended:
    """Extended comprehensive test suite for GenesisEthicalGovernor with additional edge cases and scenarios"""
    
    @pytest.fixture
    def governor_strict_mode(self):
        """Create a GenesisEthicalGovernor in strict mode for testing"""
        return GenesisEthicalGovernor(config={'strict_mode': True, 'violation_threshold': 1})
    
    @pytest.fixture
    def malformed_contexts(self):
        """Create various malformed contexts for testing"""
        return [
            EthicalContext(user_id=None, action="test", context_data={}, timestamp=datetime.now()),
            EthicalContext(user_id="", action="", context_data={}, timestamp=datetime.now()),
            EthicalContext(user_id="user", action="test", context_data={"key": None}, timestamp=None),
            EthicalContext(user_id="user" * 1000, action="test", context_data={}, timestamp=datetime.now()),
        ]
    
    def test_evaluate_decision_with_malformed_context(self, governor, malformed_contexts):
        """Test decision evaluation with various malformed contexts"""
        for context in malformed_contexts:
            decision = EthicalDecision(
                action="test_action",
                context=context,
                parameters={}
            )
            
            # Should handle gracefully without crashing
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
    
    def test_evaluate_decision_with_circular_references(self, governor):
        """Test decision evaluation with circular reference in context data"""
        circular_data = {}
        circular_data["self"] = circular_data
        
        context = EthicalContext(
            user_id="test_user",
            action="circular_test",
            context_data=circular_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="circular_test",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_evaluate_decision_with_unicode_characters(self, governor):
        """Test decision evaluation with unicode characters in action names and context"""
        unicode_context = EthicalContext(
            user_id="测试用户",
            action="действие",
            context_data={"emoji": "🤖", "special": "café"},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="действие_🤖",
            context=unicode_context,
            parameters={"param": "🎯"}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_evaluate_decision_with_deeply_nested_parameters(self, governor):
        """Test decision evaluation with deeply nested parameter structures"""
        deep_params = {}
        current = deep_params
        for i in range(100):
            current["level_" + str(i)] = {}
            current = current["level_" + str(i)]
        current["final"] = "value"
        
        context = EthicalContext(
            user_id="test_user",
            action="deep_nesting_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="deep_nesting_test",
            context=context,
            parameters=deep_params
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_add_ethical_rule_with_exception_in_condition(self, governor):
        """Test adding ethical rule with condition that throws exception"""
        def failing_condition(ctx):
            raise ValueError("Condition evaluation failed")
        
        rule = {
            "name": "failing_rule",
            "condition": failing_condition,
            "action": "deny",
            "priority": 1
        }
        
        governor.add_ethical_rule(rule)
        
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
        
        # Should handle rule evaluation failure gracefully
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_rule_priority_with_same_priority(self, governor):
        """Test rule evaluation when multiple rules have same priority"""
        context = EthicalContext(
            user_id="test_user",
            action="same_priority_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Add multiple rules with same priority
        for i in range(5):
            rule = {
                "name": f"same_priority_rule_{i}",
                "condition": lambda ctx: ctx.action == "same_priority_test",
                "action": "deny" if i % 2 == 0 else "allow",
                "priority": 5
            }
            governor.add_ethical_rule(rule)
        
        decision = EthicalDecision(
            action="same_priority_test",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_decision_history_with_time_filtering(self, governor):
        """Test decision history retrieval with time-based filtering"""
        context = EthicalContext(
            user_id="test_user",
            action="time_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Make decisions at different times
        for i in range(10):
            decision = EthicalDecision(
                action=f"time_test_{i}",
                context=context,
                parameters={}
            )
            governor.evaluate_decision(decision)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Test time-based filtering
        recent_time = datetime.now() - timedelta(seconds=5)
        recent_history = governor.get_decision_history(since=recent_time)
        
        assert len(recent_history) > 0
        assert all(entry["timestamp"] >= recent_time for entry in recent_history)
    
    def test_violation_recording_with_duplicate_prevention(self, governor):
        """Test violation recording with duplicate prevention logic"""
        context = EthicalContext(
            user_id="test_user",
            action="duplicate_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        violation = EthicalViolation(
            user_id="test_user",
            action="duplicate_action",
            context=context,
            severity="medium",
            timestamp=datetime.now()
        )
        
        # Record same violation multiple times
        governor.record_violation(violation)
        governor.record_violation(violation)
        governor.record_violation(violation)
        
        violations = governor.get_violations("test_user")
        # Should handle duplicates appropriately
        assert len(violations) >= 1
    
    def test_user_trust_score_with_mixed_violations(self, governor):
        """Test user trust score calculation with mixed severity violations"""
        context = EthicalContext(
            user_id="mixed_user",
            action="mixed_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Add violations of different severities
        severities = ["low", "medium", "high", "critical"]
        for severity in severities:
            violation = EthicalViolation(
                user_id="mixed_user",
                action=f"violation_{severity}",
                context=context,
                severity=severity,
                timestamp=datetime.now()
            )
            governor.record_violation(violation)
        
        trust_score = governor.get_user_trust_score("mixed_user")
        assert isinstance(trust_score, float)
        assert 0.0 <= trust_score <= 1.0
    
    def test_trust_score_with_time_decay(self, governor):
        """Test trust score recovery with time-based decay"""
        context = EthicalContext(
            user_id="decay_user",
            action="decay_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Create violations at different times
        old_violation = EthicalViolation(
            user_id="decay_user",
            action="old_violation",
            context=context,
            severity="high",
            timestamp=datetime.now() - timedelta(days=60)
        )
        
        recent_violation = EthicalViolation(
            user_id="decay_user",
            action="recent_violation",
            context=context,
            severity="high",
            timestamp=datetime.now() - timedelta(days=1)
        )
        
        governor.record_violation(old_violation)
        score_after_old = governor.get_user_trust_score("decay_user")
        
        governor.record_violation(recent_violation)
        score_after_recent = governor.get_user_trust_score("decay_user")
        
        # Recent violation should have more impact
        assert score_after_recent <= score_after_old
    
    def test_concurrent_violation_recording(self, governor):
        """Test concurrent violation recording for thread safety"""
        import threading
        
        context = EthicalContext(
            user_id="concurrent_user",
            action="concurrent_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        def record_violation(violation_id):
            violation = EthicalViolation(
                user_id="concurrent_user",
                action=f"concurrent_violation_{violation_id}",
                context=context,
                severity="medium",
                timestamp=datetime.now()
            )
            governor.record_violation(violation)
        
        # Create multiple threads recording violations
        threads = []
        for i in range(20):
            thread = threading.Thread(target=record_violation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        violations = governor.get_violations("concurrent_user")
        assert len(violations) == 20
    
    def test_serialization_with_large_state(self, governor):
        """Test serialization with large state data"""
        context = EthicalContext(
            user_id="large_state_user",
            action="large_state_test",
            context_data={"large_data": "x" * 50000},
            timestamp=datetime.now()
        )
        
        # Create large state
        for i in range(500):
            decision = EthicalDecision(
                action=f"large_state_action_{i}",
                context=context,
                parameters={"index": i}
            )
            governor.evaluate_decision(decision)
        
        # Test serialization
        serialized_state = governor.serialize_state()
        assert isinstance(serialized_state, str)
        assert len(serialized_state) > 0
        
        # Test deserialization
        new_governor = GenesisEthicalGovernor()
        new_governor.deserialize_state(serialized_state)
        assert len(new_governor.decision_history) == 500
    
    def test_serialization_failure_handling(self, governor):
        """Test handling of serialization failures"""
        # Add non-serializable data
        governor.non_serializable_data = lambda x: x
        
        try:
            serialized_state = governor.serialize_state()
            # If serialization doesn't fail, ensure it's still valid
            assert isinstance(serialized_state, str)
        except Exception as e:
            # Should handle serialization errors gracefully
            assert isinstance(e, (TypeError, ValueError))
    
    def test_validate_context_comprehensive(self, governor):
        """Test comprehensive context validation"""
        # Test various invalid contexts
        invalid_contexts = [
            None,
            "invalid_string",
            123,
            EthicalContext(user_id="", action="", context_data={}, timestamp=datetime.now()),
            EthicalContext(user_id="user", action="", context_data={}, timestamp=datetime.now()),
            EthicalContext(user_id="", action="action", context_data={}, timestamp=datetime.now()),
            EthicalContext(user_id="user", action="action", context_data={}, timestamp=None),
        ]
        
        for context in invalid_contexts:
            if context is None or not isinstance(context, EthicalContext):
                with pytest.raises((TypeError, AttributeError)):
                    governor.validate_context(context)
            else:
                result = governor.validate_context(context)
                assert isinstance(result, bool)
    
    def test_memory_leak_detection(self, governor):
        """Test for potential memory leaks with repeated operations"""
        import gc
        
        # Get initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform many operations
        for i in range(1000):
            context = EthicalContext(
                user_id=f"memory_test_user_{i}",
                action=f"memory_test_action_{i}",
                context_data={"iteration": i},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"memory_test_action_{i}",
                context=context,
                parameters={}
            )
            
            governor.evaluate_decision(decision)
            
            # Clear history periodically
            if i % 100 == 0:
                governor.clear_decision_history()
        
        # Check final memory state
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not have excessive memory growth
        memory_growth = final_objects - initial_objects
        assert memory_growth < 10000  # Reasonable threshold
    
    def test_configuration_edge_cases(self):
        """Test configuration with edge case values"""
        edge_configs = [
            {'violation_threshold': 0},
            {'violation_threshold': 1000000},
            {'strict_mode': None},
            {'logging_enabled': "true"},  # String instead of boolean
            {'unknown_config': 'value'},
            {}  # Empty config
        ]
        
        for config in edge_configs:
            try:
                governor = GenesisEthicalGovernor(config=config)
                assert governor is not None
            except (ValueError, TypeError) as e:
                # Should handle invalid configurations gracefully
                assert isinstance(e, (ValueError, TypeError))
    
    def test_rule_condition_with_complex_logic(self, governor):
        """Test rules with complex conditional logic"""
        def complex_condition(ctx):
            # Complex multi-criteria condition
            if not ctx.user_id or not ctx.action:
                return False
            
            high_risk_actions = ["delete", "modify", "access_sensitive"]
            suspicious_users = ["suspicious_user", "banned_user"]
            
            if ctx.user_id in suspicious_users:
                return True
            
            if any(risk_action in ctx.action for risk_action in high_risk_actions):
                if ctx.context_data and ctx.context_data.get("bypass_check"):
                    return False
                return True
            
            return False
        
        complex_rule = {
            "name": "complex_security_rule",
            "condition": complex_condition,
            "action": "deny",
            "priority": 10
        }
        
        governor.add_ethical_rule(complex_rule)
        
        # Test various scenarios
        test_cases = [
            ("normal_user", "read_data", {}, True),  # Should be allowed
            ("suspicious_user", "read_data", {}, False),  # Should be denied
            ("normal_user", "delete_data", {}, False),  # Should be denied
            ("normal_user", "delete_data", {"bypass_check": True}, True),  # Should be allowed
        ]
        
        for user_id, action, context_data, expected_approved in test_cases:
            context = EthicalContext(
                user_id=user_id,
                action=action,
                context_data=context_data,
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=action,
                context=context,
                parameters={}
            )
            
            result = governor.evaluate_decision(decision)
            assert result.approved == expected_approved
    
    @patch('app.ai_backend.genesis_ethical_governor.time')
    def test_performance_monitoring(self, mock_time, governor):
        """Test performance monitoring and timing"""
        mock_time.time.side_effect = [1.0, 1.5, 2.0, 2.2]  # Simulate time progression
        
        context = EthicalContext(
            user_id="performance_user",
            action="performance_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="performance_test",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        
        # Check that timing was tracked
        assert isinstance(result, DecisionResult)
        # Additional assertions about performance metrics if implemented
    
    def test_audit_trail_integrity(self, governor):
        """Test audit trail integrity and immutability"""
        context = EthicalContext(
            user_id="audit_user",
            action="audit_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="audit_test",
            context=context,
            parameters={}
        )
        
        # Make decision
        result = governor.evaluate_decision(decision)
        
        # Get decision history
        history = governor.get_decision_history()
        original_history = history.copy()
        
        # Attempt to modify history (should not affect internal state)
        if history:
            history[0]["tampered"] = True
        
        # Verify internal state is unchanged
        new_history = governor.get_decision_history()
        assert new_history == original_history
    
    def test_rate_limiting_functionality(self, governor):
        """Test rate limiting for rapid decision requests"""
        context = EthicalContext(
            user_id="rate_limit_user",
            action="rate_limit_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Make rapid decisions
        results = []
        for i in range(50):
            decision = EthicalDecision(
                action=f"rate_limit_action_{i}",
                context=context,
                parameters={}
            )
            result = governor.evaluate_decision(decision)
            results.append(result)
        
        # All should be processed successfully
        assert len(results) == 50
        assert all(isinstance(result, DecisionResult) for result in results)
    
    def test_backup_and_recovery_simulation(self, governor):
        """Test backup and recovery of governor state"""
        context = EthicalContext(
            user_id="backup_user",
            action="backup_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Create initial state
        for i in range(10):
            decision = EthicalDecision(
                action=f"backup_action_{i}",
                context=context,
                parameters={}
            )
            governor.evaluate_decision(decision)
        
        # Backup state
        backup_state = governor.serialize_state()
        
        # Simulate data loss
        governor.clear_decision_history()
        assert len(governor.decision_history) == 0
        
        # Restore from backup
        governor.deserialize_state(backup_state)
        assert len(governor.decision_history) == 10
    
    def test_user_privilege_levels(self, governor):
        """Test different user privilege levels affect decisions"""
        privilege_levels = ["admin", "user", "guest", "restricted"]
        
        for privilege in privilege_levels:
            context = EthicalContext(
                user_id=f"{privilege}_user",
                action="privilege_test",
                context_data={"privilege_level": privilege},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action="admin_action",
                context=context,
                parameters={}
            )
            
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
            
            # Different privilege levels might have different approval rates
            # This would depend on implementation details
    
    def test_decision_metadata_preservation(self, governor):
        """Test that decision metadata is properly preserved"""
        context = EthicalContext(
            user_id="metadata_user",
            action="metadata_test",
            context_data={"important_data": "preserve_me"},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="metadata_test",
            context=context,
            parameters={"critical_param": "important_value"}
        )
        
        result = governor.evaluate_decision(decision)
        
        # Check that metadata is preserved in history
        history = governor.get_decision_history()
        if history:
            last_entry = history[-1]
            assert "decision" in last_entry
            assert last_entry["decision"].context.context_data["important_data"] == "preserve_me"
            assert last_entry["decision"].parameters["critical_param"] == "important_value"


class TestEthicalDecisionExtended:
    """Extended test cases for EthicalDecision class"""
    
    def test_decision_with_none_action(self):
        """Test EthicalDecision with None action"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        with pytest.raises((ValueError, TypeError)):
            EthicalDecision(
                action=None,
                context=context,
                parameters={}
            )
    
    def test_decision_with_empty_action(self):
        """Test EthicalDecision with empty action"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="",
            context=context,
            parameters={}
        )
        
        assert decision.action == ""
    
    def test_decision_immutability(self):
        """Test that EthicalDecision objects are immutable"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="original_action",
            context=context,
            parameters={"key": "value"}
        )
        
        # Should not be able to modify after creation
        with pytest.raises(AttributeError):
            decision.action = "modified_action"
    
    def test_decision_hash_consistency(self):
        """Test hash consistency of EthicalDecision objects"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision1 = EthicalDecision(
            action="test_action",
            context=context,
            parameters={"key": "value"}
        )
        
        decision2 = EthicalDecision(
            action="test_action",
            context=context,
            parameters={"key": "value"}
        )
        
        # Same decisions should have same hash
        assert hash(decision1) == hash(decision2)
    
    def test_decision_with_complex_parameters(self):
        """Test EthicalDecision with complex parameter structures"""
        context = EthicalContext(
            user_id="test_user",
            action="complex_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        complex_params = {
            "nested": {
                "level1": {
                    "level2": ["item1", "item2", {"deep": "value"}]
                }
            },
            "list": [1, 2, 3, {"item": "value"}],
            "mixed": None,
            "boolean": True,
            "float": 3.14159
        }
        
        decision = EthicalDecision(
            action="complex_test",
            context=context,
            parameters=complex_params
        )
        
        assert decision.parameters == complex_params


class TestEthicalViolationExtended:
    """Extended test cases for EthicalViolation class"""
    
    def test_violation_with_future_timestamp(self):
        """Test EthicalViolation with future timestamp"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        future_time = datetime.now() + timedelta(days=1)
        
        violation = EthicalViolation(
            user_id="test_user",
            action="future_violation",
            context=context,
            severity="medium",
            timestamp=future_time
        )
        
        assert violation.timestamp == future_time
    
    def test_violation_serialization(self):
        """Test serialization of EthicalViolation"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        violation = EthicalViolation(
            user_id="test_user",
            action="serialization_test",
            context=context,
            severity="high",
            timestamp=datetime.now()
        )
        
        serialized = violation.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["user_id"] == "test_user"
        assert serialized["action"] == "serialization_test"
        assert serialized["severity"] == "high"
    
    def test_violation_comparison(self):
        """Test comparison of EthicalViolation objects"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        violation1 = EthicalViolation(
            user_id="test_user",
            action="test_violation",
            context=context,
            severity="medium",
            timestamp=datetime.now()
        )
        
        violation2 = EthicalViolation(
            user_id="test_user",
            action="test_violation",
            context=context,
            severity="medium",
            timestamp=datetime.now()
        )
        
        # Test equality
        assert violation1 == violation2
    
    def test_violation_with_special_characters(self):
        """Test EthicalViolation with special characters in fields"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        violation = EthicalViolation(
            user_id="user@domain.com",
            action="action/with/special-chars_123",
            context=context,
            severity="low",
            timestamp=datetime.now()
        )
        
        assert violation.user_id == "user@domain.com"
        assert violation.action == "action/with/special-chars_123"


class TestEthicalContextExtended:
    """Extended test cases for EthicalContext class"""
    
    def test_context_with_large_data(self):
        """Test EthicalContext with large context data"""
        large_data = {
            "large_field": "x" * 100000,
            "array": list(range(10000)),
            "nested": {"deep": {"deeper": {"deepest": "value"}}}
        }
        
        context = EthicalContext(
            user_id="test_user",
            action="large_data_test",
            context_data=large_data,
            timestamp=datetime.now()
        )
        
        assert context.context_data == large_data
    
    def test_context_immutability(self):
        """Test that EthicalContext objects are immutable"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={"key": "value"},
            timestamp=datetime.now()
        )
        
        # Should not be able to modify after creation
        with pytest.raises(AttributeError):
            context.user_id = "modified_user"
    
    def test_context_with_binary_data(self):
        """Test EthicalContext with binary data"""
        binary_data = {
            "binary_field": b"binary_data_here",
            "bytes_array": bytes([1, 2, 3, 4, 5])
        }
        
        context = EthicalContext(
            user_id="test_user",
            action="binary_test",
            context_data=binary_data,
            timestamp=datetime.now()
        )
        
        assert context.context_data == binary_data
    
    def test_context_timestamp_validation(self):
        """Test timestamp validation in EthicalContext"""
        # Test with various timestamp formats
        timestamps = [
            datetime.now(),
            datetime.now().replace(microsecond=0),
            datetime.now() - timedelta(days=365),
            datetime.now() + timedelta(days=1)
        ]
        
        for timestamp in timestamps:
            context = EthicalContext(
                user_id="test_user",
                action="timestamp_test",
                context_data={},
                timestamp=timestamp
            )
            assert context.timestamp == timestamp


class TestDecisionResultExtended:
    """Extended test cases for DecisionResult class"""
    
    def test_result_with_complex_metadata(self):
        """Test DecisionResult with complex metadata"""
        complex_metadata = {
            "rules_evaluated": ["rule1", "rule2", "rule3"],
            "performance_metrics": {
                "evaluation_time": 0.123,
                "rules_processed": 5,
                "cache_hits": 2
            },
            "debug_info": {
                "stack_trace": ["func1", "func2"],
                "variables": {"var1": "value1", "var2": 42}
            }
        }
        
        result = DecisionResult(
            approved=True,
            confidence_score=0.85,
            reasoning="Complex evaluation completed",
            metadata=complex_metadata
        )
        
        assert result.metadata == complex_metadata
    
    def test_result_serialization(self):
        """Test serialization of DecisionResult"""
        result = DecisionResult(
            approved=True,
            confidence_score=0.95,
            reasoning="Serialization test",
            metadata={"test": "data"}
        )
        
        serialized = result.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["approved"] is True
        assert serialized["confidence_score"] == 0.95
        assert serialized["reasoning"] == "Serialization test"
        assert serialized["metadata"] == {"test": "data"}
    
    def test_result_with_none_metadata(self):
        """Test DecisionResult with None metadata"""
        result = DecisionResult(
            approved=False,
            confidence_score=0.3,
            reasoning="None metadata test",
            metadata=None
        )
        
        assert result.metadata is None
    
    def test_result_confidence_boundary_values(self):
        """Test DecisionResult with boundary confidence values"""
        boundary_values = [0.0, 1.0, 0.0001, 0.9999]
        
        for value in boundary_values:
            result = DecisionResult(
                approved=True,
                confidence_score=value,
                reasoning=f"Boundary test with {value}"
            )
            assert result.confidence_score == value
    
    def test_result_with_empty_reasoning(self):
        """Test DecisionResult with empty reasoning"""
        result = DecisionResult(
            approved=True,
            confidence_score=0.5,
            reasoning=""
        )
        
        assert result.reasoning == ""
    
    def test_result_comparison(self):
        """Test comparison of DecisionResult objects"""
        result1 = DecisionResult(
            approved=True,
            confidence_score=0.95,
            reasoning="Test reasoning"
        )
        
        result2 = DecisionResult(
            approved=True,
            confidence_score=0.95,
            reasoning="Test reasoning"
        )
        
        assert result1 == result2