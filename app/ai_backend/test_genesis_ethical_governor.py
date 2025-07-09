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

class TestGenesisEthicalGovernorAdvanced:
    """Advanced test cases for GenesisEthicalGovernor covering edge cases and complex scenarios"""
    
    @pytest.fixture
    def governor_with_config(self):
        """Create governor with specific configuration for advanced testing"""
        config = {
            'violation_threshold': 3,
            'strict_mode': True,
            'logging_enabled': True,
            'max_history_size': 100,
            'trust_score_decay_rate': 0.1
        }
        return GenesisEthicalGovernor(config=config)
    
    def test_rule_conflict_resolution(self, governor_with_config):
        """Test handling of conflicting ethical rules"""
        context = EthicalContext(
            user_id="conflict_user",
            action="conflict_action",
            context_data={"level": "test"},
            timestamp=datetime.now()
        )
        
        # Add conflicting rules with same priority
        rule1 = {
            "name": "allow_rule",
            "condition": lambda ctx: ctx.action == "conflict_action",
            "action": "allow",
            "priority": 5
        }
        rule2 = {
            "name": "deny_rule", 
            "condition": lambda ctx: ctx.action == "conflict_action",
            "action": "deny",
            "priority": 5
        }
        
        governor_with_config.add_ethical_rule(rule1)
        governor_with_config.add_ethical_rule(rule2)
        
        decision = EthicalDecision(
            action="conflict_action",
            context=context,
            parameters={}
        )
        
        result = governor_with_config.evaluate_decision(decision)
        
        # Should have a deterministic resolution strategy
        assert isinstance(result, DecisionResult)
        assert result.confidence_score < 0.8  # Lower confidence due to conflict
        assert "conflict" in result.reasoning.lower()
    
    def test_rule_execution_error_handling(self, governor_with_config):
        """Test handling of errors in rule execution"""
        context = EthicalContext(
            user_id="error_user",
            action="error_action",
            context_data={"test": "data"},
            timestamp=datetime.now()
        )
        
        # Add rule that raises exception
        def faulty_condition(ctx):
            raise ValueError("Rule execution error")
        
        faulty_rule = {
            "name": "faulty_rule",
            "condition": faulty_condition,
            "action": "deny",
            "priority": 1
        }
        
        governor_with_config.add_ethical_rule(faulty_rule)
        
        decision = EthicalDecision(
            action="error_action",
            context=context,
            parameters={}
        )
        
        # Should handle rule execution errors gracefully
        result = governor_with_config.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
        assert "error" in result.reasoning.lower()
    
    def test_trust_score_boundary_conditions(self, governor_with_config):
        """Test trust score calculations at boundary conditions"""
        user_id = "boundary_user"
        
        # Test with maximum violations
        context = EthicalContext(
            user_id=user_id,
            action="violation_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Add violations up to threshold
        for i in range(10):  # More than threshold
            violation = EthicalViolation(
                user_id=user_id,
                action=f"violation_{i}",
                context=context,
                severity="critical",
                timestamp=datetime.now() - timedelta(hours=i)
            )
            governor_with_config.record_violation(violation)
        
        trust_score = governor_with_config.get_user_trust_score(user_id)
        assert trust_score >= 0.0  # Should not go below 0
        assert trust_score <= 1.0  # Should not exceed 1
        assert trust_score < 0.3   # Should be very low with many violations
    
    def test_decision_history_size_limits(self, governor_with_config):
        """Test decision history size management"""
        context = EthicalContext(
            user_id="history_user",
            action="history_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Add more decisions than max history size
        for i in range(150):  # More than max_history_size (100)
            decision = EthicalDecision(
                action=f"action_{i}",
                context=context,
                parameters={"index": i}
            )
            governor_with_config.evaluate_decision(decision)
        
        # History should be limited to max size
        history = governor_with_config.get_decision_history()
        assert len(history) <= 100
        
        # Should contain most recent decisions
        assert history[-1]["decision"].parameters["index"] == 149
    
    def test_context_data_sanitization(self, governor_with_config):
        """Test handling of potentially malicious context data"""
        malicious_data = {
            "script": "<script>alert('xss')</script>",
            "sql": "'; DROP TABLE users; --",
            "code": "eval('malicious code')",
            "large_data": "x" * 100000  # Very large string
        }
        
        context = EthicalContext(
            user_id="malicious_user",
            action="malicious_action",
            context_data=malicious_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="malicious_action",
            context=context,
            parameters={}
        )
        
        # Should handle malicious data without crashing
        result = governor_with_config.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_concurrent_violation_recording(self, governor_with_config):
        """Test concurrent violation recording thread safety"""
        import threading
        
        context = EthicalContext(
            user_id="concurrent_user",
            action="concurrent_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        violations_recorded = []
        
        def record_violation(violation_id):
            violation = EthicalViolation(
                user_id=f"user_{violation_id}",
                action=f"action_{violation_id}",
                context=context,
                severity="medium",
                timestamp=datetime.now()
            )
            governor_with_config.record_violation(violation)
            violations_recorded.append(violation_id)
        
        # Create multiple threads recording violations
        threads = []
        for i in range(20):
            thread = threading.Thread(target=record_violation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        assert len(violations_recorded) == 20
    
    def test_rule_priority_edge_cases(self, governor_with_config):
        """Test rule priority handling with edge cases"""
        context = EthicalContext(
            user_id="priority_user",
            action="priority_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Add rules with extreme priority values
        rules = [
            {
                "name": "max_priority",
                "condition": lambda ctx: True,
                "action": "allow",
                "priority": float('inf')
            },
            {
                "name": "min_priority",
                "condition": lambda ctx: True,
                "action": "deny",
                "priority": float('-inf')
            },
            {
                "name": "zero_priority",
                "condition": lambda ctx: True,
                "action": "allow",
                "priority": 0
            }
        ]
        
        for rule in rules:
            governor_with_config.add_ethical_rule(rule)
        
        decision = EthicalDecision(
            action="priority_action",
            context=context,
            parameters={}
        )
        
        result = governor_with_config.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
        # Max priority rule should win
        assert result.approved is True
    
    def test_serialization_with_complex_data(self, governor_with_config):
        """Test serialization with complex nested data structures"""
        context = EthicalContext(
            user_id="complex_user",
            action="complex_action",
            context_data={
                "nested": {
                    "deep": {
                        "data": [1, 2, 3, {"key": "value"}]
                    }
                },
                "list": [1, 2, 3],
                "tuple": (1, 2, 3),
                "datetime": datetime.now().isoformat()
            },
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="complex_action",
            context=context,
            parameters={"complex_param": {"nested": "data"}}
        )
        
        governor_with_config.evaluate_decision(decision)
        
        # Test serialization
        serialized = governor_with_config.serialize_state()
        assert isinstance(serialized, str)
        
        # Test deserialization
        new_governor = GenesisEthicalGovernor()
        new_governor.deserialize_state(serialized)
        
        assert len(new_governor.decision_history) == 1
    
    def test_violation_severity_impact(self, governor_with_config):
        """Test impact of different violation severities on trust score"""
        user_id = "severity_user"
        context = EthicalContext(
            user_id=user_id,
            action="severity_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        initial_score = governor_with_config.get_user_trust_score(user_id)
        
        # Record low severity violation
        low_violation = EthicalViolation(
            user_id=user_id,
            action="low_action",
            context=context,
            severity="low",
            timestamp=datetime.now()
        )
        governor_with_config.record_violation(low_violation)
        
        low_score = governor_with_config.get_user_trust_score(user_id)
        
        # Record critical severity violation
        critical_violation = EthicalViolation(
            user_id=user_id,
            action="critical_action",
            context=context,
            severity="critical",
            timestamp=datetime.now()
        )
        governor_with_config.record_violation(critical_violation)
        
        critical_score = governor_with_config.get_user_trust_score(user_id)
        
        # Critical violation should have more impact
        assert initial_score > low_score > critical_score
    
    def test_rule_condition_with_none_values(self, governor_with_config):
        """Test rule conditions with None values in context"""
        context = EthicalContext(
            user_id=None,
            action=None,
            context_data=None,
            timestamp=datetime.now()
        )
        
        # Rule that handles None values
        none_rule = {
            "name": "none_handler",
            "condition": lambda ctx: ctx.user_id is None,
            "action": "deny",
            "priority": 1
        }
        
        governor_with_config.add_ethical_rule(none_rule)
        
        decision = EthicalDecision(
            action=None,
            context=context,
            parameters=None
        )
        
        result = governor_with_config.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
        assert result.approved is False
    
    def test_decision_result_metadata_handling(self, governor_with_config):
        """Test handling of decision result metadata"""
        context = EthicalContext(
            user_id="metadata_user",
            action="metadata_action",
            context_data={"test": "data"},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="metadata_action",
            context=context,
            parameters={}
        )
        
        result = governor_with_config.evaluate_decision(decision)
        
        # Should have metadata
        assert hasattr(result, 'metadata')
        if result.metadata:
            assert isinstance(result.metadata, dict)
    
    def test_context_timestamp_validation(self, governor_with_config):
        """Test validation of context timestamps"""
        # Future timestamp
        future_context = EthicalContext(
            user_id="future_user",
            action="future_action",
            context_data={},
            timestamp=datetime.now() + timedelta(days=1)
        )
        
        # Very old timestamp
        old_context = EthicalContext(
            user_id="old_user",
            action="old_action",
            context_data={},
            timestamp=datetime.now() - timedelta(days=365)
        )
        
        # Both should be handled gracefully
        assert governor_with_config.validate_context(future_context) is not None
        assert governor_with_config.validate_context(old_context) is not None
    
    def test_rule_removal_while_processing(self, governor_with_config):
        """Test removing rules while decisions are being processed"""
        context = EthicalContext(
            user_id="removal_user",
            action="removal_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Add a rule
        test_rule = {
            "name": "removable_rule",
            "condition": lambda ctx: True,
            "action": "allow",
            "priority": 1
        }
        governor_with_config.add_ethical_rule(test_rule)
        
        decision = EthicalDecision(
            action="removal_action",
            context=context,
            parameters={}
        )
        
        # Process decision
        result1 = governor_with_config.evaluate_decision(decision)
        
        # Remove rule
        governor_with_config.remove_ethical_rule("removable_rule")
        
        # Process same decision again
        result2 = governor_with_config.evaluate_decision(decision)
        
        # Both should work without errors
        assert isinstance(result1, DecisionResult)
        assert isinstance(result2, DecisionResult)
    
    def test_memory_cleanup_after_history_clear(self, governor_with_config):
        """Test memory cleanup after clearing decision history"""
        context = EthicalContext(
            user_id="cleanup_user",
            action="cleanup_action",
            context_data={"large_data": "x" * 1000},
            timestamp=datetime.now()
        )
        
        # Generate large history
        for i in range(50):
            decision = EthicalDecision(
                action=f"cleanup_action_{i}",
                context=context,
                parameters={"large_param": "y" * 1000}
            )
            governor_with_config.evaluate_decision(decision)
        
        # Clear history
        governor_with_config.clear_decision_history()
        
        # Verify cleanup
        assert len(governor_with_config.decision_history) == 0
        
        # Should still work normally
        new_decision = EthicalDecision(
            action="post_cleanup_action",
            context=context,
            parameters={}
        )
        result = governor_with_config.evaluate_decision(new_decision)
        assert isinstance(result, DecisionResult)


class TestEthicalDecisionAdvanced:
    """Advanced tests for EthicalDecision class"""
    
    def test_decision_with_circular_reference(self):
        """Test decision with circular reference in parameters"""
        context = EthicalContext(
            user_id="circular_user",
            action="circular_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Create circular reference
        circular_dict = {"key": "value"}
        circular_dict["self"] = circular_dict
        
        decision = EthicalDecision(
            action="circular_action",
            context=context,
            parameters={"circular": circular_dict}
        )
        
        # Should handle circular reference
        assert decision.action == "circular_action"
        assert decision.parameters["circular"]["key"] == "value"
    
    def test_decision_hash_consistency(self):
        """Test hash consistency of EthicalDecision objects"""
        context = EthicalContext(
            user_id="hash_user",
            action="hash_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision1 = EthicalDecision(
            action="hash_action",
            context=context,
            parameters={"param": "value"}
        )
        
        decision2 = EthicalDecision(
            action="hash_action",
            context=context,
            parameters={"param": "value"}
        )
        
        # Hash should be consistent for equal objects
        if hasattr(decision1, '__hash__'):
            assert hash(decision1) == hash(decision2)
    
    def test_decision_with_lambda_parameters(self):
        """Test decision with lambda functions in parameters"""
        context = EthicalContext(
            user_id="lambda_user",
            action="lambda_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="lambda_action",
            context=context,
            parameters={"callback": lambda x: x * 2}
        )
        
        # Should handle lambda parameters
        assert decision.action == "lambda_action"
        assert callable(decision.parameters["callback"])


class TestEthicalViolationAdvanced:
    """Advanced tests for EthicalViolation class"""
    
    def test_violation_with_unicode_data(self):
        """Test violation with unicode characters in data"""
        context = EthicalContext(
            user_id="unicode_user_🚫",
            action="unicode_action_⚠️",
            context_data={"message": "违规操作 🚨"},
            timestamp=datetime.now()
        )
        
        violation = EthicalViolation(
            user_id="unicode_user_🚫",
            action="unicode_action_⚠️",
            context=context,
            severity="high",
            timestamp=datetime.now()
        )
        
        assert violation.user_id == "unicode_user_🚫"
        assert violation.action == "unicode_action_⚠️"
        assert violation.context.context_data["message"] == "违规操作 🚨"
    
    def test_violation_timestamp_precision(self):
        """Test violation timestamp precision"""
        context = EthicalContext(
            user_id="precision_user",
            action="precision_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        timestamp1 = datetime.now()
        time.sleep(0.001)  # Small delay
        timestamp2 = datetime.now()
        
        violation1 = EthicalViolation(
            user_id="precision_user",
            action="precision_action",
            context=context,
            severity="low",
            timestamp=timestamp1
        )
        
        violation2 = EthicalViolation(
            user_id="precision_user",
            action="precision_action",
            context=context,
            severity="low",
            timestamp=timestamp2
        )
        
        assert violation1.timestamp != violation2.timestamp
    
    def test_violation_severity_case_sensitivity(self):
        """Test violation severity case sensitivity"""
        context = EthicalContext(
            user_id="case_user",
            action="case_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Test various cases
        test_cases = ["LOW", "Medium", "HIGH", "Critical"]
        for severity in test_cases:
            violation = EthicalViolation(
                user_id="case_user",
                action="case_action",
                context=context,
                severity=severity,
                timestamp=datetime.now()
            )
            # Should normalize to lowercase
            assert violation.severity == severity.lower()


class TestEthicalContextAdvanced:
    """Advanced tests for EthicalContext class"""
    
    def test_context_with_complex_nested_data(self):
        """Test context with deeply nested data structures"""
        complex_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "data": [1, 2, 3],
                        "info": {"nested": True}
                    }
                }
            },
            "arrays": [
                {"id": 1, "data": [1, 2, 3]},
                {"id": 2, "data": [4, 5, 6]}
            ]
        }
        
        context = EthicalContext(
            user_id="complex_user",
            action="complex_action",
            context_data=complex_data,
            timestamp=datetime.now()
        )
        
        assert context.context_data["level1"]["level2"]["level3"]["data"] == [1, 2, 3]
        assert context.context_data["arrays"][0]["id"] == 1
    
    def test_context_serialization_with_datetime(self):
        """Test context serialization with datetime objects"""
        now = datetime.now()
        context = EthicalContext(
            user_id="datetime_user",
            action="datetime_action",
            context_data={
                "created_at": now,
                "expires_at": now + timedelta(hours=1)
            },
            timestamp=now
        )
        
        serialized = context.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["user_id"] == "datetime_user"
        # Datetime should be serialized appropriately
        assert "created_at" in serialized["context_data"]
    
    def test_context_immutability(self):
        """Test context data immutability concerns"""
        original_data = {"key": "original_value"}
        context = EthicalContext(
            user_id="immutable_user",
            action="immutable_action",
            context_data=original_data,
            timestamp=datetime.now()
        )
        
        # Modify original data
        original_data["key"] = "modified_value"
        
        # Context should maintain integrity
        assert context.context_data is not None


class TestDecisionResultAdvanced:
    """Advanced tests for DecisionResult class"""
    
    def test_result_with_complex_metadata(self):
        """Test result with complex metadata structures"""
        complex_metadata = {
            "rules_evaluated": [
                {"name": "rule1", "result": "pass", "confidence": 0.9},
                {"name": "rule2", "result": "fail", "confidence": 0.8}
            ],
            "performance_metrics": {
                "evaluation_time_ms": 15.7,
                "rules_processed": 5
            },
            "user_context": {
                "trust_score": 0.85,
                "previous_violations": 2
            }
        }
        
        result = DecisionResult(
            approved=True,
            confidence_score=0.87,
            reasoning="Approved based on comprehensive rule evaluation",
            metadata=complex_metadata
        )
        
        assert result.metadata["rules_evaluated"][0]["name"] == "rule1"
        assert result.metadata["performance_metrics"]["evaluation_time_ms"] == 15.7
        assert result.metadata["user_context"]["trust_score"] == 0.85
    
    def test_result_confidence_score_precision(self):
        """Test confidence score precision handling"""
        precise_scores = [0.123456789, 0.987654321, 0.5000000001]
        
        for score in precise_scores:
            result = DecisionResult(
                approved=True,
                confidence_score=score,
                reasoning="Precision test"
            )
            assert result.confidence_score == score
    
    def test_result_reasoning_length_handling(self):
        """Test handling of very long reasoning strings"""
        very_long_reasoning = "This is a very long reasoning string. " * 1000
        
        result = DecisionResult(
            approved=False,
            confidence_score=0.3,
            reasoning=very_long_reasoning
        )
        
        assert len(result.reasoning) == len(very_long_reasoning)
        assert result.reasoning.startswith("This is a very long reasoning string.")


class TestPerformanceAndStress:
    """Performance and stress tests for the ethical governor system"""
    
    def test_high_frequency_decision_evaluation(self):
        """Test system performance under high-frequency decision evaluation"""
        governor = GenesisEthicalGovernor()
        
        start_time = time.time()
        
        # Simulate high-frequency decision making
        for i in range(500):
            context = EthicalContext(
                user_id=f"perf_user_{i % 50}",
                action=f"perf_action_{i % 10}",
                context_data={"iteration": i},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"perf_action_{i % 10}",
                context=context,
                parameters={"batch": i // 50}
            )
            
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should handle high frequency efficiently
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert len(governor.decision_history) == 500
    
    def test_memory_usage_with_large_context_data(self):
        """Test memory usage with large context data"""
        governor = GenesisEthicalGovernor()
        
        # Create context with large data
        large_data = {
            "file_content": "x" * 50000,  # 50KB string
            "metadata": {f"key_{i}": f"value_{i}" for i in range(1000)},
            "array_data": list(range(10000))
        }
        
        context = EthicalContext(
            user_id="memory_user",
            action="memory_action",
            context_data=large_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="memory_action",
            context=context,
            parameters={"large_param": "y" * 10000}
        )
        
        # Should handle large data without issues
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_concurrent_rule_modification(self):
        """Test concurrent rule modification safety"""
        governor = GenesisEthicalGovernor()
        
        import threading
        
        def add_rules(thread_id):
            for i in range(10):
                rule = {
                    "name": f"thread_{thread_id}_rule_{i}",
                    "condition": lambda ctx: True,
                    "action": "allow",
                    "priority": i
                }
                governor.add_ethical_rule(rule)
        
        def remove_rules(thread_id):
            time.sleep(0.1)  # Small delay
            for i in range(5):
                try:
                    governor.remove_ethical_rule(f"thread_{thread_id}_rule_{i}")
                except ValueError:
                    pass  # Rule might not exist
        
        # Start multiple threads
        threads = []
        for i in range(5):
            add_thread = threading.Thread(target=add_rules, args=(i,))
            remove_thread = threading.Thread(target=remove_rules, args=(i,))
            threads.extend([add_thread, remove_thread])
            add_thread.start()
            remove_thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # System should remain stable
        assert len(governor.ethical_rules) >= 0