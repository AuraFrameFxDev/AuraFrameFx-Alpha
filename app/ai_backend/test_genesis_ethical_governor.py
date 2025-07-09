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

# Additional comprehensive test cases

class TestGenesisEthicalGovernorAdvanced:
    """Advanced test cases for GenesisEthicalGovernor"""
    
    @pytest.fixture
    def governor_with_rules(self):
        """Create governor with pre-configured rules"""
        governor = GenesisEthicalGovernor()
        
        # Add some test rules
        test_rules = [
            {
                "name": "data_access_rule",
                "condition": lambda ctx: "sensitive" in ctx.context_data.get("data_type", ""),
                "action": "deny",
                "priority": 5
            },
            {
                "name": "admin_override",
                "condition": lambda ctx: ctx.context_data.get("user_role") == "admin",
                "action": "allow",
                "priority": 10
            },
            {
                "name": "time_restriction",
                "condition": lambda ctx: ctx.timestamp.hour < 9 or ctx.timestamp.hour > 17,
                "action": "deny",
                "priority": 3
            }
        ]
        
        for rule in test_rules:
            governor.add_ethical_rule(rule)
        
        return governor
    
    def test_rule_conflict_resolution(self, governor_with_rules):
        """Test resolution of conflicting rules based on priority"""
        context = EthicalContext(
            user_id="test_user",
            action="access_data",
            context_data={
                "data_type": "sensitive_data",
                "user_role": "admin"
            },
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="access_data",
            context=context,
            parameters={}
        )
        
        result = governor_with_rules.evaluate_decision(decision)
        
        # Admin rule (priority 10) should override data access rule (priority 5)
        assert result.approved is True
    
    def test_multiple_violation_threshold(self, governor):
        """Test behavior when user exceeds violation threshold"""
        context = EthicalContext(
            user_id="threshold_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Record multiple violations
        for i in range(governor.violation_threshold + 1):
            violation = EthicalViolation(
                user_id="threshold_user",
                action=f"violation_{i}",
                context=context,
                severity="medium",
                timestamp=datetime.now()
            )
            governor.record_violation(violation)
        
        # Check if user is flagged
        trust_score = governor.get_user_trust_score("threshold_user")
        assert trust_score < 0.5  # Should be significantly impacted
    
    def test_time_based_rule_evaluation(self, governor):
        """Test rules that depend on time context"""
        # Create time-sensitive rule
        time_rule = {
            "name": "business_hours_only",
            "condition": lambda ctx: 9 <= ctx.timestamp.hour <= 17,
            "action": "allow",
            "priority": 5
        }
        governor.add_ethical_rule(time_rule)
        
        # Test during business hours
        business_hours_context = EthicalContext(
            user_id="time_user",
            action="time_sensitive_action",
            context_data={},
            timestamp=datetime.now().replace(hour=14)  # 2 PM
        )
        
        decision = EthicalDecision(
            action="time_sensitive_action",
            context=business_hours_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert result.approved is True
        
        # Test outside business hours
        after_hours_context = EthicalContext(
            user_id="time_user",
            action="time_sensitive_action",
            context_data={},
            timestamp=datetime.now().replace(hour=22)  # 10 PM
        )
        
        decision_after_hours = EthicalDecision(
            action="time_sensitive_action",
            context=after_hours_context,
            parameters={}
        )
        
        result_after_hours = governor.evaluate_decision(decision_after_hours)
        assert result_after_hours.approved is False
    
    def test_recursive_rule_evaluation(self, governor):
        """Test that rules don't cause infinite recursion"""
        # Create a rule that might trigger recursion
        recursive_rule = {
            "name": "recursive_rule",
            "condition": lambda ctx: ctx.action == "recursive_action",
            "action": "evaluate_sub_action",
            "priority": 5
        }
        governor.add_ethical_rule(recursive_rule)
        
        context = EthicalContext(
            user_id="recursive_user",
            action="recursive_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="recursive_action",
            context=context,
            parameters={}
        )
        
        # Should not hang or cause stack overflow
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_malformed_context_data_handling(self, governor):
        """Test handling of malformed or malicious context data"""
        malformed_contexts = [
            {"circular_ref": None},
            {"very_deep": {"level1": {"level2": {"level3": {"level4": "data"}}}}},
            {"special_chars": "';DROP TABLE users;--"},
            {"unicode": "🚀🔥💯"},
            {"large_number": 999999999999999999999999999999999999999999999999},
            {"binary_data": b"\x00\x01\x02\x03"},
            {"function": lambda x: x},
            {"class_instance": datetime.now()},
        ]
        
        for i, malformed_data in enumerate(malformed_contexts):
            context = EthicalContext(
                user_id=f"malformed_user_{i}",
                action="malformed_test",
                context_data=malformed_data,
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action="malformed_test",
                context=context,
                parameters={}
            )
            
            # Should handle gracefully without crashing
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
    
    def test_rate_limiting_simulation(self, governor):
        """Test behavior under rapid decision requests"""
        context = EthicalContext(
            user_id="rate_limit_user",
            action="rapid_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Simulate rapid requests
        results = []
        for i in range(100):
            decision = EthicalDecision(
                action=f"rapid_action_{i}",
                context=context,
                parameters={"request_id": i}
            )
            result = governor.evaluate_decision(decision)
            results.append(result)
        
        # All requests should be processed
        assert len(results) == 100
        assert all(isinstance(result, DecisionResult) for result in results)
    
    def test_cross_user_interaction_effects(self, governor):
        """Test that one user's actions don't improperly affect another user"""
        user1_context = EthicalContext(
            user_id="user1",
            action="user1_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        user2_context = EthicalContext(
            user_id="user2",
            action="user2_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Record violation for user1
        violation = EthicalViolation(
            user_id="user1",
            action="bad_action",
            context=user1_context,
            severity="high",
            timestamp=datetime.now()
        )
        governor.record_violation(violation)
        
        # Check that user2's trust score is unaffected
        user1_score = governor.get_user_trust_score("user1")
        user2_score = governor.get_user_trust_score("user2")
        
        assert user1_score < user2_score
    
    def test_rule_modification_during_evaluation(self, governor):
        """Test thread safety when rules are modified during evaluation"""
        import threading
        
        context = EthicalContext(
            user_id="concurrent_user",
            action="concurrent_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        results = []
        errors = []
        
        def evaluate_decision():
            try:
                decision = EthicalDecision(
                    action="concurrent_action",
                    context=context,
                    parameters={}
                )
                result = governor.evaluate_decision(decision)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        def modify_rules():
            try:
                for i in range(10):
                    rule = {
                        "name": f"dynamic_rule_{i}",
                        "condition": lambda ctx: False,
                        "action": "allow",
                        "priority": i
                    }
                    governor.add_ethical_rule(rule)
                    governor.remove_ethical_rule(f"dynamic_rule_{i}")
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=evaluate_decision))
        
        threads.append(threading.Thread(target=modify_rules))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should not have any errors
        assert len(errors) == 0
        assert len(results) == 5
    
    def test_memory_leak_prevention(self, governor):
        """Test that repeated operations don't cause memory leaks"""
        import gc
        
        initial_objects = len(gc.get_objects())
        
        # Perform many operations
        for i in range(1000):
            context = EthicalContext(
                user_id=f"memory_user_{i % 10}",
                action=f"memory_action_{i}",
                context_data={"iteration": i},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"memory_action_{i}",
                context=context,
                parameters={}
            )
            
            governor.evaluate_decision(decision)
            
            # Periodically clear history to prevent legitimate growth
            if i % 100 == 0:
                governor.clear_decision_history()
        
        # Force garbage collection
        gc.collect()
        
        final_objects = len(gc.get_objects())
        
        # Object count should not have grown excessively
        # Allow some growth for legitimate caching/optimization
        assert final_objects < initial_objects * 1.5
    
    def test_rule_dependency_chains(self, governor):
        """Test complex rule dependency scenarios"""
        # Create interdependent rules
        rules = [
            {
                "name": "prerequisite_rule",
                "condition": lambda ctx: ctx.context_data.get("has_prerequisite", False),
                "action": "allow",
                "priority": 1
            },
            {
                "name": "dependent_rule",
                "condition": lambda ctx: ctx.context_data.get("needs_prerequisite", False),
                "action": "check_prerequisite",
                "priority": 2
            }
        ]
        
        for rule in rules:
            governor.add_ethical_rule(rule)
        
        # Test with prerequisite
        context_with_prereq = EthicalContext(
            user_id="dependency_user",
            action="dependent_action",
            context_data={
                "has_prerequisite": True,
                "needs_prerequisite": True
            },
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="dependent_action",
            context=context_with_prereq,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_rule_performance_impact(self, governor):
        """Test performance impact of many rules"""
        import time
        
        # Add many rules
        for i in range(100):
            rule = {
                "name": f"performance_rule_{i}",
                "condition": lambda ctx, i=i: ctx.context_data.get("rule_id") == i,
                "action": "allow" if i % 2 == 0 else "deny",
                "priority": i
            }
            governor.add_ethical_rule(rule)
        
        context = EthicalContext(
            user_id="performance_user",
            action="performance_action",
            context_data={"rule_id": 50},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="performance_action",
            context=context,
            parameters={}
        )
        
        start_time = time.time()
        result = governor.evaluate_decision(decision)
        end_time = time.time()
        
        # Should still be reasonably fast
        assert end_time - start_time < 1.0  # Less than 1 second
        assert isinstance(result, DecisionResult)
    
    def test_context_inheritance_scenarios(self, governor):
        """Test scenarios where context is inherited or derived"""
        parent_context = EthicalContext(
            user_id="parent_user",
            action="parent_action",
            context_data={"parent_key": "parent_value"},
            timestamp=datetime.now()
        )
        
        # Create child context that inherits from parent
        child_context = EthicalContext(
            user_id="child_user",
            action="child_action",
            context_data={
                "parent_key": "parent_value",
                "child_key": "child_value",
                "inherited_from": "parent_user"
            },
            timestamp=datetime.now()
        )
        
        parent_decision = EthicalDecision(
            action="parent_action",
            context=parent_context,
            parameters={}
        )
        
        child_decision = EthicalDecision(
            action="child_action",
            context=child_context,
            parameters={}
        )
        
        parent_result = governor.evaluate_decision(parent_decision)
        child_result = governor.evaluate_decision(child_decision)
        
        assert isinstance(parent_result, DecisionResult)
        assert isinstance(child_result, DecisionResult)


class TestGenesisEthicalGovernorErrorRecovery:
    """Test error recovery and resilience scenarios"""
    
    @pytest.fixture
    def governor(self):
        return GenesisEthicalGovernor()
    
    def test_recovery_from_corrupted_state(self, governor):
        """Test recovery from corrupted internal state"""
        # Corrupt the internal state
        governor.decision_history = "corrupted_string"
        
        context = EthicalContext(
            user_id="recovery_user",
            action="recovery_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="recovery_action",
            context=context,
            parameters={}
        )
        
        # Should recover gracefully
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_handling_rule_evaluation_exceptions(self, governor):
        """Test handling of exceptions during rule evaluation"""
        # Add a rule that throws an exception
        problematic_rule = {
            "name": "exception_rule",
            "condition": lambda ctx: 1 / 0,  # Division by zero
            "action": "deny",
            "priority": 5
        }
        governor.add_ethical_rule(problematic_rule)
        
        context = EthicalContext(
            user_id="exception_user",
            action="exception_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="exception_action",
            context=context,
            parameters={}
        )
        
        # Should handle exception gracefully
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_resource_exhaustion_handling(self, governor):
        """Test handling of resource exhaustion scenarios"""
        # Create a context with extremely large data
        large_data = {"large_field": "x" * 1000000}  # 1MB of data
        
        context = EthicalContext(
            user_id="resource_user",
            action="resource_action",
            context_data=large_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="resource_action",
            context=context,
            parameters={}
        )
        
        # Should handle without crashing
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_network_timeout_simulation(self, governor):
        """Test handling of network timeout scenarios"""
        # Mock a network-dependent operation
        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = TimeoutError("Network timeout")
            
            context = EthicalContext(
                user_id="timeout_user",
                action="network_action",
                context_data={},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action="network_action",
                context=context,
                parameters={}
            )
            
            # Should handle timeout gracefully
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)


class TestGenesisEthicalGovernorEdgeCases:
    """Test additional edge cases and boundary conditions"""
    
    @pytest.fixture
    def governor(self):
        return GenesisEthicalGovernor()
    
    def test_unicode_and_special_characters(self, governor):
        """Test handling of unicode and special characters"""
        special_chars = [
            "🎯🔥💯",  # Emojis
            "测试用户",  # Chinese characters
            "пользователь",  # Russian characters
            "café",  # Accented characters
            "user\x00null",  # Null bytes
            "user\t\n\r",  # Control characters
            "user'\"<>&",  # HTML/SQL special chars
        ]
        
        for char_set in special_chars:
            context = EthicalContext(
                user_id=f"special_{char_set}",
                action=f"action_{char_set}",
                context_data={"special_data": char_set},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"action_{char_set}",
                context=context,
                parameters={}
            )
            
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
    
    def test_extreme_timestamp_values(self, governor):
        """Test handling of extreme timestamp values"""
        extreme_timestamps = [
            datetime.min,
            datetime.max,
            datetime(1900, 1, 1),
            datetime(2100, 12, 31),
            datetime.now() + timedelta(days=36500),  # 100 years in future
            datetime.now() - timedelta(days=36500),  # 100 years in past
        ]
        
        for timestamp in extreme_timestamps:
            try:
                context = EthicalContext(
                    user_id="timestamp_user",
                    action="timestamp_action",
                    context_data={},
                    timestamp=timestamp
                )
                
                decision = EthicalDecision(
                    action="timestamp_action",
                    context=context,
                    parameters={}
                )
                
                result = governor.evaluate_decision(decision)
                assert isinstance(result, DecisionResult)
            except (ValueError, OverflowError):
                # Some extreme values may legitimately fail
                pass
    
    def test_deeply_nested_parameters(self, governor):
        """Test handling of deeply nested parameter structures"""
        # Create deeply nested structure
        nested_params = {}
        current = nested_params
        for i in range(100):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]
        current["final_value"] = "deep_value"
        
        context = EthicalContext(
            user_id="nested_user",
            action="nested_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="nested_action",
            context=context,
            parameters=nested_params
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_circular_reference_handling(self, governor):
        """Test handling of circular references in data structures"""
        # Create circular reference
        circular_data = {"self": None}
        circular_data["self"] = circular_data
        
        context = EthicalContext(
            user_id="circular_user",
            action="circular_action",
            context_data=circular_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="circular_action",
            context=context,
            parameters={}
        )
        
        # Should handle without infinite recursion
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_mixed_data_types_in_context(self, governor):
        """Test handling of mixed data types in context"""
        mixed_data = {
            "string": "value",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
            "bytes": b"binary_data",
            "datetime": datetime.now(),
            "complex": complex(1, 2),
        }
        
        context = EthicalContext(
            user_id="mixed_user",
            action="mixed_action",
            context_data=mixed_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="mixed_action",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_zero_and_negative_values(self, governor):
        """Test handling of zero and negative values"""
        zero_negative_data = {
            "zero": 0,
            "negative_int": -42,
            "negative_float": -3.14,
            "empty_string": "",
            "empty_list": [],
            "empty_dict": {},
            "zero_timestamp": datetime.fromtimestamp(0),
        }
        
        context = EthicalContext(
            user_id="zero_user",
            action="zero_action",
            context_data=zero_negative_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="zero_action",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)


class TestGenesisEthicalGovernorAuditAndCompliance:
    """Test audit trail and compliance features"""
    
    @pytest.fixture
    def governor(self):
        return GenesisEthicalGovernor()
    
    def test_audit_trail_completeness(self, governor):
        """Test that audit trail captures all necessary information"""
        context = EthicalContext(
            user_id="audit_user",
            action="audit_action",
            context_data={"sensitive": True},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="audit_action",
            context=context,
            parameters={"audit_test": True}
        )
        
        result = governor.evaluate_decision(decision)
        
        # Check audit trail
        history = governor.get_decision_history()
        latest_entry = history[-1]
        
        assert "timestamp" in latest_entry
        assert "decision" in latest_entry
        assert "result" in latest_entry
        assert "user_id" in latest_entry
        assert latest_entry["user_id"] == "audit_user"
    
    def test_compliance_reporting(self, governor):
        """Test compliance reporting capabilities"""
        # Create various types of decisions
        decisions_data = [
            ("user1", "read_data", "approved"),
            ("user2", "delete_data", "denied"),
            ("user1", "modify_data", "approved"),
            ("user3", "export_data", "denied"),
        ]
        
        for user_id, action, expected_result in decisions_data:
            context = EthicalContext(
                user_id=user_id,
                action=action,
                context_data={},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=action,
                context=context,
                parameters={}
            )
            
            governor.evaluate_decision(decision)
        
        # Generate compliance report
        history = governor.get_decision_history()
        
        # Verify report structure
        assert len(history) == 4
        
        # Check that all required fields are present
        for entry in history:
            assert "timestamp" in entry
            assert "decision" in entry
            assert "result" in entry
            assert "user_id" in entry
    
    def test_data_retention_policies(self, governor):
        """Test data retention and cleanup policies"""
        # Create old decisions
        old_timestamp = datetime.now() - timedelta(days=400)
        
        context = EthicalContext(
            user_id="retention_user",
            action="old_action",
            context_data={},
            timestamp=old_timestamp
        )
        
        decision = EthicalDecision(
            action="old_action",
            context=context,
            parameters={}
        )
        
        governor.evaluate_decision(decision)
        
        # Simulate data cleanup based on retention policy
        initial_history_length = len(governor.decision_history)
        
        # Apply retention policy (if implemented)
        if hasattr(governor, 'apply_retention_policy'):
            governor.apply_retention_policy(days=365)
            
            # Check that old data was cleaned up
            final_history_length = len(governor.decision_history)
            assert final_history_length <= initial_history_length
    
    def test_anonymization_capabilities(self, governor):
        """Test data anonymization for privacy compliance"""
        context = EthicalContext(
            user_id="sensitive_user@example.com",
            action="privacy_action",
            context_data={
                "email": "user@example.com",
                "phone": "123-456-7890",
                "ssn": "123-45-6789"
            },
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="privacy_action",
            context=context,
            parameters={}
        )
        
        governor.evaluate_decision(decision)
        
        # Test anonymization (if implemented)
        if hasattr(governor, 'anonymize_history'):
            governor.anonymize_history()
            
            history = governor.get_decision_history()
            latest_entry = history[-1]
            
            # Check that sensitive data is anonymized
            assert "sensitive_user@example.com" not in str(latest_entry)
            assert "123-456-7890" not in str(latest_entry)
            assert "123-45-6789" not in str(latest_entry)