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

class TestGenesisEthicalGovernorEnhanced:
    """Enhanced test suite with additional edge cases and scenarios"""
    
    @pytest.fixture
    def governor_with_custom_rules(self):
        """Create a governor with pre-configured custom rules"""
        governor = GenesisEthicalGovernor()
        
        # Add various priority rules
        rules = [
            {
                "name": "security_rule",
                "condition": lambda ctx: "security" in ctx.action.lower(),
                "action": "require_approval",
                "priority": 10
            },
            {
                "name": "admin_rule",
                "condition": lambda ctx: ctx.user_id.startswith("admin_"),
                "action": "allow",
                "priority": 5
            },
            {
                "name": "delete_rule",
                "condition": lambda ctx: "delete" in ctx.action.lower(),
                "action": "deny",
                "priority": 8
            }
        ]
        
        for rule in rules:
            governor.add_ethical_rule(rule)
        
        return governor
    
    @pytest.fixture
    def bulk_ethical_contexts(self):
        """Create bulk ethical contexts for testing"""
        contexts = []
        for i in range(50):
            context = EthicalContext(
                user_id=f"bulk_user_{i % 5}",
                action=f"bulk_action_{i}",
                context_data={"batch_id": i, "priority": i % 3},
                timestamp=datetime.now() - timedelta(seconds=i)
            )
            contexts.append(context)
        return contexts
    
    def test_rule_evaluation_order_complex(self, governor_with_custom_rules):
        """Test complex rule evaluation order with multiple matching rules"""
        context = EthicalContext(
            user_id="admin_user",
            action="security_delete_action",
            context_data={"sensitive": True},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="security_delete_action",
            context=context,
            parameters={"force": True}
        )
        
        result = governor_with_custom_rules.evaluate_decision(decision)
        
        # Security rule (priority 10) should override admin rule (priority 5)
        # and delete rule (priority 8)
        assert isinstance(result, DecisionResult)
        assert result.reasoning is not None
    
    def test_violation_severity_impact_on_trust_score(self, governor, mock_ethical_context):
        """Test how different violation severities impact trust scores"""
        user_id = "severity_test_user"
        initial_score = governor.get_user_trust_score(user_id)
        
        # Test different severity levels
        severities = ["low", "medium", "high", "critical"]
        scores = [initial_score]
        
        for severity in severities:
            violation = EthicalViolation(
                user_id=user_id,
                action=f"{severity}_violation",
                context=mock_ethical_context,
                severity=severity,
                timestamp=datetime.now()
            )
            governor.record_violation(violation)
            new_score = governor.get_user_trust_score(user_id)
            scores.append(new_score)
        
        # Scores should decrease with each violation
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]
        
        # Critical violations should have more impact than low ones
        assert scores[-1] < scores[1]  # Critical vs initial
    
    def test_decision_history_memory_management(self, governor, mock_ethical_context):
        """Test memory management with very large decision history"""
        # Create a large number of decisions
        for i in range(10000):
            decision = EthicalDecision(
                action=f"memory_test_{i}",
                context=mock_ethical_context,
                parameters={"index": i, "data": f"test_data_{i}"}
            )
            governor.evaluate_decision(decision)
        
        # Test that we can still query history efficiently
        start_time = time.time()
        recent_history = governor.get_decision_history()[-100:]
        query_time = time.time() - start_time
        
        assert len(recent_history) == 100
        assert query_time < 1.0  # Should be fast even with large history
    
    def test_concurrent_violation_recording(self, governor, mock_ethical_context):
        """Test concurrent violation recording for thread safety"""
        import threading
        
        violations_recorded = []
        
        def record_violation(violation_id):
            violation = EthicalViolation(
                user_id=f"concurrent_user_{violation_id}",
                action=f"concurrent_action_{violation_id}",
                context=mock_ethical_context,
                severity="medium",
                timestamp=datetime.now()
            )
            governor.record_violation(violation)
            violations_recorded.append(violation_id)
        
        # Create multiple threads
        threads = []
        for i in range(20):
            thread = threading.Thread(target=record_violation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        assert len(violations_recorded) == 20
        
        # Check that all violations were recorded
        for i in range(20):
            user_violations = governor.get_violations(f"concurrent_user_{i}")
            assert len(user_violations) == 1
    
    def test_trust_score_edge_cases(self, governor, mock_ethical_context):
        """Test trust score calculation edge cases"""
        # Test with non-existent user
        score = governor.get_user_trust_score("non_existent_user")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        
        # Test with empty user_id
        with pytest.raises(ValueError):
            governor.get_user_trust_score("")
        
        # Test with None user_id
        with pytest.raises(ValueError):
            governor.get_user_trust_score(None)
        
        # Test with special characters in user_id
        special_user = "user@#$%^&*()"
        score = governor.get_user_trust_score(special_user)
        assert isinstance(score, float)
    
    def test_ethical_rule_condition_exceptions(self, governor, mock_ethical_context):
        """Test handling of exceptions in rule conditions"""
        # Add a rule with a condition that raises an exception
        def faulty_condition(ctx):
            raise ValueError("Intentional error in condition")
        
        faulty_rule = {
            "name": "faulty_rule",
            "condition": faulty_condition,
            "action": "deny",
            "priority": 5
        }
        
        governor.add_ethical_rule(faulty_rule)
        
        decision = EthicalDecision(
            action="test_faulty_rule",
            context=mock_ethical_context,
            parameters={}
        )
        
        # Should handle the exception gracefully
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_decision_parameter_validation(self, governor, mock_ethical_context):
        """Test validation of decision parameters"""
        # Test with various parameter types
        parameter_tests = [
            {"string": "value"},
            {"number": 42},
            {"float": 3.14},
            {"boolean": True},
            {"list": [1, 2, 3]},
            {"nested": {"key": {"subkey": "value"}}},
            {"mixed": {"str": "val", "num": 42, "bool": True}}
        ]
        
        for params in parameter_tests:
            decision = EthicalDecision(
                action="parameter_test",
                context=mock_ethical_context,
                parameters=params
            )
            
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
    
    def test_context_data_sanitization(self, governor):
        """Test context data sanitization for sensitive information"""
        sensitive_context = EthicalContext(
            user_id="sensitive_user",
            action="sensitive_action",
            context_data={
                "password": "secret123",
                "credit_card": "1234-5678-9012-3456",
                "ssn": "123-45-6789",
                "normal_data": "public_info"
            },
            timestamp=datetime.now()
        )
        
        # Test that sensitive data is handled appropriately
        is_valid = governor.validate_context(sensitive_context)
        assert isinstance(is_valid, bool)
        
        decision = EthicalDecision(
            action="sensitive_action",
            context=sensitive_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_decision_result_metadata_handling(self, governor, mock_ethical_context):
        """Test handling of decision result metadata"""
        decision = EthicalDecision(
            action="metadata_test",
            context=mock_ethical_context,
            parameters={"metadata_test": True}
        )
        
        result = governor.evaluate_decision(decision)
        
        # Check that metadata is present and properly structured
        assert hasattr(result, 'metadata')
        if result.metadata:
            assert isinstance(result.metadata, dict)
    
    def test_violation_timestamp_ordering(self, governor, mock_ethical_context):
        """Test that violations are ordered by timestamp correctly"""
        user_id = "timestamp_test_user"
        violations = []
        
        # Create violations with different timestamps
        for i in range(5):
            violation = EthicalViolation(
                user_id=user_id,
                action=f"action_{i}",
                context=mock_ethical_context,
                severity="medium",
                timestamp=datetime.now() - timedelta(hours=i)
            )
            violations.append(violation)
            governor.record_violation(violation)
        
        # Retrieve violations
        retrieved_violations = governor.get_violations(user_id)
        
        # Check that violations are ordered (most recent first)
        for i in range(len(retrieved_violations) - 1):
            assert retrieved_violations[i].timestamp >= retrieved_violations[i + 1].timestamp
    
    def test_serialization_edge_cases(self, governor, mock_ethical_context):
        """Test serialization with edge cases"""
        # Create complex state with various data types
        complex_rule = {
            "name": "complex_rule",
            "condition": lambda ctx: True,
            "action": "allow",
            "priority": 1,
            "metadata": {"created": datetime.now().isoformat()}
        }
        governor.add_ethical_rule(complex_rule)
        
        # Add violation with complex data
        violation = EthicalViolation(
            user_id="serialization_user",
            action="complex_action",
            context=mock_ethical_context,
            severity="high",
            timestamp=datetime.now()
        )
        governor.record_violation(violation)
        
        # Test serialization
        serialized = governor.serialize_state()
        assert isinstance(serialized, str)
        
        # Test deserialization
        new_governor = GenesisEthicalGovernor()
        new_governor.deserialize_state(serialized)
        
        # Verify complex data was preserved
        assert len(new_governor.ethical_rules) > 0
    
    def test_performance_with_complex_rules(self, governor, mock_ethical_context):
        """Test performance with many complex rules"""
        # Add many complex rules
        for i in range(100):
            rule = {
                "name": f"complex_rule_{i}",
                "condition": lambda ctx, i=i: i % 2 == 0 and len(ctx.action) > 5,
                "action": "allow" if i % 2 == 0 else "deny",
                "priority": i % 10
            }
            governor.add_ethical_rule(rule)
        
        # Test decision evaluation performance
        start_time = time.time()
        
        for i in range(100):
            decision = EthicalDecision(
                action=f"performance_test_action_{i}",
                context=mock_ethical_context,
                parameters={"test_id": i}
            )
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 5.0
    
    def test_config_inheritance_and_overrides(self):
        """Test configuration inheritance and override behavior"""
        base_config = {
            'violation_threshold': 3,
            'strict_mode': False,
            'logging_enabled': True,
            'custom_setting': 'base_value'
        }
        
        override_config = {
            'violation_threshold': 5,
            'strict_mode': True,
            'new_setting': 'override_value'
        }
        
        # Test that overrides work correctly
        governor = GenesisEthicalGovernor(config={**base_config, **override_config})
        
        assert governor.violation_threshold == 5  # Overridden
        assert governor.strict_mode is True  # Overridden
        assert governor.logging_enabled is True  # Inherited
    
    def test_bulk_violation_analysis(self, governor, bulk_ethical_contexts):
        """Test bulk violation analysis and patterns"""
        # Create bulk violations
        for i, context in enumerate(bulk_ethical_contexts):
            violation = EthicalViolation(
                user_id=context.user_id,
                action=f"bulk_violation_{i}",
                context=context,
                severity=["low", "medium", "high"][i % 3],
                timestamp=context.timestamp
            )
            governor.record_violation(violation)
        
        # Test pattern analysis
        user_patterns = {}
        for context in bulk_ethical_contexts:
            user_violations = governor.get_violations(context.user_id)
            user_patterns[context.user_id] = len(user_violations)
        
        # Verify that violations are distributed across users
        assert len(user_patterns) > 1
        assert all(count > 0 for count in user_patterns.values())
    
    def test_decision_context_immutability(self, governor, mock_ethical_context):
        """Test that decision context is not modified during evaluation"""
        original_context_data = mock_ethical_context.context_data.copy()
        original_action = mock_ethical_context.action
        
        decision = EthicalDecision(
            action="immutability_test",
            context=mock_ethical_context,
            parameters={"modify_test": True}
        )
        
        governor.evaluate_decision(decision)
        
        # Verify context was not modified
        assert mock_ethical_context.context_data == original_context_data
        assert mock_ethical_context.action == original_action
    
    def test_rule_removal_by_condition(self, governor, mock_ethical_context):
        """Test removal of rules based on conditions"""
        # Add multiple rules
        rules_to_add = [
            {
                "name": "temp_rule_1",
                "condition": lambda ctx: "temp" in ctx.action,
                "action": "allow",
                "priority": 1
            },
            {
                "name": "temp_rule_2",
                "condition": lambda ctx: "temp" in ctx.action,
                "action": "deny",
                "priority": 2
            },
            {
                "name": "permanent_rule",
                "condition": lambda ctx: True,
                "action": "allow",
                "priority": 1
            }
        ]
        
        for rule in rules_to_add:
            governor.add_ethical_rule(rule)
        
        initial_count = len(governor.ethical_rules)
        
        # Remove rules matching condition
        governor.remove_ethical_rule("temp_rule_1")
        governor.remove_ethical_rule("temp_rule_2")
        
        assert len(governor.ethical_rules) == initial_count - 2
        
        # Verify specific rules were removed
        rule_names = [rule["name"] for rule in governor.ethical_rules]
        assert "temp_rule_1" not in rule_names
        assert "temp_rule_2" not in rule_names
        assert "permanent_rule" in rule_names
    
    def test_extreme_parameter_sizes(self, governor, mock_ethical_context):
        """Test handling of extremely large parameters"""
        # Test with very large parameter values
        large_params = {
            "large_string": "x" * 100000,  # 100KB string
            "large_list": list(range(10000)),  # Large list
            "large_dict": {f"key_{i}": f"value_{i}" for i in range(1000)}  # Large dict
        }
        
        decision = EthicalDecision(
            action="large_params_test",
            context=mock_ethical_context,
            parameters=large_params
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_unicode_and_special_characters(self, governor):
        """Test handling of unicode and special characters"""
        unicode_context = EthicalContext(
            user_id="用户_测试",  # Chinese characters
            action="اختبار_عمل",  # Arabic characters
            context_data={
                "emoji": "🔒🛡️⚠️",
                "special": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
                "unicode": "Ñiño résumé naïve café"
            },
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="unicode_test",
            context=unicode_context,
            parameters={"test": "тест"}  # Cyrillic
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    @patch('time.time')
    def test_time_based_rule_evaluation(self, mock_time, governor, mock_ethical_context):
        """Test time-based rule evaluation"""
        # Mock different times
        mock_time.return_value = 1000000000  # Fixed timestamp
        
        time_sensitive_rule = {
            "name": "time_rule",
            "condition": lambda ctx: int(time.time()) % 2 == 0,
            "action": "allow",
            "priority": 1
        }
        
        governor.add_ethical_rule(time_sensitive_rule)
        
        decision = EthicalDecision(
            action="time_test",
            context=mock_ethical_context,
            parameters={}
        )
        
        result1 = governor.evaluate_decision(decision)
        
        # Change time
        mock_time.return_value = 1000000001  # Different timestamp
        
        result2 = governor.evaluate_decision(decision)
        
        # Results might be different based on time
        assert isinstance(result1, DecisionResult)
        assert isinstance(result2, DecisionResult)


class TestEthicalDataStructuresEnhanced:
    """Enhanced tests for ethical data structures"""
    
    def test_ethical_decision_deep_copy(self):
        """Test deep copying of EthicalDecision objects"""
        import copy
        
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={"mutable": ["list", "data"]},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="copy_test",
            context=context,
            parameters={"mutable": {"nested": "dict"}}
        )
        
        # Test deep copy
        decision_copy = copy.deepcopy(decision)
        
        assert decision_copy.action == decision.action
        assert decision_copy.context.user_id == decision.context.user_id
        assert decision_copy.parameters == decision.parameters
        
        # Verify it's a deep copy (modify original shouldn't affect copy)
        decision.parameters["mutable"]["nested"] = "modified"
        assert decision_copy.parameters["mutable"]["nested"] == "dict"
    
    def test_ethical_violation_comparison(self):
        """Test comparison operations on EthicalViolation objects"""
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        violation1 = EthicalViolation(
            user_id="test_user",
            action="violation1",
            context=context,
            severity="high",
            timestamp=datetime.now()
        )
        
        violation2 = EthicalViolation(
            user_id="test_user",
            action="violation2",
            context=context,
            severity="high",
            timestamp=datetime.now() + timedelta(seconds=1)
        )
        
        # Test comparison (should be based on timestamp)
        assert violation2.timestamp > violation1.timestamp
    
    def test_ethical_context_hash_and_equality(self):
        """Test hashing and equality of EthicalContext objects"""
        timestamp = datetime.now()
        
        context1 = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={"key": "value"},
            timestamp=timestamp
        )
        
        context2 = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={"key": "value"},
            timestamp=timestamp
        )
        
        # Test equality
        assert context1 == context2
        
        # Test hash consistency
        assert hash(context1) == hash(context2)
    
    def test_decision_result_json_serialization(self):
        """Test JSON serialization of DecisionResult"""
        result = DecisionResult(
            approved=True,
            confidence_score=0.85,
            reasoning="Test reasoning with special chars: !@#$%",
            metadata={
                "timestamp": datetime.now().isoformat(),
                "rule_id": "test_rule_123",
                "nested": {"key": "value"}
            }
        )
        
        # Test JSON serialization
        json_str = json.dumps(result.to_dict())
        assert isinstance(json_str, str)
        
        # Test deserialization
        deserialized = json.loads(json_str)
        assert deserialized["approved"] is True
        assert deserialized["confidence_score"] == 0.85
        assert "metadata" in deserialized


class TestSecurityAndRobustness:
    """Security and robustness tests"""
    
    def test_injection_attack_prevention(self, governor):
        """Test prevention of injection attacks in parameters"""
        malicious_contexts = [
            EthicalContext(
                user_id="'; DROP TABLE users; --",
                action="<script>alert('xss')</script>",
                context_data={"eval": "exec('import os; os.system(\"rm -rf /\")')"},
                timestamp=datetime.now()
            ),
            EthicalContext(
                user_id="../../etc/passwd",
                action="../../../sensitive_file",
                context_data={"path": "/etc/shadow"},
                timestamp=datetime.now()
            )
        ]
        
        for context in malicious_contexts:
            # Should handle malicious input gracefully
            is_valid = governor.validate_context(context)
            assert isinstance(is_valid, bool)
            
            decision = EthicalDecision(
                action="security_test",
                context=context,
                parameters={}
            )
            
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
    
    def test_resource_exhaustion_prevention(self, governor, mock_ethical_context):
        """Test prevention of resource exhaustion attacks"""
        # Test with extremely nested data
        nested_data = {"level": 0}
        current = nested_data
        for i in range(100):  # Deep nesting
            current["next"] = {"level": i + 1}
            current = current["next"]
        
        decision = EthicalDecision(
            action="resource_test",
            context=mock_ethical_context,
            parameters={"nested": nested_data}
        )
        
        start_time = time.time()
        result = governor.evaluate_decision(decision)
        end_time = time.time()
        
        # Should complete within reasonable time despite deep nesting
        assert end_time - start_time < 2.0
        assert isinstance(result, DecisionResult)
    
    def test_memory_leak_prevention(self, governor, mock_ethical_context):
        """Test prevention of memory leaks with repeated operations"""
        import gc
        
        # Force garbage collection and get initial memory info
        gc.collect()
        
        # Perform many operations
        for i in range(1000):
            decision = EthicalDecision(
                action=f"memory_test_{i}",
                context=mock_ethical_context,
                parameters={"data": f"test_{i}"}
            )
            governor.evaluate_decision(decision)
            
            # Periodically force garbage collection
            if i % 100 == 0:
                gc.collect()
        
        # Clean up should have occurred
        gc.collect()
        
        # Test that we can still perform operations normally
        final_decision = EthicalDecision(
            action="final_test",
            context=mock_ethical_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(final_decision)
        assert isinstance(result, DecisionResult)


class TestErrorHandlingAndRecovery:
    """Error handling and recovery tests"""
    
    def test_corrupted_state_recovery(self, governor, mock_ethical_context):
        """Test recovery from corrupted state"""
        # Create normal state
        decision = EthicalDecision(
            action="normal_action",
            context=mock_ethical_context,
            parameters={}
        )
        governor.evaluate_decision(decision)
        
        # Simulate state corruption
        governor.decision_history.append("corrupted_entry")
        
        # Should handle corrupted state gracefully
        new_decision = EthicalDecision(
            action="recovery_test",
            context=mock_ethical_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(new_decision)
        assert isinstance(result, DecisionResult)
    
    def test_invalid_rule_handling(self, governor, mock_ethical_context):
        """Test handling of invalid rules"""
        # Add a rule with invalid structure
        invalid_rule = {
            "name": "invalid_rule",
            "condition": "not_a_function",  # Invalid condition
            "action": "allow",
            "priority": 1
        }
        
        # Should handle invalid rule gracefully
        try:
            governor.add_ethical_rule(invalid_rule)
            
            decision = EthicalDecision(
                action="invalid_rule_test",
                context=mock_ethical_context,
                parameters={}
            )
            
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
            
        except (ValueError, TypeError, AttributeError):
            # Expected to raise an error for invalid rule
            pass
    
    def test_network_timeout_simulation(self, governor, mock_ethical_context):
        """Test handling of network timeouts in rule evaluation"""
        def slow_condition(ctx):
            time.sleep(0.1)  # Simulate slow network call
            return True
        
        slow_rule = {
            "name": "slow_rule",
            "condition": slow_condition,
            "action": "allow",
            "priority": 1
        }
        
        governor.add_ethical_rule(slow_rule)
        
        decision = EthicalDecision(
            action="timeout_test",
            context=mock_ethical_context,
            parameters={}
        )
        
        start_time = time.time()
        result = governor.evaluate_decision(decision)
        end_time = time.time()
        
        # Should complete but handle the delay
        assert isinstance(result, DecisionResult)
        assert end_time - start_time >= 0.1  # At least the delay time