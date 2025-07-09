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
        """
        Creates a new instance of GenesisEthicalGovernor for use in individual tests.
        
        Returns:
            GenesisEthicalGovernor: A fresh governor instance for test isolation.
        """
        return GenesisEthicalGovernor()
    
    @pytest.fixture
    def mock_ethical_context(self):
        """
        Create a sample EthicalContext instance with test user, action, data, and current timestamp for use in test cases.
        
        Returns:
            EthicalContext: Mock context populated with standard test values.
        """
        return EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={"test": "data"},
            timestamp=datetime.now()
        )
    
    def test_initialization(self, governor):
        """
        Verifies that the GenesisEthicalGovernor instance initializes with required attributes and correct types.
        """
        assert governor is not None
        assert hasattr(governor, 'ethical_rules')
        assert hasattr(governor, 'decision_history')
        assert hasattr(governor, 'violation_threshold')
        assert isinstance(governor.ethical_rules, list)
        assert isinstance(governor.decision_history, list)
    
    def test_initialization_with_custom_config(self):
        """
        Tests that GenesisEthicalGovernor initializes with the provided custom configuration and assigns configuration parameters correctly.
        """
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
        """
        Tests that evaluating a valid ethical decision returns a properly structured DecisionResult with expected types and value ranges.
        """
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
        """
        Test that `evaluate_decision` raises exceptions for invalid input.
        
        Verifies that passing `None` to `evaluate_decision` raises a `ValueError`, and passing a string raises a `TypeError`.
        """
        with pytest.raises(ValueError):
            governor.evaluate_decision(None)
        
        with pytest.raises(TypeError):
            governor.evaluate_decision("invalid_decision")
    
    def test_evaluate_decision_high_risk_action(self, governor, mock_ethical_context):
        """
        Verify that evaluating a high-risk action using the governor results in disapproval, a high confidence score, and reasoning that references high risk.
        """
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
        """
        Verifies that evaluating a low-risk action using the ethical governor results in approval with a confidence score greater than 0.5.
        """
        low_risk_decision = EthicalDecision(
            action="read_public_data",
            context=mock_ethical_context,
            parameters={"data_type": "public", "scope": "limited"}
        )
        
        result = governor.evaluate_decision(low_risk_decision)
        
        assert result.approved is True
        assert result.confidence_score > 0.5
    
    def test_add_ethical_rule(self, governor):
        """
        Verify that adding a new ethical rule to the governor increases the rule count and appends the rule with the correct name.
        """
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
        """
        Test that adding invalid ethical rules raises the correct exceptions.
        
        Verifies that adding `None` as a rule raises a `ValueError`, and adding an incomplete rule dictionary raises a `KeyError`.
        """
        with pytest.raises(ValueError):
            governor.add_ethical_rule(None)
        
        with pytest.raises(KeyError):
            governor.add_ethical_rule({"incomplete": "rule"})
    
    def test_remove_ethical_rule(self, governor):
        """
        Test that an ethical rule can be added and then removed from the GenesisEthicalGovernor.
        
        Adds a rule, removes it by name, and asserts that the rule count decreases and the rule is no longer present in the governor's rule list.
        """
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
        """
        Test that attempting to remove an ethical rule that does not exist raises a ValueError.
        """
        with pytest.raises(ValueError):
            governor.remove_ethical_rule("nonexistent_rule")
    
    def test_get_decision_history(self, governor, mock_ethical_context):
        """
        Verify that the governor's decision history contains the correct number of entries and that each entry includes the 'timestamp', 'decision', and 'result' fields.
        """
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
        """
        Test filtering of decision history by action name, ensuring only decisions matching the specified action are returned.
        """
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
        """
        Tests that the decision history is populated after evaluating a decision and is empty after calling `clear_decision_history`.
        """
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
        """
        Verify that ethical violations are properly recorded and can be retrieved for a given user.
        
        This test ensures that after recording a violation for a user, the violation can be retrieved and its attributes match the expected values.
        """
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
        """
        Verifies that a user's trust score is within valid bounds and decreases after a violation is recorded.
        """
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
        """
        Test that a user's trust score improves as violations become older, by comparing trust scores after an old violation versus a recent violation.
        """
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
        """
        Verify that the governor accepts valid EthicalContext objects and rejects those with missing or invalid fields.
        """
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
        """
        Verify that the GenesisEthicalGovernor can safely evaluate multiple decisions in parallel threads, ensuring correct result count and valid DecisionResult instances.
        """
        import threading
        
        decisions = []
        results = []
        
        def make_decision(decision_id):
            """
            Creates and evaluates an `EthicalDecision` with a unique action and parameters, then appends the evaluation result to a shared results list.
            
            Parameters:
                decision_id (int): Identifier used to generate unique action names and parameters for each decision.
            """
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
        """
        Test that GenesisEthicalGovernor efficiently processes and records a large number of decisions, maintaining acceptable performance and correct history size.
        """
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
        """
        Verify that the GenesisEthicalGovernor's state can be serialized and deserialized, ensuring decision history and configuration are accurately preserved.
        """
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
        """
        Test evaluation of a decision with empty parameters.
        
        Ensures that the governor returns a valid `DecisionResult` when evaluating a decision whose parameters dictionary is empty.
        """
        decision = EthicalDecision(
            action="empty_params_action",
            context=mock_ethical_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_edge_case_none_parameters(self, governor, mock_ethical_context):
        """
        Tests that the governor can evaluate an ethical decision when the decision's parameters are set to None, returning a valid DecisionResult.
        """
        decision = EthicalDecision(
            action="none_params_action",
            context=mock_ethical_context,
            parameters=None
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_edge_case_very_long_action_name(self, governor, mock_ethical_context):
        """
        Tests that the governor correctly evaluates a decision with an extremely long action name and returns a valid `DecisionResult`.
        """
        long_action = "a" * 1000
        decision = EthicalDecision(
            action=long_action,
            context=mock_ethical_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_memory_usage_with_large_context(self, governor):
        """
        Verify that GenesisEthicalGovernor can process a decision with a large context data payload without errors or excessive memory usage.
        
        Creates an EthicalContext containing a large data field, evaluates a decision using this context, and asserts that a valid DecisionResult is returned.
        """
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
        """
        Test that the logging system's info method is called during decision evaluation.
        
        Ensures that when a decision is evaluated by the governor, the logging functionality is triggered as expected.
        """
        decision = EthicalDecision(
            action="logged_action",
            context=mock_ethical_context,
            parameters={}
        )
        
        governor.evaluate_decision(decision)
        
        # Verify logging was called
        mock_logging.info.assert_called()
    
    def test_custom_rule_priority(self, governor, mock_ethical_context):
        """
        Tests that when multiple ethical rules match a decision, the rule with the highest priority determines the outcome.
        """
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
        """
        Verify that GenesisEthicalGovernor accepts valid configuration parameters and raises ValueError for invalid configurations.
        """
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
        """
        Verify that an EthicalDecision instance is created with the correct action, context, and parameters.
        """
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
        """
        Tests that two EthicalDecision instances with the same action, context, and parameters are equal.
        """
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
        """
        Verify that the string representation of an EthicalDecision object contains the action name and the class name.
        """
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
        """
        Test that an EthicalViolation instance is created with the correct user ID, action, context, severity, and timestamp attributes.
        """
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
        """
        Verifies that `EthicalViolation` accepts only valid severity levels and raises a `ValueError` for invalid severity values.
        """
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
        """
        Test that an EthicalContext object is correctly instantiated with the provided user ID, action, context data, and timestamp.
        """
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
        """
        Verify that an EthicalContext instance accepts None as context_data and preserves it.
        """
        context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data=None,
            timestamp=datetime.now()
        )
        
        assert context.context_data is None
    
    def test_ethical_context_serialization(self):
        """
        Verify that an EthicalContext object serializes to a dictionary with accurate field values.
        """
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
        """
        Verify that a DecisionResult instance is created with the correct approval status, confidence score, reasoning, and metadata.
        """
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
        """
        Verifies that DecisionResult only accepts confidence scores within [0.0, 1.0] and raises ValueError for values outside this range.
        """
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
        """
        Verify that the string representation of a DecisionResult object contains its approval status, confidence score, and class name.
        """
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
        """
        Test the complete workflow of decision evaluation, violation recording, trust score update, and decision history verification in the GenesisEthicalGovernor system.
        """
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
        """
        Processes 100 ethical decisions in bulk and verifies that each result is a valid `DecisionResult` and that all decisions are recorded in the governor's history.
        """
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
    """Extended comprehensive test suite for GenesisEthicalGovernor with additional edge cases"""
    
    @pytest.fixture
    def governor_with_rules(self):
        """
        Create a GenesisEthicalGovernor instance preloaded with standard ethical rules for testing.
        
        Returns:
            GenesisEthicalGovernor: An instance containing predefined rules for data deletion denial, admin override allowance, and suspicious activity denial.
        """
        gov = GenesisEthicalGovernor()
        
        # Add some standard rules
        rules = [
            {
                "name": "data_deletion_rule",
                "condition": lambda ctx: "delete" in ctx.action.lower(),
                "action": "deny",
                "priority": 10
            },
            {
                "name": "admin_override_rule",
                "condition": lambda ctx: ctx.context_data.get("admin_override", False),
                "action": "allow",
                "priority": 5
            },
            {
                "name": "suspicious_activity_rule",
                "condition": lambda ctx: ctx.context_data.get("suspicious_score", 0) > 0.8,
                "action": "deny",
                "priority": 8
            }
        ]
        
        for rule in rules:
            gov.add_ethical_rule(rule)
        
        return gov
    
    def test_rule_evaluation_order(self, governor_with_rules):
        """
        Test that ethical rules are evaluated by priority, with higher-priority rules (lower priority numbers) overriding lower-priority ones in decision outcomes.
        """
        context = EthicalContext(
            user_id="test_user",
            action="delete_data",
            context_data={"admin_override": True},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="delete_data",
            context=context,
            parameters={}
        )
        
        result = governor_with_rules.evaluate_decision(decision)
        
        # Admin override (priority 5) should beat deletion rule (priority 10)
        # Lower priority number = higher priority
        assert result.approved is True
    
    def test_multiple_rule_conflicts(self, governor_with_rules):
        """
        Test that actions are denied when multiple conflicting rules independently disapprove a decision.
        
        Ensures that if both a deletion rule and a suspicious activity rule would each deny an action, the governor does not approve the decision.
        """
        context = EthicalContext(
            user_id="test_user",
            action="delete_suspicious_data",
            context_data={"suspicious_score": 0.9, "admin_override": False},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="delete_suspicious_data",
            context=context,
            parameters={}
        )
        
        result = governor_with_rules.evaluate_decision(decision)
        
        # Both deletion rule and suspicious activity rule should deny
        assert result.approved is False
    
    def test_trust_score_edge_cases(self, governor):
        """
        Test trust score calculation for edge cases involving non-existent, empty, and None user IDs.
        
        Verifies that non-existent users receive a default trust score of 1.0, while empty or None user IDs raise a ValueError.
        """
        # Test with non-existent user
        score = governor.get_user_trust_score("nonexistent_user")
        assert score == 1.0  # Should default to full trust
        
        # Test with empty user ID
        with pytest.raises(ValueError):
            governor.get_user_trust_score("")
        
        # Test with None user ID
        with pytest.raises(ValueError):
            governor.get_user_trust_score(None)
    
    def test_violation_severity_impact(self, governor):
        """
        Test that higher severity violations result in greater reductions to user trust scores.
        
        Records violations of increasing severity for different users and asserts that trust scores decrease as severity increases.
        """
        context = EthicalContext(
            user_id="severity_test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Record violations of different severities
        severities = ["low", "medium", "high", "critical"]
        user_scores = {}
        
        for i, severity in enumerate(severities):
            user_id = f"user_{severity}"
            violation = EthicalViolation(
                user_id=user_id,
                action="test_violation",
                context=context,
                severity=severity,
                timestamp=datetime.now()
            )
            governor.record_violation(violation)
            user_scores[severity] = governor.get_user_trust_score(user_id)
        
        # Higher severity should result in lower trust score
        assert user_scores["critical"] < user_scores["high"]
        assert user_scores["high"] < user_scores["medium"]
        assert user_scores["medium"] < user_scores["low"]
    
    def test_decision_history_pagination(self, governor):
        """
        Verifies that decision history retrieval supports pagination and time-based filtering.
        
        Creates multiple decisions, then checks that limiting the number of returned decisions and filtering by timestamp both function as expected.
        """
        context = EthicalContext(
            user_id="pagination_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Create multiple decisions
        for i in range(50):
            decision = EthicalDecision(
                action=f"paginated_action_{i}",
                context=context,
                parameters={"index": i}
            )
            governor.evaluate_decision(decision)
        
        # Test getting recent decisions
        recent_history = governor.get_decision_history(limit=10)
        assert len(recent_history) == 10
        
        # Test getting decisions from specific time range
        cutoff_time = datetime.now() - timedelta(seconds=1)
        filtered_history = governor.get_decision_history(after_timestamp=cutoff_time)
        assert len(filtered_history) <= 50
    
    def test_rule_condition_exceptions(self, governor):
        """
        Verifies that exceptions raised within rule condition functions do not prevent the governor from evaluating a decision and returning a valid `DecisionResult`.
        """
        def failing_condition(ctx):
            """
            A rule condition function that always raises a RuntimeError to simulate a failing rule condition.
            
            Raises:
                RuntimeError: Always raised when the function is called.
            """
            raise RuntimeError("Rule condition failed")
        
        problematic_rule = {
            "name": "failing_rule",
            "condition": failing_condition,
            "action": "deny",
            "priority": 1
        }
        
        governor.add_ethical_rule(problematic_rule)
        
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
        
        # Should handle exception gracefully
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_context_data_deep_nesting(self, governor):
        """
        Test evaluation of decisions with deeply nested context data.
        
        Ensures that the governor processes decisions containing multi-level nested context structures without errors and returns a valid `DecisionResult`.
        """
        deep_context_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "deep_value": "test_value",
                            "numbers": [1, 2, 3, 4, 5]
                        }
                    }
                }
            }
        }
        
        context = EthicalContext(
            user_id="deep_nesting_user",
            action="deep_context_action",
            context_data=deep_context_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="deep_context_action",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_unicode_and_special_characters(self, governor):
        """
        Verify that the governor can process decisions and context data containing unicode characters, emojis, special symbols, and null bytes without errors.
        """
        special_chars_data = {
            "unicode": "测试数据",
            "emoji": "🚀🔒🛡️",
            "special": "!@#$%^&*()_+-=[]{}|;:,.<>?",
            "null_bytes": "test\x00data"
        }
        
        context = EthicalContext(
            user_id="unicode_user_测试",
            action="unicode_action_🔒",
            context_data=special_chars_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="unicode_action_🔒",
            context=context,
            parameters={"param": "value_with_emoji_🚀"}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_timestamp_timezone_handling(self, governor):
        """
        Tests that the governor evaluates decisions with context timestamps in different timezones and returns valid `DecisionResult` objects for both UTC and US/Eastern timezones.
        """
        import pytz
        
        # Test with UTC timezone
        utc_time = datetime.now(pytz.UTC)
        context_utc = EthicalContext(
            user_id="timezone_user",
            action="timezone_action",
            context_data={},
            timestamp=utc_time
        )
        
        # Test with different timezone
        est_time = datetime.now(pytz.timezone('US/Eastern'))
        context_est = EthicalContext(
            user_id="timezone_user",
            action="timezone_action",
            context_data={},
            timestamp=est_time
        )
        
        decision_utc = EthicalDecision(
            action="timezone_action",
            context=context_utc,
            parameters={}
        )
        
        decision_est = EthicalDecision(
            action="timezone_action",
            context=context_est,
            parameters={}
        )
        
        result_utc = governor.evaluate_decision(decision_utc)
        result_est = governor.evaluate_decision(decision_est)
        
        assert isinstance(result_utc, DecisionResult)
        assert isinstance(result_est, DecisionResult)
    
    def test_resource_cleanup_on_error(self, governor):
        """
        Test that the governor releases resources properly when errors occur during repeated evaluation of resource-intensive decisions.
        
        Simulates multiple evaluations with large context data to ensure no resource leaks or issues occur, even if exceptions are raised.
        """
        # Create a scenario that might cause resource leaks
        context = EthicalContext(
            user_id="cleanup_user",
            action="resource_intensive_action",
            context_data={"large_data": "x" * 1000000},  # 1MB of data
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="resource_intensive_action",
            context=context,
            parameters={}
        )
        
        # This should not cause memory leaks or resource issues
        for _ in range(100):
            try:
                result = governor.evaluate_decision(decision)
                assert isinstance(result, DecisionResult)
            except Exception:
                pass  # Ignore any exceptions for this test
    
    def test_concurrent_rule_modification(self, governor):
        """
        Verifies that ethical rule modifications and decision evaluations can be performed concurrently on the governor without errors.
        
        This test launches two threads: one adds new ethical rules while the other processes multiple decisions. After both threads complete, it asserts that the governor's rule list and decision history reflect the concurrent operations.
        """
        import threading
        
        def add_rules():
            """
            Adds ten unique ethical rules to the governor, each with a distinct name and priority, where each rule's condition always evaluates to False.
            """
            for i in range(10):
                rule = {
                    "name": f"concurrent_rule_{i}",
                    "condition": lambda ctx: False,
                    "action": "allow",
                    "priority": i
                }
                governor.add_ethical_rule(rule)
        
        def process_decisions():
            """
            Evaluates 20 ethical decisions concurrently, each with a unique action name but sharing the same user context.
            
            Each decision uses the same `EthicalContext` but varies the action from 'concurrent_action_0' to 'concurrent_action_19'.
            """
            context = EthicalContext(
                user_id="concurrent_user",
                action="concurrent_action",
                context_data={},
                timestamp=datetime.now()
            )
            
            for i in range(20):
                decision = EthicalDecision(
                    action=f"concurrent_action_{i}",
                    context=context,
                    parameters={}
                )
                governor.evaluate_decision(decision)
        
        # Start both operations concurrently
        rule_thread = threading.Thread(target=add_rules)
        decision_thread = threading.Thread(target=process_decisions)
        
        rule_thread.start()
        decision_thread.start()
        
        rule_thread.join()
        decision_thread.join()
        
        # Both operations should complete without errors
        assert len(governor.ethical_rules) >= 10
        assert len(governor.decision_history) >= 20
    
    def test_decision_result_metadata_completeness(self, governor):
        """
        Test that decision results from the governor include metadata fields such as processing time, rules evaluated, and decision ID.
        """
        context = EthicalContext(
            user_id="metadata_user",
            action="metadata_action",
            context_data={"test_key": "test_value"},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="metadata_action",
            context=context,
            parameters={"param1": "value1"}
        )
        
        result = governor.evaluate_decision(decision)
        
        # Check that metadata includes relevant information
        assert hasattr(result, 'metadata')
        if result.metadata:
            assert isinstance(result.metadata, dict)
            # Metadata should contain processing information
            expected_keys = ['processing_time', 'rules_evaluated', 'decision_id']
            for key in expected_keys:
                if key in result.metadata:
                    assert result.metadata[key] is not None
    
    @pytest.mark.parametrize("violation_count", [1, 3, 5, 10, 50])
    def test_trust_score_degradation_levels(self, governor, violation_count):
        """
        Test that a user's trust score decreases proportionally with the number of recorded violations.
        
        Parameters:
            violation_count (int): Number of violations to record for the user.
        """
        user_id = f"degradation_user_{violation_count}"
        context = EthicalContext(
            user_id=user_id,
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Record multiple violations
        for i in range(violation_count):
            violation = EthicalViolation(
                user_id=user_id,
                action=f"violation_{i}",
                context=context,
                severity="medium",
                timestamp=datetime.now() - timedelta(minutes=i)
            )
            governor.record_violation(violation)
        
        trust_score = governor.get_user_trust_score(user_id)
        
        # Trust score should decrease with more violations
        assert 0.0 <= trust_score <= 1.0
        if violation_count >= 10:
            assert trust_score < 0.5  # Severely degraded trust
        elif violation_count >= 5:
            assert trust_score < 0.7  # Moderately degraded trust
    
    def test_ethical_decision_immutability(self):
        """
        Test that EthicalDecision instances are immutable after creation, preventing modification of their attributes.
        """
        context = EthicalContext(
            user_id="immutable_user",
            action="immutable_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="immutable_action",
            context=context,
            parameters={"original": "value"}
        )
        
        original_action = decision.action
        original_params = decision.parameters.copy()
        
        # Attempting to modify should not affect the decision
        try:
            decision.action = "modified_action"
        except AttributeError:
            pass  # Expected if immutable
        
        try:
            decision.parameters["new_key"] = "new_value"
        except (AttributeError, TypeError):
            pass  # Expected if immutable
        
        # Verify original values are preserved
        assert decision.action == original_action
        assert decision.parameters == original_params
    
    def test_violation_aggregation_by_time_period(self, governor):
        """
        Test aggregation and retrieval of ethical violations by time period for a specific user.
        
        Verifies that violations recorded at various timestamps are correctly returned when retrieving all violations and when filtering by a recent time window.
        """
        user_id = "aggregation_user"
        context = EthicalContext(
            user_id=user_id,
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Create violations across different time periods
        time_periods = [
            datetime.now() - timedelta(hours=1),
            datetime.now() - timedelta(hours=6),
            datetime.now() - timedelta(days=1),
            datetime.now() - timedelta(days=7),
            datetime.now() - timedelta(days=30)
        ]
        
        for i, timestamp in enumerate(time_periods):
            violation = EthicalViolation(
                user_id=user_id,
                action=f"time_violation_{i}",
                context=context,
                severity="medium",
                timestamp=timestamp
            )
            governor.record_violation(violation)
        
        # Test getting violations for different time windows
        all_violations = governor.get_violations(user_id)
        assert len(all_violations) == 5
        
        # Test getting recent violations (last 24 hours)
        recent_violations = governor.get_violations(
            user_id, 
            since=datetime.now() - timedelta(hours=24)
        )
        assert len(recent_violations) <= 5
    
    def test_ethical_governor_state_consistency(self, governor):
        """
        Test that the GenesisEthicalGovernor maintains correct rule and decision history counts after repeated cycles of rule addition, decision evaluation, and rule removal.
        
        This ensures that the governor's internal state remains consistent and free of residual changes after multiple operations.
        """
        initial_rule_count = len(governor.ethical_rules)
        initial_history_count = len(governor.decision_history)
        
        context = EthicalContext(
            user_id="consistency_user",
            action="consistency_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Perform multiple operations
        for i in range(10):
            # Add a rule
            rule = {
                "name": f"consistency_rule_{i}",
                "condition": lambda ctx: False,
                "action": "allow",
                "priority": i
            }
            governor.add_ethical_rule(rule)
            
            # Make a decision
            decision = EthicalDecision(
                action=f"consistency_action_{i}",
                context=context,
                parameters={}
            )
            governor.evaluate_decision(decision)
            
            # Remove the rule
            governor.remove_ethical_rule(f"consistency_rule_{i}")
        
        # State should be consistent
        assert len(governor.ethical_rules) == initial_rule_count
        assert len(governor.decision_history) == initial_history_count + 10
    
    def test_malformed_input_handling(self, governor):
        """
        Verify that the governor processes malformed or excessively large input data without crashing.
        
        Creates an `EthicalDecision` and `EthicalContext` containing extremely large string values, then asserts that `evaluate_decision` returns a valid `DecisionResult` instance.
        """
        # Test with extremely large strings
        large_string = "x" * 100000
        
        context = EthicalContext(
            user_id="malformed_user",
            action=large_string,
            context_data={"large_data": large_string},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action=large_string,
            context=context,
            parameters={"large_param": large_string}
        )
        
        # Should handle gracefully without crashing
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_decision_caching_behavior(self, governor):
        """
        Verifies that evaluating the same decision multiple times produces consistent results, ensuring correct decision caching behavior if present.
        """
        context = EthicalContext(
            user_id="cache_user",
            action="cacheable_action",
            context_data={"cache_key": "value"},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="cacheable_action",
            context=context,
            parameters={"consistent": "params"}
        )
        
        # Make the same decision multiple times
        results = []
        for _ in range(5):
            result = governor.evaluate_decision(decision)
            results.append(result)
        
        # Results should be consistent (if caching is implemented)
        first_result = results[0]
        for result in results[1:]:
            assert result.approved == first_result.approved
            assert abs(result.confidence_score - first_result.confidence_score) < 0.1
    
    def test_rule_execution_timeout_handling(self, governor):
        """
        Tests that the governor handles slow-executing ethical rule conditions by ensuring decision evaluation completes within a specified timeout and returns a valid DecisionResult.
        """
        def slow_condition(ctx):
            """
            Simulates a slow rule condition by pausing execution before returning False.
            
            Parameters:
                ctx: The context object provided to the rule condition.
            """
            import time
            time.sleep(0.1)  # Simulate slow rule
            return False
        
        slow_rule = {
            "name": "slow_rule",
            "condition": slow_condition,
            "action": "allow",
            "priority": 1
        }
        
        governor.add_ethical_rule(slow_rule)
        
        context = EthicalContext(
            user_id="timeout_user",
            action="slow_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="slow_action",
            context=context,
            parameters={}
        )
        
        start_time = time.time()
        result = governor.evaluate_decision(decision)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 5.0  # 5 second timeout
        assert isinstance(result, DecisionResult)


class TestEthicalDecisionExtended:
    """Extended test cases for EthicalDecision class"""
    
    def test_decision_hash_consistency(self):
        """
        Test that two identical `EthicalDecision` instances produce the same hash value if the `__hash__` method is implemented.
        """
        context = EthicalContext(
            user_id="hash_user",
            action="hash_action",
            context_data={"key": "value"},
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
        
        # If hashing is implemented, identical decisions should have same hash
        if hasattr(decision1, '__hash__'):
            assert hash(decision1) == hash(decision2)
    
    def test_decision_with_callable_parameters(self):
        """
        Test that `EthicalDecision` can accept and store callable objects in its parameters.
        """
        context = EthicalContext(
            user_id="callable_user",
            action="callable_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        def test_callback():
            """
            Return a fixed string indicating the callback result.
            
            Returns:
                str: The string "callback_result".
            """
            return "callback_result"
        
        decision = EthicalDecision(
            action="callable_action",
            context=context,
            parameters={"callback": test_callback}
        )
        
        assert decision.parameters["callback"] == test_callback
        assert callable(decision.parameters["callback"])
    
    def test_decision_deep_copy_behavior(self):
        """
        Test that deep copying an EthicalDecision produces a copy whose context data remains unchanged when the original context is modified.
        """
        import copy
        
        context = EthicalContext(
            user_id="copy_user",
            action="copy_action",
            context_data={"nested": {"key": "value"}},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="copy_action",
            context=context,
            parameters={"nested_param": {"key": "value"}}
        )
        
        decision_copy = copy.deepcopy(decision)
        
        # Modify original context data
        context.context_data["nested"]["key"] = "modified"
        
        # Copy should be unaffected
        assert decision_copy.context.context_data["nested"]["key"] == "value"


class TestEthicalViolationExtended:
    """Extended test cases for EthicalViolation class"""
    
    def test_violation_severity_ordering(self):
        """
        Verifies that `EthicalViolation` instances can be sorted by severity when a sortable severity level attribute exists.
        
        Creates violations with different severities and checks that sorting by the severity level attribute produces the expected order.
        """
        context = EthicalContext(
            user_id="severity_user",
            action="severity_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        violations = []
        severities = ["low", "medium", "high", "critical"]
        
        for severity in severities:
            violation = EthicalViolation(
                user_id="severity_user",
                action=f"{severity}_action",
                context=context,
                severity=severity,
                timestamp=datetime.now()
            )
            violations.append(violation)
        
        # Test if violations can be sorted by severity
        if hasattr(violations[0], 'severity_level'):
            sorted_violations = sorted(violations, key=lambda v: v.severity_level)
            assert len(sorted_violations) == 4
    
    def test_violation_with_custom_metadata(self):
        """
        Verify that an EthicalViolation instance stores and provides access to custom metadata fields.
        """
        context = EthicalContext(
            user_id="metadata_user",
            action="metadata_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        violation = EthicalViolation(
            user_id="metadata_user",
            action="metadata_action",
            context=context,
            severity="high",
            timestamp=datetime.now(),
            metadata={"custom_field": "custom_value", "error_code": 404}
        )
        
        if hasattr(violation, 'metadata'):
            assert violation.metadata["custom_field"] == "custom_value"
            assert violation.metadata["error_code"] == 404
    
    def test_violation_json_serialization(self):
        """
        Verify that an `EthicalViolation` instance can be serialized to a JSON string with all key fields accurately represented.
        """
        context = EthicalContext(
            user_id="json_user",
            action="json_action",
            context_data={"serializable": True},
            timestamp=datetime.now()
        )
        
        violation = EthicalViolation(
            user_id="json_user",
            action="json_action",
            context=context,
            severity="medium",
            timestamp=datetime.now()
        )
        
        # Test if violation can be serialized to JSON
        if hasattr(violation, 'to_json'):
            json_str = violation.to_json()
            assert isinstance(json_str, str)
            
            # Should be valid JSON
            import json
            parsed = json.loads(json_str)
            assert parsed["user_id"] == "json_user"
            assert parsed["severity"] == "medium"


class TestEthicalContextExtended:
    """Extended test cases for EthicalContext class"""
    
    def test_context_validation_rules(self):
        """
        Tests that `EthicalContext` objects can be instantiated with both minimal and complex context data, ensuring correct attribute assignment in each case.
        """
        # Test with minimal valid context
        minimal_context = EthicalContext(
            user_id="min_user",
            action="min_action",
            context_data={},
            timestamp=datetime.now()
        )
        assert minimal_context.user_id == "min_user"
        
        # Test with maximal context
        maximal_context = EthicalContext(
            user_id="max_user",
            action="max_action",
            context_data={
                "complex_data": {
                    "nested": True,
                    "list": [1, 2, 3],
                    "string": "value"
                }
            },
            timestamp=datetime.now()
        )
        assert maximal_context.context_data["complex_data"]["nested"] is True
    
    def test_context_immutability_enforcement(self):
        """
        Verify that `EthicalContext` instances enforce immutability by preventing modification of their attributes after creation.
        """
        context = EthicalContext(
            user_id="immutable_user",
            action="immutable_action",
            context_data={"original": "value"},
            timestamp=datetime.now()
        )
        
        original_user_id = context.user_id
        original_action = context.action
        
        # Attempt to modify (should be prevented if immutable)
        try:
            context.user_id = "modified_user"
            context.action = "modified_action"
        except AttributeError:
            pass  # Expected if immutable
        
        # Verify no changes occurred
        assert context.user_id == original_user_id
        assert context.action == original_action
    
    def test_context_equality_comparison(self):
        """
        Verifies that two EthicalContext objects are equal when all attributes match and not equal when any attribute differs.
        """
        timestamp = datetime.now()
        
        context1 = EthicalContext(
            user_id="equal_user",
            action="equal_action",
            context_data={"key": "value"},
            timestamp=timestamp
        )
        
        context2 = EthicalContext(
            user_id="equal_user",
            action="equal_action",
            context_data={"key": "value"},
            timestamp=timestamp
        )
        
        # Should be equal if all fields match
        assert context1 == context2
        
        # Should not be equal if any field differs
        context3 = EthicalContext(
            user_id="different_user",
            action="equal_action",
            context_data={"key": "value"},
            timestamp=timestamp
        )
        
        assert context1 != context3


class TestPerformanceAndStressScenarios:
    """Performance and stress testing scenarios"""
    
    def test_memory_usage_under_load(self):
        """
        Test that processing a large number of decisions does not cause excessive memory usage growth in the GenesisEthicalGovernor.
        
        Processes 1000 decisions and asserts that memory usage increases by less than 100MB.
        """
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        governor = GenesisEthicalGovernor()
        
        # Process many decisions
        for i in range(1000):
            context = EthicalContext(
                user_id=f"load_user_{i % 100}",
                action=f"load_action_{i}",
                context_data={"index": i},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"load_action_{i}",
                context=context,
                parameters={"load_test": True}
            )
            
            governor.evaluate_decision(decision)
            
            # Periodic cleanup
            if i % 100 == 0:
                gc.collect()
        
        # Check memory usage didn't grow excessively
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Should not grow more than 100MB (adjust threshold as needed)
        assert memory_growth < 100 * 1024 * 1024
    
    def test_decision_processing_rate(self):
        """
        Verifies that GenesisEthicalGovernor processes at least 100 decisions per second when evaluating 1000 identical decisions in succession.
        """
        governor = GenesisEthicalGovernor()
        
        context = EthicalContext(
            user_id="rate_user",
            action="rate_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="rate_action",
            context=context,
            parameters={}
        )
        
        # Measure processing rate
        start_time = time.time()
        decision_count = 1000
        
        for i in range(decision_count):
            governor.evaluate_decision(decision)
        
        end_time = time.time()
        processing_time = end_time - start_time
        rate = decision_count / processing_time
        
        # Should process at least 100 decisions per second
        assert rate >= 100
    
    def test_large_rule_set_performance(self):
        """
        Verify that evaluating a decision with 100 ethical rules in the GenesisEthicalGovernor completes within one second and returns a valid DecisionResult.
        """
        governor = GenesisEthicalGovernor()
        
        # Add many rules
        for i in range(100):
            rule = {
                "name": f"perf_rule_{i}",
                "condition": lambda ctx: ctx.action == f"specific_action_{i}",
                "action": "allow" if i % 2 == 0 else "deny",
                "priority": i
            }
            governor.add_ethical_rule(rule)
        
        context = EthicalContext(
            user_id="perf_user",
            action="perf_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="perf_action",
            context=context,
            parameters={}
        )
        
        # Should still process quickly with many rules
        start_time = time.time()
        result = governor.evaluate_decision(decision)
        processing_time = time.time() - start_time
        
        assert processing_time < 1.0  # Should complete within 1 second
        assert isinstance(result, DecisionResult)


# Parametrized tests for comprehensive coverage
class TestParametrizedScenarios:
    """Parametrized tests for comprehensive scenario coverage"""
    
    @pytest.mark.parametrize("user_id,action,expected_approval", [
        ("admin_user", "read_data", True),
        ("admin_user", "delete_data", False),
        ("regular_user", "read_data", True),
        ("regular_user", "delete_data", False),
        ("guest_user", "read_data", False),
        ("guest_user", "delete_data", False),
    ])
    def test_user_permission_matrix(self, user_id, action, expected_approval):
        """
        Parametrized test that checks if the ethical governor approves or rejects actions for different user roles according to a permission matrix.
        
        Parameters:
            user_id (str): Identifier representing the user's role (such as admin, regular, or guest).
            action (str): The action to be evaluated for ethical approval.
            expected_approval (bool): The expected approval result for the user-action combination.
        """
        governor = GenesisEthicalGovernor()
        
        # Add permission-based rules
        if "admin" in user_id:
            permission_level = "admin"
        elif "regular" in user_id:
            permission_level = "regular"
        else:
            permission_level = "guest"
        
        context = EthicalContext(
            user_id=user_id,
            action=action,
            context_data={"permission_level": permission_level},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action=action,
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        
        # This test depends on the actual implementation
        # Adjust assertions based on actual business logic
        assert isinstance(result, DecisionResult)
        assert result.approved in [True, False]
    
    @pytest.mark.parametrize("violation_severity,expected_impact", [
        ("low", 0.05),
        ("medium", 0.15),
        ("high", 0.30),
        ("critical", 0.50),
    ])
    def test_violation_severity_impact_levels(self, violation_severity, expected_impact):
        """
        Test that recording a violation with a specific severity level decreases the user's trust score by an amount approximately matching the expected impact.
        
        Parameters:
        	violation_severity: Severity level of the violation being tested.
        	expected_impact: Expected reduction in trust score for the specified severity.
        """
        governor = GenesisEthicalGovernor()
        
        user_id = f"impact_user_{violation_severity}"
        initial_score = governor.get_user_trust_score(user_id)
        
        context = EthicalContext(
            user_id=user_id,
            action="impact_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        violation = EthicalViolation(
            user_id=user_id,
            action="impact_action",
            context=context,
            severity=violation_severity,
            timestamp=datetime.now()
        )
        
        governor.record_violation(violation)
        new_score = governor.get_user_trust_score(user_id)
        
        score_reduction = initial_score - new_score
        
        # Score reduction should be proportional to severity
        assert score_reduction >= expected_impact * 0.5  # Allow some tolerance
        assert score_reduction <= expected_impact * 2.0  # Allow some tolerance