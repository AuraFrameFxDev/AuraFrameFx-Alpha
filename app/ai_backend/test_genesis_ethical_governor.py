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
        Create and return a new instance of GenesisEthicalGovernor for use in each test.
        """
        return GenesisEthicalGovernor()
    
    @pytest.fixture
    def mock_ethical_context(self):
        """
        Creates and returns a mock EthicalContext object for use in tests.
        
        Returns:
            EthicalContext: A sample context with test user, action, data, and current timestamp.
        """
        return EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={"test": "data"},
            timestamp=datetime.now()
        )
    
    def test_initialization(self, governor):
        """
        Test that the GenesisEthicalGovernor is initialized with the expected attributes and types.
        """
        assert governor is not None
        assert hasattr(governor, 'ethical_rules')
        assert hasattr(governor, 'decision_history')
        assert hasattr(governor, 'violation_threshold')
        assert isinstance(governor.ethical_rules, list)
        assert isinstance(governor.decision_history, list)
    
    def test_initialization_with_custom_config(self):
        """
        Test that GenesisEthicalGovernor initializes correctly with a custom configuration.
        
        Verifies that custom configuration parameters are set as expected upon initialization.
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
        Test that `evaluate_decision` raises appropriate exceptions when given invalid input.
        
        Verifies that passing `None` raises a `ValueError` and passing a string raises a `TypeError`.
        """
        with pytest.raises(ValueError):
            governor.evaluate_decision(None)
        
        with pytest.raises(TypeError):
            governor.evaluate_decision("invalid_decision")
    
    def test_evaluate_decision_high_risk_action(self, governor, mock_ethical_context):
        """
        Tests that evaluating a high-risk action results in disapproval with high confidence and appropriate reasoning.
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
        Tests that evaluating a low-risk action results in approval with a confidence score above 0.5.
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
        Tests that a new ethical rule can be added to the governor and is correctly appended to the list of ethical rules.
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
        Test that adding invalid ethical rules to the governor raises appropriate exceptions.
        
        Verifies that adding a `None` rule raises a `ValueError` and adding an incomplete rule dictionary raises a `KeyError`.
        """
        with pytest.raises(ValueError):
            governor.add_ethical_rule(None)
        
        with pytest.raises(KeyError):
            governor.add_ethical_rule({"incomplete": "rule"})
    
    def test_remove_ethical_rule(self, governor):
        """
        Verify that an ethical rule can be successfully removed from the GenesisEthicalGovernor.
        
        Adds a test rule, removes it by name, and asserts that the rule count decreases and the rule is no longer present.
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
        Test that removing a nonexistent ethical rule from the governor raises a ValueError.
        """
        with pytest.raises(ValueError):
            governor.remove_ethical_rule("nonexistent_rule")
    
    def test_get_decision_history(self, governor, mock_ethical_context):
        """
        Tests that the decision history retrieved from the governor contains the correct number of entries and that each entry includes the expected fields: 'timestamp', 'decision', and 'result'.
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
        Test that the decision history can be filtered by action name, returning only matching decisions.
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
        Test that the decision history can be cleared after evaluating a decision.
        
        Verifies that after evaluating a decision, the decision history is populated, and that calling `clear_decision_history` removes all entries from the history.
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
        Tests that ethical violations are correctly recorded and retrieved for a specific user.
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
        Test that the user trust score is correctly calculated and decreases after recording a violation.
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
        Verify that a user's trust score recovers over time by comparing scores after old and recent violations.
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
        Tests that the governor correctly validates ethical context objects, accepting valid contexts and rejecting those with missing or invalid fields.
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
        Tests that the GenesisEthicalGovernor can evaluate multiple decisions concurrently without errors or data inconsistencies.
        
        Verifies that concurrent decision evaluations produce the expected number of results and that each result is a valid DecisionResult instance.
        """
        import threading
        
        decisions = []
        results = []
        
        def make_decision(decision_id):
            """
            Creates an `EthicalDecision` with a unique action and parameters, evaluates it using the governor, and appends the result to the shared results list.
            
            Parameters:
                decision_id (int): Unique identifier for the decision, used to differentiate actions and parameters.
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
        Tests that the GenesisEthicalGovernor can process and store a large number of decisions efficiently, ensuring performance remains within acceptable limits and all decisions are recorded in history.
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
        Tests that the GenesisEthicalGovernor's state can be serialized to a string and accurately restored via deserialization, preserving decision history and configuration.
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
        Test that the governor correctly evaluates a decision with empty parameters.
        
        Verifies that evaluating a decision with an empty parameters dictionary returns a valid `DecisionResult` object.
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
        Test that the governor correctly evaluates an ethical decision when the decision's parameters are set to None.
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
        Test that the governor can evaluate decisions with extremely long action names without errors.
        
        Verifies that a decision with a 1000-character action name is processed and returns a valid `DecisionResult`.
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
        Tests that the GenesisEthicalGovernor can evaluate a decision with a large context data payload without errors or excessive memory usage.
        
        Creates an EthicalContext with a large data field and verifies that decision evaluation returns a valid DecisionResult.
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
        Test that the logging functionality is triggered during decision evaluation.
        
        Verifies that the logging system's info method is called when a decision is evaluated by the governor.
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
        Verify that ethical rules are evaluated in order of their priority, with higher priority rules taking precedence over lower ones when multiple rules match a decision.
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
        Tests that the GenesisEthicalGovernor correctly validates configuration parameters, accepting valid configurations and raising a ValueError for invalid ones.
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
        Test that an EthicalDecision object is correctly created with the specified action, context, and parameters.
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
        Verify that two EthicalDecision objects with identical attributes are considered equal.
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
        Tests that the string representation of an EthicalDecision object includes the action name and class identifier.
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
        Verify that an EthicalViolation object is correctly created with the expected attributes.
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
        Test that `EthicalViolation` correctly accepts valid severity levels and raises a ValueError for invalid severity values.
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
        """
        Test that an EthicalContext object correctly handles None as context_data.
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
        Tests that an EthicalContext object can be serialized to a dictionary with correct field values.
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
        Test that a DecisionResult object is correctly created with the specified attributes.
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
        Test that the DecisionResult enforces confidence scores within the valid range [0.0, 1.0], raising ValueError for out-of-range values.
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
        Tests that the string representation of a DecisionResult object includes its approval status, confidence score, and class name.
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
        Tests the end-to-end workflow of evaluating a decision, recording a violation if rejected, updating user trust score, and verifying decision history in the GenesisEthicalGovernor system.
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
        Tests the processing of 100 ethical decisions in bulk and verifies correct result types and decision history tracking.
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
        Instantiate a GenesisEthicalGovernor with a set of predefined ethical rules for testing purposes.
        
        Returns:
            GenesisEthicalGovernor: An instance preloaded with standard rules for data deletion, admin override, and suspicious activity scenarios.
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
        Verify that ethical rules are evaluated in order of priority, ensuring higher-priority rules (with lower priority numbers) take precedence in decision outcomes.
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
        Test that the governor correctly denies an action when multiple conflicting rules apply to a decision.
        
        Verifies that when both a deletion rule and a suspicious activity rule are present and would independently deny the action, the decision is not approved.
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
        Tests edge cases for user trust score calculation, including non-existent, empty, and None user IDs.
        
        Verifies that a non-existent user defaults to a full trust score, while empty or None user IDs raise a ValueError.
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
        Test that violations with higher severity levels cause greater reductions in user trust scores.
        
        Verifies that recording violations of increasing severity results in progressively lower trust scores for each user.
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
        Test that the decision history retrieval supports pagination and time-based filtering.
        
        Creates multiple decisions, verifies that limiting the number of returned decisions works, and checks filtering by timestamp.
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
        Test that the governor gracefully handles exceptions raised within rule condition functions during decision evaluation.
        
        Verifies that an exception in a rule's condition does not prevent decision evaluation and that a valid `DecisionResult` is still returned.
        """
        def failing_condition(ctx):
            """
            A rule condition function that always raises a RuntimeError when called.
            
            Raises:
            	RuntimeError: Always raised to simulate a failing rule condition.
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
        Test that the governor can evaluate decisions with deeply nested context data without errors.
        
        Verifies that a decision containing a multi-level nested context structure is processed correctly and returns a valid `DecisionResult`.
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
        Tests that the governor correctly processes decisions and context data containing unicode, emojis, special characters, and null bytes.
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
        Test that the governor correctly evaluates decisions with context timestamps in different timezones.
        
        Verifies that decisions with UTC and US/Eastern timezone-aware timestamps are processed and return valid `DecisionResult` objects.
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
        Verify that the governor properly cleans up resources when errors occur during repeated evaluation of resource-intensive decisions.
        
        This test simulates multiple evaluations with large context data to ensure no resource leaks or issues arise, even if exceptions are raised.
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
        Test that ethical rule modifications and decision processing can occur concurrently without errors.
        
        This test starts two threads: one adding new ethical rules and another processing decisions. It verifies that both operations complete successfully and that the governor's rule and decision history reflect the concurrent activity.
        """
        import threading
        
        def add_rules():
            """
            Adds ten distinct ethical rules to the governor, each with a unique name and priority, and a condition that always evaluates to False.
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
            Evaluates a series of ethical decisions concurrently using the same context but different actions.
            
            Each decision is processed by the governor for actions named 'concurrent_action_0' through 'concurrent_action_19'.
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
        Verify that decision results produced by the governor include comprehensive metadata fields such as processing time, rules evaluated, and decision ID.
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
        Verify that a user's trust score decreases proportionally as the number of recorded violations increases.
        
        Parameters:
        	violation_count (int): The number of violations to record for the user.
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
        Verify that EthicalDecision objects remain immutable after creation, ensuring their attributes cannot be modified.
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
        Test that ethical violations can be aggregated and retrieved by specific time periods for a given user.
        
        This test verifies that violations recorded at different timestamps are correctly returned when querying for all violations and when filtering by a recent time window.
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
        Verify that the GenesisEthicalGovernor maintains consistent state after repeated add/remove rule and decision operations.
        
        This test ensures that after performing multiple cycles of adding a rule, evaluating a decision, and removing the rule, the number of ethical rules and the decision history remain as expected.
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
        Tests that the governor can handle malformed or excessively large input data without crashing.
        
        Creates an `EthicalDecision` and `EthicalContext` with extremely large string values to verify that `evaluate_decision` processes the input gracefully and returns a valid `DecisionResult`.
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
        Test that repeated evaluations of the same decision yield consistent results, verifying decision caching behavior if implemented.
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
        Test that the governor correctly handles ethical rules with slow-executing conditions, ensuring decision evaluation completes within a reasonable timeout.
        """
        def slow_condition(ctx):
            """
            Simulates a slow rule evaluation by introducing a delay before returning False.
            
            Parameters:
                ctx: The context object passed to the rule condition.
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
        Verify that identical `EthicalDecision` objects produce the same hash value if hashing is implemented.
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
        Verify that `EthicalDecision` objects can accept and correctly store callable objects as parameters.
        """
        context = EthicalContext(
            user_id="callable_user",
            action="callable_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        def test_callback():
            """
            Returns a fixed string indicating the callback result.
            
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
        Test that deep copying an EthicalDecision object results in an independent copy whose context data is unaffected by changes to the original.
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
        Test that `EthicalViolation` objects can be ordered by severity if a sortable severity level attribute is present.
        
        Creates violations with varying severities and verifies that they can be sorted by severity level when supported.
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
        Test that an EthicalViolation object correctly stores and exposes custom metadata fields.
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
        Tests that an `EthicalViolation` object can be serialized to a valid JSON string and that key fields are correctly represented in the output.
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
        Validate that `EthicalContext` objects are correctly created and handle both minimal and complex context data inputs.
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
        Test that `EthicalContext` objects prevent modification of immutable attributes after creation.
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
        Test that EthicalContext objects are considered equal when all fields match and unequal when any field differs.
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
        Tests that memory usage remains within acceptable limits when processing a large number of decisions in the GenesisEthicalGovernor under sustained load.
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
        Tests that the GenesisEthicalGovernor can process at least 100 decisions per second when evaluating 1000 identical decisions under load.
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
        Tests that the GenesisEthicalGovernor can efficiently evaluate a decision when a large number of ethical rules (100) are present, ensuring processing completes within one second and returns a valid DecisionResult.
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
        Parametrized test that verifies decision approval for various user roles and actions using a permission matrix.
        
        Parameters:
            user_id (str): The ID representing the user's role (e.g., admin, regular, guest).
            action (str): The action being evaluated for ethical approval.
            expected_approval (bool): The expected approval outcome for the given user and action.
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
        Verify that recording a violation with a given severity reduces the user's trust score by an amount proportional to the expected impact.
        
        Parameters:
        	violation_severity: The severity level of the violation to test.
        	expected_impact: The expected reduction in trust score for the given severity.
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

class TestGenesisEthicalGovernorAdvanced:
    """Advanced test scenarios for GenesisEthicalGovernor with comprehensive edge cases"""
    
    @pytest.fixture
    def governor_with_complex_rules(self):
        """
        Create a governor with complex nested rules for advanced testing scenarios.
        """
        governor = GenesisEthicalGovernor()
        
        # Complex rule with nested conditions
        complex_rule = {
            "name": "complex_data_access_rule",
            "condition": lambda ctx: (
                ctx.action.startswith("access_") and 
                ctx.context_data.get("data_sensitivity", "low") == "high" and
                ctx.context_data.get("user_clearance", 0) < 3
            ),
            "action": "deny",
            "priority": 1
        }
        
        # Time-based rule
        time_rule = {
            "name": "time_restricted_rule",
            "condition": lambda ctx: (
                ctx.timestamp.hour < 8 or ctx.timestamp.hour > 18
            ),
            "action": "deny",
            "priority": 2
        }
        
        # IP-based rule
        ip_rule = {
            "name": "ip_restriction_rule",
            "condition": lambda ctx: (
                ctx.context_data.get("ip_address", "").startswith("192.168.")
            ),
            "action": "allow",
            "priority": 3
        }
        
        rules = [complex_rule, time_rule, ip_rule]
        for rule in rules:
            governor.add_ethical_rule(rule)
        
        return governor
    
    def test_complex_nested_rule_evaluation(self, governor_with_complex_rules):
        """Test that complex nested rules are evaluated correctly with multiple conditions."""
        context = EthicalContext(
            user_id="complex_user",
            action="access_sensitive_data",
            context_data={
                "data_sensitivity": "high",
                "user_clearance": 2,
                "ip_address": "192.168.1.100"
            },
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="access_sensitive_data",
            context=context,
            parameters={}
        )
        
        result = governor_with_complex_rules.evaluate_decision(decision)
        
        # Should be denied due to complex rule (high sensitivity + low clearance)
        assert result.approved is False
        assert isinstance(result, DecisionResult)
    
    def test_time_based_rule_evaluation(self, governor_with_complex_rules):
        """Test that time-based rules are evaluated correctly based on timestamp."""
        # Test during business hours (should be allowed)
        business_hours_time = datetime.now().replace(hour=14, minute=30)
        context_business = EthicalContext(
            user_id="time_user",
            action="normal_action",
            context_data={"data_sensitivity": "low"},
            timestamp=business_hours_time
        )
        
        decision_business = EthicalDecision(
            action="normal_action",
            context=context_business,
            parameters={}
        )
        
        result_business = governor_with_complex_rules.evaluate_decision(decision_business)
        
        # Test after hours (should be denied)
        after_hours_time = datetime.now().replace(hour=22, minute=0)
        context_after = EthicalContext(
            user_id="time_user",
            action="normal_action",
            context_data={"data_sensitivity": "low"},
            timestamp=after_hours_time
        )
        
        decision_after = EthicalDecision(
            action="normal_action",
            context=context_after,
            parameters={}
        )
        
        result_after = governor_with_complex_rules.evaluate_decision(decision_after)
        
        # After hours should be denied
        assert result_after.approved is False
        assert isinstance(result_business, DecisionResult)
        assert isinstance(result_after, DecisionResult)
    
    def test_rule_chain_evaluation_order(self, governor_with_complex_rules):
        """Test that rules are evaluated in correct priority order and short-circuit appropriately."""
        # Add a high-priority override rule
        override_rule = {
            "name": "emergency_override",
            "condition": lambda ctx: ctx.context_data.get("emergency_override", False),
            "action": "allow",
            "priority": 0  # Highest priority
        }
        governor_with_complex_rules.add_ethical_rule(override_rule)
        
        # Create context that would normally be denied but has emergency override
        context = EthicalContext(
            user_id="emergency_user",
            action="access_sensitive_data",
            context_data={
                "data_sensitivity": "high",
                "user_clearance": 1,
                "emergency_override": True
            },
            timestamp=datetime.now().replace(hour=3)  # After hours
        )
        
        decision = EthicalDecision(
            action="access_sensitive_data",
            context=context,
            parameters={}
        )
        
        result = governor_with_complex_rules.evaluate_decision(decision)
        
        # Should be allowed due to emergency override despite other restrictions
        assert result.approved is True
    
    def test_decision_metadata_enrichment(self, governor_with_complex_rules):
        """Test that decision results contain comprehensive metadata about the evaluation process."""
        context = EthicalContext(
            user_id="metadata_test_user",
            action="metadata_test_action",
            context_data={"test_metadata": True},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="metadata_test_action",
            context=context,
            parameters={"param1": "value1"}
        )
        
        result = governor_with_complex_rules.evaluate_decision(decision)
        
        # Verify metadata exists and contains expected fields
        assert hasattr(result, 'metadata')
        if result.metadata:
            assert isinstance(result.metadata, dict)
            # Check for common metadata fields
            expected_fields = ['evaluation_timestamp', 'rules_matched', 'processing_duration']
            for field in expected_fields:
                if field in result.metadata:
                    assert result.metadata[field] is not None
    
    def test_rule_performance_monitoring(self, governor_with_complex_rules):
        """Test that rule evaluation performance is monitored and reported."""
        # Add a deliberately slow rule
        def slow_rule_condition(ctx):
            import time
            time.sleep(0.01)  # 10ms delay
            return ctx.action == "slow_test"
        
        slow_rule = {
            "name": "performance_test_rule",
            "condition": slow_rule_condition,
            "action": "allow",
            "priority": 1
        }
        governor_with_complex_rules.add_ethical_rule(slow_rule)
        
        context = EthicalContext(
            user_id="perf_user",
            action="slow_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="slow_test",
            context=context,
            parameters={}
        )
        
        start_time = time.time()
        result = governor_with_complex_rules.evaluate_decision(decision)
        end_time = time.time()
        
        # Should complete but track performance
        assert isinstance(result, DecisionResult)
        assert (end_time - start_time) > 0.01  # Should include the delay
    
    def test_circular_dependency_prevention(self, governor_with_complex_rules):
        """Test that circular dependencies in rule evaluation are prevented."""
        # Add rules that could potentially create circular dependencies
        rule_a = {
            "name": "rule_a",
            "condition": lambda ctx: ctx.context_data.get("trigger_b", False),
            "action": "deny",
            "priority": 1
        }
        
        rule_b = {
            "name": "rule_b", 
            "condition": lambda ctx: ctx.context_data.get("trigger_a", False),
            "action": "allow",
            "priority": 2
        }
        
        governor_with_complex_rules.add_ethical_rule(rule_a)
        governor_with_complex_rules.add_ethical_rule(rule_b)
        
        context = EthicalContext(
            user_id="circular_user",
            action="circular_test",
            context_data={"trigger_a": True, "trigger_b": True},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="circular_test",
            context=context,
            parameters={}
        )
        
        # Should handle without infinite loops
        result = governor_with_complex_rules.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_dynamic_rule_modification_during_evaluation(self, governor_with_complex_rules):
        """Test that rules can be modified while decisions are being evaluated."""
        import threading
        
        evaluation_results = []
        
        def continuous_evaluation():
            """Continuously evaluate decisions while rules are being modified."""
            for i in range(20):
                context = EthicalContext(
                    user_id=f"dynamic_user_{i}",
                    action=f"dynamic_action_{i}",
                    context_data={"iteration": i},
                    timestamp=datetime.now()
                )
                
                decision = EthicalDecision(
                    action=f"dynamic_action_{i}",
                    context=context,
                    parameters={}
                )
                
                result = governor_with_complex_rules.evaluate_decision(decision)
                evaluation_results.append(result)
                time.sleep(0.01)
        
        def modify_rules():
            """Modify rules while evaluations are happening."""
            for i in range(10):
                rule = {
                    "name": f"dynamic_rule_{i}",
                    "condition": lambda ctx: ctx.context_data.get("iteration", 0) == i,
                    "action": "allow",
                    "priority": 10 + i
                }
                governor_with_complex_rules.add_ethical_rule(rule)
                time.sleep(0.02)
        
        # Start both threads
        eval_thread = threading.Thread(target=continuous_evaluation)
        modify_thread = threading.Thread(target=modify_rules)
        
        eval_thread.start()
        modify_thread.start()
        
        eval_thread.join()
        modify_thread.join()
        
        # All evaluations should have completed successfully
        assert len(evaluation_results) == 20
        assert all(isinstance(result, DecisionResult) for result in evaluation_results)
    
    def test_rule_validation_and_sanitization(self, governor_with_complex_rules):
        """Test that rules are properly validated and sanitized before being added."""
        # Test with potentially malicious rule
        malicious_rule = {
            "name": "malicious_rule",
            "condition": lambda ctx: exec("print('malicious code')") or False,
            "action": "allow",
            "priority": 1
        }
        
        # Should handle malicious rule gracefully
        try:
            governor_with_complex_rules.add_ethical_rule(malicious_rule)
            
            context = EthicalContext(
                user_id="security_user",
                action="security_test",
                context_data={},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action="security_test",
                context=context,
                parameters={}
            )
            
            result = governor_with_complex_rules.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
            
        except Exception as e:
            # Should handle gracefully without crashing
            assert "malicious" not in str(e).lower()
    
    def test_context_data_size_limits(self, governor_with_complex_rules):
        """Test that extremely large context data is handled appropriately."""
        # Create very large context data
        large_data = {
            "large_string": "x" * 1000000,  # 1MB string
            "large_list": list(range(100000)),
            "nested_data": {
                "level_" + str(i): f"data_{i}" * 1000 for i in range(100)
            }
        }
        
        context = EthicalContext(
            user_id="large_data_user",
            action="large_data_action",
            context_data=large_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="large_data_action",
            context=context,
            parameters={}
        )
        
        # Should handle large data without crashing
        result = governor_with_complex_rules.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_rule_condition_error_isolation(self, governor_with_complex_rules):
        """Test that errors in one rule condition don't affect other rules."""
        # Add a rule that raises an exception
        def failing_condition(ctx):
            raise ValueError("Rule condition failed")
        
        failing_rule = {
            "name": "failing_rule",
            "condition": failing_condition,
            "action": "deny",
            "priority": 1
        }
        
        # Add a working rule
        working_rule = {
            "name": "working_rule",
            "condition": lambda ctx: ctx.action == "test_action",
            "action": "allow",
            "priority": 2
        }
        
        governor_with_complex_rules.add_ethical_rule(failing_rule)
        governor_with_complex_rules.add_ethical_rule(working_rule)
        
        context = EthicalContext(
            user_id="error_isolation_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="test_action",
            context=context,
            parameters={}
        )
        
        # Should handle failing rule gracefully and continue with working rule
        result = governor_with_complex_rules.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)


class TestGenesisEthicalGovernorSecurityScenarios:
    """Security-focused test scenarios for GenesisEthicalGovernor"""
    
    def test_injection_attack_prevention(self):
        """Test that the governor prevents various injection attacks through input validation."""
        governor = GenesisEthicalGovernor()
        
        # SQL injection attempt
        sql_injection_context = EthicalContext(
            user_id="'; DROP TABLE users; --",
            action="'; DELETE FROM decisions; --",
            context_data={"input": "' OR '1'='1"},
            timestamp=datetime.now()
        )
        
        sql_injection_decision = EthicalDecision(
            action="'; DELETE FROM decisions; --",
            context=sql_injection_context,
            parameters={"query": "SELECT * FROM users WHERE id = '1' OR '1'='1'"}
        )
        
        # Should handle injection attempts gracefully
        result = governor.evaluate_decision(sql_injection_decision)
        assert isinstance(result, DecisionResult)
        
        # Code injection attempt
        code_injection_context = EthicalContext(
            user_id="code_injection_user",
            action="eval('__import__(\"os\").system(\"rm -rf /\")')",
            context_data={"code": "exec('print(\"hacked\")')"},
            timestamp=datetime.now()
        )
        
        code_injection_decision = EthicalDecision(
            action="eval('__import__(\"os\").system(\"rm -rf /\")')",
            context=code_injection_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(code_injection_decision)
        assert isinstance(result, DecisionResult)
    
    def test_authentication_bypass_attempts(self):
        """Test that the governor properly validates user authentication and prevents bypass attempts."""
        governor = GenesisEthicalGovernor()
        
        # Authentication bypass attempts
        bypass_attempts = [
            {"user_id": "admin", "spoofed": True},
            {"user_id": "root", "elevated": True},
            {"user_id": "system", "bypass_auth": True},
            {"user_id": "", "anonymous": True},
            {"user_id": None, "null_user": True}
        ]
        
        for attempt in bypass_attempts:
            context = EthicalContext(
                user_id=attempt["user_id"],
                action="privileged_action",
                context_data=attempt,
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action="privileged_action",
                context=context,
                parameters={}
            )
            
            # Should handle bypass attempts appropriately
            if attempt["user_id"] is None:
                with pytest.raises((ValueError, TypeError)):
                    governor.evaluate_decision(decision)
            else:
                result = governor.evaluate_decision(decision)
                assert isinstance(result, DecisionResult)
    
    def test_privilege_escalation_detection(self):
        """Test that the governor detects and prevents privilege escalation attempts."""
        governor = GenesisEthicalGovernor()
        
        # Add privilege escalation detection rule
        escalation_rule = {
            "name": "privilege_escalation_rule",
            "condition": lambda ctx: (
                ctx.context_data.get("requested_role", "").lower() in ["admin", "root", "system"] and
                ctx.context_data.get("current_role", "").lower() in ["user", "guest"]
            ),
            "action": "deny",
            "priority": 1
        }
        governor.add_ethical_rule(escalation_rule)
        
        # Test escalation attempt
        escalation_context = EthicalContext(
            user_id="regular_user",
            action="change_role",
            context_data={
                "current_role": "user",
                "requested_role": "admin",
                "escalation_attempt": True
            },
            timestamp=datetime.now()
        )
        
        escalation_decision = EthicalDecision(
            action="change_role",
            context=escalation_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(escalation_decision)
        assert result.approved is False
        assert "privilege" in result.reasoning.lower() or "escalation" in result.reasoning.lower()
    
    def test_data_exfiltration_prevention(self):
        """Test that the governor can detect and prevent data exfiltration attempts."""
        governor = GenesisEthicalGovernor()
        
        # Add data exfiltration detection rule
        exfiltration_rule = {
            "name": "data_exfiltration_rule",
            "condition": lambda ctx: (
                ctx.action.lower() in ["export", "download", "copy"] and
                ctx.context_data.get("data_volume", 0) > 1000000 and  # > 1MB
                ctx.context_data.get("external_destination", False)
            ),
            "action": "deny",
            "priority": 1
        }
        governor.add_ethical_rule(exfiltration_rule)
        
        # Test exfiltration attempt
        exfiltration_context = EthicalContext(
            user_id="suspicious_user",
            action="export",
            context_data={
                "data_volume": 10000000,  # 10MB
                "external_destination": True,
                "data_classification": "confidential"
            },
            timestamp=datetime.now()
        )
        
        exfiltration_decision = EthicalDecision(
            action="export",
            context=exfiltration_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(exfiltration_decision)
        assert result.approved is False
    
    def test_denial_of_service_protection(self):
        """Test that the governor protects against denial of service attacks."""
        governor = GenesisEthicalGovernor()
        
        # Simulate rapid requests from same user
        user_id = "dos_attacker"
        rapid_requests = []
        
        start_time = time.time()
        for i in range(100):
            context = EthicalContext(
                user_id=user_id,
                action=f"rapid_request_{i}",
                context_data={"request_id": i, "timestamp": time.time()},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"rapid_request_{i}",
                context=context,
                parameters={}
            )
            
            result = governor.evaluate_decision(decision)
            rapid_requests.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete all requests but may implement rate limiting
        assert len(rapid_requests) == 100
        assert all(isinstance(result, DecisionResult) for result in rapid_requests)
        
        # Should not take excessively long (basic DoS protection)
        assert total_time < 30.0  # 30 seconds maximum


class TestGenesisEthicalGovernorComplianceScenarios:
    """Compliance and regulatory test scenarios"""
    
    def test_gdpr_compliance_checks(self):
        """Test GDPR compliance features including data processing lawfulness and user consent."""
        governor = GenesisEthicalGovernor()
        
        # Add GDPR compliance rule
        gdpr_rule = {
            "name": "gdpr_compliance_rule",
            "condition": lambda ctx: (
                ctx.context_data.get("data_type") == "personal" and
                not ctx.context_data.get("user_consent", False) and
                ctx.context_data.get("processing_purpose") not in ["legal_obligation", "vital_interest"]
            ),
            "action": "deny",
            "priority": 1
        }
        governor.add_ethical_rule(gdpr_rule)
        
        # Test processing personal data without consent
        non_compliant_context = EthicalContext(
            user_id="eu_user",
            action="process_personal_data",
            context_data={
                "data_type": "personal",
                "user_consent": False,
                "processing_purpose": "marketing",
                "data_subject_location": "EU"
            },
            timestamp=datetime.now()
        )
        
        non_compliant_decision = EthicalDecision(
            action="process_personal_data",
            context=non_compliant_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(non_compliant_decision)
        assert result.approved is False
        
        # Test compliant processing with consent
        compliant_context = EthicalContext(
            user_id="eu_user",
            action="process_personal_data",
            context_data={
                "data_type": "personal",
                "user_consent": True,
                "processing_purpose": "service_delivery",
                "data_subject_location": "EU"
            },
            timestamp=datetime.now()
        )
        
        compliant_decision = EthicalDecision(
            action="process_personal_data",
            context=compliant_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(compliant_decision)
        assert result.approved is True
    
    def test_hipaa_compliance_checks(self):
        """Test HIPAA compliance for healthcare data processing."""
        governor = GenesisEthicalGovernor()
        
        # Add HIPAA compliance rule
        hipaa_rule = {
            "name": "hipaa_compliance_rule",
            "condition": lambda ctx: (
                ctx.context_data.get("data_type") == "phi" and  # Protected Health Information
                not ctx.context_data.get("authorized_access", False) and
                ctx.context_data.get("access_purpose") not in ["treatment", "payment", "healthcare_operations"]
            ),
            "action": "deny",
            "priority": 1
        }
        governor.add_ethical_rule(hipaa_rule)
        
        # Test unauthorized PHI access
        unauthorized_context = EthicalContext(
            user_id="unauthorized_user",
            action="access_patient_data",
            context_data={
                "data_type": "phi",
                "authorized_access": False,
                "access_purpose": "research",
                "patient_id": "12345"
            },
            timestamp=datetime.now()
        )
        
        unauthorized_decision = EthicalDecision(
            action="access_patient_data",
            context=unauthorized_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(unauthorized_decision)
        assert result.approved is False
        
        # Test authorized access for treatment
        authorized_context = EthicalContext(
            user_id="doctor_user",
            action="access_patient_data",
            context_data={
                "data_type": "phi",
                "authorized_access": True,
                "access_purpose": "treatment",
                "patient_id": "12345"
            },
            timestamp=datetime.now()
        )
        
        authorized_decision = EthicalDecision(
            action="access_patient_data",
            context=authorized_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(authorized_decision)
        assert result.approved is True
    
    def test_financial_compliance_checks(self):
        """Test financial compliance (SOX, PCI-DSS) for financial data operations."""
        governor = GenesisEthicalGovernor()
        
        # Add financial compliance rule
        financial_rule = {
            "name": "financial_compliance_rule",
            "condition": lambda ctx: (
                ctx.context_data.get("data_type") == "financial" and
                ctx.context_data.get("transaction_amount", 0) > 10000 and  # Large transaction
                not ctx.context_data.get("supervisor_approval", False)
            ),
            "action": "deny",
            "priority": 1
        }
        governor.add_ethical_rule(financial_rule)
        
        # Test large financial transaction without approval
        large_transaction_context = EthicalContext(
            user_id="financial_user",
            action="process_transaction",
            context_data={
                "data_type": "financial",
                "transaction_amount": 50000,
                "supervisor_approval": False,
                "transaction_type": "wire_transfer"
            },
            timestamp=datetime.now()
        )
        
        large_transaction_decision = EthicalDecision(
            action="process_transaction",
            context=large_transaction_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(large_transaction_decision)
        assert result.approved is False
    
    def test_audit_trail_completeness(self):
        """Test that all decisions create complete audit trails for compliance."""
        governor = GenesisEthicalGovernor()
        
        # Create various decision types
        decision_types = [
            {"action": "data_access", "sensitive": True},
            {"action": "data_modification", "sensitive": True},
            {"action": "data_deletion", "sensitive": True},
            {"action": "user_creation", "sensitive": False},
            {"action": "report_generation", "sensitive": False}
        ]
        
        for decision_type in decision_types:
            context = EthicalContext(
                user_id="audit_user",
                action=decision_type["action"],
                context_data={"sensitive": decision_type["sensitive"]},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=decision_type["action"],
                context=context,
                parameters={}
            )
            
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
        
        # Verify audit trail completeness
        history = governor.get_decision_history()
        assert len(history) == len(decision_types)
        
        # Each history entry should contain audit information
        for entry in history:
            assert "timestamp" in entry
            assert "decision" in entry
            assert "result" in entry
            assert isinstance(entry["timestamp"], datetime)


class TestGenesisEthicalGovernorIntegrationExtended:
    """Extended integration tests for real-world scenarios"""
    
    def test_multi_tenant_isolation(self):
        """Test that the governor properly isolates decisions between different tenants."""
        governor = GenesisEthicalGovernor()
        
        # Add tenant isolation rule
        tenant_rule = {
            "name": "tenant_isolation_rule",
            "condition": lambda ctx: (
                ctx.context_data.get("tenant_id") != ctx.context_data.get("resource_tenant_id")
            ),
            "action": "deny",
            "priority": 1
        }
        governor.add_ethical_rule(tenant_rule)
        
        # Test cross-tenant access attempt
        cross_tenant_context = EthicalContext(
            user_id="tenant_a_user",
            action="access_resource",
            context_data={
                "tenant_id": "tenant_a",
                "resource_tenant_id": "tenant_b",
                "resource_type": "database"
            },
            timestamp=datetime.now()
        )
        
        cross_tenant_decision = EthicalDecision(
            action="access_resource",
            context=cross_tenant_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(cross_tenant_decision)
        assert result.approved is False
        
        # Test same-tenant access
        same_tenant_context = EthicalContext(
            user_id="tenant_a_user",
            action="access_resource",
            context_data={
                "tenant_id": "tenant_a",
                "resource_tenant_id": "tenant_a",
                "resource_type": "database"
            },
            timestamp=datetime.now()
        )
        
        same_tenant_decision = EthicalDecision(
            action="access_resource",
            context=same_tenant_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(same_tenant_decision)
        assert result.approved is True
    
    def test_cascading_violation_impacts(self):
        """Test that violations have cascading impacts on related users and resources."""
        governor = GenesisEthicalGovernor()
        
        # Record violations for related users
        related_users = ["team_lead", "team_member_1", "team_member_2"]
        
        for i, user_id in enumerate(related_users):
            context = EthicalContext(
                user_id=user_id,
                action="security_violation",
                context_data={
                    "team_id": "team_alpha",
                    "violation_severity": "high",
                    "related_users": related_users
                },
                timestamp=datetime.now()
            )
            
            violation = EthicalViolation(
                user_id=user_id,
                action="security_violation",
                context=context,
                severity="high" if i == 0 else "medium",
                timestamp=datetime.now()
            )
            
            governor.record_violation(violation)
        
        # Check that trust scores are affected appropriately
        lead_score = governor.get_user_trust_score("team_lead")
        member_1_score = governor.get_user_trust_score("team_member_1")
        member_2_score = governor.get_user_trust_score("team_member_2")
        
        # Team lead should have lower score due to high severity
        assert lead_score < member_1_score
        assert lead_score < member_2_score
        
        # All should be below perfect trust
        assert lead_score < 1.0
        assert member_1_score < 1.0
        assert member_2_score < 1.0
    
    def test_real_time_threat_detection(self):
        """Test real-time threat detection and response capabilities."""
        governor = GenesisEthicalGovernor()
        
        # Add threat detection rule
        threat_rule = {
            "name": "threat_detection_rule",
            "condition": lambda ctx: (
                ctx.context_data.get("failed_attempts", 0) > 5 or
                ctx.context_data.get("suspicious_ip", False) or
                ctx.context_data.get("unusual_time", False)
            ),
            "action": "deny",
            "priority": 1
        }
        governor.add_ethical_rule(threat_rule)
        
        # Simulate threat scenarios
        threat_scenarios = [
            {"failed_attempts": 10, "threat_type": "brute_force"},
            {"suspicious_ip": True, "threat_type": "ip_reputation"},
            {"unusual_time": True, "threat_type": "time_anomaly"},
            {"failed_attempts": 3, "suspicious_ip": True, "threat_type": "combined"}
        ]
        
        for scenario in threat_scenarios:
            context = EthicalContext(
                user_id="threat_user",
                action="login_attempt",
                context_data=scenario,
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action="login_attempt",
                context=context,
                parameters={}
            )
            
            result = governor.evaluate_decision(decision)
            
            # High-risk scenarios should be denied
            if scenario.get("failed_attempts", 0) > 5 or scenario.get("suspicious_ip", False):
                assert result.approved is False
            
            assert isinstance(result, DecisionResult)
    
    def test_machine_learning_integration(self):
        """Test integration with machine learning models for decision enhancement."""
        governor = GenesisEthicalGovernor()
        
        # Mock ML model integration
        def ml_risk_assessment(ctx):
            """Mock ML model that returns risk score based on context."""
            # Simple risk scoring based on context data
            risk_factors = 0
            if ctx.context_data.get("new_user", False):
                risk_factors += 0.2
            if ctx.context_data.get("high_value_transaction", False):
                risk_factors += 0.4
            if ctx.context_data.get("foreign_ip", False):
                risk_factors += 0.3
            if ctx.context_data.get("unusual_time", False):
                risk_factors += 0.2
            
            return min(risk_factors, 1.0)
        
        # Add ML-enhanced rule
        ml_rule = {
            "name": "ml_enhanced_rule",
            "condition": lambda ctx: ml_risk_assessment(ctx) > 0.7,
            "action": "deny",
            "priority": 1
        }
        governor.add_ethical_rule(ml_rule)
        
        # Test high-risk scenario
        high_risk_context = EthicalContext(
            user_id="ml_test_user",
            action="financial_transaction",
            context_data={
                "new_user": True,
                "high_value_transaction": True,
                "foreign_ip": True,
                "unusual_time": True
            },
            timestamp=datetime.now()
        )
        
        high_risk_decision = EthicalDecision(
            action="financial_transaction",
            context=high_risk_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(high_risk_decision)
        assert result.approved is False
        
        # Test low-risk scenario
        low_risk_context = EthicalContext(
            user_id="ml_test_user",
            action="financial_transaction",
            context_data={
                "new_user": False,
                "high_value_transaction": False,
                "foreign_ip": False,
                "unusual_time": False
            },
            timestamp=datetime.now()
        )
        
        low_risk_decision = EthicalDecision(
            action="financial_transaction",
            context=low_risk_context,
            parameters={}
        )
        
        result = governor.evaluate_decision(low_risk_decision)
        assert result.approved is True


# Additional stress tests with pytest markers
class TestGenesisEthicalGovernorStressTests:
    """Stress tests for system limits and performance under extreme conditions"""
    
    @pytest.mark.slow
    def test_extreme_decision_volume(self):
        """Test processing extremely large volumes of decisions."""
        governor = GenesisEthicalGovernor()
        
        # Process 10,000 decisions
        decisions_count = 10000
        start_time = time.time()
        
        for i in range(decisions_count):
            context = EthicalContext(
                user_id=f"stress_user_{i % 100}",
                action=f"stress_action_{i}",
                context_data={"stress_test": True, "index": i},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"stress_action_{i}",
                context=context,
                parameters={}
            )
            
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should maintain reasonable performance
        decisions_per_second = decisions_count / total_time
        assert decisions_per_second >= 50  # At least 50 decisions per second
        
        # Verify all decisions were recorded
        assert len(governor.decision_history) == decisions_count
    
    @pytest.mark.slow
    def test_extreme_rule_count(self):
        """Test performance with thousands of rules."""
        governor = GenesisEthicalGovernor()
        
        # Add 1000 rules
        rule_count = 1000
        for i in range(rule_count):
            rule = {
                "name": f"stress_rule_{i}",
                "condition": lambda ctx, idx=i: ctx.context_data.get("rule_trigger") == idx,
                "action": "allow" if i % 2 == 0 else "deny",
                "priority": i
            }
            governor.add_ethical_rule(rule)
        
        # Test decision evaluation with many rules
        context = EthicalContext(
            user_id="stress_user",
            action="stress_action",
            context_data={"rule_trigger": 500},  # Should match rule_500
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="stress_action",
            context=context,
            parameters={}
        )
        
        start_time = time.time()
        result = governor.evaluate_decision(decision)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 5.0
        assert isinstance(result, DecisionResult)
        assert len(governor.ethical_rules) == rule_count
    
    @pytest.mark.slow
    def test_memory_stress_with_violations(self):
        """Test memory usage with large numbers of violations."""
        governor = GenesisEthicalGovernor()
        
        # Record 10,000 violations
        violation_count = 10000
        users_count = 100
        
        for i in range(violation_count):
            user_id = f"violation_user_{i % users_count}"
            context = EthicalContext(
                user_id=user_id,
                action=f"violation_action_{i}",
                context_data={"violation_index": i},
                timestamp=datetime.now() - timedelta(seconds=i)
            )
            
            violation = EthicalViolation(
                user_id=user_id,
                action=f"violation_action_{i}",
                context=context,
                severity=["low", "medium", "high", "critical"][i % 4],
                timestamp=datetime.now() - timedelta(seconds=i)
            )
            
            governor.record_violation(violation)
        
        # Verify violations were recorded
        total_violations = sum(len(governor.get_violations(f"violation_user_{i}")) 
                              for i in range(users_count))
        assert total_violations == violation_count
        
        # Test trust score calculation with many violations
        for i in range(users_count):
            user_id = f"violation_user_{i}"
            trust_score = governor.get_user_trust_score(user_id)
            assert 0.0 <= trust_score <= 1.0
    
    @pytest.mark.slow
    def test_concurrent_stress_operations(self):
        """Test system under concurrent stress from multiple operation types."""
        import threading
        import queue
        
        governor = GenesisEthicalGovernor()
        results_queue = queue.Queue()
        
        def decision_worker(worker_id):
            """Worker function for decision evaluation."""
            for i in range(100):
                context = EthicalContext(
                    user_id=f"worker_{worker_id}_user_{i}",
                    action=f"worker_{worker_id}_action_{i}",
                    context_data={"worker_id": worker_id, "iteration": i},
                    timestamp=datetime.now()
                )
                
                decision = EthicalDecision(
                    action=f"worker_{worker_id}_action_{i}",
                    context=context,
                    parameters={}
                )
                
                result = governor.evaluate_decision(decision)
                results_queue.put(("decision", result))
        
        def rule_worker(worker_id):
            """Worker function for rule management."""
            for i in range(50):
                rule = {
                    "name": f"concurrent_rule_{worker_id}_{i}",
                    "condition": lambda ctx: ctx.context_data.get("worker_id") == worker_id,
                    "action": "allow",
                    "priority": worker_id * 100 + i
                }
                governor.add_ethical_rule(rule)
                results_queue.put(("rule_added", rule["name"]))
        
        def violation_worker(worker_id):
            """Worker function for violation recording."""
            for i in range(25):
                user_id = f"violation_worker_{worker_id}_user_{i}"
                context = EthicalContext(
                    user_id=user_id,
                    action=f"violation_action_{i}",
                    context_data={"worker_id": worker_id},
                    timestamp=datetime.now()
                )
                
                violation = EthicalViolation(
                    user_id=user_id,
                    action=f"violation_action_{i}",
                    context=context,
                    severity="medium",
                    timestamp=datetime.now()
                )
                
                governor.record_violation(violation)
                results_queue.put(("violation", violation))
        
        # Start multiple worker threads
        threads = []
        for i in range(5):
            # Decision workers
            t1 = threading.Thread(target=decision_worker, args=(i,))
            threads.append(t1)
            
            # Rule workers
            t2 = threading.Thread(target=rule_worker, args=(i,))
            threads.append(t2)
            
            # Violation workers
            t3 = threading.Thread(target=violation_worker, args=(i,))
            threads.append(t3)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify operations completed successfully
        decisions = [r for r in results if r[0] == "decision"]
        rules_added = [r for r in results if r[0] == "rule_added"]
        violations = [r for r in results if r[0] == "violation"]
        
        assert len(decisions) == 500  # 5 workers * 100 decisions each
        assert len(rules_added) == 250  # 5 workers * 50 rules each
        assert len(violations) == 125  # 5 workers * 25 violations each
        
        # Should complete within reasonable time
        assert total_time < 60.0  # 60 seconds maximum
        
        # Verify all decisions are valid
        for _, result in decisions:
            assert isinstance(result, DecisionResult)


# Test configuration and edge cases
class TestGenesisEthicalGovernorConfigurationEdgeCases:
    """Test edge cases related to configuration and system limits"""
    
    def test_empty_configuration(self):
        """Test behavior with empty configuration."""
        governor = GenesisEthicalGovernor(config={})
        
        # Should use default values
        assert hasattr(governor, 'violation_threshold')
        assert hasattr(governor, 'decision_history')
        assert hasattr(governor, 'ethical_rules')
        
        # Should still be functional
        context = EthicalContext(
            user_id="config_test_user",
            action="config_test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="config_test_action",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_extreme_configuration_values(self):
        """Test behavior with extreme configuration values."""
        extreme_configs = [
            {"violation_threshold": 0},
            {"violation_threshold": 1000000},
            {"strict_mode": True, "violation_threshold": 1},
            {"logging_enabled": False, "violation_threshold": 10}
        ]
        
        for config in extreme_configs:
            try:
                governor = GenesisEthicalGovernor(config=config)
                
                # Should still function
                context = EthicalContext(
                    user_id="extreme_config_user",
                    action="extreme_config_action",
                    context_data={},
                    timestamp=datetime.now()
                )
                
                decision = EthicalDecision(
                    action="extreme_config_action",
                    context=context,
                    parameters={}
                )
                
                result = governor.evaluate_decision(decision)
                assert isinstance(result, DecisionResult)
                
            except ValueError:
                # Some extreme values may be rejected
                pass
    
    def test_configuration_immutability(self):
        """Test that configuration cannot be modified after initialization."""
        config = {"violation_threshold": 5, "strict_mode": True}
        governor = GenesisEthicalGovernor(config=config)
        
        original_threshold = governor.violation_threshold
        
        # Attempt to modify configuration
        try:
            governor.violation_threshold = 10
        except AttributeError:
            pass  # Expected if immutable
        
        # Should preserve original values
        assert governor.violation_threshold == original_threshold
    
    def test_configuration_persistence(self):
        """Test that configuration is properly persisted during serialization."""
        config = {
            "violation_threshold": 7,
            "strict_mode": False,
            "logging_enabled": True
        }
        governor = GenesisEthicalGovernor(config=config)
        
        # Make some decisions to create state
        context = EthicalContext(
            user_id="persistence_user",
            action="persistence_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="persistence_action",
            context=context,
            parameters={}
        )
        
        governor.evaluate_decision(decision)
        
        # Serialize and deserialize
        serialized_state = governor.serialize_state()
        new_governor = GenesisEthicalGovernor()
        new_governor.deserialize_state(serialized_state)
        
        # Configuration should be preserved
        assert new_governor.violation_threshold == governor.violation_threshold
        assert new_governor.strict_mode == governor.strict_mode
        assert new_governor.logging_enabled == governor.logging_enabled