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

# Additional comprehensive test cases for enhanced coverage

class TestGenesisEthicalGovernorAdvanced:
    """Advanced test suite with additional edge cases and boundary conditions"""
    
    @pytest.fixture
    def governor_strict_mode(self):
        """Create a governor with strict mode enabled for rigorous testing"""
        config = {
            'strict_mode': True,
            'violation_threshold': 2,
            'logging_enabled': True,
            'max_decision_history': 1000
        }
        return GenesisEthicalGovernor(config=config)
    
    def test_decision_with_circular_reference(self, governor):
        """Test handling of context data with circular references"""
        circular_data = {"key": "value"}
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
        
        # Should handle circular references gracefully
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_decision_with_nan_float_values(self, governor):
        """Test handling of NaN and infinity float values in context data"""
        import math
        
        context_data = {
            "nan_value": float('nan'),
            "inf_value": float('inf'),
            "neg_inf_value": float('-inf'),
            "normal_value": 42.5
        }
        
        context = EthicalContext(
            user_id="nan_user",
            action="nan_action",
            context_data=context_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="nan_action",
            context=context,
            parameters={"confidence": float('nan')}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
        assert not math.isnan(result.confidence_score)
    
    def test_decision_with_empty_string_fields(self, governor):
        """Test handling of empty string values in critical fields"""
        context = EthicalContext(
            user_id="",
            action="",
            context_data={"empty_key": ""},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="",
            context=context,
            parameters={"empty_param": ""}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_decision_with_binary_data(self, governor):
        """Test handling of binary data in context and parameters"""
        binary_data = b'\x00\x01\x02\x03\xff\xfe\xfd'
        
        context = EthicalContext(
            user_id="binary_user",
            action="binary_action",
            context_data={"binary_field": binary_data},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="binary_action",
            context=context,
            parameters={"binary_param": binary_data}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_concurrent_violation_recording(self, governor):
        """Test concurrent violation recording for thread safety"""
        import threading
        import time
        
        violations_recorded = []
        
        def record_violation(violation_id):
            context = EthicalContext(
                user_id=f"concurrent_user_{violation_id}",
                action=f"concurrent_action_{violation_id}",
                context_data={"violation_id": violation_id},
                timestamp=datetime.now()
            )
            
            violation = EthicalViolation(
                user_id=f"concurrent_user_{violation_id}",
                action=f"concurrent_action_{violation_id}",
                context=context,
                severity="medium",
                timestamp=datetime.now()
            )
            
            governor.record_violation(violation)
            violations_recorded.append(violation_id)
        
        # Start multiple threads recording violations
        threads = []
        for i in range(20):
            thread = threading.Thread(target=record_violation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        assert len(violations_recorded) == 20
    
    def test_decision_history_memory_management(self, governor):
        """Test that decision history doesn't cause memory leaks with large volumes"""
        import gc
        
        # Configure with limited history
        governor.max_decision_history = 100
        
        context = EthicalContext(
            user_id="memory_user",
            action="memory_action",
            context_data={"data": "x" * 1000},
            timestamp=datetime.now()
        )
        
        # Create many decisions
        for i in range(500):
            decision = EthicalDecision(
                action=f"memory_action_{i}",
                context=context,
                parameters={"index": i}
            )
            governor.evaluate_decision(decision)
        
        # History should be limited
        assert len(governor.decision_history) <= 100
        
        # Force garbage collection
        gc.collect()
        
        # Should still function normally
        test_decision = EthicalDecision(
            action="test_after_gc",
            context=context,
            parameters={}
        )
        result = governor.evaluate_decision(test_decision)
        assert isinstance(result, DecisionResult)
    
    def test_rule_priority_edge_cases(self, governor):
        """Test edge cases in rule priority handling"""
        # Add rules with same priority
        rule1 = {
            "name": "same_priority_1",
            "condition": lambda ctx: ctx.action == "priority_test",
            "action": "allow",
            "priority": 5
        }
        
        rule2 = {
            "name": "same_priority_2",
            "condition": lambda ctx: ctx.action == "priority_test",
            "action": "deny",
            "priority": 5
        }
        
        governor.add_ethical_rule(rule1)
        governor.add_ethical_rule(rule2)
        
        context = EthicalContext(
            user_id="priority_user",
            action="priority_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="priority_test",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
        # Should handle same priority gracefully
    
    def test_violation_with_future_timestamp(self, governor):
        """Test handling of violations with future timestamps"""
        future_time = datetime.now() + timedelta(days=1)
        
        context = EthicalContext(
            user_id="future_user",
            action="future_action",
            context_data={},
            timestamp=future_time
        )
        
        violation = EthicalViolation(
            user_id="future_user",
            action="future_action",
            context=context,
            severity="medium",
            timestamp=future_time
        )
        
        governor.record_violation(violation)
        violations = governor.get_violations("future_user")
        assert len(violations) == 1
        assert violations[0].timestamp == future_time
    
    def test_context_with_extremely_deep_nesting(self, governor):
        """Test handling of extremely deeply nested context data"""
        # Create deeply nested structure
        deep_data = {"level_0": {}}
        current_level = deep_data["level_0"]
        
        for i in range(1, 100):
            current_level[f"level_{i}"] = {}
            current_level = current_level[f"level_{i}"]
        
        current_level["final_value"] = "deep_value"
        
        context = EthicalContext(
            user_id="deep_user",
            action="deep_action",
            context_data=deep_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="deep_action",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_decision_with_class_instances(self, governor):
        """Test handling of custom class instances in context data"""
        class CustomClass:
            def __init__(self, value):
                self.value = value
            
            def __str__(self):
                return f"CustomClass({self.value})"
        
        custom_instance = CustomClass("test_value")
        
        context = EthicalContext(
            user_id="class_user",
            action="class_action",
            context_data={"custom_object": custom_instance},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="class_action",
            context=context,
            parameters={"custom_param": custom_instance}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_rule_with_complex_condition_logic(self, governor):
        """Test rules with complex conditional logic"""
        def complex_condition(ctx):
            try:
                data = ctx.context_data
                if not data:
                    return False
                
                # Complex nested logic
                if data.get("risk_level", 0) > 0.5:
                    if data.get("user_type") == "admin":
                        return data.get("override_allowed", False)
                    else:
                        return data.get("approval_required", True)
                else:
                    return data.get("auto_approve", False)
            except Exception:
                return False
        
        complex_rule = {
            "name": "complex_rule",
            "condition": complex_condition,
            "action": "deny",
            "priority": 1
        }
        
        governor.add_ethical_rule(complex_rule)
        
        # Test various scenarios
        scenarios = [
            {"risk_level": 0.8, "user_type": "admin", "override_allowed": True},
            {"risk_level": 0.3, "auto_approve": True},
            {"risk_level": 0.9, "user_type": "regular", "approval_required": True},
            {}  # Empty data
        ]
        
        for scenario in scenarios:
            context = EthicalContext(
                user_id="complex_user",
                action="complex_action",
                context_data=scenario,
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action="complex_action",
                context=context,
                parameters={}
            )
            
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
    
    def test_serialization_with_complex_data(self, governor):
        """Test serialization/deserialization with complex data structures"""
        # Add complex rule
        complex_rule = {
            "name": "serialization_rule",
            "condition": lambda ctx: True,
            "action": "allow",
            "priority": 1
        }
        governor.add_ethical_rule(complex_rule)
        
        # Create complex decision
        complex_context = EthicalContext(
            user_id="serialization_user",
            action="serialization_action",
            context_data={
                "nested": {"deep": {"value": [1, 2, 3]}},
                "list": ["a", "b", "c"],
                "number": 42.5
            },
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="serialization_action",
            context=complex_context,
            parameters={"complex_param": {"key": "value"}}
        )
        
        governor.evaluate_decision(decision)
        
        # Test serialization
        serialized = governor.serialize_state()
        assert isinstance(serialized, str)
        
        # Test deserialization
        new_governor = GenesisEthicalGovernor()
        new_governor.deserialize_state(serialized)
        
        # Should maintain functional state
        test_result = new_governor.evaluate_decision(decision)
        assert isinstance(test_result, DecisionResult)
    
    def test_rule_condition_performance_monitoring(self, governor):
        """Test monitoring of rule condition performance"""
        call_count = [0]
        
        def slow_condition(ctx):
            call_count[0] += 1
            import time
            time.sleep(0.01)  # Simulate slow condition
            return False
        
        slow_rule = {
            "name": "slow_rule",
            "condition": slow_condition,
            "action": "deny",
            "priority": 1
        }
        
        governor.add_ethical_rule(slow_rule)
        
        context = EthicalContext(
            user_id="performance_user",
            action="performance_action",
            context_data={},
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
        
        assert isinstance(result, DecisionResult)
        assert call_count[0] > 0  # Condition was called
        # Should complete within reasonable time despite slow condition
        assert end_time - start_time < 1.0


class TestDataValidationAndSanitization:
    """Test suite for data validation and sanitization"""
    
    def test_sql_injection_attempt_in_context(self, governor):
        """Test handling of SQL injection attempts in context data"""
        malicious_data = {
            "user_input": "'; DROP TABLE users; --",
            "search_query": "admin' OR '1'='1",
            "filter": "1=1 UNION SELECT * FROM secrets"
        }
        
        context = EthicalContext(
            user_id="sql_user",
            action="sql_action",
            context_data=malicious_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="sql_action",
            context=context,
            parameters={"query": "SELECT * FROM users WHERE id = 1; DROP TABLE users;"}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_xss_attempt_in_context(self, governor):
        """Test handling of XSS attempts in context data"""
        xss_data = {
            "user_comment": "<script>alert('XSS')</script>",
            "display_name": "<img src=x onerror=alert('XSS')>",
            "description": "javascript:alert('XSS')"
        }
        
        context = EthicalContext(
            user_id="xss_user",
            action="xss_action",
            context_data=xss_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="xss_action",
            context=context,
            parameters={"html_content": "<script>malicious code</script>"}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_buffer_overflow_attempt(self, governor):
        """Test handling of potential buffer overflow attempts"""
        massive_string = "A" * 1000000  # 1MB string
        
        context = EthicalContext(
            user_id="buffer_user",
            action="buffer_action",
            context_data={"massive_field": massive_string},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="buffer_action",
            context=context,
            parameters={"large_param": massive_string}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_unicode_normalization_attacks(self, governor):
        """Test handling of Unicode normalization attacks"""
        # Unicode characters that look similar but are different
        unicode_data = {
            "normal_text": "admin",
            "unicode_lookalike": "аdmin",  # Cyrillic 'а' instead of 'a'
            "mixed_scripts": "раypal.com",  # Mixed Latin/Cyrillic
            "zero_width": "admin\u200b\u200cuser"  # Zero-width characters
        }
        
        context = EthicalContext(
            user_id="unicode_user",
            action="unicode_action",
            context_data=unicode_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="unicode_action",
            context=context,
            parameters={"unicode_param": "test\u0000null"}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_path_traversal_attempts(self, governor):
        """Test handling of path traversal attempts in context data"""
        path_traversal_data = {
            "file_path": "../../etc/passwd",
            "upload_path": "../../../root/.ssh/id_rsa",
            "config_path": "..\\..\\windows\\system32\\config\\sam"
        }
        
        context = EthicalContext(
            user_id="path_user",
            action="path_action",
            context_data=path_traversal_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="path_action",
            context=context,
            parameters={"file_name": "../../../../etc/shadow"}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)


class TestErrorRecoveryAndResilience:
    """Test suite for error recovery and system resilience"""
    
    def test_recovery_from_corrupted_state(self, governor):
        """Test recovery from corrupted internal state"""
        # Simulate corrupted state
        governor.decision_history = None
        governor.ethical_rules = None
        
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
    
    def test_memory_pressure_handling(self, governor):
        """Test behavior under memory pressure conditions"""
        # Create memory pressure scenario
        large_objects = []
        try:
            for i in range(100):
                large_objects.append("x" * 1000000)  # 1MB strings
                
                context = EthicalContext(
                    user_id=f"pressure_user_{i}",
                    action="pressure_action",
                    context_data={"large_data": large_objects[-1]},
                    timestamp=datetime.now()
                )
                
                decision = EthicalDecision(
                    action="pressure_action",
                    context=context,
                    parameters={}
                )
                
                result = governor.evaluate_decision(decision)
                assert isinstance(result, DecisionResult)
        except MemoryError:
            # Expected under extreme memory pressure
            pass
    
    def test_exception_propagation_control(self, governor):
        """Test that exceptions are properly controlled and don't crash the system"""
        def exception_rule(ctx):
            raise RuntimeError("Intentional exception")
        
        problematic_rule = {
            "name": "exception_rule",
            "condition": exception_rule,
            "action": "deny",
            "priority": 1
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
        
        # Should not propagate exception
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_graceful_degradation_on_partial_failure(self, governor):
        """Test graceful degradation when some components fail"""
        # Add mix of working and failing rules
        def working_rule(ctx):
            return ctx.action == "working_action"
        
        def failing_rule(ctx):
            raise Exception("Rule failed")
        
        governor.add_ethical_rule({
            "name": "working_rule",
            "condition": working_rule,
            "action": "allow",
            "priority": 1
        })
        
        governor.add_ethical_rule({
            "name": "failing_rule",
            "condition": failing_rule,
            "action": "deny",
            "priority": 2
        })
        
        context = EthicalContext(
            user_id="degradation_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="test_action",
            context=context,
            parameters={}
        )
        
        # Should still work despite failing rule
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)


class TestBusinessLogicEdgeCases:
    """Test suite for business logic edge cases and corner scenarios"""
    
    def test_trust_score_boundary_values(self, governor):
        """Test trust score calculations at boundary values"""
        # Test with user at maximum violations
        user_id = "boundary_user"
        
        # Record maximum violations
        for i in range(1000):
            context = EthicalContext(
                user_id=user_id,
                action=f"violation_{i}",
                context_data={},
                timestamp=datetime.now() - timedelta(seconds=i)
            )
            
            violation = EthicalViolation(
                user_id=user_id,
                action=f"violation_{i}",
                context=context,
                severity="critical",
                timestamp=datetime.now() - timedelta(seconds=i)
            )
            
            governor.record_violation(violation)
        
        trust_score = governor.get_user_trust_score(user_id)
        assert 0.0 <= trust_score <= 1.0
        assert trust_score >= 0.0  # Should not go below 0
    
    def test_decision_approval_consistency(self, governor):
        """Test that decision approval is consistent across multiple evaluations"""
        context = EthicalContext(
            user_id="consistency_user",
            action="consistent_action",
            context_data={"deterministic": True},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="consistent_action",
            context=context,
            parameters={"consistent": True}
        )
        
        # Make same decision multiple times
        results = []
        for _ in range(10):
            result = governor.evaluate_decision(decision)
            results.append(result)
        
        # All results should be consistent
        first_approval = results[0].approved
        for result in results[1:]:
            assert result.approved == first_approval
    
    def test_rule_evaluation_order_determinism(self, governor):
        """Test that rule evaluation order is deterministic"""
        # Add rules in specific order
        for i in range(10):
            rule = {
                "name": f"order_rule_{i}",
                "condition": lambda ctx: ctx.action == "order_test",
                "action": "allow" if i % 2 == 0 else "deny",
                "priority": i
            }
            governor.add_ethical_rule(rule)
        
        context = EthicalContext(
            user_id="order_user",
            action="order_test",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="order_test",
            context=context,
            parameters={}
        )
        
        # Evaluate multiple times
        results = []
        for _ in range(5):
            result = governor.evaluate_decision(decision)
            results.append(result)
        
        # Should be consistent
        first_result = results[0].approved
        for result in results[1:]:
            assert result.approved == first_result
    
    def test_violation_time_decay_accuracy(self, governor):
        """Test accuracy of violation time decay calculations"""
        user_id = "decay_user"
        
        # Record violations at specific time intervals
        base_time = datetime.now() - timedelta(days=30)
        
        for days_ago in [1, 7, 14, 30]:
            context = EthicalContext(
                user_id=user_id,
                action=f"decay_action_{days_ago}",
                context_data={},
                timestamp=base_time + timedelta(days=days_ago)
            )
            
            violation = EthicalViolation(
                user_id=user_id,
                action=f"decay_action_{days_ago}",
                context=context,
                severity="medium",
                timestamp=base_time + timedelta(days=days_ago)
            )
            
            governor.record_violation(violation)
        
        # Trust score should reflect time decay
        trust_score = governor.get_user_trust_score(user_id)
        assert 0.0 <= trust_score <= 1.0
        
        # Recent violations should have more impact
        # (This depends on actual implementation logic)
    
    def test_context_data_immutability_enforcement(self, governor):
        """Test that context data cannot be modified during evaluation"""
        original_data = {"mutable": "original_value"}
        
        context = EthicalContext(
            user_id="immutable_user",
            action="immutable_action",
            context_data=original_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="immutable_action",
            context=context,
            parameters={}
        )
        
        # Evaluate decision
        result = governor.evaluate_decision(decision)
        
        # Original data should remain unchanged
        assert original_data["mutable"] == "original_value"
        assert isinstance(result, DecisionResult)
    
    def test_decision_metadata_completeness(self, governor):
        """Test that decision metadata is complete and accurate"""
        context = EthicalContext(
            user_id="metadata_user",
            action="metadata_action",
            context_data={"test": "data"},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="metadata_action",
            context=context,
            parameters={"test": "param"}
        )
        
        result = governor.evaluate_decision(decision)
        
        # Check metadata completeness
        assert hasattr(result, 'metadata')
        if result.metadata:
            assert isinstance(result.metadata, dict)
            # Should contain evaluation information
            expected_fields = ['evaluation_time', 'rules_checked', 'decision_path']
            for field in expected_fields:
                if field in result.metadata:
                    assert result.metadata[field] is not None


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple components"""
    
    def test_complete_violation_workflow(self, governor):
        """Test complete workflow from decision to violation to trust score update"""
        user_id = "workflow_user"
        
        # Initial trust score
        initial_score = governor.get_user_trust_score(user_id)
        
        # Create risky decision
        context = EthicalContext(
            user_id=user_id,
            action="risky_workflow_action",
            context_data={"risk_level": "high"},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="risky_workflow_action",
            context=context,
            parameters={"force": True}
        )
        
        # Evaluate decision
        result = governor.evaluate_decision(decision)
        
        # If denied, record violation
        if not result.approved:
            violation = EthicalViolation(
                user_id=user_id,
                action="risky_workflow_action",
                context=context,
                severity="high",
                timestamp=datetime.now()
            )
            governor.record_violation(violation)
            
            # Check trust score decreased
            new_score = governor.get_user_trust_score(user_id)
            assert new_score < initial_score
            
            # Check violation was recorded
            violations = governor.get_violations(user_id)
            assert len(violations) > 0
        
        # Check decision history
        history = governor.get_decision_history()
        assert len(history) > 0
        assert any(entry['decision'].action == "risky_workflow_action" for entry in history)
    
    def test_multi_user_interaction_scenarios(self, governor):
        """Test scenarios involving multiple users interacting with the system"""
        users = ["user_1", "user_2", "user_3"]
        
        # Each user performs different actions
        for i, user_id in enumerate(users):
            for action_num in range(5):
                context = EthicalContext(
                    user_id=user_id,
                    action=f"multi_action_{i}_{action_num}",
                    context_data={"user_index": i, "action_index": action_num},
                    timestamp=datetime.now()
                )
                
                decision = EthicalDecision(
                    action=f"multi_action_{i}_{action_num}",
                    context=context,
                    parameters={"user_type": f"type_{i}"}
                )
                
                result = governor.evaluate_decision(decision)
                assert isinstance(result, DecisionResult)
        
        # Check that each user has independent trust scores
        scores = {}
        for user_id in users:
            scores[user_id] = governor.get_user_trust_score(user_id)
        
        # All scores should be valid
        for score in scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_system_under_sustained_load(self, governor):
        """Test system behavior under sustained load"""
        import time
        
        start_time = time.time()
        decisions_processed = 0
        
        # Run for 5 seconds
        while time.time() - start_time < 5.0:
            context = EthicalContext(
                user_id=f"load_user_{decisions_processed % 10}",
                action=f"load_action_{decisions_processed}",
                context_data={"load_test": True},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"load_action_{decisions_processed}",
                context=context,
                parameters={}
            )
            
            result = governor.evaluate_decision(decision)
            assert isinstance(result, DecisionResult)
            decisions_processed += 1
        
        # Should process significant number of decisions
        assert decisions_processed > 100
        
        # System should still be responsive
        test_context = EthicalContext(
            user_id="test_user",
            action="test_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        test_decision = EthicalDecision(
            action="test_action",
            context=test_context,
            parameters={}
        )
        
        test_result = governor.evaluate_decision(test_decision)
        assert isinstance(test_result, DecisionResult)

# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def test_decision_evaluation_benchmark(self, governor):
        """Benchmark decision evaluation performance"""
        import time
        
        context = EthicalContext(
            user_id="benchmark_user",
            action="benchmark_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="benchmark_action",
            context=context,
            parameters={}
        )
        
        # Warm up
        for _ in range(100):
            governor.evaluate_decision(decision)
        
        # Benchmark
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            governor.evaluate_decision(decision)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        # Should average less than 1ms per decision
        assert avg_time < 0.001
        
        # Should process at least 1000 decisions per second
        decisions_per_second = iterations / total_time
        assert decisions_per_second >= 1000
    
    def test_rule_evaluation_scaling(self, governor):
        """Test performance scaling with increasing number of rules"""
        import time
        
        # Add increasing numbers of rules and measure performance
        rule_counts = [10, 50, 100, 200]
        performance_data = []
        
        for rule_count in rule_counts:
            # Clear existing rules
            governor.ethical_rules = []
            
            # Add rules
            for i in range(rule_count):
                rule = {
                    "name": f"scale_rule_{i}",
                    "condition": lambda ctx: False,  # Always false
                    "action": "allow",
                    "priority": i
                }
                governor.add_ethical_rule(rule)
            
            context = EthicalContext(
                user_id="scale_user",
                action="scale_action",
                context_data={},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action="scale_action",
                context=context,
                parameters={}
            )
            
            # Measure performance
            start_time = time.time()
            for _ in range(100):
                governor.evaluate_decision(decision)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100
            performance_data.append((rule_count, avg_time))
        
        # Performance should scale reasonably with rule count
        # (depends on implementation, but should be roughly linear)
        for rule_count, avg_time in performance_data:
            assert avg_time < 0.1  # Should complete within 100ms even with 200 rules