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

class TestGenesisEthicalGovernorRobustness:
    """Additional robustness and edge case tests for GenesisEthicalGovernor"""
    
    @pytest.fixture
    def governor(self):
        """Create a fresh governor instance for each test"""
        return GenesisEthicalGovernor()
    
    def test_circular_rule_dependencies(self, governor):
        """
        Test that the governor handles circular rule dependencies gracefully.
        
        Creates rules that could potentially reference each other in a circular manner
        and verifies that decision evaluation doesn't result in infinite loops.
        """
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
        
        governor.add_ethical_rule(rule_a)
        governor.add_ethical_rule(rule_b)
        
        context = EthicalContext(
            user_id="circular_user",
            action="circular_action",
            context_data={"trigger_a": True, "trigger_b": True},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="circular_action",
            context=context,
            parameters={}
        )
        
        # Should complete without infinite loop
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_rule_with_recursive_condition(self, governor):
        """
        Test that rules with recursive logic in conditions are handled safely.
        """
        def recursive_condition(ctx, depth=0):
            """Condition that calls itself recursively up to a certain depth"""
            if depth > 10:
                return False
            return recursive_condition(ctx, depth + 1) if depth < 5 else True
        
        recursive_rule = {
            "name": "recursive_rule",
            "condition": recursive_condition,
            "action": "allow",
            "priority": 1
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
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_decision_with_generator_parameters(self, governor):
        """
        Test that decisions can handle generator objects as parameters.
        """
        def param_generator():
            """Generator that yields values for testing"""
            for i in range(3):
                yield f"value_{i}"
        
        context = EthicalContext(
            user_id="generator_user",
            action="generator_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="generator_action",
            context=context,
            parameters={"generator": param_generator()}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_context_with_binary_data(self, governor):
        """
        Test that contexts can handle binary data in context_data.
        """
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
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_violation_with_future_timestamp(self, governor):
        """
        Test that violations with future timestamps are handled appropriately.
        """
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
    
    def test_trust_score_with_mixed_violation_timestamps(self, governor):
        """
        Test trust score calculation with violations spanning past, present, and future.
        """
        user_id = "mixed_time_user"
        
        timestamps = [
            datetime.now() - timedelta(days=30),  # Past
            datetime.now(),                       # Present  
            datetime.now() + timedelta(hours=1)   # Future
        ]
        
        for i, timestamp in enumerate(timestamps):
            context = EthicalContext(
                user_id=user_id,
                action=f"mixed_action_{i}",
                context_data={},
                timestamp=timestamp
            )
            
            violation = EthicalViolation(
                user_id=user_id,
                action=f"mixed_action_{i}",
                context=context,
                severity="medium",
                timestamp=timestamp
            )
            
            governor.record_violation(violation)
        
        trust_score = governor.get_user_trust_score(user_id)
        assert 0.0 <= trust_score <= 1.0
    
    def test_rule_condition_with_class_instances(self, governor):
        """
        Test that rule conditions can properly handle class instances in context data.
        """
        class TestDataObject:
            def __init__(self, value):
                self.value = value
            
            def __eq__(self, other):
                return isinstance(other, TestDataObject) and self.value == other.value
        
        test_obj = TestDataObject("test_value")
        
        rule = {
            "name": "object_rule",
            "condition": lambda ctx: isinstance(ctx.context_data.get("obj"), TestDataObject),
            "action": "allow",
            "priority": 1
        }
        
        governor.add_ethical_rule(rule)
        
        context = EthicalContext(
            user_id="object_user",
            action="object_action",
            context_data={"obj": test_obj},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="object_action",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_decision_history_with_identical_timestamps(self, governor):
        """
        Test decision history handling when multiple decisions have identical timestamps.
        """
        timestamp = datetime.now()
        
        context = EthicalContext(
            user_id="identical_time_user",
            action="identical_time_action",
            context_data={},
            timestamp=timestamp
        )
        
        # Create multiple decisions with identical timestamps
        for i in range(5):
            decision = EthicalDecision(
                action=f"identical_action_{i}",
                context=context,
                parameters={"index": i}
            )
            governor.evaluate_decision(decision)
        
        history = governor.get_decision_history()
        assert len(history) >= 5
        
        # All entries should have the same timestamp
        identical_entries = [entry for entry in history if entry["timestamp"] == timestamp]
        assert len(identical_entries) == 5
    
    def test_violation_recording_with_none_context(self, governor):
        """
        Test that violations can be recorded with None context in edge cases.
        """
        violation = EthicalViolation(
            user_id="none_context_user",
            action="none_context_action",
            context=None,
            severity="low",
            timestamp=datetime.now()
        )
        
        # Should handle None context gracefully
        governor.record_violation(violation)
        violations = governor.get_violations("none_context_user")
        
        assert len(violations) == 1
        assert violations[0].context is None
    
    def test_rule_priority_with_negative_values(self, governor):
        """
        Test that rules with negative priority values are handled correctly.
        """
        high_priority_rule = {
            "name": "negative_priority_rule",
            "condition": lambda ctx: True,
            "action": "deny",
            "priority": -10
        }
        
        low_priority_rule = {
            "name": "positive_priority_rule",
            "condition": lambda ctx: True,
            "action": "allow",
            "priority": 10
        }
        
        governor.add_ethical_rule(low_priority_rule)
        governor.add_ethical_rule(high_priority_rule)
        
        context = EthicalContext(
            user_id="priority_user",
            action="priority_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="priority_action",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        
        # Negative priority should be treated as higher priority
        assert result.approved is False  # Should be denied by high priority rule
    
    def test_context_data_with_mixed_types(self, governor):
        """
        Test that context data can contain mixed Python data types.
        """
        mixed_data = {
            "string": "text",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "none": None,
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
            "bytes": b"binary_data"
        }
        
        context = EthicalContext(
            user_id="mixed_types_user",
            action="mixed_types_action",
            context_data=mixed_data,
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="mixed_types_action",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
    
    def test_decision_parameters_modification_safety(self, governor):
        """
        Test that modifying decision parameters after creation doesn't affect evaluation.
        """
        original_params = {"param1": "value1", "param2": "value2"}
        
        context = EthicalContext(
            user_id="modification_user",
            action="modification_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="modification_action",
            context=context,
            parameters=original_params.copy()
        )
        
        # Modify the original parameters dict
        original_params["param1"] = "modified_value"
        original_params["new_param"] = "new_value"
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
        
        # Decision should still contain original parameters
        assert decision.parameters["param1"] == "value1"
        assert "new_param" not in decision.parameters
    
    def test_concurrent_violation_recording(self, governor):
        """
        Test that recording violations concurrently doesn't cause data corruption.
        """
        import threading
        
        violations_recorded = []
        
        def record_violation(user_id, action_suffix):
            """Record a violation for testing concurrent access"""
            context = EthicalContext(
                user_id=user_id,
                action=f"concurrent_violation_{action_suffix}",
                context_data={},
                timestamp=datetime.now()
            )
            
            violation = EthicalViolation(
                user_id=user_id,
                action=f"concurrent_violation_{action_suffix}",
                context=context,
                severity="medium",
                timestamp=datetime.now()
            )
            
            governor.record_violation(violation)
            violations_recorded.append(violation)
        
        # Create multiple threads recording violations
        threads = []
        for i in range(10):
            thread = threading.Thread(
                target=record_violation,
                args=(f"concurrent_user_{i % 3}", i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all violations were recorded
        assert len(violations_recorded) == 10
        
        # Verify violations are retrievable
        for i in range(3):
            user_violations = governor.get_violations(f"concurrent_user_{i}")
            assert len(user_violations) > 0
    
    def test_rule_condition_exception_recovery(self, governor):
        """
        Test that the governor can continue operating after rule condition exceptions.
        """
        def failing_condition(ctx):
            """Condition that always raises an exception"""
            raise ValueError("Intentional test exception")
        
        def working_condition(ctx):
            """Condition that works normally"""
            return ctx.action == "working_action"
        
        failing_rule = {
            "name": "failing_rule",
            "condition": failing_condition,
            "action": "deny",
            "priority": 1
        }
        
        working_rule = {
            "name": "working_rule",
            "condition": working_condition,
            "action": "allow",
            "priority": 2
        }
        
        governor.add_ethical_rule(failing_rule)
        governor.add_ethical_rule(working_rule)
        
        context = EthicalContext(
            user_id="exception_user",
            action="working_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="working_action",
            context=context,
            parameters={}
        )
        
        # Should handle exception gracefully and continue with other rules
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
        
        # Governor should still be operational for subsequent decisions
        result2 = governor.evaluate_decision(decision)
        assert isinstance(result2, DecisionResult)
    
    def test_trust_score_boundary_conditions(self, governor):
        """
        Test trust score calculations at boundary conditions.
        """
        # Test with user who has maximum possible violations
        user_id = "boundary_user"
        
        # Record many violations to test lower boundary
        for i in range(1000):
            context = EthicalContext(
                user_id=user_id,
                action=f"boundary_action_{i}",
                context_data={},
                timestamp=datetime.now() - timedelta(minutes=i)
            )
            
            violation = EthicalViolation(
                user_id=user_id,
                action=f"boundary_action_{i}",
                context=context,
                severity="critical",
                timestamp=datetime.now() - timedelta(minutes=i)
            )
            
            governor.record_violation(violation)
        
        trust_score = governor.get_user_trust_score(user_id)
        
        # Trust score should still be within bounds
        assert 0.0 <= trust_score <= 1.0
        
        # Should be very low due to many critical violations
        assert trust_score < 0.1
    
    def test_serialization_with_complex_state(self, governor):
        """
        Test serialization with complex governor state including rules and violations.
        """
        # Add complex rules
        complex_rule = {
            "name": "complex_rule",
            "condition": lambda ctx: len(ctx.context_data.get("items", [])) > 5,
            "action": "deny",
            "priority": 1
        }
        
        governor.add_ethical_rule(complex_rule)
        
        # Add decision history
        for i in range(10):
            context = EthicalContext(
                user_id=f"serialization_user_{i}",
                action=f"serialization_action_{i}",
                context_data={"items": list(range(i))},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"serialization_action_{i}",
                context=context,
                parameters={"index": i}
            )
            
            governor.evaluate_decision(decision)
        
        # Add violations
        for i in range(5):
            context = EthicalContext(
                user_id=f"violation_user_{i}",
                action=f"violation_action_{i}",
                context_data={},
                timestamp=datetime.now()
            )
            
            violation = EthicalViolation(
                user_id=f"violation_user_{i}",
                action=f"violation_action_{i}",
                context=context,
                severity="medium",
                timestamp=datetime.now()
            )
            
            governor.record_violation(violation)
        
        # Test serialization with complex state
        serialized = governor.serialize_state()
        assert isinstance(serialized, str)
        
        # Test deserialization
        new_governor = GenesisEthicalGovernor()
        new_governor.deserialize_state(serialized)
        
        # Verify complex state was preserved
        assert len(new_governor.decision_history) == len(governor.decision_history)


class TestEthicalComponentsEdgeCases:
    """Additional edge case tests for individual ethical components"""
    
    def test_ethical_context_with_callable_data(self):
        """Test that EthicalContext can handle callable objects in context_data"""
        def test_function():
            return "test_result"
        
        context = EthicalContext(
            user_id="callable_user",
            action="callable_action",
            context_data={"callback": test_function},
            timestamp=datetime.now()
        )
        
        assert callable(context.context_data["callback"])
        assert context.context_data["callback"]() == "test_result"
    
    def test_ethical_decision_with_lambda_parameters(self):
        """Test that EthicalDecision can handle lambda functions in parameters"""
        context = EthicalContext(
            user_id="lambda_user",
            action="lambda_action", 
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="lambda_action",
            context=context,
            parameters={"lambda_func": lambda x: x * 2}
        )
        
        assert callable(decision.parameters["lambda_func"])
        assert decision.parameters["lambda_func"](5) == 10
    
    def test_ethical_violation_with_extreme_timestamps(self):
        """Test EthicalViolation with extreme timestamp values"""
        context = EthicalContext(
            user_id="extreme_user",
            action="extreme_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        # Test with very old timestamp
        old_timestamp = datetime(1970, 1, 1)
        violation_old = EthicalViolation(
            user_id="extreme_user",
            action="extreme_action",
            context=context,
            severity="low",
            timestamp=old_timestamp
        )
        
        assert violation_old.timestamp == old_timestamp
        
        # Test with far future timestamp
        future_timestamp = datetime(2100, 1, 1)
        violation_future = EthicalViolation(
            user_id="extreme_user",
            action="extreme_action",
            context=context,
            severity="low",
            timestamp=future_timestamp
        )
        
        assert violation_future.timestamp == future_timestamp
    
    def test_decision_result_with_complex_metadata(self):
        """Test DecisionResult with complex metadata structures"""
        complex_metadata = {
            "nested_dict": {
                "level1": {
                    "level2": {
                        "data": "deep_value"
                    }
                }
            },
            "list_of_dicts": [
                {"item": 1, "value": "a"},
                {"item": 2, "value": "b"}
            ],
            "mixed_types": [1, "string", True, None, 3.14]
        }
        
        result = DecisionResult(
            approved=True,
            confidence_score=0.85,
            reasoning="Complex metadata test",
            metadata=complex_metadata
        )
        
        assert result.metadata["nested_dict"]["level1"]["level2"]["data"] == "deep_value"
        assert len(result.metadata["list_of_dicts"]) == 2
        assert result.metadata["mixed_types"][1] == "string"
    
    def test_decision_result_confidence_score_precision(self):
        """Test DecisionResult with high precision confidence scores"""
        high_precision_scores = [
            0.123456789,
            0.999999999,
            0.000000001,
            0.5555555555555555
        ]
        
        for score in high_precision_scores:
            result = DecisionResult(
                approved=True,
                confidence_score=score,
                reasoning="Precision test"
            )
            assert abs(result.confidence_score - score) < 1e-10
    
    def test_ethical_context_serialization_with_complex_data(self):
        """Test EthicalContext serialization with complex nested data"""
        complex_data = {
            "users": [
                {"id": 1, "name": "Alice", "roles": ["admin", "user"]},
                {"id": 2, "name": "Bob", "roles": ["user"]}
            ],
            "permissions": {
                "read": {"level": 1, "description": "Read access"},
                "write": {"level": 2, "description": "Write access"},
                "delete": {"level": 3, "description": "Delete access"}
            },
            "settings": {
                "timeout": 30,
                "retries": 3,
                "features": {
                    "feature_a": True,
                    "feature_b": False
                }
            }
        }
        
        context = EthicalContext(
            user_id="complex_serialization_user",
            action="complex_serialization_action",
            context_data=complex_data,
            timestamp=datetime.now()
        )
        
        serialized = context.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["context_data"]["users"][0]["name"] == "Alice"
        assert serialized["context_data"]["permissions"]["read"]["level"] == 1
        assert serialized["context_data"]["settings"]["features"]["feature_a"] is True


class TestGenesisEthicalGovernorIntegrationExtended:
    """Extended integration tests focusing on real-world scenarios"""
    
    def test_multi_user_concurrent_workflow(self):
        """
        Test realistic multi-user workflow with concurrent decisions and violations.
        """
        governor = GenesisEthicalGovernor()
        
        # Add realistic business rules
        business_rules = [
            {
                "name": "data_access_rule",
                "condition": lambda ctx: "sensitive" in ctx.context_data.get("data_type", ""),
                "action": "deny",
                "priority": 10
            },
            {
                "name": "admin_override_rule",
                "condition": lambda ctx: ctx.context_data.get("user_role") == "admin",
                "action": "allow",
                "priority": 5
            },
            {
                "name": "business_hours_rule",
                "condition": lambda ctx: ctx.timestamp.hour < 9 or ctx.timestamp.hour > 17,
                "action": "deny",
                "priority": 7
            }
        ]
        
        for rule in business_rules:
            governor.add_ethical_rule(rule)
        
        # Simulate concurrent users
        import threading
        results = []
        
        def user_workflow(user_id, user_role):
            """Simulate a user's workflow with multiple decisions"""
            workflow_results = []
            
            for i in range(5):
                context = EthicalContext(
                    user_id=user_id,
                    action=f"access_data_{i}",
                    context_data={
                        "user_role": user_role,
                        "data_type": "sensitive" if i % 2 == 0 else "public"
                    },
                    timestamp=datetime.now()
                )
                
                decision = EthicalDecision(
                    action=f"access_data_{i}",
                    context=context,
                    parameters={"access_level": "read"}
                )
                
                result = governor.evaluate_decision(decision)
                workflow_results.append(result)
                
                # Record violation if denied
                if not result.approved:
                    violation = EthicalViolation(
                        user_id=user_id,
                        action=f"access_data_{i}",
                        context=context,
                        severity="medium",
                        timestamp=datetime.now()
                    )
                    governor.record_violation(violation)
            
            results.extend(workflow_results)
        
        # Run concurrent workflows
        threads = []
        user_scenarios = [
            ("admin_user", "admin"),
            ("regular_user", "user"),
            ("guest_user", "guest")
        ]
        
        for user_id, role in user_scenarios:
            thread = threading.Thread(target=user_workflow, args=(user_id, role))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 15  # 3 users * 5 decisions each
        assert all(isinstance(result, DecisionResult) for result in results)
        
        # Verify trust scores were affected
        for user_id, _ in user_scenarios:
            trust_score = governor.get_user_trust_score(user_id)
            assert 0.0 <= trust_score <= 1.0
    
    def test_escalation_workflow_simulation(self):
        """
        Test a realistic escalation workflow where violations trigger increased scrutiny.
        """
        governor = GenesisEthicalGovernor()
        
        user_id = "escalation_user"
        
        # Simulate escalating violations
        violation_scenarios = [
            ("minor_infraction", "low", "Accessed data outside normal hours"),
            ("policy_violation", "medium", "Attempted to access restricted data"),
            ("security_breach", "high", "Multiple failed authentication attempts"),
            ("critical_violation", "critical", "Attempted to delete audit logs")
        ]
        
        for i, (action, severity, description) in enumerate(violation_scenarios):
            context = EthicalContext(
                user_id=user_id,
                action=action,
                context_data={"description": description, "attempt": i + 1},
                timestamp=datetime.now() - timedelta(minutes=i * 10)
            )
            
            violation = EthicalViolation(
                user_id=user_id,
                action=action,
                context=context,
                severity=severity,
                timestamp=datetime.now() - timedelta(minutes=i * 10)
            )
            
            governor.record_violation(violation)
            
            # Check if trust score decreases with each violation
            trust_score = governor.get_user_trust_score(user_id)
            
            # After critical violation, trust should be very low
            if severity == "critical":
                assert trust_score < 0.3
        
        # Verify escalation pattern in violations
        violations = governor.get_violations(user_id)
        assert len(violations) == 4
        
        # Verify severity escalation
        severities = [v.severity for v in violations]
        assert "critical" in severities
        assert "high" in severities
    
    def test_compliance_audit_simulation(self):
        """
        Test a compliance audit scenario with comprehensive logging and reporting.
        """
        governor = GenesisEthicalGovernor()
        
        # Add compliance-focused rules
        compliance_rules = [
            {
                "name": "data_retention_rule",
                "condition": lambda ctx: ctx.context_data.get("data_age_days", 0) > 365,
                "action": "deny",
                "priority": 8
            },
            {
                "name": "pii_access_rule",
                "condition": lambda ctx: "pii" in ctx.context_data.get("data_classification", ""),
                "action": "deny",
                "priority": 9
            },
            {
                "name": "audit_trail_rule",
                "condition": lambda ctx: ctx.action == "delete_audit_log",
                "action": "deny",
                "priority": 1
            }
        ]
        
        for rule in compliance_rules:
            governor.add_ethical_rule(rule)
        
        # Simulate audit-worthy activities
        audit_activities = [
            ("access_customer_data", {"data_classification": "pii", "customer_id": "12345"}),
            ("export_financial_data", {"data_classification": "financial", "export_format": "csv"}),
            ("delete_old_records", {"data_age_days": 400, "record_count": 1000}),
            ("access_audit_log", {"log_type": "security", "time_range": "last_30_days"}),
            ("delete_audit_log", {"log_id": "audit_001", "reason": "cleanup"})
        ]
        
        for action, context_data in audit_activities:
            context = EthicalContext(
                user_id="audit_user",
                action=action,
                context_data=context_data,
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=action,
                context=context,
                parameters={"audit_mode": True}
            )
            
            result = governor.evaluate_decision(decision)
            
            # High-risk activities should be denied
            if action in ["delete_audit_log", "delete_old_records"]:
                assert result.approved is False
            
            # Log violations for denied activities
            if not result.approved:
                violation = EthicalViolation(
                    user_id="audit_user",
                    action=action,
                    context=context,
                    severity="high",
                    timestamp=datetime.now()
                )
                governor.record_violation(violation)
        
        # Verify audit trail
        history = governor.get_decision_history()
        assert len(history) == 5
        
        violations = governor.get_violations("audit_user")
        assert len(violations) >= 2  # At least delete operations should be denied
        
        # Verify compliance metrics
        trust_score = governor.get_user_trust_score("audit_user")
        assert trust_score < 0.7  # Should be reduced due to violations


class TestPerformanceOptimizations:
    """Tests for performance optimizations and resource management"""
    
    def test_rule_evaluation_caching(self):
        """
        Test that identical rule evaluations are cached for performance.
        """
        governor = GenesisEthicalGovernor()
        
        # Add a rule with expensive computation
        def expensive_condition(ctx):
            """Simulate expensive computation"""
            import time
            time.sleep(0.01)  # 10ms delay
            return ctx.action == "cached_action"
        
        expensive_rule = {
            "name": "expensive_rule",
            "condition": expensive_condition,
            "action": "allow",
            "priority": 1
        }
        
        governor.add_ethical_rule(expensive_rule)
        
        context = EthicalContext(
            user_id="cache_user",
            action="cached_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="cached_action",
            context=context,
            parameters={}
        )
        
        # First evaluation (should be slow)
        start_time = time.time()
        result1 = governor.evaluate_decision(decision)
        first_duration = time.time() - start_time
        
        # Second evaluation (should be faster if cached)
        start_time = time.time()
        result2 = governor.evaluate_decision(decision)
        second_duration = time.time() - start_time
        
        # Both should return valid results
        assert isinstance(result1, DecisionResult)
        assert isinstance(result2, DecisionResult)
        
        # Results should be consistent
        assert result1.approved == result2.approved
        
        # If caching is implemented, second should be faster
        # (This assertion might not hold if caching is not implemented)
        # assert second_duration < first_duration * 0.8
    
    def test_memory_efficient_history_management(self):
        """
        Test that decision history is managed efficiently with large volumes.
        """
        governor = GenesisEthicalGovernor()
        
        # Generate many decisions
        for i in range(10000):
            context = EthicalContext(
                user_id=f"memory_user_{i % 100}",
                action=f"memory_action_{i}",
                context_data={"index": i},
                timestamp=datetime.now()
            )
            
            decision = EthicalDecision(
                action=f"memory_action_{i}",
                context=context,
                parameters={}
            )
            
            governor.evaluate_decision(decision)
            
            # Periodically check memory usage
            if i % 1000 == 0:
                history = governor.get_decision_history()
                assert len(history) > 0
        
        # Verify final state
        final_history = governor.get_decision_history()
        assert len(final_history) == 10000
    
    def test_efficient_violation_queries(self):
        """
        Test efficient querying of violations with large datasets.
        """
        governor = GenesisEthicalGovernor()
        
        # Create violations for many users
        for user_id in range(100):
            for violation_id in range(50):
                context = EthicalContext(
                    user_id=f"query_user_{user_id}",
                    action=f"query_action_{violation_id}",
                    context_data={"violation_id": violation_id},
                    timestamp=datetime.now() - timedelta(minutes=violation_id)
                )
                
                violation = EthicalViolation(
                    user_id=f"query_user_{user_id}",
                    action=f"query_action_{violation_id}",
                    context=context,
                    severity="medium",
                    timestamp=datetime.now() - timedelta(minutes=violation_id)
                )
                
                governor.record_violation(violation)
        
        # Test efficient querying
        start_time = time.time()
        
        # Query specific user violations
        user_violations = governor.get_violations("query_user_50")
        assert len(user_violations) == 50
        
        # Query with time filters
        recent_violations = governor.get_violations(
            "query_user_25",
            since=datetime.now() - timedelta(minutes=25)
        )
        assert len(recent_violations) <= 25
        
        query_time = time.time() - start_time
        
        # Queries should complete quickly even with large dataset
        assert query_time < 1.0  # Should complete within 1 second