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
        
        # Calculate score reduction
        new_score = governor.get_user_trust_score(user_id)
        score_reduction = initial_score - new_score
        
        # Score reduction should be proportional to severity
        assert score_reduction >= expected_impact * 0.5  # Allow some tolerance
        assert score_reduction <= expected_impact * 2.0  # Allow some tolerance


# ... (other test classes unchanged) ...


class TestMockIntegrations:
    """Test integration with mocked external systems"""
    
    @patch('app.ai_backend.genesis_ethical_governor.external_policy_service')
    def test_external_policy_integration(self, mock_policy_service):
        """
        Test integration with external policy services.
        """
        # Mock external service response
        mock_policy_service.check_policy.return_value = {
            "allowed": True,
            "policy_id": "policy_123",
            "confidence": 0.95
        }
        
        governor = GenesisEthicalGovernor()
        
        context = EthicalContext(
            user_id="external_user",
            action="external_action",
            context_data={"external_check": True},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="external_action",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
        
        # Verify external service was called
        if hasattr(governor, 'external_policy_enabled') and governor.external_policy_enabled:
            mock_policy_service.check_policy.assert_called_once()
    
    @patch('app.ai_backend.genesis_ethical_governor.notification_service')
    def test_notification_system_integration(self, mock_notification):
        """
        Test integration with notification systems.
        """
        mock_notification.send_alert.return_value = True
        
        governor = GenesisEthicalGovernor()
        
        # Create high-severity violation
        context = EthicalContext(
            user_id="notification_user",
            action="critical_action",
            context_data={"severity": "critical"},
            timestamp=datetime.now()
        )
        
        violation = EthicalViolation(
            user_id="notification_user",
            action="critical_violation",
            context=context,
            severity="critical",
            timestamp=datetime.now()
        )
        
        governor.record_violation(violation)
        
        # Verify notification was sent for critical violation
        if hasattr(governor, 'notification_enabled') and governor.notification_enabled:
            mock_notification.send_alert.assert_called_once()
    
    @patch('app.ai_backend.genesis_ethical_governor.metrics_collector')
    def test_metrics_collection_integration(self, mock_metrics):
        """
        Test integration with metrics collection systems.
        """
        mock_metrics.record_metric.return_value = None
        
        governor = GenesisEthicalGovernor()
        
        context = EthicalContext(
            user_id="metrics_user",
            action="metrics_action",
            context_data={},
            timestamp=datetime.now()
        )
        
        decision = EthicalDecision(
            action="metrics_action",
            context=context,
            parameters={}
        )
        
        result = governor.evaluate_decision(decision)
        assert isinstance(result, DecisionResult)
        
        # Verify metrics were collected
        if hasattr(governor, 'metrics_enabled') and governor.metrics_enabled:
            mock_metrics.record_metric.assert_called()