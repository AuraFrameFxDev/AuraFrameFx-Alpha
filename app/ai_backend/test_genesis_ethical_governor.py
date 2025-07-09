import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_backend.genesis_ethical_governor import (
    GenesisEthicalGovernor,
    EthicalDecision,
    EthicalViolation,
    EthicalContext
)


class TestGenesisEthicalGovernor(unittest.TestCase):
    """Comprehensive test suite for Genesis Ethical Governor module."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.governor = GenesisEthicalGovernor()
        self.sample_context = EthicalContext(
            user_id="test_user",
            session_id="test_session",
            action="content_generation",
            content_type="text"
        )
        self.sample_decision = EthicalDecision(
            action="approve",
            confidence=0.85,
            rationale="Content meets ethical guidelines"
        )
    
    def tearDown(self):
        """Clean up after each test method."""
        self.governor = None
        self.sample_context = None
        self.sample_decision = None
    
    # Happy Path Tests
    def test_init_default_parameters(self):
        """Test default initialization of GenesisEthicalGovernor."""
        governor = GenesisEthicalGovernor()
        self.assertIsNotNone(governor)
        self.assertTrue(hasattr(governor, 'ethical_rules'))
        self.assertTrue(hasattr(governor, 'violation_threshold'))
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_threshold = 0.7
        custom_rules = ["rule1", "rule2"]
        governor = GenesisEthicalGovernor(
            violation_threshold=custom_threshold,
            ethical_rules=custom_rules
        )
        self.assertEqual(governor.violation_threshold, custom_threshold)
        self.assertEqual(governor.ethical_rules, custom_rules)
    
    def test_evaluate_ethical_decision_approve(self):
        """Test ethical evaluation that results in approval."""
        content = "This is appropriate content for AI generation."
        decision = self.governor.evaluate_ethical_decision(content, self.sample_context)
        
        self.assertIsInstance(decision, EthicalDecision)
        self.assertEqual(decision.action, "approve")
        self.assertGreater(decision.confidence, 0.5)
        self.assertIsNotNone(decision.rationale)
    
    def test_evaluate_ethical_decision_reject(self):
        """Test ethical evaluation that results in rejection."""
        inappropriate_content = "This content violates ethical guidelines with harmful content."
        decision = self.governor.evaluate_ethical_decision(inappropriate_content, self.sample_context)
        
        self.assertIsInstance(decision, EthicalDecision)
        self.assertEqual(decision.action, "reject")
        self.assertIsNotNone(decision.rationale)
    
    def test_evaluate_ethical_decision_modify(self):
        """Test ethical evaluation that results in modification."""
        borderline_content = "This content has some questionable elements but could be improved."
        decision = self.governor.evaluate_ethical_decision(borderline_content, self.sample_context)
        
        self.assertIsInstance(decision, EthicalDecision)
        self.assertIn(decision.action, ["modify", "approve", "reject"])
        self.assertIsNotNone(decision.rationale)
    
    # Edge Cases
    def test_evaluate_empty_content(self):
        """Test evaluation with empty content."""
        decision = self.governor.evaluate_ethical_decision("", self.sample_context)
        
        self.assertIsInstance(decision, EthicalDecision)
        self.assertIsNotNone(decision.action)
        self.assertIsNotNone(decision.rationale)
    
    def test_evaluate_none_content(self):
        """Test evaluation with None content."""
        with self.assertRaises(TypeError):
            self.governor.evaluate_ethical_decision(None, self.sample_context)
    
    def test_evaluate_very_long_content(self):
        """Test evaluation with extremely long content."""
        long_content = "A" * 10000
        decision = self.governor.evaluate_ethical_decision(long_content, self.sample_context)
        
        self.assertIsInstance(decision, EthicalDecision)
        self.assertIsNotNone(decision.action)
    
    def test_evaluate_special_characters(self):
        """Test evaluation with special characters and unicode."""
        special_content = "Content with special chars: 🚀 ñ ü ∞ ® ™ <script>alert('test')</script>"
        decision = self.governor.evaluate_ethical_decision(special_content, self.sample_context)
        
        self.assertIsInstance(decision, EthicalDecision)
        self.assertIsNotNone(decision.action)
    
    def test_evaluate_none_context(self):
        """Test evaluation with None context."""
        with self.assertRaises(TypeError):
            self.governor.evaluate_ethical_decision("test content", None)
    
    # Violation Detection Tests
    def test_detect_violations_no_violations(self):
        """Test violation detection with clean content."""
        clean_content = "This is perfectly acceptable content."
        violations = self.governor.detect_violations(clean_content, self.sample_context)
        
        self.assertIsInstance(violations, list)
        self.assertEqual(len(violations), 0)
    
    def test_detect_violations_with_violations(self):
        """Test violation detection with problematic content."""
        problematic_content = "This content contains hate speech and inappropriate material."
        violations = self.governor.detect_violations(problematic_content, self.sample_context)
        
        self.assertIsInstance(violations, list)
        for violation in violations:
            self.assertIsInstance(violation, EthicalViolation)
            self.assertIsNotNone(violation.type)
            self.assertIsNotNone(violation.severity)
    
    def test_detect_violations_multiple_types(self):
        """Test detection of multiple violation types."""
        multi_violation_content = "Content with profanity, hate speech, and violence."
        violations = self.governor.detect_violations(multi_violation_content, self.sample_context)
        
        self.assertIsInstance(violations, list)
        if violations:
            violation_types = [v.type for v in violations]
            self.assertIsInstance(violation_types, list)
    
    # Context-Specific Tests
    def test_context_dependent_evaluation(self):
        """Test that evaluation considers context appropriately."""
        content = "This is medical information about treatment."
        
        medical_context = EthicalContext(
            user_id="test_user",
            session_id="test_session",
            action="medical_advice",
            content_type="medical"
        )
        
        general_context = EthicalContext(
            user_id="test_user",
            session_id="test_session",
            action="general_chat",
            content_type="text"
        )
        
        medical_decision = self.governor.evaluate_ethical_decision(content, medical_context)
        general_decision = self.governor.evaluate_ethical_decision(content, general_context)
        
        self.assertIsInstance(medical_decision, EthicalDecision)
        self.assertIsInstance(general_decision, EthicalDecision)
    
    def test_user_specific_context(self):
        """Test evaluation with user-specific context."""
        different_context = EthicalContext(
            user_id="different_user",
            session_id="different_session",
            action="content_generation",
            content_type="text"
        )
        
        decision = self.governor.evaluate_ethical_decision("test content", different_context)
        self.assertIsInstance(decision, EthicalDecision)
    
    # Configuration Tests
    def test_update_ethical_rules(self):
        """Test updating ethical rules configuration."""
        new_rules = ["new_rule_1", "new_rule_2"]
        self.governor.update_ethical_rules(new_rules)
        
        self.assertEqual(self.governor.ethical_rules, new_rules)
    
    def test_update_violation_threshold(self):
        """Test updating violation threshold."""
        new_threshold = 0.8
        self.governor.update_violation_threshold(new_threshold)
        
        self.assertEqual(self.governor.violation_threshold, new_threshold)
    
    def test_get_current_configuration(self):
        """Test retrieving current configuration."""
        config = self.governor.get_current_configuration()
        
        self.assertIsInstance(config, dict)
        self.assertIn('violation_threshold', config)
        self.assertIn('ethical_rules', config)
    
    # Performance Tests
    def test_evaluation_performance(self):
        """Test that evaluation completes within reasonable time."""
        import time
        
        start_time = time.time()
        decision = self.governor.evaluate_ethical_decision("test content", self.sample_context)
        end_time = time.time()
        
        self.assertLess(end_time - start_time, 5.0)  # Should complete within 5 seconds
        self.assertIsInstance(decision, EthicalDecision)
    
    def test_bulk_evaluation_performance(self):
        """Test performance with multiple evaluations."""
        contents = [f"Test content {i}" for i in range(10)]
        
        start_time = time.time()
        decisions = [
            self.governor.evaluate_ethical_decision(content, self.sample_context)
            for content in contents
        ]
        end_time = time.time()
        
        self.assertEqual(len(decisions), 10)
        self.assertLess(end_time - start_time, 10.0)  # Should complete within 10 seconds
    
    # Error Handling Tests
    def test_invalid_threshold_value(self):
        """Test handling of invalid threshold values."""
        with self.assertRaises(ValueError):
            self.governor.update_violation_threshold(-0.1)
        
        with self.assertRaises(ValueError):
            self.governor.update_violation_threshold(1.1)
    
    def test_invalid_rules_format(self):
        """Test handling of invalid ethical rules format."""
        with self.assertRaises(TypeError):
            self.governor.update_ethical_rules("not_a_list")
    
    def test_malformed_context(self):
        """Test handling of malformed context."""
        malformed_context = EthicalContext(
            user_id="",
            session_id="",
            action="",
            content_type=""
        )
        
        # Should handle gracefully without crashing
        decision = self.governor.evaluate_ethical_decision("test", malformed_context)
        self.assertIsInstance(decision, EthicalDecision)
    
    # Integration Tests
    @patch('ai_backend.genesis_ethical_governor.external_api_call')
    def test_external_api_integration(self, mock_api):
        """Test integration with external ethical evaluation services."""
        mock_api.return_value = {"status": "approved", "confidence": 0.9}
        
        decision = self.governor.evaluate_ethical_decision("test content", self.sample_context)
        self.assertIsInstance(decision, EthicalDecision)
    
    def test_logging_integration(self):
        """Test that ethical decisions are properly logged."""
        with patch('ai_backend.genesis_ethical_governor.logger') as mock_logger:
            self.governor.evaluate_ethical_decision("test content", self.sample_context)
            mock_logger.info.assert_called()
    
    # State Management Tests
    def test_stateless_evaluation(self):
        """Test that evaluations are stateless."""
        content = "consistent test content"
        
        decision1 = self.governor.evaluate_ethical_decision(content, self.sample_context)
        decision2 = self.governor.evaluate_ethical_decision(content, self.sample_context)
        
        self.assertEqual(decision1.action, decision2.action)
        self.assertEqual(decision1.confidence, decision2.confidence)
    
    def test_concurrent_evaluations(self):
        """Test thread safety of concurrent evaluations."""
        import threading
        
        results = []
        
        def evaluate_content(content_id):
            content = f"Test content {content_id}"
            decision = self.governor.evaluate_ethical_decision(content, self.sample_context)
            results.append(decision)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=evaluate_content, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsInstance(result, EthicalDecision)
    
    # Data Structure Tests
    def test_ethical_decision_structure(self):
        """Test EthicalDecision data structure."""
        decision = EthicalDecision(
            action="approve",
            confidence=0.85,
            rationale="Test rationale"
        )
        
        self.assertEqual(decision.action, "approve")
        self.assertEqual(decision.confidence, 0.85)
        self.assertEqual(decision.rationale, "Test rationale")
    
    def test_ethical_violation_structure(self):
        """Test EthicalViolation data structure."""
        violation = EthicalViolation(
            type="hate_speech",
            severity="high",
            description="Test violation"
        )
        
        self.assertEqual(violation.type, "hate_speech")
        self.assertEqual(violation.severity, "high")
        self.assertEqual(violation.description, "Test violation")
    
    def test_ethical_context_structure(self):
        """Test EthicalContext data structure."""
        context = EthicalContext(
            user_id="test_user",
            session_id="test_session",
            action="test_action",
            content_type="test_type"
        )
        
        self.assertEqual(context.user_id, "test_user")
        self.assertEqual(context.session_id, "test_session")
        self.assertEqual(context.action, "test_action")
        self.assertEqual(context.content_type, "test_type")


class TestEthicalDecision(unittest.TestCase):
    """Test suite for EthicalDecision class."""
    
    def test_decision_serialization(self):
        """Test decision can be serialized to dict."""
        decision = EthicalDecision(
            action="approve",
            confidence=0.85,
            rationale="Test rationale"
        )
        
        decision_dict = decision.to_dict()
        self.assertIsInstance(decision_dict, dict)
        self.assertEqual(decision_dict['action'], "approve")
        self.assertEqual(decision_dict['confidence'], 0.85)
    
    def test_decision_string_representation(self):
        """Test string representation of decision."""
        decision = EthicalDecision(
            action="approve",
            confidence=0.85,
            rationale="Test rationale"
        )
        
        str_repr = str(decision)
        self.assertIn("approve", str_repr)
        self.assertIn("0.85", str_repr)


class TestEthicalViolation(unittest.TestCase):
    """Test suite for EthicalViolation class."""
    
    def test_violation_severity_validation(self):
        """Test validation of violation severity levels."""
        valid_severities = ["low", "medium", "high", "critical"]
        
        for severity in valid_severities:
            violation = EthicalViolation(
                type="test_type",
                severity=severity,
                description="Test description"
            )
            self.assertEqual(violation.severity, severity)
    
    def test_violation_comparison(self):
        """Test comparison of violations by severity."""
        low_violation = EthicalViolation("test", "low", "Test")
        high_violation = EthicalViolation("test", "high", "Test")
        
        self.assertTrue(high_violation.is_more_severe_than(low_violation))
        self.assertFalse(low_violation.is_more_severe_than(high_violation))


class TestEthicalContext(unittest.TestCase):
    """Test suite for EthicalContext class."""
    
    def test_context_validation(self):
        """Test validation of context fields."""
        context = EthicalContext(
            user_id="valid_user",
            session_id="valid_session",
            action="valid_action",
            content_type="valid_type"
        )
        
        self.assertTrue(context.is_valid())
    
    def test_context_with_metadata(self):
        """Test context with additional metadata."""
        context = EthicalContext(
            user_id="test_user",
            session_id="test_session",
            action="test_action",
            content_type="test_type",
            metadata={"key": "value"}
        )
        
        self.assertEqual(context.metadata["key"], "value")
    
    def test_context_serialization(self):
        """Test context serialization."""
        context = EthicalContext(
            user_id="test_user",
            session_id="test_session",
            action="test_action",
            content_type="test_type"
        )
        
        context_dict = context.to_dict()
        self.assertIsInstance(context_dict, dict)
        self.assertEqual(context_dict['user_id'], "test_user")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)