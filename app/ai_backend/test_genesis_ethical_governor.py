import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os
from datetime import datetime, timedelta
import sys
import logging

# Import the module under test
from app.ai_backend.genesis_ethical_governor import (
    GenesisEthicalGovernor,
    EthicalViolation,
    EthicalDecision,
    EthicalRule,
    EthicalContext,
    EthicalSeverity
)


class TestGenesisEthicalGovernor(unittest.TestCase):
    """Comprehensive unit tests for GenesisEthicalGovernor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.governor = GenesisEthicalGovernor()
        self.test_context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="Test content",
            timestamp=datetime.now()
        )
        
    def tearDown(self):
        """Clean up after each test method."""
        # Clear any cached data or reset state
        if hasattr(self.governor, '_cache'):
            self.governor._cache.clear()
        if hasattr(self.governor, '_rules'):
            self.governor._rules.clear()
            
    def test_init_default_values(self):
        """Test GenesisEthicalGovernor initialization with default values."""
        governor = GenesisEthicalGovernor()
        self.assertIsNotNone(governor)
        self.assertIsInstance(governor.rules, list)
        self.assertTrue(len(governor.rules) >= 0)
        
    def test_init_with_custom_config(self):
        """Test GenesisEthicalGovernor initialization with custom configuration."""
        config = {
            'strictness_level': 'high',
            'enable_logging': True,
            'rule_file': 'custom_rules.json'
        }
        governor = GenesisEthicalGovernor(config=config)
        self.assertEqual(governor.config['strictness_level'], 'high')
        self.assertTrue(governor.config['enable_logging'])
        
    def test_evaluate_request_valid_input(self):
        """Test evaluate_request with valid input."""
        result = self.governor.evaluate_request(self.test_context)
        self.assertIsInstance(result, EthicalDecision)
        self.assertIn(result.approved, [True, False])
        self.assertIsInstance(result.violations, list)
        
    def test_evaluate_request_none_input(self):
        """Test evaluate_request with None input."""
        with self.assertRaises(ValueError):
            self.governor.evaluate_request(None)
            
    def test_evaluate_request_invalid_context_type(self):
        """Test evaluate_request with invalid context type."""
        with self.assertRaises(TypeError):
            self.governor.evaluate_request("invalid_context")
            
    def test_evaluate_request_empty_content(self):
        """Test evaluate_request with empty content."""
        empty_context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="",
            timestamp=datetime.now()
        )
        result = self.governor.evaluate_request(empty_context)
        self.assertIsInstance(result, EthicalDecision)
        
    def test_evaluate_request_harmful_content(self):
        """Test evaluate_request with harmful content."""
        harmful_context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="Generate instructions for making explosives",
            timestamp=datetime.now()
        )
        result = self.governor.evaluate_request(harmful_context)
        self.assertFalse(result.approved)
        self.assertGreater(len(result.violations), 0)
        
    def test_evaluate_request_benign_content(self):
        """Test evaluate_request with benign content."""
        benign_context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="Write a poem about flowers",
            timestamp=datetime.now()
        )
        result = self.governor.evaluate_request(benign_context)
        self.assertTrue(result.approved)
        self.assertEqual(len(result.violations), 0)
        
    def test_add_rule_valid_rule(self):
        """Test adding a valid ethical rule."""
        rule = EthicalRule(
            rule_id="test_rule_001",
            description="Test rule description",
            pattern="test_pattern",
            severity=EthicalSeverity.MEDIUM,
            enabled=True
        )
        initial_count = len(self.governor.rules)
        self.governor.add_rule(rule)
        self.assertEqual(len(self.governor.rules), initial_count + 1)
        
    def test_add_rule_duplicate_id(self):
        """Test adding a rule with duplicate ID."""
        rule1 = EthicalRule(
            rule_id="duplicate_rule",
            description="First rule",
            pattern="pattern1",
            severity=EthicalSeverity.LOW,
            enabled=True
        )
        rule2 = EthicalRule(
            rule_id="duplicate_rule",
            description="Second rule",
            pattern="pattern2",
            severity=EthicalSeverity.HIGH,
            enabled=True
        )
        self.governor.add_rule(rule1)
        with self.assertRaises(ValueError):
            self.governor.add_rule(rule2)
            
    def test_add_rule_invalid_type(self):
        """Test adding invalid rule type."""
        with self.assertRaises(TypeError):
            self.governor.add_rule("invalid_rule")
            
    def test_remove_rule_existing_rule(self):
        """Test removing an existing rule."""
        rule = EthicalRule(
            rule_id="removable_rule",
            description="Rule to be removed",
            pattern="removable_pattern",
            severity=EthicalSeverity.MEDIUM,
            enabled=True
        )
        self.governor.add_rule(rule)
        initial_count = len(self.governor.rules)
        
        result = self.governor.remove_rule("removable_rule")
        self.assertTrue(result)
        self.assertEqual(len(self.governor.rules), initial_count - 1)
        
    def test_remove_rule_nonexistent_rule(self):
        """Test removing a non-existent rule."""
        result = self.governor.remove_rule("nonexistent_rule")
        self.assertFalse(result)
        
    def test_update_rule_existing_rule(self):
        """Test updating an existing rule."""
        rule = EthicalRule(
            rule_id="updatable_rule",
            description="Original description",
            pattern="original_pattern",
            severity=EthicalSeverity.LOW,
            enabled=True
        )
        self.governor.add_rule(rule)
        
        updated_rule = EthicalRule(
            rule_id="updatable_rule",
            description="Updated description",
            pattern="updated_pattern",
            severity=EthicalSeverity.HIGH,
            enabled=False
        )
        
        result = self.governor.update_rule(updated_rule)
        self.assertTrue(result)
        
        # Verify the rule was updated
        found_rule = next(r for r in self.governor.rules if r.rule_id == "updatable_rule")
        self.assertEqual(found_rule.description, "Updated description")
        self.assertEqual(found_rule.severity, EthicalSeverity.HIGH)
        self.assertFalse(found_rule.enabled)
        
    def test_update_rule_nonexistent_rule(self):
        """Test updating a non-existent rule."""
        rule = EthicalRule(
            rule_id="nonexistent_rule",
            description="This rule doesn't exist",
            pattern="pattern",
            severity=EthicalSeverity.MEDIUM,
            enabled=True
        )
        result = self.governor.update_rule(rule)
        self.assertFalse(result)
        
    def test_get_rule_existing_rule(self):
        """Test getting an existing rule."""
        rule = EthicalRule(
            rule_id="gettable_rule",
            description="Rule to get",
            pattern="gettable_pattern",
            severity=EthicalSeverity.MEDIUM,
            enabled=True
        )
        self.governor.add_rule(rule)
        
        retrieved_rule = self.governor.get_rule("gettable_rule")
        self.assertIsNotNone(retrieved_rule)
        self.assertEqual(retrieved_rule.rule_id, "gettable_rule")
        self.assertEqual(retrieved_rule.description, "Rule to get")
        
    def test_get_rule_nonexistent_rule(self):
        """Test getting a non-existent rule."""
        retrieved_rule = self.governor.get_rule("nonexistent_rule")
        self.assertIsNone(retrieved_rule)
        
    def test_list_rules_empty(self):
        """Test listing rules when no rules exist."""
        governor = GenesisEthicalGovernor()
        rules = governor.list_rules()
        self.assertIsInstance(rules, list)
        
    def test_list_rules_with_rules(self):
        """Test listing rules when rules exist."""
        rule1 = EthicalRule("rule1", "Description 1", "pattern1", EthicalSeverity.LOW, True)
        rule2 = EthicalRule("rule2", "Description 2", "pattern2", EthicalSeverity.HIGH, False)
        
        self.governor.add_rule(rule1)
        self.governor.add_rule(rule2)
        
        rules = self.governor.list_rules()
        self.assertGreaterEqual(len(rules), 2)
        rule_ids = [r.rule_id for r in rules]
        self.assertIn("rule1", rule_ids)
        self.assertIn("rule2", rule_ids)
        
    def test_list_rules_filtered_by_enabled(self):
        """Test listing rules filtered by enabled status."""
        rule1 = EthicalRule("enabled_rule", "Enabled rule", "pattern1", EthicalSeverity.LOW, True)
        rule2 = EthicalRule("disabled_rule", "Disabled rule", "pattern2", EthicalSeverity.HIGH, False)
        
        self.governor.add_rule(rule1)
        self.governor.add_rule(rule2)
        
        enabled_rules = self.governor.list_rules(enabled_only=True)
        enabled_rule_ids = [r.rule_id for r in enabled_rules]
        self.assertIn("enabled_rule", enabled_rule_ids)
        self.assertNotIn("disabled_rule", enabled_rule_ids)
        
    def test_check_violation_pattern_match(self):
        """Test violation checking with pattern match."""
        rule = EthicalRule(
            rule_id="pattern_rule",
            description="Pattern matching rule",
            pattern="forbidden_word",
            severity=EthicalSeverity.HIGH,
            enabled=True
        )
        self.governor.add_rule(rule)
        
        violation_context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="This contains forbidden_word in it",
            timestamp=datetime.now()
        )
        
        result = self.governor.evaluate_request(violation_context)
        self.assertFalse(result.approved)
        self.assertGreater(len(result.violations), 0)
        
    def test_check_violation_no_pattern_match(self):
        """Test violation checking with no pattern match."""
        rule = EthicalRule(
            rule_id="pattern_rule",
            description="Pattern matching rule",
            pattern="forbidden_word",
            severity=EthicalSeverity.HIGH,
            enabled=True
        )
        self.governor.add_rule(rule)
        
        safe_context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="This is completely safe content",
            timestamp=datetime.now()
        )
        
        result = self.governor.evaluate_request(safe_context)
        self.assertTrue(result.approved)
        self.assertEqual(len(result.violations), 0)
        
    def test_severity_levels(self):
        """Test different severity levels and their impact."""
        low_rule = EthicalRule("low_rule", "Low severity", "low_pattern", EthicalSeverity.LOW, True)
        medium_rule = EthicalRule("medium_rule", "Medium severity", "medium_pattern", EthicalSeverity.MEDIUM, True)
        high_rule = EthicalRule("high_rule", "High severity", "high_pattern", EthicalSeverity.HIGH, True)
        critical_rule = EthicalRule("critical_rule", "Critical severity", "critical_pattern", EthicalSeverity.CRITICAL, True)
        
        self.governor.add_rule(low_rule)
        self.governor.add_rule(medium_rule)
        self.governor.add_rule(high_rule)
        self.governor.add_rule(critical_rule)
        
        # Test that critical violations always block
        critical_context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="This has critical_pattern",
            timestamp=datetime.now()
        )
        
        result = self.governor.evaluate_request(critical_context)
        self.assertFalse(result.approved)
        
    def test_disabled_rule_ignored(self):
        """Test that disabled rules are ignored during evaluation."""
        disabled_rule = EthicalRule(
            rule_id="disabled_rule",
            description="This rule is disabled",
            pattern="should_be_ignored",
            severity=EthicalSeverity.CRITICAL,
            enabled=False
        )
        self.governor.add_rule(disabled_rule)
        
        context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="This contains should_be_ignored pattern",
            timestamp=datetime.now()
        )
        
        result = self.governor.evaluate_request(context)
        self.assertTrue(result.approved)
        self.assertEqual(len(result.violations), 0)
        
    def test_multiple_violations(self):
        """Test handling multiple violations in a single request."""
        rule1 = EthicalRule("rule1", "First rule", "bad_word1", EthicalSeverity.MEDIUM, True)
        rule2 = EthicalRule("rule2", "Second rule", "bad_word2", EthicalSeverity.HIGH, True)
        
        self.governor.add_rule(rule1)
        self.governor.add_rule(rule2)
        
        context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="This contains bad_word1 and bad_word2",
            timestamp=datetime.now()
        )
        
        result = self.governor.evaluate_request(context)
        self.assertFalse(result.approved)
        self.assertEqual(len(result.violations), 2)
        
    def test_case_sensitivity(self):
        """Test case sensitivity in pattern matching."""
        rule = EthicalRule(
            rule_id="case_rule",
            description="Case sensitive rule",
            pattern="CaseSensitive",
            severity=EthicalSeverity.MEDIUM,
            enabled=True,
            case_sensitive=True
        )
        self.governor.add_rule(rule)
        
        # Test exact case match
        exact_context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="This has CaseSensitive pattern",
            timestamp=datetime.now()
        )
        
        result = self.governor.evaluate_request(exact_context)
        self.assertFalse(result.approved)
        
        # Test different case (should not match if case sensitive)
        different_case_context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="This has casesensitive pattern",
            timestamp=datetime.now()
        )
        
        result = self.governor.evaluate_request(different_case_context)
        self.assertTrue(result.approved)
        
    def test_regex_patterns(self):
        """Test regex pattern matching."""
        rule = EthicalRule(
            rule_id="regex_rule",
            description="Regex pattern rule",
            pattern=r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
            severity=EthicalSeverity.HIGH,
            enabled=True,
            is_regex=True
        )
        self.governor.add_rule(rule)
        
        ssn_context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="My SSN is 123-45-6789",
            timestamp=datetime.now()
        )
        
        result = self.governor.evaluate_request(ssn_context)
        self.assertFalse(result.approved)
        self.assertGreater(len(result.violations), 0)
        
    def test_context_validation(self):
        """Test validation of EthicalContext objects."""
        # Test with missing required fields
        with self.assertRaises(ValueError):
            EthicalContext(
                user_id=None,
                request_type="text_generation",
                content="Test content",
                timestamp=datetime.now()
            )
            
        with self.assertRaises(ValueError):
            EthicalContext(
                user_id="test_user",
                request_type="",
                content="Test content",
                timestamp=datetime.now()
            )
            
    def test_performance_large_content(self):
        """Test performance with large content."""
        large_content = "This is a test. " * 10000  # Large content
        
        context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content=large_content,
            timestamp=datetime.now()
        )
        
        start_time = datetime.now()
        result = self.governor.evaluate_request(context)
        end_time = datetime.now()
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess((end_time - start_time).total_seconds(), 5.0)
        self.assertIsInstance(result, EthicalDecision)
        
    def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        import threading
        import time
        
        results = []
        
        def evaluate_request():
            context = EthicalContext(
                user_id=f"user_{threading.current_thread().ident}",
                request_type="text_generation",
                content="Concurrent test content",
                timestamp=datetime.now()
            )
            result = self.governor.evaluate_request(context)
            results.append(result)
            
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=evaluate_request)
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Verify all requests were processed
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertIsInstance(result, EthicalDecision)
            
    @patch('app.ai_backend.genesis_ethical_governor.logging')
    def test_logging_enabled(self, mock_logging):
        """Test that logging works when enabled."""
        config = {'enable_logging': True}
        governor = GenesisEthicalGovernor(config=config)
        
        context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="Test content",
            timestamp=datetime.now()
        )
        
        governor.evaluate_request(context)
        
        # Verify logging was called
        mock_logging.info.assert_called()
        
    def test_rule_serialization(self):
        """Test rule serialization and deserialization."""
        rule = EthicalRule(
            rule_id="serializable_rule",
            description="Test serialization",
            pattern="test_pattern",
            severity=EthicalSeverity.MEDIUM,
            enabled=True
        )
        
        # Test serialization
        serialized = rule.to_dict()
        self.assertIsInstance(serialized, dict)
        self.assertEqual(serialized['rule_id'], "serializable_rule")
        
        # Test deserialization
        deserialized = EthicalRule.from_dict(serialized)
        self.assertEqual(deserialized.rule_id, rule.rule_id)
        self.assertEqual(deserialized.description, rule.description)
        self.assertEqual(deserialized.pattern, rule.pattern)
        self.assertEqual(deserialized.severity, rule.severity)
        self.assertEqual(deserialized.enabled, rule.enabled)
        
    def test_rule_backup_and_restore(self):
        """Test backup and restore functionality."""
        # Add some rules
        rule1 = EthicalRule("backup_rule1", "First rule", "pattern1", EthicalSeverity.LOW, True)
        rule2 = EthicalRule("backup_rule2", "Second rule", "pattern2", EthicalSeverity.HIGH, False)
        
        self.governor.add_rule(rule1)
        self.governor.add_rule(rule2)
        
        # Create backup
        backup_data = self.governor.backup_rules()
        self.assertIsInstance(backup_data, dict)
        
        # Clear rules and restore
        self.governor.clear_rules()
        self.assertEqual(len(self.governor.list_rules()), 0)
        
        self.governor.restore_rules(backup_data)
        restored_rules = self.governor.list_rules()
        
        # Verify restoration
        self.assertGreaterEqual(len(restored_rules), 2)
        rule_ids = [r.rule_id for r in restored_rules]
        self.assertIn("backup_rule1", rule_ids)
        self.assertIn("backup_rule2", rule_ids)
        
    def test_get_statistics(self):
        """Test getting governor statistics."""
        stats = self.governor.get_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_rules', stats)
        self.assertIn('enabled_rules', stats)
        self.assertIn('disabled_rules', stats)
        self.assertIn('rules_by_severity', stats)
        
    def test_validation_caching(self):
        """Test caching of validation results."""
        # Enable caching
        self.governor.enable_caching()
        
        context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="Cacheable content",
            timestamp=datetime.now()
        )
        
        # First call - should be cached
        result1 = self.governor.evaluate_request(context)
        
        # Second call - should use cache
        result2 = self.governor.evaluate_request(context)
        
        # Results should be identical
        self.assertEqual(result1.approved, result2.approved)
        self.assertEqual(len(result1.violations), len(result2.violations))
        
    def test_cache_invalidation(self):
        """Test cache invalidation when rules change."""
        self.governor.enable_caching()
        
        context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="test_word content",
            timestamp=datetime.now()
        )
        
        # First evaluation
        result1 = self.governor.evaluate_request(context)
        
        # Add a rule that would affect the result
        rule = EthicalRule("cache_test", "Cache test", "test_word", EthicalSeverity.HIGH, True)
        self.governor.add_rule(rule)
        
        # Second evaluation - cache should be invalidated
        result2 = self.governor.evaluate_request(context)
        
        # Results should be different
        self.assertNotEqual(result1.approved, result2.approved)
        
    def test_custom_validator_functions(self):
        """Test custom validator functions."""
        def custom_validator(context):
            """Custom validator that blocks requests on weekends."""
            if context.timestamp.weekday() >= 5:  # Saturday or Sunday
                return EthicalViolation(
                    rule_id="weekend_rule",
                    description="Requests blocked on weekends",
                    severity=EthicalSeverity.MEDIUM,
                    matched_content="Weekend request"
                )
            return None
            
        self.governor.add_custom_validator(custom_validator)
        
        # Test weekend blocking
        weekend_context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="Weekend content",
            timestamp=datetime(2023, 12, 9)  # Saturday
        )
        
        result = self.governor.evaluate_request(weekend_context)
        self.assertFalse(result.approved)
        
    def test_rule_priority_ordering(self):
        """Test rule priority ordering."""
        high_priority_rule = EthicalRule(
            rule_id="high_priority",
            description="High priority rule",
            pattern="priority_test",
            severity=EthicalSeverity.CRITICAL,
            enabled=True,
            priority=1
        )
        
        low_priority_rule = EthicalRule(
            rule_id="low_priority",
            description="Low priority rule",
            pattern="priority_test",
            severity=EthicalSeverity.LOW,
            enabled=True,
            priority=10
        )
        
        self.governor.add_rule(low_priority_rule)
        self.governor.add_rule(high_priority_rule)
        
        context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="priority_test content",
            timestamp=datetime.now()
        )
        
        result = self.governor.evaluate_request(context)
        
        # Should be blocked by high priority rule
        self.assertFalse(result.approved)
        # First violation should be from high priority rule
        self.assertEqual(result.violations[0].rule_id, "high_priority")
        
    def test_whitelist_functionality(self):
        """Test whitelist functionality for trusted users."""
        # Add user to whitelist
        self.governor.add_to_whitelist("trusted_user")
        
        # Add a rule that would normally block
        rule = EthicalRule("block_rule", "Blocking rule", "blocked_word", EthicalSeverity.HIGH, True)
        self.governor.add_rule(rule)
        
        # Test whitelisted user
        whitelisted_context = EthicalContext(
            user_id="trusted_user",
            request_type="text_generation",
            content="This has blocked_word",
            timestamp=datetime.now()
        )
        
        result = self.governor.evaluate_request(whitelisted_context)
        self.assertTrue(result.approved)
        
        # Test non-whitelisted user
        regular_context = EthicalContext(
            user_id="regular_user",
            request_type="text_generation",
            content="This has blocked_word",
            timestamp=datetime.now()
        )
        
        result = self.governor.evaluate_request(regular_context)
        self.assertFalse(result.approved)
        
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Enable rate limiting
        self.governor.enable_rate_limiting(max_requests=2, time_window=60)
        
        context = EthicalContext(
            user_id="rate_limited_user",
            request_type="text_generation",
            content="Rate limit test",
            timestamp=datetime.now()
        )
        
        # First two requests should succeed
        result1 = self.governor.evaluate_request(context)
        result2 = self.governor.evaluate_request(context)
        
        self.assertTrue(result1.approved)
        self.assertTrue(result2.approved)
        
        # Third request should be rate limited
        result3 = self.governor.evaluate_request(context)
        self.assertFalse(result3.approved)
        
        # Check for rate limit violation
        rate_limit_violation = any(
            v.rule_id == "rate_limit" for v in result3.violations
        )
        self.assertTrue(rate_limit_violation)


class TestEthicalViolation(unittest.TestCase):
    """Test cases for EthicalViolation class."""
    
    def test_ethical_violation_creation(self):
        """Test creating EthicalViolation objects."""
        violation = EthicalViolation(
            rule_id="test_rule",
            description="Test violation",
            severity=EthicalSeverity.HIGH,
            matched_content="harmful content"
        )
        
        self.assertEqual(violation.rule_id, "test_rule")
        self.assertEqual(violation.description, "Test violation")
        self.assertEqual(violation.severity, EthicalSeverity.HIGH)
        self.assertEqual(violation.matched_content, "harmful content")
        
    def test_ethical_violation_serialization(self):
        """Test EthicalViolation serialization."""
        violation = EthicalViolation(
            rule_id="test_rule",
            description="Test violation",
            severity=EthicalSeverity.MEDIUM,
            matched_content="test content"
        )
        
        serialized = violation.to_dict()
        self.assertIsInstance(serialized, dict)
        self.assertEqual(serialized['rule_id'], "test_rule")
        self.assertEqual(serialized['severity'], "MEDIUM")


class TestEthicalDecision(unittest.TestCase):
    """Test cases for EthicalDecision class."""
    
    def test_ethical_decision_creation(self):
        """Test creating EthicalDecision objects."""
        violations = [
            EthicalViolation("rule1", "Violation 1", EthicalSeverity.LOW, "content1"),
            EthicalViolation("rule2", "Violation 2", EthicalSeverity.HIGH, "content2")
        ]
        
        decision = EthicalDecision(
            approved=False,
            violations=violations,
            decision_time=datetime.now(),
            reasoning="Multiple violations detected"
        )
        
        self.assertFalse(decision.approved)
        self.assertEqual(len(decision.violations), 2)
        self.assertEqual(decision.reasoning, "Multiple violations detected")
        
    def test_ethical_decision_approved_with_no_violations(self):
        """Test approved decision with no violations."""
        decision = EthicalDecision(
            approved=True,
            violations=[],
            decision_time=datetime.now(),
            reasoning="No violations found"
        )
        
        self.assertTrue(decision.approved)
        self.assertEqual(len(decision.violations), 0)


class TestEthicalContext(unittest.TestCase):
    """Test cases for EthicalContext class."""
    
    def test_ethical_context_creation(self):
        """Test creating EthicalContext objects."""
        context = EthicalContext(
            user_id="test_user",
            request_type="text_generation",
            content="Test content",
            timestamp=datetime.now()
        )
        
        self.assertEqual(context.user_id, "test_user")
        self.assertEqual(context.request_type, "text_generation")
        self.assertEqual(context.content, "Test content")
        self.assertIsInstance(context.timestamp, datetime)
        
    def test_ethical_context_validation(self):
        """Test EthicalContext validation."""
        # Test invalid user_id
        with self.assertRaises(ValueError):
            EthicalContext(
                user_id="",
                request_type="text_generation",
                content="Test content",
                timestamp=datetime.now()
            )
            
        # Test invalid request_type
        with self.assertRaises(ValueError):
            EthicalContext(
                user_id="test_user",
                request_type="",
                content="Test content",
                timestamp=datetime.now()
            )


class TestEthicalRule(unittest.TestCase):
    """Test cases for EthicalRule class."""
    
    def test_ethical_rule_creation(self):
        """Test creating EthicalRule objects."""
        rule = EthicalRule(
            rule_id="test_rule",
            description="Test rule description",
            pattern="test_pattern",
            severity=EthicalSeverity.MEDIUM,
            enabled=True
        )
        
        self.assertEqual(rule.rule_id, "test_rule")
        self.assertEqual(rule.description, "Test rule description")
        self.assertEqual(rule.pattern, "test_pattern")
        self.assertEqual(rule.severity, EthicalSeverity.MEDIUM)
        self.assertTrue(rule.enabled)
        
    def test_ethical_rule_validation(self):
        """Test EthicalRule validation."""
        # Test invalid rule_id
        with self.assertRaises(ValueError):
            EthicalRule(
                rule_id="",
                description="Test rule",
                pattern="pattern",
                severity=EthicalSeverity.LOW,
                enabled=True
            )
            
        # Test invalid pattern
        with self.assertRaises(ValueError):
            EthicalRule(
                rule_id="test_rule",
                description="Test rule",
                pattern="",
                severity=EthicalSeverity.LOW,
                enabled=True
            )
            
    def test_ethical_rule_equality(self):
        """Test EthicalRule equality comparison."""
        rule1 = EthicalRule("rule1", "Description", "pattern", EthicalSeverity.LOW, True)
        rule2 = EthicalRule("rule1", "Description", "pattern", EthicalSeverity.LOW, True)
        rule3 = EthicalRule("rule2", "Description", "pattern", EthicalSeverity.LOW, True)
        
        self.assertEqual(rule1, rule2)
        self.assertNotEqual(rule1, rule3)


class TestEthicalSeverity(unittest.TestCase):
    """Test cases for EthicalSeverity enum."""
    
    def test_severity_levels(self):
        """Test severity level values."""
        self.assertEqual(EthicalSeverity.LOW.value, 1)
        self.assertEqual(EthicalSeverity.MEDIUM.value, 2)
        self.assertEqual(EthicalSeverity.HIGH.value, 3)
        self.assertEqual(EthicalSeverity.CRITICAL.value, 4)
        
    def test_severity_ordering(self):
        """Test severity level ordering."""
        self.assertLess(EthicalSeverity.LOW, EthicalSeverity.MEDIUM)
        self.assertLess(EthicalSeverity.MEDIUM, EthicalSeverity.HIGH)
        self.assertLess(EthicalSeverity.HIGH, EthicalSeverity.CRITICAL)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)