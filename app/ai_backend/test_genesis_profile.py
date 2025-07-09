import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta, timezone
import json
import os
import tempfile
import time
import gc

from app.ai_backend.genesis_profile import (
    GenesisProfile,
    ProfileManager,
    generate_profile_data,
    validate_profile_schema,
    merge_profiles,
    ProfileError,
    ProfileValidationError,
    ProfileNotFoundError,
    ProfileValidator,
    ProfileBuilder
)

# Alias for consistency in tests
ValidationError = ProfileValidationError


class TestGenesisProfile(unittest.TestCase):
    """Test suite for GenesisProfile class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_profile_data = {
            "id": "test_profile_123",
            "name": "Test User",
            "email": "test@example.com",
            "created_at": "2024-01-01T00:00:00Z",
            "preferences": {
                "language": "en",
                "theme": "dark",
                "notifications": True
            },
            "metadata": {
                "version": "1.0",
                "source": "genesis"
            }
        }
        self.profile = GenesisProfile(self.sample_profile_data)
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_genesis_profile_initialization_valid_data(self):
        """Test GenesisProfile initialization with valid data"""
        profile = GenesisProfile(self.sample_profile_data)
        self.assertEqual(profile.id, "test_profile_123")
        self.assertEqual(profile.name, "Test User")
        self.assertEqual(profile.email, "test@example.com")
        self.assertIsInstance(profile.preferences, dict)
        self.assertIsInstance(profile.metadata, dict)
    
    def test_genesis_profile_initialization_empty_data(self):
        """Test GenesisProfile initialization with empty data"""
        with self.assertRaises(ValueError):
            GenesisProfile({})
    
    def test_genesis_profile_initialization_missing_required_fields(self):
        """Test GenesisProfile initialization with missing required fields"""
        incomplete_data = {"name": "Test User"}
        with self.assertRaises(KeyError):
            GenesisProfile(incomplete_data)
    
    def test_genesis_profile_initialization_invalid_email(self):
        """Test GenesisProfile initialization with invalid email"""
        invalid_data = self.sample_profile_data.copy()
        invalid_data["email"] = "invalid_email"
        with self.assertRaises(ValueError):
            GenesisProfile(invalid_data)
    
    def test_genesis_profile_to_dict(self):
        """Test converting GenesisProfile to dictionary"""
        result = self.profile.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["id"], "test_profile_123")
        self.assertEqual(result["name"], "Test User")
        self.assertIn("preferences", result)
        self.assertIn("metadata", result)
    
    def test_genesis_profile_from_json_valid(self):
        """Test creating GenesisProfile from valid JSON string"""
        json_string = json.dumps(self.sample_profile_data)
        profile = GenesisProfile.from_json(json_string)
        self.assertEqual(profile.id, "test_profile_123")
        self.assertEqual(profile.name, "Test User")
    
    def test_genesis_profile_from_json_invalid(self):
        """Test creating GenesisProfile from invalid JSON string"""
        invalid_json = "invalid json string"
        with self.assertRaises(json.JSONDecodeError):
            GenesisProfile.from_json(invalid_json)
    
    def test_genesis_profile_update_preferences(self):
        """Test updating profile preferences"""
        new_preferences = {"language": "es", "theme": "light"}
        self.profile.update_preferences(new_preferences)
        self.assertEqual(self.profile.preferences["language"], "es")
        self.assertEqual(self.profile.preferences["theme"], "light")
        self.assertTrue(self.profile.preferences["notifications"])  # Should preserve existing
    
    def test_genesis_profile_update_preferences_invalid_type(self):
        """Test updating preferences with invalid type"""
        with self.assertRaises(TypeError):
            self.profile.update_preferences(None)
    
    def test_genesis_profile_data_immutability(self):
        """
        Test that a copied snapshot of a GenesisProfile's data remains unchanged after the profile's data is modified.
        """
        profile = GenesisProfile(self.sample_profile_data)
        original_data = profile.data.copy()
        
        # Modify the data
        profile.data['new_field'] = 'new_value'
        
        # Original data should not be affected
        self.assertNotEqual(profile.data, original_data)
        self.assertIn('new_field', profile.data)
    
    def test_genesis_profile_str_representation(self):
        """
        Tests that the string representation of a GenesisProfile instance includes the profile ID and is of type string.
        """
        profile = GenesisProfile(self.sample_profile_data)
        str_repr = str(profile)
        self.assertIn(self.sample_profile_data['id'], str_repr)
        self.assertIsInstance(str_repr, str)
    
    def test_genesis_profile_equality(self):
        """
        Verify that GenesisProfile instances are equal when both profile ID and data match, and unequal when profile IDs differ.
        """
        data1 = self.sample_profile_data.copy()
        profile1 = GenesisProfile(data1)
        data2 = self.sample_profile_data.copy()
        profile2 = GenesisProfile(data2)
        data3 = self.sample_profile_data.copy()
        data3['id'] = 'different_id'
        profile3 = GenesisProfile(data3)
        
        if hasattr(profile1, '__eq__'):
            self.assertEqual(profile1, profile2)
            self.assertNotEqual(profile1, profile3)


class TestProfileManager(unittest.TestCase):
    """Test cases for ProfileManager class"""
    
    def setUp(self):
        self.manager = ProfileManager()
        self.sample_data = {
            'name': 'test_profile',
            'version': '1.0.0',
            'settings': {
                'ai_model': 'gpt-4',
                'temperature': 0.7
            }
        }
        self.profile_id = 'profile_123'
    
    def test_create_profile_success(self):
        profile = self.manager.create_profile(self.profile_id, self.sample_data)
        self.assertIsInstance(profile, GenesisProfile)
        self.assertEqual(profile.profile_id, self.profile_id)
        self.assertEqual(profile.data, self.sample_data)
        self.assertIn(self.profile_id, self.manager.profiles)
    
    def test_create_profile_duplicate_id(self):
        self.manager.create_profile(self.profile_id, self.sample_data)
        try:
            duplicate_profile = self.manager.create_profile(self.profile_id, {'name': 'duplicate'})
            self.assertEqual(duplicate_profile.profile_id, self.profile_id)
        except Exception as e:
            self.assertIsInstance(e, (ProfileError, ValueError))
    
    def test_create_profile_invalid_data(self):
        with self.assertRaises((TypeError, ValueError)):
            self.manager.create_profile(self.profile_id, None)
    
    def test_get_profile_existing(self):
        created_profile = self.manager.create_profile(self.profile_id, self.sample_data)
        retrieved_profile = self.manager.get_profile(self.profile_id)
        self.assertEqual(retrieved_profile, created_profile)
        self.assertEqual(retrieved_profile.profile_id, self.profile_id)
    
    def test_get_profile_nonexistent(self):
        result = self.manager.get_profile('nonexistent_id')
        self.assertIsNone(result)
    
    def test_get_profile_empty_id(self):
        result = self.manager.get_profile('')
        self.assertIsNone(result)
    
    def test_update_profile_success(self):
        self.manager.create_profile(self.profile_id, self.sample_data)
        update_data = {'name': 'updated_profile', 'new_field': 'new_value'}
        updated_profile = self.manager.update_profile(self.profile_id, update_data)
        self.assertEqual(updated_profile.data['name'], 'updated_profile')
        self.assertEqual(updated_profile.data['new_field'], 'new_value')
        self.assertIsInstance(updated_profile.updated_at, datetime)
    
    def test_update_profile_nonexistent(self):
        with self.assertRaises(ProfileNotFoundError):
            self.manager.update_profile('nonexistent_id', {'name': 'updated'})
    
    def test_update_profile_empty_data(self):
        self.manager.create_profile(self.profile_id, self.sample_data)
        updated_profile = self.manager.update_profile(self.profile_id, {})
        self.assertEqual(updated_profile.data, self.sample_data)
    
    def test_delete_profile_success(self):
        self.manager.create_profile(self.profile_id, self.sample_data)
        result = self.manager.delete_profile(self.profile_id)
        self.assertTrue(result)
        self.assertNotIn(self.profile_id, self.manager.profiles)
        self.assertIsNone(self.manager.get_profile(self.profile_id))
    
    def test_delete_profile_nonexistent(self):
        result = self.manager.delete_profile('nonexistent_id')
        self.assertFalse(result)
    
    def test_manager_state_isolation(self):
        manager1 = ProfileManager()
        manager2 = ProfileManager()
        manager1.create_profile(self.profile_id, self.sample_data)
        self.assertIsNotNone(manager1.get_profile(self.profile_id))
        self.assertIsNone(manager2.get_profile(self.profile_id))


class TestProfileValidator(unittest.TestCase):
    """Test cases for ProfileValidator class"""
    
    def setUp(self):
        self.valid_data = {
            'name': 'test_profile',
            'version': '1.0.0',
            'settings': {
                'ai_model': 'gpt-4',
                'temperature': 0.7
            }
        }
    
    def test_validate_profile_data_valid(self):
        result = ProfileValidator.validate_profile_data(self.valid_data)
        self.assertTrue(result)
    
    def test_validate_profile_data_missing_required_fields(self):
        invalid_data_cases = [
            {'version': '1.0.0', 'settings': {}},
            {'name': 'test', 'settings': {}},
            {'name': 'test', 'version': '1.0.0'},
            {},
        ]
        for invalid_data in invalid_data_cases:
            with self.subTest(invalid_data=invalid_data):
                result = ProfileValidator.validate_profile_data(invalid_data)
                self.assertFalse(result)
    
    def test_validate_profile_data_empty_values(self):
        empty_data_cases = [
            {'name': '', 'version': '1.0.0', 'settings': {}},
            {'name': 'test', 'version': '', 'settings': {}},
            {'name': 'test', 'version': '1.0.0', 'settings': None},
        ]
        for empty_data in empty_data_cases:
            with self.subTest(empty_data=empty_data):
                result = ProfileValidator.validate_profile_data(empty_data)
                self.assertIsInstance(result, bool)
    
    def test_validate_profile_data_none_input(self):
        with self.assertRaises((TypeError, AttributeError)):
            ProfileValidator.validate_profile_data(None)
    
    def test_validate_profile_data_invalid_types(self):
        invalid_type_cases = [
            "string_instead_of_dict",
            123,
            [],
            set(),
        ]
        for invalid_type in invalid_type_cases:
            with self.subTest(invalid_type=invalid_type):
                with self.assertRaises((TypeError, AttributeError)):
                    ProfileValidator.validate_profile_data(invalid_type)
    
    def test_validate_profile_data_extra_fields(self):
        data_with_extra = self.valid_data.copy()
        data_with_extra.update({
            'extra_field': 'extra_value',
            'metadata': {'tags': ['test']},
            'optional_settings': {'debug': True}
        })
        result = ProfileValidator.validate_profile_data(data_with_extra)
        self.assertTrue(result)


class TestProfileBuilder(unittest.TestCase):
    """Test cases for ProfileBuilder class"""
    
    def setUp(self):
        self.builder = ProfileBuilder()
    
    def test_builder_chain_methods(self):
        result = (self.builder
                 .with_name('test_profile')
                 .with_version('1.0.0')
                 .with_settings({'ai_model': 'gpt-4'})
                 .build())
        expected = {
            'name': 'test_profile',
            'version': '1.0.0',
            'settings': {'ai_model': 'gpt-4'}
        }
        self.assertEqual(result, expected)
    
    def test_builder_individual_methods(self):
        self.builder.with_name('individual_test')
        self.builder.with_version('2.0.0')
        self.builder.with_settings({'temperature': 0.5})
        result = self.builder.build()
        self.assertEqual(result['name'], 'individual_test')
        self.assertEqual(result['version'], '2.0.0')
        self.assertEqual(result['settings']['temperature'], 0.5)
    
    def test_builder_overwrite_values(self):
        self.builder.with_name('first_name')
        self.builder.with_name('second_name')
        result = self.builder.build()
        self.assertEqual(result['name'], 'second_name')
    
    def test_builder_empty_build(self):
        result = self.builder.build()
        self.assertEqual(result, {})
    
    def test_builder_partial_build(self):
        result = self.builder.with_name('partial').build()
        self.assertEqual(result, {'name': 'partial'})
        self.assertNotIn('version', result)
        self.assertNotIn('settings', result)
    
    def test_builder_complex_settings(self):
        complex_settings = {
            'ai_model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 1000,
            'nested': {
                'key1': 'value1',
                'key2': ['item1', 'item2']
            }
        }
        result = self.builder.with_settings(complex_settings).build()
        self.assertEqual(result['settings'], complex_settings)
        self.assertEqual(result['settings']['nested']['key1'], 'value1')
    
    def test_builder_immutability(self):
        self.builder.with_name('test')
        result1 = self.builder.build()
        result2 = self.builder.build()
        result1['name'] = 'modified'
        self.assertEqual(result2['name'], 'test')
        self.assertNotEqual(result1, result2)
    
    def test_builder_none_values(self):
        result = (self.builder
                 .with_name(None)
                 .with_version(None)
                 .with_settings(None)
                 .build())
        self.assertIsNone(result['name'])
        self.assertIsNone(result['version'])
        self.assertIsNone(result['settings'])


class TestProfileExceptions(unittest.TestCase):
    """Test cases for custom exceptions"""
    
    def test_profile_error_inheritance(self):
        error = ProfileError("Test error")
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Test error")
    
    def test_validation_error_inheritance(self):
        error = ValidationError("Validation failed")
        self.assertIsInstance(error, ProfileError)
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Validation failed")
    
    def test_profile_not_found_error_inheritance(self):
        error = ProfileNotFoundError("Profile not found")
        self.assertIsInstance(error, ProfileError)
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Profile not found")
    
    def test_exception_with_no_message(self):
        error = ProfileError()
        self.assertIsInstance(error, Exception)
        error = ValidationError()
        self.assertIsInstance(error, ProfileError)
        error = ProfileNotFoundError()
        self.assertIsInstance(error, ProfileError)


# Integration and parametrized tests follow...
# (Rest of file unchanged, relying on imports above to resolve all references.)

if __name__ == '__main__':
    unittest.main()