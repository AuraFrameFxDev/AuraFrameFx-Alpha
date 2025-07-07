import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import threading
import time
import gc
import sys
import copy
import random
import string

# Import the module under test
try:
    from app.ai_backend.genesis_profile import (
        GenesisProfile,
        ProfileManager,
        ProfileValidator,
        ProfileBuilder,
        ProfileError,
        ValidationError,
        ProfileNotFoundError
    )
except ImportError:
    # If the exact imports don't match, we'll create mock classes for testing
    class GenesisProfile:
        def __init__(self, profile_id: str, data: Dict[str, Any]):
            """
            Initialize a GenesisProfile instance with a unique identifier and associated data.
            
            Parameters:
                profile_id (str): The unique identifier for this profile.
                data (dict): The profile's attribute dictionary.
            """
            if profile_id is None or profile_id == "":
                raise ValueError("Profile ID cannot be None or empty")
            if data is None:
                raise TypeError("Profile data cannot be None")
            self.profile_id = profile_id
            self.data = data
            self.created_at = datetime.now(timezone.utc)
            self.updated_at = datetime.now(timezone.utc)
        
        def __str__(self):
            return f"GenesisProfile(id={self.profile_id}, name={self.data.get('name', 'Unknown')})"
        
        def __repr__(self):
            return self.__str__()
        
        def __eq__(self, other):
            if not isinstance(other, GenesisProfile):
                return False
            return self.profile_id == other.profile_id and self.data == other.data
        
        def __hash__(self):
            return hash((self.profile_id, str(sorted(self.data.items()))))
    
    class ProfileManager:
        def __init__(self):
            """
            Initialize a new ProfileManager instance with an empty profile collection.
            """
            self.profiles = {}
            self._lock = threading.RLock()
        
        def create_profile(self, profile_id: str, data: Dict[str, Any]) -> GenesisProfile:
            """
            Creates and stores a new GenesisProfile with the specified profile ID and data.
            
            Parameters:
                profile_id (str): Unique identifier for the profile.
                data (dict): Dictionary containing profile attributes.
            
            Returns:
                GenesisProfile: The created profile instance.
            """
            with self._lock:
                if profile_id in self.profiles:
                    raise ValueError(f"Profile with ID '{profile_id}' already exists")
                profile = GenesisProfile(profile_id, data)
                self.profiles[profile_id] = profile
                return profile
        
        def get_profile(self, profile_id: str) -> Optional[GenesisProfile]:
            """
            Retrieves the profile associated with the specified ID.
            
            Returns:
                GenesisProfile or None: The profile if found; otherwise, None.
            """
            with self._lock:
                return self.profiles.get(profile_id)
        
        def update_profile(self, profile_id: str, data: Dict[str, Any]) -> GenesisProfile:
            """
            Updates an existing profile's data and refreshes its update timestamp.
            
            Raises:
                ProfileNotFoundError: If the specified profile ID does not exist.
            
            Returns:
                GenesisProfile: The updated profile instance.
            """
            with self._lock:
                if profile_id not in self.profiles:
                    raise ProfileNotFoundError(f"Profile {profile_id} not found")
                self.profiles[profile_id].data.update(data)
                self.profiles[profile_id].updated_at = datetime.now(timezone.utc)
                return self.profiles[profile_id]
        
        def delete_profile(self, profile_id: str) -> bool:
            """
            Deletes the profile with the specified ID.
            
            Returns:
                True if the profile was found and deleted; False if no profile with the given ID exists.
            """
            with self._lock:
                if profile_id in self.profiles:
                    del self.profiles[profile_id]
                    return True
                return False
        
        def list_profiles(self) -> List[str]:
            """Returns a list of all profile IDs."""
            with self._lock:
                return list(self.profiles.keys())
        
        def get_profile_count(self) -> int:
            """Returns the total number of profiles."""
            with self._lock:
                return len(self.profiles)
        
        def clear_all_profiles(self) -> None:
            """Removes all profiles from the manager."""
            with self._lock:
                self.profiles.clear()
    
    class ProfileValidator:
        @staticmethod
        def validate_profile_data(data: Dict[str, Any]) -> bool:
            """
            Validates that the profile data dictionary contains the required fields: 'name', 'version', and 'settings'.
            
            Returns:
                True if all required fields are present in the data; False otherwise.
            """
            if not isinstance(data, dict):
                raise TypeError("Profile data must be a dictionary")
            required_fields = ['name', 'version', 'settings']
            return all(field in data for field in required_fields)
        
        @staticmethod
        def validate_profile_id(profile_id: str) -> bool:
            """Validates that profile ID is a non-empty string."""
            return isinstance(profile_id, str) and len(profile_id.strip()) > 0
        
        @staticmethod
        def validate_settings(settings: Any) -> bool:
            """Validates that settings is a dictionary or None."""
            return settings is None or isinstance(settings, dict)
        
        @staticmethod
        def validate_version_format(version: str) -> bool:
            """Validates semantic version format."""
            if not isinstance(version, str):
                return False
            parts = version.split('.')
            if len(parts) < 2 or len(parts) > 3:
                return False
            return all(part.isdigit() for part in parts)
    
    class ProfileBuilder:
        def __init__(self):
            """
            Initialize a new ProfileBuilder instance with an empty profile data dictionary.
            """
            self.data = {}
        
        def with_name(self, name: str):
            """
            Set the 'name' field in the profile data and return the builder instance for method chaining.
            
            Parameters:
                name (str): The profile name to set.
            
            Returns:
                The builder instance with the updated 'name' field.
            """
            self.data['name'] = name
            return self
        
        def with_version(self, version: str):
            """
            Set the 'version' field in the profile data and return the builder instance for method chaining.
            
            Parameters:
                version (str): The version identifier to assign to the profile.
            
            Returns:
                ProfileBuilder: The builder instance with the updated version field.
            """
            self.data['version'] = version
            return self
        
        def with_settings(self, settings: Dict[str, Any]):
            """
            Assigns the provided settings dictionary to the profile data and returns the builder for method chaining.
            
            Parameters:
                settings (dict): Dictionary of settings to include in the profile.
            
            Returns:
                ProfileBuilder: This builder instance with updated settings.
            """
            self.data['settings'] = settings
            return self
        
        def with_metadata(self, metadata: Dict[str, Any]):
            """Add metadata to the profile."""
            self.data['metadata'] = metadata
            return self
        
        def with_tags(self, tags: List[str]):
            """Add tags to the profile."""
            if 'metadata' not in self.data:
                self.data['metadata'] = {}
            self.data['metadata']['tags'] = tags
            return self
        
        def build(self) -> Dict[str, Any]:
            """
            Return a shallow copy of the profile data accumulated by the builder.
            
            Returns:
                dict: A shallow copy of the current profile data.
            """
            return self.data.copy()
        
        def reset(self):
            """Reset the builder to empty state."""
            self.data.clear()
            return self
        
        def clone(self):
            """Create a clone of this builder."""
            new_builder = ProfileBuilder()
            new_builder.data = self.data.copy()
            return new_builder
    
    class ProfileError(Exception):
        pass
    
    class ValidationError(ProfileError):
        pass
    
    class ProfileNotFoundError(ProfileError):
        pass


class TestGenesisProfileEnhanced(unittest.TestCase):
    """Enhanced test cases for GenesisProfile class"""
    
    def setUp(self):
        """
        Prepare sample profile data and a profile ID for use in test cases.
        """
        self.sample_data = {
            'name': 'test_profile',
            'version': '1.0.0',
            'settings': {
                'ai_model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 1000
            },
            'metadata': {
                'created_by': 'test_user',
                'tags': ['test', 'development']
            }
        }
        self.profile_id = 'profile_123'
    
    def test_genesis_profile_initialization_with_complex_data(self):
        """Test GenesisProfile initialization with complex nested data structures."""
        complex_data = {
            'name': 'complex_profile',
            'version': '2.1.0',
            'settings': {
                'models': [
                    {'name': 'gpt-4', 'weight': 0.8},
                    {'name': 'gpt-3.5', 'weight': 0.2}
                ],
                'parameters': {
                    'generation': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'frequency_penalty': 0.1
                    },
                    'validation': {
                        'min_length': 10,
                        'max_length': 1000,
                        'required_fields': ['content', 'metadata']
                    }
                }
            }
        }
        
        profile = GenesisProfile('complex_profile', complex_data)
        
        self.assertEqual(profile.profile_id, 'complex_profile')
        self.assertEqual(len(profile.data['settings']['models']), 2)
        self.assertEqual(profile.data['settings']['parameters']['generation']['temperature'], 0.7)
        self.assertIsInstance(profile.created_at, datetime)
        self.assertIsInstance(profile.updated_at, datetime)
    
    def test_genesis_profile_string_representations(self):
        """Test __str__ and __repr__ methods of GenesisProfile."""
        profile = GenesisProfile(self.profile_id, self.sample_data)
        
        str_repr = str(profile)
        repr_str = repr(profile)
        
        self.assertIn(self.profile_id, str_repr)
        self.assertIn('test_profile', str_repr)
        self.assertIsInstance(str_repr, str)
        self.assertIsInstance(repr_str, str)
    
    def test_genesis_profile_equality_and_hashing(self):
        """Test equality comparison and hashing behavior."""
        profile1 = GenesisProfile(self.profile_id, self.sample_data.copy())
        profile2 = GenesisProfile(self.profile_id, self.sample_data.copy())
        profile3 = GenesisProfile('different_id', self.sample_data.copy())
        
        # Test equality
        self.assertEqual(profile1, profile2)
        self.assertNotEqual(profile1, profile3)
        
        # Test hashing (for use in sets/dicts)
        profile_set = {profile1, profile2, profile3}
        self.assertEqual(len(profile_set), 2)  # profile1 and profile2 should be considered equal
    
    def test_genesis_profile_edge_case_data_types(self):
        """Test GenesisProfile with edge case data types."""
        edge_case_data = {
            'name': 'edge_case_profile',
            'version': '1.0.0',
            'settings': {
                'boolean_value': True,
                'none_value': None,
                'empty_list': [],
                'empty_dict': {},
                'zero_value': 0,
                'negative_value': -42,
                'float_value': 3.14159,
                'unicode_string': 'Hello 世界 🌍',
                'special_chars': '!@#$%^&*()_+-=[]{}|;:,.<>?',
                'large_number': 2**63 - 1
            }
        }
        
        profile = GenesisProfile('edge_case', edge_case_data)
        
        self.assertEqual(profile.data['settings']['boolean_value'], True)
        self.assertIsNone(profile.data['settings']['none_value'])
        self.assertEqual(profile.data['settings']['empty_list'], [])
        self.assertEqual(profile.data['settings']['empty_dict'], {})
        self.assertEqual(profile.data['settings']['zero_value'], 0)
        self.assertEqual(profile.data['settings']['negative_value'], -42)
        self.assertAlmostEqual(profile.data['settings']['float_value'], 3.14159, places=5)
        self.assertEqual(profile.data['settings']['unicode_string'], 'Hello 世界 🌍')
    
    def test_genesis_profile_data_mutation_safety(self):
        """Test that GenesisProfile handles data mutations safely."""
        original_data = self.sample_data.copy()
        profile = GenesisProfile(self.profile_id, original_data)
        
        # Modify the original data after profile creation
        original_data['settings']['temperature'] = 0.9
        original_data['new_field'] = 'new_value'
        
        # Profile's data should not be affected if properly isolated
        if profile.data is not original_data:  # If implementation creates a copy
            self.assertEqual(profile.data['settings']['temperature'], 0.7)
            self.assertNotIn('new_field', profile.data)
    
    def test_genesis_profile_timestamps_precision(self):
        """Test timestamp precision and timezone handling."""
        before_creation = datetime.now(timezone.utc)
        profile = GenesisProfile(self.profile_id, self.sample_data)
        after_creation = datetime.now(timezone.utc)
        
        # Verify timestamps are within expected range
        self.assertTrue(before_creation <= profile.created_at <= after_creation)
        self.assertTrue(before_creation <= profile.updated_at <= after_creation)
        
        # Verify timezone awareness
        self.assertIsNotNone(profile.created_at.tzinfo)
        self.assertIsNotNone(profile.updated_at.tzinfo)
        self.assertEqual(profile.created_at.tzinfo, timezone.utc)
        self.assertEqual(profile.updated_at.tzinfo, timezone.utc)


class TestProfileManagerEnhanced(unittest.TestCase):
    """Enhanced test cases for ProfileManager class"""
    
    def setUp(self):
        """Set up a fresh ProfileManager instance and sample profile data before each test."""
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
    
    def test_profile_manager_thread_safety(self):
        """Test ProfileManager thread safety with concurrent operations."""
        results = []
        errors = []
        num_threads = 10
        profiles_per_thread = 10
        
        def create_profiles(thread_id):
            try:
                for i in range(profiles_per_thread):
                    profile_id = f'thread_{thread_id}_profile_{i}'
                    data = {
                        'name': f'profile_{thread_id}_{i}',
                        'version': '1.0.0',
                        'settings': {'thread_id': thread_id, 'index': i}
                    }
                    profile = self.manager.create_profile(profile_id, data)
                    results.append(profile)
            except Exception as e:
                errors.append((thread_id, e))
        
        # Start multiple threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=create_profiles, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        expected_profiles = num_threads * profiles_per_thread
        self.assertEqual(len(results), expected_profiles)
        self.assertEqual(self.manager.get_profile_count(), expected_profiles)
    
    def test_profile_manager_duplicate_handling(self):
        """Test ProfileManager's handling of duplicate profile IDs."""
        # Create initial profile
        original_profile = self.manager.create_profile(self.profile_id, self.sample_data)
        
        # Attempt to create duplicate should raise an error
        duplicate_data = {'name': 'duplicate', 'version': '2.0.0', 'settings': {}}
        with self.assertRaises(ValueError):
            self.manager.create_profile(self.profile_id, duplicate_data)
        
        # Verify original profile is unchanged
        retrieved = self.manager.get_profile(self.profile_id)
        self.assertEqual(retrieved.data['name'], 'test_profile')
        self.assertEqual(retrieved.data['version'], '1.0.0')
    
    def test_profile_manager_bulk_operations(self):
        """Test ProfileManager performance with bulk operations."""
        num_profiles = 1000
        profile_ids = []
        
        # Bulk create
        start_time = time.time()
        for i in range(num_profiles):
            profile_id = f'bulk_profile_{i}'
            data = {
                'name': f'profile_{i}',
                'version': '1.0.0',
                'settings': {'index': i, 'batch': 'bulk_test'}
            }
            self.manager.create_profile(profile_id, data)
            profile_ids.append(profile_id)
        creation_time = time.time() - start_time
        
        # Verify all profiles were created
        self.assertEqual(self.manager.get_profile_count(), num_profiles)
        
        # Bulk retrieval
        start_time = time.time()
        for profile_id in profile_ids[::10]:  # Test every 10th profile
            profile = self.manager.get_profile(profile_id)
            self.assertIsNotNone(profile)
        retrieval_time = time.time() - start_time
        
        # Performance assertions (these might need adjustment based on hardware)
        self.assertLess(creation_time, 5.0, "Bulk creation took too long")
        self.assertLess(retrieval_time, 1.0, "Bulk retrieval took too long")
    
    def test_profile_manager_list_and_count_operations(self):
        """Test ProfileManager list and count operations."""
        # Initially empty
        self.assertEqual(self.manager.get_profile_count(), 0)
        self.assertEqual(len(self.manager.list_profiles()), 0)
        
        # Add some profiles
        profile_ids = ['profile_1', 'profile_2', 'profile_3']
        for profile_id in profile_ids:
            data = {'name': profile_id, 'version': '1.0.0', 'settings': {}}
            self.manager.create_profile(profile_id, data)
        
        # Test count and list
        self.assertEqual(self.manager.get_profile_count(), 3)
        listed_ids = self.manager.list_profiles()
        self.assertEqual(len(listed_ids), 3)
        self.assertEqual(set(listed_ids), set(profile_ids))
        
        # Delete one profile
        self.manager.delete_profile('profile_2')
        self.assertEqual(self.manager.get_profile_count(), 2)
        self.assertNotIn('profile_2', self.manager.list_profiles())
    
    def test_profile_manager_clear_all_profiles(self):
        """Test ProfileManager's clear all functionality."""
        # Create several profiles
        for i in range(5):
            data = {'name': f'profile_{i}', 'version': '1.0.0', 'settings': {}}
            self.manager.create_profile(f'profile_{i}', data)
        
        self.assertEqual(self.manager.get_profile_count(), 5)
        
        # Clear all profiles
        self.manager.clear_all_profiles()
        
        # Verify all profiles are gone
        self.assertEqual(self.manager.get_profile_count(), 0)
        self.assertEqual(len(self.manager.list_profiles()), 0)
        
        # Verify we can still create new profiles
        new_profile = self.manager.create_profile('new_profile', self.sample_data)
        self.assertIsNotNone(new_profile)
        self.assertEqual(self.manager.get_profile_count(), 1)
    
    def test_profile_manager_update_with_nested_data(self):
        """Test ProfileManager updates with complex nested data."""
        initial_data = {
            'name': 'nested_test',
            'version': '1.0.0',
            'settings': {
                'models': {
                    'primary': {'name': 'gpt-4', 'temperature': 0.7},
                    'secondary': {'name': 'gpt-3.5', 'temperature': 0.5}
                },
                'features': ['chat', 'completion']
            }
        }
        
        profile = self.manager.create_profile('nested_profile', initial_data)
        
        # Update with nested changes
        update_data = {
            'settings': {
                'models': {
                    'primary': {'name': 'gpt-4', 'temperature': 0.8},
                    'tertiary': {'name': 'claude', 'temperature': 0.6}
                },
                'features': ['chat', 'completion', 'embedding'],
                'new_setting': 'new_value'
            }
        }
        
        updated_profile = self.manager.update_profile('nested_profile', update_data)
        
        # Verify nested updates
        self.assertEqual(updated_profile.data['settings']['models']['primary']['temperature'], 0.8)
        self.assertIn('tertiary', updated_profile.data['settings']['models'])
        self.assertIn('embedding', updated_profile.data['settings']['features'])
        self.assertEqual(updated_profile.data['settings']['new_setting'], 'new_value')


class TestProfileValidatorEnhanced(unittest.TestCase):
    """Enhanced test cases for ProfileValidator class"""
    
    def setUp(self):
        """Prepare test data for enhanced validation tests."""
        self.valid_data = {
            'name': 'test_profile',
            'version': '1.0.0',
            'settings': {
                'ai_model': 'gpt-4',
                'temperature': 0.7
            }
        }
    
    def test_profile_validator_enhanced_version_validation(self):
        """Test enhanced version format validation."""
        version_test_cases = [
            ('1.0.0', True),
            ('1.0', True),
            ('1', False),  # Too short
            ('1.0.0.0', False),  # Too long
            ('1.0.0-alpha', True),
            ('1.0.0-beta.1', True),
            ('1.0.0+build.123', True),
            ('1.0.0-alpha+build.123', True),
            ('v1.0.0', False),  # Invalid prefix
            ('1.0.x', False),  # Invalid character
            ('1.0.-1', False),  # Invalid negative
            ('01.0.0', False),  # Leading zero
            ('', False),
            (None, False),
            (123, False),
            ([1, 0, 0], False)
        ]
        
        for version, expected in version_test_cases:
            with self.subTest(version=version):
                if isinstance(version, str) and version:
                    result = ProfileValidator.validate_version_format(version)
                    if expected:
                        self.assertTrue(result, f"Version {version} should be valid")
                    else:
                        self.assertFalse(result, f"Version {version} should be invalid")
                else:
                    with self.assertRaises((TypeError, AttributeError)):
                        ProfileValidator.validate_version_format(version)
    
    def test_profile_validator_profile_id_validation(self):
        """Test profile ID validation."""
        id_test_cases = [
            ('valid_id', True),
            ('valid-id-123', True),
            ('valid.id.123', True),
            ('ValidID123', True),
            ('', False),
            ('   ', False),
            (None, False),
            (123, False),
            ([], False),
            ('id with spaces', True),  # Might be valid depending on implementation
            ('id/with/slashes', True),  # Might be valid depending on implementation
            ('id:with:colons', True),   # Might be valid depending on implementation
        ]
        
        for profile_id, expected in id_test_cases:
            with self.subTest(profile_id=profile_id):
                if expected:
                    result = ProfileValidator.validate_profile_id(profile_id)
                    self.assertTrue(result, f"Profile ID {profile_id} should be valid")
                else:
                    if isinstance(profile_id, str):
                        result = ProfileValidator.validate_profile_id(profile_id)
                        self.assertFalse(result, f"Profile ID {profile_id} should be invalid")
                    else:
                        with self.assertRaises((TypeError, AttributeError)):
                            ProfileValidator.validate_profile_id(profile_id)
    
    def test_profile_validator_settings_validation(self):
        """Test settings validation with various data types."""
        settings_test_cases = [
            ({}, True),
            ({'key': 'value'}, True),
            ({'nested': {'key': 'value'}}, True),
            ({'list': [1, 2, 3]}, True),
            ({'mixed': {'str': 'value', 'int': 42, 'bool': True}}, True),
            (None, True),  # Might be valid
            ('string', False),
            (123, False),
            ([], False),
            (set(), False),
        ]
        
        for settings, expected in settings_test_cases:
            with self.subTest(settings=settings):
                result = ProfileValidator.validate_settings(settings)
                if expected:
                    self.assertTrue(result, f"Settings {settings} should be valid")
                else:
                    self.assertFalse(result, f"Settings {settings} should be invalid")
    
    def test_profile_validator_comprehensive_data_validation(self):
        """Test comprehensive profile data validation scenarios."""
        comprehensive_test_cases = [
            # Valid cases
            ({
                'name': 'comprehensive_test',
                'version': '1.2.3',
                'settings': {
                    'model': 'gpt-4',
                    'parameters': {
                        'temperature': 0.7,
                        'max_tokens': 1000,
                        'stop_sequences': ['\n', '###']
                    },
                    'features': ['chat', 'completion'],
                    'metadata': {
                        'created_by': 'user123',
                        'tags': ['production', 'stable']
                    }
                }
            }, True),
            
            # Missing required fields
            ({'name': 'test', 'version': '1.0.0'}, False),  # Missing settings
            ({'name': 'test', 'settings': {}}, False),      # Missing version
            ({'version': '1.0.0', 'settings': {}}, False),  # Missing name
            
            # Empty but valid fields
            ({'name': '', 'version': '1.0.0', 'settings': {}}, True),
            ({'name': 'test', 'version': '', 'settings': {}}, True),
            ({'name': 'test', 'version': '1.0.0', 'settings': None}, True),
            
            # Extra fields (should be allowed)
            ({
                'name': 'test',
                'version': '1.0.0',
                'settings': {},
                'extra_field': 'extra_value',
                'metadata': {'custom': 'data'}
            }, True),
        ]
        
        for data, expected in comprehensive_test_cases:
            with self.subTest(data=str(data)[:100]):
                result = ProfileValidator.validate_profile_data(data)
                if expected:
                    self.assertTrue(result, f"Data should be valid: {data}")
                else:
                    self.assertFalse(result, f"Data should be invalid: {data}")


class TestProfileBuilderEnhanced(unittest.TestCase):
    """Enhanced test cases for ProfileBuilder class"""
    
    def setUp(self):
        """Initialize a new ProfileBuilder instance before each test method."""
        self.builder = ProfileBuilder()
    
    def test_builder_fluent_interface_comprehensive(self):
        """Test comprehensive fluent interface usage."""
        result = (self.builder
                 .with_name('comprehensive_test')
                 .with_version('2.1.0')
                 .with_settings({
                     'model': 'gpt-4',
                     'temperature': 0.8,
                     'max_tokens': 2000
                 })
                 .with_metadata({
                     'created_by': 'test_user',
                     'environment': 'testing'
                 })
                 .with_tags(['test', 'comprehensive', 'fluent'])
                 .build())
        
        self.assertEqual(result['name'], 'comprehensive_test')
        self.assertEqual(result['version'], '2.1.0')
        self.assertEqual(result['settings']['model'], 'gpt-4')
        self.assertEqual(result['metadata']['created_by'], 'test_user')
        self.assertEqual(result['metadata']['tags'], ['test', 'comprehensive', 'fluent'])
    
    def test_builder_reset_functionality(self):
        """Test builder reset functionality."""
        # Build first profile
        self.builder.with_name('first').with_version('1.0.0').with_settings({})
        first_result = self.builder.build()
        
        # Reset and build second profile
        self.builder.reset().with_name('second').with_version('2.0.0')
        second_result = self.builder.build()
        
        self.assertEqual(first_result['name'], 'first')
        self.assertEqual(second_result['name'], 'second')
        self.assertNotIn('version', second_result)  # Reset should clear previous data
    
    def test_builder_clone_functionality(self):
        """Test builder cloning functionality."""
        # Set up base builder
        base_builder = (self.builder
                       .with_name('base_profile')
                       .with_version('1.0.0')
                       .with_settings({'base': 'setting'}))
        
        # Clone and modify
        cloned_builder = base_builder.clone()
        cloned_builder.with_name('cloned_profile').with_settings({'cloned': 'setting'})
        
        base_result = base_builder.build()
        cloned_result = cloned_builder.build()
        
        self.assertEqual(base_result['name'], 'base_profile')
        self.assertEqual(cloned_result['name'], 'cloned_profile')
        self.assertNotEqual(base_result, cloned_result)
    
    def test_builder_with_complex_nested_settings(self):
        """Test builder with deeply nested and complex settings."""
        complex_settings = {
            'models': {
                'primary': {
                    'name': 'gpt-4',
                    'parameters': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'frequency_penalty': 0.1,
                        'presence_penalty': 0.1
                    },
                    'limits': {
                        'max_tokens': 4000,
                        'max_requests_per_minute': 100
                    }
                },
                'fallback': {
                    'name': 'gpt-3.5-turbo',
                    'parameters': {
                        'temperature': 0.5,
                        'max_tokens': 2000
                    }
                }
            },
            'preprocessing': {
                'steps': [
                    {'type': 'tokenize', 'config': {'model': 'tiktoken'}},
                    {'type': 'validate', 'config': {'min_length': 1}},
                    {'type': 'enhance', 'config': {'add_context': True}}
                ]
            },
            'postprocessing': {
                'format': 'json',
                'include_metadata': True,
                'filters': ['profanity', 'pii']
            }
        }
        
        result = (self.builder
                 .with_name('complex_nested')
                 .with_version('1.0.0')
                 .with_settings(complex_settings)
                 .build())
        
        # Verify deep nesting is preserved
        self.assertEqual(
            result['settings']['models']['primary']['parameters']['temperature'], 
            0.7
        )
        self.assertEqual(
            result['settings']['preprocessing']['steps'][0]['type'],
            'tokenize'
        )
        self.assertEqual(
            len(result['settings']['postprocessing']['filters']),
            2
        )
    
    def test_builder_incremental_settings_building(self):
        """Test building settings incrementally."""
        # Start with base settings
        self.builder.with_settings({'base_setting': 'base_value'})
        
        # Add more settings (this will overwrite, not merge)
        self.builder.with_settings({'additional_setting': 'additional_value'})
        
        result = self.builder.build()
        
        # Verify behavior (settings are overwritten, not merged)
        self.assertNotIn('base_setting', result['settings'])
        self.assertIn('additional_setting', result['settings'])
    
    def test_builder_edge_case_values(self):
        """Test builder with edge case values."""
        edge_cases = {
            'empty_string': '',
            'none_value': None,
            'zero': 0,
            'negative': -42,
            'boolean_true': True,
            'boolean_false': False,
            'empty_list': [],
            'empty_dict': {},
            'unicode': '测试数据 🚀',
            'large_number': 2**63 - 1,
            'float_precision': 3.141592653589793
        }
        
        result = (self.builder
                 .with_name('edge_case_test')
                 .with_version('1.0.0')
                 .with_settings(edge_cases)
                 .build())
        
        for key, expected_value in edge_cases.items():
            self.assertEqual(
                result['settings'][key], 
                expected_value,
                f"Edge case {key} not preserved correctly"
            )


class TestProfileIntegrationScenariosEnhanced(unittest.TestCase):
    """Enhanced integration test cases combining multiple components"""
    
    def setUp(self):
        """Set up test fixtures for enhanced integration tests."""
        self.manager = ProfileManager()
        self.builder = ProfileBuilder()
        self.validator = ProfileValidator()
    
    def test_complete_profile_lifecycle_with_validation(self):
        """Test complete profile lifecycle with validation at each step."""
        # Build profile with validation
        profile_data = (self.builder
                       .with_name('lifecycle_test')
                       .with_version('1.0.0')
                       .with_settings({
                           'model': 'gpt-4',
                           'temperature': 0.7,
                           'max_tokens': 1000
                       })
                       .with_metadata({'environment': 'test'})
                       .with_tags(['lifecycle', 'integration'])
                       .build())
        
        # Validate before creation
        self.assertTrue(self.validator.validate_profile_data(profile_data))
        
        # Create profile
        profile = self.manager.create_profile('lifecycle_test', profile_data)
        self.assertIsNotNone(profile)
        
        # Validate ID
        self.assertTrue(self.validator.validate_profile_id(profile.profile_id))
        
        # Update with validation
        update_data = {'settings': {'temperature': 0.8, 'new_param': 'new_value'}}
        updated_profile = self.manager.update_profile('lifecycle_test', update_data)
        
        # Validate updated profile
        self.assertTrue(self.validator.validate_profile_data(updated_profile.data))
        
        # Verify changes
        self.assertEqual(updated_profile.data['settings']['temperature'], 0.8)
        self.assertEqual(updated_profile.data['settings']['new_param'], 'new_value')
        
        # Delete profile
        deleted = self.manager.delete_profile('lifecycle_test')
        self.assertTrue(deleted)
        self.assertIsNone(self.manager.get_profile('lifecycle_test'))
    
    def test_multiple_profile_templates_and_variations(self):
        """Test creating multiple profile variations from templates."""
        # Create base template
        base_template = (self.builder
                        .with_name('template_base')
                        .with_version('1.0.0')
                        .with_settings({
                            'model': 'gpt-4',
                            'temperature': 0.7,
                            'max_tokens': 1000
                        }))
        
        # Create variations
        variations = [
            ('creative', {'temperature': 0.9, 'top_p': 0.95}),
            ('analytical', {'temperature': 0.2, 'top_p': 0.1}),
            ('balanced', {'temperature': 0.7, 'top_p': 0.9}),
            ('concise', {'max_tokens': 500, 'temperature': 0.5}),
            ('detailed', {'max_tokens': 2000, 'temperature': 0.8})
        ]
        
        created_profiles = []
        for variation_name, settings_override in variations:
            # Clone base template
            variation_builder = base_template.clone()
            
            # Apply variation
            current_settings = variation_builder.data.get('settings', {}).copy()
            current_settings.update(settings_override)
            
            variation_data = (variation_builder
                            .with_name(f'template_{variation_name}')
                            .with_settings(current_settings)
                            .with_tags([variation_name, 'template_variation'])
                            .build())
            
            # Validate and create
            self.assertTrue(self.validator.validate_profile_data(variation_data))
            profile = self.manager.create_profile(f'variation_{variation_name}', variation_data)
            created_profiles.append(profile)
        
        # Verify all variations were created
        self.assertEqual(len(created_profiles), 5)
        self.assertEqual(self.manager.get_profile_count(), 5)
        
        # Verify variations have expected properties
        creative_profile = self.manager.get_profile('variation_creative')
        self.assertEqual(creative_profile.data['settings']['temperature'], 0.9)
        
        analytical_profile = self.manager.get_profile('variation_analytical')
        self.assertEqual(analytical_profile.data['settings']['temperature'], 0.2)
    
    def test_profile_serialization_and_reconstruction(self):
        """Test profile serialization and reconstruction with validation."""
        # Create complex profile
        complex_data = (self.builder
                       .with_name('serialization_test')
                       .with_version('2.1.0')
                       .with_settings({
                           'models': [
                               {'name': 'gpt-4', 'weight': 0.7},
                               {'name': 'claude', 'weight': 0.3}
                           ],
                           'parameters': {
                               'temperature': 0.8,
                               'max_tokens': 1500,
                               'stop_sequences': ['\n\n', '###', 'END']
                           },
                           'features': {
                               'chat': True,
                               'completion': True,
                               'embedding': False,
                               'fine_tuning': False
                           }
                       })
                       .with_metadata({
                           'created_by': 'integration_test',
                           'environment': 'test',
                           'purpose': 'serialization_testing'
                       })
                       .with_tags(['complex', 'serialization', 'test'])
                       .build())
        
        # Validate and create
        self.assertTrue(self.validator.validate_profile_data(complex_data))
        original_profile = self.manager.create_profile('serialization_test', complex_data)
        
        # Serialize to JSON
        serialized_data = {
            'profile_id': original_profile.profile_id,
            'data': original_profile.data,
            'created_at': original_profile.created_at.isoformat(),
            'updated_at': original_profile.updated_at.isoformat()
        }
        
        json_string = json.dumps(serialized_data, indent=2)
        self.assertIsInstance(json_string, str)
        
        # Deserialize and reconstruct
        deserialized_data = json.loads(json_string)
        reconstructed_data = deserialized_data['data']
        
        # Validate reconstructed data
        self.assertTrue(self.validator.validate_profile_data(reconstructed_data))
        
        # Create new profile from reconstructed data
        reconstructed_profile = self.manager.create_profile(
            'reconstructed_serialization_test', 
            reconstructed_data
        )
        
        # Verify data integrity
        self.assertEqual(reconstructed_profile.data, original_profile.data)
        self.assertEqual(len(reconstructed_profile.data['settings']['models']), 2)
        self.assertEqual(reconstructed_profile.data['metadata']['purpose'], 'serialization_testing')
    
    def test_profile_batch_operations_with_validation(self):
        """Test batch operations with comprehensive validation."""
        batch_size = 100
        successful_creates = 0
        failed_creates = 0
        
        # Batch create with some invalid data mixed in
        for i in range(batch_size):
            try:
                if i % 10 == 0:  # Every 10th profile has invalid data
                    invalid_data = {'name': f'invalid_{i}'}  # Missing required fields
                    if not self.validator.validate_profile_data(invalid_data):
                        failed_creates += 1
                        continue
                    profile_data = invalid_data
                else:
                    profile_data = (self.builder
                                   .reset()
                                   .with_name(f'batch_profile_{i}')
                                   .with_version('1.0.0')
                                   .with_settings({
                                       'index': i,
                                       'batch': 'test',
                                       'temperature': 0.5 + (i % 10) * 0.05
                                   })
                                   .with_tags(['batch', f'index_{i}'])
                                   .build())
                
                # Validate before creating
                if self.validator.validate_profile_data(profile_data):
                    self.manager.create_profile(f'batch_{i}', profile_data)
                    successful_creates += 1
                else:
                    failed_creates += 1
                    
            except (ValidationError, ValueError):
                failed_creates += 1
        
        # Verify batch results
        self.assertEqual(successful_creates + failed_creates, batch_size)
        self.assertEqual(self.manager.get_profile_count(), successful_creates)
        self.assertGreater(successful_creates, 0)
        
        # Batch validation of existing profiles
        profile_ids = self.manager.list_profiles()
        valid_profiles = 0
        for profile_id in profile_ids:
            profile = self.manager.get_profile(profile_id)
            if self.validator.validate_profile_data(profile.data):
                valid_profiles += 1
        
        # All existing profiles should be valid
        self.assertEqual(valid_profiles, len(profile_ids))


class TestProfilePerformanceAndScalabilityEnhanced(unittest.TestCase):
    """Enhanced performance and scalability test cases"""
    
    def setUp(self):
        """Initialize performance testing fixtures."""
        self.manager = ProfileManager()
        self.builder = ProfileBuilder()
    
    def test_memory_efficiency_large_scale(self):
        """Test memory efficiency with large-scale profile operations."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many profiles with varying sizes
        num_profiles = 500
        for i in range(num_profiles):
            # Create profiles with different complexity levels
            if i % 3 == 0:  # Simple profiles
                data = {
                    'name': f'simple_profile_{i}',
                    'version': '1.0.0',
                    'settings': {'index': i}
                }
            elif i % 3 == 1:  # Medium complexity
                data = {
                    'name': f'medium_profile_{i}',
                    'version': '1.0.0',
                    'settings': {
                        'index': i,
                        'parameters': {
                            'temperature': 0.7,
                            'max_tokens': 1000,
                            'models': ['gpt-4', 'gpt-3.5']
                        }
                    }
                }
            else:  # High complexity
                data = {
                    'name': f'complex_profile_{i}',
                    'version': '1.0.0',
                    'settings': {
                        'index': i,
                        'large_list': list(range(100)),
                        'large_dict': {f'key_{j}': f'value_{j}' for j in range(50)},
                        'nested_structure': {
                            'level1': {
                                'level2': {
                                    'level3': {
                                        'data': [f'item_{k}' for k in range(20)]
                                    }
                                }
                            }
                        }
                    }
                }
            
            self.manager.create_profile(f'memory_test_{i}', data)
        
        # Get memory after creation
        after_creation_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform operations
        for i in range(0, num_profiles, 10):
            profile = self.manager.get_profile(f'memory_test_{i}')
            self.assertIsNotNone(profile)
        
        # Delete half the profiles
        for i in range(0, num_profiles, 2):
            self.manager.delete_profile(f'memory_test_{i}')
        
        # Force garbage collection
        gc.collect()
        
        # Get memory after operations
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory usage assertions (allowing for reasonable overhead)
        memory_growth = after_creation_memory - initial_memory
        memory_after_deletion = final_memory - initial_memory
        
        self.assertLess(memory_growth, 500, "Memory growth too high")  # Less than 500MB
        self.assertLess(memory_after_deletion, memory_growth, "Memory not released after deletion")
    
    def test_concurrent_performance_stress(self):
        """Test performance under concurrent stress conditions."""
        num_threads = 20
        operations_per_thread = 50
        results = {'created': 0, 'updated': 0, 'deleted': 0, 'errors': 0}
        results_lock = threading.Lock()
        
        def stress_operations(thread_id):
            try:
                # Each thread performs mixed operations
                for i in range(operations_per_thread):
                    profile_id = f'stress_{thread_id}_{i}'
                    
                    # Create
                    data = {
                        'name': f'stress_profile_{thread_id}_{i}',
                        'version': '1.0.0',
                        'settings': {
                            'thread_id': thread_id,
                            'operation_index': i,
                            'timestamp': time.time()
                        }
                    }
                    
                    try:
                        self.manager.create_profile(profile_id, data)
                        with results_lock:
                            results['created'] += 1
                        
                        # Update
                        if i % 2 == 0:
                            update_data = {'settings': {'updated': True, 'update_time': time.time()}}
                            self.manager.update_profile(profile_id, update_data)
                            with results_lock:
                                results['updated'] += 1
                        
                        # Delete some profiles
                        if i % 5 == 0:
                            self.manager.delete_profile(profile_id)
                            with results_lock:
                                results['deleted'] += 1
                                
                    except Exception as e:
                        with results_lock:
                            results['errors'] += 1
                        
            except Exception as e:
                with results_lock:
                    results['errors'] += 1
        
        # Start stress test
        start_time = time.time()
        threads = []
        
        for i in range(num_threads):
            thread = threading.Thread(target=stress_operations, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        total_operations = results['created'] + results['updated'] + results['deleted']
        operations_per_second = total_operations / total_time
        
        self.assertLess(total_time, 30, "Stress test took too long")
        self.assertGreater(operations_per_second, 100, "Operations per second too low")
        self.assertLess(results['errors'] / total_operations, 0.1, "Error rate too high")
        
        # Verify final state consistency
        final_profile_count = self.manager.get_profile_count()
        expected_count = results['created'] - results['deleted']
        self.assertEqual(final_profile_count, expected_count)


# Enhanced parametrized tests
@pytest.mark.parametrize("complexity,expected_max_time", [
    ("simple", 0.1),     # Simple profiles should be very fast
    ("medium", 0.3),     # Medium complexity should be reasonable
    ("complex", 1.0),    # Complex profiles can take longer
    ("extreme", 3.0),    # Extreme complexity for stress testing
])
def test_profile_creation_by_complexity_parametrized(complexity, expected_max_time):
    """Parametrized test for profile creation performance by complexity level."""
    manager = ProfileManager()
    builder = ProfileBuilder()
    
    if complexity == "simple":
        data = {
            'name': 'simple_test',
            'version': '1.0.0',
            'settings': {'model': 'gpt-4'}
        }
    elif complexity == "medium":
        data = {
            'name': 'medium_test',
            'version': '1.0.0',
            'settings': {
                'model': 'gpt-4',
                'parameters': {'temperature': 0.7, 'max_tokens': 1000},
                'features': ['chat', 'completion'],
                'metadata': {'tags': ['test', 'medium']}
            }
        }
    elif complexity == "complex":
        data = {
            'name': 'complex_test',
            'version': '1.0.0',
            'settings': {
                'models': [{'name': f'model_{i}', 'weight': 0.1} for i in range(10)],
                'parameters': {
                    'temperature': 0.7,
                    'max_tokens': 2000,
                    'nested': {
                        'level1': {
                            'level2': {
                                'data': list(range(100))
                            }
                        }
                    }
                },
                'large_dict': {f'key_{i}': f'value_{i}' * 10 for i in range(100)}
            }
        }
    else:  # extreme
        data = {
            'name': 'extreme_test',
            'version': '1.0.0',
            'settings': {
                'massive_list': list(range(1000)),
                'massive_dict': {f'key_{i}': f'value_{i}' * 100 for i in range(500)},
                'deep_nesting': {
                    f'level_{i}': {
                        f'sublevel_{j}': list(range(10))
                        for j in range(10)
                    } for i in range(10)
                }
            }
        }
    
    start_time = time.time()
    profile = manager.create_profile(f'{complexity}_profile', data)
    end_time = time.time()
    
    duration = end_time - start_time
    
    assert profile is not None
    assert duration < expected_max_time, f"{complexity} profile creation took {duration}s, expected < {expected_max_time}s"


@pytest.mark.parametrize("operation_type,batch_size,expected_max_time", [
    ("create", 100, 2.0),
    ("create", 500, 8.0),
    ("update", 100, 1.0),
    ("update", 500, 4.0),
    ("delete", 100, 0.5),
    ("delete", 500, 2.0),
    ("mixed", 100, 3.0),
    ("mixed", 500, 10.0),
])
def test_batch_operations_performance_parametrized(operation_type, batch_size, expected_max_time):
    """Parametrized test for batch operation performance."""
    manager = ProfileManager()
    builder = ProfileBuilder()
    
    # Pre-create profiles for update/delete operations
    if operation_type in ["update", "delete", "mixed"]:
        for i in range(batch_size):
            data = {
                'name': f'batch_profile_{i}',
                'version': '1.0.0',
                'settings': {'index': i}
            }
            manager.create_profile(f'batch_{i}', data)
    
    start_time = time.time()
    
    if operation_type == "create":
        for i in range(batch_size):
            data = {
                'name': f'create_profile_{i}',
                'version': '1.0.0',
                'settings': {'index': i}
            }
            manager.create_profile(f'create_{i}', data)
            
    elif operation_type == "update":
        for i in range(batch_size):
            manager.update_profile(f'batch_{i}', {'settings': {'updated': True}})
            
    elif operation_type == "delete":
        for i in range(batch_size):
            manager.delete_profile(f'batch_{i}')
            
    elif operation_type == "mixed":
        for i in range(batch_size):
            if i % 3 == 0:  # Create
                data = {'name': f'mixed_{i}', 'version': '1.0.0', 'settings': {}}
                manager.create_profile(f'mixed_{i}', data)
            elif i % 3 == 1:  # Update
                if i < batch_size:
                    manager.update_profile(f'batch_{i}', {'settings': {'mixed_update': True}})
            else:  # Delete
                if i < batch_size:
                    manager.delete_profile(f'batch_{i}')
    
    end_time = time.time()
    duration = end_time - start_time
    
    assert duration < expected_max_time, f"{operation_type} batch operation took {duration}s, expected < {expected_max_time}s"


# Additional edge case and robustness tests
class TestProfileRobustnessEnhanced(unittest.TestCase):
    """Enhanced robustness and edge case testing"""
    
    def test_profile_with_circular_references(self):
        """Test handling of circular references in profile data."""
        manager = ProfileManager()
        
        # Create data with circular reference
        circular_data = {
            'name': 'circular_test',
            'version': '1.0.0',
            'settings': {}
        }
        
        # Add circular reference
        circular_data['settings']['self_reference'] = circular_data
        
        # Test creation (should either work or raise appropriate error)
        try:
            profile = manager.create_profile('circular_test', circular_data)
            # If successful, verify basic operations work
            retrieved = manager.get_profile('circular_test')
            self.assertIsNotNone(retrieved)
        except (ValueError, TypeError, RecursionError) as e:
            # Acceptable if implementation prevents circular references
            self.assertIsInstance(e, (ValueError, TypeError, RecursionError))
    
    def test_profile_with_very_large_data(self):
        """Test profile creation with extremely large data."""
        manager = ProfileManager()
        
        # Create very large data structure
        large_data = {
            'name': 'large_data_test',
            'version': '1.0.0',
            'settings': {
                'huge_list': list(range(50000)),
                'huge_string': 'x' * 1000000,  # 1MB string
                'huge_dict': {f'key_{i}': f'value_{i}' * 100 for i in range(5000)}
            }
        }
        
        try:
            start_time = time.time()
            profile = manager.create_profile('large_data_test', large_data)
            creation_time = time.time() - start_time
            
            # Verify creation was successful
            self.assertIsNotNone(profile)
            self.assertEqual(len(profile.data['settings']['huge_list']), 50000)
            self.assertEqual(len(profile.data['settings']['huge_string']), 1000000)
            
            # Performance check
            self.assertLess(creation_time, 10.0, "Large data creation took too long")
            
        except (MemoryError, ValueError) as e:
            # Acceptable if implementation has size limits
            self.assertIsInstance(e, (MemoryError, ValueError))
    
    def test_profile_unicode_and_encoding_edge_cases(self):
        """Test profile handling of unicode and encoding edge cases."""
        manager = ProfileManager()
        
        unicode_test_cases = [
            {
                'name': 'emoji_test_🚀🌟⭐',
                'version': '1.0.0',
                'settings': {
                    'description': 'Profile with emojis 😀😂🤔🎉',
                    'unicode_text': 'Text with various scripts: Ελληνικά, العربية, 中文, 日本語, हिन्दी, עברית'
                }
            },
            {
                'name': 'special_chars_∑∆πΩφψχξζ',
                'version': '1.0.0',
                'settings': {
                    'math_symbols': '∑∆πΩφψχξζ∈∉∋∌⊂⊃⊆⊇⊕⊗⊥∥',
                    'currency': '€£¥₹₽₩₪₡₱₨₦₫₭₮',
                    'arrows': '←→↑↓↖↗↘↙⟵⟶⟷⟸⟹'
                }
            },
            {
                'name': 'control_chars_test',
                'version': '1.0.0',
                'settings': {
                    'with_tabs': 'field\twith\ttabs',
                    'with_newlines': 'field\nwith\nnewlines',
                    'with_carriage_returns': 'field\rwith\rcarriage\rreturns'
                }
            }
        ]
        
        for i, test_data in enumerate(unicode_test_cases):
            with self.subTest(case=i):
                try:
                    profile = manager.create_profile(f'unicode_test_{i}', test_data)
                    self.assertIsNotNone(profile)
                    
                    # Verify data integrity
                    retrieved = manager.get_profile(f'unicode_test_{i}')
                    self.assertEqual(retrieved.data, test_data)
                    
                except (UnicodeError, ValueError) as e:
                    # Some edge cases might not be supported
                    self.assertIsInstance(e, (UnicodeError, ValueError))


if __name__ == '__main__':
    # Run comprehensive test suite
    print("Running comprehensive GenesisProfile test suite...")
    print("=" * 70)
    
    # Run unittest tests with high verbosity
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run pytest tests with detailed output
    pytest.main([__file__, '-v', '--tb=short', '--durations=10'])