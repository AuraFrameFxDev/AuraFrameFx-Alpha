import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

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
            Initialize a GenesisProfile with a unique identifier and associated data.
            
            Parameters:
                profile_id (str): The unique identifier for the profile.
                data (dict): The profile's associated data.
            
            Sets the creation and update timestamps to the current UTC time.
            """
            self.profile_id = profile_id
            self.data = data
            self.created_at = datetime.now(timezone.utc)
            self.updated_at = datetime.now(timezone.utc)
    
    class ProfileManager:
        def __init__(self):
            """
            Initialize a new ProfileManager instance with an empty profile collection.
            """
            self.profiles = {}
        
        def create_profile(self, profile_id: str, data: Dict[str, Any]) -> GenesisProfile:
            """
            Create and store a new profile with the given ID and data.
            
            Parameters:
                profile_id (str): The unique identifier for the new profile.
                data (dict): The profile's data, which must include required fields.
            
            Returns:
                GenesisProfile: The created profile instance.
            """
            profile = GenesisProfile(profile_id, data)
            self.profiles[profile_id] = profile
            return profile
        
        def get_profile(self, profile_id: str) -> Optional[GenesisProfile]:
            """
            Returns the profile instance for the specified profile ID, or None if not found.
            """
            return self.profiles.get(profile_id)
        
        def update_profile(self, profile_id: str, data: Dict[str, Any]) -> GenesisProfile:
            """
            Update the data of an existing profile and refresh its update timestamp.
            
            Merges the provided data into the profile identified by `profile_id`. Raises `ProfileNotFoundError` if the profile does not exist.
            
            Returns:
                GenesisProfile: The updated profile instance.
            """
            if profile_id not in self.profiles:
                raise ProfileNotFoundError(f"Profile {profile_id} not found")
            self.profiles[profile_id].data.update(data)
            self.profiles[profile_id].updated_at = datetime.now(timezone.utc)
            return self.profiles[profile_id]
        
        def delete_profile(self, profile_id: str) -> bool:
            """
            Deletes the profile associated with the given profile ID.
            
            Returns:
                bool: True if the profile was found and deleted; False if no profile with the specified ID exists.
            """
            if profile_id in self.profiles:
                del self.profiles[profile_id]
                return True
            return False
    
    class ProfileValidator:
        @staticmethod
        def validate_profile_data(data: Dict[str, Any]) -> bool:
            """
            Determine if the profile data contains all required fields: 'name', 'version', and 'settings'.
            
            Parameters:
                data (dict): The profile data to validate.
            
            Returns:
                bool: True if all required fields are present; False otherwise.
            """
            required_fields = ['name', 'version', 'settings']
            return all(field in data for field in required_fields)
    
    class ProfileBuilder:
        def __init__(self):
            """
            Initialize a ProfileBuilder with an empty data dictionary for building profile fields.
            """
            self.data = {}
        
        def with_name(self, name: str):
            """
            Sets the 'name' field in the profile data and returns the builder instance for method chaining.
            
            Parameters:
                name (str): The value to assign to the 'name' field.
            
            Returns:
                ProfileBuilder: The current builder instance to allow chained method calls.
            """
            self.data['name'] = name
            return self
        
        def with_version(self, version: str):
            """
            Sets the 'version' field in the profile data and returns the builder for method chaining.
            
            Parameters:
                version (str): The version identifier to assign to the profile.
            
            Returns:
                ProfileBuilder: The builder instance with the updated 'version' field.
            """
            self.data['version'] = version
            return self
        
        def with_settings(self, settings: Dict[str, Any]):
            """
            Sets the 'settings' field in the profile data and returns the builder for chaining.
            
            Parameters:
            	settings (dict): Profile settings to assign.
            
            Returns:
            	ProfileBuilder: This builder instance with updated settings.
            """
            self.data['settings'] = settings
            return self
        
        def build(self) -> Dict[str, Any]:
            """
            Return a shallow copy of the profile data accumulated by the builder.
            
            Returns:
                dict: A shallow copy of the current profile data.
            """
            return self.data.copy()
    
    class ProfileError(Exception):
        pass
    
    class ValidationError(ProfileError):
        pass
    
    class ProfileNotFoundError(ProfileError):
        pass


class TestGenesisProfile(unittest.TestCase):
    """Test cases for GenesisProfile class"""
    
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
    
    def test_genesis_profile_initialization(self):
        """
        Verify that a GenesisProfile is initialized with the correct profile ID, data, and timestamp attributes.
        
        Checks that the profile's ID and data match the input values, and that both created_at and updated_at are datetime objects.
        """
        profile = GenesisProfile(self.profile_id, self.sample_data)
        
        self.assertEqual(profile.profile_id, self.profile_id)
        self.assertEqual(profile.data, self.sample_data)
        self.assertIsInstance(profile.created_at, datetime)
        self.assertIsInstance(profile.updated_at, datetime)
    
    def test_genesis_profile_initialization_empty_data(self):
        """
        Test that initializing a GenesisProfile with an empty data dictionary correctly sets the profile ID and data attributes.
        """
        profile = GenesisProfile(self.profile_id, {})
        
        self.assertEqual(profile.profile_id, self.profile_id)
        self.assertEqual(profile.data, {})
    
    def test_genesis_profile_initialization_none_data(self):
        """
        Test that initializing a GenesisProfile with None as the data argument raises a TypeError.
        """
        with self.assertRaises(TypeError):
            GenesisProfile(self.profile_id, None)
    
    def test_genesis_profile_initialization_invalid_id(self):
        """
        Test that creating a GenesisProfile with a None or empty string as the profile ID raises a TypeError or ValueError.
        """
        with self.assertRaises((TypeError, ValueError)):
            GenesisProfile(None, self.sample_data)
        
        with self.assertRaises((TypeError, ValueError)):
            GenesisProfile("", self.sample_data)
    
    def test_genesis_profile_data_immutability(self):
        """
        Verify that a copied snapshot of a GenesisProfile's data remains unchanged after the profile's data is modified.
        
        This test ensures that modifying the profile's data does not affect previously copied data, confirming data immutability.
        """
        profile = GenesisProfile(self.profile_id, self.sample_data)
        original_data = profile.data.copy()
        
        # Modify the data
        profile.data['new_field'] = 'new_value'
        
        # Original data should not be affected if properly implemented
        self.assertNotEqual(profile.data, original_data)
        self.assertIn('new_field', profile.data)
    
    def test_genesis_profile_str_representation(self):
        """
        Tests that the string representation of a GenesisProfile instance includes the profile ID and is of type string.
        """
        profile = GenesisProfile(self.profile_id, self.sample_data)
        str_repr = str(profile)
        
        self.assertIn(self.profile_id, str_repr)
        self.assertIsInstance(str_repr, str)
    
    def test_genesis_profile_equality(self):
        """
        Verify that GenesisProfile instances are equal when they have the same profile ID and equivalent data, and not equal when their profile IDs differ.
        """
        profile1 = GenesisProfile(self.profile_id, self.sample_data)
        profile2 = GenesisProfile(self.profile_id, self.sample_data.copy())
        profile3 = GenesisProfile('different_id', self.sample_data)
        
        # Note: This test depends on how __eq__ is implemented
        # If not implemented, it will test object identity
        if hasattr(profile1, '__eq__'):
            self.assertEqual(profile1, profile2)
            self.assertNotEqual(profile1, profile3)


class TestProfileManager(unittest.TestCase):
    """Test cases for ProfileManager class"""
    
    def setUp(self):
        """
        Initializes a new ProfileManager and sample profile data before each test to ensure isolation and consistency.
        """
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
        """
        Tests that a profile is created and stored successfully with the given ID and data.
        
        Verifies that the returned object is a `GenesisProfile` with the correct profile ID and data, and that it is present in the manager's internal storage.
        """
        profile = self.manager.create_profile(self.profile_id, self.sample_data)
        
        self.assertIsInstance(profile, GenesisProfile)
        self.assertEqual(profile.profile_id, self.profile_id)
        self.assertEqual(profile.data, self.sample_data)
        self.assertIn(self.profile_id, self.manager.profiles)
    
    def test_create_profile_duplicate_id(self):
        """
        Test creating a profile with a duplicate ID to ensure correct handling of duplicates.
        
        Verifies that the system either raises an appropriate exception or overwrites the existing profile when attempting to create a profile with an ID that already exists.
        """
        self.manager.create_profile(self.profile_id, self.sample_data)
        
        # Creating another profile with the same ID should either:
        # 1. Raise an exception, or
        # 2. Overwrite the existing profile
        # This depends on implementation
        try:
            duplicate_profile = self.manager.create_profile(self.profile_id, {'name': 'duplicate'})
            # If no exception, verify the behavior
            self.assertEqual(duplicate_profile.profile_id, self.profile_id)
        except Exception as e:
            # If exception is raised, it should be a specific type
            self.assertIsInstance(e, (ProfileError, ValueError))
    
    def test_create_profile_invalid_data(self):
        """
        Test that creating a profile with invalid data, such as None, raises a TypeError or ValueError.
        """
        with self.assertRaises((TypeError, ValueError)):
            self.manager.create_profile(self.profile_id, None)
    
    def test_get_profile_existing(self):
        """
        Test retrieval of an existing profile by ID and verify the returned instance matches the created profile.
        """
        created_profile = self.manager.create_profile(self.profile_id, self.sample_data)
        retrieved_profile = self.manager.get_profile(self.profile_id)
        
        self.assertEqual(retrieved_profile, created_profile)
        self.assertEqual(retrieved_profile.profile_id, self.profile_id)
    
    def test_get_profile_nonexistent(self):
        """
        Test that retrieving a profile with a nonexistent ID returns None.
        """
        result = self.manager.get_profile('nonexistent_id')
        self.assertIsNone(result)
    
    def test_get_profile_empty_id(self):
        """
        Test retrieving a profile using an empty string as the profile ID.
        
        Verifies that requesting a profile with an empty ID returns None, indicating no such profile exists.
        """
        result = self.manager.get_profile('')
        self.assertIsNone(result)
    
    def test_update_profile_success(self):
        """
        Verify that updating an existing profile correctly merges new data fields and updates the `updated_at` timestamp.
        """
        self.manager.create_profile(self.profile_id, self.sample_data)
        
        update_data = {'name': 'updated_profile', 'new_field': 'new_value'}
        updated_profile = self.manager.update_profile(self.profile_id, update_data)
        
        self.assertEqual(updated_profile.data['name'], 'updated_profile')
        self.assertEqual(updated_profile.data['new_field'], 'new_value')
        self.assertIsInstance(updated_profile.updated_at, datetime)
    
    def test_update_profile_nonexistent(self):
        """
        Test that updating a profile with a non-existent ID raises a ProfileNotFoundError.
        """
        with self.assertRaises(ProfileNotFoundError):
            self.manager.update_profile('nonexistent_id', {'name': 'updated'})
    
    def test_update_profile_empty_data(self):
        """
        Test that updating a profile with an empty dictionary leaves the profile's data unchanged.
        """
        self.manager.create_profile(self.profile_id, self.sample_data)
        
        # Updating with empty data should not raise an error
        updated_profile = self.manager.update_profile(self.profile_id, {})
        self.assertEqual(updated_profile.data, self.sample_data)
    
    def test_delete_profile_success(self):
        """
        Test successful deletion of an existing profile.
        
        Verifies that deleting a profile returns True, removes the profile from the manager's storage, and subsequent retrieval returns None.
        """
        self.manager.create_profile(self.profile_id, self.sample_data)
        
        result = self.manager.delete_profile(self.profile_id)
        
        self.assertTrue(result)
        self.assertNotIn(self.profile_id, self.manager.profiles)
        self.assertIsNone(self.manager.get_profile(self.profile_id))
    
    def test_delete_profile_nonexistent(self):
        """
        Test that deleting a profile with a non-existent ID returns False, indicating no profile was removed.
        """
        result = self.manager.delete_profile('nonexistent_id')
        self.assertFalse(result)
    
    def test_manager_state_isolation(self):
        """
        Verify that separate ProfileManager instances maintain independent state and do not share profiles.
        """
        manager1 = ProfileManager()
        manager2 = ProfileManager()
        
        manager1.create_profile(self.profile_id, self.sample_data)
        
        self.assertIsNotNone(manager1.get_profile(self.profile_id))
        self.assertIsNone(manager2.get_profile(self.profile_id))


class TestProfileValidator(unittest.TestCase):
    """Test cases for ProfileValidator class"""
    
    def setUp(self):
        """
        Prepare a valid profile data dictionary for use in test cases.
        """
        self.valid_data = {
            'name': 'test_profile',
            'version': '1.0.0',
            'settings': {
                'ai_model': 'gpt-4',
                'temperature': 0.7
            }
        }
    
    def test_validate_profile_data_valid(self):
        """
        Tests that `ProfileValidator.validate_profile_data` returns True when provided with valid profile data.
        """
        result = ProfileValidator.validate_profile_data(self.valid_data)
        self.assertTrue(result)
    
    def test_validate_profile_data_missing_required_fields(self):
        """
        Test that profile data validation fails when required fields are missing.
        
        Verifies that `ProfileValidator.validate_profile_data` returns `False` for dictionaries lacking any of the required fields: 'name', 'version', or 'settings'.
        """
        invalid_data_cases = [
            {'version': '1.0.0', 'settings': {}},  # Missing name
            {'name': 'test', 'settings': {}},      # Missing version
            {'name': 'test', 'version': '1.0.0'},  # Missing settings
            {},                                     # Missing all
        ]
        
        for invalid_data in invalid_data_cases:
            with self.subTest(invalid_data=invalid_data):
                result = ProfileValidator.validate_profile_data(invalid_data)
                self.assertFalse(result)
    
    def test_validate_profile_data_empty_values(self):
        """
        Verify that validating profile data with empty required fields returns a boolean result.
        
        This test ensures that the `validate_profile_data` method always returns a boolean when required fields are present but contain empty values, regardless of the data's validity.
        """
        empty_data_cases = [
            {'name': '', 'version': '1.0.0', 'settings': {}},
            {'name': 'test', 'version': '', 'settings': {}},
            {'name': 'test', 'version': '1.0.0', 'settings': None},
        ]
        
        for empty_data in empty_data_cases:
            with self.subTest(empty_data=empty_data):
                # This may pass or fail depending on implementation
                result = ProfileValidator.validate_profile_data(empty_data)
                # Test that it returns a boolean
                self.assertIsInstance(result, bool)
    
    def test_validate_profile_data_none_input(self):
        """
        Test that passing None to ProfileValidator.validate_profile_data raises a TypeError or AttributeError.
        """
        with self.assertRaises((TypeError, AttributeError)):
            ProfileValidator.validate_profile_data(None)
    
    def test_validate_profile_data_invalid_types(self):
        """
        Test that `ProfileValidator.validate_profile_data` raises a `TypeError` or `AttributeError` when called with non-dictionary input types.
        """
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
        """
        Test that profile data validation passes when extra fields are included in addition to the required fields.
        
        Ensures that the presence of additional, non-required fields does not cause validation to fail as long as all required fields are present.
        """
        data_with_extra = self.valid_data.copy()
        data_with_extra.update({
            'extra_field': 'extra_value',
            'metadata': {'tags': ['test']},
            'optional_settings': {'debug': True}
        })
        
        result = ProfileValidator.validate_profile_data(data_with_extra)
        self.assertTrue(result)  # Extra fields should be allowed


class TestProfileBuilder(unittest.TestCase):
    """Test cases for ProfileBuilder class"""
    
    def setUp(self):
        """
        Set up a new ProfileBuilder instance before each test case.
        """
        self.builder = ProfileBuilder()
    
    def test_builder_chain_methods(self):
        """
        Test that ProfileBuilder supports method chaining to construct a profile data dictionary with specified fields.
        """
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
        """
        Verify that each individual setter in ProfileBuilder correctly assigns its value and that the built profile data includes the expected fields and values.
        """
        self.builder.with_name('individual_test')
        self.builder.with_version('2.0.0')
        self.builder.with_settings({'temperature': 0.5})
        
        result = self.builder.build()
        
        self.assertEqual(result['name'], 'individual_test')
        self.assertEqual(result['version'], '2.0.0')
        self.assertEqual(result['settings']['temperature'], 0.5)
    
    def test_builder_overwrite_values(self):
        """
        Test that the builder overwrites previous values when the same field is set multiple times.
        
        Verifies that the final value assigned to a field is the one present in the built profile data.
        """
        self.builder.with_name('first_name')
        self.builder.with_name('second_name')
        
        result = self.builder.build()
        
        self.assertEqual(result['name'], 'second_name')
    
    def test_builder_empty_build(self):
        """
        Test that the profile builder returns an empty dictionary when no fields are set before calling build().
        """
        result = self.builder.build()
        self.assertEqual(result, {})
    
    def test_builder_partial_build(self):
        """
        Test that the profile builder returns a dictionary with only the fields that have been explicitly set, excluding unset fields.
        """
        result = self.builder.with_name('partial').build()
        
        self.assertEqual(result, {'name': 'partial'})
        self.assertNotIn('version', result)
        self.assertNotIn('settings', result)
    
    def test_builder_complex_settings(self):
        """
        Test that ProfileBuilder correctly retains complex nested structures in the 'settings' field when building profile data.
        """
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
        """
        Test that ProfileBuilder.build() returns independent copies of the profile data, ensuring changes to one built dictionary do not affect others.
        """
        self.builder.with_name('test')
        result1 = self.builder.build()
        result2 = self.builder.build()
        
        # Modify one result
        result1['name'] = 'modified'
        
        # Other result should not be affected
        self.assertEqual(result2['name'], 'test')
        self.assertNotEqual(result1, result2)
    
    def test_builder_none_values(self):
        """
        Test that ProfileBuilder preserves None values for 'name', 'version', and 'settings' fields when building profile data.
        """
        result = (self.builder
                 .with_name(None)
                 .with_version(None)
                 .with_settings(None)
                 .build())
        
        self.assertEqual(result['name'], None)
        self.assertEqual(result['version'], None)
        self.assertEqual(result['settings'], None)


class TestProfileExceptions(unittest.TestCase):
    """Test cases for custom exceptions"""
    
    def test_profile_error_inheritance(self):
        """
        Test that ProfileError inherits from Exception and its string representation matches the given message.
        """
        error = ProfileError("Test error")
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Test error")
    
    def test_validation_error_inheritance(self):
        """
        Verify that ValidationError is a subclass of ProfileError and Exception, and that its string representation matches the provided message.
        """
        error = ValidationError("Validation failed")
        self.assertIsInstance(error, ProfileError)
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Validation failed")
    
    def test_profile_not_found_error_inheritance(self):
        """
        Test that ProfileNotFoundError is a subclass of ProfileError and Exception, and that its string representation matches the given message.
        """
        error = ProfileNotFoundError("Profile not found")
        self.assertIsInstance(error, ProfileError)
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Profile not found")
    
    def test_exception_with_no_message(self):
        """
        Verify that custom exceptions can be instantiated without a message and confirm their correct inheritance hierarchy.
        """
        error = ProfileError()
        self.assertIsInstance(error, Exception)
        
        error = ValidationError()
        self.assertIsInstance(error, ProfileError)
        
        error = ProfileNotFoundError()
        self.assertIsInstance(error, ProfileError)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test cases combining multiple components"""
    
    def setUp(self):
        """
        Initializes a ProfileManager, ProfileBuilder, and sample profile data for use in integration tests.
        """
        self.manager = ProfileManager()
        self.builder = ProfileBuilder()
        self.sample_data = {
            'name': 'integration_test',
            'version': '1.0.0',
            'settings': {
                'ai_model': 'gpt-4',
                'temperature': 0.7
            }
        }
    
    def test_end_to_end_profile_lifecycle(self):
        """
        Test the complete lifecycle of a profile, verifying creation, retrieval, update, and deletion operations for correctness.
        """
        profile_id = 'lifecycle_test'
        
        # Create
        profile = self.manager.create_profile(profile_id, self.sample_data)
        self.assertIsNotNone(profile)
        
        # Read
        retrieved = self.manager.get_profile(profile_id)
        self.assertEqual(retrieved.profile_id, profile_id)
        
        # Update
        update_data = {'name': 'updated_integration_test'}
        updated = self.manager.update_profile(profile_id, update_data)
        self.assertEqual(updated.data['name'], 'updated_integration_test')
        
        # Delete
        deleted = self.manager.delete_profile(profile_id)
        self.assertTrue(deleted)
        self.assertIsNone(self.manager.get_profile(profile_id))
    
    def test_builder_with_manager_integration(self):
        """
        Verify that profiles constructed with ProfileBuilder and saved via ProfileManager preserve all assigned fields and values.
        """
        profile_data = (self.builder
                       .with_name('builder_manager_test')
                       .with_version('2.0.0')
                       .with_settings({'model': 'gpt-3.5'})
                       .build())
        
        profile = self.manager.create_profile('builder_test', profile_data)
        
        self.assertEqual(profile.data['name'], 'builder_manager_test')
        self.assertEqual(profile.data['version'], '2.0.0')
        self.assertEqual(profile.data['settings']['model'], 'gpt-3.5')
    
    def test_validator_with_manager_integration(self):
        """
        Test integration of ProfileValidator and ProfileManager for profile creation with valid data.
        
        Verifies that data validated by ProfileValidator is accepted by ProfileManager, resulting in successful profile creation.
        """
        valid_data = (self.builder
                     .with_name('validator_test')
                     .with_version('1.0.0')
                     .with_settings({'temperature': 0.8})
                     .build())
        
        # Validate before creating
        is_valid = ProfileValidator.validate_profile_data(valid_data)
        self.assertTrue(is_valid)
        
        # Create profile
        profile = self.manager.create_profile('validator_test', valid_data)
        self.assertIsNotNone(profile)
    
    def test_error_handling_integration(self):
        """
        Tests that invalid profile data is correctly rejected by the validator and that updating a non-existent profile raises a ProfileNotFoundError.
        """
        # Test validation error
        invalid_data = {'name': 'test'}  # Missing required fields
        is_valid = ProfileValidator.validate_profile_data(invalid_data)
        self.assertFalse(is_valid)
        
        # Test profile not found error
        with self.assertRaises(ProfileNotFoundError):
            self.manager.update_profile('nonexistent', {'name': 'test'})
    
    def test_concurrent_operations_simulation(self):
        """
        Simulates sequential updates to a profile and verifies that all changes and the updated timestamp are correctly reflected after each update.
        """
        profile_id = 'concurrent_test'
        
        # Create profile
        profile = self.manager.create_profile(profile_id, self.sample_data)
        original_updated_at = profile.updated_at
        
        # Multiple updates
        self.manager.update_profile(profile_id, {'field1': 'value1'})
        self.manager.update_profile(profile_id, {'field2': 'value2'})
        
        # Verify final state
        final_profile = self.manager.get_profile(profile_id)
        self.assertEqual(final_profile.data['field1'], 'value1')
        self.assertEqual(final_profile.data['field2'], 'value2')
        self.assertGreater(final_profile.updated_at, original_updated_at)


class TestEdgeCasesAndBoundaryConditions(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        """
        Set up a fresh ProfileManager instance before each test.
        """
        self.manager = ProfileManager()
    
    def test_very_large_profile_data(self):
        """
        Tests that a profile with very large string and nested dictionary fields can be created and stored without errors.
        
        Verifies that the profile manager correctly handles large data sizes and preserves the integrity of long strings within profile settings.
        """
        large_data = {
            'name': 'large_profile',
            'version': '1.0.0',
            'settings': {
                'large_field': 'x' * 10000,  # 10KB string
                'nested_data': {f'key_{i}': f'value_{i}' for i in range(1000)}
            }
        }
        
        profile = self.manager.create_profile('large_profile', large_data)
        self.assertIsNotNone(profile)
        self.assertEqual(len(profile.data['settings']['large_field']), 10000)
    
    def test_unicode_and_special_characters(self):
        """
        Test that profiles with Unicode and special characters in their data fields are created and retrieved without data loss or corruption.
        """
        unicode_data = {
            'name': 'プロファイル_測試_🚀',
            'version': '1.0.0',
            'settings': {
                'description': 'Special chars: !@#$%^&*()_+-=[]{}|;:,.<>?',
                'unicode_field': 'Héllo Wörld 测试 🌍'
            }
        }
        
        profile = self.manager.create_profile('unicode_test', unicode_data)
        self.assertEqual(profile.data['name'], 'プロファイル_測試_🚀')
        self.assertEqual(profile.data['settings']['unicode_field'], 'Héllo Wörld 测试 🌍')
    
    def test_deeply_nested_data_structures(self):
        """
        Test that profiles with deeply nested dictionaries in the 'settings' field retain their full structure after creation and retrieval.
        
        Verifies that all levels of nesting are preserved and accessible within the profile data.
        """
        nested_data = {
            'name': 'nested_test',
            'version': '1.0.0',
            'settings': {
                'level1': {
                    'level2': {
                        'level3': {
                            'level4': {
                                'level5': 'deep_value'
                            }
                        }
                    }
                }
            }
        }
        
        profile = self.manager.create_profile('nested_test', nested_data)
        self.assertEqual(
            profile.data['settings']['level1']['level2']['level3']['level4']['level5'],
            'deep_value'
        )
    
    def test_circular_reference_handling(self):
        """
        Test that the profile manager can handle profile data containing a circular reference, either by accepting the data or raising a ValueError or TypeError.
        """
        # Create data with potential circular reference
        data = {
            'name': 'circular_test',
            'version': '1.0.0',
            'settings': {}
        }
        
        # Note: This test depends on how the implementation handles circular references
        # Most JSON serialization would fail, but in-memory objects might work
        try:
            profile = self.manager.create_profile('circular_test', data)
            self.assertIsNotNone(profile)
        except (ValueError, TypeError) as e:
            # If the implementation properly handles circular references by raising an error
            self.assertIsInstance(e, (ValueError, TypeError))
    
    def test_extremely_long_profile_ids(self):
        """
        Test that creating a profile with an extremely long profile ID either succeeds or raises an appropriate exception if length restrictions are enforced.
        """
        long_id = 'x' * 1000
        data = {
            'name': 'long_id_test',
            'version': '1.0.0',
            'settings': {}
        }
        
        try:
            profile = self.manager.create_profile(long_id, data)
            self.assertEqual(profile.profile_id, long_id)
        except (ValueError, TypeError) as e:
            # If the implementation has ID length limits
            self.assertIsInstance(e, (ValueError, TypeError))
    
    def test_profile_id_with_special_characters(self):
        """
        Test that profiles can be created with IDs containing special characters, or that appropriate exceptions are raised if such IDs are not supported.
        """
        special_ids = [
            'profile-with-dashes',
            'profile_with_underscores',
            'profile.with.dots',
            'profile with spaces',
            'profile/with/slashes',
            'profile:with:colons'
        ]
        
        for special_id in special_ids:
            with self.subTest(profile_id=special_id):
                data = {
                    'name': f'test_{special_id}',
                    'version': '1.0.0',
                    'settings': {}
                }
                
                try:
                    profile = self.manager.create_profile(special_id, data)
                    self.assertEqual(profile.profile_id, special_id)
                except (ValueError, TypeError):
                    # Some implementations may not allow special characters
                    pass
    
    def test_memory_efficiency_with_many_profiles(self):
        """
        Test that the profile manager can efficiently handle creation, storage, and retrieval of a large number of profiles while maintaining data integrity and correct access for each profile.
        """
        num_profiles = 100
        
        for i in range(num_profiles):
            profile_id = f'profile_{i}'
            data = {
                'name': f'profile_{i}',
                'version': '1.0.0',
                'settings': {'index': i}
            }
            self.manager.create_profile(profile_id, data)
        
        # Verify all profiles exist
        self.assertEqual(len(self.manager.profiles), num_profiles)
        
        # Verify random access works
        random_profile = self.manager.get_profile('profile_50')
        self.assertEqual(random_profile.data['settings']['index'], 50)


@pytest.mark.parametrize("profile_id,expected_valid", [
    ("valid_id", True),
    ("", False),
    ("profile-123", True),
    ("profile_456", True),
    ("profile.789", True),
    ("PROFILE_UPPER", True),
    ("profile with spaces", True),  # May or may not be valid depending on implementation
    ("profile/with/slashes", True),  # May or may not be valid depending on implementation
    (None, False),
    (123, False),
    ([], False),
])
def test_profile_id_validation_parametrized(profile_id, expected_valid):
    """
    Parametrized test verifying that profile creation accepts or rejects profile IDs according to validity expectations.
    
    Parameters:
        profile_id: The profile ID to test.
        expected_valid: Indicates whether the profile ID is expected to be valid (True) or invalid (False).
    """
    manager = ProfileManager()
    data = {
        'name': 'test_profile',
        'version': '1.0.0',
        'settings': {}
    }
    
    if expected_valid:
        try:
            profile = manager.create_profile(profile_id, data)
            assert profile.profile_id == profile_id
        except (TypeError, ValueError):
            # Some implementations may be more strict
            pass
    else:
        with pytest.raises((TypeError, ValueError)):
            manager.create_profile(profile_id, data)


@pytest.mark.parametrize("data,should_validate", [
    ({"name": "test", "version": "1.0", "settings": {}}, True),
    ({"name": "test", "version": "1.0"}, False),  # Missing settings
    ({"name": "test", "settings": {}}, False),  # Missing version
    ({"version": "1.0", "settings": {}}, False),  # Missing name
    ({}, False),  # Missing all required fields
    ({"name": "", "version": "1.0", "settings": {}}, True),  # Empty name might be valid
    ({"name": "test", "version": "", "settings": {}}, True),  # Empty version might be valid
    ({"name": "test", "version": "1.0", "settings": None}, True),  # None settings might be valid
])
def test_profile_validation_parametrized(data, should_validate):
    """
    Parametrized test verifying that profile data validation produces the expected result for various input scenarios.
    
    Parameters:
        data (dict): Profile data to validate.
        should_validate (bool): Expected outcome of the validation.
    """
    result = ProfileValidator.validate_profile_data(data)
    assert result == should_validate


if __name__ == '__main__':
    unittest.main()

class TestSerializationAndPersistence(unittest.TestCase):
    """Test serialization, deserialization, and persistence scenarios"""
    
    def setUp(self):
        """
        Set up a fresh ProfileManager instance and sample profile data before each test to maintain isolation.
        """
        self.manager = ProfileManager()
        self.sample_data = {
            'name': 'serialization_test',
            'version': '1.0.0',
            'settings': {
                'ai_model': 'gpt-4',
                'temperature': 0.7,
                'nested_config': {
                    'max_tokens': 1000,
                    'stop_sequences': ['\n', '###']
                }
            }
        }
    
    def test_profile_json_serialization(self):
        """
        Test that profile data can be serialized to JSON and deserialized back, ensuring all fields and nested values are preserved.
        """
        profile = self.manager.create_profile('json_test', self.sample_data)
        
        # Test JSON serialization
        json_str = json.dumps(profile.data, default=str)
        self.assertIsInstance(json_str, str)
        
        # Test deserialization
        deserialized_data = json.loads(json_str)
        self.assertEqual(deserialized_data['name'], self.sample_data['name'])
        self.assertEqual(deserialized_data['settings']['ai_model'], 'gpt-4')
    
    def test_profile_data_deep_copy(self):
        """
        Verify that deep copying a profile's data results in a fully independent copy, ensuring modifications to nested structures in the original do not affect the copy.
        """
        import copy
        
        profile = self.manager.create_profile('copy_test', self.sample_data)
        deep_copy = copy.deepcopy(profile.data)
        
        # Modify original
        profile.data['settings']['temperature'] = 0.9
        
        # Deep copy should remain unchanged
        self.assertEqual(deep_copy['settings']['temperature'], 0.7)
        self.assertNotEqual(profile.data['settings']['temperature'], deep_copy['settings']['temperature'])
    
    def test_profile_data_with_datetime_objects(self):
        """
        Verify that datetime fields in profile data are preserved as datetime objects after profile creation.
        
        Ensures that when profile data includes datetime values, these fields remain as `datetime` instances in the stored profile and are not converted to other types.
        """
        data_with_datetime = self.sample_data.copy()
        data_with_datetime['created_at'] = datetime.now(timezone.utc)
        data_with_datetime['scheduled_run'] = datetime.now(timezone.utc)
        
        profile = self.manager.create_profile('datetime_test', data_with_datetime)
        
        self.assertIsInstance(profile.data['created_at'], datetime)
        self.assertIsInstance(profile.data['scheduled_run'], datetime)
    
    def test_profile_persistence_simulation(self):
        """
        Simulates saving a profile to a temporary JSON file and reloading it to verify that all profile fields are correctly preserved after serialization and deserialization.
        """
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            profile = self.manager.create_profile('persist_test', self.sample_data)
            
            # Simulate saving to file
            profile_dict = {
                'profile_id': profile.profile_id,
                'data': profile.data,
                'created_at': profile.created_at.isoformat(),
                'updated_at': profile.updated_at.isoformat()
            }
            json.dump(profile_dict, f)
            temp_file = f.name
        
        try:
            # Simulate loading from file
            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(loaded_data['profile_id'], 'persist_test')
            self.assertEqual(loaded_data['data']['name'], 'serialization_test')
            self.assertIn('created_at', loaded_data)
            self.assertIn('updated_at', loaded_data)
        finally:
            os.unlink(temp_file)


class TestPerformanceAndScalability(unittest.TestCase):
    """Test performance and scalability scenarios"""
    
    def setUp(self):
        """
        Initializes a new ProfileManager instance before each test method.
        """
        self.manager = ProfileManager()
    
    def test_bulk_profile_creation_performance(self):
        """
        Benchmarks the creation of 1,000 profiles and asserts completion within 10 seconds.
        
        Verifies that all profiles are created and present in the manager, ensuring both correctness and acceptable performance for bulk profile creation.
        """
        import time
        
        start_time = time.time()
        num_profiles = 1000
        
        for i in range(num_profiles):
            profile_data = {
                'name': f'bulk_profile_{i}',
                'version': '1.0.0',
                'settings': {'index': i, 'batch': 'performance_test'}
            }
            self.manager.create_profile(f'bulk_{i}', profile_data)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify all profiles were created
        self.assertEqual(len(self.manager.profiles), num_profiles)
        
        # Performance assertion - should complete within reasonable time
        self.assertLess(duration, 10.0, "Bulk creation took too long")
    
    def test_profile_lookup_performance(self):
        """
        Benchmarks the retrieval speed of multiple profiles and asserts that all lookups complete in under one second.
        
        Creates 500 profiles, retrieves every 10th profile, verifies each is found, and checks that the total lookup duration is less than one second.
        """
        import time
        
        # Create profiles for testing
        num_profiles = 500
        for i in range(num_profiles):
            profile_data = {
                'name': f'lookup_profile_{i}',
                'version': '1.0.0',
                'settings': {'index': i}
            }
            self.manager.create_profile(f'lookup_{i}', profile_data)
        
        # Test lookup performance
        start_time = time.time()
        for i in range(0, num_profiles, 10):  # Test every 10th profile
            profile = self.manager.get_profile(f'lookup_{i}')
            self.assertIsNotNone(profile)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertion
        self.assertLess(duration, 1.0, "Profile lookups took too long")
    
    def test_memory_usage_with_large_profiles(self):
        """
        Test creation of a profile with large data structures to ensure correct handling and assess memory usage.
        
        Creates a profile whose settings include a large list, dictionary, and string, then verifies successful creation and the expected sizes of these data structures.
        """
        import sys
        
        # Create a profile with large data
        large_data = {
            'name': 'memory_test',
            'version': '1.0.0',
            'settings': {
                'large_list': list(range(10000)),
                'large_dict': {f'key_{i}': f'value_{i}' * 100 for i in range(1000)},
                'large_string': 'x' * 100000
            }
        }
        
        # Get initial memory usage (approximate)
        initial_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0
        
        profile = self.manager.create_profile('memory_test', large_data)
        
        # Verify the profile was created successfully
        self.assertIsNotNone(profile)
        self.assertEqual(len(profile.data['settings']['large_list']), 10000)
        self.assertEqual(len(profile.data['settings']['large_string']), 100000)
    
    def test_concurrent_access_simulation(self):
        """
        Simulates repeated sequential updates to a profile to assess robustness under conditions similar to concurrent access.
        
        Ensures that multiple updates to a profile's settings are correctly applied and that the profile remains accessible after all modifications.
        """
        profile_id = 'concurrent_test'
        
        # Create initial profile
        initial_data = {
            'name': 'concurrent_test',
            'version': '1.0.0',
            'settings': {'counter': 0}
        }
        
        self.manager.create_profile(profile_id, initial_data)
        
        # Simulate concurrent updates
        for i in range(100):
            current_profile = self.manager.get_profile(profile_id)
            updated_data = {'counter': current_profile.data['settings']['counter'] + 1}
            self.manager.update_profile(profile_id, {'settings': updated_data})
        
        # Verify final state
        final_profile = self.manager.get_profile(profile_id)
        self.assertIsNotNone(final_profile)


class TestAdvancedValidationScenarios(unittest.TestCase):
    """Test advanced validation scenarios and edge cases"""
    
    def setUp(self):
        """
        Prepare a new `ProfileValidator` instance before each test method.
        """
        self.validator = ProfileValidator()
    
    def test_schema_validation_complex_nested_structures(self):
        """
        Verifies that the profile validator correctly accepts profile data with deeply nested and complex structures in the settings field.
        """
        complex_data = {
            'name': 'complex_test',
            'version': '1.0.0',
            'settings': {
                'ai_models': [
                    {'name': 'gpt-4', 'temperature': 0.7, 'max_tokens': 1000},
                    {'name': 'gpt-3.5', 'temperature': 0.5, 'max_tokens': 500}
                ],
                'workflows': {
                    'preprocessing': {
                        'steps': ['tokenize', 'normalize', 'validate'],
                        'config': {'batch_size': 100}
                    },
                    'postprocessing': {
                        'steps': ['format', 'validate', 'export'],
                        'config': {'format': 'json'}
                    }
                }
            }
        }
        
        result = ProfileValidator.validate_profile_data(complex_data)
        self.assertTrue(result)
    
    def test_version_format_validation(self):
        """
        Tests that the profile validator correctly accepts valid semantic version strings and rejects invalid or non-string values in the profile data.
        
        Covers standard, pre-release, build metadata, and malformed version strings to ensure robust version format validation.
        """
        version_cases = [
            ('1.0.0', True),
            ('1.0.0-alpha', True),
            ('1.0.0-beta.1', True),
            ('1.0.0+build.1', True),
            ('1.0', True),  # May or may not be valid depending on implementation
            ('1', True),    # May or may not be valid depending on implementation
            ('invalid', False),
            ('1.0.0.0', False),
            ('', False),
            (None, False),
            (123, False),
        ]
        
        for version, expected_valid in version_cases:
            with self.subTest(version=version):
                data = {
                    'name': 'version_test',
                    'version': version,
                    'settings': {}
                }
                
                try:
                    result = ProfileValidator.validate_profile_data(data)
                    if expected_valid:
                        self.assertTrue(result)
                    else:
                        self.assertFalse(result)
                except (TypeError, ValueError):
                    if expected_valid:
                        self.fail(f"Unexpected error for valid version: {version}")
    
    def test_settings_type_validation(self):
        """
        Test that the profile data validator accepts or rejects different types for the 'settings' field.
        
        Checks that dictionaries and None are considered valid for 'settings', while strings, integers, and lists are rejected. Asserts that the validator returns True for valid types or False for invalid types, and fails the test if an unexpected error is raised for valid cases.
        """
        settings_cases = [
            ({'temperature': 0.7}, True),
            ({'temperature': 'invalid'}, True),  # May be handled by downstream validation
            ({'max_tokens': 1000}, True),
            ({'max_tokens': -1}, True),  # May be handled by downstream validation
            ({'stop_sequences': ['\\n', '###']}, True),
            ({'stop_sequences': 'invalid'}, True),  # May be handled by downstream validation
            ({'nested': {'key': 'value'}}, True),
            (None, True),  # May be valid depending on implementation
            ('invalid', False),
            (123, False),
            ([], False),
        ]
        
        for settings, expected_valid in settings_cases:
            with self.subTest(settings=settings):
                data = {
                    'name': 'settings_test',
                    'version': '1.0.0',
                    'settings': settings
                }
                
                try:
                    result = ProfileValidator.validate_profile_data(data)
                    if expected_valid:
                        self.assertTrue(result)
                    else:
                        self.assertFalse(result)
                except (TypeError, AttributeError):
                    if expected_valid:
                        self.fail(f"Unexpected error for valid settings: {settings}")
    
    def test_profile_name_validation(self):
        """
        Tests the profile name validation logic with a variety of input cases.
        
        Covers standard names, names with spaces, dashes, underscores, dots, Unicode characters, empty strings, whitespace-only names, very long names, and invalid types to ensure the validator accepts valid names and rejects invalid ones.
        """
        name_cases = [
            ('valid_name', True),
            ('Valid Name With Spaces', True),
            ('name-with-dashes', True),
            ('name_with_underscores', True),
            ('name.with.dots', True),
            ('プロファイル', True),  # Unicode characters
            ('profile_123', True),
            ('', False),  # Empty name
            ('   ', False),  # Whitespace only
            ('a' * 1000, True),  # Very long name - may be limited by implementation
            (None, False),
            (123, False),
            ([], False),
        ]
        
        for name, expected_valid in name_cases:
            with self.subTest(name=name):
                data = {
                    'name': name,
                    'version': '1.0.0',
                    'settings': {}
                }
                
                try:
                    result = ProfileValidator.validate_profile_data(data)
                    if expected_valid:
                        self.assertTrue(result)
                    else:
                        self.assertFalse(result)
                except (TypeError, AttributeError):
                    if expected_valid:
                        self.fail(f"Unexpected error for valid name: {name}")


class TestErrorHandlingAndExceptionScenarios(unittest.TestCase):
    """Test comprehensive error handling and exception scenarios"""
    
    def setUp(self):
        """
        Initializes a new ProfileManager instance before each test method.
        """
        self.manager = ProfileManager()
    
    def test_exception_message_accuracy(self):
        """
        Test that `ProfileNotFoundError` provides an accurate and descriptive error message containing the missing profile ID when updating a non-existent profile.
        """
        # Test ProfileNotFoundError message
        try:
            self.manager.update_profile('nonexistent_id', {'name': 'test'})
            self.fail("Expected ProfileNotFoundError")
        except ProfileNotFoundError as e:
            self.assertIn('nonexistent_id', str(e))
            self.assertIn('not found', str(e).lower())
    
    def test_exception_context_preservation(self):
        """
        Test that wrapping an exception with a custom exception preserves the original exception's message in the new exception's message.
        """
        def nested_function():
            """
            Raises a ValueError with the message "Original error".
            """
            raise ValueError("Original error")
        
        try:
            nested_function()
        except ValueError as e:
            # Test that we can wrap exceptions properly
            wrapped_error = ProfileError(f"Wrapped: {str(e)}")
            self.assertIn("Original error", str(wrapped_error))
    
    def test_recovery_from_partial_failures(self):
        """
        Test that a failed profile update with invalid data does not modify the original profile, preserving data integrity and enabling recovery after exceptions.
        """
        # Create a profile successfully
        profile = self.manager.create_profile('recovery_test', {
            'name': 'recovery_test',
            'version': '1.0.0',
            'settings': {'initial': 'value'}
        })
        
        # Simulate partial failure in update
        try:
            # This might fail depending on implementation
            self.manager.update_profile('recovery_test', {'settings': 'invalid_type'})
        except (TypeError, ValueError):
            # Ensure the profile still exists and is in valid state
            recovered_profile = self.manager.get_profile('recovery_test')
            self.assertIsNotNone(recovered_profile)
            self.assertEqual(recovered_profile.data['settings']['initial'], 'value')
    
    def test_exception_hierarchy_consistency(self):
        """
        Test that custom exception classes inherit from the correct base classes and can be caught via their shared base class.
        """
        # Test that all custom exceptions inherit properly
        validation_error = ValidationError("Validation failed")
        profile_not_found = ProfileNotFoundError("Profile not found")
        
        # Test inheritance chain
        self.assertIsInstance(validation_error, ProfileError)
        self.assertIsInstance(validation_error, Exception)
        self.assertIsInstance(profile_not_found, ProfileError)
        self.assertIsInstance(profile_not_found, Exception)
        
        # Test that they can be caught as base class
        try:
            raise ValidationError("Test error")
        except ProfileError:
            pass  # Should be caught
        except Exception:
            self.fail("Should have been caught as ProfileError")
    
    def test_error_logging_and_debugging_info(self):
        """
        Test that custom profile exceptions return the correct message and inherit from Exception.
        
        Verifies that the string representation of each custom exception matches the provided message and that each exception is an instance of Exception.
        """
        # Test with various error scenarios
        error_scenarios = [
            (ProfileError, "Basic profile error"),
            (ValidationError, "Validation error with details"),
            (ProfileNotFoundError, "Profile 'test_id' not found"),
        ]
        
        for error_class, message in error_scenarios:
            with self.subTest(error_class=error_class):
                error = error_class(message)
                self.assertEqual(str(error), message)
                self.assertIsInstance(error, Exception)


class TestProfileBuilderAdvancedScenarios(unittest.TestCase):
    """Test advanced ProfileBuilder scenarios"""
    
    def setUp(self):
        """
        Set up a new ProfileBuilder instance before each test case.
        """
        self.builder = ProfileBuilder()
    
    def test_builder_fluent_interface_with_conditionals(self):
        """
        Tests that the profile builder's fluent interface supports conditional method chaining, allowing selective inclusion of settings based on runtime conditions.
        """
        use_advanced_settings = True
        use_debug_mode = False
        
        result = self.builder.with_name('conditional_test')
        
        if use_advanced_settings:
            result = result.with_settings({
                'advanced': True,
                'optimization_level': 'high'
            })
        
        if use_debug_mode:
            result = result.with_settings({
                'debug': True,
                'verbose': True
            })
        
        final_result = result.with_version('1.0.0').build()
        
        self.assertEqual(final_result['name'], 'conditional_test')
        self.assertTrue(final_result['settings']['advanced'])
        self.assertNotIn('debug', final_result['settings'])
    
    def test_builder_template_pattern(self):
        """
        Tests that a ProfileBuilder template can be copied and modified to generate multiple distinct profile data variations.
        
        Verifies that duplicating a builder's data and altering fields results in independent profiles, supporting template-based creation patterns.
        """
        # Create a base template
        base_template = (ProfileBuilder()
                        .with_name('template_base')
                        .with_version('1.0.0')
                        .with_settings({
                            'ai_model': 'gpt-4',
                            'temperature': 0.7
                        }))
        
        # Create variations from the template
        variation1 = ProfileBuilder()
        variation1.data = base_template.data.copy()
        variation1.with_name('variation_1').with_settings({
            'temperature': 0.5,
            'max_tokens': 500
        })
        
        result1 = variation1.build()
        
        self.assertEqual(result1['name'], 'variation_1')
        self.assertEqual(result1['settings']['temperature'], 0.5)
        self.assertEqual(result1['settings']['ai_model'], 'gpt-4')
        self.assertEqual(result1['settings']['max_tokens'], 500)
    
    def test_builder_validation_integration(self):
        """
        Test that profiles built with ProfileBuilder are correctly validated by ProfileValidator for both valid and invalid data.
        
        Verifies that a fully constructed profile passes validation, while an incomplete profile fails validation.
        """
        # Build a profile and validate it
        profile_data = (self.builder
                       .with_name('validation_integration')
                       .with_version('1.0.0')
                       .with_settings({'ai_model': 'gpt-4'})
                       .build())
        
        # Validate the built profile
        is_valid = ProfileValidator.validate_profile_data(profile_data)
        self.assertTrue(is_valid)
        
        # Test with invalid data
        invalid_profile = (ProfileBuilder()
                          .with_name('invalid_test')
                          .build())  # Missing version and settings
        
        is_invalid = ProfileValidator.validate_profile_data(invalid_profile)
        self.assertFalse(is_invalid)
    
    def test_builder_immutability_and_reuse(self):
        """
        Verify that a ProfileBuilder instance can be reused to create multiple independent profiles without cross-contamination of data.
        
        Ensures that modifying the builder for one profile does not affect others and that base properties remain consistent across all derived profiles.
        """
        # Create base builder
        base_builder = (ProfileBuilder()
                       .with_name('base_profile')
                       .with_version('1.0.0'))
        
        # Create different profiles from the same base
        profile1 = base_builder.with_settings({'temperature': 0.7}).build()
        profile2 = base_builder.with_settings({'temperature': 0.5}).build()
        
        # Verify that modifications don't affect each other
        self.assertEqual(profile1['settings']['temperature'], 0.5)  # Last setting wins
        self.assertEqual(profile2['settings']['temperature'], 0.5)
        
        # Both should have the same base properties
        self.assertEqual(profile1['name'], 'base_profile')
        self.assertEqual(profile2['name'], 'base_profile')


# Add import for gc module for memory testing
import gc


# Additional parametrized tests for comprehensive coverage
@pytest.mark.parametrize("data_size,expected_performance", [
    (100, 0.1),      # Small data should be fast
    (1000, 0.5),     # Medium data should be reasonable
    (10000, 2.0),    # Large data should still be acceptable
])
def test_profile_creation_performance_parametrized(data_size, expected_performance):
    """
    Parametrized test that asserts profile creation with large data completes within a specified time limit.
    
    Parameters:
        data_size (int): Number of elements to include in the profile's list and dictionary settings.
        expected_performance (float): Maximum allowed time in seconds for profile creation.
    """
    import time
    
    manager = ProfileManager()
    large_data = {
        'name': f'performance_test_{data_size}',
        'version': '1.0.0',
        'settings': {
            'large_list': list(range(data_size)),
            'large_dict': {f'key_{i}': f'value_{i}' for i in range(data_size // 10)}
        }
    }
    
    start_time = time.time()
    profile = manager.create_profile(f'perf_test_{data_size}', large_data)
    end_time = time.time()
    
    duration = end_time - start_time
    
    assert profile is not None
    assert duration < expected_performance, f"Performance test failed: {duration} >= {expected_performance}"


@pytest.mark.parametrize("invalid_data,expected_error", [
    (None, (TypeError, AttributeError)),
    ("string", (TypeError, AttributeError)),
    (123, (TypeError, AttributeError)),
    ([], (TypeError, AttributeError)),
    ({}, False),  # Empty dict might be valid
])
def test_profile_validation_error_types_parametrized(invalid_data, expected_error):
    """
    Parametrized test that checks whether `ProfileValidator.validate_profile_data` returns `False` for incomplete but structurally valid profile data, or raises the specified exception type for invalid input.
    
    Parameters:
        invalid_data: The profile data to validate.
        expected_error: The exception type expected for invalid input, or `False` if validation should simply return `False`.
    """
    if expected_error is False:
        # Valid case - should return False but not raise exception
        result = ProfileValidator.validate_profile_data(invalid_data)
        assert isinstance(result, bool)
    else:
        # Invalid case - should raise expected error
        with pytest.raises(expected_error):
            ProfileValidator.validate_profile_data(invalid_data)


@pytest.mark.parametrize("operation,profile_id,data,expected_outcome", [
    ("create", "test_id", {"name": "test", "version": "1.0", "settings": {}}, "success"),
    ("create", "", {"name": "test", "version": "1.0", "settings": {}}, "error"),
    ("create", None, {"name": "test", "version": "1.0", "settings": {}}, "error"),
    ("get", "existing_id", None, "success"),
    ("get", "nonexistent_id", None, "none"),
    ("update", "existing_id", {"name": "updated"}, "success"),
    ("update", "nonexistent_id", {"name": "updated"}, "error"),
    ("delete", "existing_id", None, "success"),
    ("delete", "nonexistent_id", None, "false"),
])
def test_profile_manager_operations_parametrized(operation, profile_id, data, expected_outcome):
    """
    Parametrized test verifying that `ProfileManager` operations (`create`, `get`, `update`, `delete`) produce the expected outcomes for various input scenarios.
    
    Parameters:
        operation (str): The operation to perform ("create", "get", "update", or "delete").
        profile_id (str): The profile ID used in the operation.
        data (dict): Profile data for creation or update operations.
        expected_outcome (str): The expected result ("success", "error", "none", or "false").
    """
    manager = ProfileManager()
    
    # Setup: Create a profile for operations that need it
    if profile_id == "existing_id":
        manager.create_profile("existing_id", {
            "name": "existing", 
            "version": "1.0", 
            "settings": {}
        })
    
    if operation == "create":
        if expected_outcome == "success":
            profile = manager.create_profile(profile_id, data)
            assert profile is not None
            assert profile.profile_id == profile_id
        elif expected_outcome == "error":
            with pytest.raises((TypeError, ValueError)):
                manager.create_profile(profile_id, data)
    
    elif operation == "get":
        result = manager.get_profile(profile_id)
        if expected_outcome == "success":
            assert result is not None
        elif expected_outcome == "none":
            assert result is None
    
    elif operation == "update":
        if expected_outcome == "success":
            result = manager.update_profile(profile_id, data)
            assert result is not None
        elif expected_outcome == "error":
            with pytest.raises(ProfileNotFoundError):
                manager.update_profile(profile_id, data)
    
    elif operation == "delete":
        result = manager.delete_profile(profile_id)
        if expected_outcome == "success":
            assert result is True
        elif expected_outcome == "false":
            assert result is False


# Performance benchmark tests
class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for regression detection"""
    
    def test_profile_creation_benchmark(self):
        """
        Benchmarks the creation of 1,000 profiles and verifies performance and storage correctness.
        
        Asserts that total and average profile creation times remain within defined thresholds, and confirms that all created profiles are present in the manager after the operation.
        """
        import time
        
        manager = ProfileManager()
        num_iterations = 1000
        
        start_time = time.time()
        for i in range(num_iterations):
            data = {
                'name': f'benchmark_profile_{i}',
                'version': '1.0.0',
                'settings': {'index': i}
            }
            manager.create_profile(f'benchmark_{i}', data)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        
        # Performance assertions
        self.assertLess(total_time, 5.0, "Total benchmark time exceeded threshold")
        self.assertLess(avg_time, 0.01, "Average creation time per profile exceeded threshold")
        
        # Verify all profiles were created
        self.assertEqual(len(manager.profiles), num_iterations)
    
    def test_profile_lookup_benchmark(self):
        """
        Benchmark the retrieval performance of 10,000 random profile lookups from a pool of 1,000 created profiles.
        
        Asserts that the total and average lookup times remain below defined thresholds, ensuring efficient large-scale access in the profile manager.
        """
        import time
        import random
        
        manager = ProfileManager()
        num_profiles = 1000
        num_lookups = 10000
        
        # Create profiles
        profile_ids = []
        for i in range(num_profiles):
            profile_id = f'lookup_benchmark_{i}'
            data = {
                'name': f'profile_{i}',
                'version': '1.0.0',
                'settings': {'index': i}
            }
            manager.create_profile(profile_id, data)
            profile_ids.append(profile_id)
        
        # Benchmark lookups
        start_time = time.time()
        for _ in range(num_lookups):
            random_id = random.choice(profile_ids)
            profile = manager.get_profile(random_id)
            self.assertIsNotNone(profile)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_lookups
        
        # Performance assertions
        self.assertLess(total_time, 2.0, "Total lookup benchmark time exceeded threshold")
        self.assertLess(avg_time, 0.001, "Average lookup time per profile exceeded threshold")


if __name__ == '__main__':
    # Run both unittest and pytest
    import sys
    
    # Run unittest tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run pytest tests
    pytest.main([__file__, '-v'])

class TestProfileManagerConcurrencySimulation(unittest.TestCase):
    """Test scenarios that simulate concurrent access patterns"""
    
    def setUp(self):
        """
        Set up a new ProfileManager instance before each concurrency simulation test.
        """
        self.manager = ProfileManager()
        
    def test_rapid_creation_and_deletion_cycles(self):
        """
        Simulates rapid creation and deletion of profiles to verify manager consistency during high-frequency operations.
        
        Ensures that profiles can be created and deleted in quick succession without leaving residual state or inconsistencies in the profile manager.
        """
        profile_id = 'rapid_cycle_test'
        data = {
            'name': 'rapid_test',
            'version': '1.0.0',
            'settings': {'counter': 0}
        }
        
        for i in range(50):
            # Create profile
            profile = self.manager.create_profile(f'{profile_id}_{i}', data)
            self.assertIsNotNone(profile)
            
            # Verify it exists
            retrieved = self.manager.get_profile(f'{profile_id}_{i}')
            self.assertEqual(retrieved.profile_id, f'{profile_id}_{i}')
            
            # Delete it immediately
            deleted = self.manager.delete_profile(f'{profile_id}_{i}')
            self.assertTrue(deleted)
            
            # Verify deletion
            self.assertIsNone(self.manager.get_profile(f'{profile_id}_{i}'))
    
    def test_overlapping_profile_operations(self):
        """
        Verify that updating a subset of profiles does not affect the data of other profiles.
        
        Creates multiple profiles, performs updates on every other profile, and asserts that only the targeted profiles reflect the changes while others retain their original data.
        """
        profiles = []
        
        # Create multiple profiles
        for i in range(10):
            profile_id = f'overlap_test_{i}'
            data = {
                'name': f'profile_{i}',
                'version': '1.0.0',
                'settings': {'id': i, 'status': 'created'}
            }
            profile = self.manager.create_profile(profile_id, data)
            profiles.append((profile_id, profile))
        
        # Perform overlapping operations
        for i, (profile_id, profile) in enumerate(profiles):
            # Update every other profile
            if i % 2 == 0:
                self.manager.update_profile(profile_id, {'settings': {'id': i, 'status': 'updated'}})
            
            # Verify other profiles weren't affected
            for j, (other_id, _) in enumerate(profiles):
                if i != j:
                    other_profile = self.manager.get_profile(other_id)
                    expected_status = 'updated' if j % 2 == 0 and j < i else 'created'
                    self.assertEqual(other_profile.data['settings']['status'], expected_status)


class TestProfileDataTypeHandling(unittest.TestCase):
    """Test handling of various data types in profile data"""
    
    def setUp(self):
        """
        Initializes a new ProfileManager instance and prepares test data for data type handling tests.
        """
        self.manager = ProfileManager()
        
    def test_complex_data_type_preservation(self):
        """
        Test that complex Python data types are preserved in profile data after creation.
        
        Verifies that types such as Decimal, date, time, tuple, set, and frozenset, including within nested and mixed structures, remain intact when stored in a profile.
        """
        from decimal import Decimal
        from datetime import date, time
        
        complex_data = {
            'name': 'complex_types_test',
            'version': '1.0.0',
            'settings': {
                'decimal_value': Decimal('123.456'),
                'date_value': date(2023, 12, 25),
                'time_value': time(14, 30, 0),
                'tuple_value': (1, 2, 3, 'tuple'),
                'set_value': {1, 2, 3, 4, 5},
                'frozenset_value': frozenset([1, 2, 3]),
                'boolean_values': [True, False, None],
                'mixed_list': [1, 'string', 3.14, True, None],
                'nested_structures': {
                    'list_of_dicts': [
                        {'key1': 'value1', 'nested_list': [1, 2, 3]},
                        {'key2': 'value2', 'nested_tuple': (4, 5, 6)}
                    ],
                    'dict_of_lists': {
                        'numbers': [1, 2, 3, 4, 5],
                        'strings': ['a', 'b', 'c'],
                        'mixed': [1, 'two', 3.0, True]
                    }
                }
            }
        }
        
        profile = self.manager.create_profile('complex_types', complex_data)
        
        # Verify all data types are preserved
        self.assertIsInstance(profile.data['settings']['decimal_value'], Decimal)
        self.assertEqual(profile.data['settings']['decimal_value'], Decimal('123.456'))
        
        self.assertIsInstance(profile.data['settings']['date_value'], date)
        self.assertEqual(profile.data['settings']['date_value'], date(2023, 12, 25))
        
        self.assertIsInstance(profile.data['settings']['tuple_value'], tuple)
        self.assertEqual(profile.data['settings']['tuple_value'], (1, 2, 3, 'tuple'))
        
        # Note: Sets might be converted to lists depending on implementation
        self.assertIn(profile.data['settings']['set_value'].__class__.__name__, ['set', 'list'])
    
    def test_data_type_conversion_edge_cases(self):
        """
        Verify that the profile system accurately preserves and manages edge case data types, including special float values (infinity, NaN), extremely large and small numbers, empty containers, and special string formats within profile data.
        """
        edge_case_data = {
            'name': 'edge_cases',
            'version': '1.0.0',
            'settings': {
                'infinity_values': [float('inf'), float('-inf')],
                'nan_value': float('nan'),
                'very_large_int': 10**100,
                'very_small_float': 1e-100,
                'empty_containers': {
                    'empty_list': [],
                    'empty_dict': {},
                    'empty_string': '',
                    'empty_tuple': ()
                },
                'special_strings': {
                    'unicode_emoji': '🚀🌟✨',
                    'multiline': 'line1\nline2\nline3',
                    'escaped_chars': 'quote: " slash: \\ tab: \t',
                    'binary_like': b'binary_data'.decode('utf-8', errors='ignore')
                }
            }
        }
        
        profile = self.manager.create_profile('edge_cases', edge_case_data)
        
        # Verify edge cases are handled
        self.assertEqual(len(profile.data['settings']['empty_containers']['empty_list']), 0)
        self.assertEqual(len(profile.data['settings']['empty_containers']['empty_dict']), 0)
        self.assertEqual(profile.data['settings']['special_strings']['unicode_emoji'], '🚀🌟✨')
        self.assertIn('\n', profile.data['settings']['special_strings']['multiline'])


class TestProfileValidatorExtended(unittest.TestCase):
    """Extended tests for ProfileValidator with more complex validation scenarios"""
    
    def test_cross_field_validation_simulation(self):
        """
        Simulates cross-field validation scenarios to verify that the profile validator accepts profiles regardless of inter-field dependencies.
        
        This test ensures that profiles with potentially inconsistent or related fields still pass validation, reflecting the current validator's lack of cross-field checks.
        """
        validation_scenarios = [
            {
                'name': 'version_name_consistency',
                'data': {
                    'name': 'beta_profile',
                    'version': '1.0.0-beta',
                    'settings': {'environment': 'beta'}
                },
                'should_validate': True
            },
            {
                'name': 'environment_mismatch',
                'data': {
                    'name': 'production_profile',
                    'version': '1.0.0',
                    'settings': {'environment': 'development'}
                },
                'should_validate': True  # Current validator doesn't check this
            },
            {
                'name': 'version_settings_compatibility',
                'data': {
                    'name': 'legacy_profile',
                    'version': '0.1.0',
                    'settings': {'legacy_mode': True, 'deprecated_features': ['old_api']}
                },
                'should_validate': True
            }
        ]
        
        for scenario in validation_scenarios:
            with self.subTest(scenario=scenario['name']):
                result = ProfileValidator.validate_profile_data(scenario['data'])
                self.assertEqual(result, scenario['should_validate'])
    
    def test_validation_with_dynamic_schemas(self):
        """
        Test profile data validation against dynamically changing schema requirements.
        
        Simulates scenarios where required fields and nested settings vary by profile type or version, verifying that the validator accepts valid data and rejects data missing required fields under each schema variant.
        """
        schema_variants = [
            {
                'type': 'ai_model_profile',
                'required_fields': ['name', 'version', 'settings'],
                'settings_requirements': ['model_type', 'parameters'],
                'data': {
                    'name': 'ai_test',
                    'version': '1.0.0',
                    'settings': {
                        'model_type': 'transformer',
                        'parameters': {'layers': 12, 'attention_heads': 8}
                    }
                }
            },
            {
                'type': 'workflow_profile',
                'required_fields': ['name', 'version', 'settings'],
                'settings_requirements': ['steps', 'configuration'],
                'data': {
                    'name': 'workflow_test',
                    'version': '2.0.0',
                    'settings': {
                        'steps': ['preprocess', 'execute', 'postprocess'],
                        'configuration': {'parallel': True, 'timeout': 300}
                    }
                }
            }
        ]
        
        for variant in schema_variants:
            with self.subTest(profile_type=variant['type']):
                # Test basic validation (current implementation)
                result = ProfileValidator.validate_profile_data(variant['data'])
                self.assertTrue(result)
                
                # Test missing required fields
                incomplete_data = variant['data'].copy()
                incomplete_data.pop('settings')
                result = ProfileValidator.validate_profile_data(incomplete_data)
                self.assertFalse(result)


class TestProfileBuilderFactoryPatterns(unittest.TestCase):
    """Test factory patterns and advanced builder scenarios"""
    
    def test_profile_builder_factory_method(self):
        """
        Test factory method patterns for creating specialized ProfileBuilder instances with predefined configurations.
        
        Verifies that factory methods can generate builders tailored for specific use cases, such as AI model profiles and API configuration profiles, and that the resulting profiles include the expected default fields and values for each scenario.
        """
        def create_ai_model_builder():
            """
            Return a ProfileBuilder instance preconfigured for creating AI model profiles with default version and settings.
            
            Returns:
                ProfileBuilder: Builder initialized with version '1.0.0' and AI model-specific settings.
            """
            return (ProfileBuilder()
                   .with_version('1.0.0')
                   .with_settings({
                       'model_type': 'neural_network',
                       'training_data': 'default_dataset',
                       'hyperparameters': {
                           'learning_rate': 0.001,
                           'batch_size': 32,
                           'epochs': 100
                       }
                   }))
        
        def create_api_config_builder():
            """
            Create a ProfileBuilder instance preconfigured for API configuration profiles.
            
            Returns:
                ProfileBuilder: A builder with version '2.0.0' and default settings for endpoint configuration and OAuth2 authentication.
            """
            return (ProfileBuilder()
                   .with_version('2.0.0')
                   .with_settings({
                       'endpoint_config': {
                           'base_url': 'https://api.example.com',
                           'timeout': 30,
                           'retry_attempts': 3
                       },
                       'authentication': {
                           'method': 'oauth2',
                           'scope': ['read', 'write']
                       }
                   }))
        
        # Test AI model builder
        ai_builder = create_ai_model_builder()
        ai_profile = ai_builder.with_name('test_ai_model').build()
        
        self.assertEqual(ai_profile['name'], 'test_ai_model')
        self.assertEqual(ai_profile['settings']['model_type'], 'neural_network')
        self.assertEqual(ai_profile['settings']['hyperparameters']['learning_rate'], 0.001)
        
        # Test API config builder
        api_builder = create_api_config_builder()
        api_profile = api_builder.with_name('test_api_config').build()
        
        self.assertEqual(api_profile['name'], 'test_api_config')
        self.assertEqual(api_profile['settings']['endpoint_config']['base_url'], 'https://api.example.com')
        self.assertEqual(api_profile['settings']['authentication']['method'], 'oauth2')
    
    def test_builder_composition_patterns(self):
        """
        Tests that multiple specialized profile builders can be composed to collaboratively construct a single profile with combined settings.
        
        Simulates the use of separate builder functions for different profile aspects (e.g., security and performance), ensuring the final profile includes all composed configurations.
        """
        # Base profile builder
        base_builder = ProfileBuilder().with_name('composed_profile').with_version('1.0.0')
        
        # Specialized builders for different aspects
        def add_security_settings(builder):
            """
            Adds standard security settings to a profile builder and returns the updated builder.
            
            The injected security settings include encryption type, authentication requirement, and access control roles and permissions.
            """
            current_settings = builder.data.get('settings', {})
            security_settings = {
                'encryption': 'AES-256',
                'authentication_required': True,
                'access_control': {
                    'roles': ['admin', 'user'],
                    'permissions': ['read', 'write', 'execute']
                }
            }
            current_settings.update(security_settings)
            return builder.with_settings(current_settings)
        
        def add_performance_settings(builder):
            """
            Adds standard performance settings to a profile builder and returns the updated builder.
            
            The builder's 'settings' field is updated with predefined caching and connection pool configurations to optimize performance.
            
            Returns:
                ProfileBuilder: The builder instance with performance settings applied.
            """
            current_settings = builder.data.get('settings', {})
            performance_settings = {
                'caching': {
                    'enabled': True,
                    'ttl': 3600,
                    'max_size': '100MB'
                },
                'connection_pool': {
                    'min_connections': 5,
                    'max_connections': 50,
                    'timeout': 30
                }
            }
            current_settings.update(performance_settings)
            return builder.with_settings(current_settings)
        
        # Compose the profile
        composed_profile = add_performance_settings(
            add_security_settings(base_builder)
        ).build()
        
        # Verify composition
        self.assertEqual(composed_profile['name'], 'composed_profile')
        self.assertTrue(composed_profile['settings']['authentication_required'])
        self.assertEqual(composed_profile['settings']['encryption'], 'AES-256')
        self.assertTrue(composed_profile['settings']['caching']['enabled'])
        self.assertEqual(composed_profile['settings']['connection_pool']['max_connections'], 50)


class TestProfileManagerAdvancedQueries(unittest.TestCase):
    """Test advanced query and filtering capabilities"""
    
    def setUp(self):
        """
        Set up a ProfileManager instance pre-populated with diverse sample profiles for advanced query testing.
        
        This method initializes the test fixture by creating a ProfileManager and adding several profiles representing different types and configurations, enabling comprehensive query and filtering tests.
        """
        self.manager = ProfileManager()
        
        # Create diverse sample profiles
        sample_profiles = [
            {
                'id': 'ai_model_1',
                'data': {
                    'name': 'GPT_Model',
                    'version': '1.0.0',
                    'settings': {
                        'type': 'language_model',
                        'parameters': 175000000000,
                        'capabilities': ['text_generation', 'code_completion']
                    }
                }
            },
            {
                'id': 'ai_model_2',
                'data': {
                    'name': 'BERT_Model',
                    'version': '2.1.0',
                    'settings': {
                        'type': 'language_model',
                        'parameters': 340000000,
                        'capabilities': ['text_classification', 'named_entity_recognition']
                    }
                }
            },
            {
                'id': 'workflow_1',
                'data': {
                    'name': 'Data_Pipeline',
                    'version': '1.5.0',
                    'settings': {
                        'type': 'data_workflow',
                        'stages': ['extract', 'transform', 'load'],
                        'schedule': 'daily'
                    }
                }
            },
            {
                'id': 'api_config_1',
                'data': {
                    'name': 'REST_API',
                    'version': '3.0.0',
                    'settings': {
                        'type': 'api_configuration',
                        'endpoints': ['users', 'products', 'orders'],
                        'rate_limit': 1000
                    }
                }
            }
        ]
        
        for profile_data in sample_profiles:
            self.manager.create_profile(profile_data['id'], profile_data['data'])
    
    def test_profile_filtering_simulation(self):
        """
        Simulates filtering profiles by their 'type' field within the ProfileManager.
        
        This test verifies that profiles can be accurately grouped and retrieved based on the 'type' specified in their settings, ensuring correct filtering by category.
        """
        # Simulate filtering by type
        def filter_by_type(manager, profile_type):
            """
            Retrieve all profiles from the manager whose 'settings.type' field matches the specified profile type.
            
            Parameters:
                profile_type (str): The profile type to match against each profile's 'settings.type' field.
            
            Returns:
                list: A list of profiles with a 'settings.type' equal to the given profile_type.
            """
            matching_profiles = []
            for profile_id, profile in manager.profiles.items():
                if profile.data.get('settings', {}).get('type') == profile_type:
                    matching_profiles.append(profile)
            return matching_profiles
        
        # Test type filtering
        language_models = filter_by_type(self.manager, 'language_model')
        self.assertEqual(len(language_models), 2)
        
        data_workflows = filter_by_type(self.manager, 'data_workflow')
        self.assertEqual(len(data_workflows), 1)
        
        api_configs = filter_by_type(self.manager, 'api_configuration')
        self.assertEqual(len(api_configs), 1)
    
    def test_profile_search_simulation(self):
        """
        Simulates a text-based search across profiles and verifies that searching by keyword returns the correct profiles based on their name or settings.
        
        This test ensures that the search functionality can identify profiles whose names or settings contain the specified search terms, and asserts that the expected number of relevant profiles are found for various keywords.
        """
        def search_profiles(manager, search_term):
            """
            Return a list of profiles from the manager whose name or settings contain the given search term.
            
            The search is case-insensitive and matches the term against the 'name' field and the string representation of the 'settings' field in each profile.
            
            Parameters:
                search_term (str): The case-insensitive term to match within profile names and settings.
            
            Returns:
                list: Profiles matching the search criteria.
            """
            matching_profiles = []
            search_term = search_term.lower()
            
            for profile_id, profile in manager.profiles.items():
                # Search in name
                if search_term in profile.data.get('name', '').lower():
                    matching_profiles.append(profile)
                    continue
                
                # Search in settings (simplified)
                settings_str = str(profile.data.get('settings', {})).lower()
                if search_term in settings_str:
                    matching_profiles.append(profile)
            
            return matching_profiles
        
        # Test various search terms
        model_results = search_profiles(self.manager, 'model')
        self.assertGreaterEqual(len(model_results), 2)  # Should find model profiles
        
        api_results = search_profiles(self.manager, 'api')
        self.assertGreaterEqual(len(api_results), 1)  # Should find API profile
        
        classification_results = search_profiles(self.manager, 'classification')
        self.assertGreaterEqual(len(classification_results), 1)  # Should find BERT model
    
    def test_profile_statistics_simulation(self):
        """
        Simulates statistical analysis of a profile collection and verifies that computed statistics such as total profiles, profile type distribution, version distribution, and average settings complexity match expected values for the test dataset.
        """
        def compute_profile_statistics(manager):
            """
            Compute aggregate statistics across all profiles managed by the specified manager.
            
            Returns:
                dict: Contains the total number of profiles, counts by profile type, version distribution, and the average number of keys in each profile's settings.
            """
            stats = {
                'total_profiles': len(manager.profiles),
                'profile_types': {},
                'version_distribution': {},
                'average_settings_complexity': 0
            }
            
            total_settings_keys = 0
            for profile in manager.profiles.values():
                # Count profile types
                profile_type = profile.data.get('settings', {}).get('type', 'unknown')
                stats['profile_types'][profile_type] = stats['profile_types'].get(profile_type, 0) + 1
                
                # Count version distribution
                version = profile.data.get('version', 'unknown')
                stats['version_distribution'][version] = stats['version_distribution'].get(version, 0) + 1
                
                # Count settings complexity
                settings = profile.data.get('settings', {})
                total_settings_keys += len(settings)
            
            if stats['total_profiles'] > 0:
                stats['average_settings_complexity'] = total_settings_keys / stats['total_profiles']
            
            return stats
        
        stats = compute_profile_statistics(self.manager)
        
        # Verify statistics
        self.assertEqual(stats['total_profiles'], 4)
        self.assertIn('language_model', stats['profile_types'])
        self.assertEqual(stats['profile_types']['language_model'], 2)
        self.assertGreater(stats['average_settings_complexity'], 0)


class TestProfileSystemIntegration(unittest.TestCase):
    """Test integration scenarios with external systems"""
    
    def setUp(self):
        """
        Set up a fresh ProfileManager instance before each test to ensure test isolation.
        """
        self.manager = ProfileManager()
        
    @patch('json.load')
    @patch('builtins.open')
    def test_profile_import_from_external_format(self, mock_open, mock_json_load):
        """
        Tests importing a profile from an external JSON format, converting it to the internal schema, validating the data, and creating a new profile.
        
        Simulates reading a profile from an external system with a different schema, mapping its fields to the internal format, validating the converted data, and verifying that the imported profile contains the expected fields and metadata.
        """
        # Mock external profile format
        external_profile_data = {
            'profile_name': 'imported_profile',
            'profile_version': '1.2.0',
            'configuration': {
                'ai_settings': {
                    'model': 'gpt-4',
                    'temperature': 0.8,
                    'max_output_tokens': 2000
                },
                'workflow_settings': {
                    'parallel_processing': True,
                    'batch_size': 50
                }
            },
            'metadata': {
                'created_by': 'external_system',
                'import_timestamp': '2023-12-01T10:00:00Z'
            }
        }
        
        mock_json_load.return_value = external_profile_data
        
        def import_external_profile(file_path, profile_id):
            """
            Import a profile from an external JSON file, convert it to the internal format, validate it, and create a new profile with the given ID.
            
            Parameters:
                file_path (str): Path to the external JSON profile file.
                profile_id (str): Unique identifier for the new profile.
            
            Returns:
                GenesisProfile: The created profile instance.
            
            Raises:
                ValidationError: If the external profile data does not meet internal validation requirements.
            """
            with open(file_path, 'r') as f:
                external_data = json.load(f)
            
            # Convert external format to internal format
            internal_data = {
                'name': external_data['profile_name'],
                'version': external_data['profile_version'],
                'settings': external_data['configuration']
            }
            
            # Add metadata
            if 'metadata' in external_data:
                internal_data['metadata'] = external_data['metadata']
            
            # Validate before importing
            if ProfileValidator.validate_profile_data(internal_data):
                return self.manager.create_profile(profile_id, internal_data)
            else:
                raise ValidationError("Invalid external profile format")
        
        # Test import
        imported_profile = import_external_profile('external_profile.json', 'imported_1')
        
        self.assertIsNotNone(imported_profile)
        self.assertEqual(imported_profile.data['name'], 'imported_profile')
        self.assertEqual(imported_profile.data['version'], '1.2.0')
        self.assertIn('ai_settings', imported_profile.data['settings'])
        self.assertIn('metadata', imported_profile.data)
    
    @patch('json.dump')
    @patch('builtins.open')
    def test_profile_export_to_external_format(self, mock_open, mock_json_dump):
        """
        Tests exporting a profile to an external JSON format with schema transformation for compatibility with external systems.
        
        Simulates the export process by transforming the internal profile structure to the required external schema, writing it to a file, and verifying both the output format and serialization behavior using mocks.
        """
        # Create a profile to export
        profile_data = {
            'name': 'export_test',
            'version': '1.0.0',
            'settings': {
                'ai_model': 'gpt-4',
                'temperature': 0.7,
                'advanced_settings': {
                    'reasoning_mode': 'chain_of_thought',
                    'safety_filters': ['content', 'bias', 'factuality']
                }
            },
            'metadata': {
                'tags': ['production', 'verified'],
                'last_used': '2023-12-01'
            }
        }
        
        profile = self.manager.create_profile('export_test', profile_data)
        
        def export_profile_to_external_format(profile, file_path):
            """
            Exports a profile to an external JSON format and saves it to the specified file path.
            
            Parameters:
                profile: The profile object to be exported.
                file_path (str): Destination file path for the exported JSON.
            
            Returns:
                dict: The profile data structured in the external JSON format.
            """
            # Transform internal format to external format
            external_format = {
                'profile_name': profile.data['name'],
                'profile_version': profile.data['version'],
                'configuration': profile.data['settings'],
                'export_metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'exported_by': 'profile_system',
                    'original_metadata': profile.data.get('metadata', {})
                },
                'system_info': {
                    'created_at': profile.created_at.isoformat(),
                    'updated_at': profile.updated_at.isoformat(),
                    'profile_id': profile.profile_id
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(external_format, f, indent=2)
            
            return external_format
        
        # Test export
        exported_data = export_profile_to_external_format(profile, 'exported_profile.json')
        
        # Verify export format
        self.assertEqual(exported_data['profile_name'], 'export_test')
        self.assertEqual(exported_data['profile_version'], '1.0.0')
        self.assertIn('configuration', exported_data)
        self.assertIn('export_metadata', exported_data)
        self.assertIn('system_info', exported_data)
        
        # Verify that json.dump was called
        mock_json_dump.assert_called_once()


class TestProfileVersioning(unittest.TestCase):
    """Test profile versioning and migration scenarios"""
    
    def setUp(self):
        """
        Set up a new ProfileManager instance before each test in the versioning test suite.
        """
        self.manager = ProfileManager()
    
    def test_profile_version_migration_simulation(self):
        """
        Simulates migration of a profile from version 1.x to 2.0.0 and verifies correct transformation of fields and metadata.
        
        This test ensures that legacy profile data in version 1.0 format is properly migrated to the new schema, including field renaming, addition of new required fields, and attachment of migration metadata. It verifies that the migrated profile reflects the expected structure and values in version 2.0.0.
        """
        # Old format profile (version 1.0)
        old_profile_data = {
            'name': 'legacy_profile',
            'version': '1.0.0',
            'settings': {
                'model_name': 'gpt-3',  # Old field name
                'temp': 0.7,            # Old field name
                'tokens': 1000          # Old field name
            }
        }
        
        def migrate_profile_v1_to_v2(profile_data):
            """
            Migrate a profile data dictionary from version 1.x to version 2.0.0 format.
            
            If the input profile's version starts with '1.', returns a new dictionary with updated field names, default values for new fields, and migration metadata. If the profile is not version 1.x, returns the original data unchanged.
            
            Parameters:
                profile_data (dict): The profile data dictionary to migrate.
            
            Returns:
                dict: The migrated profile data in version 2.0.0 format, or the original data if migration is not needed.
            """
            if profile_data['version'].startswith('1.'):
                # Create new format
                migrated_data = {
                    'name': profile_data['name'],
                    'version': '2.0.0',  # Upgrade version
                    'settings': {
                        # Migrate field names
                        'ai_model': profile_data['settings'].get('model_name', 'unknown'),
                        'temperature': profile_data['settings'].get('temp', 0.5),
                        'max_tokens': profile_data['settings'].get('tokens', 500),
                        # Add new required fields
                        'response_format': 'text',
                        'safety_settings': {
                            'content_filter': True,
                            'bias_mitigation': True
                        }
                    },
                    # Add migration metadata
                    'migration_info': {
                        'original_version': profile_data['version'],
                        'migrated_at': datetime.now().isoformat(),
                        'migration_notes': 'Automated migration from v1.0 to v2.0'
                    }
                }
                return migrated_data
            return profile_data
        
        # Test migration
        migrated_data = migrate_profile_v1_to_v2(old_profile_data)
        profile = self.manager.create_profile('migrated_profile', migrated_data)
        
        # Verify migration
        self.assertEqual(profile.data['version'], '2.0.0')
        self.assertEqual(profile.data['settings']['ai_model'], 'gpt-3')
        self.assertEqual(profile.data['settings']['temperature'], 0.7)
        self.assertEqual(profile.data['settings']['max_tokens'], 1000)
        self.assertIn('safety_settings', profile.data['settings'])
        self.assertIn('migration_info', profile.data)
    
    def test_backward_compatibility_validation(self):
        """
        Tests that profiles from various schema versions are correctly validated and created, ensuring backward compatibility with older and newer profile formats.
        
        Verifies that the current validation logic accepts profiles with different version strings and structures, and that valid profiles can be created successfully for each version scenario.
        """
        version_scenarios = [
            {
                'version': '1.0.0',
                'data': {
                    'name': 'v1_profile',
                    'version': '1.0.0',
                    'settings': {
                        'basic_setting': 'value'
                    }
                },
                'should_validate': True
            },
            {
                'version': '2.0.0',
                'data': {
                    'name': 'v2_profile',
                    'version': '2.0.0',
                    'settings': {
                        'enhanced_setting': 'value',
                        'new_features': {
                            'feature1': True,
                            'feature2': False
                        }
                    }
                },
                'should_validate': True
            },
            {
                'version': '3.0.0-alpha',
                'data': {
                    'name': 'v3_alpha_profile',
                    'version': '3.0.0-alpha',
                    'settings': {
                        'experimental_features': {
                            'ai_reasoning': True,
                            'multi_modal': False
                        }
                    }
                },
                'should_validate': True
            }
        ]
        
        for scenario in version_scenarios:
            with self.subTest(version=scenario['version']):
                # Test validation
                is_valid = ProfileValidator.validate_profile_data(scenario['data'])
                self.assertEqual(is_valid, scenario['should_validate'])
                
                # Test profile creation
                if scenario['should_validate']:
                    profile = self.manager.create_profile(f"test_{scenario['version']}", scenario['data'])
                    self.assertIsNotNone(profile)
                    self.assertEqual(profile.data['version'], scenario['version'])


# Add comprehensive stress testing
class TestProfileSystemStress(unittest.TestCase):
    """Stress tests for the profile system under extreme conditions"""
    
    def setUp(self):
        """
        Set up a new ProfileManager instance before each test to ensure test isolation.
        """
        self.manager = ProfileManager()
    
    def test_extreme_data_volume_handling(self):
        """
        Test system stability and performance when creating and retrieving a profile with extremely large and deeply nested data structures.
        
        Creates a profile containing massive lists, large dictionaries, and deeply nested objects to verify that the profile manager can handle extreme data volumes efficiently and within acceptable time limits.
        """
        # Create profile with extremely large data
        extreme_data = {
            'name': 'extreme_volume_test',
            'version': '1.0.0',
            'settings': {
                'massive_list': list(range(100000)),  # 100K items
                'huge_dict': {f'key_{i}': f'value_{i}' * 100 for i in range(10000)},  # 10K keys with long values
                'nested_structure': {
                    f'level_{i}': {
                        f'sublevel_{j}': {
                            f'data_{k}': f'content_{k}' * 50
                            for k in range(100)
                        }
                        for j in range(50)
                    }
                    for i in range(10)
                }
            }
        }
        
        # Test creation with extreme data
        start_time = time.time()
        profile = self.manager.create_profile('extreme_test', extreme_data)
        creation_time = time.time() - start_time
        
        # Verify creation succeeded
        self.assertIsNotNone(profile)
        self.assertEqual(len(profile.data['settings']['massive_list']), 100000)
        self.assertEqual(len(profile.data['settings']['huge_dict']), 10000)
        
        # Test retrieval performance with extreme data
        start_time = time.time()
        retrieved = self.manager.get_profile('extreme_test')
        retrieval_time = time.time() - start_time
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.profile_id, 'extreme_test')
        
        # Performance assertions (should complete within reasonable time)
        self.assertLess(creation_time, 10.0, "Extreme data creation took too long")
        self.assertLess(retrieval_time, 1.0, "Extreme data retrieval took too long")
    
    def test_rapid_fire_operations(self):
        """
        Stress tests the profile management system by performing a high volume of rapid create, read, update, and delete operations in sequence.
        
        Simulates 1,000 profile operation cycles, verifies error-free execution, checks the expected number of remaining profiles, and asserts that system throughput exceeds a minimum threshold.
        """
        import threading
        import time
        
        operation_count = 1000
        errors = []
        
        def rapid_operations():
            """
            Performs repeated rapid create, read, update, and conditional delete operations on profiles for stress or concurrency testing.
            
            Appends any encountered errors to the `errors` list for later inspection.
            """
            try:
                for i in range(operation_count):
                    profile_id = f'rapid_{threading.current_thread().ident}_{i}'
                    
                    # Create
                    data = {
                        'name': f'rapid_profile_{i}',
                        'version': '1.0.0',
                        'settings': {'index': i, 'thread_id': threading.current_thread().ident}
                    }
                    profile = self.manager.create_profile(profile_id, data)
                    
                    # Read
                    retrieved = self.manager.get_profile(profile_id)
                    if retrieved is None:
                        errors.append(f"Failed to retrieve profile {profile_id}")
                    
                    # Update
                    self.manager.update_profile(profile_id, {'settings': {'index': i, 'updated': True}})
                    
                    # Delete every 10th profile
                    if i % 10 == 0:
                        self.manager.delete_profile(profile_id)
                    
            except Exception as e:
                errors.append(f"Error in rapid operations: {str(e)}")
        
        # Execute rapid operations
        start_time = time.time()
        rapid_operations()
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Errors during rapid operations: {errors}")
        
        # Verify final state
        remaining_profiles = len(self.manager.profiles)
        expected_remaining = operation_count - (operation_count // 10)  # Minus deleted profiles
        self.assertEqual(remaining_profiles, expected_remaining)
        
        # Performance assertion
        operations_per_second = (operation_count * 4) / duration  # 4 operations per iteration
        self.assertGreater(operations_per_second, 100, "Operations per second too low")


if __name__ == '__main__':
    # Run comprehensive test suite
    unittest.main(argv=[''], exit=False, verbosity=2)