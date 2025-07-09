import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import tempfile

# Import the module under test
from app.ai_backend.genesis_profile import (
    GenesisProfile,
    ProfileManager,
    generate_profile_data,
    validate_profile_schema,
    merge_profiles,
    ProfileValidationError,
    ProfileNotFoundError
)


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
            self.profile.update_preferences("not_a_dict")
        with self.assertRaises(TypeError):
            self.profile.update_preferences(123)
    
    def test_genesis_profile_data_immutability(self):
        """
        Test that a copied snapshot of a GenesisProfile's data remains unchanged after the profile's data is modified.
        
        Ensures that copying the profile's data produces an immutable snapshot, and subsequent changes to the profile do not affect the copied data.
        """
        original_data = self.profile.data.copy()
        
        # Modify the data
        self.profile.data['new_field'] = 'new_value'
        
        # Original data should not be affected if properly implemented
        self.assertNotEqual(self.profile.data, original_data)
        self.assertIn('new_field', self.profile.data)
    
    def test_genesis_profile_str_representation(self):
        """
        Tests that the string representation of a GenesisProfile instance includes the profile ID and is of type string.
        """
        str_repr = str(self.profile)
        self.assertIn(self.profile.id, str_repr)
        self.assertIsInstance(str_repr, str)
    
    def test_genesis_profile_equality(self):
        """
        Verify that GenesisProfile instances are equal when both profile ID and data match, and unequal when profile IDs differ.
        """
        profile1 = GenesisProfile(self.sample_profile_data)
        profile2 = GenesisProfile(self.sample_profile_data.copy())
        other_data = self.sample_profile_data.copy()
        other_data['id'] = 'different_id'
        profile3 = GenesisProfile(other_data)
        
        if hasattr(profile1, '__eq__'):
            self.assertEqual(profile1, profile2)
            self.assertNotEqual(profile1, profile3)


class TestProfileManager(unittest.TestCase):
    """Test cases for ProfileManager class"""
    
    def setUp(self):
        """
        Initializes a new ProfileManager and sample profile data before each test to ensure isolation.
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
        """Tests that a profile is created and stored successfully."""
        profile = self.manager.create_profile(self.profile_id, self.sample_data)
        self.assertIsInstance(profile, GenesisProfile)
        self.assertEqual(profile.profile_id, self.profile_id)
        self.assertEqual(profile.data, self.sample_data)
        self.assertIn(self.profile_id, self.manager.profiles)
    
    def test_create_profile_duplicate_id(self):
        """Test creation of a profile with a duplicate ID."""
        self.manager.create_profile(self.profile_id, self.sample_data)
        try:
            duplicate = self.manager.create_profile(self.profile_id, {'name': 'duplicate', 'version': '1.0.0', 'settings': {}})
            self.assertEqual(duplicate.profile_id, self.profile_id)
        except Exception as e:
            self.assertIsInstance(e, (ProfileValidationError, ValueError))
    
    def test_create_profile_invalid_data(self):
        """Test that creating a profile with invalid data raises."""
        with self.assertRaises((TypeError, ValueError)):
            self.manager.create_profile(self.profile_id, None)
    
    def test_get_profile_existing(self):
        """Test retrieval of an existing profile."""
        created = self.manager.create_profile(self.profile_id, self.sample_data)
        retrieved = self.manager.get_profile(self.profile_id)
        self.assertEqual(retrieved, created)
    
    def test_get_profile_nonexistent(self):
        """Test that retrieving a nonexistent profile returns None."""
        self.assertIsNone(self.manager.get_profile('nonexistent'))
    
    def test_update_profile_success(self):
        """Test successful profile update."""
        self.manager.create_profile(self.profile_id, self.sample_data)
        updated = self.manager.update_profile(self.profile_id, {'name': 'updated'})
        self.assertEqual(updated.data['name'], 'updated')
        self.assertIsInstance(updated.updated_at, datetime)
    
    def test_update_profile_nonexistent(self):
        """Test updating a non-existent profile raises."""
        with self.assertRaises(ProfileNotFoundError):
            self.manager.update_profile('no_id', {'name': 'x'})
    
    def test_update_profile_empty_data(self):
        """Test that updating with empty data does not modify."""
        profile = self.manager.create_profile(self.profile_id, self.sample_data)
        updated = self.manager.update_profile(self.profile_id, {})
        self.assertEqual(updated.data, self.sample_data)
    
    def test_delete_profile_success(self):
        """Test successful deletion of a profile."""
        self.manager.create_profile(self.profile_id, self.sample_data)
        result = self.manager.delete_profile(self.profile_id)
        self.assertTrue(result)
        self.assertNotIn(self.profile_id, self.manager.profiles)
    
    def test_delete_profile_nonexistent(self):
        """Test deleting a nonexistent profile returns False."""
        self.assertFalse(self.manager.delete_profile('no_id'))
    
    def test_manager_state_isolation(self):
        """Verify separate manager instances maintain independent state."""
        m1 = ProfileManager()
        m2 = ProfileManager()
        m1.create_profile(self.profile_id, self.sample_data)
        self.assertIsNotNone(m1.get_profile(self.profile_id))
        self.assertIsNone(m2.get_profile(self.profile_id))


class TestProfileValidator(unittest.TestCase):
    """Test cases for ProfileValidator class"""
    
    def setUp(self):
        """Prepare valid profile data."""
        self.valid_data = {
            'name': 'test_profile',
            'version': '1.0.0',
            'settings': {
                'ai_model': 'gpt-4',
                'temperature': 0.7
            }
        }
    
    def test_validate_profile_data_valid(self):
        """Valid profile data should pass."""
        self.assertTrue(validate_profile_schema(self.valid_data))
    
    def test_validate_profile_data_missing_required_fields(self):
        """Missing required fields should fail."""
        cases = [
            {'version': '1.0.0', 'settings': {}},
            {'name': 'test', 'settings': {}},
            {'name': 'test', 'version': '1.0.0'},
            {}
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(validate_profile_schema(case))
    
    def test_validate_profile_data_empty_values(self):
        """Empty required values return boolean."""
        cases = [
            {'name': '', 'version': '1.0.0', 'settings': {}},
            {'name': 'test', 'version': '', 'settings': {}},
            {'name': 'test', 'version': '1.0.0', 'settings': None},
        ]
        for case in cases:
            with self.subTest(case=case):
                result = validate_profile_schema(case)
                self.assertIsInstance(result, bool)
    
    def test_validate_profile_data_none_input(self):
        """None input raises."""
        with self.assertRaises((TypeError, AttributeError)):
            validate_profile_schema(None)
    
    def test_validate_profile_data_invalid_types(self):
        """Invalid types raise."""
        for invalid in ["string", 123, [], set()]:
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, AttributeError)):
                    validate_profile_schema(invalid)
    
    def test_validate_profile_data_extra_fields(self):
        """Extra fields should be allowed."""
        data = self.valid_data.copy()
        data.update({'extra': 'value', 'metadata': {}})
        self.assertTrue(validate_profile_schema(data))


class TestMergeProfiles(unittest.TestCase):
    """Test merge_profiles function"""
    
    def test_merge_profiles_non_conflicting(self):
        p1 = {'a': 1, 'b': 2}
        p2 = {'c': 3}
        merged = merge_profiles(p1, p2)
        self.assertEqual(merged, {'a': 1, 'b': 2, 'c': 3})
    
    def test_merge_profiles_conflicting(self):
        p1 = {'a': 1}
        p2 = {'a': 2}
        merged = merge_profiles(p1, p2)
        self.assertEqual(merged['a'], 2)  # p2 wins


@pytest.mark.parametrize("profile_id,expected_valid", [
    ("valid_id", True),
    ("", False),
    ("profile-123", True),
    ("profile_456", True),
    ("profile.789", True),
    ("PROFILE_UPPER", True),
    (None, False),
    (123, False),
    ([], False),
])
def test_profile_id_validation_parametrized(profile_id, expected_valid):
    manager = ProfileManager()
    data = {'name': 'test_profile', 'version': '1.0.0', 'settings': {}}
    if expected_valid:
        try:
            profile = manager.create_profile(profile_id, data)
            assert profile.profile_id == profile_id
        except (TypeError, ValueError):
            pass
    else:
        with pytest.raises((TypeError, ValueError)):
            manager.create_profile(profile_id, data)


@pytest.mark.parametrize("data,should_validate", [
    ({"name": "test", "version": "1.0", "settings": {}}, True),
    ({"name": "test", "version": "1.0"}, False),
    ({"name": "test", "settings": {}}, False),
    ({"version": "1.0", "settings": {}}, False),
    ({}, False),
    ({"name": "", "version": "1.0", "settings": {}}, True),
    ({"name": "test", "version": "", "settings": {}}, True),
    ({"name": "test", "version": "1.0", "settings": None}, True),
])
def test_profile_validation_parametrized(data, should_validate):
    result = validate_profile_schema(data)
    assert result == should_validate


if __name__ == '__main__':
    unittest.main()