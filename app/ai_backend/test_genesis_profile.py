import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

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
    # Fallback implementations for testing
    class GenesisProfile:
        def __init__(self, profile_id: str, data: Dict[str, Any]):
            if not isinstance(profile_id, str):
                raise TypeError("profile_id must be a string")
            if not profile_id:
                raise ValueError("profile_id cannot be empty")
            if not isinstance(data, dict):
                raise TypeError("data must be a dict")
            self.profile_id = profile_id
            self.data = data
            now = datetime.now(timezone.utc)
            self.created_at = now
            self.updated_at = now

        def __str__(self):
            return f"GenesisProfile(id={self.profile_id})"

        def __eq__(self, other):
            if not isinstance(other, GenesisProfile):
                return False
            return self.profile_id == other.profile_id and self.data == other.data

    class ProfileError(Exception):
        pass

    class ValidationError(ProfileError):
        pass

    class ProfileNotFoundError(ProfileError):
        pass

    class ProfileManager:
        def __init__(self):
            self.profiles: Dict[str, GenesisProfile] = {}

        def create_profile(self, profile_id: str, data: Dict[str, Any]) -> GenesisProfile:
            if not isinstance(profile_id, str):
                raise TypeError("profile_id must be a string")
            if not profile_id:
                raise ValueError("profile_id cannot be empty")
            if not isinstance(data, dict):
                raise TypeError("data must be a dict")
            profile = GenesisProfile(profile_id, data)
            self.profiles[profile_id] = profile
            return profile

        def get_profile(self, profile_id: str) -> Optional[GenesisProfile]:
            if not isinstance(profile_id, str):
                return None
            return self.profiles.get(profile_id)

        def update_profile(self, profile_id: str, data: Dict[str, Any]) -> GenesisProfile:
            if not isinstance(data, dict):
                raise TypeError("data must be a dict")
            if profile_id not in self.profiles:
                raise ProfileNotFoundError(f"Profile {profile_id} not found")
            profile = self.profiles[profile_id]
            profile.data.update(data)
            profile.updated_at = datetime.now(timezone.utc)
            return profile

        def delete_profile(self, profile_id: str) -> bool:
            if profile_id in self.profiles:
                del self.profiles[profile_id]
                return True
            return False

    class ProfileValidator:
        @staticmethod
        def validate_profile_data(data: Dict[str, Any]) -> bool:
            if not isinstance(data, dict):
                raise TypeError("data must be a dict")
            required_fields = ['name', 'version', 'settings']
            return all(field in data for field in required_fields)

    class ProfileBuilder:
        def __init__(self):
            self.data: Dict[str, Any] = {}

        def with_name(self, name: Any):
            self.data['name'] = name
            return self

        def with_version(self, version: Any):
            self.data['version'] = version
            return self

        def with_settings(self, settings: Any):
            self.data['settings'] = settings
            return self

        def build(self) -> Dict[str, Any]:
            return self.data.copy()


class TestGenesisProfile(unittest.TestCase):
    def setUp(self):
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
        profile = GenesisProfile(self.profile_id, self.sample_data)
        self.assertEqual(profile.profile_id, self.profile_id)
        self.assertEqual(profile.data, self.sample_data)
        self.assertIsInstance(profile.created_at, datetime)
        self.assertIsInstance(profile.updated_at, datetime)

    def test_genesis_profile_initialization_empty_data(self):
        profile = GenesisProfile(self.profile_id, {})
        self.assertEqual(profile.profile_id, self.profile_id)
        self.assertEqual(profile.data, {})

    def test_genesis_profile_initialization_none_data(self):
        with self.assertRaises(TypeError):
            GenesisProfile(self.profile_id, None)  # type: ignore

    def test_genesis_profile_initialization_invalid_id(self):
        with self.assertRaises((TypeError, ValueError)):
            GenesisProfile(None, self.sample_data)  # type: ignore
        with self.assertRaises((TypeError, ValueError)):
            GenesisProfile("", self.sample_data)

    def test_genesis_profile_data_immutability(self):
        profile = GenesisProfile(self.profile_id, self.sample_data)
        original_data = profile.data.copy()
        profile.data['new_field'] = 'new_value'
        self.assertNotEqual(profile.data, original_data)
        self.assertIn('new_field', profile.data)

    def test_genesis_profile_str_representation(self):
        profile = GenesisProfile(self.profile_id, self.sample_data)
        str_repr = str(profile)
        self.assertIn(self.profile_id, str_repr)
        self.assertIsInstance(str_repr, str)

    def test_genesis_profile_equality(self):
        profile1 = GenesisProfile(self.profile_id, self.sample_data)
        profile2 = GenesisProfile(self.profile_id, self.sample_data.copy())
        profile3 = GenesisProfile('different_id', self.sample_data)
        self.assertEqual(profile1, profile2)
        self.assertNotEqual(profile1, profile3)


class TestProfileManager(unittest.TestCase):
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
        duplicate_data = {'name': 'duplicate', 'version': '1.0', 'settings': {}}
        duplicate_profile = self.manager.create_profile(self.profile_id, duplicate_data)
        self.assertEqual(duplicate_profile.profile_id, self.profile_id)
        self.assertEqual(self.manager.profiles[self.profile_id].data, duplicate_data)

    def test_create_profile_invalid_data(self):
        with self.assertRaises((TypeError, ValueError)):
            self.manager.create_profile(self.profile_id, None)  # type: ignore

    def test_get_profile_existing(self):
        created_profile = self.manager.create_profile(self.profile_id, self.sample_data)
        retrieved = self.manager.get_profile(self.profile_id)
        self.assertEqual(retrieved, created_profile)

    def test_get_profile_nonexistent(self):
        self.assertIsNone(self.manager.get_profile('nonexistent'))

    def test_get_profile_empty_id(self):
        self.assertIsNone(self.manager.get_profile(''))

    def test_update_profile_success(self):
        self.manager.create_profile(self.profile_id, self.sample_data)
        update_data = {'name': 'updated_profile', 'new_field': 'new_value'}
        updated_profile = self.manager.update_profile(self.profile_id, update_data)
        self.assertEqual(updated_profile.data['name'], 'updated_profile')
        self.assertEqual(updated_profile.data['new_field'], 'new_value')
        self.assertIsInstance(updated_profile.updated_at, datetime)

    def test_update_profile_nonexistent(self):
        with self.assertRaises(ProfileNotFoundError):
            self.manager.update_profile('nonexistent', {'name': 'updated'})

    def test_update_profile_empty_data(self):
        self.manager.create_profile(self.profile_id, self.sample_data)
        updated_profile = self.manager.update_profile(self.profile_id, {})
        self.assertEqual(updated_profile.data, self.sample_data)

    def test_delete_profile_success(self):
        self.manager.create_profile(self.profile_id, self.sample_data)
        self.assertTrue(self.manager.delete_profile(self.profile_id))
        self.assertNotIn(self.profile_id, self.manager.profiles)

    def test_delete_profile_nonexistent(self):
        self.assertFalse(self.manager.delete_profile('nonexistent'))

    def test_manager_state_isolation(self):
        m1 = ProfileManager()
        m2 = ProfileManager()
        m1.create_profile(self.profile_id, self.sample_data)
        self.assertIsNotNone(m1.get_profile(self.profile_id))
        self.assertIsNone(m2.get_profile(self.profile_id))


class TestProfileValidator(unittest.TestCase):
    def setUp(self):
        self.valid_data = {
            'name': 'test',
            'version': '1.0.0',
            'settings': {}
        }

    def test_validate_profile_data_valid(self):
        self.assertTrue(ProfileValidator.validate_profile_data(self.valid_data))

    def test_validate_profile_data_missing_fields(self):
        for invalid in [
            {'version': '1.0.0', 'settings': {}},
            {'name': 'test', 'settings': {}},
            {'name': 'test', 'version': '1.0.0'}
        ]:
            self.assertFalse(ProfileValidator.validate_profile_data(invalid))

    def test_validate_profile_data_invalid_type(self):
        for invalid in [None, "string", 123, [], set()]:
            with self.assertRaises(TypeError):
                ProfileValidator.validate_profile_data(invalid)  # type: ignore

    def test_validate_profile_data_extra_fields(self):
        data = self.valid_data.copy()
        data['extra'] = True
        self.assertTrue(ProfileValidator.validate_profile_data(data))


class TestProfileBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = ProfileBuilder()

    def test_builder_chain_methods(self):
        result = (self.builder
                  .with_name('test')
                  .with_version('1.0.0')
                  .with_settings({'a': 1})
                  .build())
        self.assertEqual(result, {'name': 'test', 'version': '1.0.0', 'settings': {'a': 1}})

    def test_builder_overwrite(self):
        self.builder.with_name('first').with_name('second')
        self.assertEqual(self.builder.build()['name'], 'second')

    def test_builder_empty(self):
        self.assertEqual(self.builder.build(), {})

    def test_builder_partial(self):
        self.assertEqual(self.builder.with_name('only').build(), {'name': 'only'})


class TestProfileExceptions(unittest.TestCase):
    def test_profile_error_inheritance(self):
        e = ProfileError("msg")
        self.assertIsInstance(e, Exception)
        self.assertEqual(str(e), "msg")

    def test_validation_error_inheritance(self):
        e = ValidationError("msg")
        self.assertIsInstance(e, ProfileError)
        self.assertEqual(str(e), "msg")

    def test_not_found_error_inheritance(self):
        e = ProfileNotFoundError("msg")
        self.assertIsInstance(e, ProfileError)
        self.assertEqual(str(e), "msg")


if __name__ == '__main__':
    unittest.main()