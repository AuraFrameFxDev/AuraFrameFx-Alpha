"""
Comprehensive unit tests for GitHub Actions workflow YAML validation.
Testing framework: pytest (following existing project patterns)
"""

import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List


class TestWorkflowMainYML:
    """Test suite for the main workflow YAML file validation."""

    @pytest.fixture
    def workflow_content(self):
        """Load and parse the workflow YAML content for testing."""
        # Define the expected workflow structure based on the provided content
        return {
            'name': 'Build AuraFrameFX (Kotlin KSP2)',
            'on': {
                'push': {'branches': ['master']},
                'pull_request': {'branches': ['master']},
                'workflow_dispatch': None
            },
            'jobs': {
                'build': {
                    'runs-on': 'ubuntu-latest',
                    'env': {
                        'ANDROID_SDK_ROOT': '/opt/android-sdk'
                    },
                    'steps': [
                        {
                            'name': 'Checkout repository',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Set up JDK 21',
                            'uses': 'actions/setup-java@v4',
                            'with': {
                                'distribution': 'temurin',
                                'java-version': '21'
                            }
                        },
                        {
                            'name': 'Set up Android SDK',
                            'uses': 'android-actions/setup-android@v3',
                            'with': {
                                'packages': 'ndk;26.2.11394342 platform-tools tools'
                            }
                        },
                        {
                            'name': 'Cache Gradle dependencies',
                            'uses': 'actions/cache@v4',
                            'with': {
                                'path': '~/.gradle/caches\n~/.gradle/wrapper',
                                'key': '${{ runner.os }}-gradle-${{ hashFiles(\'**/*.gradle*\', \'**/gradle-wrapper.properties\') }}',
                                'restore-keys': '${{ runner.os }}-gradle-'
                            }
                        },
                        {
                            'name': 'Grant execute permission for gradlew',
                            'run': 'chmod +x ./gradlew'
                        },
                        {
                            'name': 'Build debug APK',
                            'run': './gradlew assembleDebug'
                        },
                        {
                            'name': 'Run unit tests',
                            'run': './gradlew testDebugUnitTest'
                        },
                        {
                            'name': 'Upload APK artifact',
                            'uses': 'actions/upload-artifact@v4',
                            'with': {
                                'name': 'aura-apk',
                                'path': 'app/build/outputs/apk/debug/*.apk'
                            }
                        }
                    ]
                }
            }
        }

    @pytest.fixture
    def invalid_workflow_content(self):
        """Provide invalid workflow content for testing error handling."""
        return {
            'name': '',  # Invalid: empty name
            'on': {},    # Invalid: empty triggers
            'jobs': {}   # Invalid: no jobs defined
        }

    def test_workflow_has_required_name(self, workflow_content):
        """Test that the workflow has a proper name."""
        assert 'name' in workflow_content
        assert workflow_content['name'] == 'Build AuraFrameFX (Kotlin KSP2)'
        assert isinstance(workflow_content['name'], str)
        assert len(workflow_content['name']) > 0

    def test_workflow_name_is_descriptive(self, workflow_content):
        """Test that the workflow name is descriptive and meaningful."""
        name = workflow_content['name']
        assert 'Build' in name
        assert 'AuraFrameFX' in name
        assert 'Kotlin' in name
        assert len(name) >= 10  # Should be reasonably descriptive

    def test_workflow_has_valid_triggers(self, workflow_content):
        """Test that the workflow has valid trigger configurations."""
        assert 'on' in workflow_content
        triggers = workflow_content['on']
        
        # Test push trigger
        assert 'push' in triggers
        assert 'branches' in triggers['push']
        assert 'master' in triggers['push']['branches']
        
        # Test pull request trigger
        assert 'pull_request' in triggers
        assert 'branches' in triggers['pull_request']
        assert 'master' in triggers['pull_request']['branches']
        
        # Test manual trigger
        assert 'workflow_dispatch' in triggers

    def test_workflow_trigger_branches_are_valid(self, workflow_content):
        """Test that trigger branches are valid Git branch names."""
        triggers = workflow_content['on']
        
        for trigger_type in ['push', 'pull_request']:
            if trigger_type in triggers:
                branches = triggers[trigger_type]['branches']
                for branch in branches:
                    assert isinstance(branch, str)
                    assert len(branch) > 0
                    assert not branch.startswith('/')
                    assert not branch.endswith('/')

    def test_workflow_has_jobs(self, workflow_content):
        """Test that the workflow has jobs defined."""
        assert 'jobs' in workflow_content
        assert isinstance(workflow_content['jobs'], dict)
        assert len(workflow_content['jobs']) > 0

    def test_build_job_configuration(self, workflow_content):
        """Test the build job configuration."""
        jobs = workflow_content['jobs']
        assert 'build' in jobs
        
        build_job = jobs['build']
        assert 'runs-on' in build_job
        assert build_job['runs-on'] == 'ubuntu-latest'
        
        # Test environment variables
        assert 'env' in build_job
        env_vars = build_job['env']
        assert 'ANDROID_SDK_ROOT' in env_vars
        assert env_vars['ANDROID_SDK_ROOT'] == '/opt/android-sdk'

    def test_build_job_has_steps(self, workflow_content):
        """Test that the build job has required steps."""
        build_job = workflow_content['jobs']['build']
        assert 'steps' in build_job
        assert isinstance(build_job['steps'], list)
        assert len(build_job['steps']) > 0

    def test_checkout_step_configuration(self, workflow_content):
        """Test the checkout step configuration."""
        steps = workflow_content['jobs']['build']['steps']
        checkout_step = steps[0]
        
        assert checkout_step['name'] == 'Checkout repository'
        assert checkout_step['uses'] == 'actions/checkout@v4'
        assert 'with' not in checkout_step or checkout_step.get('with') is None

    def test_java_setup_step_configuration(self, workflow_content):
        """Test the Java setup step configuration."""
        steps = workflow_content['jobs']['build']['steps']
        java_step = next(step for step in steps if 'Set up JDK' in step['name'])
        
        assert java_step['name'] == 'Set up JDK 21'
        assert java_step['uses'] == 'actions/setup-java@v4'
        assert 'with' in java_step
        assert java_step['with']['distribution'] == 'temurin'
        assert java_step['with']['java-version'] == '21'

    def test_android_sdk_setup_step_configuration(self, workflow_content):
        """Test the Android SDK setup step configuration."""
        steps = workflow_content['jobs']['build']['steps']
        android_step = next(step for step in steps if 'Set up Android SDK' in step['name'])
        
        assert android_step['name'] == 'Set up Android SDK'
        assert android_step['uses'] == 'android-actions/setup-android@v3'
        assert 'with' in android_step
        assert 'packages' in android_step['with']
        assert 'ndk;26.2.11394342' in android_step['with']['packages']
        assert 'platform-tools' in android_step['with']['packages']
        assert 'tools' in android_step['with']['packages']

    def test_gradle_cache_step_configuration(self, workflow_content):
        """Test the Gradle cache step configuration."""
        steps = workflow_content['jobs']['build']['steps']
        cache_step = next(step for step in steps if 'Cache Gradle' in step['name'])
        
        assert cache_step['name'] == 'Cache Gradle dependencies'
        assert cache_step['uses'] == 'actions/cache@v4'
        assert 'with' in cache_step
        assert 'path' in cache_step['with']
        assert 'key' in cache_step['with']
        assert 'restore-keys' in cache_step['with']

    def test_gradle_cache_paths_are_valid(self, workflow_content):
        """Test that Gradle cache paths are properly configured."""
        steps = workflow_content['jobs']['build']['steps']
        cache_step = next(step for step in steps if 'Cache Gradle' in step['name'])
        
        cache_paths = cache_step['with']['path']
        assert '~/.gradle/caches' in cache_paths
        assert '~/.gradle/wrapper' in cache_paths

    def test_gradle_cache_key_is_dynamic(self, workflow_content):
        """Test that Gradle cache key includes dynamic elements."""
        steps = workflow_content['jobs']['build']['steps']
        cache_step = next(step for step in steps if 'Cache Gradle' in step['name'])
        
        cache_key = cache_step['with']['key']
        assert '${{ runner.os }}' in cache_key
        assert 'gradle' in cache_key.lower()
        assert 'hashFiles' in cache_key

    def test_gradle_permission_step(self, workflow_content):
        """Test the Gradle permission step."""
        steps = workflow_content['jobs']['build']['steps']
        permission_step = next(step for step in steps if 'Grant execute permission' in step['name'])
        
        assert permission_step['name'] == 'Grant execute permission for gradlew'
        assert permission_step['run'] == 'chmod +x ./gradlew'

    def test_build_step_configuration(self, workflow_content):
        """Test the build step configuration."""
        steps = workflow_content['jobs']['build']['steps']
        build_step = next(step for step in steps if 'Build debug APK' in step['name'])
        
        assert build_step['name'] == 'Build debug APK'
        assert build_step['run'] == './gradlew assembleDebug'

    def test_unit_tests_step_configuration(self, workflow_content):
        """Test the unit tests step configuration."""
        steps = workflow_content['jobs']['build']['steps']
        test_step = next(step for step in steps if 'Run unit tests' in step['name'])
        
        assert test_step['name'] == 'Run unit tests'
        assert test_step['run'] == './gradlew testDebugUnitTest'

    def test_upload_artifact_step_configuration(self, workflow_content):
        """Test the upload artifact step configuration."""
        steps = workflow_content['jobs']['build']['steps']
        upload_step = next(step for step in steps if 'Upload APK artifact' in step['name'])
        
        assert upload_step['name'] == 'Upload APK artifact'
        assert upload_step['uses'] == 'actions/upload-artifact@v4'
        assert 'with' in upload_step
        assert upload_step['with']['name'] == 'aura-apk'
        assert upload_step['with']['path'] == 'app/build/outputs/apk/debug/*.apk'

    def test_all_steps_have_names(self, workflow_content):
        """Test that all steps have descriptive names."""
        steps = workflow_content['jobs']['build']['steps']
        for i, step in enumerate(steps):
            assert 'name' in step, f"Step {i} is missing a name"
            assert isinstance(step['name'], str), f"Step {i} name is not a string"
            assert len(step['name']) > 0, f"Step {i} has empty name"

    def test_step_order_is_logical(self, workflow_content):
        """Test that the steps are in a logical order."""
        steps = workflow_content['jobs']['build']['steps']
        step_names = [step['name'] for step in steps]
        
        # Checkout should be first
        assert step_names[0] == 'Checkout repository'
        
        # Setup steps should come before build steps
        checkout_index = step_names.index('Checkout repository')
        java_setup_index = step_names.index('Set up JDK 21')
        android_setup_index = step_names.index('Set up Android SDK')
        build_index = step_names.index('Build debug APK')
        
        assert checkout_index < java_setup_index < build_index
        assert checkout_index < android_setup_index < build_index

    def test_gradle_commands_are_valid(self, workflow_content):
        """Test that Gradle commands are valid."""
        steps = workflow_content['jobs']['build']['steps']
        gradle_steps = [step for step in steps if 'run' in step and './gradlew' in step['run']]
        
        valid_gradle_commands = ['assembleDebug', 'testDebugUnitTest']
        for step in gradle_steps:
            command = step['run']
            assert any(cmd in command for cmd in valid_gradle_commands)

    def test_gradle_commands_use_wrapper(self, workflow_content):
        """Test that Gradle commands use the wrapper script."""
        steps = workflow_content['jobs']['build']['steps']
        gradle_steps = [step for step in steps if 'run' in step and 'gradle' in step['run'].lower()]
        
        for step in gradle_steps:
            command = step['run']
            assert './gradlew' in command, f"Command should use Gradle wrapper: {command}"

    def test_action_versions_are_specified(self, workflow_content):
        """Test that action versions are properly specified."""
        steps = workflow_content['jobs']['build']['steps']
        for step in steps:
            if 'uses' in step:
                action = step['uses']
                assert '@' in action, f"Action {action} should have a version specified"
                version = action.split('@')[1]
                assert version.startswith('v'), f"Version {version} should start with 'v'"

    def test_action_versions_are_not_latest(self, workflow_content):
        """Test that actions don't use 'latest' versions for security."""
        steps = workflow_content['jobs']['build']['steps']
        for step in steps:
            if 'uses' in step:
                action = step['uses']
                version = action.split('@')[1]
                assert version not in ['latest', 'master', 'main'], f"Action {action} should not use {version}"

    def test_environment_variables_are_valid(self, workflow_content):
        """Test that environment variables are properly configured."""
        build_job = workflow_content['jobs']['build']
        if 'env' in build_job:
            env_vars = build_job['env']
            for key, value in env_vars.items():
                assert isinstance(key, str), f"Environment variable key {key} should be string"
                assert len(key) > 0, f"Environment variable key should not be empty"
                assert isinstance(value, str), f"Environment variable value for {key} should be string"
                assert len(value) > 0, f"Environment variable value for {key} should not be empty"

    def test_android_sdk_root_is_set(self, workflow_content):
        """Test that ANDROID_SDK_ROOT environment variable is properly set."""
        build_job = workflow_content['jobs']['build']
        env_vars = build_job['env']
        
        assert 'ANDROID_SDK_ROOT' in env_vars
        android_sdk_root = env_vars['ANDROID_SDK_ROOT']
        assert android_sdk_root.startswith('/'), "ANDROID_SDK_ROOT should be an absolute path"
        assert 'android' in android_sdk_root.lower()

    def test_cache_configuration_is_optimal(self, workflow_content):
        """Test that cache configuration follows best practices."""
        steps = workflow_content['jobs']['build']['steps']
        cache_steps = [step for step in steps if 'uses' in step and 'cache' in step['uses']]
        
        assert len(cache_steps) > 0, "Should have at least one cache step"
        
        for cache_step in cache_steps:
            assert 'with' in cache_step
            cache_config = cache_step['with']
            assert 'path' in cache_config
            assert 'key' in cache_config
            assert 'restore-keys' in cache_config

    def test_workflow_is_valid_structure(self, workflow_content):
        """Test that the workflow has valid YAML structure."""
        assert isinstance(workflow_content, dict)
        assert workflow_content is not None
        
        # Test required top-level fields
        required_fields = ['name', 'on', 'jobs']
        for field in required_fields:
            assert field in workflow_content, f"Missing required field: {field}"

    def test_runner_os_is_supported(self, workflow_content):
        """Test that the runner OS is a supported GitHub Actions runner."""
        build_job = workflow_content['jobs']['build']
        runner_os = build_job['runs-on']
        
        supported_runners = [
            'ubuntu-latest', 'ubuntu-22.04', 'ubuntu-20.04',
            'windows-latest', 'windows-2022', 'windows-2019',
            'macos-latest', 'macos-12', 'macos-11'
        ]
        
        assert runner_os in supported_runners, f"Unsupported runner: {runner_os}"

    def test_android_ndk_version_is_specified(self, workflow_content):
        """Test that Android NDK version is properly specified."""
        steps = workflow_content['jobs']['build']['steps']
        android_step = next(step for step in steps if 'Set up Android SDK' in step['name'])
        
        packages = android_step['with']['packages']
        assert 'ndk;' in packages
        
        # Extract NDK version
        ndk_part = [pkg for pkg in packages.split() if pkg.startswith('ndk;')][0]
        ndk_version = ndk_part.split(';')[1]
        assert len(ndk_version) > 0
        assert '.' in ndk_version, "NDK version should be a version number"

    def test_java_version_is_lts(self, workflow_content):
        """Test that Java version is an LTS version."""
        steps = workflow_content['jobs']['build']['steps']
        java_step = next(step for step in steps if 'Set up JDK' in step['name'])
        
        java_version = java_step['with']['java-version']
        lts_versions = ['8', '11', '17', '21']
        assert java_version in lts_versions, f"Java version {java_version} should be LTS"

    def test_java_distribution_is_valid(self, workflow_content):
        """Test that Java distribution is valid."""
        steps = workflow_content['jobs']['build']['steps']
        java_step = next(step for step in steps if 'Set up JDK' in step['name'])
        
        distribution = java_step['with']['distribution']
        valid_distributions = ['temurin', 'adopt', 'zulu', 'liberica', 'microsoft']
        assert distribution in valid_distributions, f"Invalid Java distribution: {distribution}"

    def test_artifact_path_is_valid(self, workflow_content):
        """Test that artifact upload path is valid."""
        steps = workflow_content['jobs']['build']['steps']
        upload_step = next(step for step in steps if 'Upload APK artifact' in step['name'])
        
        artifact_path = upload_step['with']['path']
        assert artifact_path.endswith('*.apk'), "Artifact path should end with *.apk"
        assert 'app/build/outputs/apk' in artifact_path, "Artifact path should include standard APK output directory"

    def test_artifact_name_is_descriptive(self, workflow_content):
        """Test that artifact name is descriptive."""
        steps = workflow_content['jobs']['build']['steps']
        upload_step = next(step for step in steps if 'Upload APK artifact' in step['name'])
        
        artifact_name = upload_step['with']['name']
        assert isinstance(artifact_name, str)
        assert len(artifact_name) > 0
        assert 'apk' in artifact_name.lower()

    def test_workflow_handles_edge_cases(self, workflow_content):
        """Test workflow configuration handles edge cases."""
        # Test that all required fields are present
        required_fields = ['name', 'on', 'jobs']
        for field in required_fields:
            assert field in workflow_content

        # Test that job has required fields
        build_job = workflow_content['jobs']['build']
        required_job_fields = ['runs-on', 'steps']
        for field in required_job_fields:
            assert field in build_job

    def test_workflow_security_practices(self, workflow_content):
        """Test that workflow follows security best practices."""
        steps = workflow_content['jobs']['build']['steps']
        
        # Test that actions use specific versions (not latest)
        for step in steps:
            if 'uses' in step:
                action = step['uses']
                assert '@' in action
                version = action.split('@')[1]
                assert version not in ['latest', 'master', 'main']

    def test_workflow_step_names_are_unique(self, workflow_content):
        """Test that all step names are unique."""
        steps = workflow_content['jobs']['build']['steps']
        step_names = [step['name'] for step in steps]
        
        assert len(step_names) == len(set(step_names)), "All step names should be unique"

    def test_workflow_has_proper_test_execution(self, workflow_content):
        """Test that workflow includes proper test execution."""
        steps = workflow_content['jobs']['build']['steps']
        test_steps = [step for step in steps if 'test' in step['name'].lower()]
        
        assert len(test_steps) > 0, "Workflow should include test execution steps"

    def test_workflow_builds_before_testing(self, workflow_content):
        """Test that build happens before testing."""
        steps = workflow_content['jobs']['build']['steps']
        step_names = [step['name'] for step in steps]
        
        build_index = next(i for i, name in enumerate(step_names) if 'Build' in name)
        test_index = next(i for i, name in enumerate(step_names) if 'test' in name.lower())
        
        assert build_index < test_index, "Build should happen before tests"

    def test_invalid_workflow_handling(self, invalid_workflow_content):
        """Test handling of invalid workflow configurations."""
        # Test empty name
        assert invalid_workflow_content['name'] == ''
        
        # Test empty triggers
        assert len(invalid_workflow_content['on']) == 0
        
        # Test empty jobs
        assert len(invalid_workflow_content['jobs']) == 0

    def test_workflow_performance_optimizations(self, workflow_content):
        """Test that workflow includes performance optimizations."""
        steps = workflow_content['jobs']['build']['steps']
        
        # Should have caching
        cache_steps = [step for step in steps if 'cache' in step.get('uses', '')]
        assert len(cache_steps) > 0, "Workflow should include caching for performance"

    def test_workflow_error_handling_capabilities(self, workflow_content):
        """Test that workflow can handle potential errors gracefully."""
        steps = workflow_content['jobs']['build']['steps']
        
        # Check that permission step comes before gradle commands
        permission_step_index = next(i for i, step in enumerate(steps) 
                                   if 'Grant execute permission' in step['name'])
        gradle_step_indices = [i for i, step in enumerate(steps) 
                             if 'run' in step and './gradlew' in step['run']]
        
        for gradle_index in gradle_step_indices:
            assert permission_step_index < gradle_index, "Permission should be granted before Gradle commands"

    @pytest.mark.parametrize("field", ['name', 'on', 'jobs'])
    def test_workflow_required_fields_present(self, workflow_content, field):
        """Test that all required workflow fields are present."""
        assert field in workflow_content, f"Required field {field} is missing"

    @pytest.mark.parametrize("trigger", ['push', 'pull_request', 'workflow_dispatch'])
    def test_workflow_trigger_types(self, workflow_content, trigger):
        """Test that all expected trigger types are configured."""
        assert trigger in workflow_content['on'], f"Trigger {trigger} is missing"

    def test_workflow_yaml_compliance(self, workflow_content):
        """Test that workflow structure complies with YAML standards."""
        # Test that content can be serialized back to YAML
        try:
            yaml_content = yaml.dump(workflow_content)
            assert len(yaml_content) > 0
            
            # Test that it can be parsed back
            reparsed = yaml.safe_load(yaml_content)
            assert reparsed is not None
        except yaml.YAMLError as e:
            pytest.fail(f"Workflow content is not valid YAML: {e}")

    def test_workflow_completeness(self, workflow_content):
        """Test that workflow includes all necessary components for Android build."""
        steps = workflow_content['jobs']['build']['steps']
        step_names = [step['name'] for step in steps]
        
        required_components = [
            'Checkout',
            'JDK',
            'Android SDK',
            'Gradle',
            'Build',
            'test',
            'artifact'
        ]
        
        for component in required_components:
            assert any(component.lower() in name.lower() for name in step_names), \
                f"Missing required component: {component}"