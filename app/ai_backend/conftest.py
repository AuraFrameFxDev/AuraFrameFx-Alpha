import pytest
import sys
import os

# Add the app directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

@pytest.fixture
def sample_ethical_framework():
    """
    Provides a pytest fixture that returns an `EthicalFramework` instance initialized with a test framework name and a set of ethical principles for use in unit tests.
    
    Returns:
        EthicalFramework: An instance configured with the name "TestFramework" and principles: fairness, transparency, accountability, and privacy.
    """
    from app.ai_backend.test_genesis_ethical_governor import EthicalFramework
    return EthicalFramework(
        "TestFramework",
        ["fairness", "transparency", "accountability", "privacy"]
    )

@pytest.fixture
def sample_governance_policies():
    """
    Provides a pytest fixture that returns a list of sample GovernancePolicy objects for testing.
    
    Returns:
        List of GovernancePolicy: Sample policies targeting privacy and safety, each with associated rules.
    """
    from app.ai_backend.test_genesis_ethical_governor import GovernancePolicy
    return [
        GovernancePolicy("privacy", ["no_pii_without_consent", "data_minimization"]),
        GovernancePolicy("safety", ["harm_prevention", "risk_assessment"])
    ]

@pytest.fixture
def ethical_governor(sample_ethical_framework, sample_governance_policies):
    """
    Pytest fixture that returns an EthicalGovernor instance configured with a sample ethical framework and governance policies.
    
    Returns:
        EthicalGovernor: An instance initialized with the provided sample ethical framework and governance policies for use in tests.
    """
    from app.ai_backend.test_genesis_ethical_governor import EthicalGovernor
    return EthicalGovernor(sample_ethical_framework, sample_governance_policies)