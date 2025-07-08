import pytest
import sys
import os

# Add the app directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

@pytest.fixture
def sample_ethical_framework():
    """
    Pytest fixture that returns an `EthicalFramework` instance with sample ethical principles for testing.
    """
    from app.ai_backend.test_genesis_ethical_governor import EthicalFramework
    return EthicalFramework(
        "TestFramework",
        ["fairness", "transparency", "accountability", "privacy"]
    )

@pytest.fixture
def sample_governance_policies():
    """
    Return a list of sample GovernancePolicy instances representing example privacy and safety policies for testing.
    
    Returns:
        List[GovernancePolicy]: Example governance policies for use in tests.
    """
    from app.ai_backend.test_genesis_ethical_governor import GovernancePolicy
    return [
        GovernancePolicy("privacy", ["no_pii_without_consent", "data_minimization"]),
        GovernancePolicy("safety", ["harm_prevention", "risk_assessment"])
    ]

@pytest.fixture
def ethical_governor(sample_ethical_framework, sample_governance_policies):
    """
    Pytest fixture that provides an EthicalGovernor instance initialized with sample ethical framework and governance policies for testing purposes.
    """
    from app.ai_backend.test_genesis_ethical_governor import EthicalGovernor
    return EthicalGovernor(sample_ethical_framework, sample_governance_policies)