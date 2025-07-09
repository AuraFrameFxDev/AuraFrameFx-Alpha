import pytest
import sys
import os

# Add the app directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

@pytest.fixture
def sample_ethical_framework():
    """
    Provides a pytest fixture that returns an `EthicalFramework` instance initialized with the name "TestFramework" and a set of sample ethical principles for use in tests.
    
    Returns:
        EthicalFramework: An instance with predefined ethical principles for testing purposes.
    """
    from app.ai_backend.test_genesis_ethical_governor import EthicalFramework
    return EthicalFramework(
        "TestFramework",
        ["fairness", "transparency", "accountability", "privacy"]
    )

@pytest.fixture
def sample_governance_policies():
    """
    Return a list of sample GovernancePolicy objects for use in tests.
    
    The returned list includes example privacy and safety policies, each initialized with representative rules.
    
    Returns:
        List[GovernancePolicy]: Example governance policies for testing purposes.
    """
    from app.ai_backend.test_genesis_ethical_governor import GovernancePolicy
    return [
        GovernancePolicy("privacy", ["no_pii_without_consent", "data_minimization"]),
        GovernancePolicy("safety", ["harm_prevention", "risk_assessment"])
    ]

@pytest.fixture
def ethical_governor(sample_ethical_framework, sample_governance_policies):
    """
    Pytest fixture that returns an EthicalGovernor instance initialized with a sample ethical framework and governance policies for testing.
    """
    from app.ai_backend.test_genesis_ethical_governor import EthicalGovernor
    return EthicalGovernor(sample_ethical_framework, sample_governance_policies)