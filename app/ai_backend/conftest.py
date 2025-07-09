import pytest
import sys
import os

# Add the app directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

@pytest.fixture
def sample_ethical_framework():
    """
    Return an EthicalFramework instance named "TestFramework" with the principles fairness, transparency, accountability, and privacy for use in tests.
    
    Returns:
        EthicalFramework: The test ethical framework instance.
    """
    from app.ai_backend.test_genesis_ethical_governor import EthicalFramework
    return EthicalFramework(
        "TestFramework",
        ["fairness", "transparency", "accountability", "privacy"]
    )

@pytest.fixture
def sample_governance_policies():
    """
    Return a list of sample GovernancePolicy instances representing privacy and safety policies.
    
    Returns:
        List[GovernancePolicy]: Two policies, one for privacy with rules on PII and data minimization, and one for safety with harm prevention and risk assessment rules.
    """
    from app.ai_backend.test_genesis_ethical_governor import GovernancePolicy
    return [
        GovernancePolicy("privacy", ["no_pii_without_consent", "data_minimization"]),
        GovernancePolicy("safety", ["harm_prevention", "risk_assessment"])
    ]

@pytest.fixture
def ethical_governor(sample_ethical_framework, sample_governance_policies):
    """
    Pytest fixture that provides an EthicalGovernor instance initialized with sample ethical framework and governance policies.
    
    Returns:
        EthicalGovernor: An instance configured for use in ethical governance tests.
    """
    from app.ai_backend.test_genesis_ethical_governor import EthicalGovernor
    return EthicalGovernor(sample_ethical_framework, sample_governance_policies)