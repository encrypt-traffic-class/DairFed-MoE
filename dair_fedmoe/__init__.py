"""
DAIR-FedMoE: Hierarchical MoE for Federated Encrypted Traffic Classification
under Distributed Feature, Concept, and Label Drift
"""

__version__ = "0.1.0"

from .models import GShardTransformer, HierarchicalMoE
from .drift import DriftDetector, DriftScorer
from .rl import PolicyNetwork, ExpertManager
from .training import FederatedTrainer, LocalTrainer
from .privacy import DPMechanism, SecureAggregation

__all__ = [
    'GShardTransformer',
    'HierarchicalMoE',
    'DriftDetector',
    'DriftScorer',
    'PolicyNetwork',
    'ExpertManager',
    'FederatedTrainer',
    'LocalTrainer',
    'DPMechanism',
    'SecureAggregation',
] 