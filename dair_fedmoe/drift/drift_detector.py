"""
Drift detection module for DAIR-FedMoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np
from scipy.stats import entropy

from ..config import ModelConfig

class JensenShannonDivergence:
    """Jensen-Shannon divergence for drift detection."""
    
    @staticmethod
    def compute(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute JSD between two probability distributions."""
        m = 0.5 * (p + q)
        jsd = 0.5 * (
            F.kl_div(m.log(), p, reduction='batchmean') +
            F.kl_div(m.log(), q, reduction='batchmean')
        )
        return jsd

class DriftDetector:
    """Drift detection module using Jensen-Shannon divergence."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.window_size = config.drift_window_size
        self.threshold = config.drift_threshold
        
        # Initialize reference distribution
        self.reference_dist = None
        self.drift_history = []
        
    def update_reference(self, features: torch.Tensor):
        """Update reference distribution with new features."""
        if self.reference_dist is None:
            self.reference_dist = features.detach()
        else:
            # Update reference using exponential moving average
            alpha = self.config.reference_update_rate
            self.reference_dist = alpha * features.detach() + (1 - alpha) * self.reference_dist
            
    def detect_drift(
        self,
        features: torch.Tensor,
        return_metrics: bool = False
    ) -> Tuple[bool, Optional[Dict[str, float]]]:
        """Detect drift in input features."""
        if self.reference_dist is None:
            self.update_reference(features)
            return False, None
            
        # Compute feature distributions
        current_dist = F.softmax(features, dim=-1)
        ref_dist = F.softmax(self.reference_dist, dim=-1)
        
        # Compute Jensen-Shannon divergence
        jsd = JensenShannonDivergence.compute(current_dist, ref_dist)
        
        # Update drift history
        self.drift_history.append(jsd.item())
        if len(self.drift_history) > self.window_size:
            self.drift_history.pop(0)
            
        # Detect drift
        drift_detected = jsd > self.threshold
        
        if return_metrics:
            metrics = {
                'jsd': jsd.item(),
                'drift_history_mean': np.mean(self.drift_history),
                'drift_history_std': np.std(self.drift_history)
            }
            return drift_detected, metrics
            
        return drift_detected, None
    
    def get_drift_metrics(self) -> Dict[str, float]:
        """Get current drift detection metrics."""
        if not self.drift_history:
            return {
                'jsd': 0.0,
                'drift_history_mean': 0.0,
                'drift_history_std': 0.0
            }
            
        return {
            'jsd': self.drift_history[-1],
            'drift_history_mean': np.mean(self.drift_history),
            'drift_history_std': np.std(self.drift_history)
        }
        
    def reset(self):
        """Reset drift detector state."""
        self.reference_dist = None
        self.drift_history = [] 