"""
Privacy mechanisms for DAIR-FedMoE including differential privacy and secure aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np
from dataclasses import dataclass

from ..config import ModelConfig

@dataclass
class PrivacyBudget:
    """Privacy budget for differential privacy."""
    epsilon: float
    delta: float

class GaussianNoise:
    """Gaussian noise mechanism for differential privacy."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.sensitivity = config.dp_sensitivity
        self.epsilon = config.dp_epsilon
        self.delta = config.dp_delta
        
    def compute_noise_scale(self) -> float:
        """Compute noise scale based on privacy budget."""
        # Using analytical Gaussian mechanism
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to tensor."""
        noise_scale = self.compute_noise_scale()
        noise = torch.randn_like(x) * noise_scale
        return x + noise

class SecureAggregator:
    """Secure aggregation for federated learning."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.num_clients = config.num_clients
        self.mask_generator = torch.Generator()
        self.mask_generator.manual_seed(config.seed)
        
    def generate_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random masks for secure aggregation."""
        # Generate random masks for each client
        masks = []
        for _ in range(self.num_clients):
            mask = torch.randn(
                self.config.hidden_dim,
                generator=self.mask_generator
            )
            masks.append(mask)
            
        # Compute pairwise masks
        pairwise_masks = torch.zeros(
            self.num_clients,
            self.num_clients,
            self.config.hidden_dim
        )
        
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                pairwise_mask = torch.randn(
                    self.config.hidden_dim,
                    generator=self.mask_generator
                )
                pairwise_masks[i, j] = pairwise_mask
                pairwise_masks[j, i] = -pairwise_mask
                
        return torch.stack(masks), pairwise_masks
    
    def aggregate(
        self,
        client_updates: List[torch.Tensor],
        masks: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Securely aggregate client updates."""
        client_masks, pairwise_masks = masks
        
        # Apply masks to client updates
        masked_updates = []
        for i, update in enumerate(client_updates):
            # Add client's own mask
            masked_update = update + client_masks[i]
            
            # Add pairwise masks
            for j in range(self.num_clients):
                if i != j:
                    masked_update = masked_update + pairwise_masks[i, j]
                    
            masked_updates.append(masked_update)
            
        # Aggregate masked updates
        aggregated = torch.stack(masked_updates).mean(dim=0)
        
        return aggregated

class PrivacyManager:
    """Privacy manager combining differential privacy and secure aggregation."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.noise_mechanism = GaussianNoise(config)
        self.secure_aggregator = SecureAggregator(config)
        
    def apply_privacy(
        self,
        client_updates: List[torch.Tensor],
        privacy_budget: Optional[PrivacyBudget] = None
    ) -> torch.Tensor:
        """Apply privacy mechanisms to client updates."""
        # Update privacy budget if provided
        if privacy_budget is not None:
            self.noise_mechanism.epsilon = privacy_budget.epsilon
            self.noise_mechanism.delta = privacy_budget.delta
            
        # Generate masks for secure aggregation
        masks = self.secure_aggregator.generate_masks()
        
        # Add noise to client updates
        noisy_updates = [
            self.noise_mechanism.add_noise(update)
            for update in client_updates
        ]
        
        # Securely aggregate updates
        aggregated = self.secure_aggregator.aggregate(noisy_updates, masks)
        
        return aggregated
    
    def get_privacy_budget(self) -> PrivacyBudget:
        """Get current privacy budget."""
        return PrivacyBudget(
            epsilon=self.noise_mechanism.epsilon,
            delta=self.noise_mechanism.delta
        ) 