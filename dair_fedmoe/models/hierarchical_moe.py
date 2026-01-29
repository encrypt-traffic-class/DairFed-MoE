"""
Hierarchical Mixture of Experts (HMoE) layer implementation for DAIR-FedMoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from ..config import ModelConfig

class Expert(nn.Module):
    """Single expert network in the MoE layer."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.hidden_dim, config.expert_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.expert_dim, config.hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TopLevelRouter(nn.Module):
    """Top-level router for traffic routing."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(config.hidden_dim, config.num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.router(x)

class BottomLevelRouter(nn.Module):
    """Bottom-level router for drift detection."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(config.hidden_dim, config.num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.router(x)

class HierarchicalMoE(nn.Module):
    """Hierarchical Mixture of Experts layer."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(config.num_experts)
        ])
        
        # Create routers
        self.top_router = TopLevelRouter(config)
        self.bottom_router = BottomLevelRouter(config)
        
        # Load balancing loss
        self.aux_loss_weight = config.aux_loss_weight
        
    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_dim = x.size()
        
        # Reshape for expert processing
        x_reshaped = x.view(-1, hidden_dim)
        
        # Get routing probabilities
        top_probs = self.top_router(x_reshaped)  # [batch_size * seq_len, num_experts]
        bottom_probs = self.bottom_router(x_reshaped)  # [batch_size * seq_len, num_experts]
        
        # Combine routing probabilities
        routing_probs = top_probs * bottom_probs
        
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x_reshaped))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size * seq_len, num_experts, hidden_dim]
        
        # Weighted sum of expert outputs
        weighted_output = torch.bmm(
            routing_probs.unsqueeze(1),
            expert_outputs
        ).squeeze(1)  # [batch_size * seq_len, hidden_dim]
        
        # Reshape back to original dimensions
        output = weighted_output.view(batch_size, seq_len, hidden_dim)
        
        if return_aux_loss:
            # Calculate load balancing loss
            top_importance = top_probs.sum(0) / top_probs.size(0)
            bottom_importance = bottom_probs.sum(0) / bottom_probs.size(0)
            
            top_balance_loss = torch.sum(top_importance * torch.log(top_importance + 1e-6))
            bottom_balance_loss = torch.sum(bottom_importance * torch.log(bottom_importance + 1e-6))
            
            aux_loss = self.aux_loss_weight * (top_balance_loss + bottom_balance_loss)
            return output, aux_loss
        
        return output, None 