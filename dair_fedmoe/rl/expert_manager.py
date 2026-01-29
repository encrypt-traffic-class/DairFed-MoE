"""
Reinforcement learning module for expert lifecycle management in DAIR-FedMoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np
from dataclasses import dataclass

from ..config import ModelConfig

@dataclass
class ExpertState:
    """State representation for expert management."""
    utilization: torch.Tensor  # Expert utilization rates
    performance: torch.Tensor  # Expert performance metrics
    drift_metrics: torch.Tensor  # Drift detection metrics
    resource_usage: torch.Tensor  # Resource usage metrics

class ExpertPolicyNetwork(nn.Module):
    """Policy network for expert lifecycle management."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config.num_experts * 4, 256),  # 4 metrics per expert
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_experts * 3)  # 3 actions per expert
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(
        self,
        state: ExpertState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate state features
        state_features = torch.cat([
            state.utilization,
            state.performance,
            state.drift_metrics,
            state.resource_usage
        ], dim=-1)
        
        # Encode state
        encoded_state = self.state_encoder(state_features)
        
        # Get policy and value
        policy_logits = self.policy_head(encoded_state)
        value = self.value_head(encoded_state)
        
        return policy_logits, value

class ExpertManager:
    """Expert lifecycle manager using PPO."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.policy_network = ExpertPolicyNetwork(config)
        
        # PPO hyperparameters
        self.clip_ratio = config.ppo_clip_ratio
        self.value_coef = config.ppo_value_coef
        self.entropy_coef = config.ppo_entropy_coef
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate
        )
        
    def get_action(
        self,
        state: ExpertState,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from policy network."""
        policy_logits, value = self.policy_network(state)
        
        # Reshape policy logits
        policy_logits = policy_logits.view(-1, self.config.num_experts, 3)
        
        if deterministic:
            action = torch.argmax(policy_logits, dim=-1)
        else:
            action_probs = F.softmax(policy_logits, dim=-1)
            action = torch.multinomial(action_probs.view(-1, 3), 1)
            action = action.view(-1, self.config.num_experts)
            
        return action, policy_logits, value
    
    def compute_loss(
        self,
        old_policy_logits: torch.Tensor,
        new_policy_logits: torch.Tensor,
        value: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss."""
        # Policy loss
        old_probs = F.softmax(old_policy_logits, dim=-1)
        new_probs = F.softmax(new_policy_logits, dim=-1)
        
        ratio = new_probs / (old_probs + 1e-8)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # Value loss
        value_loss = F.mse_loss(value, returns)
        
        # Entropy loss
        entropy = -torch.sum(new_probs * torch.log(new_probs + 1e-8), dim=-1).mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss -
            self.entropy_coef * entropy
        )
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
        
        return total_loss, metrics
    
    def update(
        self,
        states: List[ExpertState],
        actions: torch.Tensor,
        old_policy_logits: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> Dict[str, float]:
        """Update policy network using PPO."""
        self.optimizer.zero_grad()
        
        # Get new policy logits and value
        new_policy_logits, value = self.policy_network(states[-1])
        
        # Compute loss
        loss, metrics = self.compute_loss(
            old_policy_logits,
            new_policy_logits,
            value,
            returns,
            advantages
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer']) 