"""
Main DAIR-FedMoE model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from ..config import ModelConfig
from .gshard_transformer import GShardTransformer
from .hierarchical_moe import HierarchicalMoE

class DAIRFedMoE(nn.Module):
    """DAIR-FedMoE model for federated encrypted traffic classification."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # GShard Transformer backbone
        self.transformer = GShardTransformer(config)
        
        # Hierarchical MoE layer
        self.hmoe = HierarchicalMoE(config)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Get transformer features
        features = self.transformer(x, mask)
        
        # Apply HMoE layer
        hmoe_output, aux_loss = self.hmoe(features, return_aux_loss)
        
        # Final classification
        logits = self.classifier(hmoe_output)
        
        # Prepare output dictionary
        outputs = {
            'logits': logits,
            'features': features,
            'hmoe_output': hmoe_output
        }
        
        if return_aux_loss and aux_loss is not None:
            outputs['aux_loss'] = aux_loss
            
        return outputs
    
    def get_expert_utilization(self) -> Dict[str, torch.Tensor]:
        """Get expert utilization statistics."""
        with torch.no_grad():
            # Get routing probabilities from both routers
            top_router = self.hmoe.top_router
            bottom_router = self.hmoe.bottom_router
            
            # Create dummy input for router analysis
            dummy_input = torch.randn(1, self.config.hidden_dim)
            
            # Get routing probabilities
            top_probs = top_router(dummy_input)
            bottom_probs = bottom_router(dummy_input)
            
            return {
                'top_router_probs': top_probs,
                'bottom_router_probs': bottom_probs
            }
    
    def get_model_size(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Get the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 