"""
GShard Transformer backbone implementation for DAIR-FedMoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..config import ModelConfig

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(context)

class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * config.ff_expansion),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * config.ff_expansion, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x

class GShardTransformer(nn.Module):
    """GShard Transformer backbone for DAIR-FedMoE."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, 1000, config.hidden_dim)  # Max sequence length of 1000
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
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
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Token embedding
        x = self.token_embedding(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
            
        # Classification head
        x = self.classifier(x)
        
        return x 