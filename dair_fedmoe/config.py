"""
Configuration parameters for DAIR-FedMoE framework.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import yaml

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Transformer parameters
    num_layers: int = 6
    hidden_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    num_classes: int = 2  # VPN vs non-VPN
    
    # Input features
    max_packets_per_flow: int = 10
    max_payload_size: int = 1500  # Ethernet MTU
    num_header_features: int = 20  # TCP/UDP header size
    num_statistical_features: int = 10  # Statistical features per packet
    
    # Calculate input dimension
    @property
    def input_dim(self) -> int:
        """Calculate input dimension based on features."""
        # Header features (per packet)
        header_dim = self.num_header_features * self.max_packets_per_flow
        
        # Payload features (per packet)
        payload_dim = self.max_payload_size * self.max_packets_per_flow
        
        # Statistical features (per packet)
        stats_dim = self.num_statistical_features * self.max_packets_per_flow
        
        # Flow-level features
        flow_dim = 5  # Basic flow features (duration, bytes, etc.)
        
        return header_dim + payload_dim + stats_dim + flow_dim
    
    # MoE parameters
    num_experts: int = 8
    expert_capacity: int = 64
    router_jitter: float = 0.01
    router_loss_weight: float = 0.01
    
    # Drift detection
    drift_window_size: int = 500
    drift_smoothing_factor: float = 0.95
    
    # Loss reweighting
    min_weight: float = 1.0
    max_weight: float = 5.0
    weight_epsilon: float = 1e-6
    
    # Privacy configuration
    privacy_budget: float = 1.0
    privacy_delta: float = 1e-5
    clip_norm: float = 1.0
    
    # Training configuration
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 10
    
    # PCAP processing
    flow_timeout: int = 600  # 10 minutes in seconds
    include_headers: bool = True
    include_stats: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'num_classes': self.num_classes,
            'max_packets_per_flow': self.max_packets_per_flow,
            'max_payload_size': self.max_payload_size,
            'num_header_features': self.num_header_features,
            'num_statistical_features': self.num_statistical_features,
            'num_experts': self.num_experts,
            'expert_capacity': self.expert_capacity,
            'router_jitter': self.router_jitter,
            'router_loss_weight': self.router_loss_weight,
            'drift_window_size': self.drift_window_size,
            'drift_smoothing_factor': self.drift_smoothing_factor,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'weight_epsilon': self.weight_epsilon,
            'privacy_budget': self.privacy_budget,
            'privacy_delta': self.privacy_delta,
            'clip_norm': self.clip_norm,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'flow_timeout': self.flow_timeout,
            'include_headers': self.include_headers,
            'include_stats': self.include_stats
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
        
    def save_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f)

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Federated learning
    num_clients: int = 20
    num_rounds: int = 250
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # RL parameters
    ppo_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    
    # Reward weights
    drift_reward_weight: float = 2.0
    expert_count_weight: float = 0.5

@dataclass
class PrivacyConfig:
    """Privacy configuration."""
    # Differential privacy
    clip_norm: float = 1.0
    noise_scale: float = 1.2
    delta: float = 1e-5
    
    # Secure aggregation
    use_secure_agg: bool = True
    min_clients: int = 10

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    # ISCX-VPN
    iscx_vpn_classes: int = 14
    iscx_vpn_samples_per_client: Tuple[int, int] = (3000, 5000)
    
    # ISCX-Tor
    iscx_tor_classes: int = 20
    iscx_tor_samples_per_client: Tuple[int, int] = (3000, 4500)
    
    # CIC-IDS2017
    cic_ids_classes: int = 10
    cic_ids_samples_per_client: Tuple[int, int] = (2500, 3500)
    
    # UNSW-NB15
    unsw_nb15_classes: int = 10
    unsw_nb15_samples_per_client: Tuple[int, int] = (3000, 4000)

@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    dataset: DatasetConfig = DatasetConfig()
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.model.num_experts > 0, "Number of experts must be positive"
        assert self.training.num_clients > 0, "Number of clients must be positive"
        assert self.training.num_rounds > 0, "Number of rounds must be positive"
        assert self.privacy.clip_norm > 0, "Clip norm must be positive"
        assert self.privacy.noise_scale > 0, "Noise scale must be positive" 