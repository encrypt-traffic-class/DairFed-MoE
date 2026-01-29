"""
Federated training module for DAIR-FedMoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm

from ..config import ModelConfig, TrainingConfig
from ..models.dair_fedmoe import DAIRFedMoE
from ..drift.drift_detector import DriftDetector
from ..rl.expert_manager import ExpertManager, ExpertState
from ..privacy.privacy_mechanism import PrivacyManager, PrivacyBudget

@dataclass
class ClientState:
    """State for each federated client."""
    model: DAIRFedMoE
    optimizer: torch.optim.Optimizer
    drift_detector: DriftDetector
    local_epoch: int
    performance_metrics: Dict[str, float]

class FederatedTrainer:
    """Federated trainer for DAIR-FedMoE."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig
    ):
        self.model_config = model_config
        self.training_config = training_config
        
        # Initialize components
        self.server_model = DAIRFedMoE(model_config)
        self.expert_manager = ExpertManager(model_config)
        self.privacy_manager = PrivacyManager(model_config)
        
        # Client states
        self.clients: Dict[int, ClientState] = {}
        
        # Training metrics
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'drift_metrics': [],
            'privacy_budget': []
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def add_client(self, client_id: int, client_model: DAIRFedMoE):
        """Add a new client to the federation."""
        optimizer = torch.optim.Adam(
            client_model.parameters(),
            lr=self.training_config.learning_rate
        )
        
        drift_detector = DriftDetector(self.model_config)
        
        self.clients[client_id] = ClientState(
            model=client_model,
            optimizer=optimizer,
            drift_detector=drift_detector,
            local_epoch=0,
            performance_metrics={}
        )
        
    def train_client(
        self,
        client_id: int,
        train_data: torch.utils.data.DataLoader,
        val_data: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, float]:
        """Train a single client."""
        client = self.clients[client_id]
        client.model.train()
        
        epoch_metrics = {
            'train_loss': 0.0,
            'train_acc': 0.0,
            'drift_detected': False
        }
        
        # Training loop
        for batch in tqdm(train_data, desc=f'Training client {client_id}'):
            inputs, targets = batch
            
            # Forward pass
            outputs = client.model(inputs, return_aux_loss=True)
            logits = outputs['logits']
            
            # Compute loss
            loss = F.cross_entropy(logits, targets)
            if 'aux_loss' in outputs:
                loss = loss + outputs['aux_loss']
                
            # Backward pass
            client.optimizer.zero_grad()
            loss.backward()
            client.optimizer.step()
            
            # Update metrics
            epoch_metrics['train_loss'] += loss.item()
            epoch_metrics['train_acc'] += (
                (logits.argmax(dim=-1) == targets).float().mean().item()
            )
            
            # Update drift detector
            client.drift_detector.update_reference(outputs['features'])
            drift_detected, drift_metrics = client.drift_detector.detect_drift(
                outputs['features'],
                return_metrics=True
            )
            epoch_metrics['drift_detected'] |= drift_detected
            
        # Normalize metrics
        num_batches = len(train_data)
        epoch_metrics['train_loss'] /= num_batches
        epoch_metrics['train_acc'] /= num_batches
        
        # Validation if provided
        if val_data is not None:
            val_metrics = self.evaluate_client(client_id, val_data)
            epoch_metrics.update(val_metrics)
            
        # Update client state
        client.local_epoch += 1
        client.performance_metrics.update(epoch_metrics)
        
        return epoch_metrics
    
    def evaluate_client(
        self,
        client_id: int,
        val_data: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate a single client."""
        client = self.clients[client_id]
        client.model.eval()
        
        metrics = {
            'val_loss': 0.0,
            'val_acc': 0.0
        }
        
        with torch.no_grad():
            for batch in val_data:
                inputs, targets = batch
                
                # Forward pass
                outputs = client.model(inputs)
                logits = outputs['logits']
                
                # Compute metrics
                loss = F.cross_entropy(logits, targets)
                acc = (logits.argmax(dim=-1) == targets).float().mean()
                
                metrics['val_loss'] += loss.item()
                metrics['val_acc'] += acc.item()
                
        # Normalize metrics
        num_batches = len(val_data)
        metrics['val_loss'] /= num_batches
        metrics['val_acc'] /= num_batches
        
        return metrics
    
    def aggregate_updates(self) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using secure aggregation."""
        client_updates = []
        
        for client_id, client in self.clients.items():
            # Get model parameters
            client_params = {
                name: param.data.clone()
                for name, param in client.model.named_parameters()
            }
            
            # Compute update
            update = {
                name: param - self.server_model.state_dict()[name]
                for name, param in client_params.items()
            }
            
            client_updates.append(update)
            
        # Apply privacy mechanisms
        aggregated_updates = {}
        for param_name in client_updates[0].keys():
            param_updates = [
                update[param_name] for update in client_updates
            ]
            aggregated_updates[param_name] = self.privacy_manager.apply_privacy(
                param_updates
            )
            
        return aggregated_updates
    
    def update_server_model(self, updates: Dict[str, torch.Tensor]):
        """Update server model with aggregated updates."""
        server_state = self.server_model.state_dict()
        
        for name, param in server_state.items():
            if name in updates:
                param.data += updates[name]
                
        self.server_model.load_state_dict(server_state)
        
    def train_round(
        self,
        train_loaders: Dict[int, torch.utils.data.DataLoader],
        val_loaders: Optional[Dict[int, torch.utils.data.DataLoader]] = None
    ) -> Dict[str, float]:
        """Train one round of federated learning."""
        round_metrics = {
            'train_loss': 0.0,
            'train_acc': 0.0,
            'val_loss': 0.0,
            'val_acc': 0.0,
            'drift_detected': False
        }
        
        # Train clients
        for client_id, train_loader in train_loaders.items():
            val_loader = val_loaders.get(client_id) if val_loaders else None
            client_metrics = self.train_client(client_id, train_loader, val_loader)
            
            # Update round metrics
            for key, value in client_metrics.items():
                if key in round_metrics:
                    round_metrics[key] += value
                    
        # Normalize metrics
        num_clients = len(train_loaders)
        for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            if key in round_metrics:
                round_metrics[key] /= num_clients
                
        # Aggregate updates
        updates = self.aggregate_updates()
        self.update_server_model(updates)
        
        # Update metrics history
        for key, value in round_metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
                
        # Log round metrics
        self.logger.info(f'Round metrics: {round_metrics}')
        
        return round_metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'server_model': self.server_model.state_dict(),
            'expert_manager': self.expert_manager.policy_network.state_dict(),
            'client_states': {
                client_id: {
                    'model': client.model.state_dict(),
                    'optimizer': client.optimizer.state_dict(),
                    'drift_detector': client.drift_detector.reference_dist
                }
                for client_id, client in self.clients.items()
            },
            'metrics_history': self.metrics_history
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        
        self.server_model.load_state_dict(checkpoint['server_model'])
        self.expert_manager.policy_network.load_state_dict(
            checkpoint['expert_manager']
        )
        
        for client_id, client_state in checkpoint['client_states'].items():
            if client_id in self.clients:
                self.clients[client_id].model.load_state_dict(
                    client_state['model']
                )
                self.clients[client_id].optimizer.load_state_dict(
                    client_state['optimizer']
                )
                self.clients[client_id].drift_detector.reference_dist = (
                    client_state['drift_detector']
                )
                
        self.metrics_history = checkpoint['metrics_history'] 