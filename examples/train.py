"""
Main training script for DAIR-FedMoE.
"""

import os
import argparse
import torch
import logging
from typing import Dict, Optional, Tuple
import yaml

from dair_fedmoe.config import ModelConfig, TrainingConfig
from dair_fedmoe.models.dair_fedmoe import DAIRFedMoE
from dair_fedmoe.training.federated_trainer import FederatedTrainer
from dair_fedmoe.utils.dataset import DatasetProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='Train DAIR-FedMoE')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['iscx-vpn', 'iscx-tor', 'cic-ids2017', 'unsw-nb15'],
                      help='Dataset to use')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to dataset')
    
    # Training arguments
    parser.add_argument('--num_clients', type=int, default=10,
                      help='Number of federated clients')
    parser.add_argument('--num_rounds', type=int, default=250,
                      help='Number of federated rounds')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate')
    
    # Model arguments
    parser.add_argument('--num_experts', type=int, default=8,
                      help='Number of experts in HMoE layer')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                      help='Number of attention heads')
    
    # Privacy arguments
    parser.add_argument('--dp_epsilon', type=float, default=1.0,
                      help='Differential privacy epsilon')
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                      help='Differential privacy delta')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to config file')
    
    return parser.parse_args()

def load_config(args) -> Tuple[ModelConfig, TrainingConfig]:
    """Load configuration from args and config file."""
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}
        
    # Create model config
    model_config = ModelConfig(
        num_experts=args.num_experts,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        **config_dict.get('model', {})
    )
    
    # Create training config
    training_config = TrainingConfig(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        learning_rate=args.learning_rate,
        **config_dict.get('training', {})
    )
    
    return model_config, training_config

def setup_logging(output_dir: str):
    """Setup logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f'Arguments: {args}')
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load configuration
    model_config, training_config = load_config(args)
    logger.info(f'Model config: {model_config}')
    logger.info(f'Training config: {training_config}')
    
    # Create dataset processor
    dataset_processor = DatasetProcessor(model_config)
    
    # Load dataset
    if args.dataset == 'iscx-vpn':
        train_loader, val_loader, test_loader = dataset_processor.load_iscx_vpn(
            args.data_path
        )
    elif args.dataset == 'iscx-tor':
        train_loader, val_loader, test_loader = dataset_processor.load_iscx_tor(
            args.data_path
        )
    elif args.dataset == 'cic-ids2017':
        train_loader, val_loader, test_loader = dataset_processor.load_cic_ids2017(
            args.data_path
        )
    elif args.dataset == 'unsw-nb15':
        train_loader, val_loader, test_loader = dataset_processor.load_unsw_nb15(
            args.data_path
        )
        
    # Create federated dataloaders
    client_loaders = dataset_processor.create_federated_loaders(
        train_loader,
        args.num_clients
    )
    
    # Create federated trainer
    trainer = FederatedTrainer(model_config, training_config)
    
    # Add clients
    for client_id in range(args.num_clients):
        client_model = DAIRFedMoE(model_config)
        trainer.add_client(client_id, client_model)
        
    # Training loop
    logger.info('Starting training...')
    
    for round_idx in range(args.num_rounds):
        logger.info(f'Round {round_idx + 1}/{args.num_rounds}')
        
        # Train round
        round_metrics = trainer.train_round(client_loaders)
        
        # Log metrics
        logger.info(f'Round metrics: {round_metrics}')
        
        # Save checkpoint
        if (round_idx + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                args.output_dir,
                f'checkpoint_round_{round_idx + 1}.pt'
            )
            trainer.save_checkpoint(checkpoint_path)
            logger.info(f'Saved checkpoint to {checkpoint_path}')
            
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    trainer.save_checkpoint(final_path)
    logger.info(f'Saved final model to {final_path}')
    
    logger.info('Training completed!')

if __name__ == '__main__':
    main() 