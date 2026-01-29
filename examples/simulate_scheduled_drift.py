import os
import yaml
import torch
import logging
import argparse
import json
from datetime import datetime
from torch.utils.data import DataLoader
from dair_fedmoe.simulation.federated_env import FederatedEnvironment
from dair_fedmoe.utils.dataset import DatasetProcessor
from dair_fedmoe.config import ModelConfig

def setup_logging(output_dir: str):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'scheduled_drift.log')),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='Simulate federated learning with scheduled drifts')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='iscx-vpn',
                      help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='datasets/processed/iscx-vpn',
                      help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs/scheduled_drift',
                      help='Output directory')
    parser.add_argument('--num_rounds', type=int, default=250,
                      help='Number of training rounds')
    parser.add_argument('--num_clients', type=int, default=10,
                      help='Number of clients')
    parser.add_argument('--drift_strength', type=float, default=0.5,
                      help='Strength of drift injection')
    return parser.parse_args()

def save_metrics(metrics: dict, output_dir: str, round_num: int):
    """Save metrics to JSON file"""
    metrics_file = os.path.join(output_dir, f'metrics_round_{round_num}.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    # Parse arguments
    args = parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup
    setup_logging(output_dir)
    config = load_config(args.config)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Define drift schedule
    drift_schedule = {
        50: "feature",    # Feature drift at round 50
        100: "concept",   # Concept drift at round 100
        150: "label",     # Label drift at round 150
        200: "combined"   # Combined drift at round 200
    }
    
    # Initialize components
    dataset_processor = DatasetProcessor(config)
    
    # Create federated environment
    env = FederatedEnvironment(
        config=config,
        dataset_processor=dataset_processor,
        num_clients=args.num_clients,
        drift_strength=args.drift_strength,
        privacy_epsilon=1.0,
        privacy_delta=1e-5,
        drift_schedule=drift_schedule
    )
    
    # Setup environment
    env.setup(args.data_dir)
    
    # Training loop
    for round_num in range(args.num_rounds):
        # Train round
        metrics = env.train_round(round_num)
        
        # Log metrics
        logging.info(f"\nRound {round_num + 1}/{args.num_rounds}")
        logging.info(env.get_metrics_summary())
        
        # Check if drift was injected
        if round_num in drift_schedule:
            drift_type = drift_schedule[round_num]
            logging.info(f"Drift injected: {drift_type}")
        
        # Save metrics
        save_metrics(metrics, output_dir, round_num)
        
        # Evaluate periodically
        if (round_num + 1) % 10 == 0:
            test_loader = DataLoader(
                dataset_processor.get_test_dataset(),
                batch_size=config["training"]["batch_size"],
                shuffle=False
            )
            eval_metrics = env.evaluate(test_loader)
            logging.info("\nEvaluation Metrics:")
            logging.info(env.get_metrics_summary())
            
            # Save evaluation metrics
            save_metrics(eval_metrics, output_dir, f'eval_{round_num}')
    
    # Save final model
    torch.save(env.server_model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    
    # Save drift history
    drift_history = env.get_drift_history()
    with open(os.path.join(output_dir, "drift_history.txt"), "w") as f:
        for round_num, drift_type in drift_history:
            f.write(f"Round {round_num}: {drift_type}\n")
    
    # Save final metrics summary
    with open(os.path.join(output_dir, "final_metrics.txt"), "w") as f:
        f.write(env.get_metrics_summary())
    
    logging.info("Training completed. All metrics and model saved.")

if __name__ == "__main__":
    main() 