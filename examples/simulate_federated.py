import os
import yaml
import torch
import logging
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
            logging.FileHandler(os.path.join(output_dir, 'simulation.log')),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Configuration
    config_path = "configs/default.yaml"
    output_dir = "outputs/simulation"
    dataset_path = "datasets/processed/iscx-vpn"
    
    # Setup
    setup_logging(output_dir)
    config = load_config(config_path)
    
    # Initialize components
    dataset_processor = DatasetProcessor(config)
    
    # Create federated environment
    env = FederatedEnvironment(
        config=config,
        dataset_processor=dataset_processor,
        num_clients=10,
        drift_type="feature",  # or "concept" or "label"
        drift_strength=0.5,
        privacy_epsilon=1.0,
        privacy_delta=1e-5
    )
    
    # Setup environment
    env.setup(dataset_path)
    
    # Training loop
    num_rounds = config["training"]["num_rounds"]
    for round_num in range(num_rounds):
        # Train round
        metrics = env.train_round(round_num)
        
        # Log metrics
        logging.info(f"Round {round_num + 1}/{num_rounds}")
        logging.info(f"Average Loss: {metrics['loss']:.4f}")
        logging.info(f"Average Accuracy: {metrics['accuracy']:.4f}")
        
        # Evaluate periodically
        if (round_num + 1) % 10 == 0:
            test_loader = DataLoader(
                dataset_processor.get_test_dataset(),
                batch_size=config["training"]["batch_size"],
                shuffle=False
            )
            eval_metrics = env.evaluate(test_loader)
            logging.info("Evaluation Metrics:")
            logging.info(f"Test Loss: {eval_metrics['loss']:.4f}")
            logging.info(f"Test Accuracy: {eval_metrics['accuracy']:.4f}")
    
    # Save final model
    torch.save(env.server_model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    logging.info("Training completed. Model saved.")

if __name__ == "__main__":
    main() 