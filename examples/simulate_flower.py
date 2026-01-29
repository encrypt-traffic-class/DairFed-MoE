import os
import sys
import logging
import argparse
import torch
import flwr as fl
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dair_fedmoe.config import ModelConfig
from dair_fedmoe.models.dair_fedmoe import DAIRFedMoE
from dair_fedmoe.utils.dataset import DatasetProcessor
from dair_fedmoe.utils.metrics import MetricsTracker
from dair_fedmoe.simulation.flower_client import DAIRFedMoEClient
from dair_fedmoe.simulation.flower_server import DAIRFedMoEServer

def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    log_file = output_dir / "simulation.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Run federated learning simulation using Flower")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                      help="Path to configuration file")
    parser.add_argument("--num-clients", type=int, default=10,
                      help="Number of federated clients")
    parser.add_argument("--num-rounds", type=int, default=250,
                      help="Number of federated rounds")
    parser.add_argument("--output-dir", type=str, default="results",
                      help="Output directory for results")
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"flower_simulation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = ModelConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Initialize model
    model = DAIRFedMoE(config)
    logger.info("Initialized DAIR-FedMoE model")
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Initialize dataset processor
    dataset_processor = DatasetProcessor(config)
    
    # Load and process dataset
    train_data, test_data = dataset_processor.load_iscx_vpn()
    logger.info("Loaded ISCX-VPN dataset")
    
    # Split data for federated learning
    client_data = dataset_processor.split_data(train_data, args.num_clients)
    logger.info(f"Split data among {args.num_clients} clients")
    
    # Initialize Flower clients
    clients = []
    for i in range(args.num_clients):
        client = DAIRFedMoEClient(
            model=model,
            config=config,
            train_data=client_data[i],
            test_data=test_data,
            client_id=i
        )
        clients.append(client)
    logger.info("Initialized Flower clients")
    
    # Initialize Flower server
    server = DAIRFedMoEServer(
        model=model,
        config=config,
        metrics_tracker=metrics_tracker,
        min_fit_clients=args.num_clients,
        min_eval_clients=args.num_clients,
        min_available_clients=args.num_clients,
        fraction_fit=1.0,
        fraction_eval=1.0,
        initial_parameters=fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for val in model.state_dict().values()]
        )
    )
    logger.info("Initialized Flower server")
    
    # Start federated learning
    logger.info("Starting federated learning simulation")
    fl.simulation.start_simulation(
        client_fn=lambda cid: clients[cid],
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=server
    )
    
    # Save final results
    metrics_summary = metrics_tracker.get_metrics_summary()
    with open(output_dir / "final_metrics.txt", "w") as f:
        f.write(metrics_summary)
    
    # Save model
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    
    logger.info("Simulation completed")
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 