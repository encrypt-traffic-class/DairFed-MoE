# DAIR-FedMoE: Hierarchical MoE for Federated Encrypted Traffic Classification

![Alt text](assets/model.png?raw=true "Model")

This repository contains the implementation of DAIR-FedMoE, a novel framework for federated encrypted traffic classification that addresses distributed feature drift, concept drift, and label drift.

![Alt text](assets/entaglement.png?raw=true "Entanglement")

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Federated Learning Simulation](#federated-learning-simulation)

## Prerequisites

### 1. Python Environment
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### 2. External Tools
- [Wireshark](https://www.wireshark.org/download.html) (for EditCap)
- [SplitCap](https://www.netresec.com/?page=SplitCap) (for session splitting)

### 3. Dataset
- [ISCX-VPN2016](https://www.unb.ca/cic/datasets/vpn.html)
- [ISCX-Tor2016](https://www.unb.ca/cic/datasets/tor.html)
- [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [UNSW-NB15](https://www.unb.ca/cic/datasets/nb15.html)

![Alt text](assets/performance-1.png?raw=true "Performance")
![Alt text](assets/performance-2.png?raw=true "Performance")

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dair-fedmoe.git
cd dair-fedmoe
```

2. Create and activate a conda environment:
```bash
conda create -n dair-fedmoe python=3.8
conda activate dair-fedmoe
```

3. Install PyTorch:
```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

4. Install other dependencies:
```bash
pip install -r requirements.txt
```

5. Install Wireshark:
   - Windows: Download and run the installer from [Wireshark website](https://www.wireshark.org/download.html)
   - Linux: `sudo apt-get install wireshark`
   - macOS: `brew install wireshark`

6. Install SplitCap:
   - Download from [Netresec](https://www.netresec.com/?page=SplitCap)
   - Extract to a known location
   - Add to system PATH or update the `splitcap_path` in configuration

## Dataset Setup

1. Download the datasets:
```bash
# Create datasets directory
mkdir -p datasets/raw
cd datasets/raw

# Download ISCX-VPN2016
wget https://www.unb.ca/cic/datasets/vpn.html -O iscx-vpn2016.zip
unzip iscx-vpn2016.zip

# Download ISCX-Tor2016
wget https://www.unb.ca/cic/datasets/tor.html -O iscx-tor2016.zip
unzip iscx-tor2016.zip

# Download CIC-IDS2017
wget https://www.unb.ca/cic/datasets/ids-2017.html -O cic-ids2017.zip
unzip cic-ids2017.zip

# Download UNSW-NB15
wget https://www.unb.ca/cic/datasets/nb15.html -O unsw-nb15.zip
unzip unsw-nb15.zip
```

2. Organize the datasets:
```bash
# Create processed directory
mkdir -p datasets/processed

# Process each dataset
python scripts/process_dataset.py --dataset iscx-vpn --input_dir datasets/raw/iscx-vpn2016 --output_dir datasets/processed/iscx-vpn
python scripts/process_dataset.py --dataset iscx-tor --input_dir datasets/raw/iscx-tor2016 --output_dir datasets/processed/iscx-tor
python scripts/process_dataset.py --dataset cic-ids2017 --input_dir datasets/raw/cic-ids2017 --output_dir datasets/processed/cic-ids2017
python scripts/process_dataset.py --dataset unsw-nb15 --input_dir datasets/raw/unsw-nb15 --output_dir datasets/processed/unsw-nb15
```

## Configuration

1. Update the configuration file:
```bash
# Copy default configuration
cp configs/default.yaml configs/custom.yaml

# Edit configuration
nano configs/custom.yaml
```

Key configuration parameters:
```yaml
model:
  # Model architecture
  hidden_dim: 512
  num_heads: 8
  num_layers: 6
  dropout: 0.1
  num_classes: 2  # VPN vs non-VPN

  # Input features
  max_packets_per_flow: 10
  max_payload_size: 1500  # Ethernet MTU
  num_header_features: 20
  num_statistical_features: 10

  # MoE configuration
  num_experts: 8
  expert_capacity: 64
  router_jitter: 0.01
  router_loss_weight: 0.01

training:
  # Federated learning
  num_clients: 10
  num_rounds: 250
  learning_rate: 1e-3
  batch_size: 32

  # PCAP processing
  flow_timeout: 600  # 10 minutes
  include_headers: true
  include_stats: true
```

## Training

1. Train the model:
```bash
# Single GPU training
python examples/train.py \
    --config configs/custom.yaml \
    --dataset iscx-vpn \
    --data_dir datasets/processed/iscx-vpn \
    --output_dir outputs/iscx-vpn \
    --num_rounds 250 \
    --batch_size 32

# Multi-GPU training
python examples/train.py \
    --config configs/custom.yaml \
    --dataset iscx-vpn \
    --data_dir datasets/processed/iscx-vpn \
    --output_dir outputs/iscx-vpn \
    --num_rounds 250 \
    --batch_size 32 \
    --num_gpus 4
```

2. Monitor training:
```bash
# View training logs
tail -f outputs/iscx-vpn/training.log

# View TensorBoard
tensorboard --logdir outputs/iscx-vpn/tensorboard
```

![Alt text](assets/expert1.png?raw=true "Expert1")
![Alt text](assets/expert2.png?raw=true "Expert2")

## Evaluation

1. Evaluate the model:
```bash
python examples/evaluate.py \
    --config configs/custom.yaml \
    --dataset iscx-vpn \
    --data_dir datasets/processed/iscx-vpn \
    --model_path outputs/iscx-vpn/model.pt \
    --output_dir outputs/iscx-vpn/evaluation
```

2. Generate evaluation reports:
```bash
python examples/generate_report.py \
    --eval_dir outputs/iscx-vpn/evaluation \
    --output_dir outputs/iscx-vpn/reports
```

## Project Structure

```
dair-fedmoe/
├── assets/                 # Images and diagrams
├── configs/               # Configuration files
├── datasets/             # Dataset storage
│   ├── raw/             # Raw datasets
│   └── processed/       # Processed datasets
├── dair_fedmoe/         # Main package
│   ├── models/          # Model implementations
│   ├── training/        # Training code
│   ├── drift/           # Drift detection
│   ├── privacy/         # Privacy mechanisms
│   ├── rl/              # Reinforcement learning
│   └── utils/           # Utilities
├── docs/                # Documentation
├── examples/            # Example scripts
├── scripts/             # Utility scripts
├── tests/               # Unit tests
├── outputs/             # Training outputs
├── requirements.txt     # Python dependencies
└── README.md           # This file
```
![Alt text](assets/policy.png?raw=true "Policy")
![Alt text](assets/redgeline.png?raw=true "Redgeline")
![Alt text](assets/tradeoff.png?raw=true "Tradeoff")



## Federated Learning Simulation

The project includes a simulation environment for federated learning with drift injection capabilities. This allows you to test the model's performance under various drift scenarios.

### Simulation Setup

1. Configure the simulation parameters in `configs/default.yaml`:
```yaml
simulation:
  num_clients: 10
  drift_type: "feature"  # "feature", "concept", or "label"
  drift_strength: 0.5
  privacy_epsilon: 1.0
  privacy_delta: 1e-5
  non_iid_alpha: 0.5  # Controls non-IIDness of data distribution
```

2. Run the simulation:
```bash
python examples/simulate_federated.py \
    --config configs/default.yaml \
    --dataset iscx-vpn \
    --data_dir datasets/processed/iscx-vpn \
    --output_dir outputs/simulation \
    --num_rounds 250
```

### Drift Types

The simulation supports three types of drift:

1. **Feature Drift**:
   - Injects noise into input features
   - Simulates changes in network traffic patterns
   - Controlled by `drift_strength` parameter

2. **Concept Drift**:
   - Modifies feature-label relationships
   - Simulates changes in traffic classification rules
   - Affects model's decision boundaries

3. **Label Drift**:
   - Changes label distribution
   - Simulates shifts in traffic types
   - Affects class balance

### Non-IID Data Distribution

The simulation creates non-IID data distributions among clients using the Dirichlet distribution:

```python
# Example of non-IID data split
alpha = 0.5  # Lower alpha = more non-IID
proportions = np.random.dirichlet([alpha] * num_clients)
```

### Privacy Mechanisms

The simulation includes differential privacy:

1. **Client-level Privacy**:
   - Applies noise to model updates
   - Controlled by `privacy_epsilon` and `privacy_delta`
   - Protects individual client data

2. **Secure Aggregation**:
   - Aggregates client updates securely
   - Prevents information leakage
   - Maintains model performance

### Monitoring and Evaluation

The simulation provides comprehensive monitoring:

1. **Training Metrics**:
   - Per-round loss and accuracy
   - Client-specific metrics
   - Drift detection alerts

2. **Evaluation**:
   - Periodic model evaluation
   - Test set performance
   - Drift impact analysis

3. **Logging**:
   - Detailed training logs
   - Drift detection events
   - Performance metrics

### Example Output

```
Round 1/250
Average Loss: 0.6931
Average Accuracy: 0.5234

Round 10/250
Average Loss: 0.5123
Average Accuracy: 0.7123
Drift detected in client 3

Evaluation Metrics:
Test Loss: 0.4987
Test Accuracy: 0.7234
```


To run the scheduled drift simulation:

```bash
python examples/simulate_scheduled_drift.py \
    --config configs/default.yaml \
    --dataset iscx-vpn \
    --data_dir datasets/processed/iscx-vpn \
    --output_dir outputs/scheduled_drift \
    --num_rounds 250 \
    --num_clients 10 \
    --drift_strength 0.5
```

The simulation will:
1. Train the model for 250 rounds
2. Inject drifts
3. Log training metrics and drift events
4. Save the final model and drift history

Example output:
```
Round 50/250
Average Loss: 0.5123
Average Accuracy: 0.7123
Drift injected: feature

Round 100/250
Average Loss: 0.6234
Average Accuracy: 0.6543
Drift injected: concept

Round 150/250
Average Loss: 0.5890
Average Accuracy: 0.6789
Drift injected: label

Round 200/250
Average Loss: 0.7123
Average Accuracy: 0.5890
Drift injected: combined
```

The drift history is saved in `outputs/scheduled_drift/drift_history.txt`:
```
Round 50: feature
Round 100: concept
Round 150: label
Round 200: combined
```

## Implementation

The framework is implemented using:
- PyTorch 1.12 for model construction
- Flower 0.19 for federated orchestration
- Ray RLlib 1.8 for reinforcement learning

### Running the Simulation

#### Using Custom Implementation

To run the simulation using our custom implementation:

```bash
python examples/simulate_scheduled_drift.py --config configs/default.yaml --num-clients 10 --num-rounds 250
```

#### Using Flower Implementation

To run the simulation using Flower's federated learning framework:

```bash
python examples/simulate_flower.py --config configs/default.yaml --num-clients 10 --num-rounds 250
```

The Flower implementation provides the following advantages:
- Built-in support for federated learning protocols
- Efficient communication between clients and server
- Scalable to large numbers of clients
- Support for various aggregation strategies

Both implementations support the same features:
- Dynamic expert routing
- Drift detection and adaptation
- Privacy-preserving mechanisms
- Comprehensive metrics tracking 




## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{dair-fedmoe,
  title={DAIR-FedMoE: Hierarchical MoE for Federated Encrypted Traffic Classification under Distributed Feature, Concept, and Label Drift},
  author={Shamaila Fardous, Kashif Sharif, Fan Li, Ali Asghar Manjotho},
  journal={Transactions on Dependable and Secure Computing},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ISCX-VPN2016](https://www.unb.ca/cic/datasets/vpn.html) dataset
- [ISCX-Tor2016](https://www.unb.ca/cic/datasets/tor.html) dataset
- [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset
- [UNSW-NB15](https://www.unb.ca/cic/datasets/nb15.html) dataset
