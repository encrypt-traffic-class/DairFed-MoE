"""
Dataset processing utilities for DAIR-FedMoE.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

from dair_fedmoe.config import ModelConfig
from .pcap_processor import PCAPProcessor

class EncryptedTrafficDataset(Dataset):
    """Dataset for encrypted traffic classification."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: Optional[nn.Module] = None
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            features = self.transform(features)
            
        return features, label

class DatasetProcessor:
    """Processor for encrypted traffic datasets."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.pcap_processor = PCAPProcessor(
            max_packets_per_flow=config.max_packets_per_flow,
            max_payload_size=config.max_payload_size,
            flow_timeout=config.flow_timeout,
            include_headers=config.include_headers,
            include_stats=config.include_stats
        )
        
    def _format_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Format features for model input."""
        # Extract and format header features
        header_features = []
        for i in range(self.config.max_packets_per_flow):
            packet_cols = [col for col in features_df.columns if f'p{i+1}_' in col and 'payload' not in col]
            if packet_cols:
                packet_features = features_df[packet_cols].values
                header_features.append(packet_features)
            else:
                # Pad with zeros if packet not present
                header_features.append(np.zeros((len(features_df), self.config.num_header_features)))
                
        # Extract and format payload features
        payload_features = []
        for i in range(self.config.max_packets_per_flow):
            payload_col = f'p{i+1}_payload'
            if payload_col in features_df.columns:
                payload = features_df[payload_col].values
                # Convert list of arrays to 2D array
                payload = np.stack([p if isinstance(p, np.ndarray) else np.zeros(self.config.max_payload_size) 
                                  for p in payload])
                payload_features.append(payload)
            else:
                payload_features.append(np.zeros((len(features_df), self.config.max_payload_size)))
                
        # Extract and format statistical features
        stats_features = []
        for i in range(self.config.max_packets_per_flow):
            stats_cols = [col for col in features_df.columns if f'p{i+1}_' in col and 
                         any(s in col for s in ['entropy', 'length', 'iat'])]
            if stats_cols:
                packet_stats = features_df[stats_cols].values
                stats_features.append(packet_stats)
            else:
                stats_features.append(np.zeros((len(features_df), self.config.num_statistical_features)))
                
        # Extract flow-level features
        flow_cols = ['num_packets', 'duration', 'total_bytes', 'avg_packet_size', 'std_packet_size']
        flow_features = features_df[flow_cols].values
        
        # Combine all features
        header_features = np.concatenate(header_features, axis=1)
        payload_features = np.concatenate(payload_features, axis=1)
        stats_features = np.concatenate(stats_features, axis=1)
        
        # Concatenate all feature types
        features = np.concatenate([
            header_features,
            payload_features,
            stats_features,
            flow_features
        ], axis=1)
        
        return features
        
    def load_iscx_vpn(
        self,
        data_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and process ISCX-VPN dataset."""
        # Check if data_path is a directory of PCAP files
        if os.path.isdir(data_path):
            # Process PCAP files
            features_df, labels = self.pcap_processor.process_directory(data_path)
            
            # Convert categorical features to numerical
            features_df = pd.get_dummies(features_df, columns=['flow_key'])
            
            # Format features for model input
            features = self._format_features(features_df)
            labels = np.array(labels)
        else:
            # Load from CSV
            data = pd.read_csv(data_path)
            features = data.drop('label', axis=1).values
            labels = data['label'].values
        
        # Scale features
        features = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train
        )
        
        # Create datasets
        train_dataset = EncryptedTrafficDataset(X_train, y_train)
        val_dataset = EncryptedTrafficDataset(X_val, y_val)
        test_dataset = EncryptedTrafficDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader, test_loader
    
    def load_iscx_tor(
        self,
        data_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and process ISCX-Tor dataset."""
        # Similar to ISCX-VPN processing
        return self.load_iscx_vpn(
            data_path,
            test_size,
            val_size,
            random_state
        )
    
    def load_cic_ids2017(
        self,
        data_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and process CIC-IDS2017 dataset."""
        # Similar to ISCX-VPN processing
        return self.load_iscx_vpn(
            data_path,
            test_size,
            val_size,
            random_state
        )
    
    def load_unsw_nb15(
        self,
        data_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and process UNSW-NB15 dataset."""
        # Similar to ISCX-VPN processing
        return self.load_iscx_vpn(
            data_path,
            test_size,
            val_size,
            random_state
        )
    
    def create_federated_loaders(
        self,
        train_loader: DataLoader,
        num_clients: int,
        alpha: float = 0.5,
        random_state: int = 42
    ) -> Dict[int, DataLoader]:
        """Create federated dataloaders using Dirichlet distribution."""
        # Get all data
        all_features = []
        all_labels = []
        
        for features, labels in train_loader:
            all_features.append(features)
            all_labels.append(labels)
            
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Create client data using Dirichlet distribution
        client_loaders = {}
        
        for client_id in range(num_clients):
            # Sample client data
            client_indices = np.random.dirichlet(
                np.ones(self.config.num_classes) * alpha
            )
            
            # Create client dataset
            client_features = []
            client_labels = []
            
            for class_idx in range(self.config.num_classes):
                class_mask = (all_labels == class_idx)
                class_features = all_features[class_mask]
                class_labels = all_labels[class_mask]
                
                # Sample class data
                num_samples = int(len(class_features) * client_indices[class_idx])
                indices = np.random.choice(
                    len(class_features),
                    num_samples,
                    replace=False
                )
                
                client_features.append(class_features[indices])
                client_labels.append(class_labels[indices])
                
            client_features = torch.cat(client_features, dim=0)
            client_labels = torch.cat(client_labels, dim=0)
            
            # Create client dataset and loader
            client_dataset = EncryptedTrafficDataset(
                client_features.numpy(),
                client_labels.numpy()
            )
            
            client_loader = DataLoader(
                client_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4
            )
            
            client_loaders[client_id] = client_loader
            
        return client_loaders 