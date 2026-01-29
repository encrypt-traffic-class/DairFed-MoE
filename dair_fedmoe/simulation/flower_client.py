import torch
import flwr as fl
from typing import Dict, List, Tuple
from ..models.dair_fedmoe import DAIRFedMoE
from ..utils.metrics import MetricsTracker
from ..drift.drift_detector import DriftDetector
from ..privacy.privacy_mechanism import PrivacyMechanism

class DAIRFedMoEClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: DAIRFedMoE,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: dict,
        drift_detector: DriftDetector,
        privacy_mechanism: PrivacyMechanism,
        metrics_tracker: MetricsTracker
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.drift_detector = drift_detector
        self.privacy_mechanism = privacy_mechanism
        self.metrics_tracker = metrics_tracker
        
        # Training setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def get_parameters(self, config):
        """Get model parameters as a list of NumPy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model using the provided parameters"""
        self.set_parameters(parameters)
        
        # Training setup
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"]
        )
        
        # Training loop
        self.model.train()
        for batch in self.train_loader:
            features, labels = batch
            features, labels = features.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, expert_indices = self.model(features, return_expert_indices=True)
            loss = self.model.compute_loss(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            self.metrics_tracker.update_classification_metrics(outputs, labels)
            self.metrics_tracker.update_expert_metrics(expert_indices)
        
        # Detect drift
        drift_detected = self.drift_detector.detect_drift()
        self.metrics_tracker.update_drift_metrics(config["round"], drift_detected)
        
        # Apply privacy mechanism
        parameters = self.get_parameters({})
        private_parameters = self.privacy_mechanism.apply(parameters)
        
        # Update privacy metrics
        epsilon_consumed = self.privacy_mechanism.get_epsilon_consumed()
        self.metrics_tracker.update_privacy_metrics(epsilon_consumed)
        
        return private_parameters, len(self.train_loader), self.metrics_tracker.get_metrics()
    
    def evaluate(self, parameters, config):
        """Evaluate the model using the provided parameters"""
        self.set_parameters(parameters)
        
        # Evaluation loop
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                features, labels = batch
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = self.model(features)
                self.metrics_tracker.update_classification_metrics(outputs, labels)
        
        metrics = self.metrics_tracker.get_metrics()
        return float(metrics["loss"]), len(self.val_loader), metrics 