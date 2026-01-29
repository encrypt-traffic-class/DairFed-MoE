import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, recall_score, confusion_matrix
import time

class MetricsTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            # Classification metrics
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'minority_recall': 0.0,
            
            # Drift metrics
            'drift_detection_time': 0.0,
            'drift_recovery_speed': 0.0,
            'drift_impact': 0.0,
            
            # Communication metrics
            'communication_cost': 0.0,
            'bytes_transferred': 0,
            
            # Expert metrics
            'experts_per_round': [],
            'expert_utilization': 0.0,
            
            # Performance metrics
            'runtime_overhead': 0.0,
            'training_time': 0.0,
            
            # Privacy metrics
            'privacy_budget_consumed': 0.0,
            'privacy_epsilon_remaining': 1.0
        }
        
        # Trackers
        self.start_time = None
        self.last_drift_time = None
        self.last_accuracy = None
        self.round_times = []
        
    def start_round(self):
        """Start timing a round"""
        self.start_time = time.time()
        
    def end_round(self, round_num: int):
        """End timing a round and update metrics"""
        if self.start_time is not None:
            round_time = time.time() - self.start_time
            self.round_times.append(round_time)
            self.metrics['runtime_overhead'] = np.mean(self.round_times)
            self.metrics['training_time'] = sum(self.round_times)
    
    def update_classification_metrics(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Update classification metrics"""
        predictions = outputs.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Accuracy
        self.metrics['accuracy'] = (predictions == labels).mean()
        
        # Macro-F1
        self.metrics['macro_f1'] = f1_score(labels, predictions, average='macro')
        
        # Minority class recall
        cm = confusion_matrix(labels, predictions)
        if cm.shape[0] > 1:  # Multi-class case
            minority_class = np.argmin(cm.sum(axis=1))
            self.metrics['minority_recall'] = recall_score(
                labels, predictions, labels=[minority_class], average='micro'
            )
        else:  # Binary case
            self.metrics['minority_recall'] = recall_score(
                labels, predictions, average='binary'
            )
    
    def update_drift_metrics(self, round_num: int, drift_detected: bool):
        """Update drift-related metrics"""
        if drift_detected:
            if self.last_drift_time is not None:
                # Calculate drift recovery speed
                rounds_since_last_drift = round_num - self.last_drift_time
                if self.last_accuracy is not None:
                    accuracy_drop = self.last_accuracy - self.metrics['accuracy']
                    if accuracy_drop > 0:
                        self.metrics['drift_recovery_speed'] = accuracy_drop / rounds_since_last_drift
            
            self.last_drift_time = round_num
            self.last_accuracy = self.metrics['accuracy']
    
    def update_communication_metrics(self, model_size: int, num_clients: int):
        """Update communication-related metrics"""
        # Calculate bytes transferred (model parameters + gradients)
        bytes_per_client = model_size * 4  # 4 bytes per float32
        total_bytes = bytes_per_client * num_clients * 2  # *2 for upload and download
        self.metrics['bytes_transferred'] += total_bytes
        self.metrics['communication_cost'] = self.metrics['bytes_transferred'] / (1024 * 1024)  # Convert to MB
    
    def update_expert_metrics(self, expert_usage: List[int]):
        """Update expert-related metrics"""
        self.metrics['experts_per_round'].append(len(expert_usage))
        self.metrics['expert_utilization'] = np.mean(expert_usage)
    
    def update_privacy_metrics(self, epsilon_consumed: float):
        """Update privacy-related metrics"""
        self.metrics['privacy_budget_consumed'] += epsilon_consumed
        self.metrics['privacy_epsilon_remaining'] -= epsilon_consumed
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics"""
        return self.metrics
    
    def get_metrics_summary(self) -> str:
        """Get a formatted summary of metrics"""
        summary = []
        summary.append("Metrics Summary:")
        summary.append(f"Classification:")
        summary.append(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        summary.append(f"  Macro-F1: {self.metrics['macro_f1']:.4f}")
        summary.append(f"  Minority Recall: {self.metrics['minority_recall']:.4f}")
        
        summary.append(f"\nDrift:")
        summary.append(f"  Recovery Speed: {self.metrics['drift_recovery_speed']:.4f}")
        summary.append(f"  Impact: {self.metrics['drift_impact']:.4f}")
        
        summary.append(f"\nCommunication:")
        summary.append(f"  Cost: {self.metrics['communication_cost']:.2f} MB")
        summary.append(f"  Bytes Transferred: {self.metrics['bytes_transferred']}")
        
        summary.append(f"\nExperts:")
        summary.append(f"  Average per Round: {np.mean(self.metrics['experts_per_round']):.2f}")
        summary.append(f"  Utilization: {self.metrics['expert_utilization']:.4f}")
        
        summary.append(f"\nPerformance:")
        summary.append(f"  Runtime Overhead: {self.metrics['runtime_overhead']:.2f}s")
        summary.append(f"  Total Training Time: {self.metrics['training_time']:.2f}s")
        
        summary.append(f"\nPrivacy:")
        summary.append(f"  Budget Consumed: {self.metrics['privacy_budget_consumed']:.4f}")
        summary.append(f"  Epsilon Remaining: {self.metrics['privacy_epsilon_remaining']:.4f}")
        
        return "\n".join(summary) 