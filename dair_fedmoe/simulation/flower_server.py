import flwr as fl
from typing import Dict, List, Tuple, Optional
from ..models.dair_fedmoe import DAIRFedMoE
from ..utils.metrics import MetricsTracker
import torch

class DAIRFedMoEServer(fl.server.strategy.FedAvg):
    def __init__(
        self,
        model: DAIRFedMoE,
        config: dict,
        metrics_tracker: MetricsTracker,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.config = config
        self.metrics_tracker = metrics_tracker
        
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Parameters]:
        """Aggregate model weights and update metrics"""
        # Aggregate weights using FedAvg
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_parameters is not None:
            # Update model parameters
            parameters_dict = zip(self.model.state_dict().keys(), aggregated_parameters)
            state_dict = {k: torch.from_numpy(v) for k, v in parameters_dict}
            self.model.load_state_dict(state_dict, strict=True)
            
            # Update communication metrics
            model_size = sum(p.numel() for p in self.model.parameters())
            self.metrics_tracker.update_communication_metrics(
                model_size,
                len(results)
            )
            
            # Update expert metrics from all clients
            expert_usage = []
            for _, fit_res in results:
                if "metrics" in fit_res.metrics:
                    expert_usage.extend(fit_res.metrics["metrics"].get("experts_per_round", []))
            self.metrics_tracker.update_expert_metrics(expert_usage)
        
        return aggregated_parameters
    
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation results and update metrics"""
        # Aggregate evaluation results
        aggregated_loss = super().aggregate_evaluate(rnd, results, failures)
        
        if aggregated_loss is not None:
            # Update metrics from all clients
            for _, eval_res in results:
                if "metrics" in eval_res.metrics:
                    client_metrics = eval_res.metrics["metrics"]
                    self.metrics_tracker.update_classification_metrics(
                        client_metrics.get("outputs", None),
                        client_metrics.get("labels", None)
                    )
        
        return aggregated_loss 