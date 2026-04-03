"""
BALANCE (Byzantine-robust Averaging through Local Similarity in Decentralization) aggregator.

Reference: Fang, M., Liu, J., Gong, N. Z., & Gire, E. (2024).
"Byzantine-Robust Decentralized Federated Learning."
ACM SIGSAC Conference on Computer and Communications Security (CCS '24).

Filtering condition: accept neighbor j if
    ||w_i - w_j|| <= gamma * exp(-kappa * t/T) * ||w_i||

The threshold tightens exponentially as training progresses (models converge).
Aggregation: w_new = alpha * w_local + (1-alpha) * mean(accepted_neighbors)
"""

import gc
import logging
import math

import torch

from nebula.core.aggregation.aggregator import Aggregator


class Balance(Aggregator):
    """
    BALANCE aggregation: filter neighbors by adaptive distance threshold,
    then weighted average of local model and accepted neighbors.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.alpha = config.participant.get("aggregator_args", {}).get("balance_alpha", 0.5)
        self.gamma = config.participant.get("aggregator_args", {}).get("balance_gamma", 0.3)
        self.kappa = config.participant.get("aggregator_args", {}).get("balance_kappa", 1.0)
        logging.info(f"[BALANCE] Initialized: alpha={self.alpha}, gamma={self.gamma}, kappa={self.kappa}")

    def run_aggregation(self, models):
        super().run_aggregation(models)

        models_list = list(models.items())
        if len(models_list) == 0:
            return None

        # Find local model
        local_model = None
        for addr, (model, weight) in models_list:
            if addr == self._addr:
                local_model = model
                break
        if local_model is None:
            local_model = models_list[0][1][0]

        keys = list(local_model.keys())

        # Get current round and total rounds from engine
        current_round = self.engine.round if self.engine.round is not None else 0
        total_rounds = self.engine.total_rounds if self.engine.total_rounds is not None else 100
        if total_rounds <= 0:
            total_rounds = 100

        # Compute adaptive threshold: gamma * exp(-kappa * t/T) * ||w_local||
        t_ratio = current_round / total_rounds
        decay = math.exp(-self.kappa * t_ratio)

        local_norm = sum(
            torch.norm(local_model[key].float()).item() ** 2 for key in keys
        ) ** 0.5

        threshold = self.gamma * decay * local_norm

        logging.info(
            f"[BALANCE] Round {current_round}/{total_rounds}, "
            f"decay={decay:.4f}, local_norm={local_norm:.2f}, threshold={threshold:.4f}"
        )

        # Filter neighbors
        accepted = []
        for addr, (model, weight) in models_list:
            if addr == self._addr:
                continue  # Skip local model

            dist = sum(
                torch.norm(model[key].float() - local_model[key].float()).item() ** 2 for key in keys
            ) ** 0.5

            if dist <= threshold:
                accepted.append((addr, model, weight))
                logging.info(f"[BALANCE] Accepted {addr} (dist={dist:.4f} <= {threshold:.4f})")
            else:
                logging.info(f"[BALANCE] Rejected {addr} (dist={dist:.4f} > {threshold:.4f})")

        # Aggregation: alpha * local + (1-alpha) * mean(accepted)
        accum = {key: torch.zeros_like(local_model[key], dtype=torch.float32) for key in keys}

        with torch.no_grad():
            # Local contribution
            for key in keys:
                accum[key] += self.alpha * local_model[key].float()

            # Neighbor contribution
            if accepted:
                for key in keys:
                    neighbor_sum = torch.zeros_like(local_model[key], dtype=torch.float32)
                    for addr, model, weight in accepted:
                        neighbor_sum += model[key].float()
                    accum[key] += (1 - self.alpha) * neighbor_sum / len(accepted)
            else:
                # No neighbors accepted — fallback to local model only
                logging.warning("[BALANCE] No neighbors accepted, using local model only")
                for key in keys:
                    accum[key] += (1 - self.alpha) * local_model[key].float()

        gc.collect()
        logging.info(f"[BALANCE] Aggregated with {len(accepted)} accepted neighbors out of {len(models_list) - 1}")
        return accum