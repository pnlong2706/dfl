"""
Krum Model Poisoning attack (Fang attack on Krum).

Reference: Fang, M., Cao, X., Jia, J., & Gong, N. (2020).
"Local Model Poisoning Attacks to Byzantine-Robust Federated Learning." USENIX Security.

The attack crafts a malicious model w_m = mean - lambda * sign(mean) where
lambda is found via binary search such that Krum selects the malicious model.
All Byzantine workers submit the same crafted model.
"""

import logging
import math
from collections import OrderedDict
from typing import Dict, List

import torch

from nebula.addons.attacks.model.knowledgeattack import KnowledgeModelAttack


def _flatten(state_dict: OrderedDict) -> torch.Tensor:
    """Flatten a state dict into a single 1D tensor."""
    return torch.cat([v.float().flatten() for v in state_dict.values()])


def _unflatten(flat: torch.Tensor, reference: OrderedDict) -> OrderedDict:
    """Restore a flat tensor back to OrderedDict using reference shapes."""
    result = OrderedDict()
    offset = 0
    for key, val in reference.items():
        numel = val.numel()
        result[key] = flat[offset:offset + numel].reshape(val.shape).to(val.dtype)
        offset += numel
    return result


def _krum_score(vectors: List[torch.Tensor], n_exclude: int) -> List[float]:
    """
    Compute Krum scores: for each vector, sum of distances to nearest (n - n_exclude - 2) neighbors.
    """
    n = len(vectors)
    k = max(1, n - n_exclude - 2)
    scores = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                dists.append(torch.norm(vectors[i] - vectors[j]).item())
        dists.sort()
        scores.append(sum(dists[:k]))
    return scores


class KrumModelPoisoningAttack(KnowledgeModelAttack):
    """
    Krum attack: crafts mal = mean - lambda * sign(mean), binary search on lambda
    so that Krum selects the malicious model.
    """

    def __init__(self, engine, attack_params: Dict):
        try:
            round_start = int(attack_params["round_start_attack"])
            round_stop = int(attack_params["round_stop_attack"])
            attack_interval = int(attack_params["attack_interval"])
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid attack params for Krum Attack: {e}")

        super().__init__(engine, round_start, round_stop, attack_interval)

        n_nodes = engine.config.participant["scenario_args"]["n_nodes"]
        poisoned_percent = float(attack_params.get("poisoned_node_percent", 20))
        self.n_attackers = max(1, int(n_nodes * poisoned_percent / 100))
        self.max_iterations = 20
        self.lambda_threshold = 1e-5

        logging.info(
            f"[Krum Attack] Initialized: n_attackers={self.n_attackers}, "
            f"max_iterations={self.max_iterations}"
        )

    def craft_attack(self, neighbor_models: List[OrderedDict]) -> OrderedDict:
        """
        Craft Krum malicious model via binary search on lambda.
        """
        reference = neighbor_models[0]

        # Flatten all neighbor models
        flat_models = [_flatten(m) for m in neighbor_models]
        mean_model = torch.stack(flat_models).mean(dim=0)
        deviation = torch.sign(mean_model)

        d = mean_model.numel()
        n_honest = len(flat_models)

        # Initialize lambda from Krum distance scores
        scores = _krum_score(flat_models, self.n_attackers)
        min_score = min(scores)
        max_dist = max(torch.norm(f - mean_model).item() for f in flat_models)
        lam = min_score / (max(1, n_honest - self.n_attackers - 1) * math.sqrt(d)) + max_dist / math.sqrt(d)

        # Binary search on lambda
        for iteration in range(self.max_iterations):
            if lam < self.lambda_threshold:
                break

            # Craft malicious model
            mal_flat = mean_model - lam * deviation

            # Check if Krum would select this malicious model
            all_models = [mal_flat] + flat_models
            scores = _krum_score(all_models, self.n_attackers)
            selected = scores.index(min(scores))

            if selected == 0:
                # Krum selects the malicious model — attack succeeds
                logging.info(
                    f"[Krum Attack] Success at iteration {iteration}, lambda={lam:.6f}"
                )
                return _unflatten(mal_flat, reference)

            # Reduce lambda and try again
            lam *= 0.5

        # Fallback: use best effort
        mal_flat = mean_model - lam * deviation
        logging.info(f"[Krum Attack] Fallback with lambda={lam:.6f}")
        return _unflatten(mal_flat, reference)