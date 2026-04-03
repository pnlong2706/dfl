"""
Trim Model Poisoning attack (Fang attack on Trimmed Mean).

Reference: Fang, M., Cao, X., Jia, J., & Gong, N. (2020).
"Local Model Poisoning Attacks to Byzantine-Robust Federated Learning." USENIX Security.

The attack places malicious values just beyond the honest range per coordinate,
in the direction of sign(mean), so they survive the trimming operation and bias
the aggregated result.

For positive deviation: mal in [max, b*max]
For negative deviation: mal in [b*min, min]
where b=2 is the scaling factor.
"""

import logging
from collections import OrderedDict
from typing import Dict, List

import torch

from nebula.addons.attacks.model.knowledgeattack import KnowledgeModelAttack


class TrimModelPoisoningAttack(KnowledgeModelAttack):
    """
    Trim attack: places malicious values beyond the honest range per coordinate.
    """

    def __init__(self, engine, attack_params: Dict):
        try:
            round_start = int(attack_params["round_start_attack"])
            round_stop = int(attack_params["round_stop_attack"])
            attack_interval = int(attack_params["attack_interval"])
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid attack params for Trim Attack: {e}")

        super().__init__(engine, round_start, round_stop, attack_interval)
        self.b = 2.0  # Scaling factor beyond honest range

        logging.info(f"[Trim Attack] Initialized with b={self.b}")

    def craft_attack(self, neighbor_models: List[OrderedDict]) -> OrderedDict:
        """
        Craft Trim malicious model by placing values beyond the honest range.
        """
        keys = list(neighbor_models[0].keys())
        mal_model = OrderedDict()

        for key in keys:
            stacked = torch.stack([m[key].float() for m in neighbor_models])

            # Compute per-coordinate statistics
            mean_val = stacked.mean(dim=0)
            deviation = torch.sign(mean_val)  # Direction of honest gradient
            max_val = stacked.max(dim=0).values
            min_val = stacked.min(dim=0).values

            # Random values in [0, 1] for interpolation
            r = torch.rand_like(mean_val)

            # Craft malicious values
            mal_val = torch.zeros_like(mean_val)

            # Positive deviation: place in [max, b*max] or [max, max/b] if max <= 0
            pos_mask = deviation > 0
            neg_mask = deviation <= 0

            # For positive deviation coordinates
            max_pos = max_val.clone()
            max_upper = torch.where(max_pos > 0, self.b * max_pos, max_pos / self.b)
            mal_val[pos_mask] = (max_pos[pos_mask] + r[pos_mask] * (max_upper[pos_mask] - max_pos[pos_mask]))

            # For negative deviation coordinates: place in [b*min, min]
            min_neg = min_val.clone()
            min_lower = torch.where(min_neg < 0, self.b * min_neg, min_neg / self.b)
            mal_val[neg_mask] = (min_lower[neg_mask] + r[neg_mask] * (min_neg[neg_mask] - min_lower[neg_mask]))

            mal_model[key] = mal_val.to(neighbor_models[0][key].dtype)

        logging.info(f"[Trim Attack] Crafted malicious model with b={self.b}")
        return mal_model