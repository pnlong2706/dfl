"""
ALIE (A Little Is Enough) attack.

Reference: Baruch, G., Baruch, M., & Goldberg, Y. (2019).
"A Little Is Enough: Circumventing Defenses For Distributed Learning." NeurIPS 2019.

The attack crafts malicious updates that are small enough to evade statistical
defenses but consistent enough across Byzantine workers to shift the aggregate:
    mal = mu - z_max * sigma  (per coordinate)
where z_max is derived from the inverse normal CDF based on the number of
total and Byzantine workers.
"""

import logging
import math
from collections import OrderedDict
from typing import Dict, List

import torch
from scipy.stats import norm

from nebula.addons.attacks.model.knowledgeattack import KnowledgeModelAttack


class ALIEAttack(KnowledgeModelAttack):
    """
    ALIE attack: crafts malicious model as mu - z_max * sigma per coordinate.

    All Byzantine workers send the identical crafted model.
    """

    def __init__(self, engine, attack_params: Dict):
        try:
            round_start = int(attack_params["round_start_attack"])
            round_stop = int(attack_params["round_stop_attack"])
            attack_interval = int(attack_params["attack_interval"])
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid attack params for ALIE: {e}")

        super().__init__(engine, round_start, round_stop, attack_interval)

        # Number of attackers (estimated from config)
        n_nodes = engine.config.participant["scenario_args"]["n_nodes"]
        poisoned_percent = float(attack_params.get("poisoned_node_percent", 20))
        self.n_attackers = max(1, int(n_nodes * poisoned_percent / 100))
        self.n_total = n_nodes

        logging.info(
            f"[ALIE] Initialized: n_total={self.n_total}, n_attackers={self.n_attackers}"
        )

    def _compute_z_max(self, n_total: int, n_attackers: int) -> float:
        """
        Compute z_max from the inverse normal CDF.

        z_max = Phi^{-1}((n - m - s) / (n - m))
        where s = floor(n/2 + 1) - m
        """
        n = n_total
        m = n_attackers
        s = math.floor(n / 2 + 1) - m
        if s <= 0:
            logging.warning(f"[ALIE] s={s} <= 0, too many attackers. Using z_max=0.5")
            return 0.5
        n_honest = n - m
        if n_honest <= 0:
            return 0.5
        cdf_value = (n_honest - s) / n_honest
        cdf_value = max(0.01, min(0.99, cdf_value))  # Clamp to valid range
        z_max = norm.ppf(cdf_value)
        logging.info(f"[ALIE] n={n}, m={m}, s={s}, cdf_value={cdf_value:.4f}, z_max={z_max:.4f}")
        return z_max

    def craft_attack(self, neighbor_models: List[OrderedDict]) -> OrderedDict:
        """
        Craft ALIE malicious model: mal = mu - z_max * sigma per coordinate.
        """
        z_max = self._compute_z_max(self.n_total, self.n_attackers)

        # Get layer keys from first model
        keys = list(neighbor_models[0].keys())
        mal_model = OrderedDict()

        for key in keys:
            # Stack all neighbor values for this layer
            stacked = torch.stack([m[key].float() for m in neighbor_models])

            # Compute per-coordinate mean and std
            mu = stacked.mean(dim=0)
            sigma = stacked.std(dim=0)

            # Craft malicious value: mu - z_max * sigma
            mal_model[key] = (mu - z_max * sigma).to(neighbor_models[0][key].dtype)

        logging.info(f"[ALIE] Crafted malicious model with z_max={z_max:.4f}")
        return mal_model