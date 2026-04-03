"""
Dissensus attack for decentralized federated learning.

Reference: He, L., Karimireddy, S. P., & Jaggi, M. (2022).
"Byzantine-Robust Decentralized Learning via ClippedGossip." arXiv:2202.01545.

The attack cancels or reverses the gossip progress of the target honest node
by crafting: x_byz = x_self - epsilon * gossip_progress
where gossip_progress = mean(x_neighbor - x_self) over honest neighbors.

With epsilon=1, Byzantine nodes exactly cancel honest gossip progress.
With epsilon>1, they reverse it, causing divergence.
"""

import logging
from collections import OrderedDict
from typing import Dict, List

import torch

from nebula.addons.attacks.model.knowledgeattack import KnowledgeModelAttack


class DissensusAttack(KnowledgeModelAttack):
    """
    Dissensus attack: cancels/reverses gossip progress by negating neighbor contributions.
    """

    def __init__(self, engine, attack_params: Dict):
        try:
            round_start = int(attack_params["round_start_attack"])
            round_stop = int(attack_params["round_stop_attack"])
            attack_interval = int(attack_params["attack_interval"])
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid attack params for Dissensus: {e}")

        super().__init__(engine, round_start, round_stop, attack_interval)
        self.epsilon = float(attack_params.get("epsilon", 1.0))

        logging.info(f"[Dissensus] Initialized with epsilon={self.epsilon}")

    def craft_attack(self, neighbor_models: List[OrderedDict]) -> OrderedDict:
        """
        Craft Dissensus malicious model.

        x_byz = x_self - epsilon * mean(x_neighbor - x_self)
        """
        # Get own current model
        self_model = self.engine.trainer.get_model_parameters()
        keys = list(self_model.keys())
        n_neighbors = len(neighbor_models)

        if n_neighbors == 0:
            return self_model

        mal_model = OrderedDict()

        for key in keys:
            self_param = self_model[key].float()

            # Compute gossip progress: mean of (neighbor - self) over all honest neighbors
            gossip_progress = torch.zeros_like(self_param)
            for m in neighbor_models:
                if key in m:
                    gossip_progress += (m[key].float() - self_param)
            gossip_progress /= n_neighbors

            # Craft: x_byz = x_self - epsilon * gossip_progress
            mal_param = self_param - self.epsilon * gossip_progress
            mal_model[key] = mal_param.to(self_model[key].dtype)

        logging.info(
            f"[Dissensus] Crafted malicious model with epsilon={self.epsilon}, "
            f"using {n_neighbors} neighbor models"
        )
        return mal_model