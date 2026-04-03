"""
SCClip (Self-Centered Clipping) aggregator for Byzantine-robust decentralized learning.

Reference: He, L., Karimireddy, S. P., & Jaggi, M. (2023).
"Byzantine-robust decentralized learning via self-centered clipping."
arXiv preprint arXiv:2202.01545.

For each neighbor j, the difference (x_j - x_i) is clipped to have norm at most tau:
    clipped_j = x_i + CLIP(x_j - x_i, tau)
    CLIP(v, tau) = min(1, tau / ||v||) * v

Then all clipped models are averaged.
"""

import gc
import logging

import torch

from nebula.core.aggregation.aggregator import Aggregator


class SCClip(Aggregator):
    """
    SCClip: Self-Centered Clipping aggregation.

    Clips each neighbor's difference from the local model to bound the influence
    any single neighbor can exert.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.tau = config.participant.get("aggregator_args", {}).get("tau", 10.0)
        logging.info(f"[SCClip] Initialized with tau={self.tau}")

    def _clip(self, diff, tau):
        """CLIP(v, tau) = min(1, tau / ||v||) * v"""
        norm = torch.norm(diff.float()).item()
        if norm > tau and norm > 0:
            return diff.float() * (tau / norm)
        return diff.float()

    def run_aggregation(self, models):
        super().run_aggregation(models)

        models_list = list(models.values())
        if len(models_list) == 0:
            return None

        # Use the local model (self._addr) as the center for clipping
        # If local model is in the dict, use it; otherwise use the first model
        local_model = None
        for addr, (model, weight) in models.items():
            if addr == self._addr:
                local_model = model
                break
        if local_model is None:
            local_model = models_list[0][0]

        keys = list(local_model.keys())
        accum = {key: torch.zeros_like(local_model[key], dtype=torch.float32) for key in keys}
        n_models = len(models_list)

        with torch.no_grad():
            for model_params, weight in models_list:
                for key in keys:
                    diff = model_params[key].float() - local_model[key].float()
                    clipped_diff = self._clip(diff, self.tau)
                    # clipped model = local + clipped(neighbor - local)
                    accum[key] += (local_model[key].float() + clipped_diff) / n_models

        del models_list
        gc.collect()

        logging.info(f"[SCClip] Aggregated {n_models} models with tau={self.tau}")
        return accum