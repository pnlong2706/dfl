"""
SCClip (Self-Centered Clipping) aggregator for Byzantine-robust decentralized learning.

Reference: He, L., Karimireddy, S. P., & Jaggi, M. (2023).
"Byzantine-robust decentralized learning via self-centered clipping."
arXiv preprint arXiv:2202.01545.

For each neighbor j, the full-model difference vector (x_j - x_i) is clipped:
    CLIP(v, tau) = min(1, tau / ||v||) * v
    clipped_j = x_i + CLIP(x_j - x_i, tau)

Then all clipped models are averaged using mixing weights.
"""

import gc
import logging

import torch

from nebula.core.aggregation.aggregator import Aggregator


class SCClip(Aggregator):
    """
    SCClip: Self-Centered Clipping aggregation.

    Clips each neighbor's FULL model difference from the local model,
    bounding the influence any single neighbor can exert.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.tau = config.participant.get("aggregator_args", {}).get("tau", 10.0)
        logging.info(f"[SCClip] Initialized with tau={self.tau}")

    def _model_diff_norm(self, model_a, model_b):
        """Compute L2 norm of full flattened difference vector."""
        dist_sq = 0.0
        for key in model_a:
            if key in model_b:
                dist_sq += torch.sum((model_a[key].float() - model_b[key].float()) ** 2).item()
        return dist_sq ** 0.5

    def run_aggregation(self, models):
        super().run_aggregation(models)

        models_list = list(models.items())
        if len(models_list) == 0:
            return None

        # Find local model as clipping center
        local_model = None
        for addr, (model, weight) in models_list:
            if addr == self._addr:
                local_model = model
                break
        if local_model is None:
            local_model = models_list[0][1][0]

        keys = list(local_model.keys())
        n_models = len(models_list)
        w = 1.0 / n_models  # uniform mixing weight

        accum = {key: torch.zeros_like(local_model[key], dtype=torch.float32) for key in keys}

        with torch.no_grad():
            for addr, (model_params, weight) in models_list:
                # Compute FULL model difference norm (not per-layer)
                diff_norm = self._model_diff_norm(model_params, local_model)

                # CLIP scale: min(1, tau / ||v||)
                if diff_norm > self.tau and diff_norm > 0:
                    scale = self.tau / diff_norm
                else:
                    scale = 1.0

                # Apply same scale to all layers
                for key in keys:
                    diff = model_params[key].float() - local_model[key].float()
                    clipped = local_model[key].float() + diff * scale
                    accum[key] += w * clipped

        del models_list
        gc.collect()

        logging.info(f"[SCClip] Aggregated {n_models} models with tau={self.tau}")
        return accum
