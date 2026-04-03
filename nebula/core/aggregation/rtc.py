"""
RTC (Remove-then-Clip) aggregator for Byzantine-robust decentralized learning.

Reference: Yang, C. & Ghaderi, J. (2024).
"Byzantine-Robust Decentralized Learning via Remove-then-Clip Aggregation."
AAAI Conference on Artificial Intelligence, 38(19), 21735-21743.

Two-phase aggregation:
1. REMOVE: Iteratively remove b most outlying neighbors (farthest from weighted mean)
2. CLIP: Self-centered clipping on remaining neighbors (same as SCClip)
"""

import gc
import logging

import torch

from nebula.core.aggregation.aggregator import Aggregator


class RTC(Aggregator):
    """
    RTC: Remove-then-Clip aggregation.

    Phase 1 (Remove): Iteratively remove b neighbors farthest from the weighted mean.
    Phase 2 (Clip): Clip remaining neighbors' differences from local model.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.b = config.participant.get("aggregator_args", {}).get("rtc_b", 1)
        self.tau = config.participant.get("aggregator_args", {}).get("tau", 10.0)
        logging.info(f"[RTC] Initialized with b={self.b}, tau={self.tau}")

    def _flatten(self, state_dict):
        """Flatten state dict to a single vector."""
        return torch.cat([v.float().flatten() for v in state_dict.values()])

    def _clip(self, diff, tau):
        """CLIP(v, tau) = min(1, tau / ||v||) * v"""
        norm = torch.norm(diff.float()).item()
        if norm > tau and norm > 0:
            return diff.float() * (tau / norm)
        return diff.float()

    def run_aggregation(self, models):
        super().run_aggregation(models)

        models_list = list(models.items())  # [(addr, (state_dict, weight)), ...]
        if len(models_list) == 0:
            return None

        # Find local model
        local_model = None
        local_addr = None
        for addr, (model, weight) in models_list:
            if addr == self._addr:
                local_model = model
                local_addr = addr
                break
        if local_model is None:
            local_model = models_list[0][1][0]
            local_addr = models_list[0][0]

        # Separate neighbors from local
        neighbors = [(addr, model, weight) for addr, (model, weight) in models_list if addr != local_addr]

        # ========== REMOVE phase ==========
        remaining = list(neighbors)
        for removal_round in range(min(self.b, len(remaining) - 1)):
            if len(remaining) <= 1:
                break

            # Compute weighted mean of local + remaining
            all_models = [(local_model, 1.0)] + [(m, w) for _, m, w in remaining]
            total_w = sum(w for _, w in all_models)
            keys = list(local_model.keys())
            mean_model = {}
            for key in keys:
                mean_model[key] = sum(m[key].float() * (w / total_w) for m, w in all_models)

            # Find and remove farthest neighbor from mean
            max_dist = -1
            max_idx = 0
            for i, (addr, m, w) in enumerate(remaining):
                dist = sum(
                    torch.norm(m[key].float() - mean_model[key]).item() ** 2 for key in keys
                ) ** 0.5
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i

            removed = remaining.pop(max_idx)
            logging.info(f"[RTC] Remove phase: removed {removed[0]} (dist={max_dist:.4f})")

        # ========== CLIP phase ==========
        keys = list(local_model.keys())
        accum = {key: torch.zeros_like(local_model[key], dtype=torch.float32) for key in keys}
        n_total = 1 + len(remaining)  # local + remaining neighbors

        # Add local model
        for key in keys:
            accum[key] += local_model[key].float() / n_total

        # Add clipped neighbors
        for addr, m, w in remaining:
            for key in keys:
                diff = m[key].float() - local_model[key].float()
                clipped_diff = self._clip(diff, self.tau)
                accum[key] += (local_model[key].float() + clipped_diff) / n_total

        gc.collect()
        logging.info(f"[RTC] Aggregated: {len(remaining)} neighbors after removing {self.b}, tau={self.tau}")
        return accum