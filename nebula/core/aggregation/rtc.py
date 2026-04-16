"""
RTC (Remove-then-Clip) aggregator for Byzantine-robust decentralized learning.

Reference: Yang, C. & Ghaderi, J. (2024).
"Byzantine-Robust Decentralized Learning via Remove-then-Clip Aggregation."
AAAI Conference on Artificial Intelligence, 38(19), 21735-21743.

Two-phase aggregation:
1. REMOVE: Iteratively remove neighbors farthest from local model until
   cumulative weight of removed nodes exceeds delta_max (Algo 1)
2. CLIP: Self-centered clipping with CLIP(v, tau) = min(1, tau/||v||^2) * v (Eq. 11)
   where tau is computed adaptively from remaining neighbor distances (Eq. 10)
"""

import gc
import logging
from collections import OrderedDict

import torch

from nebula.core.aggregation.aggregator import Aggregator


class RTC(Aggregator):
    """
    RTC: Remove-then-Clip aggregation (AAAI 2024).

    Phase 1 (Remove): Iteratively remove neighbors farthest from local model
                      until cumulative removed weight > delta_max.
    Phase 2 (Clip): CLIP(v, tau) = min(1, tau/||v||^2) * v on remaining neighbors.
    Final: weighted sum per Equation 9 from the paper.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        # b: max number of removals (count-based approximation of delta_max)
        self.b = config.participant.get("aggregator_args", {}).get("rtc_b", 1)
        # tau: if > 0, used as fixed clipping threshold; if 0, computed adaptively (Eq. 10)
        self.tau = config.participant.get("aggregator_args", {}).get("tau", 0)
        logging.info(f"[RTC] Initialized with b={self.b}, tau={self.tau} ({'fixed' if self.tau > 0 else 'adaptive'})")

    def _model_distance_sq(self, model_a, model_b):
        """Compute squared L2 distance between two models."""
        dist_sq = 0.0
        for key in model_a:
            if key in model_b:
                dist_sq += torch.sum((model_a[key].float() - model_b[key].float()) ** 2).item()
        return dist_sq

    def _compute_adaptive_tau(self, local_model, remaining, delta_max):
        """
        Compute adaptive tau per Equation 10:
        tau_i = (1/delta_max) * sqrt(sum_{j in S_i} w_ij * ||x_i - x_j||^2)

        For uniform weights w_ij = 1/n, this becomes:
        tau_i = (1/delta_max) * sqrt(sum ||x_i - x_j||^2 / n)
        """
        if not remaining or delta_max <= 0:
            return 10.0  # fallback

        n = len(remaining) + 1  # including self
        weighted_dist_sq = 0.0
        for _, m, w in remaining:
            weighted_dist_sq += self._model_distance_sq(m, local_model) / n

        tau = (1.0 / delta_max) * (weighted_dist_sq ** 0.5)
        return max(tau, 1e-6)  # prevent zero

    def run_aggregation(self, models):
        super().run_aggregation(models)

        models_list = list(models.items())
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
        n_total = len(models_list)  # total nodes including self
        keys = list(local_model.keys())

        # Uniform mixing weight (fully-connected: w_ij = 1/n)
        w = 1.0 / n_total

        # ========== REMOVE phase (Algorithm 1) ==========
        # Remove b neighbors farthest from local model x_i
        remaining = list(neighbors)
        removed = []
        for _ in range(min(self.b, len(remaining) - 1)):
            if len(remaining) <= 1:
                break

            max_dist_sq = -1
            max_idx = 0
            for i, (addr, m, _w) in enumerate(remaining):
                dist_sq = self._model_distance_sq(m, local_model)
                if dist_sq > max_dist_sq:
                    max_dist_sq = dist_sq
                    max_idx = i

            r = remaining.pop(max_idx)
            removed.append(r)
            logging.info(f"[RTC] Remove: {r[0]} (dist={max_dist_sq**0.5:.4f})")

        # ========== Compute tau ==========
        if self.tau > 0:
            tau = self.tau
        else:
            # Adaptive tau (Eq. 10): delta_max = b * w
            delta_max = max(self.b * w, 1e-6)
            tau = self._compute_adaptive_tau(local_model, remaining, delta_max)
            logging.info(f"[RTC] Adaptive tau={tau:.4f} (delta_max={delta_max:.4f})")

        # ========== CLIP phase + Aggregation (Equation 9) ==========
        # Eq 9: RTC_i = sum_{j in S_i} w_ij * (x_i + CLIP(x_j - x_i, tau))
        #              + sum_{j in removed} w_ij * x_i
        #              + w_ii * x_i
        #
        # CLIP(v, tau) = min(1, tau / ||v||^2) * v  (Eq. 11)

        accum = {key: torch.zeros_like(local_model[key], dtype=torch.float32) for key in keys}

        # Self-weight: w_ii * x_i
        for key in keys:
            accum[key] += w * local_model[key].float()

        # Remaining neighbors: w_ij * (x_i + CLIP(x_j - x_i, tau))
        for addr, m, _w in remaining:
            # Compute full-model difference for norm
            diff_norm_sq = self._model_distance_sq(m, local_model)

            # CLIP scale: min(1, tau / ||v||^2)
            if diff_norm_sq > tau and diff_norm_sq > 0:
                scale = tau / diff_norm_sq
            else:
                scale = 1.0

            for key in keys:
                diff = m[key].float() - local_model[key].float()
                clipped = local_model[key].float() + diff * scale
                accum[key] += w * clipped

        # Removed neighbors: w_ij * x_i (contribute local model)
        for addr, m, _w in removed:
            for key in keys:
                accum[key] += w * local_model[key].float()

        gc.collect()
        logging.info(f"[RTC] Aggregated: {len(remaining)} kept, {len(removed)} removed, tau={tau:.4f}")
        return accum
