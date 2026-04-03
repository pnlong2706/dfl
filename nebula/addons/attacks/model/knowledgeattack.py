"""
Base class for knowledge-based model poisoning attacks.

These attacks require access to neighbor models before aggregation to craft
optimized malicious updates. Used by Trim, Krum, ALIE, and Dissensus attacks.
"""

import logging
from abc import abstractmethod
from collections import OrderedDict
from functools import wraps
from typing import Dict, List

from nebula.addons.attacks.model.modelattack import ModelAttack


class KnowledgeModelAttack(ModelAttack):
    """
    Base class for attacks that need access to all neighbor models before aggregation.

    Unlike standard ModelAttack (which modifies the aggregated result), this class
    intercepts the models dict BEFORE aggregation, crafts a malicious model using
    knowledge of neighbor models, and replaces the attacker's own model in the dict.

    Subclasses must implement `craft_attack(neighbor_models)`.
    """

    def aggregator_decorator(self):
        """Override: intercept models BEFORE aggregation, replace attacker's model."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args):
                _, *new_args = args  # Exclude self argument
                models_dict = new_args[0] if new_args else {}

                # Collect neighbor models (excluding self)
                neighbor_models = []
                for addr, (model, weight) in models_dict.items():
                    if addr != self.engine.addr:
                        neighbor_models.append(model)

                # Craft malicious model and replace own entry
                if neighbor_models and self.engine.addr in models_dict:
                    try:
                        mal_model = self.craft_attack(neighbor_models)
                        original_weight = models_dict[self.engine.addr][1]
                        models_dict[self.engine.addr] = (mal_model, original_weight)
                        logging.info(
                            f"[{self.__class__.__name__}] Crafted malicious model using "
                            f"{len(neighbor_models)} neighbor models"
                        )
                    except Exception as e:
                        logging.exception(f"[{self.__class__.__name__}] Failed to craft attack: {e}")

                accum = func(*new_args)
                return accum

            return wrapper

        return decorator

    @abstractmethod
    def craft_attack(self, neighbor_models: List[OrderedDict]) -> OrderedDict:
        """
        Craft the malicious model given all neighbor models.

        Args:
            neighbor_models: List of state dicts from honest neighbors.

        Returns:
            OrderedDict: Crafted malicious model state dict.
        """
        raise NotImplementedError

    def model_attack(self, received_weights):
        """Not used by KnowledgeModelAttack — craft_attack handles poisoning."""
        return received_weights