import copy
import logging
import time
import sys
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Optional

from nebula.core.aggregation.updatehandlers.updatehandler import UpdateHandler
from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import UpdateNeighborEvent, UpdateReceivedEvent
from nebula.core.utils.locker import Locker

if TYPE_CHECKING:
    from nebula.core.aggregation.aggregator import Aggregator

logging.basicConfig(
    level=logging.INFO,  # or DEBUG if you want more detail
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class Update:
    """
    Represents a model update received from a node in a specific training round.

    Attributes:
        model (object): The model object or weights received.
        weight (float): The weight or importance of the update.
        source (str): Identifier of the node that sent the update.
        round (int): Training round this update belongs to.
        time_received (float): Timestamp when the update was received.
    """
    def __init__(self, model, weight, source, round, time_received):
        self.model = model
        self.weight = weight
        self.source = source
        self.round = round
        self.time_received = time_received

    def __eq__(self, other):
        """
        Checks if two updates belong to the same round.
        """
        return self.round == other.round


MAX_UPDATE_BUFFER_SIZE = 1  # Modify to create an historic


class DFLUpdateHandler(UpdateHandler):
    """
    Distributed Federated Learning (DFL) Update Handler.

    This handler manages the reception, storage, and tracking of model updates from federation nodes
    during asynchronous rounds. It supports partial updates, late arrivals, and maintains update history.
    """

    def __init__(self, aggregator, addr, buffersize=MAX_UPDATE_BUFFER_SIZE):
        """
        Initialize the update handler with required locks and storage.

        Args:
            aggregator (Aggregator): Aggregator instance for the federation.
            addr (str): Address of the local node.
            buffersize (int): Maximum number of historical updates to keep per node.
        """
        self._addr = addr
        self._aggregator: Aggregator = aggregator
        self._buffersize = buffersize
        self._updates_storage: dict[str, tuple[Update, deque[Update]]] = {}
        self._updates_storage_lock = Locker(name="updates_storage_lock", async_lock=True)
        self._sources_expected = set()
        self._sources_received = set()
        self._round_updates_lock = Locker(name="round_updates_lock", async_lock=True)
        self._update_federation_lock = Locker(name="update_federation_lock", async_lock=True)
        self._notification_sent_lock = Locker(name="notification_sent_lock", async_lock=True)
        self._notification = False
        self._missing_ones = set()
        self._nodes_using_historic = set()

        # Pseudo Aggregation: EMA storage for neighbor models
        self._old_models: dict[str, OrderedDict] = {}  # neighbor_id -> previous model state_dict
        self._old_weight: dict[str, float] = {}
        self._old_model_rounds: dict[str, int] = {}  # neighbor_id -> round when model was received
        self._ema_deltas: dict[str, OrderedDict] = {}  # neighbor_id -> EMA of deltaW
        self._pseudo_agg_enabled = False
        self._ema_alpha = 0.25  # Default EMA weight for new delta (will be overridden by config)
        self._max_round_staleness = 5  # Maximum round difference before excluding from pseudo agg

        # Step scheduler for adjusted weight decay
        self._weight_drop_rate = 1.0  # Multiplicative factor (< 1 to decay weight over time)
        self._weight_schedule_step = 1  # Number of physical rounds before applying drop_rate
        self._stop_pseudo_round = None  # Physical round after which to stop pseudo aggregation (None = no limit)

        # PRT (Prediction-Residual Trust) configuration
        self._prt_enabled = False
        self._prt_score_type = "exponential"
        self._prt_scale = 1.0
        self._prt_min_trust = 0.1
        self._prt_trust_smoothing = 0.5
        self._prt_warmup_rounds = 2
        self._prt_apply_to_pseudo = True
        self._prt_trust_scores: dict[str, float] = {}  # neighbor_id -> smoothed trust

        # Adaptive PRT extensions
        self._prt_adaptive = True  # Use z-score relative scoring
        self._prt_exclusion_z = 2.5  # Z-score threshold for hard exclusion
        self._prt_direction_check = True  # Enable cosine similarity check
        self._prt_direction_penalty = 0.3  # Penalty for opposing consensus direction
        self._prt_pending_residuals: dict[str, float] = {}  # neighbor_id -> raw residual (current round)
        self._prt_pending_deltas: dict[str, object] = {}  # neighbor_id -> model delta (current round)
        self._prt_suspicion_count: dict[str, int] = {}  # neighbor_id -> cumulative suspicion count

    @property
    def us(self):
        """Returns the internal updates storage dictionary."""
        return self._updates_storage

    @property
    def agg(self):
        """Returns the aggregator instance."""
        return self._aggregator

    async def init(self, config=None):
        """
        Subscribe to update-related events from the event manager.
        """
        await EventManager.get_instance().subscribe_node_event(UpdateNeighborEvent, self.notify_federation_update)
        await EventManager.get_instance().subscribe_node_event(UpdateReceivedEvent, self.storage_update)

    async def round_expected_updates(self, federation_nodes: set):
        """
        Define which nodes are expected to send updates in this round and reset internal state.

        Args:
            federation_nodes (set): Set of node IDs expected to participate this round.
        """
        await self._update_federation_lock.acquire_async()
        await self._updates_storage_lock.acquire_async()
        self._sources_expected = federation_nodes.copy()
        self._sources_received.clear()

        # Initialize new nodes
        for fn in federation_nodes:
            if fn not in self.us:
                self.us[fn] = (None, deque(maxlen=self._buffersize))

        # Clear removed nodes
        removed_nodes = [node for node in self._updates_storage.keys() if node not in federation_nodes]
        for rn in removed_nodes:
            del self._updates_storage[rn]

        # Check already received updates
        await self._check_updates_already_received()

        await self._updates_storage_lock.release_async()
        await self._update_federation_lock.release_async()

        # Lock to check if all updates received
        if self._round_updates_lock.locked():
            self._round_updates_lock.release_async()

        self._notification = False

    async def _check_updates_already_received(self):
        """
        Scan storage for updates already received in this round.
        """
        for se in self._sources_expected:
            (last_updt, node_storage) = self._updates_storage[se]
            if len(node_storage):
                try:
                    if (last_updt and node_storage[-1] and last_updt != node_storage[-1]) or (
                        node_storage[-1] and not last_updt
                    ):
                        self._sources_received.add(se)
                        logging.info(
                            f"Update already received from source: {se} | ({len(self._sources_received)}/{len(self._sources_expected)}) Updates received"
                        )
                except:
                    logging.exception(
                        f"ERROR: source expected: {se} | last_update None: {(True if not last_updt else False)}, last update storaged None: {(True if not node_storage[-1] else False)}"
                    )

    async def storage_update(self, updt_received_event: UpdateReceivedEvent):
        """
        Store an incoming update and trigger aggregation if all updates are received.

        Args:
            updt_received_event (UpdateReceivedEvent): Event with model update data.
        """
        time_received = time.time()
        (model, weight, source, round, _) = await updt_received_event.get_event_data()
        if source in self._sources_expected:
            updt = Update(model, weight, source, round, time_received)
            await self._updates_storage_lock.acquire_async()
            if updt in self.us[source][1]:
                logging.info(f"Discard | Alerady received update from source: {source} for round: {round}")
            else:
                last_update_used = self.us[source][0]
                self.us[source][1].append(updt)
                self.us[source] = (last_update_used, self.us[source][1])
                self._old_weight[source] = weight
                logging.info(
                    f"Storage Update | source={source} | round={round} | weight={weight} | federation nodes: {self._sources_expected}"
                )

                # Update EMA if pseudo aggregation is enabled (for actual aggregation rounds)
                if self._pseudo_agg_enabled:
                    # Store residual for PRT BEFORE update_ema (which overwrites _old_models)
                    if self._prt_enabled and source in self._old_models and source in self._ema_deltas:
                        self._store_prt_residual(source, model, round)
                    self._old_model_rounds[source] = round  # Track round when model was received
                    self.update_ema(source, model)

                self._sources_received.add(source)
                updates_left = self._sources_expected.difference(self._sources_received)
                logging.info(
                    f"Updates received ({len(self._sources_received)}/{len(self._sources_expected)}) | Missing nodes: {updates_left}"
                )
                if self._round_updates_lock.locked() and not updates_left:
                    all_rec = await self._all_updates_received()
                    if all_rec:
                        await self._notify()
            await self._updates_storage_lock.release_async()
        else:
            if source not in self._sources_received:
                logging.info(f"Discard update | source: {source} not in expected updates for this Round")

    async def get_round_updates(self):
        """
        Retrieve the most recent valid updates for this round, filling gaps if needed.

        Returns:
            dict: A dictionary mapping node ID to (model, weight) tuples.
        """
        await self._updates_storage_lock.acquire_async()
        updates_missing = self._sources_expected.difference(self._sources_received)
        if updates_missing:
            self._missing_ones = updates_missing
            logging.info(f"Missing updates from sources: {updates_missing}")
        else:
            self._missing_ones.clear()

        self._nodes_using_historic.clear()

        # Finalize PRT trust scores using all residuals from this round
        if self._prt_enabled and self._prt_pending_residuals:
            self._finalize_prt_trust()

        updates = {}
        for sr in self._sources_received:
            source_historic = self.us[sr][1]
            last_updt_received = self.us[sr][0]
            updt: Update = None
            updt = source_historic[-1]  # Get last update received
            if last_updt_received and last_updt_received == updt:
                logging.info(f"Missing update from source: {sr}, using last update received..")
                self._nodes_using_historic.add(sr)
            else:
                last_updt_received = updt
                self.us[sr] = (last_updt_received, source_historic)  # Update storage with new last update used
            # Apply PRT trust to weight for real rounds
            weight = updt.weight
            if self._prt_enabled and sr in self._prt_trust_scores:
                trust = self._prt_trust_scores[sr]
                if trust <= 0:
                    logging.info(f"PRT EXCLUDED | neighbor={sr} | trust=0 (hard exclusion)")
                    continue  # Fully exclude this neighbor
                weight = weight * trust
                logging.info(f"PRT weight adjustment | neighbor={sr} | original={updt.weight:.2f} | trust={trust:.4f} | effective={weight:.2f}")
            updates[sr] = (updt.model, weight)

        await self._updates_storage_lock.release_async()
        return updates

    async def notify_federation_update(self, updt_nei_event: UpdateNeighborEvent):
        """
        Handle federation node join/leave events.

        Args:
            updt_nei_event (UpdateNeighborEvent): Event with neighbor update data.
        """
        source, remove = await updt_nei_event.get_event_data()
        if not remove:
            if self._round_updates_lock.locked():
                logging.info(f"Source: {source} will be count next round")
            else:
                await self._update_source(source, remove)
        else:
            if source not in self._sources_received:  # Not received update from this source yet
                await self._update_source(source, remove=True)
                all_rec = await self._all_updates_received()  # Verify if discarding node aggregation could be done
                if all_rec:
                    await self._notify()
            else:
                logging.info(f"Already received update from: {source}, it will be discarded next round")

    async def _update_source(self, source, remove=False):
        """
        Add or remove a node from the expected sources.

        Args:
            source (str): Node ID.
            remove (bool): Whether to remove the node from the expected list.
        """
        logging.info(f"🔄 Update | remove: {remove} | source: {source}")
        await self._updates_storage_lock.acquire_async()
        if remove:
            self._sources_expected.discard(source)
        else:
            self.us[source] = (None, deque(maxlen=self._buffersize))
            self._sources_expected.add(source)
        logging.info(f"federation nodes expected this round: {self._sources_expected}")
        await self._updates_storage_lock.release_async()

    async def get_round_missing_nodes(self):
        """
        Return the set of nodes whose updates were not received this round.

        Returns:
            set: Missing node IDs.
        """
        return self._missing_ones

    async def notify_if_all_updates_received(self):
        """
        Set a notification trigger and notify aggregator if all updates are already received.
        """
        logging.info("Set notification when all expected updates received")
        await self._round_updates_lock.acquire_async()
        await self._updates_storage_lock.acquire_async()
        all_received = await self._all_updates_received()
        await self._updates_storage_lock.release_async()
        if all_received:
            await self._notify()

    async def stop_notifying_updates(self):
        """
        Cancel any notification triggers for update reception.
        """
        if self._round_updates_lock.locked():
            logging.info("Stop notification updates")
            await self._round_updates_lock.release_async()

    async def _notify(self):
        """
        Notify the aggregator that all expected updates have been received.
        """
        await self._notification_sent_lock.acquire_async()
        if self._notification:
            await self._notification_sent_lock.release_async()
            return
        self._notification = True
        await self.stop_notifying_updates()
        await self._notification_sent_lock.release_async()
        logging.info("🔄 Notifying aggregator to release aggregation")
        await self.agg.notify_all_updates_received()

    async def _all_updates_received(self):
        """
        Check if all expected updates have been received.

        Returns:
            bool: True if no updates are missing.
        """
        updates_left = self._sources_expected.difference(self._sources_received)
        all_received = False
        if len(updates_left) == 0:
            logging.info("All updates have been received this round")
            if await self._round_updates_lock.locked_async():
                await self._round_updates_lock.release_async()
            all_received = True
        return all_received

    # ========== Pseudo Aggregation Methods ==========

    def enable_pseudo_aggregation(
        self,
        ema_alpha: float = 0.25,
        weight_drop_rate: float = 1.0,
        weight_schedule_step: int = 1,
        stop_pseudo_round: int = None
    ):
        """
        Enable pseudo aggregation with EMA-based model prediction and weight scheduling.

        Args:
            ema_alpha (float): Weight for new delta in EMA calculation (default: 0.25).
                             EMA_new = (1 - ema_alpha) * EMA_old + ema_alpha * delta
            weight_drop_rate (float): Multiplicative decay factor for adjusted_weight (default: 1.0 = no decay).
                                    Applied every weight_schedule_step physical rounds.
                                    Formula: adjusted_weight *= drop_rate ^ (physical_round // schedule_step)
            weight_schedule_step (int): Number of physical rounds between weight decay applications (default: 1).
            stop_pseudo_round (int): Physical round after which pseudo aggregation stops (default: None = no limit).
        """
        self._pseudo_agg_enabled = True
        self._ema_alpha = ema_alpha
        self._weight_drop_rate = weight_drop_rate
        self._weight_schedule_step = max(1, weight_schedule_step)  # Ensure at least 1
        self._stop_pseudo_round = stop_pseudo_round
        logging.info(
            f"Pseudo Aggregation enabled: ema_alpha={ema_alpha}, "
            f"weight_drop_rate={weight_drop_rate}, "
            f"weight_schedule_step={weight_schedule_step}, "
            f"stop_pseudo_round={stop_pseudo_round}"
        )

    def disable_pseudo_aggregation(self):
        """Disable pseudo aggregation and clear EMA storage."""
        self._pseudo_agg_enabled = False
        self._old_models.clear()
        self._ema_deltas.clear()
        logging.info("Pseudo Aggregation disabled")

    def store_old_model(self, neighbor_id: str, model_state_dict: OrderedDict):
        """
        Store a neighbor's model for future EMA calculation.

        Args:
            neighbor_id (str): Identifier of the neighbor node.
            model_state_dict (OrderedDict): Model state dictionary to store.
        """
        self._old_models[neighbor_id] = copy.deepcopy(model_state_dict)
        logging.debug(f"Stored old model from neighbor {neighbor_id}")

    def update_ema(self, neighbor_id: str, new_model_state_dict: OrderedDict):
        """
        Update EMA when receiving new model from neighbor.

        This method:
        1. Calculates deltaW = newW - oldW
        2. Updates EMA: EMA_new = (1 - alpha) * EMA_old + alpha * deltaW
        3. Stores newW as oldW for next round

        Args:
            neighbor_id (str): Identifier of the neighbor node.
            new_model_state_dict (OrderedDict): Newly received model state dictionary.
        """
        if not self._pseudo_agg_enabled:
            return

        if neighbor_id not in self._old_models:
            # First time receiving from this neighbor, just store
            self.store_old_model(neighbor_id, new_model_state_dict)
            logging.info(f"First model received from neighbor {neighbor_id}, stored as baseline (no EMA yet)")
            return

        # Calculate delta: newW - oldW
        old_model = self._old_models[neighbor_id]
        delta = OrderedDict()

        # Skip BatchNorm running statistics - these are data-dependent, not gradient-dependent
        # Predicting them via EMA produces invalid statistics that corrupt model predictions
        skip_keys = {'running_mean', 'running_var', 'num_batches_tracked'}

        for key in new_model_state_dict.keys():
            # Skip buffers that shouldn't be predicted (e.g., BatchNorm running stats)
            if any(skip_key in key for skip_key in skip_keys):
                continue

            if key in old_model:
                delta[key] = new_model_state_dict[key] - old_model[key]
            else:
                # New parameter appeared, use new value as delta
                delta[key] = new_model_state_dict[key]

        # Update EMA
        if neighbor_id not in self._ema_deltas:
            # First delta calculation, initialize EMA with this delta
            self._ema_deltas[neighbor_id] = delta
            logging.info(f"Initialized EMA for neighbor {neighbor_id} with first delta")
        else:
            # Update existing EMA: EMA_new = (1 - alpha) * EMA_old + alpha * delta
            old_ema = self._ema_deltas[neighbor_id]
            new_ema = OrderedDict()

            for key in delta.keys():
                if key in old_ema:
                    new_ema[key] = (1 - self._ema_alpha) * old_ema[key] + self._ema_alpha * delta[key]
                else:
                    # New parameter in delta, initialize with current delta
                    new_ema[key] = delta[key]

            self._ema_deltas[neighbor_id] = new_ema
            logging.debug(f"Updated EMA for neighbor {neighbor_id}")

        # Store new model as old for next round
        self.store_old_model(neighbor_id, new_model_state_dict)

    def predict_neighbor_model(self, neighbor_id: str, current_round: int = None) -> Optional[OrderedDict]:
        """
        Predict neighbor's model using oldW + EMA * scaling_factor for pseudo aggregation.

        Prediction formula:
            predictedW = oldW + EMA * scaling_factor
            scaling_factor = max(current_round - model_round + 0.5, 0.5)

        The scaling factor increases with staleness:
        - Same round (diff=0): 0.5 (conservative)
        - 1 round old (diff=1): 1.5 (extrapolate further)
        - 2 rounds old (diff=2): 2.5 (extrapolate even more)

        Args:
            neighbor_id (str): Identifier of the neighbor to predict.
            current_round (int, optional): Current round number for staleness calculation.

        Returns:
            OrderedDict: Predicted model state dict, or None if no history available.
        """
        if not self._pseudo_agg_enabled:
            logging.warning("Pseudo aggregation is not enabled, cannot predict models")
            return None

        if neighbor_id not in self._old_models:
            # No history for this neighbor, skip prediction
            logging.debug(f"No history for neighbor {neighbor_id}, skipping prediction")
            return None

        old_model = self._old_models[neighbor_id]

        if neighbor_id not in self._ema_deltas:
            # Have oldW but no EMA yet (only received once)
            # Predict no change: return old model as-is
            logging.debug(f"No EMA for neighbor {neighbor_id}, predicting no change (using old model)")
            return copy.deepcopy(old_model)

        # Calculate scaling factor based on staleness
        scaling_factor = 0.5  # Default if no round info
        if current_round is not None and neighbor_id in self._old_model_rounds:
            model_round = self._old_model_rounds[neighbor_id]
            round_diff = current_round - model_round
            scaling_factor = max(round_diff + 0.5, 0.5)
            logging.debug(
                f"Neighbor {neighbor_id}: model from round {model_round}, current {current_round}, "
                f"diff={round_diff}, scaling_factor={scaling_factor}"
            )

        # Predict: oldW + EMA * scaling_factor
        # Skip BatchNorm buffers - use stored values instead of predicting
        skip_keys = {'running_mean', 'running_var', 'num_batches_tracked'}

        ema = self._ema_deltas[neighbor_id]
        predicted = OrderedDict()

        for key in old_model.keys():
            # For BatchNorm buffers, use the stored value (don't predict)
            if any(skip_key in key for skip_key in skip_keys):
                predicted[key] = old_model[key]
            elif key in ema:
                # For parameters, predict using EMA
                predicted[key] = old_model[key] + ema[key] * scaling_factor
            else:
                # Parameter exists in old model but not in EMA, keep old value
                predicted[key] = old_model[key]

        logging.debug(f"Predicted model for neighbor {neighbor_id} with scaling factor {scaling_factor:.2f}")
        return predicted

    async def get_predicted_models(self, federation_nodes: set) -> dict:
        """
        Get predicted models for all federation nodes for pseudo aggregation.

        Models are filtered based on staleness:
        - If model is older than max_round_staleness rounds: exclude from aggregation
        - Otherwise: weight is adjusted by staleness and optional step scheduler

        Step Scheduler:
        - adjusted_weight *= drop_rate ^ (physical_round // schedule_step)
        - Allows gradual weight decay over training

        Args:
            federation_nodes (set): Set of neighbor node IDs.

        Returns:
            dict: Dictionary mapping neighbor_id to (predicted_model, adjusted_weight) tuples.
                 Only includes neighbors with fresh enough predictions.
                 Returns empty dict if past stop_pseudo_round.
        """
        predicted_models = {}
        skipped_neighbors = []
        stale_neighbors = []

        # Get current round from engine (logical round)
        current_round = self._aggregator.engine.round if self._aggregator.engine.round is not None else 0

        # Calculate physical round (pseudo aggregation doubles rounds, so physical = logical // 2)
        physical_round = (current_round + 1) // 2

        # Check if we should stop pseudo aggregation
        if self._stop_pseudo_round is not None and physical_round > self._stop_pseudo_round:
            logging.info(
                f"Pseudo Aggregation stopped: physical_round={physical_round} > stop_pseudo_round={self._stop_pseudo_round}"
            )
            return {}

        for neighbor_id in federation_nodes:
            predicted_model = self.predict_neighbor_model(neighbor_id, current_round)
            if predicted_model is None:
                skipped_neighbors.append(neighbor_id)
                continue

            # Check staleness
            if neighbor_id not in self._old_model_rounds:
                # No round info, use default weight
                base_weight = self._old_weight.get(neighbor_id, 100.0)
                predicted_models[neighbor_id] = (predicted_model, base_weight)
                logging.debug(f"Neighbor {neighbor_id}: no round info, using default weight {base_weight}")
                continue

            model_round = self._old_model_rounds[neighbor_id]
            round_diff = current_round - model_round

            # Exclude if too stale (older than 5 rounds)
            if round_diff > self._max_round_staleness:
                stale_neighbors.append((neighbor_id, round_diff))
                logging.info(
                    f"Excluding neighbor {neighbor_id} from pseudo aggregation: "
                    f"model is {round_diff} rounds old (current={current_round}, model_round={model_round})"
                )
                continue

            # Adjust weight based on staleness
            base_weight = self._old_weight.get(neighbor_id, 100.0)
            staleness_penalty = max(1, round_diff + 1)
            adjusted_weight = base_weight / staleness_penalty

            # Apply step scheduler: adjusted_weight *= drop_rate ^ (physical_round // schedule_step)
            if self._weight_drop_rate != 1.0:
                schedule_steps = physical_round // self._weight_schedule_step
                weight_decay_factor = self._weight_drop_rate ** schedule_steps
                adjusted_weight *= weight_decay_factor
                logging.info(
                    f"Neighbor {neighbor_id}: model from round {model_round} (diff={round_diff}), "
                    f"weight: {base_weight:.2f} / {staleness_penalty} * {self._weight_drop_rate}^{schedule_steps} = {adjusted_weight:.2f}"
                )
            else:
                logging.info(
                    f"Neighbor {neighbor_id}: model from round {model_round} (diff={round_diff}), "
                    f"weight adjusted: {base_weight:.2f} / {staleness_penalty} = {adjusted_weight:.2f}"
                )

            # Apply PRT trust to pseudo round weights
            if self._prt_enabled and self._prt_apply_to_pseudo and neighbor_id in self._prt_trust_scores:
                trust = self._prt_trust_scores[neighbor_id]
                adjusted_weight *= trust
                logging.info(f"PRT pseudo weight | neighbor={neighbor_id} | trust={trust:.4f} | final_weight={adjusted_weight:.2f}")

            predicted_models[neighbor_id] = (predicted_model, adjusted_weight)

        # Log summary
        if stale_neighbors:
            logging.info(
                f"Pseudo Aggregation: Predicted {len(predicted_models)} models, "
                f"excluded {len(stale_neighbors)} stale neighbors: {[(n, d) for n, d in stale_neighbors]}, "
                f"skipped {len(skipped_neighbors)} without history"
            )
        elif skipped_neighbors:
            logging.info(
                f"Pseudo Aggregation: Predicted {len(predicted_models)} models, "
                f"skipped {len(skipped_neighbors)} neighbors without history: {skipped_neighbors}"
            )
        else:
            logging.info(f"Pseudo Aggregation: Predicted {len(predicted_models)} models for all neighbors")

        return predicted_models

    # ========== PRT (Prediction-Residual Trust) Methods ==========

    def enable_prt(
        self,
        score_type: str = "exponential",
        scale: float = 1.0,
        min_trust: float = 0.1,
        trust_smoothing: float = 0.5,
        warmup_rounds: int = 2,
        apply_to_pseudo: bool = True,
        adaptive: bool = True,
        exclusion_z: float = 2.5,
        direction_check: bool = True,
        direction_penalty: float = 0.3,
    ):
        """
        Enable PRT (Prediction-Residual Trust) for Byzantine defense.

        PRT compares actual received models to EMA predictions. Large residuals
        reduce a neighbor's trust score, which scales down its aggregation weight.

        Adaptive mode (default) uses z-score relative scoring across all neighbors
        instead of fixed-scale absolute scoring, plus directional consistency checks.

        Requires Pseudo Aggregation to be enabled (for EMA predictions).

        Args:
            score_type: Trust function type ("exponential" or "inverse"). Used in non-adaptive mode.
            scale: Scaling factor for residual in trust function. Used in non-adaptive mode.
            min_trust: Floor value for trust score (prevents zero weight in non-adaptive mode).
            trust_smoothing: EMA smoothing for trust updates (0=keep old, 1=use new).
            warmup_rounds: Number of rounds before PRT starts (trust=1.0 during warmup).
            apply_to_pseudo: Whether to apply trust scores to pseudo round weights.
            adaptive: Use z-score relative scoring instead of fixed-scale absolute scoring.
            exclusion_z: Z-score threshold for hard exclusion (adaptive mode only).
            direction_check: Enable cosine similarity check against consensus direction.
            direction_penalty: Trust multiplier for updates opposing consensus (0=full penalty, 1=no penalty).
        """
        self._prt_enabled = True
        self._prt_score_type = score_type
        self._prt_scale = scale
        self._prt_min_trust = min_trust
        self._prt_trust_smoothing = trust_smoothing
        self._prt_warmup_rounds = warmup_rounds
        self._prt_apply_to_pseudo = apply_to_pseudo
        self._prt_adaptive = adaptive
        self._prt_exclusion_z = exclusion_z
        self._prt_direction_check = direction_check
        self._prt_direction_penalty = direction_penalty
        logging.info(
            f"PRT enabled: adaptive={adaptive}, exclusion_z={exclusion_z}, "
            f"direction_check={direction_check}, direction_penalty={direction_penalty}, "
            f"score_type={score_type}, scale={scale}, min_trust={min_trust}, "
            f"trust_smoothing={trust_smoothing}, warmup_rounds={warmup_rounds}, apply_to_pseudo={apply_to_pseudo}"
        )

    def _store_prt_residual(self, neighbor_id: str, actual_model: OrderedDict, current_round: int):
        """
        Compute and store the prediction residual and model delta for a neighbor.
        Called when each model arrives; trust is finalized later in _finalize_prt_trust.
        """
        if not self._prt_enabled or not self._pseudo_agg_enabled:
            return
        if current_round < self._prt_warmup_rounds:
            return

        predicted_model = self.predict_neighbor_model(neighbor_id, current_round)
        if predicted_model is None:
            return

        skip_keys = {'running_mean', 'running_var', 'num_batches_tracked'}
        residual_sq_sum = 0.0
        param_count = 0
        delta_flat = []
        for key in actual_model.keys():
            if any(sk in key for sk in skip_keys):
                continue
            if key in predicted_model:
                diff = actual_model[key].float() - predicted_model[key].float()
                residual_sq_sum += diff.pow(2).sum().item()
                param_count += diff.numel()

        if param_count == 0:
            return

        normalized_residual = (residual_sq_sum / param_count) ** 0.5
        self._prt_pending_residuals[neighbor_id] = normalized_residual

        # Store model delta (actual - old) for directional check
        if self._prt_direction_check and neighbor_id in self._old_models:
            import torch
            old_model = self._old_models[neighbor_id]
            delta = []
            for key in actual_model.keys():
                if any(sk in key for sk in skip_keys):
                    continue
                if key in old_model:
                    delta.append((actual_model[key].float() - old_model[key].float()).flatten())
            if delta:
                self._prt_pending_deltas[neighbor_id] = torch.cat(delta)

    def _finalize_prt_trust(self):
        """
        Compute final trust scores using all pending residuals from this round.

        Adaptive mode: uses z-score relative scoring (outlier detection across neighbors).
        Non-adaptive mode: uses fixed-scale absolute scoring (original behavior).
        """
        import math
        import torch

        if not self._prt_pending_residuals:
            return

        residuals = self._prt_pending_residuals
        neighbor_ids = list(residuals.keys())

        if self._prt_adaptive and len(residuals) >= 3:
            # === Adaptive z-score relative scoring ===
            vals = list(residuals.values())
            mu = sum(vals) / len(vals)
            sigma = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 + 1e-10

            for nid in neighbor_ids:
                z_score = (residuals[nid] - mu) / sigma

                # Hard exclusion for extreme outliers
                if z_score > self._prt_exclusion_z:
                    raw_trust = 0.0
                    self._prt_suspicion_count[nid] = self._prt_suspicion_count.get(nid, 0) + 1
                    logging.info(
                        f"PRT ADAPTIVE | neighbor={nid} | residual={residuals[nid]:.6f} | "
                        f"z_score={z_score:.3f} > {self._prt_exclusion_z} | EXCLUDED | "
                        f"suspicion_count={self._prt_suspicion_count[nid]}"
                    )
                else:
                    # Gaussian falloff for z > 0 (worse than average = penalized)
                    raw_trust = math.exp(-max(z_score, 0) ** 2 / 2.0)
                    logging.info(
                        f"PRT ADAPTIVE | neighbor={nid} | residual={residuals[nid]:.6f} | "
                        f"z_score={z_score:.3f} | raw_trust={raw_trust:.4f}"
                    )

                # Directional consistency check
                if self._prt_direction_check and raw_trust > 0 and len(self._prt_pending_deltas) >= 2:
                    raw_trust = self._apply_direction_check(nid, raw_trust)

                # Suspicion memory: persistent penalty for repeat offenders
                suspicion = self._prt_suspicion_count.get(nid, 0)
                if suspicion > 0 and raw_trust > 0:
                    persistence_factor = max(0.7 ** suspicion, self._prt_min_trust)
                    raw_trust *= persistence_factor
                    logging.info(f"PRT SUSPICION | neighbor={nid} | suspicion_count={suspicion} | factor={persistence_factor:.4f}")

                # EMA smoothing
                old_trust = self._prt_trust_scores.get(nid, 1.0)
                smoothed = (1 - self._prt_trust_smoothing) * old_trust + self._prt_trust_smoothing * raw_trust
                self._prt_trust_scores[nid] = smoothed
                logging.info(f"PRT FINAL | neighbor={nid} | smoothed_trust={smoothed:.4f}")

                # Reduce suspicion slowly when node behaves well
                if z_score < 0.5 and nid in self._prt_suspicion_count and self._prt_suspicion_count[nid] > 0:
                    self._prt_suspicion_count[nid] = max(0, self._prt_suspicion_count[nid] - 1)

        else:
            # === Non-adaptive: fixed-scale absolute scoring (original behavior) ===
            for nid in neighbor_ids:
                normalized_residual = residuals[nid]
                if self._prt_score_type == "exponential":
                    raw_trust = math.exp(-self._prt_scale * normalized_residual)
                else:
                    raw_trust = 1.0 / (1.0 + self._prt_scale * normalized_residual)
                raw_trust = max(raw_trust, self._prt_min_trust)

                old_trust = self._prt_trust_scores.get(nid, 1.0)
                smoothed = (1 - self._prt_trust_smoothing) * old_trust + self._prt_trust_smoothing * raw_trust
                self._prt_trust_scores[nid] = smoothed
                logging.info(
                    f"PRT | neighbor={nid} | residual={normalized_residual:.6f} | "
                    f"raw_trust={raw_trust:.4f} | smoothed_trust={smoothed:.4f}"
                )

        # Clear pending state for next round
        self._prt_pending_residuals.clear()
        self._prt_pending_deltas.clear()

    def _apply_direction_check(self, neighbor_id: str, raw_trust: float) -> float:
        """
        Penalize neighbors whose model update direction opposes the consensus.
        Catches Dissensus-style attacks that reverse gossip progress.
        """
        import torch

        if neighbor_id not in self._prt_pending_deltas:
            return raw_trust

        neighbor_delta = self._prt_pending_deltas[neighbor_id]

        # Compute average delta across all neighbors (consensus direction)
        other_deltas = [d for nid, d in self._prt_pending_deltas.items() if nid != neighbor_id]
        if not other_deltas:
            return raw_trust

        avg_delta = torch.stack(other_deltas).mean(dim=0)

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            neighbor_delta.unsqueeze(0), avg_delta.unsqueeze(0)
        ).item()

        if cos_sim < 0:
            # Opposing consensus direction — apply penalty
            # cos_sim in [-1, 0]: more negative = stronger penalty
            penalty = self._prt_direction_penalty + (1 - self._prt_direction_penalty) * (1 + cos_sim)
            raw_trust *= penalty
            self._prt_suspicion_count[neighbor_id] = self._prt_suspicion_count.get(neighbor_id, 0) + 1
            logging.info(
                f"PRT DIRECTION | neighbor={neighbor_id} | cos_sim={cos_sim:.4f} | "
                f"penalty={penalty:.4f} | adjusted_trust={raw_trust:.4f}"
            )

        return raw_trust

    def get_prt_trust_scores(self) -> dict[str, float]:
        """Return a copy of current PRT trust scores for all neighbors."""
        return self._prt_trust_scores.copy()
