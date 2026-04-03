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
            updates[sr] = (updt.model, updt.weight)

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
    ):
        """
        Enable PRT (Prediction-Residual Trust) for Byzantine defense.

        PRT compares actual received models to EMA predictions. Large residuals
        reduce a neighbor's trust score, which scales down its aggregation weight.

        Requires Pseudo Aggregation to be enabled (for EMA predictions).

        Args:
            score_type: Trust function type ("exponential" or "inverse").
            scale: Scaling factor for residual in trust function.
            min_trust: Floor value for trust score (prevents zero weight).
            trust_smoothing: EMA smoothing for trust updates (0=keep old, 1=use new).
            warmup_rounds: Number of rounds before PRT starts (trust=1.0 during warmup).
            apply_to_pseudo: Whether to apply trust scores to pseudo round weights.
        """
        self._prt_enabled = True
        self._prt_score_type = score_type
        self._prt_scale = scale
        self._prt_min_trust = min_trust
        self._prt_trust_smoothing = trust_smoothing
        self._prt_warmup_rounds = warmup_rounds
        self._prt_apply_to_pseudo = apply_to_pseudo
        logging.info(
            f"PRT enabled: score_type={score_type}, scale={scale}, min_trust={min_trust}, "
            f"trust_smoothing={trust_smoothing}, warmup_rounds={warmup_rounds}, apply_to_pseudo={apply_to_pseudo}"
        )

    def compute_prt_trust(self, neighbor_id: str, actual_model: OrderedDict, current_round: int) -> float:
        """
        Compute trust score from residual between actual and predicted model.

        Called on real communication rounds when an actual model arrives.
        Compares actual model to EMA-predicted model, converts residual magnitude
        to a trust score, and applies EMA smoothing across rounds.

        Args:
            neighbor_id: Identifier of the neighbor node.
            actual_model: Actually received model state dict.
            current_round: Current round number.

        Returns:
            float: Smoothed trust score in [min_trust, 1.0].
        """
        if not self._prt_enabled or not self._pseudo_agg_enabled:
            return 1.0

        if current_round < self._prt_warmup_rounds:
            return 1.0

        predicted_model = self.predict_neighbor_model(neighbor_id, current_round)
        if predicted_model is None:
            return 1.0

        # Compute normalized residual (RMS per parameter)
        skip_keys = {'running_mean', 'running_var', 'num_batches_tracked'}
        residual_sq_sum = 0.0
        param_count = 0
        for key in actual_model.keys():
            if any(sk in key for sk in skip_keys):
                continue
            if key in predicted_model:
                diff = actual_model[key].float() - predicted_model[key].float()
                residual_sq_sum += diff.pow(2).sum().item()
                param_count += diff.numel()

        if param_count == 0:
            return 1.0

        normalized_residual = (residual_sq_sum / param_count) ** 0.5

        # Score function
        import math
        if self._prt_score_type == "exponential":
            raw_trust = math.exp(-self._prt_scale * normalized_residual)
        else:
            raw_trust = 1.0 / (1.0 + self._prt_scale * normalized_residual)

        raw_trust = max(raw_trust, self._prt_min_trust)

        # EMA smoothing
        old_trust = self._prt_trust_scores.get(neighbor_id, 1.0)
        smoothed_trust = (1 - self._prt_trust_smoothing) * old_trust + self._prt_trust_smoothing * raw_trust
        self._prt_trust_scores[neighbor_id] = smoothed_trust

        logging.info(
            f"PRT | neighbor={neighbor_id} | residual={normalized_residual:.6f} | "
            f"raw_trust={raw_trust:.4f} | smoothed_trust={smoothed_trust:.4f}"
        )
        return smoothed_trust

    def get_prt_trust_scores(self) -> dict[str, float]:
        """Return a copy of current PRT trust scores for all neighbors."""
        return self._prt_trust_scores.copy()
