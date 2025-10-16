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
        self._ema_deltas: dict[str, OrderedDict] = {}  # neighbor_id -> EMA of deltaW
        self._pseudo_agg_enabled = False
        self._ema_alpha = 0.25  # Default EMA weight for new delta (will be overridden by config)

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

    def enable_pseudo_aggregation(self, ema_alpha: float = 0.25):
        """
        Enable pseudo aggregation with specified EMA alpha.

        Args:
            ema_alpha (float): Weight for new delta in EMA calculation (default: 0.25).
                             EMA_new = (1 - ema_alpha) * EMA_old + ema_alpha * delta
        """
        self._pseudo_agg_enabled = True
        self._ema_alpha = ema_alpha
        logging.info(f"Pseudo Aggregation enabled with EMA alpha={ema_alpha}")

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

        for key in new_model_state_dict.keys():
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

    def predict_neighbor_model(self, neighbor_id: str) -> Optional[OrderedDict]:
        """
        Predict neighbor's model using oldW + EMA for pseudo aggregation.

        Prediction formula: predictedW = oldW + EMA

        Args:
            neighbor_id (str): Identifier of the neighbor to predict.

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

        # Predict: oldW + EMA
        ema = self._ema_deltas[neighbor_id]
        predicted = OrderedDict()

        for key in old_model.keys():
            if key in ema:
                predicted[key] = old_model[key] + ema[key] * 0.5
            else:
                # Parameter exists in old model but not in EMA, keep old value
                predicted[key] = old_model[key]

        logging.debug(f"Predicted model for neighbor {neighbor_id}")
        return predicted

    async def get_predicted_models(self, federation_nodes: set) -> dict:
        """
        Get predicted models for all federation nodes for pseudo aggregation.

        Args:
            federation_nodes (set): Set of neighbor node IDs.

        Returns:
            dict: Dictionary mapping neighbor_id to (predicted_model, weight) tuples.
                 Only includes neighbors with available predictions.
        """
        predicted_models = {}
        skipped_neighbors = []

        for neighbor_id in federation_nodes:
            predicted_model = self.predict_neighbor_model(neighbor_id)
            if predicted_model is not None:
                predicted_models[neighbor_id] = (predicted_model, self._old_weight[neighbor_id] if neighbor_id in self._old_weight else 100.0)
            else:
                skipped_neighbors.append(neighbor_id)

        if skipped_neighbors:
            logging.info(
                f"Pseudo Aggregation: Predicted {len(predicted_models)} models, "
                f"skipped {len(skipped_neighbors)} neighbors without history: {skipped_neighbors}"
            )
        else:
            logging.info(f"Pseudo Aggregation: Predicted {len(predicted_models)} models for all neighbors")

        return predicted_models
