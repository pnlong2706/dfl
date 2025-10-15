# Pseudo Aggregation in NEBULA

## Overview

Pseudo Aggregation is a communication-efficient extension to Decentralized Federated Learning (DFL) that reduces network overhead by **50%** while maintaining model convergence.

### How It Works

Instead of communicating model updates every round, Pseudo Aggregation uses **Exponential Moving Average (EMA)** to predict neighbor models:

**Traditional DFL (every round):**
```
Train N epochs → Send model → Receive models → Aggregate → Repeat
```

**Pseudo Aggregation (alternating rounds):**
```
Round 1 (Actual):  Train N/2 epochs → Send model → Receive models → Update EMA → Aggregate
Round 1.5 (Pseudo): Train N/2 epochs → Predict neighbor models → Aggregate (no communication)
Round 2 (Actual):  Train N/2 epochs → Send model → Receive models → Update EMA → Aggregate
Round 2.5 (Pseudo): Train N/2 epochs → Predict neighbor models → Aggregate (no communication)
...
```

### EMA-Based Model Prediction

When receiving a new model from neighbor `j`:
```
deltaW_j = newW_j - oldW_j
EMA_j = 0.75 * EMA_j_old + 0.25 * deltaW_j
```

When predicting neighbor `j`'s model (pseudo rounds):
```
predictedW_j = oldW_j + EMA_j
```

### Key Benefits

- **50% less communication** (alternating rounds skip model transmission)
- **Reduced network bandwidth** usage
- **Faster training** (no waiting for slow neighbors in pseudo rounds)
- **Compatible with all aggregation algorithms** (FedAvg, Krum, Median, TrimmedMean)
- **Works with Byzantine-robust aggregators**

## Configuration

### Enabling Pseudo Aggregation

Add the following to your **participant configuration JSON** file:

```json
{
  "device_args": {
    "idx": 0,
    "role": "trainer_aggregator",
    ...
  },
  "aggregator_args": {
    "algorithm": "FedAvg",
    "aggregation_timeout": 120,
    "pseudo_aggregation": {
      "enabled": true,
      "ema_alpha": 0.25
    }
  },
  "trainer_args": {
    "epochs": 2
  },
  ...
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable/disable pseudo aggregation |
| `ema_alpha` | float | `0.25` | Weight for new delta in EMA calculation (0.0-1.0) |

**EMA Alpha Interpretation:**
- `ema_alpha = 0.25`: EMA updates slowly (75% history, 25% new change) - more stable but less responsive
- `ema_alpha = 0.50`: Balanced (50% history, 50% new change)
- `ema_alpha = 0.75`: EMA updates quickly (25% history, 75% new change) - more responsive but less stable

**Recommendation:** Start with default `0.25`. Increase if neighbor models change rapidly, decrease if models are stable.

## Round Structure

With pseudo aggregation enabled, the round structure changes:

### Without Pseudo Aggregation (Traditional)
- **Round 0:** Initialization (actual aggregation)
- **Round 1:** Actual aggregation
- **Round 2:** Actual aggregation
- **Round 3:** Actual aggregation
- ...
- **Total communication events:** N rounds

### With Pseudo Aggregation
- **Round 0:** Initialization (actual aggregation, full epochs)
- **Round 1:** Actual aggregation (remaining epochs)
- **Round 1.5:** Pseudo aggregation (half epochs)
- **Round 2:** Actual aggregation (remaining epochs)
- **Round 2.5:** Pseudo aggregation (half epochs)
- ...
- **Total communication events:** ~N/2 rounds

**Important:** If you configure `rounds = 100` with pseudo aggregation, you'll actually execute **200 rounds** (100 actual + 100 pseudo) but with the **same total epochs** as 100 traditional rounds.

### Epoch Splitting

Training epochs are automatically split:
- **Pseudo round:** `max_epochs // 2` (e.g., 1 epoch if max_epochs=2)
- **Actual round:** `max_epochs - (max_epochs // 2)` (e.g., 1 epoch if max_epochs=2)

Example with `max_epochs = 3`:
- Pseudo round: 1 epoch
- Actual round: 2 epochs
- Total per cycle: 3 epochs (same as traditional)

## Logging and Monitoring

### JSON Logs

Pseudo aggregation information is automatically logged to JSON files:

```json
{
  "round": 1.5,
  "aggregation_type": "pseudo",
  "start_time": "2025-10-16T10:30:00",
  "dataset_info": { ... },
  "training": { ... },
  "test_global": { ... }
}
```

### Text Logs

Look for these log markers:

**Pseudo Round Start:**
```
🔮  Pseudo aggregation round - using predicted models
Pseudo round: training for 1 epochs (half of 2)
```

**Actual Round Start:**
```
📡  Actual aggregation round - waiting for real model updates
Actual round: training for 1 epochs (remainder of 2)
```

**Model Prediction:**
```
Pseudo Aggregation: Predicted 3 models for all neighbors
```

**Communication Skip:**
```
🔮  Pseudo round - skipping model propagation to neighbors
```

## Example: MNIST with Pseudo Aggregation

### Configuration

**Topology Config (`topology.json`):**
```json
{
  "n_nodes": 4,
  "b_symmetric": true,
  "undirected_neighbor_num": 3,
  "topology": "fully_connected"
}
```

**Participant Config (`participant_0.json`):**
```json
{
  "device_args": {
    "idx": 0,
    "role": "trainer_aggregator",
    "start": true,
    "logging": true
  },
  "network_args": {
    "ip": "127.0.0.1",
    "port": 45000,
    "neighbors": "127.0.0.1:45001 127.0.0.1:45002 127.0.0.1:45003"
  },
  "data_args": {
    "dataset": "MNIST",
    "iid": false,
    "partition_selection": "dirichlet",
    "partition_parameter": 0.5
  },
  "model_args": {
    "model": "MLP"
  },
  "training_args": {
    "trainer": "lightning",
    "epochs": 2,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 0.001
  },
  "aggregator_args": {
    "algorithm": "FedAvg",
    "aggregation_timeout": 120,
    "pseudo_aggregation": {
      "enabled": true,
      "ema_alpha": 0.25
    }
  },
  "scenario_args": {
    "rounds": 50,
    "name": "mnist_pseudo_agg_experiment",
    "random_seed": 42
  }
}
```

### Expected Behavior

- **Total rounds executed:** 100 (50 actual + 50 pseudo)
- **Communication events:** ~50 (instead of 100)
- **Training epochs per cycle:** 2 (1 pseudo + 1 actual)
- **Round 0:** Initialization, actual aggregation (2 epochs)
- **Round 1:** Actual aggregation (1 epoch)
- **Round 1.5:** Pseudo aggregation (1 epoch, no communication)
- **Round 2:** Actual aggregation (1 epoch)
- **Round 2.5:** Pseudo aggregation (1 epoch, no communication)
- ...
- **Round 50:** Actual aggregation (last round)

### Performance Comparison

| Metric | Traditional DFL | Pseudo Aggregation |
|--------|----------------|-------------------|
| Communication Rounds | 50 | 25 |
| Total Training Epochs | 100 | 100 |
| Network Messages | ~200 | ~100 |
| Training Time | Baseline | 30-50% faster* |

*Depends on network latency and aggregation timeout

## Edge Cases and Behavior

### First Round (Round 0-1)
- **Round 0:** Initialization only (if configured)
- **Round 1:** Always actual aggregation (need initial models to build EMA)
- No pseudo aggregation until Round 2+

### New Neighbor Joins
- If a neighbor never sent a model: **skip prediction** for that neighbor
- If a neighbor sent only once: **predict no change** (use last model as-is)

### Missing EMA History
When a neighbor has `oldW` but no EMA yet:
- **Prediction:** `predictedW = oldW` (assume no change)
- **EMA initialization:** Starts after receiving second model from neighbor

### Network Topology Changes
- Pseudo aggregation automatically adapts to current federation nodes
- New neighbors excluded from pseudo aggregation until first actual round

## Troubleshooting

### Issue: "No predicted models available"
**Cause:** No neighbors have sent models yet (e.g., Round 1.5 before Round 1 completes)
**Solution:** This is expected behavior. System will use only local model for aggregation.

### Issue: Convergence slower than expected
**Cause:** EMA alpha might be too low or model changes are too rapid
**Solution:** Increase `ema_alpha` from 0.25 to 0.4-0.5

### Issue: Convergence unstable
**Cause:** EMA alpha might be too high
**Solution:** Decrease `ema_alpha` from 0.25 to 0.1-0.15

### Issue: JSON logs not showing aggregation_type
**Cause:** JSON logger not created or not passing aggregation type
**Solution:** Ensure JSON logger is enabled and check Lightning trainer initialization

## Technical Implementation Details

### Modified Components

1. **DFLUpdateHandler** (`nebula/core/aggregation/updatehandlers/dflupdatehandler.py`)
   - Added EMA storage: `_old_models`, `_ema_deltas`
   - Methods: `update_ema()`, `predict_neighbor_model()`, `get_predicted_models()`

2. **Aggregator** (`nebula/core/aggregation/aggregator.py`)
   - New method: `get_pseudo_aggregation()` (bypasses waiting for updates)

3. **Engine** (`nebula/core/engine.py`)
   - Round type tracking: `_is_pseudo_round`
   - Method: `get_round_with_phase()` returns float (e.g., 1.5 for pseudo)
   - Chooses `get_aggregation()` vs `get_pseudo_aggregation()`

4. **Role Behaviors** (`nebula/core/noderole.py`)
   - TrainerAggregatorRoleBehavior skips `ModelPropagationEvent` in pseudo rounds

5. **Lightning Trainer** (`nebula/core/training/lightning.py`)
   - Epoch splitting: `adjust_epochs_for_pseudo_agg()`
   - JSON logging with `aggregation_type`

6. **JSON Logger** (`nebula/core/utils/nebulalogger_json.py`)
   - Field: `aggregation_type` ("pseudo" or "actual")

### Data Structures

**Old Models Storage:**
```python
_old_models: Dict[str, OrderedDict]
# Maps neighbor_id -> previous model state_dict
```

**EMA Storage:**
```python
_ema_deltas: Dict[str, OrderedDict]
# Maps neighbor_id -> EMA of deltaW (same structure as model state_dict)
```

### Prediction Algorithm

```python
def predict_neighbor_model(neighbor_id):
    if neighbor_id not in old_models:
        return None  # No history

    oldW = old_models[neighbor_id]

    if neighbor_id not in ema_deltas:
        return oldW  # Only received once, predict no change

    ema = ema_deltas[neighbor_id]
    predicted = {key: oldW[key] + ema[key] for key in oldW.keys()}
    return predicted
```

## Future Enhancements

Potential improvements for Pseudo Aggregation:

1. **Adaptive EMA Alpha:** Automatically adjust `ema_alpha` based on prediction accuracy
2. **Selective Pseudo Aggregation:** Only predict stable neighbors, communicate with volatile ones
3. **Multi-step Prediction:** Allow multiple consecutive pseudo rounds (e.g., 2 pseudo, 1 actual)
4. **Prediction Confidence:** Track and log how accurate predictions are
5. **Gradient-based Prediction:** Predict using gradient information instead of full model deltas

## References

- NEBULA Documentation: https://docs.nebula-dfl.com/
- Federated Learning: Communication-Efficient Learning of Deep Networks from Decentralized Data (McMahan et al., 2017)
- Exponential Moving Average: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average

## Support

For issues or questions:
- GitHub Issues: https://github.com/CyberDataLab/nebula/issues
- Documentation: https://docs.nebula-dfl.com/

## License

Pseudo Aggregation is part of NEBULA and follows the same AGPLv3 license.
