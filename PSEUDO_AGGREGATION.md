# Pseudo Aggregation in NEBULA

## Overview

Pseudo Aggregation reduces communication overhead in Decentralized Federated Learning (DFL) by **50%** using Exponential Moving Average (EMA) to predict neighbor models instead of waiting for transmission.

### How It Works

**Traditional DFL (every round):**
```
Train N epochs → Send model → Receive models → Aggregate → Repeat
```

**Pseudo Aggregation (alternating rounds):**
```
Round 1 (Actual):   Train N/2 epochs → Send/Receive → Update EMA → Aggregate
Round 1.5 (Pseudo): Train N/2 epochs → Predict models → Aggregate (no communication)
Round 2 (Actual):   Train N/2 epochs → Send/Receive → Update EMA → Aggregate
Round 2.5 (Pseudo): Train N/2 epochs → Predict models → Aggregate (no communication)
```

**Model Prediction:**
```python
# When receiving actual model from neighbor j:
deltaW_j = newW_j - oldW_j
EMA_j = (1 - alpha) * EMA_j_old + alpha * deltaW_j

# When predicting in pseudo rounds:
predictedW_j = oldW_j + EMA_j
```

### Key Benefits

- **50% less communication** (every other round skips transmission)
- **Faster training** (no waiting in pseudo rounds)
- **Compatible with all aggregators** (FedAvg, Krum, Median, TrimmedMean)
- **Same total training epochs** as traditional DFL

## Configuration

### Frontend (Recommended)

Enable via the deployment UI:
1. Check "Enable Pseudo Aggregation" in the aggregation settings
2. Adjust EMA Alpha (default: 0.25)
3. Deploy scenario

### Manual Configuration

Add to participant JSON:

```json
{
  "aggregator_args": {
    "algorithm": "FedAvg",
    "aggregation_timeout": 180,
    "pseudo_aggregation": {
      "enabled": true,
      "ema_alpha": 0.25
    }
  },
  "training_args": {
    "epochs": 2
  },
  "scenario_args": {
    "rounds": 20
  }
}
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Enable pseudo aggregation |
| `ema_alpha` | `0.25` | Weight for new delta (0.0-1.0). Lower = more stable, Higher = more responsive |

**Recommended `ema_alpha` values:**
- `0.1-0.2`: Stable predictions, slow adaptation
- `0.25` (default): Balanced
- `0.4-0.5`: Fast adaptation, less stable

## ⚠️ IMPORTANT: Round Numbering

**When pseudo aggregation is enabled, the internal round counter doubles to accommodate both actual and pseudo rounds.**

### Example Behavior

If you configure **20 rounds** with pseudo aggregation enabled:

- **Physical rounds executed:** 40 (20 actual + 20 pseudo)
- **Logical rounds displayed:** 0, 1.0, 1.5, 2.0, 2.5, ..., 19.5, 20.0
- **Communication events:** ~20 (only during actual rounds)
- **Total training epochs:** Same as 20 traditional rounds

**Why this matters:**
- Logs and metrics will show 40 physical rounds
- JSON logs will have twice as many entries
- Training takes the same total epochs but over more rounds
- Network traffic is still reduced by 50%

**Log interpretation:**
```
Round 10.0 (Actual) | Physical: 19/39 | Logical: 10/20
```
- **Logical round 10** = what you configured
- **Physical round 19** = internal counter (19 actual + 19 pseudo so far)

### Round Structure Details

**Without Pseudo Aggregation (Traditional):**
- Configure 20 rounds → Execute 20 rounds → 20 communication events

**With Pseudo Aggregation:**
- Configure 20 rounds → Execute 40 physical rounds → 20 communication events
- Round 0: Initialization (actual)
- Round 1: Actual (with communication)
- Round 2: Pseudo (no communication)
- Round 3: Actual (with communication)
- Round 4: Pseudo (no communication)
- ...
- Round 39: Actual (last round)

## Epoch Splitting

Training epochs are automatically divided:
- **Pseudo round:** `epochs // 2` (e.g., 1 if epochs=2)
- **Actual round:** `epochs - (epochs // 2)` (e.g., 1 if epochs=2)

Example with `epochs = 3`:
- Pseudo: 1 epoch
- Actual: 2 epochs
- **Total per cycle: 3 epochs** (same as traditional)

## Logging

### Console Logs

**Pseudo round indicators:**
```
🔮 Pseudo aggregation round - using predicted models
Pseudo round: training for 1 epochs (half of 2)
🔮 Pseudo round - skipping model propagation to neighbors
```

**Actual round indicators:**
```
📡 Actual aggregation round - waiting for real model updates
Actual round: training for 1 epochs (remainder of 2)
```

### JSON Logs

Each round includes `aggregation_type`:

```json
{
  "round": 1.5,
  "aggregation_type": "pseudo",
  "training": { "loss": 0.45, "accuracy": 0.89 },
  "test_global": { "loss": 0.52, "accuracy": 0.85 }
}
```

## Troubleshooting

### "No predicted models available"
**Cause:** Round 1 hasn't completed yet (no neighbor models received)
**Solution:** Expected behavior. System uses only local model.

### Slower convergence
**Cause:** `ema_alpha` too low
**Solution:** Increase to 0.4-0.5

### Unstable convergence
**Cause:** `ema_alpha` too high
**Solution:** Decrease to 0.1-0.15

### Aggregation timeout errors
**Cause:** Timeout too short for actual rounds
**Solution:** Increase `aggregation_timeout` to 180+ seconds in frontend

## Technical Details

### Modified Components

1. **DFLUpdateHandler** - EMA storage and prediction
2. **Aggregator** - `get_pseudo_aggregation()` bypasses waiting
3. **Engine** - Round type tracking and conditional aggregation
4. **RoleBehavior** - Skips `ModelPropagationEvent` in pseudo rounds
5. **Lightning Trainer** - Epoch splitting
6. **JSON Logger** - Tracks `aggregation_type`

### Data Structures

```python
_old_models: Dict[str, OrderedDict]        # neighbor_id -> last received model
_old_model_rounds: Dict[str, int]          # neighbor_id -> round when model was received
_old_weight: Dict[str, float]              # neighbor_id -> original model weight
_ema_deltas: Dict[str, OrderedDict]        # neighbor_id -> EMA of deltaW
_max_round_staleness: int = 5              # Maximum allowed round difference
```

### Prediction Logic with Staleness Handling

```python
def predict_neighbor_model(neighbor_id):
    if neighbor_id not in _old_models:
        return None  # No history yet

    if neighbor_id not in _ema_deltas:
        return _old_models[neighbor_id]  # Only received once, predict no change

    predicted = _old_models[neighbor_id] + _ema_deltas[neighbor_id]
    return predicted

def get_predicted_models(federation_nodes, current_round):
    predicted_models = {}

    for neighbor_id in federation_nodes:
        predicted_model = predict_neighbor_model(neighbor_id)
        if predicted_model is None:
            continue

        # Check staleness
        model_round = _old_model_rounds[neighbor_id]
        round_diff = current_round - model_round

        # Exclude if too stale (older than 5 rounds)
        if round_diff > _max_round_staleness:
            log(f"Excluding {neighbor_id}: {round_diff} rounds old")
            continue

        # Adjust weight based on staleness
        base_weight = _old_weight[neighbor_id]
        adjusted_weight = base_weight / max(1, round_diff + 1)

        predicted_models[neighbor_id] = (predicted_model, adjusted_weight)

    return predicted_models
```

**Staleness handling:**
- **Models older than 5 rounds:** Excluded from pseudo aggregation entirely
- **Recent models (0-5 rounds old):** Weight adjusted by `base_weight / (round_diff + 1)`
  - Same round: weight unchanged (diff=0, penalty=1)
  - 1 round old: weight × 0.5 (diff=1, penalty=2)
  - 2 rounds old: weight × 0.33 (diff=2, penalty=3)
  - 5 rounds old: weight × 0.167 (diff=5, penalty=6)

## Performance Verification

To confirm pseudo aggregation works correctly:

1. **Check logs for alternating round types:**
   - "Actual Aggregation" and "Pseudo Aggregation" should alternate
   - "🔮 Pseudo round - skipping model propagation" every other round

2. **Verify communication reduction:**
   - Count "Sending model to" logs
   - Should be ~50% of total rounds

3. **Monitor physical vs logical rounds:**
   - Physical rounds = 2 × configured rounds
   - Logical rounds = configured rounds

4. **Confirm no timeout in pseudo rounds:**
   - Pseudo rounds should NOT show "Aggregation timeout: X starts..."
   - Only actual rounds use timeout

## Example Configuration

**Scenario:** 4 nodes, fully connected, MNIST, 20 logical rounds

```json
{
  "aggregator_args": {
    "algorithm": "FedAvg",
    "aggregation_timeout": 180,
    "pseudo_aggregation": {
      "enabled": true,
      "ema_alpha": 0.25
    }
  },
  "training_args": {
    "epochs": 2,
    "batch_size": 32
  },
  "scenario_args": {
    "rounds": 20
  }
}
```

**Expected outcome:**
- 40 physical rounds executed
- 20 actual aggregations (with communication)
- 20 pseudo aggregations (no communication)
- ~50% network traffic reduction
- Same total epochs as traditional 20-round training

## Support

- **GitHub Issues:** https://github.com/CyberDataLab/nebula/issues
- **Documentation:** https://docs.nebula-dfl.com/

## License

Part of NEBULA platform under AGPLv3 license.
