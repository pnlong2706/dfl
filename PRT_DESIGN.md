# PRT (Prediction-Residual Trust) — Design Document

## Overview

PRT is a Byzantine-robustness module for Decentralized Federated Learning (DFL).
It leverages Pseudo Aggregation's EMA predictions to detect and exclude malicious
neighbors by comparing predicted vs actual models.

**Key insight**: Instead of comparing a node to the group (which attackers can corrupt),
PRT compares each node to its own predicted trajectory. Large deviations indicate
abnormal behavior.

## Architecture

PRT sits between model reception and aggregation:

```
Neighbor model arrives
    → Store PRT residual (actual vs EMA prediction)
    → Update EMA
    → [All models arrived]
    → Finalize PRT trust scores (MAD z-score across all residuals)
    → Apply trust to aggregation weights
    → Aggregation runs with adjusted weights
```

## Algorithm

### Phase 1: Residual Computation (per model arrival)

When a real neighbor model arrives in an actual communication round:

```
predicted_model = old_model + EMA_delta * scaling_factor
residual = RMS(actual_model - predicted_model)  # skip BatchNorm keys
```

The residual and model delta (actual - old) are stored for batch processing.

### Phase 2: MAD-Based Trust Scoring (before aggregation)

Once all models arrive, trust scores are computed using all residuals together:

**Step 1: Iterative MAD z-scores (robust to 50% contamination)**

```python
for iteration in range(3):  # up to 3 passes
    median = sorted(residuals)[n // 2]
    abs_devs = sorted(|r - median| for r in residuals)
    MAD = abs_devs[n // 2] * 1.4826  # consistency constant for normal dist

    modified_z = (residual - median) / MAD

    # Exclude nodes with z > exclusion_threshold (default 2.5)
    if modified_z > exclusion_z:
        remove from active set, mark as excluded

    if no new exclusions: break
    # Recompute with cleaned set
```

**Why MAD, not mean/std?** Mean and std have 0% breakdown point — a single outlier
can corrupt them. At 40% attacker rate, mean/std make attackers look "normal" and
honest nodes look like outliers. MAD uses medians with 50% breakdown point, so up
to 50% contamination cannot corrupt the statistics.

**Why iterative?** First pass may leave borderline attackers. Removing the worst and
recomputing gives cleaner statistics for the next pass.

**Step 2: Gaussian falloff for non-excluded nodes**

```
raw_trust = exp(-max(modified_z, 0)^2 / 2)
```

Nodes better than median (z < 0) get full trust. Nodes worse than median get
exponentially decreasing trust.

**Step 3: Directional consistency check**

```python
consensus_delta = median(all neighbor deltas)  # coordinate-wise median
cos_sim = cosine_similarity(neighbor_delta, consensus_delta)
if cos_sim < 0:  # opposing consensus
    trust *= direction_penalty  # default 0.3
```

Uses **median** consensus (not mean) to resist 40%+ contamination.
Catches Dissensus-style attacks that reverse gossip progress.

**Step 4: Suspicion memory**

```python
if excluded or opposing direction:
    suspicion_count[neighbor] += 1

trust *= max(0.7 ^ suspicion_count, min_trust)

# Slow recovery: decrement count by 1 when z < 0.5
if well_behaved:
    suspicion_count[neighbor] -= 1
```

Persistent penalty for repeat offenders. A node that attacks once then behaves
normally still carries a penalty that decays slowly.

**Step 5: EMA smoothing across rounds**

```
smoothed_trust = (1 - smoothing) * old_trust + smoothing * raw_trust
```

### Phase 3: Trust Application

**In actual rounds**: multiply aggregation weight by trust score.
Trust = 0 means hard exclusion (node skipped entirely).

**In pseudo rounds**: optionally apply latest trust to predicted model weights.

### Phase 4: Trust-Gated EMA Updates

```python
if trust[neighbor] < 0.3:
    skip EMA update for this neighbor  # don't corrupt predictions
else:
    effective_alpha = ema_alpha * trust  # scale learning rate by trust
    EMA_new = (1 - effective_alpha) * EMA_old + effective_alpha * delta
```

Prevents identified attackers from corrupting future EMA predictions,
which would undermine the entire detection pipeline.

## Configuration

```json
"prt": {
    "enabled": true,
    "score_type": "exponential",     // trust function (non-adaptive fallback)
    "scale": 1.0,                    // scaling factor (non-adaptive fallback)
    "min_trust": 0.1,                // floor for trust score
    "trust_smoothing": 0.5,          // EMA smoothing across rounds
    "warmup_rounds": 2,              // rounds before PRT activates
    "apply_to_pseudo": true,         // apply trust to pseudo round weights
    "adaptive": true,                // use MAD z-score (recommended)
    "exclusion_z": 2.5,              // z-score threshold for hard exclusion
    "direction_check": true,         // cosine similarity check
    "direction_penalty": 0.3         // penalty for opposing consensus
}
```

## Requirements

- **Pseudo Aggregation must be enabled** — PRT needs EMA predictions to compute residuals.
- Minimum 3 neighbors for adaptive mode (falls back to fixed-scale with fewer).

## Properties

| Property | Value |
|----------|-------|
| Breakdown point | 50% (from MAD) |
| Max tolerable attack rate | < 50% of neighbors |
| Warmup period | 2 rounds (configurable) |
| Detection latency | ~5-10 rounds for stable trust |
| Communication overhead | None (uses existing model exchange) |
| Computation overhead | O(n) per round (residual + z-score) |

## Comparison with Other Defenses

| Defense | Approach | Breakdown | Knows f? | Adaptive? |
|---------|----------|-----------|----------|-----------|
| **PRT** | Prediction residual + MAD | 50% | No | Yes |
| Krum | Geometric centrality | ~33% | Yes | No |
| TrimmedMean | Coordinate-wise trimming | Depends on trim % | Yes | No |
| Median | Coordinate-wise median | 50% | No | No |
| BALANCE | Distance from local model | Depends on gamma | No | Partially |
| RTC | Remove + clip | Depends on b | Yes | Partially |
| SCClip | Self-centered clipping | None (clips, doesn't exclude) | No | No |

PRT's key advantage: it **does not require knowing the number of attackers f**,
adapts automatically via MAD statistics, and provides both detection (exclusion)
and mitigation (weight reduction).

## Known Limitations

1. **Requires Pseudo Aggregation**: adds communication overhead (50% more rounds)
   and can cause mid-training accuracy dip on dense topologies.
2. **>50% attackers**: MAD breaks down; no single-node defense works.
3. **Sophisticated adaptive attacks**: an attacker that tracks the MAD threshold
   could stay just below it. Mitigated by directional check + suspicion memory.
4. **EMA calibration period**: first ~5 rounds have poor predictions, reducing
   detection accuracy. Warmup period helps but doesn't eliminate this.
