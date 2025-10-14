# NEBULA Enhanced Logging System

## Overview

NEBULA now includes an enhanced logging system that provides:
- **JSON-formatted metrics** for easy data access and analysis
- **Detailed training logs** with INFO, WARNING, and ERROR messages
- **Structured round-based logging** for federated learning experiments
- **Backward compatibility** with existing TensorBoard logs

## Log File Locations

After running an experiment, logs are organized as follows:

```
app/logs/
└── <experiment_name>/
    ├── participant_0.log                    # Main log file (INFO, WARNING, ERROR)
    ├── participant_0_debug.log              # Debug-level logs
    ├── participant_0_error.log              # Error and warning logs only
    ├── participant_0_training.log           # Training-specific logs
    ├── participant_0_metrics.json           # ⭐ NEW: Complete metrics in JSON
    ├── participant_0_rounds.json            # ⭐ NEW: Per-round summary
    ├── participant_0_summary.json           # ⭐ NEW: Experiment summary
    └── metrics/
        └── participant_0/
            └── events.out.tfevents.*        # TensorBoard event files
```

## New JSON Log Files

### 1. `participant_X_metrics.json`

Complete training history with all rounds and metrics.

**Structure:**
```json
{
  "participant_id": 0,
  "scenario": "experiment_name",
  "start_time": "2025-01-15T10:30:00.123456",
  "end_time": "2025-01-15T11:45:00.654321",
  "total_duration_seconds": 4500.5,
  "rounds": [
    {
      "round": 1,
      "start_time": "2025-01-15T10:30:00.123456",
      "end_time": "2025-01-15T10:45:00.789012",
      "duration_seconds": 900.67,
      "dataset_info": {
        "num_train_samples": 5000,
        "num_val_samples": 500,
        "num_test_local_samples": 1000,
        "num_test_global_samples": 10000
      },
      "training": {
        "epochs": [
          {
            "epoch": 0,
            "timestamp": "2025-01-15T10:30:15.123456",
            "metrics": {
              "Accuracy": 0.7523,
              "Precision": 0.7489,
              "Recall": 0.7612,
              "F1Score": 0.7550
            }
          },
          {
            "epoch": 1,
            "timestamp": "2025-01-15T10:32:30.456789",
            "metrics": {
              "Accuracy": 0.8123,
              "Precision": 0.8089,
              "Recall": 0.8212,
              "F1Score": 0.8150
            }
          }
        ]
      },
      "validation": {
        "timestamp": "2025-01-15T10:44:00.123456",
        "metrics": {
          "Accuracy": 0.7892,
          "Precision": 0.7856,
          "Recall": 0.7934,
          "F1Score": 0.7895
        }
      },
      "test_local": {
        "timestamp": "2025-01-15T10:44:30.123456",
        "metrics": {
          "Accuracy": 0.7756,
          "Precision": 0.7721,
          "Recall": 0.7834,
          "F1Score": 0.7777
        }
      },
      "test_global": {
        "timestamp": "2025-01-15T10:45:00.123456",
        "metrics": {
          "Accuracy": 0.7823,
          "Precision": 0.7789,
          "Recall": 0.7912,
          "F1Score": 0.7850
        }
      },
      "network": {
        "neighbors": ["participant_1", "participant_2"],
        "role": "trainer_aggregator"
      },
      "metadata": {}
    }
  ]
}
```

### 2. `participant_X_rounds.json`

Individual round summaries in a list format (easier to parse sequentially).

**Structure:**
```json
[
  {
    "round": 1,
    "start_time": "2025-01-15T10:30:00.123456",
    "end_time": "2025-01-15T10:45:00.789012",
    "duration_seconds": 900.67,
    "training": { ... },
    "validation": { ... },
    "test_local": { ... },
    "test_global": { ... }
  },
  {
    "round": 2,
    ...
  }
]
```

### 3. `participant_X_summary.json`

Quick experiment summary with final metrics.

**Structure:**
```json
{
  "participant_id": 0,
  "scenario": "experiment_name",
  "start_time": "2025-01-15T10:30:00.123456",
  "end_time": "2025-01-15T11:45:00.654321",
  "total_duration_seconds": 4500.5,
  "total_rounds": 10,
  "final_metrics": {
    "training": {
      "Accuracy": 0.9234,
      "Precision": 0.9201,
      "Recall": 0.9289,
      "F1Score": 0.9245
    },
    "validation": {
      "Accuracy": 0.8934,
      "Precision": 0.8901,
      "Recall": 0.8989,
      "F1Score": 0.8945
    },
    "test_local": {
      "Accuracy": 0.8823,
      "Precision": 0.8789,
      "Recall": 0.8912,
      "F1Score": 0.8850
    },
    "test_global": {
      "Accuracy": 0.8912,
      "Precision": 0.8878,
      "Recall": 0.9001,
      "F1Score": 0.8939
    }
  }
}
```

## Accessing Metrics Programmatically

### Python Example

```python
import json

# Load complete metrics
with open('app/logs/experiment_name/participant_0_metrics.json', 'r') as f:
    data = json.load(f)

# Get training metrics from round 5, epoch 2
round_5 = data['rounds'][4]  # 0-indexed
epoch_2_metrics = round_5['training']['epochs'][1]['metrics']
accuracy = epoch_2_metrics['Accuracy']

print(f"Round 5, Epoch 2 Accuracy: {accuracy:.4f}")

# Get final test accuracy
summary_file = 'app/logs/experiment_name/participant_0_summary.json'
with open(summary_file, 'r') as f:
    summary = json.load(f)

final_accuracy = summary['final_metrics']['test_local']['Accuracy']
print(f"Final Test Accuracy: {final_accuracy:.4f}")
```

### Pandas Example

```python
import json
import pandas as pd

# Load rounds data
with open('app/logs/experiment_name/participant_0_rounds.json', 'r') as f:
    rounds = json.load(f)

# Extract training metrics across all rounds
training_data = []
for round_data in rounds:
    for epoch in round_data['training']['epochs']:
        training_data.append({
            'round': round_data['round'],
            'epoch': epoch['epoch'],
            'timestamp': epoch['timestamp'],
            **epoch['metrics']
        })

df = pd.DataFrame(training_data)
print(df.head())

# Plot accuracy over time
import matplotlib.pyplot as plt
plt.plot(df['epoch'], df['Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Over Time')
plt.show()
```

## Enhanced Text Logging

The existing log files now contain more detailed information:

### `participant_X.log` (INFO level and above)

```
2025-01-15 10:30:00 | INFO | JSON Logger initialized for participant 0
2025-01-15 10:30:00 | INFO |   Metrics file: app/logs/.../participant_0_metrics.json
2025-01-15 10:30:00 | INFO |   Rounds file: app/logs/.../participant_0_rounds.json
2025-01-15 10:30:00 | INFO |   Summary file: app/logs/.../participant_0_summary.json
2025-01-15 10:30:00 | INFO | Update | current round = 1
2025-01-15 10:30:00 | INFO | [JSON Logger] Started logging round 1
2025-01-15 10:30:00 | INFO | [Dataset Info] Training samples: 5000, Validation samples: 500, Test (Local) samples: 1000, Test (Global) samples: 10000
2025-01-15 10:30:00 | INFO | Dataset sizes - Train: 5000, Val: 500, Test (Local): 1000, Test (Global): 10000
2025-01-15 10:30:00 | INFO | ========== [Training] Started ==========
2025-01-15 10:30:15 | INFO | [Training] Epoch 0 - Accuracy: 0.7523, Precision: 0.7489
2025-01-15 10:32:30 | INFO | [Training] Epoch 1 - Accuracy: 0.8123, Precision: 0.8089
2025-01-15 10:35:00 | INFO | ========== [Training] Done ==========
2025-01-15 10:44:00 | INFO | [Validation] Accuracy: 0.7892, Precision: 0.7856
2025-01-15 10:44:30 | INFO | [Test (Local)] Accuracy: 0.7756, Precision: 0.7721
2025-01-15 10:45:00 | INFO | [Test (Global)] Accuracy: 0.7823, Precision: 0.7789
2025-01-15 10:45:00 | INFO | [JSON Logger] Finished logging round 1 (duration: 900.67s)
```

### `participant_X_training.log` (Training details)

Contains PyTorch Lightning training output and detailed per-batch information.

## Metrics Available

For each phase (Training, Validation, Test), the following metrics are logged:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score (macro average)
- **Recall**: Recall score (macro average)
- **F1Score**: F1 score (macro average)

Additional information logged:
- **Dataset Info**: Number of training, validation, and test samples per round
- **Timestamps**: ISO 8601 format for all events
- **Duration**: Time taken for each round in seconds
- **Network info**: Neighbors, role, and other network metadata
- **Metadata**: Custom key-value pairs per round

### Dataset Information

At the beginning of each round, the following dataset sizes are logged:
- `num_train_samples`: Number of training samples
- `num_val_samples`: Number of validation samples (if validation set is used)
- `num_test_local_samples`: Number of local test samples
- `num_test_global_samples`: Number of global test samples

This information is useful for:
- Understanding data distribution across participants
- Verifying correct data partitioning
- Analyzing the relationship between dataset size and model performance
- Debugging data loading issues

## Backward Compatibility

- **TensorBoard logs** continue to work as before
- **Existing log files** (`participant_X.log`, `_debug.log`, `_error.log`) are unchanged
- **No configuration required**: JSON logging is automatically enabled

## Tips for Analysis

1. **Quick Summary**: Check `participant_X_summary.json` for final results
2. **Round Comparison**: Use `participant_X_rounds.json` to compare rounds
3. **Detailed Analysis**: Use `participant_X_metrics.json` for complete history
4. **Time Series**: Extract epochs with timestamps for temporal analysis
5. **Multi-Participant**: Combine JSON files from multiple participants for comparison

## Example Analysis Scripts

### Compare All Participants

```python
import json
import glob
import matplotlib.pyplot as plt

# Load all participant summaries
summaries = []
for f in glob.glob('app/logs/experiment_name/participant_*_summary.json'):
    with open(f) as file:
        summaries.append(json.load(file))

# Extract final test accuracies
accuracies = [s['final_metrics']['test_local']['Accuracy'] for s in summaries]
participant_ids = [s['participant_id'] for s in summaries]

# Plot comparison
plt.bar(participant_ids, accuracies)
plt.xlabel('Participant ID')
plt.ylabel('Final Test Accuracy')
plt.title('Model Performance Across Participants')
plt.show()
```

### Track Convergence

```python
import json
import matplotlib.pyplot as plt

with open('app/logs/experiment_name/participant_0_rounds.json') as f:
    rounds = json.load(f)

# Extract test accuracy per round
test_accuracies = [r['test_local']['metrics']['Accuracy'] for r in rounds]
round_numbers = [r['round'] for r in rounds]

plt.plot(round_numbers, test_accuracies, marker='o')
plt.xlabel('Round')
plt.ylabel('Test Accuracy')
plt.title('Model Convergence Over Federated Learning Rounds')
plt.grid(True)
plt.show()
```

## Troubleshooting

**Q: JSON files are not being created**
- Check that `device_args.logging` is set to `true` in your configuration
- Verify participant logs show "JSON Logger initialized"
- Check file permissions in the logs directory

**Q: JSON files are empty or incomplete**
- Ensure training completes normally (check error logs)
- Verify `json_logger.end_round()` is called after each round
- Check for exceptions in the main log file

**Q: Metrics are missing from JSON**
- Verify the model inherits from `NebulaModel`
- Check that `log_metrics_end()` is called for each phase
- Ensure metrics are computed successfully (check training logs)

## Future Enhancements

Planned improvements include:
- CSV export option for spreadsheet analysis
- Real-time metrics dashboard
- Automatic plot generation
- Metrics aggregation across participants
- Integration with experiment tracking tools (MLflow, Weights & Biases)
