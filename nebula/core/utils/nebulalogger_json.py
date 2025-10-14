import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class NebulaJSONLogger:
    """
    Logger for NEBULA that saves metrics and training information in JSON format.

    This logger creates easily accessible JSON files containing:
    - Training metrics (loss, accuracy, precision, recall, f1)
    - Validation metrics
    - Test metrics
    - Round information
    - Timestamps
    - Network information (neighbors, role, etc.)
    """

    def __init__(self, log_dir: str, participant_id: int, scenario_name: str):
        """
        Initialize the JSON logger.

        Args:
            log_dir: Base directory for logs
            participant_id: ID of the participant
            scenario_name: Name of the scenario/experiment
        """
        self.log_dir = Path(log_dir)
        self.participant_id = participant_id
        self.scenario_name = scenario_name

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Define log file paths
        self.metrics_file = self.log_dir / f"participant_{participant_id}_metrics.json"
        self.rounds_file = self.log_dir / f"participant_{participant_id}_rounds.json"
        self.summary_file = self.log_dir / f"participant_{participant_id}_summary.json"

        # Initialize data structures
        self.metrics_data = {
            "participant_id": participant_id,
            "scenario": scenario_name,
            "start_time": datetime.now().isoformat(),
            "rounds": []
        }

        self.current_round_data = None

        logging.info(f"JSON Logger initialized for participant {participant_id}")
        logging.info(f"  Metrics file: {self.metrics_file}")
        logging.info(f"  Rounds file: {self.rounds_file}")
        logging.info(f"  Summary file: {self.summary_file}")

    def start_round(self, round_num: int, **kwargs):
        """Start logging a new round."""
        self.current_round_data = {
            "round": round_num,
            "start_time": datetime.now().isoformat(),
            "dataset_info": {},
            "training": {},
            "validation": {},
            "test_local": {},
            "test_global": {},
            "network": {},
            "metadata": kwargs
        }
        logging.info(f"[JSON Logger] Started logging round {round_num}")

    def log_dataset_info(self, num_train_samples: int, num_val_samples: int = None,
                        num_test_local_samples: int = None, num_test_global_samples: int = None):
        """Log dataset information at the beginning of a round."""
        if self.current_round_data is None:
            logging.warning("[JSON Logger] No active round. Call start_round() first.")
            return

        dataset_info = {
            "num_train_samples": num_train_samples,
        }

        if num_val_samples is not None:
            dataset_info["num_val_samples"] = num_val_samples
        if num_test_local_samples is not None:
            dataset_info["num_test_local_samples"] = num_test_local_samples
        if num_test_global_samples is not None:
            dataset_info["num_test_global_samples"] = num_test_global_samples

        self.current_round_data["dataset_info"] = dataset_info

        # Log to console
        info_str = f"Training samples: {num_train_samples}"
        if num_val_samples is not None:
            info_str += f", Validation samples: {num_val_samples}"
        if num_test_local_samples is not None:
            info_str += f", Test (Local) samples: {num_test_local_samples}"
        if num_test_global_samples is not None:
            info_str += f", Test (Global) samples: {num_test_global_samples}"

        logging.info(f"[Dataset Info] {info_str}")

    def log_training_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Log training metrics for an epoch."""
        if self.current_round_data is None:
            logging.warning("[JSON Logger] No active round. Call start_round() first.")
            return

        if "epochs" not in self.current_round_data["training"]:
            self.current_round_data["training"]["epochs"] = []

        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "metrics": self._convert_tensors(metrics)
        }
        self.current_round_data["training"]["epochs"].append(epoch_data)

        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                                for k, v in metrics.items()])
        logging.info(f"[Training] Epoch {epoch} - {metrics_str}")

    def log_validation_metrics(self, metrics: Dict[str, Any]):
        """Log validation metrics."""
        if self.current_round_data is None:
            logging.warning("[JSON Logger] No active round. Call start_round() first.")
            return

        self.current_round_data["validation"] = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self._convert_tensors(metrics)
        }

        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                                for k, v in metrics.items()])
        logging.info(f"[Validation] {metrics_str}")

    def log_test_metrics(self, phase: str, metrics: Dict[str, Any]):
        """Log test metrics (local or global)."""
        if self.current_round_data is None:
            logging.warning("[JSON Logger] No active round. Call start_round() first.")
            return

        phase_key = "test_local" if "Local" in phase else "test_global"
        self.current_round_data[phase_key] = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self._convert_tensors(metrics)
        }

        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                                for k, v in metrics.items()])
        logging.info(f"[{phase}] {metrics_str}")

    def log_network_info(self, **kwargs):
        """Log network information (neighbors, role, etc.)."""
        if self.current_round_data is None:
            logging.warning("[JSON Logger] No active round. Call start_round() first.")
            return

        self.current_round_data["network"].update(self._convert_tensors(kwargs))

        # Log to console
        info_str = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
        logging.info(f"[Network] {info_str}")

    def end_round(self):
        """End the current round and save data."""
        if self.current_round_data is None:
            logging.warning("[JSON Logger] No active round to end.")
            return

        self.current_round_data["end_time"] = datetime.now().isoformat()

        # Calculate round duration
        start = datetime.fromisoformat(self.current_round_data["start_time"])
        end = datetime.fromisoformat(self.current_round_data["end_time"])
        self.current_round_data["duration_seconds"] = (end - start).total_seconds()

        # Add to rounds list
        self.metrics_data["rounds"].append(self.current_round_data)

        # Save to file
        self._save_metrics()
        self._save_round_summary()

        round_num = self.current_round_data["round"]
        logging.info(f"[JSON Logger] Finished logging round {round_num} "
                    f"(duration: {self.current_round_data['duration_seconds']:.2f}s)")

        self.current_round_data = None

    def log_experiment_end(self):
        """Log end of experiment and create final summary."""
        self.metrics_data["end_time"] = datetime.now().isoformat()

        # Calculate total duration
        start = datetime.fromisoformat(self.metrics_data["start_time"])
        end = datetime.fromisoformat(self.metrics_data["end_time"])
        self.metrics_data["total_duration_seconds"] = (end - start).total_seconds()

        # Save final data
        self._save_metrics()
        self._create_summary()

        logging.info(f"[JSON Logger] Experiment ended. Total duration: "
                    f"{self.metrics_data['total_duration_seconds']:.2f}s")

    def _convert_tensors(self, data: Any) -> Any:
        """Convert tensors and special types to JSON-serializable format."""
        import torch

        if isinstance(data, dict):
            return {k: self._convert_tensors(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_tensors(item) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().item() if data.numel() == 1 else data.detach().cpu().tolist()
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)

    def _save_metrics(self):
        """Save all metrics to JSON file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_data, f, indent=2)
        except Exception as e:
            logging.error(f"[JSON Logger] Error saving metrics: {e}")

    def _save_round_summary(self):
        """Save individual round data to separate file."""
        if self.current_round_data is None:
            return

        try:
            # Append mode for rounds
            rounds_data = []
            if self.rounds_file.exists():
                with open(self.rounds_file, 'r') as f:
                    rounds_data = json.load(f)

            rounds_data.append(self.current_round_data)

            with open(self.rounds_file, 'w') as f:
                json.dump(rounds_data, f, indent=2)
        except Exception as e:
            logging.error(f"[JSON Logger] Error saving round summary: {e}")

    def _create_summary(self):
        """Create a summary of the experiment."""
        summary = {
            "participant_id": self.participant_id,
            "scenario": self.scenario_name,
            "start_time": self.metrics_data["start_time"],
            "end_time": self.metrics_data.get("end_time"),
            "total_duration_seconds": self.metrics_data.get("total_duration_seconds"),
            "total_rounds": len(self.metrics_data["rounds"]),
            "final_metrics": {}
        }

        # Get final round metrics
        if self.metrics_data["rounds"]:
            last_round = self.metrics_data["rounds"][-1]

            # Training metrics from last epoch
            if last_round["training"].get("epochs"):
                last_epoch = last_round["training"]["epochs"][-1]
                summary["final_metrics"]["training"] = last_epoch["metrics"]

            # Validation metrics
            if last_round["validation"].get("metrics"):
                summary["final_metrics"]["validation"] = last_round["validation"]["metrics"]

            # Test metrics
            if last_round["test_local"].get("metrics"):
                summary["final_metrics"]["test_local"] = last_round["test_local"]["metrics"]
            if last_round["test_global"].get("metrics"):
                summary["final_metrics"]["test_global"] = last_round["test_global"]["metrics"]

        try:
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            logging.info(f"[JSON Logger] Summary saved to {self.summary_file}")
        except Exception as e:
            logging.error(f"[JSON Logger] Error creating summary: {e}")
