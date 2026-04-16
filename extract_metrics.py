#!/usr/bin/env python3
"""Extract per-round global accuracy/loss statistics across all nodes.

Usage:
    python extract_metrics.py <experiment_log_dir>
    python extract_metrics.py <experiment_log_dir> --pseudo   # for pseudo aggregation experiments

Outputs TWO json files:
    <name>_stats.json          — all nodes (including malicious)
    <name>_stats_honest.json   — honest nodes only
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_malicious_indices(experiment_dir: str) -> set:
    """Read participant configs to find which nodes are malicious."""
    config_dir = Path(experiment_dir).parent.parent / "config" / Path(experiment_dir).name
    if not config_dir.exists():
        # Try alternate path (logs might be in archive)
        config_dir = Path("app/config") / Path(experiment_dir).name
    malicious = set()
    if not config_dir.exists():
        return malicious
    for f in sorted(config_dir.glob("participant_*.json")):
        try:
            cfg = json.load(open(f))
            idx = cfg["device_args"]["idx"]
            if cfg["device_args"].get("malicious", False):
                malicious.add(idx)
        except (KeyError, json.JSONDecodeError):
            pass
    return malicious


def extract(experiment_dir: str, pseudo: bool = False, exclude_indices: set = None):
    metrics_dir = Path(experiment_dir) / "metrics"
    if not metrics_dir.exists():
        print(f"Error: {metrics_dir} not found")
        sys.exit(1)

    participant_dirs = sorted(metrics_dir.iterdir())
    if not participant_dirs:
        print(f"Error: no participant dirs in {metrics_dir}")
        sys.exit(1)

    acc_by_round = defaultdict(list)
    loss_by_round = defaultdict(list)

    for pdir in participant_dirs:
        # Extract participant index from dir name (e.g., "participant_3" -> 3)
        pname = pdir.name
        try:
            pidx = int(pname.split("_")[-1])
        except ValueError:
            pidx = -1

        if exclude_indices and pidx in exclude_indices:
            continue

        ea = EventAccumulator(str(pdir))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])

        rounds = {e.step: int(e.value) for e in ea.Scalars("A-Round")} if "A-Round" in tags else {}

        if "Test (Global)/Accuracy" in tags:
            for e in ea.Scalars("Test (Global)/Accuracy"):
                r = rounds.get(e.step, e.step)
                acc_by_round[r].append(e.value)

        if "Train/Loss" in tags:
            round_steps = sorted(rounds.keys())
            loss_events = ea.Scalars("Train/Loss")
            for i, rs in enumerate(round_steps):
                next_rs = round_steps[i + 1] if i + 1 < len(round_steps) else float("inf")
                round_losses = [e.value for e in loss_events if rs <= e.step < next_rs]
                if round_losses:
                    loss_by_round[rounds[rs]].append(round_losses[-1])

    all_rounds = sorted(set(acc_by_round.keys()) | set(loss_by_round.keys()))

    raw_rounds = []
    for r in all_rounds:
        entry = {"round": r}
        if r in acc_by_round:
            vals = np.array(acc_by_round[r])
            entry["global_accuracy"] = {
                "avg": float(np.mean(vals)),
                "max": float(np.max(vals)),
                "min": float(np.min(vals)),
                "std": float(np.std(vals)),
                "var": float(np.var(vals)),
            }
        if r in loss_by_round:
            vals = np.array(loss_by_round[r])
            entry["global_loss"] = {
                "avg": float(np.mean(vals)),
                "max": float(np.max(vals)),
                "min": float(np.min(vals)),
                "std": float(np.std(vals)),
                "var": float(np.var(vals)),
            }
        raw_rounds.append(entry)

    if pseudo:
        merged_rounds = []
        i = 0
        logical = 0
        while i < len(raw_rounds):
            r1 = raw_rounds[i]
            r2 = raw_rounds[i + 1] if i + 1 < len(raw_rounds) else None
            entry = {"round": logical}

            acc_entries = [r for r in [r1, r2] if r and "global_accuracy" in r]
            if acc_entries:
                best = max(acc_entries, key=lambda x: x["global_accuracy"]["avg"])
                entry["global_accuracy"] = dict(best["global_accuracy"])

            loss_entries = [r for r in [r1, r2] if r and "global_loss" in r]
            if loss_entries:
                best = min(loss_entries, key=lambda x: x["global_loss"]["avg"])
                entry["global_loss"] = dict(best["global_loss"])

            merged_rounds.append(entry)
            logical += 1
            i += 2
        final_rounds = merged_rounds
    else:
        final_rounds = raw_rounds

    num_included = len(participant_dirs) - (len(exclude_indices) if exclude_indices else 0)
    return {
        "experiment": Path(experiment_dir).name,
        "num_participants": num_included,
        "rounds": final_rounds,
    }


def print_summary(result, label=""):
    if label:
        print(f"  ({label})")
    for r in result["rounds"]:
        acc = r.get("global_accuracy", {})
        loss = r.get("global_loss", {})
        if acc and loss:
            print(f"  Round {r['round']:>2}: acc={acc.get('avg', 'N/A'):.4f} (std={acc.get('std', 0):.4f})  loss={loss.get('avg', 'N/A'):.4f} (std={loss.get('std', 0):.4f})")
        else:
            print(f"  Round {r['round']:>2}: partial data")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_metrics.py <experiment_log_dir> [--pseudo]")
        sys.exit(1)

    experiment_dir = sys.argv[1]
    pseudo = "--pseudo" in sys.argv

    # Find malicious nodes
    malicious = get_malicious_indices(experiment_dir)

    # Extract ALL nodes
    result_all = extract(experiment_dir, pseudo=pseudo, exclude_indices=None)
    out_all = Path("json_output") / f"{result_all['experiment']}_stats.json"
    out_all.parent.mkdir(exist_ok=True)
    with open(out_all, "w") as f:
        json.dump(result_all, f, indent=2)
    print(f"Written to {out_all} ({result_all['num_participants']} nodes)")
    if pseudo:
        print(f"  (pseudo mode: merged to {len(result_all['rounds'])} logical rounds)")

    # Extract HONEST only (if malicious nodes found)
    if malicious:
        result_honest = extract(experiment_dir, pseudo=pseudo, exclude_indices=malicious)
        out_honest = Path("json_output") / f"{result_honest['experiment']}_stats_honest.json"
        with open(out_honest, "w") as f:
            json.dump(result_honest, f, indent=2)
        print(f"Written to {out_honest} ({result_honest['num_participants']} honest nodes, excluded {len(malicious)} malicious: {malicious})")
    else:
        print("  No malicious nodes detected — honest-only output skipped")

    # Print summary (honest if available, else all)
    summary = result_honest if malicious else result_all
    print_summary(summary, "honest only" if malicious else "all nodes")
