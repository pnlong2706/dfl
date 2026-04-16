#!/usr/bin/env python3
"""
Plot DFL experiment metrics for comparison.

Usage:
    python plot_metrics.py -name comparison1 \
        "nebula_DFL_2026_04_03_23_16_24:FedAvg" \
        "nebula_DFL_2026_04_04_22_26_22:Krum" \
        "nebula_DFL_synthetic:Baseline"

Each positional arg is "experiment_name:legend_label".
Looks up json_output/<experiment_name>_stats.json for each.

Outputs 4 figures to figure/<name>/:
    - accuracy_avg.png
    - loss_avg.png
    - accuracy_std.png
    - loss_std.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_experiment(name, honest_only=False):
    if honest_only:
        candidates = [f"{name}_stats_honest.json", f"{name}_stats.json", f"{name}.json"]
    else:
        candidates = [f"{name}_stats.json", f"{name}.json"]
    for suffix in candidates:
        path = Path("json_output") / suffix
        if path.exists():
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f"Not found: json_output/{name}_stats.json or json_output/{name}.json")


def extract_series(data):
    rounds = [r["round"] for r in data["rounds"]]
    acc_avg = [r["global_accuracy"]["avg"] for r in data["rounds"] if "global_accuracy" in r]
    acc_std = [r["global_accuracy"]["std"] for r in data["rounds"] if "global_accuracy" in r]
    loss_avg = [r["global_loss"]["avg"] for r in data["rounds"] if "global_loss" in r]
    loss_std = [r["global_loss"]["std"] for r in data["rounds"] if "global_loss" in r]
    # Use only rounds that have both metrics
    n = min(len(acc_avg), len(loss_avg), len(rounds))
    return rounds[:n], acc_avg[:n], acc_std[:n], loss_avg[:n], loss_std[:n]


def plot_figure(all_series, key_idx, title, ylabel, out_path, fill_std_idx=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, (rounds, *series) in all_series:
        y = series[key_idx]
        ax.plot(rounds, y, label=label, linewidth=1.5)
        if fill_std_idx is not None:
            std = series[fill_std_idx]
            y_arr = [float(v) for v in y]
            s_arr = [float(v) for v in std]
            ax.fill_between(rounds,
                            [v - s for v, s in zip(y_arr, s_arr)],
                            [v + s for v, s in zip(y_arr, s_arr)],
                            alpha=0.15)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot DFL experiment metrics for comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("experiments", nargs="+",
                        help='"experiment_name:legend_label" pairs')
    parser.add_argument("-name", required=True,
                        help="Output folder name (figure/<name>/)")

    args = parser.parse_args()

    out_dir = Path("figure") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant, honest_only, file_suffix in [("", False, ""), ("_honest_only", True, "_honest_only")]:
        all_series = []
        for entry in args.experiments:
            if ":" in entry:
                exp_name, label = entry.split(":", 1)
            else:
                exp_name = entry
                label = entry
            data = load_experiment(exp_name, honest_only=honest_only)
            series = extract_series(data)
            all_series.append((label, series))

        title_suffix = " (honest only)" if honest_only else ""
        print(f"Plotting {len(all_series)} experiments{title_suffix}\n")

        plot_figure(all_series, 0, f"Global Accuracy (avg){title_suffix}", "Accuracy",
                    out_dir / f"accuracy_avg{file_suffix}.png", fill_std_idx=1)
        plot_figure(all_series, 2, f"Global Loss (avg){title_suffix}", "Loss",
                    out_dir / f"loss_avg{file_suffix}.png", fill_std_idx=3)
        plot_figure(all_series, 1, f"Global Accuracy (std){title_suffix}", "Std",
                    out_dir / f"accuracy_std{file_suffix}.png")
        plot_figure(all_series, 3, f"Global Loss (std){title_suffix}", "Std",
                    out_dir / f"loss_std{file_suffix}.png")


if __name__ == "__main__":
    main()
