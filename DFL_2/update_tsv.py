#!/usr/bin/env python3
"""
Script to update DFL_2/Experiment.tsv with generated scenario titles for experiments that have no name.
"""

import re
from typing import List, Tuple


def parse_byzantine_percentage(percentage_str: str) -> int:
    """Parse Byzantine percentage string to integer."""
    if not percentage_str or percentage_str.strip() == "":
        return 0
    clean = percentage_str.strip().replace("%", "")
    try:
        return int(clean)
    except ValueError:
        return 0


def parse_partition_parameter(dist_str: str) -> float:
    """Parse data distribution string to get partition parameter."""
    if not dist_str or dist_str.strip() == "":
        return 0.3
    dist_str = dist_str.strip()
    match = re.match(r"Dir\(([\d.]+)\)", dist_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.3


def get_dataset_name(dataset: str) -> str:
    """Get formatted dataset name."""
    dataset = dataset.strip().upper() if dataset else "CIFAR10"
    if "CIFAR100" in dataset or "CIFAR 100" in dataset:
        return "Cifar100"
    elif "CIFAR10" in dataset or "CIFAR 10" in dataset:
        return "Cifar10"
    elif "MNIST" in dataset:
        return "MNIST"
    return "Cifar10"


def generate_scenario_title(dataset: str, partition_param: float, aggregation: str, topology: str, attack: str = None, defense: str = None) -> str:
    """Generate a scenario title following the format of existing titles."""
    dataset_name = get_dataset_name(dataset)
    agg_clean = aggregation.strip()
    topo_clean = topology.strip() if topology else "Ring"
    
    title = f"{dataset_name} ({partition_param}) {agg_clean} {topo_clean}"
    
    # Add attack if present and not None
    if attack and attack.strip() and attack.strip() != "None":
        title += f" {attack.strip()}"
    
    # Add defense if present and not None
    if defense and defense.strip() and defense.strip() != "None":
        title += f" {defense.strip()}"
    
    return title


def update_tsv_file(tsv_path: str) -> List[Tuple[int, str]]:
    """
    Update the TSV file with generated scenario titles.
    
    Returns:
        List of (line_number, generated_title) tuples for updated lines
    """
    with open(tsv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Column indices (based on actual TSV structure)
    COL_DATASET = 1
    COL_DATA_DIST = 2
    COL_TOPOLOGY = 3
    COL_AGGREGATION = 4
    COL_STATUS = 5
    COL_BYZANTINE = 6
    COL_ATTACK = 7
    COL_DEFENSE = 8
    COL_SCENARIO_TITLE = 11
    
    # Track current values (for inheritance)
    current_dataset = ""
    current_data_dist = ""
    current_topology = ""
    current_byzantine = "0%"
    current_attack = "None"
    current_aggregation = ""
    
    updates = []
    updated_lines = []
    
    for line_num, line in enumerate(lines, 1):
        # Skip empty lines
        if not line.strip():
            updated_lines.append(line)
            continue
        
        # Split by tab
        cells = line.split('\t')
        
        # Skip header rows and metadata rows
        if len(cells) > COL_DATASET and cells[COL_DATASET].strip() == "Dataset":
            updated_lines.append(line)
            continue
        if len(cells) > 1 and (cells[1].strip().startswith("DFL Experiment") or 
                               cells[1].strip().startswith("For model:") or 
                               cells[1].strip().startswith("Default:")):
            updated_lines.append(line)
            continue
        
        # Update current values if new values are present
        if len(cells) > COL_DATASET and cells[COL_DATASET].strip():
            current_dataset = cells[COL_DATASET].strip()
        if len(cells) > COL_DATA_DIST and cells[COL_DATA_DIST].strip():
            current_data_dist = cells[COL_DATA_DIST].strip()
        if len(cells) > COL_TOPOLOGY and cells[COL_TOPOLOGY].strip():
            current_topology = cells[COL_TOPOLOGY].strip()
        if len(cells) > COL_BYZANTINE and cells[COL_BYZANTINE].strip():
            current_byzantine = cells[COL_BYZANTINE].strip()
        if len(cells) > COL_ATTACK and cells[COL_ATTACK].strip():
            current_attack = cells[COL_ATTACK].strip()
        
        # Get aggregation - update current if present, otherwise use inherited
        aggregation = cells[COL_AGGREGATION].strip() if len(cells) > COL_AGGREGATION and cells[COL_AGGREGATION].strip() else ""
        if aggregation:
            current_aggregation = aggregation
        else:
            aggregation = current_aggregation
        
        # Skip rows without aggregation
        if not aggregation:
            updated_lines.append(line)
            continue
        
        # Get other values
        defense = cells[COL_DEFENSE].strip() if len(cells) > COL_DEFENSE else ""
        scenario_title = cells[COL_SCENARIO_TITLE].strip() if len(cells) > COL_SCENARIO_TITLE else ""
        status = cells[COL_STATUS].strip() if len(cells) > COL_STATUS else ""
        
        # Only process rows with valid status
        if status not in ["Finish", "Pending", "Running", "Finish "]:
            updated_lines.append(line)
            continue
        
        # If no scenario title, generate one
        if not scenario_title:
            partition_param = parse_partition_parameter(current_data_dist)
            generated_title = generate_scenario_title(current_dataset, partition_param, aggregation, current_topology, current_attack, defense)
            
            # Update the cell
            cells[COL_SCENARIO_TITLE] = generated_title
            
            # Reconstruct the line
            updated_line = '\t'.join(cells)
            updated_lines.append(updated_line)
            
            updates.append((line_num, generated_title))
            print(f"  Line {line_num}: {generated_title}")
        else:
            updated_lines.append(line)
    
    # Write updated content
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    return updates


def main():
    """Main entry point."""
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tsv_path = os.path.join(script_dir, "Experiment.tsv")
    
    print(f"Updating TSV file: {tsv_path}")
    print()
    
    updates = update_tsv_file(tsv_path)
    
    print()
    print(f"Updated {len(updates)} scenario titles in {tsv_path}")
    
    return updates


if __name__ == "__main__":
    main()
