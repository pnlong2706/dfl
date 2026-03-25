#!/usr/bin/env python3
"""
Script to parse DFL_2/Experiment.tsv and generate JSON configuration files for each experiment.

Each experiment will be saved as a separate JSON file with the body params following
the structure of the example in DFL_2/example.txt.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple


def generate_topology(n_nodes: int, topology: str, federation: str) -> Tuple[dict, dict, list]:
    """
    Generate nodes dictionary and adjacency matrix based on topology type.
    
    Args:
        n_nodes: Number of nodes
        topology: Topology type ("Ring", "Star", "Fully", "Random")
        federation: Federation type ("CFL", "SFL", "DFL")
    
    Returns:
        Tuple of (nodes_dict, nodes_graph, matrix)
    """
    import random
    
    nodes = {}
    nodes_graph = {}
    
    # Determine role based on federation type
    for i in range(n_nodes):
        node_id = str(i)
        
        # Assign role based on federation and topology
        if federation == "CFL":
            role = "server" if i == 0 else "trainer"
        elif federation == "SFL":
            role = "aggregator" if i == 0 else "trainer"
        else:  # DFL
            role = "trainer_aggregator"
        
        # Generate IP addresses like in the example (192.168.50.x)
        ip = f"192.168.50.{i + 2}"
        port = str(45001 + i)
        
        nodes[node_id] = {
            "id": node_id,
            "ip": ip,
            "port": port,
            "role": role,
            "malicious": False,
            "proxy": False,
            "start": (i == 0),
            "neighbors": [],
            "links": [],
        }
        nodes_graph[node_id] = {
            "id": node_id,
            "role": role,
            "malicious": False,
            "proxy": False,
            "start": (i == 0),
        }
    
    # Generate links based on topology
    links = []
    if topology == "Fully":
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                links.append((i, j))
    elif topology == "Ring":
        for i in range(n_nodes):
            links.append((i, (i + 1) % n_nodes))
    elif topology == "Star":
        for i in range(1, n_nodes):
            links.append((0, i))
    elif topology == "Random":
        # Random topology with ~20% connection probability (as mentioned in TSV note)
        random.seed(42)  # For reproducibility
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if random.random() < 0.2:
                    links.append((i, j))
    
    # Update neighbors based on links
    for source, target in links:
        source_id = str(source)
        target_id = str(target)
        
        # Add to neighbors (bidirectional)
        if target_id not in nodes[source_id]["neighbors"]:
            nodes[source_id]["neighbors"].append(target_id)
        if source_id not in nodes[target_id]["neighbors"]:
            nodes[target_id]["neighbors"].append(source_id)
        
        # Add to links
        nodes[source_id]["links"].append({"source": source, "target": target})
    
    # Generate adjacency matrix
    matrix = [[0] * n_nodes for _ in range(n_nodes)]
    for source, target in links:
        matrix[source][target] = 1
        matrix[target][source] = 1  # Bidirectional
    
    return nodes, nodes_graph, matrix


def parse_byzantine_percentage(percentage_str: str) -> int:
    """Parse Byzantine percentage string to integer."""
    if not percentage_str or percentage_str.strip() == "":
        return 0
    # Remove % sign and convert to integer
    clean = percentage_str.strip().replace("%", "")
    try:
        return int(clean)
    except ValueError:
        return 0


def parse_partition_parameter(dist_str: str) -> Tuple[str, float]:
    """
    Parse data distribution string to get partition selection and parameter.
    
    Examples:
    - "Dir(0.3)" -> ("dirichlet", 0.3)
    - "Dir(0.05)" -> ("dirichlet", 0.05)
    """
    if not dist_str or dist_str.strip() == "":
        return ("dirichlet", 0.3)  # Default
    
    dist_str = dist_str.strip()
    
    # Match Dir(x.x) pattern
    match = re.match(r"Dir\(([\d.]+)\)", dist_str, re.IGNORECASE)
    if match:
        param = float(match.group(1))
        return ("dirichlet", param)
    
    return ("dirichlet", 0.3)  # Default


def get_dataset_model(dataset: str) -> Tuple[str, str]:
    """
    Get dataset name and model based on dataset string.
    
    Returns:
        Tuple of (dataset_name, model_name)
    """
    dataset = dataset.strip().upper() if dataset else "CIFAR10"
    
    if "CIFAR100" in dataset or "CIFAR 100" in dataset:
        return ("CIFAR100", "SimpleNet")
    elif "CIFAR10" in dataset or "CIFAR 10" in dataset:
        return ("CIFAR10", "SimpleNet")
    elif "MNIST" in dataset:
        return ("MNIST", "CNN")
    else:
        return ("CIFAR10", "SimpleNet")


def parse_aggregation(agg_str: str) -> Tuple[str, bool, bool, bool]:
    """
    Parse aggregation string to determine algorithm and settings.
    
    Returns:
        Tuple of (agg_algorithm, pseudo_enabled, pcr_enabled, fedsam_enabled)
    """
    if not agg_str:
        agg_str = "DFedAvg"
    
    agg_str = agg_str.strip()
    
    agg_algorithm = "FedAvg"
    pseudo_enabled = False
    pcr_enabled = False
    fedsam_enabled = False
    
    # Check for Pseu (Pseudo aggregation)
    if agg_str.lower().startswith("pseu"):
        pseudo_enabled = True
        agg_str = agg_str[4:].strip()  # Remove "Pseu" prefix
    
    # Check for DFedSAM or DFedSam
    if "sam" in agg_str.lower():
        fedsam_enabled = True
        agg_algorithm = "FedAvg"
    elif "DFedAvg" in agg_str or "fedavg" in agg_str.lower():
        agg_algorithm = "FedAvg"
    
    # Check for PCR
    if "PCR" in agg_str or "pcr" in agg_str.lower():
        pcr_enabled = True
    
    return (agg_algorithm, pseudo_enabled, pcr_enabled, fedsam_enabled)


def get_attack_params(attack: str, byzantine_percent: int) -> dict:
    """
    Get attack parameters based on attack type.
    
    Args:
        attack: Attack type string
        byzantine_percent: Percentage of Byzantine nodes (0 to 100)
    
    Returns:
        Attack parameters dictionary
    """
    attack = attack.strip() if attack else "None"
    
    attack_mapping = {
        "None": "No Attack",
        "Label Flipping": "Label Flipping",
        "Gaussian Model Poisoning": "Gaussian Noise Attack",
        "Gaussian Model Posoining": "Gaussian Noise Attack",  # Typo in TSV
        "Trim Model Poisoning": "Trim Attack",
        "Krum Model Poisoning": "Krum Attack",
        "ALIE": "ALIE",
        "DISSENSUS": "Disensus",
    }
    
    attack_name = attack_mapping.get(attack, "No Attack")
    
    return {
        "attacks": attack_name,
        "poisoned_node_percent": byzantine_percent,
        "round_start_attack": 1,
        "round_stop_attack": 100,
        "attack_interval": 1,
    }


def get_defense_params(defense: str) -> dict:
    """
    Get defense/reputation parameters based on defense type.
    
    Args:
        defense: Defense type string
    
    Returns:
        Reputation parameters dictionary
    """
    defense = defense.strip() if defense else "None"
    
    # Defense mapping - these enable reputation system with specific defenses
    defense_enabled = defense not in ["None", "", "None"]
    
    return {
        "enabled": defense_enabled,
        "metrics": {
            "model_similarity": {"enabled": False, "weight": 0},
            "num_messages": {"enabled": False, "weight": 0},
            "model_arrival_latency": {"enabled": False, "weight": 0},
            "fraction_parameters_changed": {"enabled": False, "weight": 0},
        },
        "initial_reputation": 0.2,
        "weighting_factor": "dynamic",
    }


def generate_scenario_title(dataset: str, partition_param: float, aggregation: str, topology: str, attack: str = None, defense: str = None) -> str:
    """
    Generate a scenario title following the format of existing titles.
    
    Format: "{Dataset} ({partition_param}) {Aggregation} {Topology} {Attack} {Defense}"
    Example: "Cifar10 (0.05) DFedSam Ring"
    """
    # Parse dataset
    dataset_name = dataset.strip().replace(" ", "")
    if "CIFAR" in dataset_name.upper():
        if "100" in dataset_name:
            dataset_name = "Cifar100"
        else:
            dataset_name = "Cifar10"
    elif "MNIST" in dataset_name.upper():
        dataset_name = "MNIST"
    
    # Clean aggregation name
    agg_clean = aggregation.strip()
    
    # Clean topology
    topo_clean = topology.strip() if topology else "Ring"
    
    # Build title
    title = f"{dataset_name} ({partition_param}) {agg_clean} {topo_clean}"
    
    # Add attack if present and not None
    if attack and attack.strip() and attack.strip() != "None":
        title += f" {attack.strip()}"
    
    # Add defense if present and not None
    if defense and defense.strip() and defense.strip() != "None":
        title += f" {defense.strip()}"
    
    return title


def build_scenario(
    dataset: str,
    data_dist: str,
    topology: str,
    aggregation: str,
    byzantine_percent: int,
    attack: str,
    defense: str,
    scenario_title: str = None,
) -> dict:
    """
    Build a complete scenario configuration from parameters.
    
    Args:
        dataset: Dataset name (e.g., "CIFAR 10")
        data_dist: Data distribution (e.g., "Dir(0.3)")
        topology: Topology type (e.g., "Ring")
        aggregation: Aggregation method (e.g., "DFedAvg + PCR")
        byzantine_percent: Percentage of Byzantine nodes
        attack: Attack type (e.g., "None")
        defense: Defense type (e.g., "PRT")
        scenario_title: Optional scenario title
    
    Returns:
        Complete scenario configuration dictionary
    """
    # Parse values
    dataset_name, model_name = get_dataset_model(dataset)
    partition_selection, partition_parameter = parse_partition_parameter(data_dist)
    agg_algorithm, pseudo_enabled, pcr_enabled, fedsam_enabled = parse_aggregation(aggregation)
    attack_params = get_attack_params(attack, byzantine_percent)
    reputation = get_defense_params(defense)
    
    # Generate title if not provided
    if not scenario_title or not scenario_title.strip():
        scenario_title = generate_scenario_title(dataset, partition_parameter, aggregation, topology, attack, defense)
    
    # Generate topology
    n_nodes = 20
    federation = "DFL"
    topology_clean = topology.strip() if topology and topology.strip() else "Ring"
    nodes, nodes_graph, matrix = generate_topology(n_nodes, topology_clean, federation)
    
    # Build scenario
    scenario = {
        "scenario_title": scenario_title,
        "scenario_description": "empty",
        "deployment": "docker",
        "federation": federation,
        "rounds": 100,
        "topology": topology_clean,
        "nodes": nodes,
        "nodes_graph": nodes_graph,
        "n_nodes": n_nodes,
        "matrix": matrix,
        "dataset": dataset_name,
        "iid": False,
        "partition_selection": partition_selection,
        "partition_parameter": partition_parameter,
        "model": model_name,
        "agg_algorithm": agg_algorithm,
        "pseudo_aggregation": {
            "enabled": pseudo_enabled,
            "ema_alpha": 0.25,
            "weight_drop_rate": 1,
            "weight_schedule_step": 1,
            "stop_pseudo_round": None,
        },
        "fedsam": {
            "enabled": fedsam_enabled,
            "rho": 0.5,
        },
        "pcr": {
            "enabled": pcr_enabled,
            "mu": 0.01,
            "apply_mode": "all_round",  # As per TSV note: pcr-apply-mode = all_round
        },
        "mid_round_test": False,  # As per TSV note: mid-round-testing = false
        "logginglevel": True,
        "report_status_data_queue": True,
        "epochs": 2,
        "attack_params": attack_params,
        "reputation": reputation,
        "mobility": False,
        "network_simulation": False,
        "mobility_type": "both",
        "radius_federation": 500,
        "scheme_mobility": "random",
        "round_frequency": 1,
        "mobile_participants_percent": 100,
        "random_geo": False,
        "latitude": 38.023522,
        "longitude": -1.174389,
        "with_sa": False,
        "strict_topology": False,
        "sad_candidate_selector": "Distance",
        "sad_model_handler": "std",
        "sar_arbitration_policy": "sap",
        "sar_neighbor_policy": "Distance",
        "sar_training": False,
        "sar_training_policy": "Broad-Propagation Strategy",
        "random_topology_probability": "0.5",
        "with_trustworthiness": False,
        "robustness_pillar": "20",
        "resilience_to_attacks": "40",
        "algorithm_robustness": "40",
        "client_reliability": "20",
        "privacy_pillar": "15",
        "technique": "20",
        "uncertainty": "60",
        "indistinguishability": "20",
        "fairness_pillar": "15",
        "selection_fairness": "30",
        "performance_fairness": "35",
        "class_distribution": "35",
        "explainability_pillar": "15",
        "interpretability": "40",
        "post_hoc_methods": "60",
        "accountability_pillar": "10",
        "factsheet_completeness": "100",
        "architectural_soundness_pillar": "10",
        "client_management": "50",
        "optimization": "50",
        "sustainability_pillar": "15",
        "energy_source": "50",
        "hardware_efficiency": "25",
        "federation_complexity": "25",
        "network_subnet": "172.20.0.0/16",
        "network_gateway": "172.20.0.1",
        "additional_participants": [],
        "schema_additional_participants": "random",
        "accelerator": "cpu",
        "gpu_id": [],
        "physical_ips": [],
    }
    
    return scenario


def parse_tsv_file(tsv_path: str) -> List[dict]:
    """
    Parse the TSV file and extract all experiments.
    
    The TSV has a specific structure:
    - Header rows with column names
    - Data rows where empty cells inherit from the row above
    - Multiple sections separated by empty rows
    
    Column indices (0-indexed, with column 0 being empty):
    - [1]: Dataset
    - [2]: Data distribution
    - [3]: Topology
    - [4]: Aggregation
    - [5]: Status
    - [6]: Byzantine percentage
    - [7]: Attack
    - [8]: Defense
    - [9]: Note
    - [10]: Port
    - [11]: Scenario title
    
    Args:
        tsv_path: Path to the TSV file
    
    Returns:
        List of experiment dictionaries
    """
    experiments = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Column indices (based on actual TSV structure)
    COL_DATASET = 1
    COL_DATA_DIST = 2
    COL_TOPOLOGY = 3
    COL_AGGREGATION = 4
    COL_STATUS = 5
    COL_BYZANTINE = 6
    COL_ATTACK = 7
    COL_DEFENSE = 8
    COL_NOTE = 9
    COL_PORT = 10
    COL_SCENARIO_TITLE = 11
    
    # Track current values (for inheritance)
    current_dataset = ""
    current_data_dist = ""
    current_topology = ""
    current_byzantine = "0%"
    current_attack = "None"
    current_aggregation = ""  # Track aggregation for inheritance
    
    # Process each line
    for line_num, line in enumerate(lines, 1):
        # Skip empty lines
        if not line.strip():
            continue
        
        # Split by tab
        cells = line.split('\t')
        
        # Clean cells (strip whitespace)
        cells = [c.strip() for c in cells]
        
        # Skip header rows and metadata rows
        if cells[COL_DATASET] == "Dataset":
            continue
        if cells[1].startswith("DFL Experiment") or cells[1].startswith("For model:") or cells[1].startswith("Default:"):
            continue
        
        # Update current values if new values are present
        if len(cells) > COL_DATASET and cells[COL_DATASET]:
            current_dataset = cells[COL_DATASET]
        if len(cells) > COL_DATA_DIST and cells[COL_DATA_DIST]:
            current_data_dist = cells[COL_DATA_DIST]
        if len(cells) > COL_TOPOLOGY and cells[COL_TOPOLOGY]:
            current_topology = cells[COL_TOPOLOGY]
        if len(cells) > COL_BYZANTINE and cells[COL_BYZANTINE]:
            current_byzantine = cells[COL_BYZANTINE]
        if len(cells) > COL_ATTACK and cells[COL_ATTACK]:
            current_attack = cells[COL_ATTACK]
        
        # Get aggregation - update current if present, otherwise use inherited
        aggregation = cells[COL_AGGREGATION] if len(cells) > COL_AGGREGATION and cells[COL_AGGREGATION] else ""
        if aggregation:
            current_aggregation = aggregation
        else:
            aggregation = current_aggregation
        
        # Skip rows without aggregation (even inherited)
        if not aggregation:
            continue
        
        # Get other values
        defense = cells[COL_DEFENSE] if len(cells) > COL_DEFENSE else ""
        scenario_title = cells[COL_SCENARIO_TITLE] if len(cells) > COL_SCENARIO_TITLE else ""
        
        # Check if this row has a status (indicates it's a valid experiment row)
        status = cells[COL_STATUS] if len(cells) > COL_STATUS else ""
        
        # Only add rows that have a status (Finish, Pending, Running)
        if status not in ["Finish", "Pending", "Running", "Finish "]:
            continue
        
        # Build experiment
        experiment = {
            "line": line_num,
            "dataset": current_dataset,
            "data_dist": current_data_dist,
            "topology": current_topology,
            "aggregation": aggregation,
            "byzantine_percent": parse_byzantine_percentage(current_byzantine),
            "attack": current_attack,
            "defense": defense,
            "scenario_title": scenario_title,
            "status": status,
        }
        
        experiments.append(experiment)
    
    return experiments


def main():
    """Main entry point."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tsv_path = os.path.join(script_dir, "Experiment.tsv")
    output_dir = os.path.join(script_dir, "experiments")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Parsing TSV file: {tsv_path}")
    
    # Parse TSV
    experiments = parse_tsv_file(tsv_path)
    
    print(f"Found {len(experiments)} experiments")
    
    # Track experiments that need title updates
    title_updates = []
    
    # Generate JSON files
    for i, exp in enumerate(experiments):
        # Build scenario
        scenario = build_scenario(
            dataset=exp["dataset"],
            data_dist=exp["data_dist"],
            topology=exp["topology"],
            aggregation=exp["aggregation"],
            byzantine_percent=exp["byzantine_percent"],
            attack=exp["attack"],
            defense=exp["defense"],
            scenario_title=exp["scenario_title"],
        )
        
        # Check if scenario title was generated
        if not exp["scenario_title"]:
            title_updates.append({
                "line": exp["line"],
                "generated_title": scenario["scenario_title"],
            })
        
        # Create safe filename
        safe_title = re.sub(r'[^\w\s-]', '', scenario["scenario_title"])
        safe_title = re.sub(r'[-\s]+', '_', safe_title).strip('_')
        filename = f"{i+1:03d}_{safe_title}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Write JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([scenario], f, indent=2)
        
        print(f"  Created: {filename}")
    
    print(f"\nGenerated {len(experiments)} JSON files in: {output_dir}")
    
    # Print title updates
    if title_updates:
        print(f"\n{len(title_updates)} experiments needed title updates:")
        for update in title_updates:
            print(f"  Line {update['line']}: {update['generated_title']}")
    
    return experiments, title_updates


if __name__ == "__main__":
    main()
