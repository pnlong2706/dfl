#!/usr/bin/env python3
"""
Script to run a DFL experiment with custom configuration.
Based on the first experiment from DFL Experiment.tsv:
- Dataset: CIFAR 10
- Data distribution: Dir(0.3)
- Topology: Ring
- Aggregation: DFedAvg
- Byzantine percentage: 0%
- Attack: None
- Defense: None
- Defaults: 100 rounds, 2 epochs/round, 20 nodes, SimpleNet model
"""

import asyncio
import json
import os
import sys
import subprocess
import time
import hashlib
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nebula.addons.topologymanager import TopologyManager


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

EXPERIMENT_CONFIG = {
    # Scenario settings
    "scenario_name": "cifar10_dir03_ring_dfedavg",
    "rounds": 100,
    "epochs": 2,
    "n_nodes": 20,
    
    # Dataset settings
    "dataset": "CIFAR10",
    "model": "SimpleNet",
    "iid": False,
    "partition_selection": "dirichlet",
    "partition_parameter": 0.3,  # Dir(0.3)
    
    # Topology settings
    "topology": "Ring",
    "federation": "DFL",  # Decentralized Federated Learning
    
    # Aggregation settings
    "aggregation_algorithm": "FedAvg",
    
    # Attack settings (None for this experiment)
    "attack": "No Attack",
    "byzantine_percent": 0,
    
    # Defense settings (None for this experiment)
    "defense": None,
    
    # Device settings
    "accelerator": "cpu",  # Change to "gpu" if available
    "gpu_id": None,
    "logging": True,
    
    # Network settings
    "base_port": 45000,
    "base_ip": "127.0.0.1",
    
    # Directories
    "log_dir": "./experiment_logs",
    "config_dir": "./experiment_config",
}


# =============================================================================
# PARTICIPANT CONFIGURATION TEMPLATE
# =============================================================================

def get_participant_template():
    """Return the base participant configuration template."""
    return {
        "scenario_args": {
            "name": "",
            "start_time": "",
            "federation": "DFL",
            "rounds": 100,
            "deployment": "process",
            "controller": "127.0.0.1:5000",
            "random_seed": 42,
            "n_nodes": 0,
            "config_version": "development"
        },
        "device_args": {
            "uid": "",
            "idx": 0,
            "name": "",
            "username": "pi",
            "password": "pi",
            "role": "trainer_aggregator",
            "proxy": False,
            "malicious": False,
            "start": True,
            "accelerator": "cpu",
            "gpu_id": None,
            "devices": "auto",
            "strategy": "ddp",
            "logging": True
        },
        "security_args": {
            "certfile": "",
            "keyfile": "",
            "cafile": ""
        },
        "federation_args": {
            "round": 0
        },
        "network_args": {
            "ip": "127.0.0.1",
            "port": 45000,
            "addr": "",
            "neighbors": "",
            "interface": "eth0",
            "simulation": False,
            "bandwidth": "5Gbps",
            "delay": "0ms",
            "delay-distro": "0ms",
            "delay-distribution": "normal",
            "loss": "0%",
            "duplicate": "0%",
            "corrupt": "0%",
            "reordering": "0%"
        },
        "adaptive_args": {
            "model_similarity": True
        },
        "mobility_args": {
            "latitude": "",
            "longitude": "",
            "change_geo_interval": 5,
            "grace_time_mobility": 60,
            "random_geo": True,
            "mobility": False,
            "mobility_type": "topology",
            "topology_type": "Ring",
            "radius_federation": 1000,
            "scheme_mobility": "random",
            "round_frequency": 1,
            "neighbors_distance": {},
            "additional_node": {
                "status": False,
                "time_start": 0,
                "scheme": "random"
            }
        },
        "data_args": {
            "dataset": "CIFAR10",
            "iid": False,
            "num_workers": 4,
            "partition_selection": "dirichlet",
            "partition_parameter": 0.3
        },
        "model_args": {
            "model": "SimpleNet"
        },
        "training_args": {
            "trainer": "lightning",
            "epochs": 2,
            "batch_size": 32,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "scheduler": "steplr",
            "step_size": 10,
            "gamma": 0.1
        },
        "aggregator_args": {
            "algorithm": "FedAvg",
            "aggregation_timeout": 180000,
            "aggregation_push": "slow",
            "pseudo_aggregation": {
                "enabled": False,
                "ema_alpha": 0.25
            }
        },
        "defense_args": {
            "reputation": {
                "enabled": False,
                "metrics": {},
                "initial_reputation": 0.2,
                "weighting_factor": "dynamic"
            }
        },
        "adversarial_args": {
            "attack_params": {
                "attacks": "No Attack"
            }
        },
        "tracking_args": {
            "enable_remote_tracking": False,
            "local_tracking": "basic",
            "log_dir": "./experiment_logs",
            "config_dir": "./experiment_config",
            "run_hash": ""
        },
        "mender_args": {
            "id": "",
            "mac": "",
            "device_type": ""
        },
        "message_args": {
            "max_local_messages": 10000,
            "compression": "zlib"
        },
        "reporter_args": {
            "grace_time_reporter": 10,
            "report_frequency": 5,
            "report_status_data_queue": True
        },
        "discoverer_args": {
            "grace_time_discovery": 0,
            "discovery_frequency": 10,
            "discovery_interval": 0.2
        },
        "health_args": {
            "grace_time_health": 60,
            "health_interval": 15,
            "send_alive_interval": 0.2,
            "check_alive_interval": 5,
            "alive_timeout": 120
        },
        "forwarder_args": {
            "forwarder_interval": 1,
            "forward_messages_interval": 0,
            "number_forwarded_messages": 100
        },
        "propagator_args": {
            "propagate_interval": 3,
            "propagate_model_interval": 0,
            "propagation_early_stop": 3,
            "history_size": 20
        },
        "misc_args": {
            "grace_time_connection": 10,
            "grace_time_start_federation": 10
        }
    }


# =============================================================================
# TOPOLOGY GENERATION
# =============================================================================

def generate_ring_neighbors(n_nodes):
    """
    Generate neighbor connections for a ring topology.
    Each node connects to its two neighbors in the ring.
    
    Returns a dict mapping node_idx -> list of neighbor addresses
    """
    neighbors = {}
    for i in range(n_nodes):
        # In a ring, each node connects to (i-1) and (i+1) mod n_nodes
        left = (i - 1) % n_nodes
        right = (i + 1) % n_nodes
        neighbors[i] = [left, right]
    return neighbors


# =============================================================================
# CONFIGURATION GENERATION
# =============================================================================

def generate_configs(config):
    """Generate configuration files for all nodes."""
    n_nodes = config["n_nodes"]
    base_port = config["base_port"]
    base_ip = config["base_ip"]
    config_dir = config["config_dir"]
    log_dir = config["log_dir"]
    
    # Create directories
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, config["scenario_name"]), exist_ok=True)
    
    # Generate ring topology neighbors
    neighbors_map = generate_ring_neighbors(n_nodes)
    
    # Generate timestamp
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    config_files = []
    
    for i in range(n_nodes):
        participant = get_participant_template()
        
        # Set unique port for each node
        port = base_port + i
        
        # Update scenario args
        participant["scenario_args"]["name"] = config["scenario_name"]
        participant["scenario_args"]["start_time"] = start_time
        participant["scenario_args"]["rounds"] = config["rounds"]
        participant["scenario_args"]["n_nodes"] = n_nodes
        participant["scenario_args"]["federation"] = config["federation"]
        
        # Update device args
        participant["device_args"]["idx"] = i
        participant["device_args"]["name"] = f"participant_{i}_{base_ip}_{port}"
        participant["device_args"]["role"] = "trainer_aggregator"
        participant["device_args"]["start"] = (i == 0)  # Only first node starts
        participant["device_args"]["accelerator"] = config["accelerator"]
        participant["device_args"]["gpu_id"] = config["gpu_id"]
        participant["device_args"]["logging"] = config["logging"]
        
        # Generate UID
        uid = hashlib.sha1(f"{base_ip}{port}{config['scenario_name']}".encode()).hexdigest()
        participant["device_args"]["uid"] = uid
        
        # Update network args
        participant["network_args"]["ip"] = base_ip
        participant["network_args"]["port"] = port
        participant["network_args"]["addr"] = f"{base_ip}:{port}"
        
        # Set neighbors based on ring topology
        neighbor_addrs = []
        for neighbor_idx in neighbors_map[i]:
            neighbor_port = base_port + neighbor_idx
            neighbor_addrs.append(f"{base_ip}:{neighbor_port}")
        participant["network_args"]["neighbors"] = " ".join(neighbor_addrs)
        
        # Update data args
        participant["data_args"]["dataset"] = config["dataset"]
        participant["data_args"]["iid"] = config["iid"]
        participant["data_args"]["partition_selection"] = config["partition_selection"]
        participant["data_args"]["partition_parameter"] = config["partition_parameter"]
        
        # Update model args
        participant["model_args"]["model"] = config["model"]
        
        # Update training args
        participant["training_args"]["epochs"] = config["epochs"]
        
        # Update aggregator args
        participant["aggregator_args"]["algorithm"] = config["aggregation_algorithm"]
        
        # Update adversarial args
        participant["adversarial_args"]["attack_params"]["attacks"] = config["attack"]
        
        # Update tracking args
        participant["tracking_args"]["log_dir"] = os.path.abspath(log_dir)
        participant["tracking_args"]["config_dir"] = os.path.abspath(config_dir)
        
        # Update mobility args
        participant["mobility_args"]["topology_type"] = config["topology"]
        
        # Save config file
        config_file = os.path.join(config_dir, f"participant_{i}.json")
        with open(config_file, "w") as f:
            json.dump(participant, f, indent=2)
        
        config_files.append(config_file)
        print(f"Generated config for node {i}: {config_file}")
        print(f"  - Port: {port}")
        print(f"  - Neighbors: {participant['network_args']['neighbors']}")
    
    # Save scenario summary
    scenario_summary = {
        "scenario_name": config["scenario_name"],
        "n_nodes": n_nodes,
        "topology": config["topology"],
        "dataset": config["dataset"],
        "model": config["model"],
        "rounds": config["rounds"],
        "epochs": config["epochs"],
        "partition_selection": config["partition_selection"],
        "partition_parameter": config["partition_parameter"],
        "aggregation_algorithm": config["aggregation_algorithm"],
        "start_time": start_time,
    }
    
    summary_file = os.path.join(config_dir, "scenario_summary.json")
    with open(summary_file, "w") as f:
        json.dump(scenario_summary, f, indent=2)
    
    print(f"\nScenario summary saved to: {summary_file}")
    
    return config_files


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(config_files, config):
    """Run the experiment by starting all nodes as processes."""
    processes = []
    
    print("\n" + "="*60)
    print("STARTING EXPERIMENT")
    print("="*60)
    print(f"Scenario: {config['scenario_name']}")
    print(f"Nodes: {config['n_nodes']}")
    print(f"Rounds: {config['rounds']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Model: {config['model']}")
    print("="*60 + "\n")
    
    # Start all nodes
    for i, config_file in enumerate(config_files):
        print(f"Starting node {i}...")
        
        # Start the node as a subprocess
        cmd = [sys.executable, "-m", "nebula.core.node", config_file]
        
        # Create log file for this node
        log_file = os.path.join(
            config["log_dir"], 
            config["scenario_name"], 
            f"node_{i}.log"
        )
        
        with open(log_file, "w") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
        
        processes.append(process)
        print(f"  Node {i} started with PID: {process.pid}")
        
        # Small delay between starting nodes
        time.sleep(1)
    
    print("\n" + "="*60)
    print("ALL NODES STARTED")
    print("="*60)
    print(f"Total processes: {len(processes)}")
    print("Press Ctrl+C to stop the experiment")
    print("="*60 + "\n")
    
    try:
        # Wait for all processes to complete
        for i, process in enumerate(processes):
            process.wait()
            print(f"Node {i} finished with return code: {process.returncode}")
    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal. Stopping all nodes...")
        for i, process in enumerate(processes):
            process.terminate()
            print(f"Terminated node {i}")
        
        # Wait for processes to terminate
        for process in processes:
            process.wait()
        
        print("All nodes stopped.")
    
    return processes


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("="*60)
    print("NEBULA DFL EXPERIMENT RUNNER")
    print("="*60)
    print("\nExperiment Configuration:")
    for key, value in EXPERIMENT_CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # Generate configuration files
    print("Generating configuration files...")
    config_files = generate_configs(EXPERIMENT_CONFIG)
    
    print(f"\nGenerated {len(config_files)} configuration files.")
    print(f"Config directory: {EXPERIMENT_CONFIG['config_dir']}")
    print(f"Log directory: {EXPERIMENT_CONFIG['log_dir']}")
    
    # Ask for confirmation before starting
    response = input("\nStart the experiment? (y/n): ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return
    
    # Run the experiment
    run_experiment(config_files, EXPERIMENT_CONFIG)


if __name__ == "__main__":
    main()
