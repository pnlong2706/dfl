#!/usr/bin/env python3
"""
Script to run DFL experiments via the NEBULA UI/API.

This script:
1. Starts the NEBULA platform (frontend + controller)
2. Uses the API to deploy experiments with custom configurations
3. Experiments are tracked in the database automatically

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
import logging
import os
import sys
import time
import subprocess
import signal
from datetime import datetime

import requests
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Default ports
CONTROLLER_PORT = 5050
FRONTEND_PORT = 6060

# Experiment configuration based on TSV first experiment
EXPERIMENT_CONFIG = {
    # Scenario metadata
    "scenario_title": "CIFAR10 Dir(0.3) Ring DFedAvg",
    "scenario_description": "DFL experiment with CIFAR10, Dirichlet(0.3) distribution, Ring topology, FedAvg aggregation",
    
    # Deployment settings
    "deployment": "process",  # "docker", "process", or "physical"
    "federation": "DFL",  # Decentralized Federated Learning
    
    # Topology
    "topology": "Ring",
    "n_nodes": 20,
    
    # Dataset settings
    "dataset": "CIFAR10",
    "iid": False,
    "partition_selection": "dirichlet",
    "partition_parameter": 0.3,
    
    # Model settings
    "model": "SimpleNet",
    
    # Training settings
    "rounds": 100,
    "epochs": 2,
    
    # Aggregation
    "agg_algorithm": "FedAvg",
    
    # Attack settings (None for this experiment)
    "attack_params": {
        "attacks": "No Attack",
        "poisoned_node_percent": 0,
        "poisoned_sample_percent": 0,
        "poisoned_noise_percent": 0
    },
    
    # Defense settings (None for this experiment)
    "reputation": {"enabled": False},
    
    # Device settings
    "accelerator": "gpu",  # Change to "cpu" if no GPU available
    "gpu_id": [0],  # GPU ID(s) to use
    
    # Logging
    "logginglevel": True,
    
    # Network simulation
    "network_simulation": False,
    
    # Mobility
    "mobility": False,
    "random_geo": False,
    
    # Trustworthiness
    "with_trustworthiness": False,
    
    # Situational Awareness
    "with_sa": False,
}


# =============================================================================
# API CLIENT
# =============================================================================

class NebulaClient:
    """Client for interacting with the NEBULA API."""
    
    def __init__(self, base_url: str, username: str = "admin", password: str = "admin"):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.logged_in = False
    
    def login(self) -> bool:
        """Login to NEBULA and establish session."""
        try:
            response = self.session.post(
                f"{self.base_url}/platform/login",
                data={"username": self.username, "password": self.password},
                allow_redirects=False
            )
            if response.status_code in [200, 303, 302]:
                self.logged_in = True
                logger.info(f"Logged in as {self.username}")
                return True
            else:
                logger.error(f"Login failed: {response.status_code}")
                return False
        except RequestException as e:
            logger.error(f"Login error: {e}")
            return False
    
    def wait_for_service(self, timeout: int = 60) -> bool:
        """Wait for the NEBULA service to be available."""
        logger.info(f"Waiting for NEBULA service at {self.base_url}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.base_url}/platform", timeout=5)
                if response.status_code == 200:
                    logger.info("NEBULA service is available")
                    return True
            except RequestException:
                pass
            time.sleep(2)
        
        logger.error("Timeout waiting for NEBULA service")
        return False
    
    def deploy_scenario(self, scenario_data: dict) -> dict:
        """
        Deploy a scenario via the API.
        
        Args:
            scenario_data: Scenario configuration dictionary
            
        Returns:
            Response data with scenario name
        """
        if not self.logged_in:
            if not self.login():
                return {"error": "Not logged in"}
        
        try:
            # The frontend expects a list of scenarios
            response = self.session.post(
                f"{self.base_url}/platform/dashboard/deployment/run",
                json=[scenario_data],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info("Scenario deployment initiated")
                return {"status": "success", "message": "Scenario deployed"}
            else:
                logger.error(f"Deployment failed: {response.status_code} - {response.text}")
                return {"error": f"Deployment failed: {response.status_code}"}
                
        except RequestException as e:
            logger.error(f"Deployment error: {e}")
            return {"error": str(e)}
    
    def get_running_scenarios(self) -> list:
        """Get list of running scenarios."""
        try:
            response = self.session.get(
                f"{self.base_url}/platform/api/dashboard/runningscenario"
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("scenario_status") == "running":
                    return [data]
            return []
        except RequestException as e:
            logger.error(f"Error getting running scenarios: {e}")
            return []
    
    def get_scenarios(self) -> list:
        """Get list of all scenarios."""
        try:
            response = self.session.get(f"{self.base_url}/platform/api/dashboard")
            if response.status_code == 200:
                return response.json()
            return []
        except RequestException as e:
            logger.error(f"Error getting scenarios: {e}")
            return []
    
    def stop_scenario(self, scenario_name: str, stop_all: bool = False) -> bool:
        """Stop a running scenario."""
        try:
            response = self.session.get(
                f"{self.base_url}/platform/dashboard/{scenario_name}/stop/{stop_all}"
            )
            return response.status_code in [200, 303, 302]
        except RequestException as e:
            logger.error(f"Error stopping scenario: {e}")
            return False


# =============================================================================
# PLATFORM MANAGER
# =============================================================================

class NebulaPlatform:
    """Manages the NEBULA platform lifecycle."""
    
    def __init__(self, script_path: str = "script/run_dfl.sh", username: str = "admin"):
        self.script_path = script_path
        self.username = username
        self.process = None
        self.client = None
    
    def start(self) -> bool:
        """Start the NEBULA platform."""
        logger.info("Starting NEBULA platform...")
        
        # Check if script exists
        if not os.path.exists(self.script_path):
            logger.error(f"Script not found: {self.script_path}")
            logger.info("Please run: make install")
            return False
        
        try:
            # Start the platform using the script
            self.process = subprocess.Popen(
                ["bash", self.script_path, self.username],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                cwd=os.getcwd()
            )
            
            logger.info(f"NEBULA process started (PID: {self.process.pid})")
            
            # Initialize API client
            self.client = NebulaClient(
                f"http://localhost:{FRONTEND_PORT}",
                username=self.username
            )
            
            # Wait for service to be available
            if self.client.wait_for_service(timeout=120):
                logger.info(f"NEBULA UI available at: http://localhost:{FRONTEND_PORT}")
                return True
            else:
                logger.error("Failed to start NEBULA platform")
                self.stop()
                return False
                
        except Exception as e:
            logger.error(f"Error starting platform: {e}")
            return False
    
    def stop(self):
        """Stop the NEBULA platform."""
        if self.process:
            logger.info("Stopping NEBULA platform...")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except Exception as e:
                logger.warning(f"Error stopping process: {e}")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.process = None
            logger.info("NEBULA platform stopped")
    
    def deploy_experiment(self, config: dict) -> dict:
        """Deploy an experiment via the API."""
        if not self.client:
            logger.error("Platform not started")
            return {"error": "Platform not started"}
        
        # Login if needed
        if not self.client.logged_in:
            if not self.client.login():
                return {"error": "Login failed"}
        
        return self.client.deploy_scenario(config)


# =============================================================================
# EXPERIMENT BUILDER
# =============================================================================

def build_scenario_from_config(config: dict) -> dict:
    """
    Build a complete scenario configuration from a simplified config.
    
    This creates the full scenario data structure expected by the NEBULA frontend.
    """
    scenario = {
        # Metadata
        "scenario_title": config.get("scenario_title", "Experiment"),
        "scenario_description": config.get("scenario_description", ""),
        
        # Deployment
        "deployment": config.get("deployment", "process"),
        "federation": config.get("federation", "DFL"),
        
        # Topology
        "topology": config.get("topology", "Ring"),
        "n_nodes": config.get("n_nodes", 20),
        "nodes": {},
        "nodes_graph": {},
        "matrix": [],
        
        # Dataset
        "dataset": config.get("dataset", "CIFAR10"),
        "iid": config.get("iid", False),
        "partition_selection": config.get("partition_selection", "dirichlet"),
        "partition_parameter": config.get("partition_parameter", 0.3),
        
        # Model
        "model": config.get("model", "SimpleNet"),
        
        # Training
        "rounds": config.get("rounds", 100),
        "epochs": config.get("epochs", 2),
        
        # Aggregation
        "agg_algorithm": config.get("agg_algorithm", "FedAvg"),
        "pseudo_aggregation": {"enabled": False, "ema_alpha": 0.25},
        "fedsam": {"enabled": False, "rho": 0.5},
        "pcr": {"enabled": False, "mu": 0.01, "apply_mode": "pseudo_only"},
        "mid_round_test": False,
        
        # Attack
        "attack_params": config.get("attack_params", {"attacks": "No Attack"}),
        
        # Defense
        "reputation": config.get("reputation", {"enabled": False}),
        
        # Device
        "accelerator": config.get("accelerator", "gpu"),
        "gpu_id": config.get("gpu_id", [0]),
        "logginglevel": config.get("logginglevel", True),
        "report_status_data_queue": True,
        
        # Network
        "network_simulation": config.get("network_simulation", False),
        "network_subnet": "",
        "network_gateway": "",
        
        # Mobility
        "mobility": config.get("mobility", False),
        "random_geo": config.get("random_geo", False),
        "latitude": "",
        "longitude": "",
        "mobility_type": "",
        "radius_federation": 1000,
        "scheme_mobility": "random",
        "round_frequency": 1,
        "mobile_participants_percent": 0,
        "additional_participants": [],
        "schema_additional_participants": "random",
        
        # Trustworthiness
        "with_trustworthiness": config.get("with_trustworthiness", False),
        "robustness_pillar": {},
        "resilience_to_attacks": {},
        "algorithm_robustness": {},
        "client_reliability": {},
        "privacy_pillar": {},
        "technique": {},
        "uncertainty": {},
        "indistinguishability": {},
        "fairness_pillar": {},
        "selection_fairness": {},
        "performance_fairness": {},
        "class_distribution": {},
        "explainability_pillar": {},
        "interpretability": {},
        "post_hoc_methods": {},
        "accountability_pillar": {},
        "factsheet_completeness": {},
        "architectural_soundness_pillar": {},
        "client_management": {},
        "optimization": {},
        "sustainability_pillar": {},
        "energy_source": {},
        "hardware_efficiency": {},
        "federation_complexity": {},
        
        # Situational Awareness
        "with_sa": config.get("with_sa", False),
        "strict_topology": True,
        "random_topology_probability": 0.2,
        "sad_candidate_selector": "ring",
        "sad_model_handler": "std",
        "sar_arbitration_policy": "static",
        "sar_neighbor_policy": "idle",
        "sar_training": False,
        "sar_training_policy": "idle",
    }
    
    return scenario


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DFL experiments via NEBULA UI")
    parser.add_argument("--username", default="admin", help="NEBULA username")
    parser.add_argument("--port", type=int, default=FRONTEND_PORT, help="Frontend port")
    parser.add_argument("--no-start", action="store_true", help="Don't start platform (assume already running)")
    parser.add_argument("--stop-after", action="store_true", help="Stop platform after experiment")
    parser.add_argument("--wait", action="store_true", help="Wait for experiment to complete")
    
    args = parser.parse_args()
    
    print("="*60)
    print("NEBULA DFL EXPERIMENT RUNNER (UI/API)")
    print("="*60)
    print("\nExperiment Configuration:")
    for key, value in EXPERIMENT_CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize platform
    platform = NebulaPlatform(username=args.username)
    
    try:
        if not args.no_start:
            # Start the platform
            if not platform.start():
                logger.error("Failed to start NEBULA platform")
                sys.exit(1)
        else:
            # Connect to existing platform
            platform.client = NebulaClient(f"http://localhost:{args.port}", username=args.username)
            if not platform.client.wait_for_service(timeout=30):
                logger.error("NEBULA platform not available. Start it first or remove --no-start")
                sys.exit(1)
        
        # Build scenario
        scenario = build_scenario_from_config(EXPERIMENT_CONFIG)
        
        # Deploy experiment
        logger.info("Deploying experiment...")
        result = platform.deploy_experiment(scenario)
        
        if "error" in result:
            logger.error(f"Failed to deploy experiment: {result['error']}")
            sys.exit(1)
        
        logger.info("Experiment deployed successfully!")
        logger.info(f"UI available at: http://localhost:{FRONTEND_PORT}/platform/dashboard")
        
        if args.wait:
            logger.info("Waiting for experiment to complete...")
            logger.info("Press Ctrl+C to stop waiting")
            
            try:
                while True:
                    running = platform.client.get_running_scenarios()
                    if not running:
                        logger.info("Experiment completed")
                        break
                    time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Stopped waiting")
        
        if not args.stop_after and not args.wait:
            logger.info("Experiment is running. Press Ctrl+C to stop the platform")
            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        if args.stop_after:
            platform.stop()
        else:
            logger.info(f"Platform still running at http://localhost:{FRONTEND_PORT}")
            logger.info("Run 'python app/main.py --stop' to stop it later")


if __name__ == "__main__":
    main()
