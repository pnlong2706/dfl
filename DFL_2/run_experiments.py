#!/usr/bin/env python3
"""
Script to run DFL experiments consecutively via the NEBULA API.

This script:
1. Loads experiment JSON files from the experiments directory
2. Filters experiments with 0% Byzantine attack
3. Runs them consecutively using the NEBULA API

Usage:
    python3 run_experiments.py [--experiments-dir ./experiments] [--base-url http://localhost:18000]
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import List, Dict, Any, Optional

import requests
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


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
                data={"user": self.username, "password": self.password},
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
    
    def initialize_user_data(self) -> bool:
        """Initialize user data by visiting the dashboard."""
        try:
            response = self.session.get(
                f"{self.base_url}/platform/dashboard",
                allow_redirects=True
            )
            if response.status_code == 200:
                logger.info("User data initialized")
                return True
            else:
                logger.warning(f"Dashboard visit returned: {response.status_code}")
                return False
        except RequestException as e:
            logger.error(f"Error initializing user data: {e}")
            return False
    
    def deploy_scenario(self, scenario_data: dict) -> dict:
        """
        Deploy a scenario via the API.
        
        Args:
            scenario_data: Scenario configuration dictionary
            
        Returns:
            Response data with status
        """
        if not self.logged_in:
            if not self.login():
                return {"error": "Not logged in"}
        
        # Initialize user data by visiting dashboard first
        self.initialize_user_data()
        
        try:
            # The frontend expects a list of scenarios
            response = self.session.post(
                f"{self.base_url}/platform/dashboard/deployment/run",
                json=[scenario_data],
                headers={"Content-Type": "application/json"},
                allow_redirects=False
            )
            
            # Accept 200, 303, or 302 as success (redirect to dashboard is expected)
            if response.status_code in [200, 303, 302]:
                logger.info("Scenario deployment initiated")
                return {"status": "success", "message": "Scenario deployed"}
            else:
                logger.error(f"Deployment failed: {response.status_code} - {response.text}")
                return {"error": f"Deployment failed: {response.status_code}"}
                
        except RequestException as e:
            logger.error(f"Deployment error: {e}")
            return {"error": str(e)}
    
    def get_running_scenarios(self) -> List[dict]:
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
    
    def wait_for_scenario_completion(self, poll_interval: int = 30, timeout: int = 3600) -> bool:
        """
        Wait for the current scenario to complete.
        
        Args:
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            
        Returns:
            True if scenario completed, False if timeout
        """
        logger.info("Waiting for scenario to complete...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            running = self.get_running_scenarios()
            if not running:
                logger.info("Scenario completed")
                return True
            
            elapsed = int(time.time() - start_time)
            logger.info(f"Scenario still running... ({elapsed}s elapsed)")
            time.sleep(poll_interval)
        
        logger.warning("Timeout waiting for scenario completion")
        return False


def load_experiments(experiments_dir: str, byzantine_filter: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load experiment JSON files from directory.
    
    Args:
        experiments_dir: Path to experiments directory
        byzantine_filter: If set, only load experiments with this Byzantine percentage
        
    Returns:
        List of experiment data dictionaries
    """
    experiments = []
    
    if not os.path.exists(experiments_dir):
        logger.error(f"Experiments directory not found: {experiments_dir}")
        return experiments
    
    for filename in sorted(os.listdir(experiments_dir)):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(experiments_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # The JSON is a list with one scenario
            if isinstance(data, list) and len(data) > 0:
                scenario = data[0]
            else:
                scenario = data
            
            # Filter by Byzantine percentage if specified
            if byzantine_filter is not None:
                byzantine_pct = scenario.get('attack_params', {}).get('poisoned_node_percent', 0)
                if byzantine_pct != byzantine_filter:
                    continue
            
            experiments.append({
                'filename': filename,
                'filepath': filepath,
                'scenario': scenario,
                'title': scenario.get('scenario_title', 'Unknown'),
                'byzantine_pct': scenario.get('attack_params', {}).get('poisoned_node_percent', 0),
            })
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error loading {filename}: {e}")
    
    return experiments


def run_experiments_consecutively(
    client: NebulaClient,
    experiments: List[Dict[str, Any]],
    wait_for_completion: bool = True,
    poll_interval: int = 30,
    timeout: int = 3600,
    delay_between: int = 10,
) -> Dict[str, Any]:
    """
    Run experiments consecutively.
    
    Args:
        client: NebulaClient instance
        experiments: List of experiment dictionaries
        wait_for_completion: Whether to wait for each scenario to complete
        poll_interval: Seconds between status checks
        timeout: Maximum seconds to wait per scenario
        delay_between: Seconds to wait between scenarios
        
    Returns:
        Summary of results
    """
    results = {
        'total': len(experiments),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'details': [],
    }
    
    logger.info(f"Starting consecutive run of {len(experiments)} experiments")
    print()
    
    for i, exp in enumerate(experiments, 1):
        logger.info(f"[{i}/{len(experiments)}] Running: {exp['title']}")
        logger.info(f"    File: {exp['filename']}")
        logger.info(f"    Byzantine %: {exp['byzantine_pct']}")
        
        # Deploy scenario
        result = client.deploy_scenario(exp['scenario'])
        
        if 'error' in result:
            logger.error(f"    Failed: {result['error']}")
            results['failed'] += 1
            results['details'].append({
                'title': exp['title'],
                'filename': exp['filename'],
                'status': 'failed',
                'error': result['error'],
            })
            continue
        
        results['success'] += 1
        
        if wait_for_completion:
            # Wait for scenario to complete
            completed = client.wait_for_scenario_completion(poll_interval, timeout)
            
            if not completed:
                logger.warning("    Scenario did not complete within timeout")
                results['details'].append({
                    'title': exp['title'],
                    'filename': exp['filename'],
                    'status': 'timeout',
                })
            else:
                logger.info("    Scenario completed successfully")
                results['details'].append({
                    'title': exp['title'],
                    'filename': exp['filename'],
                    'status': 'completed',
                })
            
            # Wait before next scenario
            if i < len(experiments):
                logger.info(f"Waiting {delay_between}s before next scenario...")
                time.sleep(delay_between)
        
        print()
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run DFL experiments consecutively via NEBULA API")
    parser.add_argument(
        "--experiments-dir",
        default="./experiments",
        help="Directory containing experiment JSON files (default: ./experiments)"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:6060",
        help="NEBULA base URL (default: http://localhost:6060)"
    )
    parser.add_argument(
        "--username",
        default="admin",
        help="NEBULA username (default: admin)"
    )
    parser.add_argument(
        "--password",
        default="admin",
        help="NEBULA password (default: admin)"
    )
    parser.add_argument(
        "--byzantine",
        type=int,
        default=0,
        help="Filter experiments by Byzantine percentage (default: 0)"
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for scenarios to complete"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between status checks (default: 30)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Maximum seconds to wait per scenario (default: 3600)"
    )
    parser.add_argument(
        "--delay-between",
        type=int,
        default=10,
        help="Seconds to wait between scenarios (default: 10)"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list experiments without running them"
    )
    
    args = parser.parse_args()
    
    # Get absolute path to experiments directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(script_dir, args.experiments_dir)
    
    print("=" * 60)
    print("NEBULA DFL EXPERIMENT RUNNER")
    print("=" * 60)
    print()
    
    # Load experiments
    logger.info(f"Loading experiments from: {experiments_dir}")
    logger.info(f"Filtering by Byzantine percentage: {args.byzantine}%")
    print()
    
    experiments = load_experiments(experiments_dir, args.byzantine)
    
    if not experiments:
        logger.error("No experiments found matching criteria")
        sys.exit(1)
    
    logger.info(f"Found {len(experiments)} experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['title']} (Byzantine: {exp['byzantine_pct']}%)")
    print()
    
    # If list-only, exit here
    if args.list_only:
        logger.info("List-only mode, exiting")
        return
    
    # Initialize client
    client = NebulaClient(args.base_url, args.username, args.password)
    
    # Wait for service
    if not client.wait_for_service(timeout=60):
        logger.error("NEBULA service not available")
        sys.exit(1)
    
    # Login
    if not client.login():
        logger.error("Failed to login to NEBULA")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("STARTING EXPERIMENTS")
    print("=" * 60)
    print()
    
    # Run experiments
    results = run_experiments_consecutively(
        client,
        experiments,
        wait_for_completion=not args.no_wait,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        delay_between=args.delay_between,
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {results['total']}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    print()
    
    if results['details']:
        print("Details:")
        for detail in results['details']:
            status_icon = "✓" if detail['status'] == 'completed' else "✗"
            print(f"  {status_icon} {detail['title']}: {detail['status']}")


if __name__ == "__main__":
    main()
