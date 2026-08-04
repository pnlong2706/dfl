#!/usr/bin/env python3
"""
Script to run DFL experiments consecutively via the NEBULA API.

This script:
1. Loads experiment JSON files from the experiments directory
2. Filters experiments with 0% Byzantine attack
3. Skips experiments that are already in the completed list
4. Runs them consecutively using the NEBULA API
5. Updates the completed list after each successful run

Usage:
    python3 run_experiments.py [--experiments-dir ./experiments] [--base-url http://localhost:18000]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import List, Dict, Any, Optional, Set

try:
    import requests
    from requests.exceptions import RequestException
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Default completed experiments file
COMPLETED_FILE = "completed_experiments.txt"


def load_completed_experiments(completed_file: str) -> Set[str]:
    """
    Load the set of completed experiment filenames.
    
    Args:
        completed_file: Path to the completed experiments file
        
    Returns:
        Set of completed experiment filenames
    """
    completed = set()
    
    if not os.path.exists(completed_file):
        return completed
    
    with open(completed_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            completed.add(line)
    
    return completed


def save_completed_experiment(completed_file: str, filename: str) -> None:
    """
    Append a completed experiment to the completed file.
    
    Args:
        completed_file: Path to the completed experiments file
        filename: Experiment filename to add
    """
    with open(completed_file, 'a') as f:
        f.write(f"{filename}\n")
    logger.info(f"Added {filename} to completed list")


def load_experiments(
    experiments_dir: str, 
    byzantine_filter: Optional[int] = None,
    completed: Set[str] = None,
    skip_completed: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load experiment JSON files from directory.
    
    Args:
        experiments_dir: Path to experiments directory
        byzantine_filter: If set, only load experiments with this Byzantine percentage
        completed: Set of completed experiment filenames to skip
        skip_completed: Whether to skip completed experiments
        
    Returns:
        List of experiment data dictionaries
    """
    experiments = []
    skipped = []
    
    if completed is None:
        completed = set()
    
    if not os.path.exists(experiments_dir):
        logger.error(f"Experiments directory not found: {experiments_dir}")
        return experiments
    
    for filename in sorted(os.listdir(experiments_dir)):
        if not filename.endswith('.json'):
            continue
        
        # Skip completed experiments
        if skip_completed and filename in completed:
            skipped.append(filename)
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
    
    if skipped:
        logger.info(f"Skipping {len(skipped)} completed experiments")
        for f in skipped:
            logger.debug(f"  Skipped: {f}")
    
    return experiments


class NebulaClient:
    """Client for interacting with the NEBULA API."""
    
    def __init__(self, base_url: str, username: str = "admin", password: str = "admin"):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.logged_in = False
        self._current_scenario_name = None
    
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
    
    @staticmethod
    def count_node_containers() -> int:
        """
        Count running node (participant) containers via the Docker CLI.

        This is the robust completion signal: it is independent of whether every
        node successfully POSTs /node/done to the frontend (a single missed
        report makes the frontend's runningscenario status hang forever).
        Returns -1 if docker is unavailable so callers can fall back to the API.
        """
        try:
            out = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=15,
            )
            if out.returncode != 0:
                return -1
            return sum(1 for n in out.stdout.splitlines() if "participant" in n)
        except (FileNotFoundError, subprocess.SubprocessError):
            return -1

    def is_scenario_active(self) -> bool:
        """
        A scenario is active if the frontend reports it running OR its node
        containers are still up. Combining both closes the race where the API
        has not yet flipped to 'running' and the gap where nodes are up but the
        API already reports 'not running'.
        """
        if self.get_running_scenarios():
            return True
        return self.count_node_containers() > 0

    def get_running_scenario_name(self) -> Optional[str]:
        """Return the name of the currently running scenario, or None."""
        running = self.get_running_scenarios()
        if running:
            return running[0].get("name")
        return None

    def wait_for_scenario_start(self, startup_timeout: int = 600, poll: int = 5) -> bool:
        """
        Block until the scenario has actually started (node containers exist or
        the API reports running). Without this, deployment is asynchronous and
        an immediate completion check races the launch, so the runner declares
        the scenario 'complete' in milliseconds and stampedes the queue.

        Captures the running scenario name in self._current_scenario_name so it
        can be finalized later even if the API stops reporting it.
        """
        logger.info("Waiting for scenario to start...")
        start = time.time()
        while time.time() - start < startup_timeout:
            name = self.get_running_scenario_name()
            if name:
                self._current_scenario_name = name
            if self.is_scenario_active():
                logger.info(f"Scenario is running (startup took {int(time.time() - start)}s)")
                return True
            time.sleep(poll)
        logger.warning(f"Scenario never started within {startup_timeout}s")
        return False

    def finalize_scenario(self, scenario_name: str = None, idle_timeout: int = 120, grace: int = 30) -> None:
        """
        Ensure the frontend queue advances after a scenario ends.

        Preferred (clean) path: when every node reports /node/done, the frontend
        both releases its queue AND marks the scenario 'completed' (green). We wait
        briefly for that. Only if the frontend is still stuck on 'running' after the
        grace period (a node crashed / never reported done, so the queue is wedged)
        do we force-stop via the stop endpoint — which sets 'finished' (red), the
        correct signal for a run that did not end cleanly.
        """
        scenario_name = scenario_name or self._current_scenario_name

        # Give the natural completion path a chance to release the queue.
        start = time.time()
        while time.time() - start < grace:
            if not self.get_running_scenarios():
                logger.info("Scenario released the queue on its own (marked completed)")
                self._current_scenario_name = None
                return
            time.sleep(3)

        # Still 'running' -> wedged. Force-stop to unblock the queue.
        if scenario_name:
            logger.info(f"Scenario {scenario_name} still 'running' after end; force-stopping to release the queue")
            self.stop_scenario(scenario_name, stop_all=False)
        # Wait until the frontend is idle again before the next deploy.
        start = time.time()
        while time.time() - start < idle_timeout:
            if not self.is_scenario_active():
                break
            time.sleep(3)
        self._current_scenario_name = None

    def wait_for_scenario_completion(
        self,
        poll_interval: int = 30,
        timeout: int = 3600,
        startup_timeout: int = 600,
        scenario_name: str = None,
    ) -> bool:
        """
        Wait for the current scenario to complete.

        Phase 1: wait until it actually starts (fixes the deploy/poll race).
        Phase 2: wait until neither the API nor any node containers report it
                 active. On timeout, force-stop and clean up so one stuck run
                 cannot block an unattended overnight queue.

        Returns:
            True if the scenario completed on its own, False if it never started
            or had to be force-stopped on timeout.
        """
        # Phase 1 — must be running before we can meaningfully wait for the end.
        if not self.wait_for_scenario_start(startup_timeout=startup_timeout):
            return False

        # Phase 2 — wait for genuine completion.
        # Completion is judged primarily by the node-container lifecycle: once
        # containers have appeared, the run is done when they have all exited.
        # This is authoritative even if the frontend API is stuck reporting
        # 'running' (which happens whenever a single node misses its /node/done
        # report). The API is only a fallback when docker is unavailable or no
        # container ever appears.
        logger.info("Waiting for scenario to complete...")
        start_time = time.time()
        containers_seen = False
        while time.time() - start_time < timeout:
            n = self.count_node_containers()
            if n > 0:
                containers_seen = True
            elapsed = int(time.time() - start_time)

            if containers_seen and n == 0:
                logger.info(f"Scenario completed (node containers exited after {elapsed}s)")
                self.finalize_scenario(scenario_name)
                return True
            if n < 0 and not self.get_running_scenarios():
                # docker unavailable -> fall back to API-only signal
                logger.info("Scenario completed (API reports not running)")
                self.finalize_scenario(scenario_name)
                return True

            logger.info(f"Scenario still running... ({elapsed}s elapsed, {max(n, 0)} node containers up)")
            time.sleep(poll_interval)

        # Timed out — force-stop so the queue can proceed.
        logger.warning(f"Timeout after {timeout}s; force-stopping scenario to unblock the queue")
        self.finalize_scenario(scenario_name)
        return False


def run_experiments_consecutively(
    client: NebulaClient,
    experiments: List[Dict[str, Any]],
    completed_file: str,
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
        completed_file: Path to completed experiments file
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
                
                # Add to completed file
                save_completed_experiment(completed_file, exp['filename'])
            
            # Wait before next scenario
            if i < len(experiments):
                logger.info(f"Waiting {delay_between}s before next scenario...")
                time.sleep(delay_between)
        else:
            # If not waiting, mark as success immediately
            save_completed_experiment(completed_file, exp['filename'])
            results['details'].append({
                'title': exp['title'],
                'filename': exp['filename'],
                'status': 'deployed',
            })
        
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
        "--completed-file",
        default=None,
        help="File to track completed experiments (default: completed_experiments.txt)"
    )
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="Include experiments that are already in the completed list"
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
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(script_dir, args.experiments_dir)
    
    if args.completed_file:
        completed_file = args.completed_file
    else:
        completed_file = os.path.join(script_dir, COMPLETED_FILE)
    
    print("=" * 60)
    print("NEBULA DFL EXPERIMENT RUNNER")
    print("=" * 60)
    print()
    
    # Load completed experiments
    logger.info(f"Loading completed experiments from: {completed_file}")
    completed = load_completed_experiments(completed_file)
    logger.info(f"Found {len(completed)} completed experiments")
    print()
    
    # Load experiments
    logger.info(f"Loading experiments from: {experiments_dir}")
    logger.info(f"Filtering by Byzantine percentage: {args.byzantine}%")
    print()
    
    experiments = load_experiments(
        experiments_dir, 
        args.byzantine,
        completed,
        skip_completed=not args.include_completed,
    )
    
    if not experiments:
        logger.info("No experiments found matching criteria")
        if completed and not args.include_completed:
            logger.info(f"  ({len(completed)} experiments already completed)")
        return
    
    logger.info(f"Found {len(experiments)} experiments to run:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['title']} (Byzantine: {exp['byzantine_pct']}%)")
    print()
    
    # If list-only, exit here
    if args.list_only:
        logger.info("List-only mode, exiting")
        return
    
    # Check if requests is available
    if not REQUESTS_AVAILABLE:
        logger.error("The 'requests' module is required. Install it with: pip install requests")
        sys.exit(1)
    
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
        completed_file,
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
            status_icon = "✓" if detail['status'] in ['completed', 'deployed'] else "✗"
            print(f"  {status_icon} {detail['title']}: {detail['status']}")


if __name__ == "__main__":
    main()
