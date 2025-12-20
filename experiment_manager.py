"""
Centralized Experiment Manager

Coordinates task distribution to multiple GPU worker processes.
Monitors experiment completion via CSV files and assigns tasks dynamically.
"""

import socket
import json
import threading
import time
import os
import glob
import yaml
import itertools
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Set
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_manager.log'),
        logging.StreamHandler()
    ]
)

class ExperimentManager:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.tasks: Dict[str, dict] = {}
        self.completed_tasks: Set[str] = set()
        self.assigned_tasks: Dict[str, dict] = {}  # task_id -> worker info
        self.lock = threading.Lock()
        self.running = True
        self.state_file = 'manager_state.json'  # State persistence file
        
    def load_all_configs(self) -> List[dict]:
        """Load all experiment config files"""
        config_files = glob.glob('configs/config_exp*.yaml')
        configs = []
        
        for config_path in sorted(config_files):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    config['_path'] = config_path
                    configs.append(config)
                    logging.info(f"Loaded config: {config_path}")
            except Exception as e:
                logging.error(f"Error loading {config_path}: {e}")
        
        return configs
    
    def generate_experiments(self, phase_config: dict, defaults: dict) -> List[dict]:
        """Generate experiment configurations from phase config"""
        vary_params = {}
        for item in phase_config['combinations']:
            vary_params.update(item)
        
        keys = list(vary_params.keys())
        vals = list(vary_params.values())
        
        experiments = []
        for instance in itertools.product(*vals):
            exp_setup = defaults.copy()
            
            # Handle dataset
            if 'dataset' not in vary_params:
                exp_setup['dataset'] = phase_config.get('dataset', 'mnist')
            
            exp_setup['phase_name'] = phase_config.get('phase_name', 'Unknown')
            
            for k, v in zip(keys, instance):
                exp_setup[k] = v
            
            # Resolve {dataset} placeholder
            if 'output_dir' in exp_setup and '{dataset}' in exp_setup['output_dir']:
                exp_setup['output_dir'] = exp_setup['output_dir'].format(dataset=exp_setup['dataset'])
            
            experiments.append(exp_setup)
        
        return experiments
    
    def generate_all_tasks(self):
        """Generate all tasks from all config files"""
        configs = self.load_all_configs()
        
        phase_names = [
            'exp1_vary_width',
            'exp4_iid_vs_noniid',
            'exp5_defense_comparison',
            'exp2_mechanism_analysis',
            'exp3_attack_types',
        ]
        
        task_counter = 0
        task_signatures = {}  # signature -> task_id mapping for deduplication
        duplicates_skipped = 0
        
        for config in configs:
            defaults = config['defaults']
            
            
            for phase_name in phase_names:
                if phase_name not in config:
                    continue
                
                phase_cfg = config[phase_name]
                phase_cfg['phase_name'] = phase_name
                
                experiments = self.generate_experiments(phase_cfg, defaults)
                
                for exp in experiments:
                    # Seed is now part of the experiment config from combinations
                    seed = exp.get('seed', 42)  # Fallback to 42 if not specified
                    
                    # Create task signature to detect duplicates
                    signature = self._create_task_signature(exp)
                    
                    # Check if this exact experiment already exists
                    if signature in task_signatures:
                        duplicates_skipped += 1
                        existing_task_id = task_signatures[signature]
                        
                        # Add this output directory to the existing task
                        if exp['output_dir'] not in self.tasks[existing_task_id]['output_dirs']:
                            self.tasks[existing_task_id]['output_dirs'].append(exp['output_dir'])
                        
                        logging.debug(
                            f"Skipping duplicate: {phase_name} seed={seed} "
                            f"(already exists as {existing_task_id}), "
                            f"added output_dir: {exp['output_dir']}"
                        )
                        continue
                    
                    # Create unique task
                    task_id = f"{phase_name}_{task_counter}_{seed}"
                    
                    self.tasks[task_id] = {
                        'task_id': task_id,
                        'phase': phase_name,
                        'config': exp,
                        'seed': seed,
                        'output_dir': exp['output_dir'],  # Primary output dir
                        'output_dirs': [exp['output_dir']],  # All output dirs (for duplicates)
                        'status': 'pending',
                        'assigned_to': None,
                        'assigned_at': None,
                        'retry_count': 0,          # Track retry attempts
                        'last_error': None         # Track last error message
                    }
                    
                    # Register this signature
                    task_signatures[signature] = task_id
                    task_counter += 1
        
        logging.info(f"Generated {len(self.tasks)} unique tasks")
        if duplicates_skipped > 0:
            logging.info(f"Skipped {duplicates_skipped} duplicate tasks across configs")
    
    def load_completed_tasks(self):
        """Load completed tasks from centralized CSV"""
        centralized_csv = 'final_results.csv'
        
        if not os.path.exists(centralized_csv):
            logging.info("No centralized final_results.csv found - starting fresh")
            return
        
        try:
            df = pd.read_csv(centralized_csv)
            
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                sig = self._create_task_signature(row_dict)
                self.completed_tasks.add(sig)
            
            logging.info(f"Loaded {len(df)} completed experiments from {centralized_csv}")
        except Exception as e:
            logging.error(f"Error loading {centralized_csv}: {e}")
        
        # Mark matching tasks as complete
        with self.lock:
            for task_id, task in self.tasks.items():
                sig = self._create_task_signature(task['config'])
                if sig in self.completed_tasks:
                    task['status'] = 'complete'
        
        completed_count = sum(1 for t in self.tasks.values() if t['status'] == 'complete')
        logging.info(f"Marked {completed_count} tasks as already completed")
    
    def _create_task_signature(self, config: dict) -> str:
        """Create unique signature for a task based on its parameters"""
        key_params = [
            config.get('phase', config.get('phase_name', '')),
            config.get('dataset', ''),
            config.get('model_type', 'cnn'),        # Added model_type
            str(config.get('width_factor', '')),
            str(config.get('depth', '')),
            str(config.get('poison_ratio', '')),
            config.get('poison_type', 'label_flip'),
            str(config.get('alpha', '')),
            config.get('data_ordering', 'shuffle'),
            config.get('aggregator', 'fedavg'),
            str(config.get('batch_size', 64)),
            str(config.get('seed', 42))  # Include seed for deduplication
        ]
        return '|'.join(key_params)
    
    def _create_result_signature(self, config: dict) -> str:
        """Create result signature (without phase) for CSV display"""
        key_params = [
            config.get('dataset', ''),
            config.get('model_type', 'cnn'),
            str(config.get('width_factor', '')),
            str(config.get('depth', '')),
            str(config.get('poison_ratio', '')),
            config.get('poison_type', 'label_flip'),
            str(config.get('alpha', '')),
            config.get('data_ordering', 'shuffle'),
            config.get('aggregator', 'fedavg'),
            str(config.get('batch_size', 64)),
            str(config.get('seed', 42))
        ]
        return '|'.join(key_params)
    
    def save_state(self):
        """Save manager state to file for crash recovery"""
        with self.lock:
            # Convert tasks dict to serializable format
            state = {
                'tasks': {},
                'assigned_tasks': {}
            }
            
            for task_id, task in self.tasks.items():
                # Copy task and convert datetime to string
                task_copy = task.copy()
                if 'assigned_at' in task_copy and task_copy['assigned_at'] is not None:
                    task_copy['assigned_at'] = task_copy['assigned_at'].isoformat()
                state['tasks'][task_id] = task_copy
            
            for task_id, task in self.assigned_tasks.items():
                task_copy = task.copy()
                if 'assigned_at' in task_copy and task_copy['assigned_at'] is not None:
                    task_copy['assigned_at'] = task_copy['assigned_at'].isoformat()
                state['assigned_tasks'][task_id] = task_copy
            
            try:
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                logging.info(f"Saved state: {len(self.tasks)} tasks")
            except Exception as e:
                logging.error(f"Error saving state: {e}")
        # print status complete count / all tasks
        complete_count = sum(1 for t in self.tasks.values() if t['status'] == 'complete')
        logging.info(f"{complete_count}/{len(self.tasks)} tasks are complete")  
    
    def load_state(self):
        """Load manager state from file after restart"""
        if not os.path.exists(self.state_file):
            logging.info("No previous state file found - starting fresh")
            return False
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Restore tasks
            for task_id, task in state.get('tasks', {}).items():
                # Convert datetime strings back
                if 'assigned_at' in task and task['assigned_at'] is not None:
                    task['assigned_at'] = datetime.fromisoformat(task['assigned_at'])
                self.tasks[task_id] = task
            
            # Restore assigned tasks
            for task_id, task in state.get('assigned_tasks', {}).items():
                if 'assigned_at' in task and task['assigned_at'] is not None:
                    task['assigned_at'] = datetime.fromisoformat(task['assigned_at'])
                self.assigned_tasks[task_id] = task
            
            logging.info(f"Loaded state: {len(self.tasks)} tasks, {len(self.assigned_tasks)} assigned")
            
            # Check for stale assignments (workers might have died during manager downtime)
            self.check_stale_assignments()
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading state: {e}")
            return False
    
    def check_stale_assignments(self):
        """Check if assigned workers are still alive"""
        current_time = datetime.now()
        stale_tasks = []
        
        for task_id, task in list(self.assigned_tasks.items()):
            assigned_at = task.get('assigned_at')
            if assigned_at is None:
                stale_tasks.append(task_id)
                continue
            
            # If assigned more than 2 hours ago, consider stale
            elapsed = (current_time - assigned_at).total_seconds()
            if elapsed > 7200:  # 2 hours
                stale_tasks.append(task_id)
        
        # Reassign stale tasks
        for task_id in stale_tasks:
            logging.warning(f"Reassigning stale task {task_id}")
            self.tasks[task_id]['status'] = 'pending'
            self.tasks[task_id]['assigned_to'] = None
            if task_id in self.assigned_tasks:
                del self.assigned_tasks[task_id]
        
        if stale_tasks:
            logging.info(f"Reassigned {len(stale_tasks)} stale tasks")
    
    def refresh_completed(self):
        """Periodically refresh completed tasks from centralized CSV"""
        while self.running:
            time.sleep(5)  # Check every 5 seconds
            
            centralized_csv = 'final_results.csv'
            new_completed = set()
            
            if not os.path.exists(centralized_csv):
                continue
            
            try:
                df = pd.read_csv(centralized_csv)
                for _, row in df.iterrows():
                    row_dict = row.to_dict()
                    sig = self._create_task_signature(row_dict)
                    new_completed.add(sig)
            except Exception as e:
                logging.error(f"Error refreshing {centralized_csv}: {e}")
            
            # Update tasks
            with self.lock:
                newly_completed = new_completed - self.completed_tasks
                if newly_completed:
                    for task_id, task in self.tasks.items():
                        sig = self._create_task_signature(task['config'])
                        if sig in newly_completed and task['status'] != 'complete':
                            task['status'] = 'complete'
                            logging.info(f"Task {task_id} marked as complete via CSV refresh")
                
                self.completed_tasks.update(newly_completed)
    
    def get_next_task(self, worker_id: str) -> Optional[dict]:
        """Get next available task for worker"""
        with self.lock:
            # Check for stale assignments (>2 hours old)
            current_time = datetime.now()
            for task_id, task in list(self.assigned_tasks.items()):
                assigned_at = task['assigned_at']
                if (current_time - assigned_at).seconds > 7200:  # 2 hours
                    logging.warning(f"Task {task_id} timeout - reassigning")
                    self.tasks[task_id]['status'] = 'pending'
                    self.tasks[task_id]['assigned_to'] = None
                    del self.assigned_tasks[task_id]
            
            # Find next pending task
            for task_id, task in self.tasks.items():
                if task['status'] == 'pending':
                    task['status'] = 'assigned'
                    task['assigned_to'] = worker_id
                    task['assigned_at'] = current_time
                    self.assigned_tasks[task_id] = task
                    
                    logging.info(f"Assigned task {task_id} to {worker_id}")
                    return task
            
            # No pending tasks - check if all tasks are complete
            self.check_all_complete()
            
            return None
    
    def mark_complete(self, task_id: str, worker_id: str):
        """Mark task as complete"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]['status'] = 'complete'
                if task_id in self.assigned_tasks:
                    del self.assigned_tasks[task_id]
                
                logging.info(f"Task {task_id} completed by {worker_id}")
                
                # Check if all tasks are now complete
                self.check_all_complete()
            else:
                logging.warning(f"Unknown task {task_id} reported complete by {worker_id}")
    
    def save_result(self, task_id: str, result_data: dict):
        """Save experiment result to centralized CSV"""
        with self.lock:
            if task_id not in self.tasks:
                logging.warning(f"Cannot save result: task {task_id} not found")
                return
            
            try:
                # Add signature if not present
                if 'signature' not in result_data:
                    result_data['signature'] = self._create_result_signature(result_data)
                
                centralized_csv = 'final_results.csv'
                
                # Read existing centralized results
                if os.path.exists(centralized_csv):
                    df_central = pd.read_csv(centralized_csv)
                    df_central = pd.concat([df_central, pd.DataFrame([result_data])], ignore_index=True)
                else:
                    df_central = pd.DataFrame([result_data])
                
                # Save to centralized CSV
                df_central.to_csv(centralized_csv, index=False)
                logging.info(f"Saved result for {task_id} to {centralized_csv}")
                
            except Exception as e:
                logging.error(f"Error saving result to centralized CSV: {e}")
    
    def mark_failed(self, task_id: str, worker_id: str, error_msg: str, max_retries: int = 3):
        """Mark task as failed and reassign if under retry limit"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task['retry_count'] += 1
                task['last_error'] = error_msg
                
                if task_id in self.assigned_tasks:
                    del self.assigned_tasks[task_id]
                
                if task['retry_count'] <= max_retries:
                    # Reassign task
                    task['status'] = 'pending'
                    task['assigned_to'] = None
                    logging.warning(
                        f"Task {task_id} failed on {worker_id} (attempt {task['retry_count']}/{max_retries}). "
                        f"Error: {error_msg[:100]}... Reassigning."
                    )
                else:
                    # Mark as failed permanently
                    task['status'] = 'failed'
                    logging.error(
                        f"Task {task_id} failed {task['retry_count']} times. Giving up. "
                        f"Last error: {error_msg[:100]}"
                    )
            else:
                logging.warning(f"Unknown task {task_id} reported failed by {worker_id}")
    
    def check_all_complete(self):
        """Check if all tasks are complete and initiate shutdown if so"""
        # Note: Should be called while holding self.lock
        total = len(self.tasks)
        complete = sum(1 for t in self.tasks.values() if t['status'] == 'complete')
        
        if total > 0 and complete == total:
            logging.info("="*60)
            logging.info("ALL TASKS COMPLETE!")
            logging.info(f"Total experiments: {total}")
            logging.info("Shutting down experiment manager...")
            logging.info("="*60)
            
            # Set running flag to False to stop server
            self.running = False
    
    def get_status(self) -> dict:
        """Get current status"""
        with self.lock:
            total = len(self.tasks)
            pending = sum(1 for t in self.tasks.values() if t['status'] == 'pending')
            assigned = sum(1 for t in self.tasks.values() if t['status'] == 'assigned')
            complete = sum(1 for t in self.tasks.values() if t['status'] == 'complete')
            failed = sum(1 for t in self.tasks.values() if t['status'] == 'failed')
            
            return {
                'total': total,
                'pending': pending,
                'assigned': assigned,
                'complete': complete,
                'failed': failed,
                'progress': f"{complete}/{total} ({100*complete/total:.1f}%)" if total > 0 else "0/0"
            }
    
    def handle_client(self, conn, addr):
        """Handle worker connection"""
        try:
            data = conn.recv(4096).decode('utf-8')
            if not data:
                return
            
            request = json.loads(data)
            
            if request['type'] == 'request_task':
                worker_id = request['worker_id']
                task = self.get_next_task(worker_id)
                
                if task:
                    # Create a clean task dict without internal fields
                    clean_task = {
                        'task_id': task['task_id'],
                        'phase': task['phase'],
                        'config': task['config'],
                        'seed': task['seed'],
                        'output_dir': task['output_dir']
                    }
                    
                    response = {
                        'type': 'task',
                        'task': clean_task
                    }
                else:
                    response = {'type': 'no_task'}
                
                conn.sendall(json.dumps(response).encode('utf-8'))
            
            elif request['type'] == 'task_complete':
                task_id = request['task_id']
                worker_id = request['worker_id']
                self.mark_complete(task_id, worker_id)
                
                response = {'type': 'ack'}
                conn.sendall(json.dumps(response).encode('utf-8'))
            
            elif request['type'] == 'submit_result':
                task_id = request['task_id']
                result_data = request['result_data']
                worker_id = request.get('worker_id', 'unknown')
                
                # Save result to all output directories
                self.save_result(task_id, result_data)
                
                # Mark task as complete
                self.mark_complete(task_id, worker_id)
                
                response = {'type': 'ack'}
                conn.sendall(json.dumps(response).encode('utf-8'))
            
            elif request['type'] == 'task_failed':
                task_id = request['task_id']
                worker_id = request['worker_id']
                error_msg = request.get('error', 'Unknown error')
                self.mark_failed(task_id, worker_id, error_msg)
                
                response = {'type': 'ack'}
                conn.sendall(json.dumps(response).encode('utf-8'))
            
            elif request['type'] == 'status':
                status = self.get_status()
                response = {'type': 'status', 'status': status}
                conn.sendall(json.dumps(response).encode('utf-8'))
        
        except Exception as e:
            logging.error(f"Error handling client {addr}: {e}")
        finally:
            conn.close()
    
    def run_server(self):
        """Run socket server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        server_socket.settimeout(1.0)  # Allow checking running flag
        
        logging.info(f"Experiment Manager listening on {self.host}:{self.port}")
        
        while self.running:
            try:
                conn, addr = server_socket.accept()
                threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                logging.error(f"Server error: {e}")
        
        server_socket.close()
    
    def start(self):
        """Start the experiment manager"""
        logging.info("Starting Experiment Manager")
        
        # Generate all tasks
        self.generate_all_tasks()
        
        # Load completed tasks
        self.load_completed_tasks()
        
        # Print initial status
        status = self.get_status()
        logging.info(f"Initial status: {status['progress']}")
        
        # Start CSV refresh thread
        refresh_thread = threading.Thread(target=self.refresh_completed, daemon=True)
        refresh_thread.start()
        
        # Start server
        try:
            self.run_server()
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            self.running = False


if __name__ == "__main__":
    manager = ExperimentManager()
    manager.start()
