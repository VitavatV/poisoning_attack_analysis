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
        
        for config in configs:
            defaults = config['defaults']
            seed_list = config.get('seeds', [42])
            
            for phase_name in phase_names:
                if phase_name not in config:
                    continue
                
                phase_cfg = config[phase_name]
                phase_cfg['phase_name'] = phase_name
                
                experiments = self.generate_experiments(phase_cfg, defaults)
                
                for exp in experiments:
                    for seed in seed_list:
                        task_id = f"{phase_name}_{task_counter}_{seed}"
                        
                        self.tasks[task_id] = {
                            'task_id': task_id,
                            'phase': phase_name,
                            'config': exp,
                            'seed': seed,
                            'output_dir': exp['output_dir'],
                            'status': 'pending',
                            'assigned_to': None,
                            'assigned_at': None
                        }
                        
                        task_counter += 1
        
        logging.info(f"Generated {len(self.tasks)} total tasks")
    
    def load_completed_tasks(self):
        """Scan all result directories for completed tasks"""
        result_dirs = glob.glob('results_*')
        
        for result_dir in result_dirs:
            csv_path = os.path.join(result_dir, 'final_results.csv')
            if not os.path.exists(csv_path):
                continue
            
            try:
                df = pd.read_csv(csv_path)
                
                for _, row in df.iterrows():
                    # Create a signature for this completed experiment
                    sig = self._create_task_signature(row.to_dict())
                    self.completed_tasks.add(sig)
                
                logging.info(f"Loaded {len(df)} completed experiments from {csv_path}")
            except Exception as e:
                logging.error(f"Error loading {csv_path}: {e}")
        
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
            str(config.get('width_factor', '')),
            str(config.get('depth', '')),
            str(config.get('poison_ratio', '')),
            config.get('poison_type', 'label_flip'),
            str(config.get('alpha', '')),
            config.get('data_ordering', 'shuffle'),
            config.get('aggregator', 'fedavg'),
            str(config.get('batch_size', 64))
        ]
        return '|'.join(key_params)
    
    def refresh_completed(self):
        """Periodically refresh completed tasks from CSV files"""
        while self.running:
            time.sleep(5)  # Check every 5 seconds
            
            result_dirs = glob.glob('results_*')
            new_completed = set()
            
            for result_dir in result_dirs:
                csv_path = os.path.join(result_dir, 'final_results.csv')
                if not os.path.exists(csv_path):
                    continue
                
                try:
                    df = pd.read_csv(csv_path)
                    for _, row in df.iterrows():
                        sig = self._create_task_signature(row.to_dict())
                        new_completed.add(sig)
                except Exception as e:
                    logging.error(f"Error refreshing {csv_path}: {e}")
            
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
            
            return {
                'total': total,
                'pending': pending,
                'assigned': assigned,
                'complete': complete,
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
                    response = {
                        'type': 'task',
                        'task': task
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
