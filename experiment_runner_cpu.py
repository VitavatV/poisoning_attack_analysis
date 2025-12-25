"""
CPU Worker Process

Requests tasks from experiment_manager and executes them using CPU.
Uses multiprocessing to parallelize client training for better CPU utilization.
"""

import random
import yaml
import socket
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
import logging
from multiprocessing import Pool, cpu_count
import sys
import os

# Import modules
from models import ScalableCNN, LogisticRegression
from data_utils import load_global_dataset, partition_data_dirichlet, get_client_dataloader
from utils import train_client, evaluate_model, fed_avg, fed_median, EarlyStopping

import pandas as pd


def request_task_from_manager(worker_id: str, host='localhost', port=5000) -> dict:
    """
    Request a task from the experiment manager.
    
    Returns:
        Task dict if available, None otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        
        request = {
            'type': 'request_task',
            'worker_id': worker_id,
            'worker_type': 'cpu'  # Identify as CPU worker
        }
        
        sock.sendall(json.dumps(request).encode('utf-8'))
        response_data = sock.recv(8192).decode('utf-8')
        sock.close()
        
        response = json.loads(response_data)
        
        if response['type'] == 'task':
            return response['task']
        elif response['type'] == 'no_task':
            return None
        
    except Exception as e:
        print(f"Error requesting task: {e}")
        return None


def notify_task_complete(task_id: str, worker_id: str, host='localhost', port=5000):
    """Notify manager that task is complete"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        
        request = {
            'type': 'task_complete',
            'task_id': task_id,
            'worker_id': worker_id
        }
        
        sock.sendall(json.dumps(request).encode('utf-8'))
        response = sock.recv(1024).decode('utf-8')
        sock.close()
        
        print(f"Task {task_id} marked complete")
    except Exception as e:
        print(f"Error notifying completion: {e}")


def notify_task_failed(task_id: str, worker_id: str, error_msg: str, host='localhost', port=5000):
    """Notify manager that task has failed"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        
        request = {
            'type': 'task_failed',
            'task_id': task_id,
            'worker_id': worker_id,
            'error': error_msg
        }
        
        sock.sendall(json.dumps(request).encode('utf-8'))
        response = sock.recv(1024).decode('utf-8')
        sock.close()
        
        print(f"Task {task_id} marked as failed")
    except Exception as e:
        print(f"Error notifying failure: {e}")



def submit_result_to_manager(task_id: str, worker_id: str, result_data: dict, host='localhost', port=5000):
    """Submit experiment result to manager"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        
        request = {
            'type': 'submit_result',
            'task_id': task_id,
            'worker_id': worker_id,
            'result_data': result_data
        }
        
        sock.sendall(json.dumps(request).encode('utf-8'))
        response = sock.recv(1024).decode('utf-8')
        sock.close()
        
        print(f"Result submitted for task {task_id}")
        return True
    except Exception as e:
        print(f"Error submitting result: {e}")
        return False


def create_model(config, num_classes, in_channels, img_size):
    """
    Create model based on model_type in config.
    
    Args:
        config: Experiment configuration dict
        num_classes: Number of output classes
        in_channels: Number of input channels
        img_size: Spatial dimension of input
    
    Returns:
        PyTorch model instance
    """
    model_type = config.get('model_type', 'cnn')  # Default to CNN
    
    # Get width and depth with defaults
    width_factor = config.get('width_factor', 4)
    depth = config.get('depth', 4)
    
    if model_type == 'lr':
        # Multi-Layer Perceptron (scalable logistic regression)
        model = LogisticRegression(
            num_classes=num_classes,
            width_factor=width_factor,
            depth=depth,
            in_channels=in_channels,
            img_size=img_size
        )
    else:  # model_type == 'cnn'
        # Convolutional Neural Network
        model = ScalableCNN(
            num_classes=num_classes,
            width_factor=width_factor,
            depth=depth,
            in_channels=in_channels,
            img_size=img_size
        )
    
    return model


def train_client_worker(args):
    """Worker function for parallel client training using CPU multiprocessing"""
    client_id, global_weights, config, train_ds_only, client_indices_subset, device_str, malicious_clients = args
    
    from models import ScalableCNN, LogisticRegression
    from data_utils import get_client_dataloader
    from utils import train_client
    import torch
    
    device = torch.device(device_str)
    
    is_malicious = client_id in malicious_clients
    train_loader = get_client_dataloader(
        train_ds_only,
        client_indices_subset[client_id],
        config,
        is_malicious_client=is_malicious
    )
    
    if config['dataset'] == 'mnist':
        num_classes, in_channels, img_size = 10, 1, 28
    elif config['dataset'] == 'cifar10':
        num_classes, in_channels, img_size = 10, 3, 32
    elif config['dataset'] == 'cifar100':
        num_classes, in_channels, img_size = 100, 3, 32
    else:
        num_classes = config.get('num_classes', 10)
        in_channels = config.get('in_channels', 3)
        img_size = config.get('img_size', 32)
    
    local_model = create_model(config, num_classes, in_channels, img_size).to(device)
    
    local_model.load_state_dict(global_weights)
    
    trained_weights = train_client(
        local_model,
        train_loader,
        epochs=config['local_epochs'],
        lr=config['lr'],
        device=device,
        momentum=config.get('momentum', 0.9),
        weight_decay=float(config.get('weight_decay', 0)),
        max_grad_norm=config.get('max_grad_norm', 1.0)
    )
    
    return trained_weights


def run_single_experiment(config, seed, num_cpu_workers=None):
    """Run a single experiment on CPU with multiprocessing"""
    # Force CPU device
    config['device'] = 'cpu'
    
    # Set all seeds
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Determine number of CPU workers for parallel training
    if num_cpu_workers is None:
        num_cpu_workers = max(1, cpu_count() - 2)  # Leave 2 cores for system
    
    print(f"\n>>> RUNNING Seed: {seed} on CPU (using {num_cpu_workers} workers)")
    print("\n" + "="*60)

    device = torch.device(config['device'])
    device_str = str(device)
    
    train_ds_full, test_ds = load_global_dataset(config['dataset'])
    
    # Validation Split
    val_size = int(len(train_ds_full) * config['validation_split'])
    train_size = len(train_ds_full) - val_size
    
    train_ds, val_ds = random_split(train_ds_full, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(42))
    
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
    
    # Partition data
    train_subset_indices = train_ds.indices
    train_ds_only = Subset(train_ds_full, train_subset_indices)
    client_indices_subset = partition_data_dirichlet(train_ds_only, config['num_clients'], 
                                                     alpha=config['alpha'])

    # Handle both tensor (MNIST) and list (CIFAR-10) targets
    if isinstance(train_ds_full.targets, torch.Tensor):
        train_ds_only.targets = train_ds_full.targets.clone().detach()
    else:
        # Convert list to tensor (CIFAR-10 case)
        train_ds_only.targets = torch.tensor(train_ds_full.targets)
    
    # Select malicious clients based on poison_ratio (client-level poisoning)
    from data_utils import select_malicious_clients
    malicious_clients = select_malicious_clients(
        config['num_clients'], 
        config['poison_ratio'], 
        seed
    )
    
    print(f"Malicious clients: {len(malicious_clients)}/{config['num_clients']} ({config['poison_ratio']*100:.0f}%)")
    if len(malicious_clients) > 0:
        print(f"IDs: {malicious_clients}")
    
    # Initialize model
    if config['dataset'] == 'mnist':
        num_classes, in_channels, img_size = 10, 1, 28
    elif config['dataset'] == 'cifar10':
        num_classes, in_channels, img_size = 10, 3, 32
    elif config['dataset'] == 'cifar100':
        num_classes, in_channels, img_size = 100, 3, 32
    else:
        num_classes = config.get('num_classes', 10)
        in_channels = config.get('in_channels', 3)
        img_size = config.get('img_size', 32)
        
    global_model = create_model(config, num_classes, in_channels, img_size).to(device)
    global_weights = global_model.state_dict()
    
    # Early Stopping
    early_stopper = EarlyStopping(
        patience=config['early_stopping_patience'], 
        min_delta=config['min_delta']
    )
    
    # Training Loop
    best_acc = 0.0
    best_loss = np.inf
    best_epoch = 0
    
    for round_idx in range(config['global_rounds']):
        local_weights = []
        
        m = max(int(config['fraction_fit'] * config['num_clients']), 1)
        global_weights = global_model.state_dict()
        selected_clients = np.random.choice(range(config['num_clients']), m, replace=False)
        
        # Parallel client training using multiprocessing
        workers_to_use = min(len(selected_clients), num_cpu_workers)
        
        if workers_to_use > 1:
            # Use multiprocessing for parallel training
            with Pool(workers_to_use) as pool:
                args = [(client_id, global_weights, config, train_ds_only, 
                         client_indices_subset, device_str, malicious_clients) 
                        for client_id in selected_clients]
                local_weights = pool.map(train_client_worker, args)
        else:
            # Sequential training for single worker
            for client_id in selected_clients:
                local_model = create_model(config, num_classes, in_channels, img_size).to(device)
                local_model.load_state_dict(global_weights)
                
                is_malicious = client_id in malicious_clients
                client_loader = get_client_dataloader(
                    train_ds_only, 
                    client_indices_subset[client_id], 
                    config, 
                    is_malicious_client=is_malicious
                )
                
                trained_weights = train_client(
                    local_model,
                    client_loader,
                    epochs=config['local_epochs'],
                    lr=config['lr'],
                    device=device,
                    momentum=config.get('momentum', 0.9),
                    weight_decay=float(config.get('weight_decay', 0)),
                    max_grad_norm=config.get('max_grad_norm', 1.0)
                )
                
                local_weights.append(trained_weights)
            
        # Aggregation
        aggregator_name = config.get('aggregator', 'fedavg')
        if aggregator_name == 'median':
            global_weights = fed_median(local_weights)
        else:
            global_weights = fed_avg(local_weights)
        global_model.load_state_dict(global_weights)
        
        # Validation
        val_loss, val_acc = evaluate_model(global_model, val_loader, device)
        
        # Check if validation returns NaN (all clients failed to train)
        if np.isnan(val_loss) or np.isinf(val_loss):
            logging.error(f"🛑 EXPERIMENT FAILED: Validation returned NaN/Inf at round {round_idx+1}")
            logging.error(f"   This indicates severe training instability across all clients.")
            logging.error(f"   Stopping experiment early. Returning NaN results.")
            # Return NaN results
            return np.nan, np.nan, np.nan, np.nan, round_idx, sum(p.numel() for p in global_model.parameters()), global_model
        
        print(f"Round {round_idx+1}/{config['global_rounds']} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        early_stopper(val_loss, global_weights)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            best_epoch = round_idx
            
        if early_stopper.early_stop:
            print(">>> Early Stopping Triggered!")
            global_model.load_state_dict(early_stopper.get_best_weights())
            break
            
    # Final Test
    test_loss, test_acc = evaluate_model(global_model, test_loader, device)
    print(f"FINAL RESULT: Test Acc: {test_acc:.4f}")
    num_params = sum(p.numel() for p in global_model.parameters())
    
    return test_acc, test_loss, best_acc, best_loss, best_epoch, num_params, global_model


def run_task(task: dict, worker_id: str, num_cpu_workers=None):
    """Execute a single task"""
    config = task['config']
    seed = task['seed']
    output_dir = task['output_dir']
    phase = task['phase']
    
    # Setup logging per task
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"worker_{task['task_id']}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logging.info(f"Starting task {task['task_id']}")
    logging.info(f"Config: {config}")
    
    try:
        # Run experiment
        t_acc, t_loss, v_acc, v_loss, best_epoch, num_params, model = run_single_experiment(
            config, seed, num_cpu_workers
        )
        
        # Check if results contain NaN/Inf (training failed)
        if np.isnan(t_acc) or np.isinf(t_acc) or np.isnan(t_loss) or np.isinf(t_loss):
            logging.warning("⚠️ Training resulted in NaN/Inf values. Recording as failed experiment.")
        
        # Save result (including NaN if training failed)
        result_entry = {
            "phase": phase,
            "signature": f"{config['dataset']}|{config.get('model_type', 'cnn')}|{config.get('width_factor', 4)}|{config.get('depth', 4)}|{config['poison_ratio']}|{config.get('poison_type', 'label_flip')}|{config['alpha']}|{config.get('data_ordering', 'shuffle')}|{config.get('aggregator', 'fedavg')}|{config.get('batch_size', 64)}|{seed}",
            "dataset": config['dataset'],
            "model_type": config.get('model_type', 'cnn'),
            "width_factor": config.get('width_factor', 4),
            "depth": config.get('depth', 4),
            "poison_ratio": config['poison_ratio'],
            "poison_type": config.get('poison_type', 'label_flip'),
            "alpha": config['alpha'],
            "data_ordering": config.get('data_ordering', 'shuffle'),
            "aggregator": config.get('aggregator', 'fedavg'),
            "batch_size": config.get('batch_size', 64),
            "mean_test_acc": float(t_acc) if not np.isnan(t_acc) else np.nan,
            "std_test_acc": 0.0,
            "mean_test_loss": float(t_loss) if not np.isnan(t_loss) else np.nan,
            "std_test_loss": 0.0,
            "mean_val_acc": float(v_acc) if not np.isnan(v_acc) else np.nan,
            "std_val_acc": 0.0,
            "mean_val_loss": float(v_loss) if not np.isnan(v_loss) else np.nan,
            "std_val_loss": 0.0,
            "num_parameters": num_params,
            "best_epoch": best_epoch,
            "seed": seed
        }
        
        # Submit result to manager (manager will save to all output directories)
        success = submit_result_to_manager(task['task_id'], worker_id, result_entry)
        
        if not success:
            # Fallback: save locally if manager submission failed
            logging.warning("Failed to submit to manager, saving locally...")
            csv_path = os.path.join(output_dir, "final_results.csv")
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df = pd.concat([df, pd.DataFrame([result_entry])], ignore_index=True)
            else:
                df = pd.DataFrame([result_entry])
            
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved result to {csv_path} (fallback)")
        
        logging.info(f"Task {task['task_id']} completed successfully")
        return True
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error running task {task['task_id']}: {error_msg}")
        
        import traceback
        traceback.print_exc()
        
        # Notify manager that task failed so it can be reassigned
        notify_task_failed(task['task_id'], worker_id, error_msg)
        
        return False


def worker_loop(manager_host='localhost', manager_port=5000, num_cpu_workers=None):
    """Main worker loop"""
    import multiprocessing
    
    # Set multiprocessing method
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    print("="*60)
    print("CPU Worker Starting")
    print("="*60)
    
    pid = os.getpid()
    worker_id = f"cpu_worker_{pid}"
    
    # Determine CPU workers
    if num_cpu_workers is None:
        num_cpu_workers = max(1, cpu_count() - 2)
    
    # Automatic thread control to optimize CPU utilization
    # Calculate optimal threads per worker to avoid oversubscription
    total_cpus = cpu_count()
    
    if num_cpu_workers == 1:
        # Single worker: use most of the CPU (leave some for system)
        optimal_threads = max(1, int(total_cpus * 0.7))
    else:
        # Multiple workers: distribute threads evenly
        optimal_threads = max(1, (total_cpus * 0.7) // num_cpu_workers)
    
    # Set PyTorch thread limits
    torch.set_num_threads(optimal_threads)
    torch.set_num_interop_threads(optimal_threads)
    
    # Also set environment variables for BLAS/MKL libraries
    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
    os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(optimal_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
    
    print(f"\nWorker ID: {worker_id}")
    print(f"Using CPU with {num_cpu_workers} parallel workers")
    print(f"Threads per worker: {optimal_threads} (Total CPUs: {total_cpus})")
    print(f"Expected CPU utilization: ~{min(100, (num_cpu_workers * optimal_threads / total_cpus) * 100):.0f}%")
    print(f"Manager: {manager_host}:{manager_port}")
    print("="*60)
    
    task_count = 0
    
    while True:
        # Request task
        print(f"\n[{worker_id}] Requesting task from manager...")
        task = request_task_from_manager(worker_id, manager_host, manager_port)
        
        if task is None:
            print(f"[{worker_id}] No more tasks available. Shutting down.")
            break
        
        task_count += 1
        print(f"\n[{worker_id}] Received task {task['task_id']} (Task #{task_count})")
        
        # Execute task
        success = run_task(task, worker_id, num_cpu_workers)
        
        # Notify completion
        if success:
            notify_task_complete(task['task_id'], worker_id, manager_host, manager_port)
        
        # Wait before next request
        print(f"\n[{worker_id}] Waiting 10 seconds before next request...")
        time.sleep(10)
    
    print(f"\n[{worker_id}] Worker shutting down. Completed {task_count} tasks.")


if __name__ == "__main__":
    # Parse command-line args for manager host/port and CPU workers
    manager_host = sys.argv[1] if len(sys.argv) > 1 else 'localhost'
    manager_port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    num_cpu_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    worker_loop(manager_host, manager_port, num_cpu_workers)
