import random
import yaml
import itertools
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

# Import modules ของเรา
from models import ScalableCNN
from data_utils import load_global_dataset, partition_data_dirichlet, get_client_dataloader
from utils import train_client, evaluate_model, fed_avg, fed_median, EarlyStopping

import pandas as pd
import os


def train_client_worker(args):
    """Worker function for parallel client training.
    
    This function is designed to be executed in a separate process.
    It creates a local model, trains it, and returns the trained weights.
    
    Args:
        args: Tuple of (client_id, global_weights, config, train_ds_only, client_indices_subset, device_str)
    
    Returns:
        Trained model weights (state_dict)
    """
    client_id, global_weights, config, train_ds_only, client_indices_subset, device_str = args
    
    # Import necessary modules in worker (required for multiprocessing)
    from models import ScalableCNN
    from data_utils import get_client_dataloader
    from utils import train_client
    import torch
    
    device = torch.device(device_str)
    
    # Recreate the dataloader for this client
    is_victim = config['poison_ratio'] > 0
    train_loader = get_client_dataloader(
        train_ds_only,
        client_indices_subset[client_id],
        config,
        is_attacker=is_victim
    )
    
    # Determine model parameters based on dataset
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
    
    # Create local model
    local_model = ScalableCNN(
        num_classes=num_classes,
        width_factor=config['width_factor'],
        depth=config['depth'],
        in_channels=in_channels,
        img_size=img_size
    ).to(device)
    
    # Load global weights
    local_model.load_state_dict(global_weights)
    
    # Train the client
    trained_weights = train_client(
        local_model,
        train_loader,
        epochs=config['local_epochs'],
        lr=config['lr'],
        device=device,
        momentum=config.get('momentum', 0.9),
        weight_decay=float(config.get('weight_decay', 0))
    )
    
    return trained_weights

def load_config(path="config_definitive.yaml"):
    """Load and validate YAML configuration file"""
    try:
        with open(path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{path}' not found.")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        raise

def load_existing_results(output_dir):
    """Load existing results from final_results.csv if it exists"""
    results_path = os.path.join(output_dir, "final_results.csv")
    if os.path.exists(results_path):
        try:
            df = pd.read_csv(results_path)
            logging.info(f"Loaded {len(df)} existing results from {results_path}")
            return df
        except Exception as e:
            logging.warning(f"Failed to load existing results: {e}")
            return pd.DataFrame()
    else:
        logging.info("No existing results found. Starting fresh.")
        return pd.DataFrame()

def is_experiment_completed(exp, phase_name, existing_results_df):
    """Check if an experiment with the same configuration has already been run"""
    if existing_results_df.empty:
        return False
    
    # Define key parameters that uniquely identify an experiment
    key_params = {
        'phase': phase_name,
        'dataset': exp['dataset'],
        'width_factor': exp['width_factor'],
        'depth': exp['depth'],
        'poison_ratio': exp['poison_ratio'],
        'alpha': exp['alpha'],
        'data_ordering': exp.get('data_ordering', 'shuffle'),
        'aggregator': exp.get('aggregator', 'fedavg'),
        'batch_size': exp.get('batch_size', 64)
    }
    
    # Check if a row exists with all matching parameters
    mask = pd.Series([True] * len(existing_results_df))
    for param, value in key_params.items():
        if param in existing_results_df.columns:
            mask &= (existing_results_df[param] == value)
    
    return mask.any()

def generate_experiments(phase_config, defaults):
    # (โค้ดเดิมสำหรับแตก Grid Search)
    vary_params = {}
    for item in phase_config['combinations']:
        vary_params.update(item)
    keys = list(vary_params.keys())
    vals = list(vary_params.values())
    
    experiments = []
    for instance in itertools.product(*vals):
        exp_setup = defaults.copy()
        exp_setup['dataset'] = phase_config['dataset']
        exp_setup['phase_name'] = phase_config.get('phase_name', 'Unknown')
        for k, v in zip(keys, instance):
            exp_setup[k] = v
        experiments.append(exp_setup)
    return experiments

def run_single_experiment(config, seed):
    # Set all seeds
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    print(f"\n>>> RUNNING Seed: {seed}")
    print("\n" + "="*60)

    device_config = config.get('device', 'cpu')
    device = torch.device(device_config)
    device_str = str(device)
    
    train_ds_full, test_ds = load_global_dataset(config['dataset'])
    
    # --- Validation Split for Early Stopping ---
    val_size = int(len(train_ds_full) * config['validation_split'])
    train_size = len(train_ds_full) - val_size
    
    train_ds, val_ds = random_split(train_ds_full, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(42))
    
    # DataLoader สำหรับ Server ไว้ Valid/Test
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
    
    # Partition ข้อมูล Train ให้ Clients
    train_subset_indices = train_ds.indices
    train_ds_only = Subset(train_ds_full, train_subset_indices)
    client_indices_subset = partition_data_dirichlet(train_ds_only, config['num_clients'], 
                                                     alpha=config['alpha'])

    # Pre-create all client dataloaders once
    train_ds_only.targets = torch.tensor(train_ds_full.targets)
    client_dataloaders = {}
    is_victim = config['poison_ratio'] > 0
    for client_id in range(config['num_clients']):
        client_config = config.copy()
        client_dataloaders[client_id] = get_client_dataloader(
            train_ds_only, 
            client_indices_subset[client_id], 
            client_config, 
            is_attacker=is_victim
        )

    # 2. Initialize Global Model
    # Determine num_classes and in_channels based on dataset
    if config['dataset'] == 'mnist':
        num_classes = 10
        in_channels = 1
        img_size = 28
    elif config['dataset'] == 'cifar10':
        num_classes = 10
        in_channels = 3
        img_size = 32
    elif config['dataset'] == 'cifar100':
        num_classes = 100
        in_channels = 3
        img_size = 32
    else:
        # Fallback/Default
        num_classes = config.get('num_classes', 10)
        in_channels = config.get('in_channels', 3)
        img_size = config.get('img_size', 32)
    global_model = ScalableCNN(
        num_classes=num_classes, 
        width_factor=config['width_factor'], 
        depth=config['depth'],
        in_channels=in_channels,
        img_size=img_size
    ).to(device)
    global_weights = global_model.state_dict()
    
    # 3. Setup Early Stopping
    early_stopper = EarlyStopping(
        patience=config['early_stopping_patience'], 
        min_delta=config['min_delta']
    )
    
    # 4. Global Training Loop
    best_acc = 0.0
    best_loss = np.inf
    
    for round_idx in range(config['global_rounds']):
        local_weights = []
        
        # Select Clients
        m = max(int(config['fraction_fit'] * config['num_clients']), 1)
        global_weights = global_model.state_dict()
        selected_clients = np.random.choice(range(config['num_clients']), m, replace=False)
        
        # --- Parallel Local Training ---
        # Determine number of parallel workers
        num_workers = config.get('num_parallel_workers', -1)
        if num_workers == -1:
            num_workers = max(1, cpu_count() - 1)  # Leave one core free
        num_workers = max(1, min(num_workers, cpu_count()))
        
        # Prepare arguments for each client
        worker_args = [
            (client_id, global_weights, config, train_ds_only, client_indices_subset, device_str)
            for client_id in selected_clients
        ]
        
        # Use multiprocessing Pool to train clients in parallel
        # If only 1 worker or 1 client, skip multiprocessing overhead
        if num_workers == 1 or len(selected_clients) == 1:
            # Sequential execution
            local_weights = [train_client_worker(args) for args in worker_args]
        else:
            # Parallel execution
            with Pool(processes=min(num_workers, len(selected_clients))) as pool:
                local_weights = pool.map(train_client_worker, worker_args)
            
        # --- Aggregation ---
        aggregator_name = config.get('aggregator', 'fedavg')
        if aggregator_name == 'median':
            global_weights = fed_median(local_weights)
        else:
            global_weights = fed_avg(local_weights)
        global_model.load_state_dict(global_weights)
        
        # --- Validation & Early Stopping ---
        val_loss, val_acc = evaluate_model(global_model, val_loader, device)
        
        print(f"Round {round_idx+1}/{config['global_rounds']} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Check Early Stopping
        early_stopper(val_loss, global_weights)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            
        if early_stopper.early_stop:
            print(">>> Early Stopping Triggered!")
            # Load best weights back
            global_model.load_state_dict(early_stopper.get_best_weights())
            break
            
    # 5. Final Test
    test_loss, test_acc = evaluate_model(global_model, test_loader, device)
    print(f"FINAL RESULT: Test Acc: {test_acc:.4f}")
    num_params = sum(p.numel() for p in global_model.parameters())
    
    return test_acc, test_loss, best_acc, best_loss, num_params, global_model

def main():
    import sys
    
    # Support command-line config path
    if len(sys.argv) > 1:
        config_path_list = [sys.argv[1]]
    else:
        config_path_list = [
            'configs/config_exp0_mnist.yaml',
            'configs/config_exp1_mnist.yaml',
            'configs/config_exp2_mnist.yaml',
            'configs/config_exp3_mnist.yaml',
            'configs/config_exp4_mnist.yaml',
            ]
    for config_path in config_path_list:
        
        config = load_config(config_path)
        defaults = config['defaults']
        seed_list = config.get('seeds', [42])
        
        # Create output directory
        output_dir = defaults['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'experiment.log'), mode='a'),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Starting experiments with config: {config_path}")
        logging.info(f"Seeds: {seed_list}")
        
        # Load existing results to check for completed experiments
        existing_results_df = load_existing_results(output_dir)
        
        # 1. เตรียม List สำหรับเก็บผลลัพธ์ทั้งหมด
        all_results = []
        
        # Load existing results into all_results if they exist
        if not existing_results_df.empty:
            all_results = existing_results_df.to_dict('records')
        
        config_name_list = [
            'exp0_vary_width',
            'exp1_fine_grained_width',
            'exp2_defense_comparison',
            'exp3_mechanism_analysis',
            'exp4_attack_types',
            # Phase 0 New Experiments
            'exp0_width_scaling',
            'exp0_depth_scaling',
            'exp0_mnist_baseline',
            'exp0_cifar10_baseline'
        ]
        phases = []
        for config_name in config_name_list:
            if config_name not in config:
                logging.warning(f"Config {config_name} not found in config file. Skipping.")
                continue    
            phases.append((config_name, config[config_name]))
    
        total_experiments = 0
        skipped_experiments = 0
        
        for phase_name, phase_cfg in phases:
            phase_cfg['phase_name'] = phase_name
            exp_list = generate_experiments(phase_cfg, defaults)
            
            for i_exp, exp in enumerate(exp_list):
                total_experiments += 1
                
                # Check if this experiment has already been completed
                n_experiments = len(existing_results_df)
                if is_experiment_completed(exp, phase_name, existing_results_df):
                    skipped_experiments += 1
                    logging.info(f"SKIPPING experiment (already completed): {phase_name} - "
                               f"Dataset={exp['dataset']}, Width={exp['width_factor']}, "
                               f"Depth={exp['depth']}, Alpha={exp['alpha']}, "
                               f"Poison={exp['poison_ratio']}")
                    print(f"\n>>> SKIPPING (already completed): {phase_name}")
                    print(f"    Dataset={exp['dataset']}, W={exp['width_factor']}, "
                          f"D={exp['depth']}, Alpha={exp['alpha']}, Poison={exp['poison_ratio']}")
                    continue
                
                seed_test_accs = []
                seed_test_losses = []
                seed_val_accs = []
                seed_val_losses = []
                seed_num_params = []
                model = None
                
                # try:
                if True:
                    for seed in seed_list:
                        t_acc, t_loss, v_acc, v_loss, num_params, model = run_single_experiment(exp, seed)
                        seed_test_accs.append(t_acc)
                        seed_test_losses.append(t_loss)
                        seed_val_accs.append(v_acc)
                        seed_val_losses.append(v_loss)
                        seed_num_params.append(num_params)
                # except Exception as e:
                #     logging.error(f"Error in experiment: {e}")
                #     print(f"Error in experiment: {e}")
                #     continue
                
                # Calculate stats
                if not seed_test_accs:
                    print(f"Warning: No valid results")
                    continue
                    
                mean_acc = np.mean(seed_test_accs)
                std_acc = np.std(seed_test_accs)
                mean_loss = np.mean(seed_test_losses)
                std_loss = np.std(seed_test_losses)
                mean_acc_val = np.mean(seed_val_accs)
                std_acc_val = np.std(seed_val_accs)
                mean_loss_val = np.mean(seed_val_losses)
                std_loss_val = np.std(seed_val_losses)
                mean_num_params = np.mean(seed_num_params)
    
                # 2. รวบรวมข้อมูลลง Dictionary
                result_entry = {
                    "phase": phase_name,
                    "dataset": exp['dataset'],
                    "width_factor": exp['width_factor'],
                    "depth": exp['depth'],
                    "poison_ratio": exp['poison_ratio'],
                    "alpha": exp['alpha'],
                    "data_ordering": exp.get('data_ordering', 'shuffle'),
                    "aggregator": exp.get('aggregator', 'fedavg'),
                    "batch_size": exp.get('batch_size', 64),
                    "mean_test_acc": mean_acc,
                    "std_test_acc": std_acc,
                    "mean_test_loss": mean_loss,
                    "std_test_loss": std_loss,
                    "mean_val_acc": mean_acc_val,
                    "std_val_acc": std_acc_val,
                    "mean_val_loss": mean_loss_val,
                    "std_val_loss": std_loss_val,
                    "num_parameters": mean_num_params,
                    "raw_seeds": str(seed_test_accs)
                }
                all_results.append(result_entry)
                
                # 3. Save ระหว่างทาง
                df = pd.DataFrame(all_results)
                df.to_csv(os.path.join(output_dir, "final_results.csv"), index=False)
                logging.info(f"Saved: {len(all_results)} results")
    
                # 4. Save Model
                if model is not None:
                    torch.save(model.state_dict(), os.path.join(output_dir, f"final_model_{i_exp+n_experiments}.pth"))
                    logging.info("Saved final model")
    
        print(f"\n{'='*60}")
        print(f"All experiments completed. Results saved to final_results.csv")
        print(f"Total experiments: {total_experiments}")
        print(f"Skipped (already completed): {skipped_experiments}")
        print(f"Newly completed: {total_experiments - skipped_experiments}")
        print(f"{'='*60}")
        logging.info(f"All experiments completed. Total: {total_experiments}, "
                    f"Skipped: {skipped_experiments}, New: {total_experiments - skipped_experiments}")

if __name__ == "__main__":
    main()