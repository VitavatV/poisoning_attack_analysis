import random
import yaml
import itertools
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
import logging

# Import modules ของเรา
from models import ScalableCNN
from data_utils import load_global_dataset, partition_data_dirichlet, get_client_dataloader
from utils import train_client, evaluate_model, fed_avg, fed_median, EarlyStopping

import pandas as pd
import os

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
    
    print(f"\\n>>> RUNNING Seed: {seed}")
    print("\\n" + "="*60)
    print(f"RUNNING: {config.get('phase_name', 'Unknown')}")
    print(f"Setting: Dataset={config['dataset']}, Alpha={config['alpha']}")
    print(f"Model: W={config['width_factor']}, D={config['depth']}")
    print(f"Poison: {config['poison_ratio']} ({config.get('data_ordering', 'shuffle')})")
    print("="*60)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare Data
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
    
    # Map subset indices กลับไปเป็น Global Indices
    client_global_indices = {}
    for cid, idxs in client_indices_subset.items():
        client_global_indices[cid] = [train_subset_indices[i] for i in idxs]

    # 2. Setup Global Model
    num_classes = 100 if config['dataset'] == 'cifar100' else 10
    global_model = ScalableCNN(
        num_classes=num_classes, 
        width_factor=config['width_factor'], 
        depth=config['depth']
    ).to(device)
    
    num_params = global_model.get_num_parameters()
    print(f"Model Parameters: {num_params:,}")
    
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
        selected_clients = np.random.choice(range(config['num_clients']), m, replace=False)
        
        # --- Local Training ---
        for client_id in selected_clients:
            # Enable poisoning when poison_ratio > 0
            is_victim = config['poison_ratio'] > 0
            
            # Setup Config เฉพาะ Client
            client_config = config.copy()
            
            train_loader = get_client_dataloader(
                train_ds_full, 
                client_global_indices[client_id], 
                client_config, 
                is_attacker=is_victim 
            )
            
            # Init Model with Global Weights
            local_model = copy.deepcopy(global_model)
            local_model.load_state_dict(global_weights)
            
            # Train with weight_decay
            w = train_client(local_model, train_loader, 
                             epochs=config['local_epochs'], 
                             lr=config['lr'], 
                             device=device,
                             momentum=config.get('momentum', 0.9),
                             weight_decay=float(config.get('weight_decay', 0)))
            local_weights.append(w)
            
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
    
    return test_acc, test_loss, best_acc, best_loss, num_params

def main():
    import sys
    
    # Support command-line config path
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_definitive.yaml"
    
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
            logging.FileHandler(os.path.join(output_dir, 'experiment.log'), mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Starting experiments with config: {config_path}")
    logging.info(f"Seeds: {seed_list}")
    
    # 1. เตรียม List สำหรับเก็บผลลัพธ์ทั้งหมด
    all_results = []
    
    config_name_list = [
        'exp0_vary_width',
        'exp1_fine_grained_width',
        'exp2_defense_comparison',
        'exp3_mechanism_analysis',
        'exp4_attack_types'
    ]
    phases = []
    for config_name in config_name_list:
        if config_name not in config:
            logging.warning(f"Config {config_name} not found in config file. Skipping.")
            continue    
        phases.append((config_name, config[config_name]))

    for phase_name, phase_cfg in phases:
        phase_cfg['phase_name'] = phase_name
        exp_list = generate_experiments(phase_cfg, defaults)
        
        for exp in exp_list:
            seed_test_accs = []
            seed_test_losses = []
            seed_val_accs = []
            seed_val_losses = []
            seed_num_params = []
            
            # try:
            if True:
                for seed in seed_list:
                    t_acc, t_loss, v_acc, v_loss, num_params = run_single_experiment(exp, seed)
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

    print("All experiments completed. Results saved to final_results.csv")
    logging.info("All experiments completed")

if __name__ == "__main__":
    main()