"""
Generate CSV of Poisoning Data Information

This script generates a CSV file containing ground truth (GT) labels, training class labels,
and training client assignments for each experiment configuration.

The CSV filename format: {experiment_index}_{dataset}_poisoning_data.csv
Columns: Sample_Index, GT, Training_Class, Training_Client, Is_Poisoned

Poisoning Mechanism:
- poison_ratio determines the percentage of CLIENTS that are malicious (not data samples)
- Malicious clients poison 100% of their data
- Label flip: 0→9, 1→8, 2→7, 3→6, 4→5, 5→4, 6→3, 7→2, 8→1, 9→0 (complementary)
- Random noise: Random class except the correct class
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import itertools


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_dataset_info(dataset_name: str) -> Tuple[int, int]:
    """
    Get dataset information.
    
    Returns:
        (num_classes, num_samples)
    """
    dataset_info = {
        'mnist': (10, 60000),      # 10 classes, 60k training samples
        'cifar10': (10, 50000),    # 10 classes, 50k training samples
    }
    return dataset_info.get(dataset_name.lower(), (10, 50000))


def partition_data_dirichlet_indices(num_samples: int, num_classes: int, num_clients: int, 
                                      alpha: float, seed: int) -> Tuple[Dict[int, List[int]], np.ndarray]:
    """
    Simulate Dirichlet partitioning to get client data indices.
    
    Returns:
        (client_indices_dict, labels_array)
    """
    np.random.seed(seed)
    
    # Create synthetic labels (evenly distributed classes)
    labels = np.array([i % num_classes for i in range(num_samples)])
    np.random.shuffle(labels)
    
    min_size = 0
    client_idcs = {}
    
    # Keep trying until all clients have at least 10 samples
    while min_size < 10:
        client_idcs = {i: [] for i in range(num_clients)}
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            client_splits = np.split(idx_k, proportions)
            for i in range(num_clients):
                client_idcs[i] += client_splits[i].tolist()
        
        min_size = min([len(client_idcs[i]) for i in range(num_clients)])
    
    return client_idcs, labels


def apply_label_flip(gt_label: int, num_classes: int = 10) -> int:
    """
    Apply complementary label flip.
    0→9, 1→8, 2→7, 3→6, 4→5, 5→4, 6→3, 7→2, 8→1, 9→0
    
    Args:
        gt_label: Ground truth label
        num_classes: Total number of classes
    
    Returns:
        Flipped label
    """
    return (num_classes - 1) - gt_label


def apply_random_noise(gt_label: int, num_classes: int = 10, seed: int = None) -> int:
    """
    Apply random noise: select random class except the correct class.
    
    Args:
        gt_label: Ground truth label
        num_classes: Total number of classes
        seed: Random seed for reproducibility
    
    Returns:
        Random label (not equal to gt_label)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get all possible classes except the ground truth
    possible_labels = [i for i in range(num_classes) if i != gt_label]
    return np.random.choice(possible_labels)


def select_malicious_clients(num_clients: int, poison_ratio: float, seed: int) -> List[int]:
    """
    Select which clients are malicious based on poison_ratio.
    
    Args:
        num_clients: Total number of clients
        poison_ratio: Percentage of clients to be malicious (0.0 to 1.0)
        seed: Random seed
    
    Returns:
        List of malicious client IDs
    """
    if poison_ratio <= 0.0:
        return []
    
    np.random.seed(seed)
    num_malicious = max(1, int(num_clients * poison_ratio))
    malicious_clients = np.random.choice(num_clients, num_malicious, replace=False).tolist()
    
    return malicious_clients


def generate_poisoning_csv(config_path: str, experiment_name: str, output_dir: str = "./poisoning_data"):
    """
    Generate poisoning data CSV for a given experiment configuration.
    
    Args:
        config_path: Path to the configuration YAML file
        experiment_name: Name/index of the experiment (e.g., 'exp2', 'exp4')
        output_dir: Directory to save CSV files
    """
    config = load_config(config_path)
    defaults = config.get('defaults', {})
    
    # Find the experiment phase
    experiment_phase = None
    for key in config.keys():
        if key != 'defaults' and isinstance(config[key], dict):
            experiment_phase = config[key]
            break
    
    if not experiment_phase:
        print(f"No experiment phase found in {config_path}")
        return
    
    # Get combinations
    combinations = experiment_phase.get('combinations', [])
    if not combinations:
        print(f"No combinations found in {config_path}")
        return
    
    # Generate all parameter combinations
    param_names = []
    param_values = []
    for combo in combinations:
        for key, values in combo.items():
            param_names.append(key)
            param_values.append(values if isinstance(values, list) else [values])
    
    # Create all combinations
    all_configs = []
    for values in itertools.product(*param_values):
        cfg = defaults.copy()
        for name, value in zip(param_names, values):
            cfg[name] = value
        all_configs.append(cfg)
    
    # Group by dataset
    datasets = set(cfg.get('dataset', 'unknown') for cfg in all_configs)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate CSV for each dataset
    for dataset in datasets:
        dataset_configs = [cfg for cfg in all_configs if cfg.get('dataset') == dataset]
        
        print(f"\nGenerating poisoning data for {experiment_name} - {dataset}")
        print(f"Number of configurations: {len(dataset_configs)}")
        
        # Process each configuration
        for config_idx, cfg in enumerate(dataset_configs, 1):
            # Extract parameters
            num_clients = cfg.get('num_clients', 10)
            alpha = cfg.get('alpha', 100.0)
            poison_ratio = cfg.get('poison_ratio', 0.0)
            poison_type = cfg.get('poison_type', 'label_flip')
            seed = cfg.get('seed', 42)
            
            # Get dataset info
            num_classes, num_samples = get_dataset_info(dataset)
            
            # Partition data among clients using Dirichlet distribution
            client_idcs, labels = partition_data_dirichlet_indices(
                num_samples, num_classes, num_clients, alpha, seed
            )
            
            # Select which clients are malicious
            malicious_clients = select_malicious_clients(num_clients, poison_ratio, seed)
            
            print(f"  Config {config_idx:03d}: {len(malicious_clients)}/{num_clients} malicious clients "
                  f"(poison_ratio={poison_ratio}, poison_type={poison_type})")
            
            # Create data rows
            data_rows = []
            poisoned_count = 0
            
            for sample_idx in range(num_samples):
                gt_label = labels[sample_idx]
                
                # Find which client this sample belongs to
                client_id = None
                for cid, indices in client_idcs.items():
                    if sample_idx in indices:
                        client_id = cid
                        break
                
                if client_id is None:
                    continue
                
                # Determine if this sample is poisoned
                # Malicious clients poison 100% of their data
                is_poisoned = False
                training_label = gt_label
                
                if client_id in malicious_clients:
                    is_poisoned = True
                    poisoned_count += 1
                    
                    # Apply poisoning based on attack type
                    if poison_type == 'label_flip':
                        # Complementary label flip: 0→9, 1→8, 2→7, etc.
                        training_label = apply_label_flip(gt_label, num_classes)
                    elif poison_type == 'random_noise':
                        # Random class except the correct class
                        training_label = apply_random_noise(gt_label, num_classes, seed + sample_idx)
                    else:
                        # Unknown poison type, default to label flip
                        training_label = apply_label_flip(gt_label, num_classes)
                
                data_rows.append({
                    'Sample_Index': sample_idx,
                    'GT': gt_label,
                    'Training_Class': training_label,
                    'Training_Client': client_id,
                    'Is_Poisoned': is_poisoned
                })
            
            # Create DataFrame
            df = pd.DataFrame(data_rows)
            
            # Generate filename
            filename = f"{experiment_name}_{dataset}_config{config_idx:03d}_seed{seed}_alpha{alpha}_poison{poison_ratio}_{poison_type}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            
            # Print statistics
            total_samples = len(df)
            poisoned_samples = df['Is_Poisoned'].sum()
            unique_training_classes = df['Training_Class'].nunique()
            
            print(f"    Saved: {filename}")
            print(f"    Total samples: {total_samples}, Poisoned: {poisoned_samples} ({100*poisoned_samples/total_samples:.1f}%)")
            print(f"    Unique training classes: {unique_training_classes}")


def main():
    """Main function to generate poisoning data CSVs for all experiments."""
    
    # Define config files and their experiment names
    experiments = [
        ('configs/config_exp1.yaml', 'exp1'),
        ('configs/config_exp2.yaml', 'exp2'),
        ('configs/config_exp3.yaml', 'exp3'),
        ('configs/config_exp4.yaml', 'exp4'),
        ('configs/config_exp5.yaml', 'exp5'),
    ]
    
    output_dir = "./poisoning_data"
    
    print("=" * 80)
    print("Poisoning Data CSV Generator")
    print("=" * 80)
    
    for config_path, exp_name in experiments:
        if os.path.exists(config_path):
            print(f"\nProcessing {exp_name}: {config_path}")
            try:
                generate_poisoning_csv(config_path, exp_name, output_dir)
            except Exception as e:
                print(f"  Error processing {exp_name}: {e}")
        else:
            print(f"\nSkipping {exp_name}: {config_path} not found")
    
    print("\n" + "=" * 80)
    print(f"All poisoning data CSVs generated in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
