"""
Migrate Existing Results to Experiment Directories

This script distributes existing results from centralized final_results.csv
to individual experiment directories (results_expN_dataset/) based on phase.
"""

import pandas as pd
import os
import yaml
import glob
import itertools
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_all_configs():
    """Load all experiment config files"""
    config_files = glob.glob('configs/config_exp*.yaml')
    configs = []
    
    for config_path in sorted(config_files):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                config['_path'] = config_path
                configs.append(config)
        except Exception as e:
            logging.error(f"Error loading {config_path}: {e}")
    
    return configs

def create_signature(config):
    """Create signature from config (without phase)"""
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

def generate_experiments(phase_config, defaults):
    """Generate experiment configurations from phase config"""
    vary_params = {}
    for item in phase_config['combinations']:
        vary_params.update(item)
    
    keys = list(vary_params.keys())
    vals = list(vary_params.values())
    
    experiments = []
    for instance in itertools.product(*vals):
        exp_setup = defaults.copy()
        
        if 'dataset' not in vary_params:
            exp_setup['dataset'] = phase_config.get('dataset', 'mnist')
        
        exp_setup['phase_name'] = phase_config.get('phase_name', 'Unknown')
        
        for k, v in zip(keys, instance):
            exp_setup[k] = v
        
        experiments.append(exp_setup)
    
    return experiments

def get_all_phase_signatures():
    """Get all experiment signatures grouped by phase"""
    configs = load_all_configs()
    
    phase_names = [
        'exp1_vary_width',
        'exp2_mechanism_analysis',
        'exp3_attack_types',
        'exp4_iid_vs_noniid',
        'exp5_defense_comparison',
    ]
    
    # Map signature -> list of phases that need it
    signature_to_phases = {}
    
    for config in configs:
        defaults = config['defaults']
        
        for phase_name in phase_names:
            if phase_name not in config:
                continue
            
            phase_cfg = config[phase_name]
            phase_cfg['phase_name'] = phase_name
            
            experiments = generate_experiments(phase_cfg, defaults)
            
            for exp in experiments:
                sig = create_signature(exp)
                if sig not in signature_to_phases:
                    signature_to_phases[sig] = []
                if phase_name not in signature_to_phases[sig]:
                    signature_to_phases[sig].append(phase_name)
    
    return signature_to_phases

def migrate_results(quiet=False):
    """Migrate results: check if signatures need duplication across phases"""
    centralized_csv = 'final_results.csv'
    
    if not os.path.exists(centralized_csv):
        if not quiet:
            logging.warning(f"{centralized_csv} not found - skipping migration")
        return
    
    # Load centralized results
    df = pd.read_csv(centralized_csv)
    original_count = len(df)
    
    # Get all possible signatures and which phases need them
    signature_to_phases = get_all_phase_signatures()
    
    # Track new rows to add
    new_rows = []
    
    for _, row in df.iterrows():
        row_sig = row['signature']
        current_phase = row['phase']
        
        # Check which other phases need this signature
        needed_phases = signature_to_phases.get(row_sig, [])
        
        for phase in needed_phases:
            if phase == current_phase:
                continue  # Already have this phase
            
            # Check if this phase+signature combo already exists
            existing = df[(df['phase'] == phase) & (df['signature'] == row_sig)]
            if len(existing) > 0:
                continue  # Already exists
            
            # Create duplicate row with different phase
            new_row = row.copy()
            new_row['phase'] = phase
            new_rows.append(new_row)
    
    if new_rows:
        # Add new rows to dataframe
        df_new = pd.DataFrame(new_rows)
        df_combined = pd.concat([df, df_new], ignore_index=True)
        
        # Save updated CSV
        df_combined.to_csv(centralized_csv, index=False)
        
        if not quiet:
            logging.info(f"Added {len(new_rows)} duplicate results to {centralized_csv}")
            logging.info(f"Total results: {original_count} → {len(df_combined)}")
            
            # Show phase distribution
            phase_counts = df_combined['phase'].value_counts()
            for phase, count in phase_counts.items():
                logging.info(f"  {phase}: {count} results")
    else:
        if not quiet:
            logging.info(f"No migration needed - all signatures already in correct phases")


if __name__ == "__main__":
    migrate_results()
