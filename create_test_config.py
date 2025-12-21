"""
Test script to verify experiment manager auto-shutdown functionality

This creates a minimal test with just 2 tasks to verify the manager
properly shuts down when all work is complete.
"""

import yaml
import os
import shutil

# Create a test config directory
os.makedirs('test_configs', exist_ok=True)

# Create a minimal test config with just 2 experiments
test_config = {
    'defaults': {
        'project_name': 'TEST_AutoShutdown',
        'output_dir': './test_results_{dataset}',
        'load_to_memory': True,
        'num_clients': 2,
        'fraction_fit': 1.0,
        'global_rounds': 5,  # Very short for testing
        'local_epochs': 1,
        'batch_size': 32,
        'optimizer': 'sgd',
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'depth': 4,
        'validation_split': 0.1,
        'early_stopping_patience': 3,
        'min_delta': 0.0001,
        'poison_type': 'label_flip',
        'target_class': 0,
        'poison_label': 1,
        'alpha': 100.0,
        'data_ordering': 'shuffle',
        'aggregator': 'fedavg'
    },
    'seeds': [42],  # Only one seed
    'exp1_vary_width': {
        'combinations': [
            {'dataset': ['mnist']},  # Only one dataset
            {'width_factor': [1]},   # Only one width
            {'depth': [4]},
            {'poison_ratio': [0.0, 0.3]},  # Only 2 values = 2 experiments
            {'aggregator': ['fedavg']}
        ]
    }
}

# Save test config
with open('test_configs/config_exp1.yaml', 'w') as f:
    yaml.dump(test_config, f)

print("=" * 60)
print("Test Config Created")
print("=" * 60)
print("\nThis config will generate only 2 tasks:")
print("  1. mnist, width=1, poison=0.0")
print("  2. mnist, width=1, poison=0.3")
print("\nTo test auto-shutdown:")
print("\n1. Temporarily rename configs/ to configs_backup/:")
print("   mv configs configs_backup  (or rename in Windows)")
print("\n2. Rename test_configs/ to configs/:")
print("   mv test_configs configs")
print("\n3. Start the manager:")
print("   python experiment_manager.py")
print("\n4. Start a worker (in another terminal):")
print("   python experiment_runner_gpu.py")
print("\n5. Watch the manager:")
print("   - It will assign 2 tasks")
print("   - Worker will complete them quickly")
print("   - Manager should automatically shutdown with message:")
print("     'ALL TASKS COMPLETE!'")
print("\n6. Restore original configs:")
print("   mv configs test_configs")
print("   mv configs_backup configs")
print("\n" + "=" * 60)
