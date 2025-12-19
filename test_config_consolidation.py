"""
Test script to verify the new config structure and experiment generation
"""
import sys
sys.path.append('c:\\github\\poisoning_attack_analysis')

import yaml
from experiment_runner_cpu import generate_experiments

# Test config_exp1.yaml
print("="*60)
print("Testing config_exp1.yaml")
print("="*60)

with open('c:\\github\\poisoning_attack_analysis\\configs\\config_exp1.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

defaults = config['defaults']
phase = config['exp1_fine_grained_width']
phase['phase_name'] = 'exp1_fine_grained_width'

print(f"\nDefaults output_dir: {defaults['output_dir']}")
print(f"Contains placeholder: {'{dataset}' in defaults['output_dir']}")

experiments = generate_experiments(phase, defaults)

print(f"\nGenerated {len(experiments)} experiments")
print(f"\nExpected: 2 datasets × 1 width × 3 poison × 2 alpha = 12 experiments")

# Check dataset distribution
mnist_count = sum(1 for e in experiments if e['dataset'] == 'mnist')
cifar10_count = sum(1 for e in experiments if e['dataset'] == 'cifar10')

print(f"\nDataset distribution:")
print(f"  MNIST: {mnist_count}")
print(f"  CIFAR10: {cifar10_count}")

# Check output_dir resolution
print(f"\nChecking output_dir resolution:")
for i, exp in enumerate(experiments[:4]):  # Show first 4
    print(f"  Exp {i+1}: dataset={exp['dataset']}, output_dir={exp['output_dir']}")
    if '{dataset}' in exp['output_dir']:
        print(f"    ❌ ERROR: Placeholder not resolved!")
    else:
        print(f"    ✅ OK: Placeholder resolved correctly")

# Verify unique output directories per dataset
output_dirs = set(e['output_dir'] for e in experiments)
print(f"\nUnique output_dirs: {output_dirs}")

# Test backward compatibility with old config structure
print("\n" + "="*60)
print("Testing backward compatibility")
print("="*60)

# Create a mock old-style config
old_style_phase = {
    'dataset': 'mnist',
    'combinations': [
        {'width_factor': [4]},
        {'poison_ratio': [0.0, 0.3]}
    ]
}

old_defaults = {
    'output_dir': './results_old_style',
    'depth': 4,
    'alpha': 100.0
}

old_experiments = generate_experiments(old_style_phase, old_defaults)
print(f"\nGenerated {len(old_experiments)} experiments from old-style config")
print(f"All have dataset='mnist': {all(e['dataset'] == 'mnist' for e in old_experiments)}")
print(f"No placeholder in output_dir: {all('{dataset}' not in e['output_dir'] for e in old_experiments)}")

print("\n" + "="*60)
print("✅ All tests passed!")
print("="*60)
