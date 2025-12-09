"""
Progress Monitoring Script for Federated Learning Experiments
Displays real-time progress of all experiments across datasets
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

experiments = {
    "EXP 0 CIFAR": ("results_exp0_cifar10", 48),
    "EXP 0 MNIST": ("results_exp0_mnist", 48),
    "EXP 1 CIFAR": ("results_exp1_cifar10", 9),
    "EXP 1 MNIST": ("results_exp1_mnist", 9),
    "EXP 2 CIFAR": ("results_exp2_cifar10", 6),
    "EXP 2 MNIST": ("results_exp2_mnist", 6),
    "EXP 3 CIFAR": ("results_exp3_cifar10", 27),
    "EXP 3 MNIST": ("results_exp3_mnist", 27),
    "EXP 4 CIFAR": ("results_exp4_cifar10", 6),
    "EXP 4 MNIST": ("results_exp4_mnist", 6),
}

print(f"\n{'='*60}")
print(f"Experiment Progress Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

total_completed = 0
total_expected = 0

for name, (dir_name, expected) in experiments.items():
    csv_path = Path(dir_name) / "final_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        completed = len(df)
    else:
        completed = 0
    
    total_completed += completed
    total_expected += expected
    
    progress = (completed / expected) * 100 if expected > 0 else 0
    bar_length = 30
    filled = int(bar_length * completed / expected) if expected > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"{name:15} [{bar}] {completed:3}/{expected:3} ({progress:5.1f}%)")

print(f"\n{'='*60}")
overall = (total_completed / total_expected) * 100 if total_expected > 0 else 0
print(f"OVERALL PROGRESS: {total_completed}/{total_expected} ({overall:.1f}%)")
print(f"Remaining: {total_expected - total_completed} experiments")
print(f"{'='*60}\n")

# Show GPU distribution if total is not complete
if total_completed < total_expected:
    print("GPU Distribution (Optimized Plan):")
    print("  GPU 0: 24 experiments (EXP 0 MNIST, EXP 2 CIFAR, EXP 3 MNIST Part A, EXP 4 CIFAR)")
    print("  GPU 1: 24 experiments (EXP 1 MNIST, EXP 2 MNIST, EXP 3 MNIST Part B)")
    print("  GPU 2: 20 experiments (EXP 3 CIFAR, EXP 3 MNIST Part C, EXP 4 MNIST)")
    print(f"\nEstimated time to completion: 18-24 hours (with 3 GPUs in parallel)\n")
