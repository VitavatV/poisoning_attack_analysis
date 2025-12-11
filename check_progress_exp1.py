"""
Progress Monitoring Script for EXP1 - Fine-Grained Width Analysis
Displays real-time progress of all experiments across datasets
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

experiments = {
    "EXP 1 CIFAR": ("results_exp1_cifar10", 20),  # 5 widths × 2 poisons × 2 alphas
    "EXP 1 MNIST": ("results_exp1_mnist", 20),    # 5 widths × 2 poisons × 2 alphas
}

print(f"\n{'='*70}")
print(f"EXP1 Progress Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Goal: High-Resolution Double Descent Landscape Analysis")
print(f"{'='*70}\n")

total_completed = 0
total_expected = 0

for name, (dir_name, expected) in experiments.items():
    csv_path = Path(dir_name) / "final_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        completed = len(df)
        
        # Show breakdown by condition if there are results
        if completed > 0:
            print(f"{name:15} Details:")
            if 'poison_ratio' in df.columns and 'alpha' in df.columns:
                breakdown = df.groupby(['poison_ratio', 'alpha']).size()
                for (pr, alpha), count in breakdown.items():
                    condition = f"  poison={pr:.1f}, α={alpha:>5}"
                    print(f"    {condition}: {count} experiments")
            print()
    else:
        completed = 0
    
    total_completed += completed
    total_expected += expected
    
    progress = (completed / expected) * 100 if expected > 0 else 0
    bar_length = 35
    filled = int(bar_length * completed / expected) if expected > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"{name:15} [{bar}] {completed:3}/{expected:3} ({progress:5.1f}%)\n")

print(f"{'='*70}")
overall = (total_completed / total_expected) * 100 if total_expected > 0 else 0
print(f"OVERALL PROGRESS: {total_completed}/{total_expected} ({overall:.1f}%)")
print(f"Remaining: {total_expected - total_completed} experiments")
print(f"{'='*70}\n")

# Show experiment configuration
if total_completed < total_expected:
    print("Experiment Configuration:")
    print("  Width Factors: [64, 128, 256, 512, 1024]")
    print("  Poison Ratios: [0.0, 0.5]")
    print("  Alpha Values: [100.0, 0.1]")
    print("  Aggregator: FedAvg")
    print(f"\nEstimated Runtime: ~40-80 hours per dataset")
    print(f"Priority: HIGH - Core experiment for double descent hypothesis\n")
