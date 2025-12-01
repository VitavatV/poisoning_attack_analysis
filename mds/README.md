# Experiment103: Federated Learning Poisoning Attack Analysis

Research on hyperparameter and model structure effects on poisoning attacks in federated learning.

## Overview

This experiment studies how model architecture (width and depth) affects robustness to poisoning attacks in federated learning systems using CIFAR-10.

## Prerequisites

```bash
# Python 3.7+ with required packages
pip install torch torchvision numpy pandas matplotlib seaborn pyyaml
```

## Quick Start

### 1. Test Run (Quick Verification - ~45-60 minutes)

Test with minimal configuration to verify everything works:

```bash
cd Experiment103
python experiment_runner.py config_test.yaml
```

**Test configuration:**
- 2 global rounds (vs 150 in full)
- 1 seed (vs 3 seeds)
- 3 clients (vs 10 clients)
- Small models (depth=2, width=2-4)
- ~8 total experiments
- Results: `./results_test/final_results.csv`

### 2. Full Experiment (All Phases - ~6.5-11.5 days)

Run all 5 experiment phases:

```bash
cd Experiment103
python experiment_runner.py config_definitive.yaml
```

### 3. Individual Experiments (Recommended)

Run experiments individually for better control:

```bash
# EXP 0: Preliminary Study (60-100 hours, 126 experiments)
python experiment_runner.py config_exp0.yaml

# EXP 1: High-Resolution Landscape (40-80 hours, 54 experiments)
python experiment_runner.py config_exp1.yaml

# EXP 2: Defense Benchmark (12-20 hours, 12 experiments)
python experiment_runner.py config_exp2.yaml

# EXP 3: Mechanism Analysis (30-55 hours, 36 experiments)
python experiment_runner.py config_exp3.yaml

# EXP 4: Attack Types (12-20 hours, 12 experiments)
python experiment_runner.py config_exp4.yaml
```

## Runtime Estimates

### Total: ~154-275 hours (6.5-11.5 days)

| Experiment | Description | # Runs | Est. Time | Priority |
|------------|-------------|--------|-----------|----------|
| Test | Quick verification | 8 | 45-60 min | **START HERE** |
| EXP 0 | Preliminary study (width×depth grid) | 126 | 60-100 hrs | Medium |
| EXP 1 | Fine-grained width analysis | 54 | 40-80 hrs | **HIGH** |
| EXP 2 | Defense comparison (FedAvg vs Median) | 12 | 12-20 hrs | **HIGH** |
| EXP 3 | Mechanism analysis (batch/ordering) | 36 | 30-55 hrs | Medium |
| EXP 4 | Attack type generalization | 12 | 12-20 hrs | Medium |

**Note:** Estimates assume CUDA GPU. CPU would be 5-10x slower. Early stopping may reduce time by 20-40%.

### Optimization Strategies:

1. **Skip EXP 0** - Run EXP 1-4 only → saves 60-100 hours
2. **Reduce seeds** - Use 2 seeds instead of 3 → saves 33%
3. **Lower rounds** - Use 100 instead of 150 → saves 30-40%
4. **Run phases separately** - Better monitoring and control

## Experiment Phases

### EXP 0: Preliminary Study
**Goal:** Understand how width and depth affect model capacity
- Varies: Width [1-64] × Depth [2-64]
- 42 model configurations × 3 seeds = 126 runs
- **Can skip if time-limited** - EXP 1 covers width well

### EXP 1: High-Resolution Landscape ⭐
**Goal:** Plot double descent with fine-grained width analysis
- Varies: Width [2-64], Poison [0.0, 0.3]
- 18 configurations × 3 seeds = 54 runs
- **Core experiment for main hypothesis**

### EXP 2: Defense Benchmark ⭐
**Goal:** Compare intrinsic (width) vs extrinsic (aggregation) defense
- Compares: FedAvg vs Median at different widths
- 4 configurations × 3 seeds = 12 runs
- **Key for defense evaluation**

### EXP 3: Mechanism Analysis
**Goal:** Understand why width helps (batch size, ordering effects)
- Varies: Batch size, data ordering
- 12 configurations × 3 seeds = 36 runs
- **Explains the mechanism**

### EXP 4: Attack Generalization
**Goal:** Verify results hold for different attack types
- Tests: Label flip vs random noise
- 4 configurations × 3 seeds = 12 runs
- **Validates generalization**

## Results Analysis

After experiment completion, generate plots and analysis:

```bash
python analyze_results_definitive.py
```

**Outputs:**
- `./results_definitive/plots/*.png` - Visualization plots
- `./results_definitive/plots/results_table.tex` - LaTeX table

## File Structure

```
Experiment103/
├── config_test.yaml           # Quick test
├── config_definitive.yaml     # All experiments
├── config_exp0.yaml           # EXP 0 only
├── config_exp1.yaml           # EXP 1 only
├── config_exp2.yaml           # EXP 2 only
├── config_exp3.yaml           # EXP 3 only
├── config_exp4.yaml           # EXP 4 only
├── experiment_runner.py       # Main runner
├── analyze_results_definitive.py
├── models.py
├── data_utils.py
├── utils.py
├── README.md
└── RUNTIME_ESTIMATES.md       # Detailed estimates
```

## Configuration Options

### Key Parameters

Edit config files to customize:

```yaml
defaults:
  num_clients: 10          # Number of federated clients
  global_rounds: 150       # Training rounds
  local_epochs: 5          # Local training epochs per round
  batch_size: 64           # Batch size
  lr: 0.01                 # Learning rate
  weight_decay: 5e-4       # L2 regularization
  
  # Poisoning attack
  poison_type: "label_flip"
  poison_ratio: 0.0        # Percentage of poisoned data (0.0-1.0)
  
  # Data distribution
  alpha: 100.0             # Dirichlet alpha (100=IID, 0.1=non-IID)
  aggregator: "fedavg"     # fedavg or median

seeds: [42, 101, 2024]     # Multiple seeds for statistics
```

## Monitoring Progress

### Check logs in real-time:

```bash
# For specific experiment
tail -f results_definitive/experiment.log

# Count completed experiments
wc -l results_definitive/final_results.csv
```

### Check current experiment status:

```bash
# See which experiment is running
tail -20 results_definitive/experiment.log | grep "RUNNING:"
```

## Recommended Workflow

### For Time-Limited Scenarios:

1. **Day 1:** Run test config to verify setup
   ```bash
   python experiment_runner.py config_test.yaml  # ~1 hour
   ```

2. **Day 2-3:** Run core experiments (EXP 1 & 2)
   ```bash
   python experiment_runner.py config_exp1.yaml  # ~2-3 days
   python experiment_runner.py config_exp2.yaml  # ~12-20 hours
   ```

3. **Day 4-5:** Run mechanism experiments (EXP 3 & 4)
   ```bash
   python experiment_runner.py config_exp3.yaml  # ~1-2 days
   python experiment_runner.py config_exp4.yaml  # ~12-20 hours
   ```

4. **Optional:** Run EXP 0 if time allows
   ```bash
   python experiment_runner.py config_exp0.yaml  # ~2-4 days
   ```

### For Full Study:

```bash
# All at once (6.5-11.5 days)
python experiment_runner.py config_definitive.yaml
```

## Troubleshooting

### CUDA out of memory
- Reduce `batch_size` in config
- Use smaller models (reduce `width_factor` and `depth`)
- Run experiments sequentially rather than in parallel

### Import errors
```bash
cd Experiment103
python -c "from models import *; from data_utils import *; from utils import *; print('OK')"
```

### Config validation
```bash
python -c "import yaml; c = yaml.safe_load(open('config_definitive.yaml')); print('Config valid')"
```

### Check GPU availability
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Research Questions

This experiment investigates:
- **H1**: How does model width affect poisoning attack resilience?
- **H2**: Is there a "double descent" phenomenon in federated learning?
- **H3**: What mechanisms make wide models more robust?
- **H4**: How do different aggregation methods compare?

## Notes

- First run downloads CIFAR-10 dataset (~170MB, 30-40 min)
- Results saved incrementally (safe to interrupt)
- GPU strongly recommended (CPU is 5-10x slower)
- Each seed runs sequentially for reproducibility
- Early stopping triggers when validation loss plateaus
