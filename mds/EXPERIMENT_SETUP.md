# Experiment103: Comprehensive Experimental Setup Documentation

## Research Overview

**Title:** Effects of Hyperparameters and Model Structure on Poisoning Attacks in Federated Learning

**Objective:** This experiment investigates how model architecture (specifically width and depth) affects robustness to poisoning attacks in federated learning systems, with a focus on understanding the double descent phenomenon and mechanism behind model width robustness.

**Research Questions:**
- **H1**: How does model width affect poisoning attack resilience?
- **H2**: Is there a "double descent" phenomenon in federated learning?
- **H3**: What mechanisms make wide models more robust?
- **H4**: How do different aggregation methods compare (intrinsic vs extrinsic defenses)?

---

## 1. Hardware Specifications

### 1.1 Recommended Hardware Configuration

#### GPU Setup (Recommended)

**Option A: Consumer GPU (Your Setup)**
- **GPU**: NVIDIA RTX 4060 (8GB VRAM)
- **CUDA Compute Capability**: 8.9 (Ada Lovelace architecture)
- **Power Requirements**: 115W TDP
- **Advantages**: 
  - Cost-effective (already owned)
  - Can run 24/7 continuously
  - 2x faster than older consumer GPUs
  - Modern architecture with tensor cores
- **Limitations**:
  - 8GB VRAM limits maximum model size (W=64, D=64 may not fit)
  - Requires batch size reduction for large models
  - 2-3x slower than high-end cloud GPUs

**Option B: High-Performance GPUs (Cloud Rental)**
- **RTX 3090**: 24GB VRAM, ~2.2x faster than RTX 4060
- **RTX 4090**: 24GB VRAM, ~3.5x faster than RTX 4060
- **A100**: 40GB VRAM, ~2.8x faster than RTX 4060
- Cost: $0.20-0.50/hour on Vast.ai or RunPod

**Option C: CPU-Only (Not Recommended)**
- Approximately 5-10x slower than GPU
- Total runtime: 50-110 days instead of 10-17 days
- Only use if GPU unavailable

#### System Requirements

**Current Testing Machine:**
- **OS**: Microsoft Windows 11 Pro (Build 26200)
- **Processor**: AMD Ryzen 5 4600G (or equivalent Intel i5/i7)
- **RAM**: 32GB DDR4 (minimum 16GB recommended)
- **Storage**: 100GB free space
  - CIFAR-10 dataset: ~170MB
  - Results storage: ~10-20GB (all experiments)
  - Checkpoint storage: ~5-10GB
  - Logs: ~1-2GB
- **System Type**: x64-based PC

**Minimum Requirements:**
- **RAM**: 16GB (8GB may work for smaller models)
- **Storage**: 50GB free
- **Internet**: Stable connection for dataset download (first run only)

### 1.2 GPU Performance Comparison

| Configuration | VRAM | Relative Speed | Total Runtime | Cost |
|---------------|------|----------------|----------------|------|
| **RTX 4060** (Your setup) | 8GB | 1.0x | 10-17 days | $0 |
| RTX 3090 (Vast.ai) | 24GB | 2.2x | 5-8 days | $30-72 |
| RTX 4090 (RunPod) | 24GB | 3.5x | 3-5 days | $90-220 |
| A100 (Cloud) | 40GB | 2.8x | 4-6 days | $100-250 |
| CPU Only | N/A | 0.1-0.2x | 50-110 days | $0 |

**Recommended Strategy:** Use RTX 4060 for all experiments (FREE, 10-17 days total)

### 1.3 Hardware Monitoring

**GPU Utilization Check:**
```bash
# On Windows with NVIDIA GPU
nvidia-smi -l 1

# Expected output for RTX 4060:
# - GPU Utilization: 90-100%
# - Memory Used: 6-7.5GB (for large models)
# - Power Draw: 110-115W
# - Temperature: 60-80°C
```

**CPU and Memory Monitoring:**
```powershell
# Windows Task Manager
# Or use PowerShell
Get-Process python | Select-Object -Property CPU,WorkingSet,PM

# Expected:
# - CPU: 10-30% (data loading)
# - RAM: 4-8GB per experiment
```

---

## 2. Software Environment

### 2.1 Operating System

- **Primary**: Windows 11 Pro (tested)
- **Also Compatible**: Windows 10, Linux (Ubuntu 20.04+), macOS

### 2.2 Python Environment

**Python Version:** 3.7+ (tested on Python 3.10+)

**Core Dependencies:**

| Package | Version (Tested) | Purpose |
|---------|------------------|---------|
| `torch` | 2.7.1 | Deep learning framework |
| `torchvision` | 0.22.1 | Computer vision datasets and transforms |
| `numpy` | 2.3.0 | Numerical computations |
| `pandas` | 2.3.0 | Data analysis and result logging |
| `matplotlib` | 3.10.3 | Plotting and visualization |
| `seaborn` | 0.13.2 | Statistical visualization |
| `pyyaml` | Latest | Configuration file parsing |

**CUDA Configuration:**
- **CUDA Version**: 11.8+ or 12.x (if using GPU)
- **cuDNN**: Automatically installed with PyTorch
- **Note**: CPU-only version also available (slower)

### 2.3 Installation Instructions

#### Option A: Using pip (Recommended)

```bash
# 1. Create virtual environment (recommended)
python -m venv experiment103_env

# 2. Activate environment
# Windows:
experiment103_env\Scripts\activate
# Linux/macOS:
source experiment103_env/bin/activate

# 3. Install PyTorch with CUDA support (for GPU)
# Visit https://pytorch.org/get-started/locally/ for your specific configuration
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision

# 4. Install other dependencies
pip install numpy pandas matplotlib seaborn pyyaml

# 5. Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### Option B: Using conda

```bash
# 1. Create conda environment
conda create -n experiment103 python=3.10

# 2. Activate environment
conda activate experiment103

# 3. Install dependencies
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install numpy pandas matplotlib seaborn pyyaml -c conda-forge
```

### 2.4 Environment Verification

```bash
# Navigate to experiment directory
cd d:\github\poisoning_attack_analysis\Experiment103

# Test imports
python -c "from models import *; from data_utils import *; from utils import *; print('All imports OK')"

# Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Validate configuration files
python -c "import yaml; c = yaml.safe_load(open('config_definitive.yaml')); print('Config valid')"
```

---

## 3. Dataset Preparation

### 3.1 Dataset: CIFAR-10

**Description:**
- **Name**: CIFAR-10 (Canadian Institute For Advanced Research)
- **Type**: Image classification
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training Images**: 50,000
- **Test Images**: 10,000
- **Image Size**: 32×32 RGB
- **File Size**: ~170MB

**Automatic Download:**
- Dataset is automatically downloaded on first run
- Stored in: `./data/` directory
- Download time: 30-40 minutes (first run only)
- Source: Official PyTorch torchvision datasets

### 3.2 Data Preprocessing

**Transformations Applied:**

```python
# Training data transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # Data augmentation
    transforms.RandomHorizontalFlip(),         # Random horizontal flip
    transforms.ToTensor(),                     # Convert to tensor
    transforms.Normalize(                      # Normalize to ImageNet stats
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

# Test data transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])
```

### 3.3 Data Distribution

**Federated Learning Partitioning:**

**Method**: Dirichlet Distribution with parameter α
- Controls non-IID-ness of data across clients
- **α = 100.0**: IID (Independent and Identically Distributed)
- **α = 10.0**: Slightly non-IID
- **α = 1.0**: Moderately non-IID
- **α = 0.1**: Highly non-IID (realistic FL scenario)

**Default Configuration:**
- **Number of Clients**: 10
- **Data Split**: Dirichlet(α) partitioning
- **Validation Split**: 10% of training data per client
- **Client Participation**: 100% per round (fraction_fit=1.0)

### 3.4 Poisoning Attack

**Attack Types:**

**1. Label Flip Attack (Primary)**
```yaml
poison_type: "label_flip"
target_class: 0        # Source class (e.g., airplane)
poison_label: 1        # Target class (e.g., automobile)
poison_ratio: 0.0-0.5  # Percentage of poisoned data
```
- Flips labels of target_class to poison_label
- Simulates backdoor/targeted poisoning
- Used in EXP 0, 1, 2, 3

**2. Random Noise Attack**
```yaml
poison_type: "random_noise"
poison_ratio: 0.0-0.5  # Percentage of poisoned data
```
- Randomly changes labels to other classes
- Untargeted attack
- Used in EXP 4 for generalization testing

**Poisoning Ratio Levels:**
- 0.0: Clean (no attack)
- 0.1-0.25: Light poisoning
- 0.3: Moderate poisoning (default for attack scenarios)
- 0.5: Heavy poisoning

---

## 4. Model Architecture

### 4.1 ScalableCNN Architecture

**Design Philosophy:**
- Scalable width and depth for experimental control
- ResNet-inspired structure without skip connections
- Designed to study overparameterization effects

**Architecture Components:**

```python
class ScalableCNN(nn.Module):
    def __init__(self, num_classes=10, width_factor=1, depth=4):
        """
        Args:
            num_classes: 10 (CIFAR-10)
            width_factor: Channel width multiplier [1, 2, 4, 8, 16, 32, 64]
            depth: Number of convolutional layers [2, 4, 8, 16, 32, 64]
        """
```

**Layer Structure:**
```
Input (3×32×32)
    ↓
[Depth × Convolutional Blocks]:
    Conv2d(in_channels, out_channels=width_factor, kernel=3, padding=1)
    BatchNorm2d(out_channels)
    ReLU()
    [MaxPool2d(2,2) every 2 layers if spatial_dim > 1]
    ↓
Flatten
    ↓
Linear(flatten_dim, num_classes=10)
    ↓
Output (10 classes)
```

**Channel Scaling:**
- Base channels: 1
- Output channels per layer: `base_channels × width_factor`
- Constant width across all layers (flat architecture)

**Spatial Dimension Reduction:**
- Input: 32×32
- MaxPool2d applied every 2 layers
- Final spatial dimension: depends on depth
  - Depth=2: 32×32
  - Depth=4: 16×16
  - Depth=8: 8×8
  - Depth=16: 4×4
  - Depth=32: 2×2

### 4.2 Model Configurations

**Parameter Range:**

| Configuration | Width Factor (W) | Depth (D) | Approx. Parameters | VRAM Usage |
|---------------|------------------|-----------|-------------------|------------|
| Tiny | 1-2 | 2-4 | ~1K-10K | <1GB |
| Small | 4-8 | 4-8 | ~10K-100K | 1-2GB |
| Medium | 10-16 | 4-16 | ~100K-1M | 2-4GB |
| Large | 32 | 4-32 | ~1M-10M | 4-6GB |
| Very Large | 64 | 4-32 | ~10M-50M | 6-8GB |

**Experiment-Specific Configurations:**

**EXP 0 (Width × Depth Grid):**
- Width: [1, 2, 4, 8, 16, 32]
- Depth: [2, 4, 8, 16, 32]
- 30 combinations × 3 poison levels = 90 configs

**EXP 1 (Fine-Grained Width):**
- Width: [2, 4, 6, 8, 10, 12, 16, 32, 64]
- Depth: 4 (fixed)
- 9 widths × 2 poison × 1 alpha = 18 configs

**EXP 2 (Defense Comparison):**
- Width: [10, 64] (critical vs robust)
- Depth: 4 (fixed)
- 2 widths × 2 aggregators = 4 configs

**EXP 3 (Mechanism Analysis):**
- Width: [10, 64]
- Depth: 4 (fixed)
- 2 widths × 2 batch × 3 ordering = 12 configs

**EXP 4 (Attack Types):**
- Width: [10, 64]
- Depth: 4 (fixed)
- 2 widths × 2 attacks = 4 configs

### 4.3 Memory Optimization for RTX 4060 (8GB)

**If encountering CUDA Out of Memory errors:**

```yaml
# Reduce batch size
defaults:
  batch_size: 32  # Default (can reduce to 16 if needed)

# Limit maximum model size
exp0_vary_width:
  combinations:
    - width_factor: [1, 2, 4, 8, 16, 32]  # Skip 64
    - depth: [2, 4, 8, 16, 32]  # Skip 64
```

---

## 5. Experimental Design

### 5.1 Federated Learning Protocol

**Training Protocol:**
```yaml
defaults:
  num_clients: 10                    # Number of federated clients
  fraction_fit: 1.0                  # 100% client participation
  global_rounds: 150                 # Communication rounds
  local_epochs: 5                    # Local training epochs per round
  batch_size: 32                     # Mini-batch size
  
  # Optimizer settings
  optimizer: "sgd"                   # SGD optimizer
  lr: 0.01                           # Learning rate
  momentum: 0.9                      # SGD momentum
  weight_decay: 5e-4                 # L2 regularization
  
  # Early stopping
  validation_split: 0.1              # 10% validation data
  early_stopping_patience: 20        # Stop if no improvement for 20 rounds
  min_delta: 0.0001                  # Minimum improvement threshold
```

**Aggregation Methods:**
1. **FedAvg (Federated Averaging)**: Weighted average of client models
2. **Median**: Component-wise median of client updates (Byzantine-robust)

### 5.2 Experimental Phases

#### EXP 0: Preliminary Study (Optional)
**Goal:** Understand width vs depth scaling effects
- **Runtime**: 100-160 hours (4-7 days) on RTX 4060
- **Experiments**: 90 configurations × 3 seeds = 270 runs
- **Priority**: Medium (can skip if time-limited)
- **Variables**:
  - Width: [1, 2, 4, 8, 16, 32]
  - Depth: [2, 4, 8, 16, 32]
  - Poison: [0.0, 0.25, 0.5]

#### EXP 1: High-Resolution Landscape ⭐ (CORE)
**Goal:** Plot double descent with fine-grained width analysis
- **Runtime**: 60-120 hours (2.5-5 days) on RTX 4060
- **Experiments**: 18 configurations × 3 seeds = 54 runs
- **Priority**: **HIGH** (core hypothesis testing)
- **Variables**:
  - Width: [2, 4, 6, 8, 10, 12, 16, 32, 64]
  - Poison: [0.0, 0.3]
  - Alpha: [0.1] (highly non-IID)

#### EXP 2: Defense Benchmark ⭐
**Goal:** Compare intrinsic (width) vs extrinsic (aggregation) defenses
- **Runtime**: 18-30 hours (0.75-1.25 days) on RTX 4060
- **Experiments**: 4 configurations × 3 seeds = 12 runs
- **Priority**: **HIGH** (defense evaluation)
- **Variables**:
  - Width: [10, 64]
  - Aggregator: [fedavg, median]
  - Poison: [0.3]

#### EXP 3: Mechanism Analysis
**Goal:** Understand why width helps (batch size, data ordering)
- **Runtime**: 45-80 hours (2-3.5 days) on RTX 4060
- **Experiments**: 12 configurations × 3 seeds = 36 runs
- **Priority**: Medium (mechanistic understanding)
- **Variables**:
  - Width: [10, 64]
  - Batch size: [32, 128]
  - Data ordering: [shuffle, bad_good, good_bad]
  - Poison: [0.3]

#### EXP 4: Attack Type Generalization
**Goal:** Verify results hold for different attack types
- **Runtime**: 18-30 hours (0.75-1.25 days) on RTX 4060
- **Experiments**: 4 configurations × 3 seeds = 12 runs
- **Priority**: Medium (generalization check)
- **Variables**:
  - Width: [10, 64]
  - Attack type: [label_flip, random_noise]
  - Poison: [0.3]

### 5.3 Randomization and Reproducibility

**Multiple Seeds:**
```yaml
seeds: [42, 101, 2024]  # 3 independent runs per configuration
```

**Purpose:**
- Compute mean and standard deviation
- Statistical significance testing
- Error bars in plots
- Verify robustness of findings

**Random Components:**
- Model initialization
- Data partitioning
- Client sampling (if fraction_fit < 1.0)
- Data augmentation
- SGD optimization

### 5.4 Total Experiment Scope

| Phase | Configurations | Seeds | Total Runs | Est. Time (RTX 4060) |
|-------|----------------|-------|------------|----------------------|
| **Test** | 8 | 1 | 8 | 45-65 min |
| **EXP 0** | 90 | 3 | 270 | 100-160 hrs |
| **EXP 1** | 18 | 3 | 54 | 60-120 hrs |
| **EXP 2** | 4 | 3 | 12 | 18-30 hrs |
| **EXP 3** | 12 | 3 | 36 | 45-80 hrs |
| **EXP 4** | 4 | 3 | 12 | 18-30 hrs |
| **TOTAL** | 136 | 3 | 384 | **241-420 hrs** |

**Total Runtime on RTX 4060:** 10-17.5 days (running 24/7)

---

## 6. Running the Experiments

### 6.1 Pre-Run Checklist

```bash
# 1. Navigate to experiment directory
cd d:\github\poisoning_attack_analysis\Experiment103

# 2. Verify environment
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 3. Test imports
python -c "from models import *; from data_utils import *; from utils import *; print('OK')"

# 4. Validate configuration
python -c "import yaml; yaml.safe_load(open('config_exp1.yaml'))"

# 5. Check disk space
# Ensure at least 50GB free
```

### 6.2 Quick Test Run (RECOMMENDED FIRST)

```bash
# Test configuration (45-65 minutes)
python experiment_runner.py config_test.yaml
```

**Expected Output:**
```
[INFO] Configuration loaded successfully
[INFO] Starting Experiment 1/8
[INFO] Seed: 42, Phase: exp0_vary_width, Config: {...}
[INFO] Downloading CIFAR-10 dataset... (first run only)
[INFO] Round 1/2: Train Loss=2.31, Val Loss=2.30, Val Acc=10.2%
[INFO] Round 2/2: Train Loss=2.15, Val Loss=2.20, Val Acc=18.5%
[INFO] Experiment 1/8 completed (2 rounds, 8.5 min)
...
[INFO] All experiments completed! Results: ./results_test/final_results.csv
```

**Verify Test Results:**
```bash
# Check results file
type results_test\final_results.csv

# Should contain 8 rows (experiments) + 1 header
```

### 6.3 Running Individual Experiments

**Option A: Sequential Execution (Recommended)**

```bash
# Week 1: Core experiments (EXP 1 + 2)
python experiment_runner.py config_exp1.yaml  # 2.5-5 days
python experiment_runner.py config_exp2.yaml  # 0.75-1.25 days

# Week 2: Mechanism experiments (EXP 3 + 4)
python experiment_runner.py config_exp3.yaml  # 2-3.5 days
python experiment_runner.py config_exp4.yaml  # 0.75-1.25 days

# Optional: Preliminary study
python experiment_runner.py config_exp0.yaml  # 4-7 days
```

**Option B: All-in-One Execution**

```bash
# Run all experiments sequentially
python experiment_runner.py config_definitive.yaml  # 10-17.5 days
```

### 6.4 Long-Running Execution Tips

**Windows: Using background processes**

```bash
# Option 1: Start with nohup equivalent (PowerShell)
Start-Process python -ArgumentList "experiment_runner.py config_exp1.yaml" -NoNewWindow -RedirectStandardOutput "exp1_log.txt"

# Option 2: Use Windows Task Scheduler for persistence

# Option 3: Keep terminal open (simpler but requires active session)
python experiment_runner.py config_exp1.yaml
```

**Linux/macOS: Using screen or tmux**

```bash
# Using screen
screen -S exp1
python experiment_runner.py config_exp1.yaml
# Detach: Ctrl+A then D
# Reattach: screen -r exp1

# Using tmux
tmux new -s exp1
python experiment_runner.py config_exp1.yaml
# Detach: Ctrl+B then D
# Reattach: tmux attach -t exp1
```

### 6.5 Monitoring Progress

**Real-time Log Monitoring:**

```bash
# Windows PowerShell
Get-Content results_exp1\experiment.log -Wait -Tail 50

# Or use any text editor to view
notepad results_exp1\experiment.log
```

**Check Completion Status:**

```bash
# Count completed experiments
python -c "import pandas as pd; df = pd.read_csv('results_exp1/final_results.csv'); print(f'Completed: {len(df)} experiments')"

# View latest results
python -c "import pandas as pd; df = pd.read_csv('results_exp1/final_results.csv'); print(df.tail())"
```

**GPU Monitoring (if using NVIDIA GPU):**

```bash
# Real-time monitoring
nvidia-smi -l 1

# Check temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -l 1
```

**Check Current Running Experiment:**

```bash
# View last 20 log lines
Get-Content results_exp1\experiment.log -Tail 20 | Select-String "RUNNING:"
```

### 6.6 Handling Interruptions

**Safe Interruption:**
- Results are saved incrementally after each experiment
- Safe to stop between experiments (not during)
- Press `Ctrl+C` to stop
- Already completed experiments are saved in `final_results.csv`

**Resuming After Interruption:**
- Simply re-run the same command
- The system will continue from the next experiment
- Already completed experiments are NOT re-run

---

## 7. Results and Output

### 7.1 Output Directory Structure

```
Experiment103/
├── results_test/               # Test run results
│   ├── final_results.csv
│   └── experiment.log
├── results_exp0/               # EXP 0 results
│   ├── final_results.csv
│   ├── experiment.log
│   └── checkpoints/
├── results_exp1/               # EXP 1 results
│   ├── final_results.csv
│   ├── experiment.log
│   └── checkpoints/
├── results_exp2/               # EXP 2 results
├── results_exp3/               # EXP 3 results
├── results_exp4/               # EXP 4 results
└── results_definitive/         # All experiments (if running config_definitive.yaml)
    ├── final_results.csv
    ├── experiment.log
    ├── plots/                  # Generated after analysis
    │   ├── exp1_double_descent.png
    │   ├── exp2_defense_comparison.png
    │   ├── exp3_batch_size_effect.png
    │   ├── exp4_attack_types.png
    │   └── results_table.tex
    └── checkpoints/
```

### 7.2 Results File Structure

**final_results.csv columns:**
```csv
seed,experiment,dataset,width_factor,depth,poison_ratio,alpha,poison_type,
aggregator,batch_size,data_ordering,final_round,best_val_acc,final_val_acc,
best_test_acc,final_test_acc,model_params,total_time_sec
```

**Example row:**
```csv
42,exp1_fine_grained_width,cifar10,16,4,0.3,0.1,label_flip,fedavg,32,shuffle,
150,75.3,74.8,72.1,71.5,2145792,3847.2
```

### 7.3 Log File Format

**experiment.log example:**
```
2025-12-01 06:00:00 [INFO] Starting experiment runner
2025-12-01 06:00:01 [INFO] Configuration: config_exp1.yaml
2025-12-01 06:00:01 [INFO] Total experiments: 54 (18 configs × 3 seeds)
2025-12-01 06:00:02 [INFO] Device: cuda (NVIDIA GeForce RTX 4060)
2025-12-01 06:00:10 [INFO] RUNNING: Experiment 1/54
2025-12-01 06:00:10 [INFO] Seed=42, W=2, D=4, Poison=0.0, Alpha=0.1
2025-12-01 06:05:45 [INFO] Round 1/150: Train Loss=2.31, Val Acc=10.2%
...
2025-12-01 08:30:22 [INFO] Experiment 1/54 completed (150 rounds, 150.2 min)
2025-12-01 08:30:22 [INFO] Best Val Acc: 68.5%, Final Test Acc: 65.2%
```

### 7.4 Results Analysis

**After experiments complete, generate visualizations:**

```bash
python analyze_results_definitive.py
```

**Generated Plots:**
1. **EXP 1**: Double descent curve (width vs accuracy, clean vs poisoned)
2. **EXP 2**: Defense comparison (FedAvg vs Median at different widths)
3. **EXP 3**: Batch size and ordering effects
4. **EXP 4**: Attack type generalization
5. **Summary Table**: LaTeX-formatted results table

**Plots Location:**
- `./results_definitive/plots/*.png`
- `./results_definitive/plots/results_table.tex`

### 7.5 Expected Results

**Key Findings:**
1. **Double Descent**: Accuracy peaks at critical width, drops in interpolation regime, recovers in overparameterized regime
2. **Width Robustness**: Wide models (W=32-64) more robust to poisoning than critical models (W=8-12)
3. **Defense Comparison**: Wide models with FedAvg outperform narrow models with Median
4. **Batch Effect**: Larger batches stabilize training, reduce poisoning impact
5. **Ordering Effect**: "good_bad" ordering (clean data first) improves robustness

---

## 8. Troubleshooting

### 8.1 Common Errors

**CUDA Out of Memory:**
```bash
# Error: RuntimeError: CUDA out of memory
# Solution 1: Reduce batch size
# Edit config file:
defaults:
  batch_size: 16  # Reduced from 32

# Solution 2: Skip very large models
exp0_vary_width:
  combinations:
    - width_factor: [1, 2, 4, 8, 16]  # Remove 32, 64
```

**Import Errors:**
```bash
# Error: ModuleNotFoundError: No module named 'torch'
# Solution: Install dependencies
pip install torch torchvision numpy pandas matplotlib seaborn pyyaml
```

**Configuration Validation Error:**
```bash
# Error: KeyError or missing parameter
# Solution: Check config file syntax
python -c "import yaml; print(yaml.safe_load(open('config_exp1.yaml')))"
```

**Dataset Download Timeout:**
```bash
# Error: URLError or timeout during CIFAR-10 download
# Solution 1: Retry (network issue)
# Solution 2: Manual download
# Download from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# Extract to: ./data/cifar-10-batches-py/
```

**Slow Training (CPU instead of GPU):**
```bash
# Check if GPU is being used
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 8.2 Performance Issues

**Training Too Slow:**
1. Check GPU utilization: `nvidia-smi`
2. Reduce model size or batch size
3. Enable early stopping (already enabled by default)
4. Use test config to benchmark speed

**Disk Space Full:**
```bash
# Check disk space
dir results_exp1

# Clean up old checkpoints (if needed)
rmdir /s results_test
```

**Memory Leak (RAM):**
```bash
# If RAM usage grows over time:
# Restart experiment
# Usually not an issue for GPU training
```

---

## 9. Configuration Files

### 9.1 Available Configurations

| File | Purpose | Experiments | Seeds | Est. Runtime (RTX 4060) |
|------|---------|-------------|-------|------------------------|
| `config_test.yaml` | Quick verification | 8 | 1 | 45-65 min |
| `config_exp0.yaml` | Preliminary study | 90 | 3 | 100-160 hrs |
| `config_exp1.yaml` | Core experiment | 18 | 3 | 60-120 hrs |
| `config_exp2.yaml` | Defense benchmark | 4 | 3 | 18-30 hrs |
| `config_exp3.yaml` | Mechanism analysis | 12 | 3 | 45-80 hrs |
| `config_exp4.yaml` | Attack generalization | 4 | 3 | 18-30 hrs |
| `config_definitive.yaml` | All experiments | 128 | 3 | 241-420 hrs |

### 9.2 Configuration Customization

**Example: Modify EXP 1 for faster testing:**

```yaml
# config_exp1.yaml
defaults:
  global_rounds: 50     # Reduced from 150
  batch_size: 64        # Increased from 32 (faster, uses more VRAM)

seeds: [42]             # Single seed instead of 3

exp1_fine_grained_width:
  combinations:
    - width_factor: [4, 16, 64]  # Reduced from 9 widths
    - poison_ratio: [0.0, 0.3]
    - alpha: [0.1]
```

**Example: Add new experiment phase:**

```yaml
# Add to any config file
exp5_custom:
  dataset: "cifar10"
  combinations:
    - width_factor: [8, 16, 32]
    - depth: [4, 8]
    - poison_ratio: [0.2, 0.4]
    - aggregator: ["fedavg", "median"]
```

---

## 10. Additional Resources

### 10.1 Documentation Files

- **README.md**: Quick start guide and basic usage
- **RTX4060_ESTIMATES.md**: Detailed GPU performance analysis
- **RUNTIME_ESTIMATES.md**: Comprehensive runtime calculations
- **CLOUD_GPU_GUIDE.md**: Cloud GPU rental instructions
- **RTX4060_READY.md**: RTX 4060 setup guide

### 10.2 Code Structure

**Core Modules:**
- `experiment_runner.py`: Main experiment orchestration
- `models.py`: ScalableCNN model definition
- `data_utils.py`: Dataset loading, partitioning, poisoning
- `utils.py`: Training loop, evaluation, aggregation
- `analyze_results_definitive.py`: Results visualization

### 10.3 Research Context

**Related Work:**
- Federated Learning (McMahan et al., 2017)
- Byzantine-Robust Aggregation (Blanchard et al., 2017)
- Double Descent Phenomenon (Belkin et al., 2019)
- Poisoning Attacks in FL (Bagdasaryan et al., 2020)

**Hypothesis:**
- **H1**: Model width provides intrinsic robustness to poisoning attacks
- **H2**: Double descent curve shifts under poisoning attack
- **H3**: Wide models average out poisoned updates better (gradient space theory)
- **H4**: Architectural defense (width) can outperform algorithmic defense (median)

---

## 11. Quick Reference

### 11.1 Command Cheat Sheet

```bash
# Quick test (1 hour)
python experiment_runner.py config_test.yaml

# Core experiments (priority)
python experiment_runner.py config_exp1.yaml  # 2-5 days
python experiment_runner.py config_exp2.yaml  # 1 day

# Full study (all experiments)
python experiment_runner.py config_definitive.yaml  # 10-17 days

# Generate plots and analysis
python analyze_results_definitive.py

# Monitor progress
Get-Content results_exp1\experiment.log -Wait -Tail 20

# Check GPU
nvidia-smi -l 1

# Verify environment
python -c "import torch; print(torch.cuda.is_available())"
```

### 11.2 Timeline Recommendations

**Week 1: Setup and Testing**
- Day 1: Environment setup, verify GPU, run test config
- Day 2-6: EXP 1 (core hypothesis)
- Day 7: EXP 2 (defense comparison)

**Week 2: Extended Experiments**
- Day 8-11: EXP 3 (mechanism analysis)
- Day 12: EXP 4 (generalization)
- Day 13-14: Results analysis and visualization

**Optional Week 3: Comprehensive Study**
- Day 15-21: EXP 0 (preliminary width×depth grid)

### 11.3 Critical Parameters

| Parameter | Impact | Recommended Range |
|-----------|--------|-------------------|
| `width_factor` | Model capacity, robustness | 2-64 |
| `depth` | Model capacity, computation | 2-32 |
| `poison_ratio` | Attack strength | 0.0-0.5 |
| `alpha` | Data heterogeneity | 0.1-100.0 |
| `global_rounds` | Convergence | 50-150 |
| `batch_size` | Memory, speed, stability | 16-128 |
| `lr` | Convergence speed | 0.001-0.1 |

---

## 12. Contact and Support

**For questions or issues:**
- Review documentation files in `Experiment103/`
- Check troubleshooting section above
- Verify environment with test config
- Review experiment logs for error messages

**Citation:**
If using this experimental setup in research, please cite:
```bibtex
@article{experiment103,
  title={Effects of Hyperparameters and Model Structure on Poisoning Attacks 
         in Federated Learning},
  author={[Your Name]},
  journal={[Venue]},
  year={2025}
}
```

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-01  
**Experiment Code Version:** Definitive (Final)  
**Hardware Tested On:** NVIDIA RTX 4060 (8GB), Windows 11 Pro, 32GB RAM
