# ✅ 3-GPU Setup Verification & Execution Guide

## 📋 Setup Verification Complete

All configurations and experiment runners are properly set up for **3-GPU parallel execution**!

---

## 📁 Configuration Files Status

### ✅ All Config Files Present (13 total)

#### GPU 0 Assignments (cuda:0)
- ✅ `config_exp0_mnist.yaml` - EXP 0 MNIST (3 remaining experiments)
- ✅ `config_exp2_cifar10.yaml` - EXP 2 CIFAR10 (6 experiments)
- ✅ `config_exp3_mnist_gpu0.yaml` - EXP 3 MNIST Part A (9 experiments, batch_size=128)
- ✅ `config_exp4_cifar10.yaml` - EXP 4 CIFAR10 (6 experiments)

**Total GPU 0: 24 experiments**

#### GPU 1 Assignments (cuda:1)
- ✅ `config_exp1_mnist.yaml` - EXP 1 MNIST (9 experiments)
- ✅ `config_exp2_mnist.yaml` - EXP 2 MNIST (6 experiments)
- ✅ `config_exp3_mnist_gpu1.yaml` - EXP 3 MNIST Part B (9 experiments, batch_size=32)

**Total GPU 1: 24 experiments**

#### GPU 2 Assignments (cuda:2)
- ✅ `config_exp3_cifar10.yaml` - EXP 3 CIFAR10 (5 remaining experiments)
- ✅ `config_exp3_mnist_gpu2.yaml` - EXP 3 MNIST Part C (9 experiments, batch_size=8)
- ✅ `config_exp4_mnist.yaml` - EXP 4 MNIST (6 experiments)

**Total GPU 2: 20 experiments**

#### Reference Configs (Not Used in This Run)
- `config_exp0_cifar10.yaml` - Already completed
- `config_exp1_cifar10.yaml` - Already completed
- `config_exp3_mnist.yaml` - Original file (replaced by gpu0/gpu1/gpu2 splits)

---

## 🔧 Experiment Runners Status

### ✅ All Runners Present (3 total)

1. ✅ `experiment_runner_ext0124_mnist.py` - Handles EXP 0, 1, 2, 4 for MNIST
2. ✅ `experiment_runner_ext024_cifar10.py` - Handles EXP 0, 2, 4 for CIFAR10
3. ✅ `experiment_runner_ext13.py` - Handles EXP 1, 3 for both datasets

**Features**:
- ✅ Multiprocessing support
- ✅ Early stopping
- ✅ Skip completed experiments (checks `final_results.csv`)
- ✅ Automatic result saving

---

## 🚀 Execution Instructions

### Step 1: Verify GPU Availability

Run this to confirm you have 3 GPUs:
```powershell
nvidia-smi
```

Expected output: 3 RTX 2000 Ada GPUs (cuda:0, cuda:1, cuda:2)

### Step 2: Open 3 Terminal Windows

Open 3 separate PowerShell or Command Prompt windows and navigate to the project directory in each:

```powershell
cd c:\github\poisoning_attack_analysis
```

### Step 3: Launch Experiments in Parallel

Copy and paste these commands into each terminal:

#### 🖥️ Terminal 1 (GPU 0) - 24 experiments
```powershell
# Activate environment (if needed)
# .\env313\Scripts\activate

# Run GPU 0 experiments sequentially
python experiment_runner_ext0124_mnist.py --config configs/config_exp0_mnist.yaml
python experiment_runner_ext024_cifar10.py --config configs/config_exp2_cifar10.yaml
python experiment_runner_ext13.py --config configs/config_exp3_mnist_gpu0.yaml
python experiment_runner_ext024_cifar10.py --config configs/config_exp4_cifar10.yaml

# Notify when complete
echo "GPU 0 COMPLETE!"
```

#### 🖥️ Terminal 2 (GPU 1) - 24 experiments
```powershell
# Activate environment (if needed)
# .\env313\Scripts\activate

# Run GPU 1 experiments sequentially
python experiment_runner_ext0124_mnist.py --config configs/config_exp1_mnist.yaml
python experiment_runner_ext0124_mnist.py --config configs/config_exp2_mnist.yaml
python experiment_runner_ext13.py --config configs/config_exp3_mnist_gpu1.yaml

# Notify when complete
echo "GPU 1 COMPLETE!"
```

#### 🖥️ Terminal 3 (GPU 2) - 20 experiments
```powershell
# Activate environment (if needed)
# .\env313\Scripts\activate

# Run GPU 2 experiments sequentially
python experiment_runner_ext13.py --config configs/config_exp3_cifar10.yaml
python experiment_runner_ext13.py --config configs/config_exp3_mnist_gpu2.yaml
python experiment_runner_ext0124_mnist.py --config configs/config_exp4_mnist.yaml

# Notify when complete
echo "GPU 2 COMPLETE!"
```

---

## 📊 Monitoring Progress

### Option 1: Check Progress Script
Open a 4th terminal and run periodically:
```powershell
cd c:\github\poisoning_attack_analysis
python check_progress.py
```

### Option 2: Monitor GPU Usage
```powershell
# Watch GPU utilization (updates every 5 seconds)
nvidia-smi -l 5

# Or more detailed monitoring
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv -l 5
```

### Option 3: Check Log Files
Logs are saved in each results directory:
- `results_exp0_mnist/experiment.log`
- `results_exp1_mnist/experiment.log`
- etc.

---

## 📈 Expected Timeline

| Time | Status |
|------|--------|
| **0:00** | All 3 terminals start experiments |
| **0:30** | First experiments should be completing |
| **2:00** | ~10-15% progress |
| **6:00** | ~30-40% progress |
| **12:00** | ~60-70% progress |
| **18:00** | ~90-95% progress |
| **18-24h** | ✅ All complete! |

---

## 💾 Results Location

Results will be saved to:
```
results_exp0_mnist/final_results.csv
results_exp1_mnist/final_results.csv
results_exp2_cifar10/final_results.csv
results_exp2_mnist/final_results.csv
results_exp3_cifar10/final_results.csv
results_exp3_mnist/final_results.csv
results_exp4_cifar10/final_results.csv
results_exp4_mnist/final_results.csv
```

**Note**: EXP 3 MNIST results from all 3 GPUs (gpu0, gpu1, gpu2) will be combined into the same `results_exp3_mnist/final_results.csv` file automatically.

---

## 💰 RunPod Cost Estimate

**Configuration**: 3x RTX 2000 Ada  
**Rate**: $0.25/GPU/hour  
**Runtime**: 18-24 hours  
**Total Cost**: **$13.50 - $18.00**

To monitor costs on RunPod:
1. Check RunPod dashboard for real-time usage
2. Set up billing alerts if available
3. Estimate: ~$0.75/hour while running

---

## 🔧 Troubleshooting

### If a Terminal Crashes
1. Note which config file was running
2. Check the error in the terminal output
3. Restart just that command - it will skip completed experiments
4. The runner automatically checks `final_results.csv` and skips duplicates

### If GPU Runs Out of Memory
- Configs are optimized for 16GB VRAM (RTX 2000 Ada)
- Should NOT happen with current settings
- If it does: Contact me for batch_size adjustment

### If Experiments Take Longer Than Expected
- This is normal - early stopping causes variation
- Some experiments may need full 2000 rounds
- Monitor with `python check_progress.py`

### If Results Are Missing
Check that results directories exist:
```powershell
ls results_exp*
```

If missing, they'll be created automatically on first run.

---

## ✅ Pre-Flight Checklist

Before starting, verify:
- [ ] 3 GPUs available (`nvidia-smi`)
- [ ] 3 terminals open at project directory
- [ ] Environment activated (if using virtualenv)
- [ ] All config files present (13 files in `configs/`)
- [ ] All experiment runners present (3 .py files)
- [ ] Sufficient disk space (~10GB recommended)
- [ ] RunPod instance running and connected

---

## 🎯 Quick Start Commands (Copy-Paste Ready)

### All-in-One Launch Script

Create a batch file `launch_all_gpus.bat`:
```batch
@echo off
echo Launching 3-GPU Experiment Suite
echo ================================

start "GPU 0" cmd /k "cd /d c:\github\poisoning_attack_analysis && python experiment_runner_ext0124_mnist.py --config configs/config_exp0_mnist.yaml && python experiment_runner_ext024_cifar10.py --config configs/config_exp2_cifar10.yaml && python experiment_runner_ext13.py --config configs/config_exp3_mnist_gpu0.yaml && python experiment_runner_ext024_cifar10.py --config configs/config_exp4_cifar10.yaml && echo GPU 0 COMPLETE!"

start "GPU 1" cmd /k "cd /d c:\github\poisoning_attack_analysis && python experiment_runner_ext0124_mnist.py --config configs/config_exp1_mnist.yaml && python experiment_runner_ext0124_mnist.py --config configs/config_exp2_mnist.yaml && python experiment_runner_ext13.py --config configs/config_exp3_mnist_gpu1.yaml && echo GPU 1 COMPLETE!"

start "GPU 2" cmd /k "cd /d c:\github\poisoning_attack_analysis && python experiment_runner_ext13.py --config configs/config_exp3_cifar10.yaml && python experiment_runner_ext13.py --config configs/config_exp3_mnist_gpu2.yaml && python experiment_runner_ext0124_mnist.py --config configs/config_exp4_mnist.yaml && echo GPU 2 COMPLETE!"

echo.
echo All GPU terminals launched!
echo Monitor progress with: python check_progress.py
```

Then simply run:
```powershell
.\launch_all_gpus.bat
```

---

## 📊 Final Verification

Run this before starting to verify everything:
```powershell
# Check GPUs
nvidia-smi

# Check config files
ls configs/*.yaml | measure

# Check experiment runners
ls experiment_runner_*.py | measure

# Check progress monitoring works
python check_progress.py

# Expected output:
# - nvidia-smi: 3 GPUs visible
# - Config count: 13 files
# - Runner count: 3 files
# - Progress: 0/192 (0.0%) or your current progress
```

---

## 🎉 When Complete

Once `python check_progress.py` shows 192/192 (100%):

1. **Verify Results**:
   ```powershell
   python check_progress.py
   ```

2. **Backup Results** (recommended):
   ```powershell
   $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
   Compress-Archive -Path results_exp* -DestinationPath "experiment_results_$timestamp.zip"
   ```

3. **Analyze Results** (if applicable):
   ```powershell
   python analyze_results_definitive.py
   ```

4. **Stop RunPod Instance** to save costs!

---

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review `experiment.log` files in results directories
3. Run `python check_progress.py` to see current status
4. Contact me with specific error messages

---

**Everything is ready to go! Start your 3 terminals and launch the experiments! Good luck! 🚀**
