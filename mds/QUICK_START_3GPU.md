# 🚀 Quick Start Guide - 3-GPU Parallel Execution

## Summary
- **Total Experiments**: 192 (124 done, 68 remaining)
- **Estimated Time**: 18-24 hours with 3 GPUs in parallel
- **Hardware**: 3x RTX 2000 Ada (16GB VRAM), 18 vCPUs, 93GB RAM

---

## Step 1: Verify Setup

Check GPU availability:
```powershell
nvidia-smi
```

You should see 3 GPUs (cuda:0, cuda:1, cuda:2).

---

## Step 2: Open 3 Terminal Windows

Open 3 separate PowerShell or Command Prompt windows.

---

## Step 3: Run Experiments in Parallel

### Terminal 1 (GPU 0) - 24 experiments
```powershell
cd c:\github\poisoning_attack_analysis

# EXP 0 MNIST - Finish remaining 3 experiments
python experiment_runner_ext0124_mnist.py --config configs/config_exp0_mnist.yaml

# EXP 2 CIFAR - 6 experiments
python experiment_runner_ext024_cifar10.py --config configs/config_exp2_cifar10.yaml

# EXP 3 MNIST Part A - 9 experiments (batch_size=128)
python experiment_runner_ext13.py --config configs/config_exp3_mnist_gpu0.yaml

# EXP 4 CIFAR - 6 experiments
python experiment_runner_ext024_cifar10.py --config configs/config_exp4_cifar10.yaml
```

### Terminal 2 (GPU 1) - 24 experiments
```powershell
cd c:\github\poisoning_attack_analysis

# EXP 1 MNIST - 9 experiments
python experiment_runner_ext0124_mnist.py --config configs/config_exp1_mnist.yaml

# EXP 2 MNIST - 6 experiments
python experiment_runner_ext0124_mnist.py --config configs/config_exp2_mnist.yaml

# EXP 3 MNIST Part B - 9 experiments (batch_size=32)
python experiment_runner_ext13.py --config configs/config_exp3_mnist_gpu1.yaml
```

### Terminal 3 (GPU 2) - 20 experiments
```powershell
cd c:\github\poisoning_attack_analysis

# EXP 3 CIFAR - Finish remaining 5 experiments
python experiment_runner_ext13.py --config configs/config_exp3_cifar10.yaml

# EXP 3 MNIST Part C - 9 experiments (batch_size=8)
python experiment_runner_ext13.py --config configs/config_exp3_mnist_gpu2.yaml

# EXP 4 MNIST - 6 experiments
python experiment_runner_ext0124_mnist.py --config configs/config_exp4_mnist.yaml
```

---

## Step 4: Monitor Progress

### Option 1: GPU Monitoring (Terminal 4)
```powershell
# Watch GPU utilization in real-time (updates every 5 seconds)
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 5
```

### Option 2: Experiment Progress
```powershell
# Check experiment completion status
python check_progress.py
```

Run this periodically (every hour) to see progress.

---

## Expected Behavior

### GPU Utilization
- **GPU 0**: Should show 80-95% utilization, 4-8GB VRAM used
- **GPU 1**: Should show 80-95% utilization, 4-8GB VRAM used
- **GPU 2**: Should show 80-95% utilization, 4-8GB VRAM used

### Timeline
- **First 6 hours**: All GPUs active, experiments running
- **6-18 hours**: Steady progress, some experiments complete
- **18-24 hours**: Most experiments should be done
- **24-48 hours**: Conservative estimate if some experiments take longer

### Results Location
Results will be saved to:
- `results_exp0_cifar10/final_results.csv`
- `results_exp0_mnist/final_results.csv`
- `results_exp1_cifar10/final_results.csv`
- `results_exp1_mnist/final_results.csv`
- `results_exp2_cifar10/final_results.csv`
- `results_exp2_mnist/final_results.csv`
- `results_exp3_cifar10/final_results.csv`
- `results_exp3_mnist/final_results.csv`
- `results_exp4_cifar10/final_results.csv`
- `results_exp4_mnist/final_results.csv`

---

## Troubleshooting

### If GPU runs out of memory
The RTX 2000 Ada has 16GB VRAM. If you see OOM errors:
1. Check `nvidia-smi` to confirm VRAM usage
2. Configs are already optimized for 16GB, so this shouldn't happen
3. If it does, contact me for adjustment

### If experiments take longer than expected
This is normal. Early stopping will terminate some experiments early, but others may need the full 2000 rounds. The 18-24 hour estimate accounts for this variation.

### If a script crashes
1. Note which experiment/config failed
2. Check error logs in the results directory
3. Restart just that specific command
4. The script will skip already-completed experiments (checks `final_results.csv`)

---

## What to Do When Complete

Once all experiments finish (check with `python check_progress.py`):

1. **Verify all results**:
   ```powershell
   python check_progress.py
   ```
   Should show 192/192 (100%) complete

2. **Analyze results** (if you have analysis scripts):
   ```powershell
   python analyze_results_definitive.py
   ```

3. **Backup results** (optional but recommended):
   ```powershell
   # Create a results archive
   Compress-Archive -Path results_exp* -DestinationPath experiment_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip
   ```

---

## Summary of Changes Made

### New Files Created
1. ✅ `configs/config_exp3_mnist_gpu0.yaml` - EXP 3 MNIST Part A (batch_size=128)
2. ✅ `configs/config_exp3_mnist_gpu1.yaml` - EXP 3 MNIST Part B (batch_size=32)
3. ✅ `configs/config_exp3_mnist_gpu2.yaml` - EXP 3 MNIST Part C (batch_size=8)
4. ✅ `check_progress.py` - Progress monitoring script

### Files Modified
1. ✅ `configs/config_exp0_mnist.yaml` - Changed device from cuda:1 → cuda:0
2. ✅ `configs/config_exp4_mnist.yaml` - Changed device from cuda:1 → cuda:2

### Impact
- **Before**: 65 hours (GPU 2 bottleneck with 32 experiments)
- **After**: 18-24 hours (balanced: 24/24/20 experiments per GPU)
- **Speed-up**: ~63% faster! 🚀

---

## Questions?

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Run `python check_progress.py` to verify current status
3. Check `nvidia-smi` to verify GPU utilization
4. Review error logs in the results directories

Good luck with your experiments! 🎯
