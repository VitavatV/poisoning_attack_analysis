# Runtime Estimates for Experiment103

## Test Configuration Analysis

**Current Test Status:** Running for 34+ minutes (still downloading CIFAR-10 data at 23%)

### Test Config Parameters:
- **Global rounds:** 2
- **Local epochs:** 1
- **Seeds:** 1
- **Clients:** 3
- **Model size:** Small (depth=2, width=2-4)
- **Total experiments:** ~8 configurations
- **Estimated total:** ~45-60 minutes (including 30-40 min download)

---

## Full Experiment Runtime Estimates

### Scaling Factors:

| Parameter | Test | Full | Multiplier |
|-----------|------|------|------------|
| Global rounds | 2 | 150 (with early stop ~50-80) | 25-40x |
| Local epochs | 1 | 5 | 5x |
| Seeds | 1 | 3 | 3x |
| Clients | 3 | 10 | 3.3x |
| Model sizes | Small | Various (1-64x width) | 2-10x avg |
| Experiments | 8 | ~250 total | 31x |

**Combined multiplier:** ~150-500x (accounting for early stopping and variable model sizes)

---

## Detailed Estimates by Experiment Phase

### EXP 0: Preliminary Study
**Configuration:**
- Width factors: [1, 2, 4, 8, 16, 32, 64] = 7 values
- Depths: [2, 4, 8, 16, 32, 64] = 6 values
- Total combinations: 7 × 6 = 42
- Seeds: 3
- **Total runs:** 42 × 3 = 126 experiments

**Estimated time per experiment:**
- Small models (W=1-4): ~20-30 min
- Medium models (W=8-16): ~40-60 min
- Large models (W=32-64): ~60-120 min

**Phase total:** ~60-100 hours

---

### EXP 1: Fine-Grained Width
**Configuration:**
- Width factors: [2, 4, 6, 8, 10, 12, 16, 32, 64] = 9 values
- Poison ratios: [0.0, 0.3] = 2 values
- Alpha: [0.1] = 1 value
- Total combinations: 9 × 2 = 18
- Seeds: 3
- **Total runs:** 18 × 3 = 54 experiments

**Estimated time per experiment:** ~30-90 min (avg ~45 min)

**Phase total:** ~40-80 hours

---

### EXP 2: Defense Comparison
**Configuration:**
- Aggregators: [fedavg, median] = 2 values
- Width factors: [10, 64] = 2 values  
- Poison ratio: [0.3] = 1 value
- Alpha: [0.1] = 1 value
- Total combinations: 2 × 2 = 4
- Seeds: 3
- **Total runs:** 4 × 3 = 12 experiments

**Estimated time per experiment:** ~40-100 min (avg ~60 min)

**Phase total:** ~12-20 hours

---

### EXP 3: Mechanism Analysis
**Configuration:**
- Batch sizes: [32, 128] = 2 values
- Data ordering: [shuffle, bad_good, good_bad] = 3 values
- Width factors: [10, 64] = 2 values
- Poison ratio: [0.3] = 1 value
- Total combinations: 2 × 3 × 2 = 12
- Seeds: 3
- **Total runs:** 12 × 3 = 36 experiments

**Estimated time per experiment:** ~40-90 min (avg ~55 min)

**Phase total:** ~30-55 hours

---

### EXP 4: Attack Types
**Configuration:**
- Poison types: [label_flip, random_noise] = 2 values
- Width factors: [10, 64] = 2 values
- Poison ratio: [0.3] = 1 value
- Total combinations: 2 × 2 = 4
- Seeds: 3
- **Total runs:** 4 × 3 = 12 experiments

**Estimated time per experiment:** ~40-100 min (avg ~60 min)

**Phase total:** ~12-20 hours

---

## Total Runtime Summary

| Phase | Experiments | Estimated Time |
|-------|-------------|----------------|
| EXP 0: Preliminary Study | 126 | 60-100 hours |
| EXP 1: Fine-Grained Width | 54 | 40-80 hours |
| EXP 2: Defense Comparison | 12 | 12-20 hours |
| EXP 3: Mechanism Analysis | 36 | 30-55 hours |
| EXP 4: Attack Types | 12 | 12-20 hours |
| **TOTAL** | **240** | **154-275 hours** |

### Conservative Estimate: **6.5-11.5 days** of continuous running

---

## Optimization Strategies

### To Reduce Runtime:

1. **Reduce global_rounds** (e.g., 100 instead of 150)
   - Savings: ~30-40%
   - New estimate: ~4-7 days

2. **Use fewer seeds** (e.g., 2 instead of 3)
   - Savings: ~33%
   - New estimate: ~4-8 days

3. **Skip large models in EXP 0** (e.g., width ≤ 32)
   - Savings: ~40-50 hours
   - New estimate: ~4.5-9 days

4. **Run phases separately**
   - Run EXP 1-4 only (skip EXP 0)
   - Time: ~94-175 hours (~4-7 days)

5. **Parallel execution** (if multiple GPUs)
   - Can run ~2-4x faster with 2-4 GPUs
   - New estimate: ~1.5-6 days

---

## Recommended Approach

### Option A: Full Sequential (1 GPU)
```bash
python experiment_runner.py config_definitive.yaml
```
**Time:** 6.5-11.5 days

### Option B: Skip EXP 0 (Focus on main hypotheses)
Edit config to comment out EXP 0, then:
```bash
python experiment_runner.py config_definitive.yaml
```
**Time:** 4-7 days

### Option C: Reduced Configuration
Create `config_fast.yaml` with:
- global_rounds: 100
- seeds: [42, 101]
- Skip width > 32 in EXP 0

**Time:** 3-5 days

---

## Notes

- **Early stopping** may reduce actual runtime by 20-40%
- **CUDA availability** critical (CPU would be 5-10x slower)
- Results saved **incrementally** (safe to interrupt and resume)
- Download time (30-40 min) only on first run
- **GPU memory** may limit largest models (W=64, D=64)

---

## Monitoring Commands

```bash
# Check progress
tail -f results_definitive/experiment.log

# Count completed experiments
wc -l results_definitive/final_results.csv

# Estimate remaining time based on completed
python -c "import pandas as pd; df=pd.read_csv('results_definitive/final_results.csv'); print(f'Completed: {len(df)}/240')"
```
