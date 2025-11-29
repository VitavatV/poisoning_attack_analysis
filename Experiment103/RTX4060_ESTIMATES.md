# Runtime Estimates: RTX 4060 vs Other GPUs

## GPU Performance Comparison

| GPU | VRAM | CUDA Cores | Relative Speed | Cost |
|-----|------|------------|----------------|------|
| **RTX 4060** | 8GB | 3072 | 1.0x (baseline) | You own it! |
| Current PC | ? | ? | ~0.4-0.6x | - |
| RTX 3090 | 24GB | 10496 | ~2.2x | Rent |
| RTX 4090 | 24GB | 16384 | ~3.5x | Rent |
| A100 | 40GB | 6912 | ~2.8x | Rent |

---

## RTX 4060 Runtime Estimates

### Test Config Analysis

**Current test runtime:** 42+ minutes (still downloading data on slower machine)

**On RTX 4060 (estimated):**
- Download time: Same (~30-40 min first time only)
- Training time: ~15-25 min for test config
- **Total test:** ~45-65 minutes

### Full Experiment Estimates on RTX 4060

Based on RTX 4060 being ~2x faster than typical older GPUs but ~2x slower than RTX 3090:

| Experiment | # Runs | RTX 4060 Time | Notes |
|------------|--------|---------------|-------|
| **Test** | 8 | 45-65 min | Good for verification |
| **EXP 0** | 126 | 100-160 hours (4-7 days) | Longest phase |
| **EXP 1** | 54 | 60-120 hours (2.5-5 days) | **Core experiment** |
| **EXP 2** | 12 | 18-30 hours (0.75-1.25 days) | **High priority** |
| **EXP 3** | 36 | 45-80 hours (2-3.5 days) | Medium priority |
| **EXP 4** | 12 | 18-30 hours (0.75-1.25 days) | Medium priority |
| **TOTAL** | 240 | **241-420 hours** | **10-17.5 days** |

---

## RTX 4060 Considerations

### ✅ Advantages:
- **FREE** - You already own it!
- Significantly faster than older GPUs
- Modern architecture (Ada Lovelace)
- Good power efficiency
- Can run 24/7 at lower cost than renting

### ⚠️ Limitations:
- **8GB VRAM** - May limit largest models
  - Models with W=64, D=64 might not fit
  - Reduce batch_size if OOM errors occur
- 2-3x slower than cloud RTX 4090/A100
- Still 10-17 days for full experiment

### 💡 Potential Issues:

**Memory constraints with 8GB:**
- Large models (W=64, D=32+) may exceed VRAM
- **Solution:** Modify configs to skip extreme sizes

**Recommended config adjustments for RTX 4060:**

```yaml
# For EXP 0, limit maximum size
exp0_vary_width:
  combinations:
    - width_factor: [1, 2, 4, 8, 16, 32]  # Skip 64
    - depth: [2, 4, 8, 16, 32]  # Skip 64
```

---

## Recommended Strategy with RTX 4060

### Option A: Run Everything on RTX 4060 (FREE)

**Timeline:** 10-17 days
**Cost:** $0 (electricity only)

```bash
# Week 1-2: Core experiments
python experiment_runner.py config_exp1.yaml  # 2.5-5 days
python experiment_runner.py config_exp2.yaml  # 0.75-1.25 days
python experiment_runner.py config_exp3.yaml  # 2-3.5 days
python experiment_runner.py config_exp4.yaml  # 0.75-1.25 days

# Week 2-3: Optional preliminary
python experiment_runner.py config_exp0.yaml  # 4-7 days
```

**Pros:**
- ✅ Completely FREE
- ✅ Can run 24/7
- ✅ No interruptions

**Cons:**
- ⚠️ Takes 10-17 days
- ⚠️ 8GB VRAM limits some configs

---

### Option B: Hybrid (RTX 4060 + Cloud) - RECOMMENDED

**Timeline:** 5-8 days total
**Cost:** $20-50

**Strategy:**
1. **RTX 4060 for fast experiments (FREE)**
   - EXP 2: 18-30 hours
   - EXP 4: 18-30 hours
   - **Subtotal:** ~2 days

2. **Cloud for longest experiments**
   - Vast.ai RTX 3090 for EXP 0: 60-100 hrs → $12-40
   - Vast.ai RTX 3090 for EXP 1: 40-80 hrs → $8-32
   - **Subtotal:** 3-5 days, $20-72

3. **RTX 4060 for EXP 3 (FREE)**
   - EXP 3: 45-80 hours → ~2-3 days

**Total Timeline:** 5-8 days
**Total Cost:** $20-72

---

### Option C: All Cloud (Fastest)

**Timeline:** 3-5 days
**Cost:** $30-110 (Vast.ai) or $90-220 (RunPod)

Only recommended if:
- ❌ RTX 4060 computer not available 24/7
- ❌ Need results urgently
- ✅ Budget available

---

## Memory Optimization for RTX 4060

If you hit VRAM limits, modify configs:

### 1. Reduce batch size:
```yaml
defaults:
  batch_size: 32  # Instead of 64
```

### 2. Skip extreme model sizes:
```yaml
exp0_vary_width:
  combinations:
    - width_factor: [1, 2, 4, 8, 16, 32]  # Skip 64
    - depth: [2, 4, 8, 16, 32]  # Skip 64
```

### 3. Use gradient accumulation (if needed):
Add to experiment_runner.py for effective larger batch:
```python
# Simulate batch_size=64 with 8GB VRAM
# batch_size=32, accumulation_steps=2
```

---

## Performance Monitoring

### Check if RTX 4060 is being utilized:

```bash
# On RTX 4060 computer, run:
nvidia-smi -l 1

# Should show:
# - GPU Utilization: 90-100%
# - Memory Used: 6-7.5GB (for large models)
# - Power: 110-115W
```

### Benchmark test config:

```bash
# Run test first to verify
python experiment_runner.py config_test.yaml

# Time it:
# Expected: 15-25 min training (after download)
# If much slower, check GPU usage
```

---

## Final Recommendation

### **Use RTX 4060 for Most Experiments!**

**Best approach:**
1. ✅ **Start test on RTX 4060** to verify speed
2. ✅ **Run EXP 1, 2, 3, 4 on RTX 4060** (8-12 days, FREE)
3. ⚠️ **Optional:** Use Vast.ai for EXP 0 only if time-critical ($12-40, saves 2-4 days)

**Why:**
- RTX 4060 is **fast enough** (2x faster than older GPUs)
- **Completely FREE** (vs $30-110 cloud)
- You can **run 24/7** without interruptions
- 8GB VRAM is **sufficient** with minor config tweaks
- **10-17 days** is reasonable for comprehensive study

**Total Cost:** $0-40 (vs $30-220 all-cloud)
**Total Time:** 10-17 days on RTX 4060 alone, or 6-10 days hybrid

---

## Setup on RTX 4060 Computer

```bash
# 1. Copy your code to RTX 4060 computer
# 2. Install dependencies
pip install torch torchvision numpy pandas matplotlib seaborn pyyaml

# 3. Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

# 4. Run test
python experiment_runner.py config_test.yaml

# 5. Start main experiments (use screen/tmux for long runs)
screen -S exp1
python experiment_runner.py config_exp1.yaml
# Ctrl+A then D to detach
```

Your RTX 4060 is a great middle ground - much better than renting for everything! 🎯
