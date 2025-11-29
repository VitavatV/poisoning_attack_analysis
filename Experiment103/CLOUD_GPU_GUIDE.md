# Cloud GPU Services for Faster Experiment103 Execution

## Quick Recommendations by Budget

| Service | GPU | Cost/Hour | Best For | Setup Time |
|---------|-----|-----------|----------|------------|
| **Google Colab Pro+** | A100 | $0.00-$50/mo | Quick tests, <24hr jobs | 5 min |
| **Kaggle** | P100/T4 | FREE | Small experiments | 5 min |
| **Vast.ai** | RTX 3090 | $0.20-$0.40 | Long runs, budget | 15 min |
| **RunPod** | RTX 4090 | $0.59-$0.79 | Best performance/$ | 10 min |
| **Lambda Labs** | A100 | $1.10 | Easy setup, reliable | 15 min |
| **Paperspace** | A100 | $3.09 | Professional use | 20 min |
| **GCP** | A100 | $2.93 | Research credits | 30 min |

---

## Top Recommendations

### 1. **Vast.ai** (BEST VALUE) ⭐

**Why:** Cheapest on-demand GPUs, perfect for 6+ day experiments

**Specs:**
- RTX 3090 (24GB): $0.20-$0.40/hour
- RTX 4090 (24GB): $0.50-$0.70/hour  
- A100 (40GB): $0.90-$1.20/hour

**Cost Estimate for Full Experiment:**
- EXP 0: $12-40 (60-100 hrs @ $0.20-$0.40)
- EXP 1: $8-32 (40-80 hrs)
- EXP 2: $2-8 (12-20 hrs)
- EXP 3: $6-22 (30-55 hrs)
- EXP 4: $2-8 (12-20 hrs)
- **TOTAL: $30-110** (vs weeks on local machine)

**Setup:**
```bash
# 1. Sign up at vast.ai
# 2. Add funds ($50-100 recommended)
# 3. Search for RTX 3090 with 24GB+ RAM, 100GB+ disk
# 4. Template: PyTorch
# 5. SSH or Jupyter access
```

**Pros:**
- ✅ Extremely cheap
- ✅ Pay only for what you use
- ✅ Wide GPU selection
- ✅ Good for long experiments

**Cons:**
- ⚠️ Instances can be interrupted (choose "interruptible: no")
- ⚠️ Variable quality (check reliability score)

---

### 2. **RunPod** (BEST PERFORMANCE) ⭐

**Why:** Latest GPUs (RTX 4090), reliable, good balance

**Specs:**
- RTX 4090 (24GB): $0.59-$0.79/hour
- RTX 3090 (24GB): $0.34-$0.44/hour
- A100 (80GB): $1.89/hour

**Cost Estimate:**
- Full experiment: $90-220 @ $0.59/hr for 154-275 hours

**Setup:**
```bash
# 1. Sign up at runpod.io
# 2. Choose "Secure Cloud" for reliability
# 3. Template: PyTorch 2.0
# 4. Deploy with 100GB+ storage
```

**Pros:**
- ✅ RTX 4090 (newest, fastest for your workload)
- ✅ Very reliable (Secure Cloud)
- ✅ Easy interface
- ✅ Good documentation

**Cons:**
- ⚠️ More expensive than Vast.ai
- ⚠️ Community Cloud can be interrupted

---

### 3. **Google Colab Pro+** (EASIEST)

**Why:** Zero setup, includes storage, easiest to start

**Specs:**
- Colab Pro: T4/P100 - $10/month
- Colab Pro+: V100/A100 - $50/month
- Up to 24 hours runtime (can reconnect)

**Limitations:**
- ❌ Max 24 hours per session (need to restart)
- ❌ Background execution issues
- ⚠️ Not ideal for 6+ day experiments

**Best Use:**
- ✅ Perfect for config_test.yaml
- ✅ Good for individual EXP 1-4 (separately)
- ❌ NOT recommended for full config_definitive.yaml

**Setup:**
```python
# In Colab notebook:
!git clone <your-repo>
%cd poisoning_attack_analysis/Experiment103
!pip install -r requirements.txt
!python experiment_runner.py config_exp1.yaml
```

---

### 4. **Lambda Labs** (MOST RELIABLE)

**Why:** Dedicated servers, great for research, excellent support

**Specs:**
- A100 (40GB): $1.10/hour
- A100 (80GB): $1.29/hour
- Always available instances

**Cost Estimate:**
- Full experiment: $169-357 @ $1.10/hr

**Setup:**
```bash
# 1. Sign up at lambdalabs.com
# 2. Launch instance (A100)
# 3. Pre-installed: PyTorch, CUDA
# 4. SSH access included
```

**Pros:**
- ✅ Most reliable
- ✅ No interruptions
- ✅ Great for research
- ✅ Academic discounts available

**Cons:**
- ⚠️ More expensive
- ⚠️ Sometimes sold out

---

### 5. **Kaggle** (FREE!)

**Why:** Completely free GPUs for experimentation

**Specs:**
- P100 or T4 (16GB)
- 30 hours/week free
- 12 hour sessions

**Limitations:**
- ❌ 30 hours total per week
- ❌ 12 hour max session
- ⚠️ Good for testing only

**Best Use:**
- ✅ Run config_test.yaml (FREE)
- ✅ Test individual configs
- ❌ Can't run full experiment

---

## Quick Setup Guide (RunPod Example)

### 1. Deploy Instance

```bash
1. Go to runpod.io → Secure Cloud
2. Filter: RTX 4090, 100GB+ storage
3. Template: PyTorch 2.1
4. Deploy Pod
```

### 2. Upload Your Code

**Option A: Git (Recommended)**
```bash
# SSH into pod
git clone https://github.com/yourusername/poisoning_attack_analysis.git
cd poisoning_attack_analysis/Experiment103
```

**Option B: Direct Upload**
```bash
# Use RunPod's web interface to upload files
```

### 3. Install Dependencies

```bash
pip install torch torchvision numpy pandas matplotlib seaborn pyyaml
```

### 4. Run Experiment

```bash
# For long experiments, use screen or tmux
screen -S experiment

# Run experiment
python experiment_runner.py config_exp1.yaml

# Detach: Ctrl+A then D
# Reattach: screen -r experiment
```

### 5. Download Results

```bash
# From your local machine:
scp -r user@pod-ip:/workspace/results_exp1 ./local_results/
```

---

## Cost Comparison: 240 Experiments (Full Study)

| Service | GPU | Hours | Total Cost | Speed |
|---------|-----|-------|------------|-------|
| **Local PC** | GTX 1080 | 400-600 | $0 | 1x (baseline) |
| **Vast.ai** | RTX 3090 | 154-275 | $30-110 | 2-3x faster |
| **RunPod** | RTX 4090 | 100-180 | $59-142 | 3-4x faster |
| **Lambda** | A100 | 80-140 | $88-154 | 4-5x faster |
| **GCP** | A100 | 80-140 | $234-410 | 4-5x faster |

---

## Optimization Strategy (RECOMMENDED)

### Hybrid Approach - $50-80 total

1. **Local test** (FREE)
   ```bash
   python experiment_runner.py config_test.yaml
   ```

2. **Vast.ai for long jobs** ($30-60)
   ```bash
   # Run EXP 0, 1, 3 (longest)
   python experiment_runner.py config_exp0.yaml
   python experiment_runner.py config_exp1.yaml
   python experiment_runner.py config_exp3.yaml
   ```

3. **RunPod for critical** ($20)
   ```bash
   # Run EXP 2, 4 (highest priority, shorter)
   python experiment_runner.py config_exp2.yaml
   python experiment_runner.py config_exp4.yaml
   ```

---

## Academic/Research Credits

### Google Cloud Platform (GCP)
- **$300 free credits** for new users
- Additional research credits: https://edu.google.com/programs/credits/research/
- A100: ~100 hours free

### AWS Educate
- Up to $100 credits
- Apply: https://aws.amazon.com/education/awseducate/

### Microsoft Azure
- $100-200 credits for students
- Apply: https://azure.microsoft.com/en-us/free/students/

---

## Monitoring & Management

### Keep Experiments Running

**Use tmux or screen:**
```bash
# Start tmux session
tmux new -s exp1

# Run experiment
python experiment_runner.py config_exp1.yaml

# Detach: Ctrl+B then D
# Reattach: tmux attach -t exp1
# List sessions: tmux ls
```

### Monitor Progress Remotely

```bash
# Check logs
tail -f results_exp1/experiment.log

# Check CSV
wc -l results_exp1/final_results.csv

# Check GPU usage
nvidia-smi
```

### Auto-restart on Failure

```bash
#!/bin/bash
# run_experiment.sh
while true; do
    python experiment_runner.py config_exp1.yaml
    if [ $? -eq 0 ]; then
        break
    fi
    echo "Restarting in 60 seconds..."
    sleep 60
done
```

---

## Final Recommendation

### For Your 6.5-11.5 Day Experiment:

**Best Choice: Vast.ai RTX 3090**
- Cost: **$30-110** total
- Time: **3-5 days** (2-3x faster than local)
- Reliability: Good (choose 99%+ reliability)

**Alternative: RunPod RTX 4090** 
- Cost: **$90-160** total
- Time: **2-3 days** (3-4x faster)
- Reliability: Excellent

**Setup Time: 15-30 minutes**
**Potential Savings: 50-70% time reduction**

---

## Quick Start Checklist

- [ ] Sign up for Vast.ai or RunPod
- [ ] Add $50-100 credits
- [ ] Deploy RTX 3090/4090 instance
- [ ] Clone your repo
- [ ] Install dependencies
- [ ] Start with config_test.yaml
- [ ] Run main experiments in screen/tmux
- [ ] Monitor via logs
- [ ] Download results when complete

**Total setup: ~30 minutes for 2-5x speedup!**
