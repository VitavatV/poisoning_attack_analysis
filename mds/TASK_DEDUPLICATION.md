# Task Deduplication

## Overview

Added automatic task deduplication to prevent running the same experiment multiple times if it appears in different config files.

## Problem

If you have overlapping configurations across multiple config files:

**config_exp1.yaml:**
```yaml
combinations:
  - model_type: ["cnn", "lr"]
  - dataset: ["mnist"]
  - width_factor: [1, 4]
  - poison_ratio: [0.0]
```

**config_exp2.yaml:**
```yaml
combinations:
  - model_type: ["cnn", "lr"]  
  - dataset: ["mnist"]
  - width_factor: [1]          # Overlaps with exp1
  - poison_ratio: [0.0]        # Overlaps with exp1
```

**Without deduplication:**
- Experiment (mnist, cnn, w=1, poison=0) runs **twice**
- Wastes time and GPU resources

## Solution

Use task signatures to identify and skip duplicates:

```python
# Task signature includes all experiment parameters
signature = "exp1|mnist|cnn|1|4|0.0|label_flip|100.0|shuffle|fedavg|64"

# Before adding task, check if signature already exists
if signature in task_signatures:
    skip  # Duplicate!
else:
    add task
    register signature
```

## How It Works

### 1. Task Signature

Created from all experiment parameters:
- Phase name
- Dataset
- Model type
- Width factor
- Depth
- Poison ratio
- Poison type
- Alpha (data distribution)
- Data ordering
- Aggregator
- Batch size
- **Seed** (added to config for deduplication)

### 2. Deduplication Process

When generating tasks:

1. Create task signature
2. Check if signature exists in `task_signatures` dict
3. If exists → **skip**, log as duplicate
4. If new → **add** task, register signature

### 3. Logging

Manager logs:
```
Generated 240 unique tasks
Skipped 60 duplicate tasks across configs
```

## Example

**Before Deduplication:**
```
Config 1: 10 tasks
Config 2: 10 tasks (5 overlap with Config 1)
Total: 20 tasks (5 duplicates)
```

**After Deduplication:**
```
Config 1: 10 tasks added
Config 2: 5 tasks added (5 duplicates skipped)
Total: 15 unique tasks
```

**Savings: 25% fewer experiments**

## Benefits

✅ **No Wasted Work**: Each experiment runs exactly once  
✅ **Automatic**: No manual config checking needed  
✅ **Transparent**: Logs how many duplicates skipped  
✅ **Flexible**: Works across any number of config files  
✅ **Safe**: Uses exact parameter matching

## Use Cases

### Use Case 1: Overlapping Grid Searches

**Scenario**: Exp1 does wide grid search, Exp2 focuses on subset

- Exp1: width=[1, 4, 16, 64], poison=[0.0, 0.3, 0.5]
- Exp2: width=[4], poison=[0.0, 0.3, 0.5] with extra parameters

**Result**: Shared (width=4, poison=X) configs run once

### Use Case 2: Multiple Configs for Organization

**Scenario**: Separate configs for different datasets but same architecture

- config_mnist.yaml: MNIST experiments
- config_cifar.yaml: CIFAR experiments
- Both test same models

**Result**: If any overlap, deduplicated automatically

### Use Case 3: Incremental Experiments

**Scenario**: Add new config without re-running old experiments

- Old configs already have some parameter combinations
- New config expands the grid
- Only new combinations run

**Result**: Saves time on reruns

## Deduplication Rules

Tasks are considered **duplicates** if they match on ALL:
- Dataset
- Model type
- Architecture parameters (width, depth)
- Poisoning parameters (type, ratio)
- Data distribution (alpha, ordering)
- Aggregation method
- Batch size
- **Seed**

Even tiny differences → separate tasks.

## Monitoring

Check deduplication results:

```bash
# Start manager
python experiment_manager.py

# Output:
# Generated 240 unique tasks
# Skipped 60 duplicate tasks across configs
```

Or check logs:
```bash
grep "duplicate" experiment_manager.log
```

## Edge Cases

### Same Experiment, Different Phases

If same config appears in two phases:
- ✅ **Deduplicated** (signatures match)
- First phase wins (arbitrary)

### Same Config, Different Seeds

- ❌ **Not deduplicated** (seed part of signature)
- Seed=42 and Seed=101 are different tasks

### Nearly Identical Configs

- ❌ **Not deduplicated** if any parameter differs
- Must match exactly on ALL parameters

## Implementation

**Key Code:**
```python
task_signatures = {}  # signature -> task_id

for config in configs:
    for experiment in experiments:
        for seed in seeds:
            signature = create_signature(experiment, seed)
            
            if signature in task_signatures:
                # Duplicate - skip
                duplicates_skipped += 1
            else:
                # New - add task
                tasks[task_id] = {...}
                task_signatures[signature] = task_id
```

## Testing

Create test configs with overlaps:
```yaml
# config_test1.yaml
combinations:
  - dataset: ["mnist"]
  - width_factor: [1, 4]

# config_test2.yaml  
combinations:
  - dataset: ["mnist"]
  - width_factor: [4, 16]  # width=4 overlaps
```

Expected: width=4 counted once, not twice.

## Performance

- **Memory**: O(N) for task_signatures dict
- **Time**: O(N) signature generation and lookup
- **Negligible overhead** for typical experiment sets (<10K tasks)
