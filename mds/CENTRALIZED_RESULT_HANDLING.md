# Centralized Result Handling

## Overview

Workers now submit results to the manager instead of saving directly. The manager saves results to **all output directories** for experiments that appear in multiple configs.

## Problem: Duplicate Results Not Shared

**Before:**
```
Config 1 needs: (mnist, cnn, w=4, poison=0.0)
Config 2 needs: (mnist, cnn, w=4, poison=0.0)  # Same experiment!

Worker runs once (deduplicated)
Saves to: results_exp1/final_results.csv only

Config 2 never gets the result! ❌
```

## Solution: Manager-Centered Result Distribution

**After:**
```
Config 1 needs: (mnist, cnn, w=4, poison=0.0)
Config 2 needs: (mnist, cnn, w=4, poison=0.0)

Manager tracks: output_dirs = ['results_exp1', 'results_exp2']

Worker runs once
Submits result to manager
Manager saves to BOTH directories ✅
```

## Architecture Changes

### 1. Task Structure (Manager)

Added `output_dirs` list to track all directories:

```python
task = {
    'task_id': 'exp1_vary_width_0_42',
    'output_dir': 'results_exp1_mnist',     # Primary
    'output_dirs': [                         # All directories (NEW)
        'results_exp1_mnist',
        'results_exp2_mnist'  # From duplicate config
    ],
    ...
}
```

### 2. Deduplication Update

When duplicate detected, add output directory:

```python
if signature in task_signatures:
    # Duplicate found!
    existing_task = tasks[existing_task_id]
    
    # Add new output directory to existing task
    if new_output_dir not in existing_task['output_dirs']:
        existing_task['output_dirs'].append(new_output_dir)
    
    skip task creation
```

### 3. Worker: Submit Result

Instead of saving CSV directly:

```python
# Old way
df.to_csv(output_dir + '/final_results.csv')

# New way
submit_result_to_manager(task_id, worker_id, result_data)
```

### 4. Manager: Save to All Directories

Manager receives result and saves everywhere:

```python
def save_result(task_id, result_data):
    task = tasks[task_id]
    
    for output_dir in task['output_dirs']:
        save_to_csv(output_dir + '/final_results.csv', result_data)
```

## Message Flow

```
┌─────────┐                    ┌─────────┐
│ Worker  │                    │ Manager │
└─────────┘                    └─────────┘
     │                              │
     │ 1. Get task                  │
     │ ◄──────────────────────────── │
     │    output_dirs: [A, B]       │
     │                              │
     │ 2. Run experiment            │
     │    ...                        │
     │                              │
     │ 3. Submit result             │
     │ ─────────────────────────► │
     │    {task_id, result_data}    │
     │                              │
     │                            4. Save to A/final_results.csv
     │                            5. Save to B/final_results.csv
     │                              │
     │ 6. Ack                       │
     │ ◄──────────────────────────── │
     │                              │
```

## Socket Protocol

### Worker → Manager: Submit Result

```python
{
    'type': 'submit_result',
    'task_id': 'exp1_vary_width_0_42',
    'worker_id': 'gpu2',
    'result_data': {
        'phase': 'exp1_vary_width',
        'dataset': 'mnist',
        'model_type': 'cnn',
        'mean_test_acc': 0.9523,
        'mean_test_loss': 0.1234,
        ...
    }
}
```

### Manager → Worker: Acknowledgment

```python
{
    'type': 'ack'
}
```

## Benefits

✅ **Duplicate Results Shared**: One experiment → all configs get results  
✅ **Centralized Control**: Manager controls where results go  
✅ **Atomic Saves**: Result saved to all directories before ack  
✅ **Fallback**: Worker saves locally if manager unreachable  
✅ **Deduplication Complete**: No redundant work, no missing results

## Example Scenario

**Setup:**
- `config_exp1.yaml`: Tests (mnist, cnn, w=4)
- `config_exp2.yaml`: Tests (mnist, cnn, w=4)  # Same!
- `config_exp3.yaml`: Tests (cifar10, cnn, w=4)

**Task Generation:**
```
Task A: (mnist, cnn, w=4)
  output_dirs: ['results_exp1', 'results_exp2']
  
Task B: (cifar10, cnn, w=4)
  output_dirs: ['results_exp3']
```

**Execution:**
```
Worker runs Task A once
Submits result to manager

Manager saves to:
  ✓ results_exp1/final_results.csv
  ✓ results_exp2/final_results.csv

Both configs have complete results!
```

## Fallback Strategy

If manager is down or unreachable:

```python
success = submit_result_to_manager(task_id, result)

if not success:
    # Fallback: save locally to primary output_dir
    save_to_csv(primary_output_dir)
    logging.warning("Saved locally (manager unreachable)")
```

## Implementation Details

### Manager Side

**save_result method:**
```python
def save_result(task_id, result_data):
    task = tasks[task_id]
    output_dirs = task.get('output_dirs', [task['output_dir']])
    
    for output_dir in output_dirs:
        csv_path = output_dir + '/final_results.csv'
        
        # Append to existing or create new
        if exists(csv_path):
            df = read_csv(csv_path)
            df = concat([df, DataFrame([result_data])])
        else:
            df = DataFrame([result_data])
        
        df.to_csv(csv_path)
```

### Worker Side

**submit_result_to_manager function:**
```python
def submit_result_to_manager(task_id, worker_id, result_data):
    sock = socket.connect(manager)
    
    request = {
        'type': 'submit_result',
        'task_id': task_id,
        'worker_id': worker_id,
        'result_data': result_data
    }
    
    sock.send(json.dumps(request))
    response = sock.recv()
    
    return response['type'] == 'ack'
```

## Testing

### Test 1: Single Config
```
Config 1: (mnist, cnn, w=4)
Expected: Result in results_exp1/ only
```

### Test 2: Duplicate Across Configs
```
Config 1: (mnist, cnn, w=4)
Config 2: (mnist, cnn, w=4)
Expected: Same result in results_exp1/ AND results_exp2/
```

### Test 3: Partial Overlap
```
Config 1: width=[1, 4, 16]
Config 2: width=[4, 16, 64]
Expected:
  w=1: results_exp1/ only
  w=4: results_exp1/ AND results_exp2/
  w=16: results_exp1/ AND results_exp2/
  w=64: results_exp2/ only
```

## Monitoring

Check result distribution:
```bash
# Count results per directory
wc -l results_exp1/final_results.csv
wc -l results_exp2/final_results.csv

# Check for duplicates
cat results_exp1/final_results.csv | grep "mnist.*cnn.*4"
cat results_exp2/final_results.csv | grep "mnist.*cnn.*4"
# Should be identical rows
```

Manager logs:
```
Result saved to 2 directories for task exp1_vary_width_0_42
Saved result for exp1_vary_width_0_42 to results_exp1/final_results.csv
Saved result for exp1_vary_width_0_42 to results_exp2/final_results.csv
```
