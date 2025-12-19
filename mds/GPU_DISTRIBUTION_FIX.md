# Exclusive GPU Allocation

## Overview

Updated GPU allocation to enforce **one worker per GPU** using PID-based lock files. No GPU sharing is allowed.

## How It Works

### 1. Lock File System

Each GPU gets a lock file in `.gpu_locks/` directory:
```
.gpu_locks/
├── gpu_0.lock  # Contains PID of worker using GPU 0
├── gpu_1.lock  # Contains PID of worker using GPU 1
├── gpu_2.lock  # Contains PID of worker using GPU 2
└── gpu_3.lock  # Contains PID of worker using GPU 3
```

### 2. GPU Claiming Process

When a worker starts:

1. **Scan all GPUs** for available memory (>4GB)
2. **Check lock files** to see which GPUs are claimed
3. **Claim first unclaimed GPU** by creating lock file with worker's PID
4. **Exit if all GPUs claimed**

### 3. Stale Lock Cleanup

Before claiming, workers clean up stale locks:
- Read PID from lock file
- Check if process still exists using `os.kill(pid, 0)`
- Remove lock if process died

### 4. Automatic Cleanup

When worker exits (normally or via Ctrl+C):
- `atexit` handler automatically removes lock file
- GPU becomes available for new workers

## Example Scenario

**4 GPUs, Starting 6 Workers:**

```bash
# Worker 1 starts
# → Checks GPU 0: available, claims it
# → Creates .gpu_locks/gpu_0.lock with PID 1234
# → Starts working

# Worker 2 starts
# → Checks GPU 0: locked by PID 1234, skip
# → Checks GPU 1: available, claims it
# → Creates .gpu_locks/gpu_1.lock with PID 1235

# Worker 3 starts
# → GPU 0: locked, GPU 1: locked
# → Checks GPU 2: available, claims it

# Worker 4 starts
# → GPU 0,1,2: locked
# → Checks GPU 3: available, claims it

# Worker 5 starts
# → GPU 0,1,2,3: ALL LOCKED
# → "No available GPUs (all claimed)"
# → Exits immediately

# Worker 6 starts
# → Same as Worker 5, exits immediately
```

## Result

**With 4 GPUs:**
- Workers 1-4: Each gets exclusive GPU (no sharing)
- Workers 5-6: Exit immediately (no GPUs available)

## Cleanup

Lock files are automatically removed when:
- Worker finishes task and exits
- Worker is killed (Ctrl+C)
- Worker crashes (atexit still runs)

Manual cleanup if needed:
```bash
rm -rf .gpu_locks/
```

## Benefits

✅ **Exclusive Access**: One worker per GPU guaranteed  
✅ **No Sharing**: Workers 5-6 don't share with 1-4  
✅ **Automatic Cleanup**: Locks released on exit  
✅ **Stale Lock Handling**: Crashed worker locks auto-removed  
✅ **Simple**: No external coordination needed

## Monitoring

Check which GPUs are claimed:
```bash
ls -la .gpu_locks/
cat .gpu_locks/gpu_0.lock  # Shows PID
ps -p $(cat .gpu_locks/gpu_0.lock)  # Shows process info
```

## Starting Workers

```bash
# Start 4 workers (one per GPU)
python experiment_runner_gpu.py &
python experiment_runner_gpu.py &
python experiment_runner_gpu.py &
python experiment_runner_gpu.py &

# These will exit immediately (all GPUs claimed)
python experiment_runner_gpu.py &
python experiment_runner_gpu.py &
```

Or use a loop that stops when no GPUs available:
```bash
for i in {1..6}; do
    python experiment_runner_gpu.py &
    sleep 1
done
```

First 4 will claim GPUs, last 2 will exit.
