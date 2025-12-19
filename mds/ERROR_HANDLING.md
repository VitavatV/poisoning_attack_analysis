# Error Handling & Task Reassignment

## Problem

When a worker encounters an error (e.g., dataset download failure), the task remains in "assigned" state indefinitely. The manager doesn't know about the failure and can't reassign the task to another worker.

## Solution

Added comprehensive error handling with automatic task reassignment.

---

## Changes Made

### 1. Worker (experiment_runner_gpu.py)

**Added `notify_task_failed()` function:**
```python
def notify_task_failed(task_id, worker_id, error_msg):
    # Sends 'task_failed' message to manager
```

**Updated `run_task()` error handling:**
- Catches all exceptions
- Notifies manager when task fails
- Logs full traceback
- Returns False to continue with next task

### 2. Manager (experiment_manager.py)

**Added retry tracking to tasks:**
```python
{
    'retry_count': 0,      # Number of retry attempts
    'last_error': None     # Last error message
}
```

**Added `mark_failed()` method:**
- Increments retry counter
- Stores error message
- Reassigns task if under retry limit (default: 3)
- Marks as permanently failed after max retries

**Updated socket handler:**
- Added 'task_failed' message type handler
- Calls `mark_failed()` when worker reports failure

**Updated status reporting:**
- Added 'failed' count to status dict
- Shows permanently failed tasks

---

## Workflow

### Normal Flow
```
1. Manager assigns task → Worker
2. Worker executes successfully
3. Worker notifies "task_complete"
4. Manager marks complete
```

### Error Flow (with retry)
```
1. Manager assigns task → Worker A
2. Worker A encounters error
3. Worker A notifies "task_failed"
4. Manager increments retry_count (1/3)
5. Manager marks task as 'pending'
6. Manager assigns task → Worker B
7. Worker B executes successfully
8. Worker B notifies "task_complete"
9. Manager marks complete
```

### Error Flow (max retries exceeded)
```
1. Task fails on Worker A → retry_count = 1
2. Task fails on Worker B → retry_count = 2
3. Task fails on Worker C → retry_count = 3
4. Manager marks task as 'failed' (permanently)
5. Task not reassigned anymore
```

---

## Configuration

**Max Retries:** Set in `mark_failed()` method (default: 3)

```python
def mark_failed(self, task_id, worker_id, error_msg, max_retries=3):
```

To change:
```python
# In experiment_manager.py, line ~256
max_retries = 5  # Allow 5 retry attempts
```

---

## Benefits

✅ **Automatic Recovery:** Failed tasks reassigned to other workers  
✅ **Fault Tolerance:** System continues despite worker errors  
✅ **Retry Limit:** Prevents infinite loops on permanently broken tasks  
✅ **Error Tracking:** Last error message logged for debugging  
✅ **Worker Independence:** Worker failures don't block other workers  

---

## Example Scenarios

### Scenario 1: Dataset Download Failure

```
Worker A: Cannot download CIFAR-10
  → Notifies manager
  → Manager reassigns to Worker B
Worker B: Has datasets pre-downloaded
  → Executes successfully
  → Task completes
```

### Scenario 2: Out of Memory

```
Worker A (4GB VRAM): OOM on large model
  → Notifies manager
  → Manager reassigns to Worker B
Worker B (24GB VRAM): Success
  → Task completes
```

### Scenario 3: Corrupted Config

```
Worker A, B, C all fail (same config error)
  → After 3 retries, task marked as 'failed'
  → Manager continues with other tasks
  → Admin can fix config and restart
```

---

## Monitoring

Check manager logs for failures:

```bash
tail -f experiment_manager.log | grep -i "failed\|error"
```

Example output:
```
2025-12-19 14:50:23 - WARNING - Task exp1_0_42 failed on gpu0 (attempt 1/3). Error: File not found... Reassigning.
2025-12-19 14:51:45 - INFO - Task exp1_0_42 completed by gpu1
```

Or check status programmatically:
```python
import socket, json

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 5000))
sock.sendall(json.dumps({'type': 'status'}).encode())
response = json.loads(sock.recv(1024).decode())
print(f"Failed: {response['status']['failed']}")
sock.close()
```

---

## Testing

This system now handles:
- ✅ Network failures during dataset download
- ✅ Out of memory errors
- ✅ Invalid configurations
- ✅ Worker crashes mid-task
- ✅ Timeout scenarios (2-hour limit still applies)

Tasks will be retried up to 3 times before being marked as permanently failed.
