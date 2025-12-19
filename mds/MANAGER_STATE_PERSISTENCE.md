# Manager State Persistence & Recovery

## Overview

Added state persistence to experiment manager so it can recover after crashes or restarts, knowing exactly what each worker was doing.

## Features

### 1. State File (`manager_state.json`)

Stores complete manager state:
```json
{
  "tasks": {
    "exp1_vary_width_0_42": {
      "task_id": "exp1_vary_width_0_42",
      "phase": "exp1_vary_width",
      "config": {...},
      "status": "assigned",
      "assigned_to": "gpu2",
      "assigned_at": "2025-12-19T23:10:15.123456",
      "retry_count": 0,
      "last_error": null
    },
    ...
  },
  "assigned_tasks": {
    "exp1_vary_width_0_42": {...}
  }
}
```

### 2. Automatic State Saving

State is saved:
- **On startup** (initial state)
- **Every 50 seconds** (during refresh_completed cycle)
- **On clean shutdown** (Ctrl+C)

### 3. State Recovery on Restart

When manager restarts:

1. **Loads state file** if it exists
2. **Restores all tasks** with their statuses
3. **Checks for stale assignments**:
   - Tasks assigned >2 hours ago
   - Assumes worker died
   - Reassigns to pending
4. **Updates from CSV files** to catch any completions during downtime

### 4. Stale Assignment Detection

Protects against:
- Workers that died while manager was down
- Tasks stuck in "assigned" state
- Zombie assignments

## Usage Scenarios

### Scenario 1: Manager Crashes

```bash
# Manager running with 4 workers
Manager: Task A assigned to gpu0, Task B to gpu1, Task C to gpu2

# Manager crashes
<CRASH>

# Restart manager
python experiment_manager.py

# Output:
# Loaded state: 240 tasks, 3 assigned
# Reassigning stale task exp1_vary_width_0_42 (assigned 0.5 hours ago - still valid)
# Initial status: 45/240 (18.8%)
```

**Manager resumes** exactly where it left off!

### Scenario 2: Manager Down for Hours

```bash
# Manager goes down
# Workers complete tasks and write to CSV
# Worker processes die naturally

# Manager restarts 3 hours later
python experiment_manager.py

# Output:
# Loaded state: 240 tasks, 3 assigned
# Reassigning stale task exp1_vary_width_0_42 (assigned 3.2 hours ago)
# Reassigning stale task exp1_vary_width_1_42 (assigned 3.2 hours ago)  
# Reassigning stale task exp1_vary_width_2_42 (assigned 3.2 hours ago)
# Updating state with CSV completions...
# Marked 3 tasks as already completed
# Initial status: 48/240 (20.0%)
```

**Manager:**
- Loaded old state
- Detected stale assignments (>2 hours)
- Reassigned those tasks
- Discovered 3 were actually complete (via CSV)
- Ready to continue

### Scenario 3: Clean Shutdown

```bash
# Manager running
Ctrl+C

# Output:
# Shutting down...
# Saved state: 240 tasks
```

State saved for perfect restart!

## Benefits

✅ **Crash Recovery**: Manager can restart without losing progress  
✅ **Stale Detection**: Automatically handles dead workers  
✅ **No Duplicate Work**: Knows what's assigned vs complete  
✅ **Seamless Resume**: Workers reconnect and continue  
✅ **Audit Trail**: State file shows exact status at any time

## State File Location

```
poisoning_attack_analysis/
├── manager_state.json          # Manager state
├── .gpu_locks/                 # Worker GPU locks
│   ├── gpu_0.lock
│   ├── gpu_1.lock
│   └── ...
└── results_*/                  # Experiment results
```

## Manual State Inspection

Check current state:
```bash
cat manager_state.json | jq '.tasks | map(select(.status=="assigned"))'
```

See assigned tasks:
```bash
cat manager_state.json | jq '.assigned_tasks'
```

## State Reset

To start completely fresh:
```bash
rm manager_state.json
python experiment_manager.py
```

## Implementation Details

### State Structure

- **tasks**: All task definitions with current status
- **assigned_tasks**: Quick lookup of currently assigned tasks
- **Datetime conversion**: ISO format for JSON serialization

### Thread Safety

All state operations protected by `self.lock` for thread-safe access.

### Save Frequency

- Too frequent: Disk I/O overhead
- Too rare: Risk losing progress
- **Chosen: 50 seconds** - Good balance

### Stale Threshold

- **2 hours**: Same as normal timeout
- Assumes worker died or is stuck
- Safe to reassign

## Recovery Workflow

```
┌─────────────────────────────────────┐
│ Manager Restart                      │
├─────────────────────────────────────┤
│ 1. Load state file                   │
│ 2. Restore tasks & assignments       │
│ 3. Check stale assignments (>2hr)    │
│ 4. Reassign stale tasks              │
│ 5. Scan CSV files for completions    │
│ 6. Update completed tasks            │
│ 7. Save updated state                │
│ 8. Resume normal operation           │
└─────────────────────────────────────┘
```

## Future Enhancements

Possible improvements:
- Compress state file (gzip)
- Keep state history (rotating backups)
- Add task start/end timestamps
- Track worker performance stats
- Add state checksum validation
