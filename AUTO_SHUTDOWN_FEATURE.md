# Auto-Shutdown Feature

## Summary

Added automatic shutdown to `experiment_manager.py` - the manager now exits gracefully when all tasks are complete.

## Changes Made

### [experiment_manager.py](file:///c:/github/poisoning_attack_analysis/experiment_manager.py)

**New Method: `check_all_complete()` (lines 254-268)**

```python
def check_all_complete(self):
    """Check if all tasks are complete and initiate shutdown if so"""
    total = len(self.tasks)
    complete = sum(1 for t in self.tasks.values() if t['status'] == 'complete')
    
    if total > 0 and complete == total:
        logging.info("="*60)
        logging.info("ALL TASKS COMPLETE!")
        logging.info(f"Total experiments: {total}")
        logging.info("Shutting down experiment manager...")
        logging.info("="*60)
        
        # Set running flag to False to stop server
        self.running = False
```

**Modified: `get_next_task()` (line 238-240)**

Added check when no pending tasks found:
```python
# No pending tasks - check if all tasks are complete
self.check_all_complete()
```

**Modified: `mark_complete()` (line 250-252)**

Added check after marking task complete:
```python
# Check if all tasks are now complete
self.check_all_complete()
```

## Behavior

**Before:**
- Manager runs indefinitely
- Must manually press `Ctrl+C` to stop
- Keeps listening on port even when all work is done

**After:**
- Manager automatically detects completion
- Logs completion status
- Gracefully shuts down
- Releases port and cleans up resources

## Example Output

```
============================================================
Starting Experiment Manager
============================================================
Generated 240 total tasks
Marked 45 tasks as already completed
Initial status: 45/240 (18.8%)
Experiment Manager listening on localhost:5000

Assigned task exp1_vary_width_0_42 to gpu0
Task exp1_vary_width_0_42 completed by gpu0
...
[many tasks later]
...
Task exp5_defense_comparison_239_2024 completed by gpu2

============================================================
ALL TASKS COMPLETE!
Total experiments: 240
Shutting down experiment manager...
============================================================

[Process exits]
```

## Testing

Created `create_test_config.py` to set up a minimal test with only 2 tasks for quick verification.

**Test Steps:**
1. Run `python create_test_config.py` 
2. Follow instructions to temporarily use test config
3. Start manager and worker
4. Observe auto-shutdown after 2 tasks complete

## Technical Details

- **Thread-safe**: Checks happen while holding `self.lock`
- **Multiple triggers**: Checked both when assigning tasks and marking complete
- **Clean exit**: Sets `self.running = False` which stops the server loop
- **Resource cleanup**: Socket server closes properly via existing shutdown logic
