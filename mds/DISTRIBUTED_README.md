# Distributed Experiment System

A multi-process experiment execution system with centralized task management and dynamic GPU allocation.

## Architecture

- **Experiment Manager** (`experiment_manager.py`): Centralized coordinator that manages task queue and monitors completion
- **GPU Workers** (`experiment_runner_gpu.py`): Worker processes that request tasks and execute them on available GPUs

## Quick Start

### Windows

**Terminal 1 - Start Manager:**
```bash
python experiment_manager.py
```

**Terminal 2+ - Start Workers:**
```bash
python experiment_runner_gpu.py
python experiment_runner_gpu.py  # Start as many as you have GPUs
python experiment_runner_gpu.py
```

Or use the launcher:
```bash
launch_distributed.bat
```

### Linux/Mac

**Terminal 1 - Start Manager:**
```bash
python experiment_manager.py
```

**Terminal 2+ - Start Workers:**
```bash
python experiment_runner_gpu.py &
python experiment_runner_gpu.py &
python experiment_runner_gpu.py &
```

Or use the launcher:
```bash
chmod +x launch_distributed.sh
./launch_distributed.sh manager      # In terminal 1
./launch_distributed.sh workers 3    # In terminal 2 (starts 3 workers)
```

## How It Works

1. **Manager starts** and scans all `configs/config_exp*.yaml` files to generate tasks
2. **Manager checks** `results_*/final_results.csv` files to identify already-completed experiments
3. **Workers start** and auto-detect available GPUs (checks for >4GB free VRAM)
4. **Workers request tasks** from manager via socket connection
5. **Manager assigns** next pending task to requesting worker
6. **Worker executes** the experiment and saves results to CSV
7. **Worker notifies** manager of completion
8. **Manager refreshes** CSV files every 5 seconds to stay updated
9. **Workers repeat** until no tasks remain, then shut down

## Features

- ✅ **Dynamic GPU Allocation**: Workers automatically find and use available GPUs
- ✅ **Parallel Execution**: Multiple workers run simultaneously on different GPUs
- ✅ **Fault Tolerance**: Manager reassigns tasks if worker crashes (2-hour timeout)
- ✅ **Progress Tracking**: Manager logs overall progress
- ✅ **Resume Support**: Automatically skips already-completed experiments
- ✅ **No Duplicate Work**: Tasks are locked when assigned to prevent duplicates

## Monitoring Progress

Check the manager terminal for status updates:
```
Generated 240 total tasks
Marked 45 tasks as already completed
Assigned task exp1_vary_width_0_42 to gpu0
Task exp1_vary_width_0_42 completed by gpu0
```

Or check the log file:
```bash
tail -f experiment_manager.log
```

## Configuration

All experiment configurations are in `configs/config_exp*.yaml` - no changes needed!

The manager automatically:
- Loads all config files
- Generates task combinations
- Resolves `{dataset}` placeholders in output directories
- Tracks completion by matching CSV parameters

## Stopping

- **Manager**: Press `Ctrl+C` in the manager terminal
- **Workers**: Press `Ctrl+C` in worker terminals (or just close them)
- **All processes**: Close all terminal windows

Workers will complete their current task before shutting down.

## Troubleshooting

**"No available GPU found"**
- Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU memory: All GPUs may be in use
- Reduce VRAM requirement in `get_available_gpu()` (currently 4GB)

**"Error requesting task"**
- Make sure manager is running first
- Check manager is listening on port 5000
- Try restarting the manager

**Tasks not completing**
- Check worker logs: `cat results_*/worker_*.log`
- Check for errors in worker terminal
- Manager will reassign after 2-hour timeout

**Duplicate experiments**
- Manager checks CSV parameters before assignment
- If CSV is corrupted, delete it and restart manager

## Advanced Usage

**Custom manager host/port:**
```python
# In experiment_manager.py, line 26:
manager = ExperimentManager(host='0.0.0.0', port=5000)

# When starting worker:
python experiment_runner_gpu.py 192.168.1.100 5000
```

**Check status programmatically:**
```python
import socket, json

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 5000))
sock.sendall(json.dumps({'type': 'status'}).encode())
response = json.loads(sock.recv(1024).decode())
print(response['status'])
sock.close()
```
