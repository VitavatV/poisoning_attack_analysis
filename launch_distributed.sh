#!/bin/bash
# Distributed Experiment System Launcher (Linux/Mac)

echo "============================================================="
echo " Distributed Experiment System"
echo "============================================================="
echo ""
echo "Usage:"
echo "  ./launch_distributed.sh manager         - Start manager"
echo "  ./launch_distributed.sh worker          - Start one worker"
echo "  ./launch_distributed.sh workers [N]     - Start N workers (default: 3)"
echo ""
echo "============================================================="
echo ""

case "$1" in
    manager)
        echo "Starting Experiment Manager..."
        python experiment_manager.py
        ;;
    
    worker)
        echo "Starting GPU Worker..."
        python experiment_runner_gpu.py
        ;;
    
    workers)
        NUM_WORKERS=${2:-3}
        echo "Starting $NUM_WORKERS GPU workers..."
        
        for i in $(seq 1 $NUM_WORKERS); do
            echo "Starting worker $i..."
            python experiment_runner_gpu.py &
            sleep 2
        done
        
        echo ""
        echo "Started $NUM_WORKERS workers"
        echo "Each worker will auto-detect an available GPU"
        echo "Press Ctrl+C to stop all workers"
        wait
        ;;
    
    *)
        echo "Error: Invalid argument"
        echo ""
        echo "Usage:"
        echo "  $0 manager          - Start manager"
        echo "  $0 worker           - Start one worker"
        echo "  $0 workers [N]      - Start N workers"
        exit 1
        ;;
esac
