@echo off
REM Distributed Experiment System Launcher (Windows)
REM
REM Usage:
REM   launch_manager.bat          - Start manager
REM   launch_worker.bat           - Start one GPU worker
REM   launch_workers.bat [N]      - Start N GPU workers (default: 3)

echo =============================================================
echo  Distributed Experiment System
echo =============================================================
echo.
echo This system consists of:
echo   1. Experiment Manager - Coordinates all tasks
echo   2. GPU Workers - Execute tasks on available GPUs
echo.
echo =============================================================
echo.

choice /C 12 /M "Press 1 to start MANAGER, 2 to start WORKER(S)"

if errorlevel 2 goto worker
if errorlevel 1 goto manager

:manager
echo.
echo Starting Experiment Manager...
echo.
python experiment_manager.py
goto end

:worker
set /p num_workers="How many workers to start? (default: 1): "
if "%num_workers%"=="" set num_workers=1

echo.
echo Starting %num_workers% GPU worker(s)...
echo.

for /l %%i in (1,1,%num_workers%) do (
    echo Starting worker %%i...
    start "GPU Worker %%i" cmd /k python experiment_runner_gpu.py
    timeout /t 2 /nobreak >nul
)

echo.
echo Started %num_workers% worker(s)
echo Each worker will auto-detect an available GPU
goto end

:end
echo.
echo Press any key to exit...
pause >nul
