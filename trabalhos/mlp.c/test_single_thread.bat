@echo off
REM Quick script to test with a specific number of threads

if "%1"=="" (
    echo Usage: test_single_thread.bat [number_of_threads]
    echo Example: test_single_thread.bat 8
    exit /b 1
)

echo ===================================================================
echo Testing OpenMP CPU with %1 thread(s)
echo ===================================================================

set OMP_NUM_THREADS=%1
mnist_mlp_openmp_cpu.exe

echo.
echo Test complete!
