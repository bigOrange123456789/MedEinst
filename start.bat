@echo off
setlocal
chcp 65001 >nul
title MedEinst - Smoke Run and Paper Metrics

echo [1/4] Activating conda environment MedEinst...
call "D:\mySoftware2\anaconda3\condabin\conda.bat" activate MedEinst
if errorlevel 1 goto :error

echo [2/4] Changing to scripts directory...
cd /d "G:\Codex\MedEinst\scripts"
if errorlevel 1 goto :error

echo [3/4] Running ECR agent...
rem Local inference automatically assigns workers across visible CUDA GPUs.
python run_ecr_agent.py --benchmark-set "G:\Codex\MedEinst_dataset\test.jsonl" --output-dir "G:\Codex\MedEinst\output_smoke" --parallel --workers 4 --console-language zh --rerun --limit 2 --detailed-timing
::  --parallel --workers 4 --console-language zh --rerun --limit 2 --detailed-timing
if errorlevel 1 goto :error

echo [4/4] Computing paper metrics...
python compute_paper_metrics.py
if errorlevel 1 goto :error

echo.
echo All steps completed successfully.
pause
endlocal
exit /b 0

:error
set "RUN_EXIT_CODE=%errorlevel%"
echo.
echo ERROR: Execution stopped with exit code %RUN_EXIT_CODE%.
echo Please check the error message above.
pause
exit /b %RUN_EXIT_CODE%
