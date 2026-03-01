@echo off
setlocal

if "%SOPILOT_DATA_DIR%"=="" set SOPILOT_DATA_DIR=data
if "%SOPILOT_PRIMARY_TASK_ID%"=="" set SOPILOT_PRIMARY_TASK_ID=filter_change

set DB_PATH=%SOPILOT_DATA_DIR%\sopilot.db
set LABELS_PATH=%1
if "%LABELS_PATH%"=="" set LABELS_PATH=docs\evaluation_labels_template.json

echo [SOPilot] PoC status
python scripts\poc_status.py --db-path %DB_PATH% --task-id %SOPILOT_PRIMARY_TASK_ID%
if errorlevel 1 goto fail

echo [SOPilot] PoC metrics and gates
python scripts\evaluate_poc.py ^
  --db-path %DB_PATH% ^
  --task-id %SOPILOT_PRIMARY_TASK_ID% ^
  --labels %LABELS_PATH% ^
  --max-critical-miss-rate 0.10 ^
  --max-critical-fpr 0.30 ^
  --max-rescore-jitter 5.0 ^
  --max-dtw-p90 0.60 ^
  --fail-on-gate
if errorlevel 1 goto fail

echo [SOPilot] Daily eval passed
exit /b 0

:fail
echo [SOPilot] Daily eval failed
exit /b 2

endlocal

