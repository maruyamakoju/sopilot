@echo off
setlocal

if "%SOPILOT_PRIMARY_TASK_ID%"=="" set SOPILOT_PRIMARY_TASK_ID=filter_change
if "%SOPILOT_PRIMARY_TASK_NAME%"=="" set SOPILOT_PRIMARY_TASK_NAME=Filter Change
if "%SOPILOT_ENFORCE_PRIMARY_TASK%"=="" set SOPILOT_ENFORCE_PRIMARY_TASK=true

:: ── DATA_DIR: TESDA検証データ入りDBを使用（未設定時のデフォルト "data" より優先）
if "%SOPILOT_DATA_DIR%"=="" set SOPILOT_DATA_DIR=data_tesda_cm

:: ── EMBEDDER: デフォルトは color-motion (CPU, GPU不使用, PC固まらない)
:: ── V-JEPA2を使う場合は明示的に set SOPILOT_EMBEDDER_BACKEND=vjepa2 してから実行
:: ── V-JEPA2はVRAM ~6GB消費 + モデルロード時にPC高負荷になる場合がある
if "%SOPILOT_EMBEDDER_BACKEND%"=="" set SOPILOT_EMBEDDER_BACKEND=color-motion

if "%SOPILOT_SCORE_WORKERS%"=="" set SOPILOT_SCORE_WORKERS=1
if "%SOPILOT_LOG_LEVEL%"=="" set SOPILOT_LOG_LEVEL=INFO
if "%SOPILOT_VJEPA2_VARIANT%"=="" set SOPILOT_VJEPA2_VARIANT=vjepa2_vit_large
if "%SOPILOT_VJEPA2_PRETRAINED%"=="" set SOPILOT_VJEPA2_PRETRAINED=true

echo [SOPilot] PoC bootstrap
echo TASK_ID=%SOPILOT_PRIMARY_TASK_ID%
echo TASK_NAME=%SOPILOT_PRIMARY_TASK_NAME%
echo EMBEDDER=%SOPILOT_EMBEDDER_BACKEND%
echo WORKERS=%SOPILOT_SCORE_WORKERS%
echo DATA_DIR=%SOPILOT_DATA_DIR%

:: V-JEPA2のときだけpreloadする（color-motionはpreload不要）
if /I "%SOPILOT_EMBEDDER_BACKEND%"=="color-motion" goto run_server
if /I "%1"=="--skip-preload" goto run_server

echo [SOPilot] Preloading V-JEPA2 cache... (GPU使用 - 完了まで待ってください)
python scripts\preload_vjepa2.py --variant %SOPILOT_VJEPA2_VARIANT% --pretrained %SOPILOT_VJEPA2_PRETRAINED% --crop-size 256
if errorlevel 1 (
  echo [SOPilot] WARNING: preload failed. continuing...
)

:run_server
echo [SOPilot] Starting API...
uvicorn sopilot.main:create_app --factory --host 0.0.0.0 --port 8000 --workers 1

endlocal
