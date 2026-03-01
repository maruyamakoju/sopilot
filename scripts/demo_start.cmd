@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title SOPilot Demo Launcher

:: ════════════════════════════════════════════
::  SOPilot — Demo Launcher
::  Usage:
::    demo_start.cmd             ← standalone demo (no server)
::    demo_start.cmd --server    ← also start live API server
:: ════════════════════════════════════════════

set "ROOT=%~dp0.."
set "DEMO_HTML=%ROOT%\demo.html"
set "MODE=demo"
if /I "%1"=="--server" set "MODE=server"

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║          SOPilot  Demo Launcher              ║
echo  ║          PoC v0.2   On-Prem AI Scoring      ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: ── Check demo.html exists ──────────────────
if not exist "%DEMO_HTML%" (
  echo [ERROR] demo.html が見つかりません: %DEMO_HTML%
  echo         このスクリプトを scripts\ フォルダ内で実行してください。
  pause & exit /b 1
)

echo  [1/3] デモファイルを確認しました:
echo        %DEMO_HTML%
echo.

:: ── Open standalone demo in default browser ─
echo  [2/3] スタンドアローンデモをブラウザで開きます...
start "" "%DEMO_HTML%"
timeout /t 1 /nobreak >nul

:: ── Optional: start live API server ─────────
if /I "%MODE%"=="server" (
  echo  [3/3] ライブAPIサーバを起動します (port 8000)...
  echo.
  :: Set defaults
  if "%SOPILOT_PRIMARY_TASK_ID%"==""   set SOPILOT_PRIMARY_TASK_ID=filter_change
  if "%SOPILOT_EMBEDDER_BACKEND%"==""  set SOPILOT_EMBEDDER_BACKEND=color_motion
  if "%SOPILOT_SCORE_WORKERS%"==""     set SOPILOT_SCORE_WORKERS=1
  if "%SOPILOT_LOG_LEVEL%"==""         set SOPILOT_LOG_LEVEL=INFO

  echo  TASK_ID  = %SOPILOT_PRIMARY_TASK_ID%
  echo  EMBEDDER = %SOPILOT_EMBEDDER_BACKEND%
  echo  WORKERS  = %SOPILOT_SCORE_WORKERS%
  echo.
  cd /d "%ROOT%"
  uvicorn sopilot.main:app --host 0.0.0.0 --port 8000 --workers 1
) else (
  echo  [3/3] スタンドアローンモード (サーバ不要)
  echo.
  echo  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  echo   デモが開きました。ブラウザを確認してください。
  echo.
  echo   ライブサーバも起動する場合:
  echo     demo_start.cmd --server
  echo  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  echo.
  pause
)

endlocal
