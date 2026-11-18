@echo off
REM PathSnap API (laptop 1). Usage:
REM   run_api.bat            serve data\evaluator  (evaluation day)
REM   run_api.bat rehearsal  serve data\raw\student_test (practice)
cd /d "%~dp0"
if /i "%1"=="rehearsal" (
    set PATHSNAP_EVAL_DIR=%~dp0data\raw\student_test
) else (
    set PATHSNAP_EVAL_DIR=%~dp0data\evaluator
)
echo Serving trips from: %PATHSNAP_EVAL_DIR%
echo.
echo Give laptop 2 the IPv4 address listed under your Wi-Fi adapter
echo (or under "Local Area Connection* N" if THIS laptop is the hotspot):
ipconfig | findstr /i /c:"adapter" /c:"IPv4"
echo   -^> http://THAT-IP:8000
echo.
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
pause
