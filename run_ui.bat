@echo off
REM PathSnap UI (laptop 2). Usage:  run_ui.bat 192.168.43.120
REM (the IP printed by run_api.bat on laptop 1; omit it to use localhost)
cd /d "%~dp0"
set API=http://localhost:8000
if not "%1"=="" set API=http://%1:8000
echo Checking %API%/health ...
curl -s %API%/health
echo.
echo If you saw JSON above, the API is reachable. The sidebar is pre-filled with %API%
set PATHSNAP_API=%API%
python -m streamlit run ui/app.py
pause
