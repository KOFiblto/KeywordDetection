@echo off
echo Starting Keyword Detection Application...

:: Start the Python backend in a new command window
start "KeywordDetectionBackend" cmd /c "cd /d "%~dp0backend" && "..\.venv\Scripts\python.exe" main.py"

:: Wait a moment to ensure backend starts
ping 127.0.0.1 -n 3 > nul

:: Start the Electron frontend in a new command window
start "KeywordDetectionFrontend" cmd /c "cd /d "%~dp0frontend" && npm start"

echo Both services have been started in separate windows!
echo You can use stop.bat to close them.
