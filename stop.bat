@echo off
echo Stopping Keyword Detection Application...

:: Kill processes that were started with our specific window titles
taskkill /FI "WindowTitle eq KeywordDetectionBackend*" /T /F > nul 2>&1
taskkill /FI "WindowTitle eq KeywordDetectionFrontend*" /T /F > nul 2>&1

:: Failsafe: kill any leftover Python backend running on port 18000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :18000') do (
    taskkill /F /PID %%a > nul 2>&1
)

echo App stopped successfully!
timeout /t 2 /nobreak > nul
