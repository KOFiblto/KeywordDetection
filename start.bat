@echo off
echo ===================================================
echo   Auditory AI Launcher
echo ===================================================
echo.
echo Your computer's local IP address(es):
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr "IPv4"') do (
    set temp_ip=%%i
    call echo   - %%temp_ip: =%%
)
echo.

:: Save original directory
set "ORIG_DIR=%CD%"

:: Run the integrated services runner script
node start-services.js

:: Restore original directory
cd /d "%ORIG_DIR%"
