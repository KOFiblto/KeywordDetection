@echo off
echo ===================================================
echo   Auditory AI Release Generator
echo ===================================================
echo.

:: 1. Build PyInstaller Python Backend (Skipped - client-side WASM inference active)
echo [1/4] Client-side WASM inference active. Skipping Python compilation...

:: 2. Clean dist (Skipped)
echo [2/4] Skipping GPU cleanup...

:: 3. Clean old frontend builds
echo.
echo [3/4] Cleaning previous builds...
if exist "frontend\dist-app" rd /s /q "frontend\dist-app"

:: 4. Build Vite Client and Package Electron Installer
echo.
echo [4/4] Building Vite client and compiling final installer...
cd frontend
call npm run dist-app
if %ERRORLEVEL% neq 0 (
    echo Error compiling Electron distributable.
    cd ..
    pause
    exit /b %ERRORLEVEL%
)
cd ..

echo.
echo ===================================================
echo   Release compiled successfully!
echo   Installer: frontend\dist-app\Auditory AI Setup 1.0.0.exe
echo ===================================================
pause
