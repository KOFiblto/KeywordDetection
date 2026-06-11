@echo off
echo ===================================================
echo   Auditory AI Release Generator
echo ===================================================
echo.

:: 1. Build PyInstaller Python Backend
echo [1/4] Compiling Python FastAPI backend with PyInstaller...
call .\.venv\Scripts\pyinstaller --noconfirm --onedir --console --name backend_api --exclude-module torch --exclude-module torchaudio --exclude-module torchvision --exclude-module scipy --exclude-module PyQt6 --exclude-module matplotlib --exclude-module seaborn --exclude-module PIL --exclude-module pandas --exclude-module tkinter --exclude-module docutils --exclude-module IPython backend/main.py
if %ERRORLEVEL% neq 0 (
    echo Error compiling backend with PyInstaller.
    pause
    exit /b %ERRORLEVEL%
)

:: 2. Shrink Backend Size (Remove CUDA/GPU binaries)
echo.
echo [2/4] Removing heavy GPU/CUDA dependencies to optimize installer size...
powershell -Command "Get-ChildItem -Path 'dist\backend_api\_internal\torch\lib' -Include *cuda*,*cublas*,*cudnn*,*cufft*,*curand*,*cusolver*,*cusparse*,*nv*,*cupti* -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force"

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
