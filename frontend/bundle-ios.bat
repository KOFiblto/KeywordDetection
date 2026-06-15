@echo off
echo =========================================================
echo Building Auditory AI web assets...
echo =========================================================
call cmd.exe /c npm run build
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Web build failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =========================================================
echo Syncing web assets to iOS Xcode project...
echo =========================================================
call cmd.exe /c npx cap sync ios
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Capacitor sync failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =========================================================
echo Packaging Xcode project to auditory-ai-xcode.zip...
echo =========================================================
powershell -Command "if (Test-Path auditory-ai-xcode.zip) { Remove-Item auditory-ai-xcode.zip }; Compress-Archive -Path ios -DestinationPath auditory-ai-xcode.zip -Force"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Failed to zip Xcode project.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =========================================================
echo Success! Created auditory-ai-xcode.zip in the frontend folder.
echo You can now send this zip file to your friend to open in Xcode.
echo =========================================================
pause
