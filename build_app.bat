@echo off
echo Astraea Build Tool
echo -----------------
echo.
echo Choose a build method:
echo 1. PyInstaller (Standard build)
echo 2. Nuitka (Optimized, faster execution)
echo 3. Simplified build (Faster, skips whisper model dependencies)
echo 4. Fixed build with DLL support (Recommended for ONNX error)
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Building with PyInstaller...
    echo.
    python build.py
) else if "%choice%"=="2" (
    echo.
    echo Building with Nuitka...
    echo.
    python nuitka_build.py
) else if "%choice%"=="3" (
    echo.
    echo Building with simplified dependencies...
    echo.
    python simplified_build.py
) else if "%choice%"=="4" (
    echo.
    echo Building with fixed DLL support...
    echo.
    python fixed_build.py
) else (
    echo.
    echo Invalid choice. Please run again and select 1-4.
)

echo.
echo.
echo Press any key to exit...
pause > nul 