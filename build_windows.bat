@echo off
echo ====================================================
echo  Building Nova Desktop for Windows
echo ====================================================

where pyinstaller >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] PyInstaller not found. Installing PyInstaller...
    python -m pip install pyinstaller
)

echo [*] Running PyInstaller with nova.spec...
pyinstaller --clean --noconfirm nova.spec

if %ERRORLEVEL% EQU 0 (
    echo ====================================================
    echo  Build Successful!
    echo  Output directory: dist\Nova\Nova.exe
    echo ====================================================
) else (
    echo [!] Build failed with error code %ERRORLEVEL%
)
