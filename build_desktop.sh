#!/usr/bin/env bash
set -e

echo "===================================================="
echo " Building Nova Desktop"
echo "===================================================="

if ! command -v pyinstaller &> /dev/null; then
    echo "[*] Installing PyInstaller..."
    python3 -m pip install pyinstaller
fi

echo "[*] Running PyInstaller with nova.spec..."
pyinstaller --clean --noconfirm nova.spec

echo "===================================================="
echo " Build Successful!"
echo " Output located in: dist/Nova/"
echo "===================================================="
