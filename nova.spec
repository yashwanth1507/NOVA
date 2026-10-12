# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification file for building Nova Desktop executable."""

import sys
from pathlib import Path

block_cipher = None

# Base path
ROOT_DIR = Path.cwd()

# Data files to include in bundle
datas = [
    (str(ROOT_DIR / 'Nova' / 'frontend' / 'desktop'), 'Nova/frontend/desktop'),
    (str(ROOT_DIR / 'config.json'), '.'),
]

# Hidden imports required by dynamic ASGI / FastAPI / pywebview / llama.cpp
hiddenimports = [
    # Embedding stack (optional hybrid retrieval). Without these the frozen
    # build silently degrades to keyword-only search.
    'sentence_transformers',
    'transformers',
    'tokenizers',
    'numpy',
    'huggingface_hub',
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'fastapi',
    'starlette',
    'pydantic',
    'webview',
    'webview.platforms.winforms',
    'webview.platforms.edgechromium',
    'webview.platforms.cocoa',
    'webview.platforms.gtk',
    'pypdf',
    'pdfplumber',
    'llama_cpp',
]

a = Analysis(
    ['main.py'],
    pathex=[str(ROOT_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'torch.testing'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Nova',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Nova',
)
