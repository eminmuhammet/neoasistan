# PyInstaller build spec for NEO.
#
# Build with:  .venv\Scripts\pyinstaller neo.spec --noconfirm
# Output:      dist\NEO\NEO.exe
#
# --onedir (not --onefile) on purpose: this bundle carries PySide6 and
# ctranslate2 native DLLs, which onefile would have to unpack to a temp dir
# on every launch -- slower to start and a common source of DLL-loading
# failures. The speech model itself is NOT bundled; faster-whisper downloads
# it to the user's Hugging Face cache on first use.

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = []
binaries = []
hiddenimports = [
    "win32com.client",
    "pythoncom",
    "pywintypes",
]

# ctranslate2 / faster-whisper ship native libraries and asset files that
# PyInstaller's static analysis doesn't see on its own.
binaries += collect_dynamic_libs("ctranslate2")
datas += collect_data_files("faster_whisper")

# sounddevice bundles the PortAudio DLL as package data.
binaries += collect_dynamic_libs("sounddevice")
datas += collect_data_files("sounddevice")

# edge-tts and the Google API client read packaged data files at runtime.
datas += collect_data_files("edge_tts")
datas += collect_data_files("googleapiclient")


a = Analysis(
    ["run_neo.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NEO",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="NEO",
)
