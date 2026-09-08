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

import os

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


def _is_unused_discovery_doc(dest_path):
    """googleapiclient ships a discovery document for every Google API it
    knows about -- 600 files, 100 MB in the built bundle. NEO calls exactly
    one of them (Calendar v3, 0.13 MB); the rest is weight in every release
    the updater has to download.

    Filtering the `datas` list before Analysis did nothing: PyInstaller's own
    googleapiclient hook collects the cache independently and put all 600
    back. They have to be removed from the finished analysis instead.
    """
    path = str(dest_path).replace("\\", "/").lower()
    if "discovery_cache/documents/" not in path or not path.endswith(".json"):
        return False
    return not os.path.basename(path).startswith("calendar.")


a = Analysis(
    ["run_neo.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "pytest",
        # NEO's interface is plain QtWidgets: no QML/Quick scene graph, no
        # 3D, charts, multimedia, PDF viewer or embedded browser.
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuickWidgets",
        "PySide6.Qt3DCore",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtMultimedia",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
    ],
    noarchive=False,
)

# Qt's own DLLs are pulled in by PySide6's hook regardless of the module
# excludes above, so they are dropped here as well. opengl32sw is Qt's
# software OpenGL fallback (19.7 MB) and a widgets-only interface never
# reaches for it.
_UNUSED_QT_BINARIES = (
    "opengl32sw",
    "Qt6Quick",
    "Qt6Qml",
    "Qt6Pdf",
    "Qt6WebEngine",
    "Qt6Multimedia",
    "Qt63D",
    "Qt6Charts",
    "Qt6DataVisualization",
)
a.binaries = [
    entry
    for entry in a.binaries
    if not any(unused.lower() in os.path.basename(entry[0]).lower() for unused in _UNUSED_QT_BINARIES)
]

a.datas = [entry for entry in a.datas if not _is_unused_discovery_doc(entry[0])]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NEO",
    icon="neo.ico",
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
