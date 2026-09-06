from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from .base import RiskLevel, Tool, ToolResult

logger = logging.getLogger(__name__)

MAX_RESULTS = 20
SEARCH_TIME_BUDGET_SECONDS = 6.0

_SKIP_DIRS = {"node_modules", "__pycache__", ".git", ".venv", "AppData", "$RECYCLE.BIN"}


def _known_folders() -> dict[str, Path]:
    home = Path.home()
    folders = {
        "masaüstü": home / "Desktop",
        "desktop": home / "Desktop",
        "indirilenler": home / "Downloads",
        "downloads": home / "Downloads",
        "belgeler": home / "Documents",
        "documents": home / "Documents",
        "resimler": home / "Pictures",
        "pictures": home / "Pictures",
        "müzik": home / "Music",
        "videolar": home / "Videos",
        "ev": home,
    }
    onedrive = os.getenv("OneDrive")
    if onedrive:
        folders["onedrive"] = Path(onedrive)
    return folders


def _resolve_folder(name: str) -> Path | None:
    candidate = Path(name).expanduser()
    if candidate.is_dir():
        return candidate
    return _known_folders().get(name.strip().lower())


class FindFileTool(Tool):
    name = "find_file"
    description = (
        "Kullanıcının klasörlerinde dosya arar (ad ya da uzantı parçasına göre). "
        "'location' verilmezse Masaüstü, İndirilenler ve Belgeler taranır. "
        "Sadece okur, hiçbir dosyayı değiştirmez."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Aranan dosya adının bir parçası ya da uzantı (ör. 'rapor', '.pdf').",
            },
            "location": {
                "type": "string",
                "description": (
                    "İsteğe bağlı klasör: 'masaüstü', 'indirilenler', 'belgeler' "
                    "gibi bir isim ya da tam yol."
                ),
            },
        },
        "required": ["query"],
    }

    async def run(self, query: str, location: str | None = None, **kwargs: object) -> ToolResult:
        return await asyncio.to_thread(self._search, query, location)

    def _search(self, query: str, location: str | None) -> ToolResult:
        needle = query.strip().lower()
        if not needle:
            return ToolResult(success=False, error="Ne arayacağımı belirtmen gerekiyor.")

        if location:
            root = _resolve_folder(location)
            if root is None:
                return ToolResult(success=False, error=f"'{location}' diye bir klasör bulamadım.")
            roots = [root]
        else:
            home = Path.home()
            roots = [home / "Desktop", home / "Downloads", home / "Documents"]

        deadline = time.monotonic() + SEARCH_TIME_BUDGET_SECONDS
        matches: list[dict[str, object]] = []
        truncated = False

        for root in roots:
            if not root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
                if time.monotonic() > deadline:
                    truncated = True
                    break
                for filename in filenames:
                    if needle not in filename.lower():
                        continue
                    full_path = Path(dirpath) / filename
                    try:
                        stat = full_path.stat()
                    except OSError:
                        continue
                    matches.append(
                        {
                            "name": filename,
                            "path": str(full_path),
                            "size_kb": round(stat.st_size / 1024, 1),
                            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                        }
                    )
                    if len(matches) >= MAX_RESULTS:
                        truncated = True
                        break
                if len(matches) >= MAX_RESULTS or truncated:
                    break
            if len(matches) >= MAX_RESULTS or truncated:
                break

        return ToolResult(
            success=True,
            data={"query": query, "matches": matches, "count": len(matches), "truncated": truncated},
        )


class OpenFolderTool(Tool):
    name = "open_folder"
    description = (
        "Bir klasörü Dosya Gezgini'nde açar. 'masaüstü', 'indirilenler', "
        "'belgeler' gibi bir isim ya da tam yol verilebilir."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "folder": {
                "type": "string",
                "description": "Açılacak klasörün adı ya da tam yolu.",
            }
        },
        "required": ["folder"],
    }

    async def run(self, folder: str, **kwargs: object) -> ToolResult:
        target = _resolve_folder(folder)
        if target is None:
            return ToolResult(success=False, error=f"'{folder}' diye bir klasör bulamadım.")
        try:
            os.startfile(str(target))
        except OSError:
            logger.exception("Klasör açılamadı: %s", target)
            return ToolResult(success=False, error=f"'{folder}' açılırken bir hata oluştu.")
        return ToolResult(success=True, data={"opened": str(target)})
