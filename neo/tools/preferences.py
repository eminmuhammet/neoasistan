from __future__ import annotations

import asyncio
import logging

from ..memory.preference_store import PreferenceStore, SensitivePreferenceError
from .base import RiskLevel, Tool, ToolResult

logger = logging.getLogger(__name__)


class RememberPreferenceTool(Tool):
    name = "remember_preference"
    description = (
        "Kullanıcı hakkında kalıcı, tekrar tekrar işine yarayacak bir gerçeği "
        "hatırlar (ör. şehri, mesleği, nasıl hitap edilmek istediği, bir "
        "alışkanlığı). 'key' kısa ve tutarlı olsun (ör. 'şehir', "
        "'hitap_şekli') -- aynı bilgiyi farklı isimlerle kaydetme, var olan "
        "anahtarı güncelle. Şifre, TC kimlik no, kart no gibi kimlik "
        "bilgilerini ASLA bu araçla kaydetme."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Kısa, tutarlı bir anahtar (ör. 'şehir')."},
            "value": {"type": "string", "description": "Hatırlanacak bilgi."},
        },
        "required": ["key", "value"],
    }

    def __init__(self, store: PreferenceStore) -> None:
        self._store = store

    async def run(self, key: str, value: str, **kwargs: object) -> ToolResult:
        try:
            await asyncio.to_thread(self._store.remember, key, value)
        except SensitivePreferenceError as exc:
            return ToolResult(success=False, error=str(exc))
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        return ToolResult(success=True, data={"key": key, "value": value})


class RecallPreferencesTool(Tool):
    name = "recall_preferences"
    description = (
        "Kullanıcı hakkında NEO'nun şu ana kadar öğrendiği bilgileri döndürür. "
        "'key' verilirse sadece o bilgiyi, verilmezse hepsini listeler. "
        "Kullanıcı 'benim hakkımda ne biliyorsun' gibi bir şey sorduğunda kullan."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Belirli bir bilginin anahtarı (opsiyonel)."},
        },
    }

    def __init__(self, store: PreferenceStore) -> None:
        self._store = store

    async def run(self, key: str | None = None, **kwargs: object) -> ToolResult:
        if key:
            pref = await asyncio.to_thread(self._store.get, key)
            if pref is None:
                return ToolResult(success=True, data={"found": False})
            return ToolResult(
                success=True,
                data={"found": True, "key": pref.key, "value": pref.value},
            )
        prefs = await asyncio.to_thread(self._store.all)
        return ToolResult(
            success=True,
            data={"preferences": [{"key": p.key, "value": p.value} for p in prefs]},
        )


class ForgetPreferenceTool(Tool):
    name = "forget_preference"
    description = (
        "Daha önce hatırlanan bir bilgiyi kalıcı olarak siler (ör. artık "
        "doğru olmadığı için, ya da kullanıcı unutulmasını istediği için). "
        "Notun 'key' değerini recall_preferences ile öğrenebilirsin."
    )
    # Deleting the user's own remembered data, same reasoning as calendar
    # note deletion: irreversible from NEO's side, so it goes through the
    # confirmation dialog rather than running silently.
    risk = RiskLevel.MEDIUM
    input_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Silinecek bilginin anahtarı."},
        },
        "required": ["key"],
    }

    def __init__(self, store: PreferenceStore) -> None:
        self._store = store

    async def run(self, key: str, **kwargs: object) -> ToolResult:
        removed = await asyncio.to_thread(self._store.forget, key)
        if not removed:
            return ToolResult(success=False, error=f"'{key}' anahtarıyla kayıtlı bir bilgi yok.")
        return ToolResult(success=True, data={"key": key})
