from __future__ import annotations

import asyncio
import ctypes

from .base import RiskLevel, Tool, ToolResult

# Virtual-key codes for the extended media keys, same values Windows itself
# uses -- sending these through keybd_event is indistinguishable from a
# physical keyboard press, so it works with whatever app currently has
# media focus (Spotify, browser, etc.) without needing to know which one.
_VK_MEDIA_PLAY_PAUSE = 0xB3
_VK_MEDIA_NEXT_TRACK = 0xB0
_VK_MEDIA_PREV_TRACK = 0xB1

_KEYEVENTF_KEYUP = 0x0002


def _press_media_key(vk_code: int) -> None:
    ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk_code, 0, _KEYEVENTF_KEYUP, 0)


def _get_endpoint_volume():
    """Every call runs inside asyncio.to_thread, i.e. a plain worker thread
    with no COM apartment set up -- comtypes objects created without this
    silently fail on any method that actually touches the COM interface
    (SetMasterVolumeLevelScalar included), so each thread needs its own
    CoInitialize call before touching pycaw."""
    import comtypes
    from pycaw.pycaw import AudioUtilities

    comtypes.CoInitialize()
    return AudioUtilities.GetSpeakers().EndpointVolume


class GetVolumeTool(Tool):
    name = "get_volume"
    description = "Sistemin mevcut ses seviyesini (%0-100) ve sessize alınıp alınmadığını döner."
    risk = RiskLevel.LOW
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        def _read() -> tuple[int, bool]:
            endpoint = _get_endpoint_volume()
            return round(endpoint.GetMasterVolumeLevelScalar() * 100), bool(endpoint.GetMute())

        try:
            level, muted = await asyncio.to_thread(_read)
        except Exception:
            return ToolResult(success=False, error="Ses seviyesi okunamadı.")
        return ToolResult(success=True, data={"volume_percent": level, "muted": muted})


class SetVolumeTool(Tool):
    name = "set_volume"
    description = "Sistemin ses seviyesini belirtilen yüzdeye (0-100) ayarlar."
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "percent": {"type": "integer", "description": "0 ile 100 arası ses seviyesi."},
        },
        "required": ["percent"],
    }

    async def run(self, percent: int, **kwargs: object) -> ToolResult:
        clamped = max(0, min(100, int(percent)))

        def _set() -> None:
            # Same thread for both calls: COM apartment state (CoInitialize)
            # is per-thread, so fetching the endpoint and using it must
            # happen inside one asyncio.to_thread call, not two.
            _get_endpoint_volume().SetMasterVolumeLevelScalar(clamped / 100, None)

        try:
            await asyncio.to_thread(_set)
        except Exception:
            return ToolResult(success=False, error="Ses seviyesi ayarlanamadı.")
        return ToolResult(success=True, data={"volume_percent": clamped})


class SetMuteTool(Tool):
    name = "set_mute"
    description = "Sistem sesini sessize alır ya da sessizden çıkarır."
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "muted": {"type": "boolean", "description": "true: sessize al, false: sesi aç."},
        },
        "required": ["muted"],
    }

    async def run(self, muted: bool, **kwargs: object) -> ToolResult:
        def _set() -> None:
            _get_endpoint_volume().SetMute(bool(muted), None)

        try:
            await asyncio.to_thread(_set)
        except Exception:
            return ToolResult(success=False, error="Sessize alma durumu değiştirilemedi.")
        return ToolResult(success=True, data={"muted": bool(muted)})


class MediaControlTool(Tool):
    name = "media_control"
    description = (
        "O an çalan medyayı kontrol eder: 'play_pause' (oynat/duraklat), "
        "'next' (sonraki parça), 'previous' (önceki parça). Hangi uygulamada "
        "çaldığını bilmesi gerekmez -- Windows'un medya tuşlarını kullanır."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["play_pause", "next", "previous"]},
        },
        "required": ["action"],
    }

    _KEYS = {
        "play_pause": _VK_MEDIA_PLAY_PAUSE,
        "next": _VK_MEDIA_NEXT_TRACK,
        "previous": _VK_MEDIA_PREV_TRACK,
    }

    async def run(self, action: str, **kwargs: object) -> ToolResult:
        vk_code = self._KEYS.get(action)
        if vk_code is None:
            return ToolResult(success=False, error=f"Bilinmeyen eylem: {action!r}")
        try:
            await asyncio.to_thread(_press_media_key, vk_code)
        except Exception:
            return ToolResult(success=False, error="Medya kontrolü gönderilemedi.")
        return ToolResult(success=True, data={"action": action})
