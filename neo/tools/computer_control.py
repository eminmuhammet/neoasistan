from __future__ import annotations

import asyncio
import logging

from ..core.automation_panic import engage_panic, is_panic_engaged
from .base import RiskLevel, Tool, ToolResult
from .screen import ScreenCaptureUnavailableError, _capture_and_encode

logger = logging.getLogger(__name__)

_PANIC_MESSAGE = (
    "Otomasyon panik tuşuyla (Ctrl+Alt+Shift+Q) veya fareyi ekran köşesine "
    "götürerek durduruldu. Devam etmek için NEO'yu yeniden başlat."
)


def _pyautogui():
    import pyautogui

    # Never turned off: moving the mouse to a screen corner must always be
    # able to abort automation, no matter which tool is mid-action.
    pyautogui.FAILSAFE = True
    return pyautogui


async def _verify_with_screenshot() -> dict:
    """Automatic post-action verification per the plan: every computer
    control tool hands back a fresh screenshot so the caller (and the
    model, on its next turn) can see what the action actually did rather
    than trusting that it worked."""
    try:
        encoded, width, height = await asyncio.to_thread(_capture_and_encode, 1)
    except ScreenCaptureUnavailableError:
        return {}
    return {"image_base64": encoded, "media_type": "image/png", "width": width, "height": height}


def _panic_result() -> ToolResult:
    return ToolResult(success=False, error=_PANIC_MESSAGE)


class ClickTool(Tool):
    name = "click"
    description = (
        "Ekranda belirtilen (x, y) koordinatına fare ile tıklar. Önce "
        "capture_screen ile ekrana bakıp doğru koordinatı belirle. Sadece "
        "yardımcı modunda çalışır."
    )
    risk = RiskLevel.MEDIUM
    input_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Ekran x koordinatı (piksel)."},
            "y": {"type": "integer", "description": "Ekran y koordinatı (piksel)."},
            "button": {"type": "string", "enum": ["left", "right", "middle"]},
            "double": {"type": "boolean", "description": "Çift tıklama."},
        },
        "required": ["x", "y"],
    }

    async def run(
        self, x: int, y: int, button: str = "left", double: bool = False, **kwargs: object
    ) -> ToolResult:
        if is_panic_engaged():
            return _panic_result()
        pyautogui = _pyautogui()
        try:
            clicks = 2 if double else 1
            await asyncio.to_thread(pyautogui.click, int(x), int(y), clicks=clicks, button=button)
        except pyautogui.FailSafeException:
            engage_panic()
            return _panic_result()
        return ToolResult(
            success=True,
            data={"x": x, "y": y, "button": button, **await _verify_with_screenshot()},
        )


class MoveMouseTool(Tool):
    name = "move_mouse"
    description = "Fareyi tıklamadan sadece belirtilen (x, y) koordinatına götürür."
    risk = RiskLevel.MEDIUM
    input_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        },
        "required": ["x", "y"],
    }

    async def run(self, x: int, y: int, **kwargs: object) -> ToolResult:
        if is_panic_engaged():
            return _panic_result()
        pyautogui = _pyautogui()
        try:
            await asyncio.to_thread(pyautogui.moveTo, int(x), int(y), 0.2)
        except pyautogui.FailSafeException:
            engage_panic()
            return _panic_result()
        return ToolResult(success=True, data={"x": x, "y": y})


class DragTool(Tool):
    name = "drag"
    description = (
        "Fareyi (start_x, start_y) konumundan basılı tutarak (end_x, end_y) "
        "konumuna sürükler (ör. bir dosyayı sürükle-bırak, bir kaydırıcıyı "
        "hareket ettirmek için)."
    )
    risk = RiskLevel.MEDIUM
    input_schema = {
        "type": "object",
        "properties": {
            "start_x": {"type": "integer"},
            "start_y": {"type": "integer"},
            "end_x": {"type": "integer"},
            "end_y": {"type": "integer"},
        },
        "required": ["start_x", "start_y", "end_x", "end_y"],
    }

    async def run(
        self, start_x: int, start_y: int, end_x: int, end_y: int, **kwargs: object
    ) -> ToolResult:
        if is_panic_engaged():
            return _panic_result()
        pyautogui = _pyautogui()

        def _do_drag() -> None:
            pyautogui.moveTo(int(start_x), int(start_y))
            pyautogui.dragTo(int(end_x), int(end_y), duration=0.3)

        try:
            await asyncio.to_thread(_do_drag)
        except pyautogui.FailSafeException:
            engage_panic()
            return _panic_result()
        return ToolResult(
            success=True,
            data={
                "start_x": start_x, "start_y": start_y, "end_x": end_x, "end_y": end_y,
                **await _verify_with_screenshot(),
            },
        )


class TypeTextTool(Tool):
    name = "type_text"
    description = (
        "İmlecin bulunduğu yere (aktif pencere/alan) verilen metni yazar. "
        "Doğru alana odaklanıldığından emin olmak için önce click ile o "
        "alana tıkla."
    )
    risk = RiskLevel.MEDIUM
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Yazılacak metin."},
        },
        "required": ["text"],
    }

    async def run(self, text: str, **kwargs: object) -> ToolResult:
        if is_panic_engaged():
            return _panic_result()
        pyautogui = _pyautogui()
        try:
            await asyncio.to_thread(pyautogui.write, text, 0.02)
        except pyautogui.FailSafeException:
            engage_panic()
            return _panic_result()
        return ToolResult(success=True, data={"length": len(text), **await _verify_with_screenshot()})


class PressKeysTool(Tool):
    name = "press_keys"
    description = (
        "Bir veya birden fazla tuşa aynı anda basar (kısayol tuşu), ör. "
        "['ctrl', 'c'] kopyalamak için, ['enter'] Enter'a basmak için, "
        "['alt', 'tab'] pencere değiştirmek için."
    )
    risk = RiskLevel.MEDIUM
    input_schema = {
        "type": "object",
        "properties": {
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Basılacak tuşlar, ör. ['ctrl', 'c'].",
            },
        },
        "required": ["keys"],
    }

    async def run(self, keys: list[str], **kwargs: object) -> ToolResult:
        if is_panic_engaged():
            return _panic_result()
        if not keys:
            return ToolResult(success=False, error="En az bir tuş belirtilmeli.")
        pyautogui = _pyautogui()
        try:
            await asyncio.to_thread(pyautogui.hotkey, *keys)
        except pyautogui.FailSafeException:
            engage_panic()
            return _panic_result()
        return ToolResult(success=True, data={"keys": keys, **await _verify_with_screenshot()})
