from __future__ import annotations

import asyncio
import base64
import io
import logging

from .base import RiskLevel, Tool, ToolResult

logger = logging.getLogger(__name__)

# Anthropic downscales images past ~1568px on the long edge anyway, so
# sending anything bigger only spends extra tokens for no extra clarity.
# 1400 sits comfortably under that with room to spare.
MAX_LONG_EDGE = 1400


class ScreenCaptureUnavailableError(Exception):
    """Raised when a screenshot can't be taken."""


def _capture_and_encode(monitor_index: int) -> tuple[str, int, int]:
    """Runs off the asyncio loop (mss/Pillow are both synchronous). Builds a
    fresh mss.mss() inside the call rather than sharing one across calls --
    mss instances are tied to the thread that created them, and
    asyncio.to_thread doesn't guarantee the same worker thread twice."""
    import mss
    from PIL import Image

    try:
        with mss.MSS() as sct:
            monitor = sct.monitors[monitor_index]
            raw = sct.grab(monitor)
            image = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    except Exception as exc:
        raise ScreenCaptureUnavailableError("Ekran görüntüsü alınamadı.") from exc

    long_edge = max(image.width, image.height)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded, image.width, image.height


class CaptureScreenTool(Tool):
    name = "capture_screen"
    description = (
        "Kullanıcının ekranının anlık bir görüntüsünü alır ve doğrudan sana "
        "gösterir -- bir hata mesajını okumak, ekrandaki bir yazıyı "
        "anlamak, açık bir pencerede ne olduğunu görmek için kullan. "
        "Sadece kullanıcı açıkça istediğinde çağır (ör. 'ekrana bak', "
        "'bu hatayı çöz', 'ekranımda ne yazıyor'); kendiliğinden, periyodik "
        "olarak veya kullanıcı istemeden asla çağırma."
    )
    # Read-only (nothing on the machine changes), but it exposes whatever is
    # currently on screen -- open chats, documents, other people's
    # messages -- to an external API call. That's a privacy decision the
    # human in front of the screen should confirm each time, the same bar
    # this codebase already sets for deleting the user's own data.
    risk = RiskLevel.MEDIUM
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        try:
            encoded, width, height = await asyncio.to_thread(_capture_and_encode, 1)
        except ScreenCaptureUnavailableError as exc:
            return ToolResult(success=False, error=str(exc))

        # No dedicated audit panel yet (planned separately), so the log is
        # the only trail right now that a screenshot was taken and sent out.
        logger.info("Ekran görüntüsü alındı ve gönderildi (%dx%d)", width, height)
        return ToolResult(
            success=True,
            data={
                "image_base64": encoded,
                "media_type": "image/png",
                "width": width,
                "height": height,
            },
        )
