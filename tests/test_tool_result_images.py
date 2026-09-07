"""_tool_result_content: how a tool's result becomes a tool_result message.

Most tools stay exactly as before (plain JSON string). A tool that captured
an image (screen vision) instead gets a real Anthropic image content block,
since the Messages API accepts tool_result content as either a string or a
list of blocks -- no separate "queue the image for later" plumbing needed.
"""

import json

from neo.core.agent import _tool_result_content


def test_ordinary_result_is_unchanged_json_string():
    """Every existing tool (calendar, weather, system info, ...) must keep
    producing exactly the same string as before this feature existed."""
    result = {"success": True, "temperature_c": 24, "city": "Bursa"}

    content = _tool_result_content(result)

    assert isinstance(content, str)
    assert json.loads(content) == result


def test_image_result_becomes_an_image_content_block():
    result = {
        "success": True,
        "image_base64": "QUJD",
        "media_type": "image/png",
        "width": 1400,
        "height": 900,
    }

    content = _tool_result_content(result)

    assert isinstance(content, list)
    assert len(content) == 2
    image_block, text_block = content
    assert image_block == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }
    assert text_block["type"] == "text"


def test_image_result_caption_excludes_the_raw_base64():
    """The text block is metadata (dimensions, success flag) for Claude to
    read alongside the picture -- it must not duplicate the (large) base64
    payload as text too, which would double the token cost for nothing."""
    result = {"success": True, "image_base64": "QUJD" * 1000, "media_type": "image/png"}

    content = _tool_result_content(result)
    caption = json.loads(content[1]["text"])

    assert "image_base64" not in caption
    assert caption["success"] is True


def test_missing_media_type_defaults_to_png():
    result = {"success": True, "image_base64": "QUJD"}
    content = _tool_result_content(result)
    assert content[0]["source"]["media_type"] == "image/png"


def test_empty_image_base64_is_treated_as_no_image():
    """A falsy value (empty string, None) must fall back to the plain JSON
    path rather than emitting a broken image block with no data."""
    result = {"success": False, "image_base64": "", "error": "yok"}
    content = _tool_result_content(result)
    assert isinstance(content, str)
