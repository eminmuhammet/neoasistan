import asyncio

import pytest

from neo.memory.preference_store import PreferenceStore
from neo.tools.base import RiskLevel
from neo.tools.preferences import (
    ForgetPreferenceTool,
    RecallPreferencesTool,
    RememberPreferenceTool,
)


@pytest.fixture
def store(tmp_path):
    return PreferenceStore(tmp_path / "preferences.db")


def test_remember_preference_succeeds(store):
    result = asyncio.run(RememberPreferenceTool(store).run(key="şehir", value="Bursa"))

    assert result.success is True
    assert store.get("şehir").value == "Bursa"


def test_remember_preference_refuses_a_credential_looking_key(store):
    """The tool must surface the store's guard as a failed ToolResult, not
    let the exception propagate and crash the tool-use loop."""
    result = asyncio.run(RememberPreferenceTool(store).run(key="şifre", value="1234"))

    assert result.success is False
    assert "kimlik" in result.error
    assert store.all() == []


def test_remember_preference_rejects_an_empty_key(store):
    result = asyncio.run(RememberPreferenceTool(store).run(key="   ", value="bir şey"))
    assert result.success is False


def test_recall_all_preferences(store):
    store.remember("şehir", "Bursa")
    store.remember("meslek", "mühendis")

    result = asyncio.run(RecallPreferencesTool(store).run())

    assert result.success is True
    keys = {p["key"] for p in result.data["preferences"]}
    assert keys == {"şehir", "meslek"}


def test_recall_specific_key_found(store):
    store.remember("şehir", "Bursa")

    result = asyncio.run(RecallPreferencesTool(store).run(key="şehir"))

    assert result.data == {"found": True, "key": "şehir", "value": "Bursa"}


def test_recall_specific_key_not_found(store):
    result = asyncio.run(RecallPreferencesTool(store).run(key="yok"))
    assert result.data == {"found": False}


def test_recall_on_empty_store_returns_empty_list(store):
    result = asyncio.run(RecallPreferencesTool(store).run())
    assert result.data["preferences"] == []


def test_forget_preference_succeeds(store):
    store.remember("şehir", "Bursa")

    result = asyncio.run(ForgetPreferenceTool(store).run(key="şehir"))

    assert result.success is True
    assert store.get("şehir") is None


def test_forget_missing_preference_fails_cleanly(store):
    result = asyncio.run(ForgetPreferenceTool(store).run(key="yok"))
    assert result.success is False


def test_risk_levels_match_calendar_note_precedent(store):
    """Adding/reading are LOW risk; deleting the user's own remembered data
    is MEDIUM, same reasoning as delete_calendar_note -- irreversible from
    NEO's side, so it goes through the confirmation dialog."""
    assert RememberPreferenceTool(store).risk is RiskLevel.LOW
    assert RecallPreferencesTool(store).risk is RiskLevel.LOW
    assert ForgetPreferenceTool(store).risk is RiskLevel.MEDIUM
