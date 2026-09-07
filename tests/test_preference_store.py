"""PreferenceStore: durable facts about the user ("şehir: Bursa"), distinct
from ConversationStore's transcript of what was said. Keyed by a normalized
key so a later correction overwrites the earlier fact instead of leaving
both a stale and a current answer sitting side by side."""

import pytest

from neo.memory.preference_store import PreferenceStore, SensitivePreferenceError


def _store(tmp_path):
    return PreferenceStore(tmp_path / "preferences.db")


def test_remember_and_get(tmp_path):
    store = _store(tmp_path)
    store.remember("şehir", "Bursa")

    pref = store.get("şehir")
    assert pref is not None
    assert pref.value == "Bursa"
    assert pref.source == "sohbet"


def test_get_missing_key_returns_none(tmp_path):
    assert _store(tmp_path).get("yok") is None


def test_remember_overwrites_rather_than_duplicates(tmp_path):
    """A later correction ("aslında İstanbul'da yaşıyorum") must replace the
    old fact, not leave two rows answering the same question differently."""
    store = _store(tmp_path)
    store.remember("şehir", "Bursa")
    store.remember("şehir", "İstanbul")

    assert store.get("şehir").value == "İstanbul"
    assert len(store.all()) == 1


def test_key_matching_is_case_and_whitespace_insensitive(tmp_path):
    """Claude's own phrasing of a key can vary slightly between calls
    ("Şehir" vs "şehir "); without normalization each variant would pile up
    as a separate, stale fact."""
    store = _store(tmp_path)
    store.remember("Şehir", "Bursa")

    assert store.get("şehir").value == "Bursa"
    assert store.get("  ŞEHİR  ").value == "Bursa"
    store.remember("şehir ", "İstanbul")
    assert len(store.all()) == 1
    assert store.get("Şehir").value == "İstanbul"


def test_forget_removes_the_preference(tmp_path):
    store = _store(tmp_path)
    store.remember("şehir", "Bursa")

    assert store.forget("şehir") is True
    assert store.get("şehir") is None


def test_forget_missing_key_reports_nothing_removed(tmp_path):
    assert _store(tmp_path).forget("yok") is False


def test_all_returns_every_preference_sorted_by_key(tmp_path):
    store = _store(tmp_path)
    store.remember("meslek", "mühendis")
    store.remember("şehir", "Bursa")

    keys = [p.key for p in store.all()]
    assert keys == sorted(keys)
    assert set(keys) == {"meslek", "şehir"}


def test_all_is_empty_for_a_fresh_store(tmp_path):
    assert _store(tmp_path).all() == []


def test_empty_key_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _store(tmp_path).remember("   ", "bir şey")


@pytest.mark.parametrize(
    "key",
    [
        "şifre",
        "Şifre",
        "sifre",
        "parola",
        "password",
        "pin",
        "cvv",
        "cvc",
        "kart no",
        "kart numarası",
        "iban",
        "tc kimlik",
        "tc no",
        "kimlik no",
        "banka şifresi",  # contains "şifre" as a substring of a longer key
    ],
)
def test_credential_looking_keys_are_refused(tmp_path, key):
    """Defense in depth: the system prompt tells Claude never to remember a
    credential, but this is a second, deterministic gate at the actual
    persistence boundary so a prompt-level slip can't end up as plaintext
    on disk."""
    store = _store(tmp_path)
    with pytest.raises(SensitivePreferenceError):
        store.remember(key, "1234")

    assert store.all() == []


def test_a_phone_number_is_not_treated_as_sensitive(tmp_path):
    """The guard matches on the key's wording, not the value's shape --
    matching on "looks like a long number" would also block a legitimate
    phone number, which the user might reasonably want remembered."""
    store = _store(tmp_path)
    store.remember("eşinin telefonu", "05321234567")
    assert store.get("eşinin telefonu").value == "05321234567"


def test_source_can_be_recorded(tmp_path):
    store = _store(tmp_path)
    store.remember("şehir", "Bursa", source="konuşma 2026-09-07")
    assert store.get("şehir").source == "konuşma 2026-09-07"
