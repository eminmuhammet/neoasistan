"""The wake-up cue: spoken acknowledgements ("Dinliyorum efendim.") cached to
disk, with a plain-tone fallback. Module-level cache state is reset around
each test since chime.py tracks readiness globally by design (the cache
outlives any single object)."""

import asyncio

import pytest

from neo.voice import chime


@pytest.fixture(autouse=True)
def reset_module_state():
    chime._cache_dir = None
    chime._ready.clear()
    yield
    chime._cache_dir = None
    chime._ready.clear()


def test_configure_sets_the_cache_dir(tmp_path):
    chime.configure(tmp_path)
    assert chime._cache_dir == tmp_path / "cues"


def test_prepare_cues_without_configure_does_nothing():
    """configure() is called once at startup; a missing call must not crash
    the prepare step, just leave the fallback tone as the only cue."""
    asyncio.run(chime.prepare_cues())
    assert chime._ready == []


def test_prepare_cues_synthesizes_missing_files(tmp_path, monkeypatch):
    chime.configure(tmp_path)
    saved = []

    class FakeCommunicate:
        def __init__(self, text, voice):
            self.text = text

        async def save(self, path):
            saved.append((self.text, path))
            with open(path, "wb") as f:
                f.write(b"fake-mp3-bytes")

    monkeypatch.setattr(
        "edge_tts.Communicate", FakeCommunicate, raising=False
    )
    import sys
    import types

    fake_module = types.SimpleNamespace(Communicate=FakeCommunicate)
    monkeypatch.setitem(sys.modules, "edge_tts", fake_module)

    asyncio.run(chime.prepare_cues())

    assert len(chime._ready) == len(chime.CUE_PHRASES)
    assert len(saved) == len(chime.CUE_PHRASES)
    for path in chime._ready:
        assert path.exists()
        assert path.stat().st_size > 0


def test_prepare_cues_skips_already_synthesized_files(tmp_path, monkeypatch):
    """Restarting NEO shouldn't re-synthesize cues that already exist on
    disk from a previous run."""
    chime.configure(tmp_path)
    (tmp_path / "cues").mkdir(parents=True)
    existing = tmp_path / "cues" / "cue_0.mp3"
    existing.write_bytes(b"already-there")

    calls = []

    class FakeCommunicate:
        def __init__(self, text, voice):
            calls.append(text)

        async def save(self, path):
            with open(path, "wb") as f:
                f.write(b"new-bytes")

    import sys
    import types

    monkeypatch.setitem(
        sys.modules, "edge_tts", types.SimpleNamespace(Communicate=FakeCommunicate)
    )

    asyncio.run(chime.prepare_cues())

    assert chime.CUE_PHRASES[0] not in calls
    assert existing.read_bytes() == b"already-there"


def test_prepare_cues_is_idempotent_across_repeated_calls(tmp_path, monkeypatch):
    """Calling prepare_cues() twice (e.g. two starts within one process)
    must not duplicate entries in the ready list -- that would silently
    skew which cue gets picked."""
    chime.configure(tmp_path)

    class FakeCommunicate:
        def __init__(self, text, voice):
            pass

        async def save(self, path):
            with open(path, "wb") as f:
                f.write(b"bytes")

    import sys
    import types

    monkeypatch.setitem(
        sys.modules, "edge_tts", types.SimpleNamespace(Communicate=FakeCommunicate)
    )

    asyncio.run(chime.prepare_cues())
    first_count = len(chime._ready)
    asyncio.run(chime.prepare_cues())

    assert len(chime._ready) == first_count == len(chime.CUE_PHRASES)


def test_prepare_cues_without_edge_tts_leaves_ready_empty(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "edge_tts":
            raise ImportError("yok")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    chime.configure(tmp_path)

    asyncio.run(chime.prepare_cues())

    assert chime._ready == []


def test_play_activation_chime_uses_a_ready_file(tmp_path, monkeypatch):
    chime.configure(tmp_path)
    fake = tmp_path / "cue_0.mp3"
    fake.write_bytes(b"x")
    chime._ready.append(fake)

    played = []
    monkeypatch.setattr(chime, "_play_file", lambda path: played.append(path))

    asyncio.run(chime.play_activation_chime())

    assert played == [fake]


def test_play_activation_chime_falls_back_when_nothing_ready(monkeypatch):
    toned = []
    monkeypatch.setattr(chime, "_fallback_tone", lambda: toned.append(True))

    asyncio.run(chime.play_activation_chime())

    assert toned == [True]


def test_play_activation_chime_falls_back_on_playback_failure(tmp_path, monkeypatch):
    """A corrupted or locked cue file must not leave the user with silence
    instead of at least the fallback tone."""
    chime.configure(tmp_path)
    fake = tmp_path / "cue_0.mp3"
    fake.write_bytes(b"x")
    chime._ready.append(fake)

    def boom(path):
        raise OSError("kilitli dosya")

    toned = []
    monkeypatch.setattr(chime, "_play_file", boom)
    monkeypatch.setattr(chime, "_fallback_tone", lambda: toned.append(True))

    asyncio.run(chime.play_activation_chime())

    assert toned == [True]
