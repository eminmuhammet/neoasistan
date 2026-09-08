"""OpenWakeWordSpotter is a thin wrapper around openwakeword.model.Model.

The real model needs a one-time download and real audio to exercise
meaningfully -- that was checked by hand against synthesized "Hey Jarvis"
audio before adopting it (0.999 for genuine attempts, 0.000 for unrelated
Turkish and English speech, see the module's docstring). These tests cover
the wrapper's own logic -- lazy loading, float->int16 conversion, threshold
plumbing -- against a fake model, the same way test_stt.py covers WhisperSTT
without downloading a Whisper model.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from neo.voice.openwakeword_spotter import (
    DEFAULT_DEBOUNCE_SECONDS,
    DEFAULT_MODEL,
    OpenWakeWordSpotter,
    WakeWordModelUnavailableError,
)


class _FakeModel:
    def __init__(self, score: float = 0.0) -> None:
        self.score = score
        self.calls: list[dict] = []

    def predict(self, pcm16, threshold, debounce_time):
        self.calls.append(
            {"pcm16": pcm16, "threshold": threshold, "debounce_time": debounce_time}
        )
        return {DEFAULT_MODEL: self.score}


def test_a_high_score_counts_as_the_wake_word():
    spotter = OpenWakeWordSpotter()
    fake = _FakeModel(score=0.93)
    with patch.object(OpenWakeWordSpotter, "_ensure_model", return_value=fake):
        heard = spotter.feed(np.zeros(1280, dtype=np.float32))
    assert heard is True


def test_a_low_score_does_not():
    spotter = OpenWakeWordSpotter()
    fake = _FakeModel(score=0.02)
    with patch.object(OpenWakeWordSpotter, "_ensure_model", return_value=fake):
        heard = spotter.feed(np.zeros(1280, dtype=np.float32))
    assert heard is False


def test_the_configured_threshold_is_what_gets_passed_to_the_model():
    spotter = OpenWakeWordSpotter(model_name="hey_jarvis", threshold=0.75)
    fake = _FakeModel(score=0.6)
    with patch.object(OpenWakeWordSpotter, "_ensure_model", return_value=fake):
        heard = spotter.feed(np.zeros(1280, dtype=np.float32))
    assert heard is False  # 0.6 < the configured 0.75
    assert fake.calls[0]["threshold"] == {"hey_jarvis": 0.75}


def test_the_debounce_setting_is_passed_through():
    spotter = OpenWakeWordSpotter(debounce_seconds=2.5)
    fake = _FakeModel(score=0.9)
    with patch.object(OpenWakeWordSpotter, "_ensure_model", return_value=fake):
        spotter.feed(np.zeros(1280, dtype=np.float32))
    assert fake.calls[0]["debounce_time"] == 2.5


def test_default_debounce_matches_the_documented_constant():
    assert DEFAULT_DEBOUNCE_SECONDS > 0


def test_float_audio_is_converted_to_int16_pcm():
    spotter = OpenWakeWordSpotter()
    fake = _FakeModel(score=0.0)
    chunk = np.array([1.0, -1.0, 0.5, -0.5, 0.0], dtype=np.float32)
    with patch.object(OpenWakeWordSpotter, "_ensure_model", return_value=fake):
        spotter.feed(chunk)
    pcm16 = fake.calls[0]["pcm16"]
    assert pcm16.dtype == np.int16
    assert pcm16[0] == 32767  # +1.0 clamped to int16 max, not overflowed
    assert pcm16[1] == -32767


def test_out_of_range_audio_is_clipped_not_wrapped():
    """Guards against a rare mic sample slightly over 1.0 becoming a
    negative int16 through overflow instead of just clipping."""
    spotter = OpenWakeWordSpotter()
    fake = _FakeModel(score=0.0)
    chunk = np.array([1.3, -1.3], dtype=np.float32)
    with patch.object(OpenWakeWordSpotter, "_ensure_model", return_value=fake):
        spotter.feed(chunk)
    pcm16 = fake.calls[0]["pcm16"]
    assert pcm16[0] > 0 and pcm16[1] < 0


def test_a_cached_model_is_returned_without_touching_imports():
    """_ensure_model is expensive (loads an ONNX session, and on first run
    downloads weights); once self._model is set, the real _ensure_model
    must return it immediately rather than importing or loading again."""
    spotter = OpenWakeWordSpotter()
    sentinel = _FakeModel()
    spotter._model = sentinel
    assert spotter._ensure_model() is sentinel


def test_missing_dependency_raises_a_clear_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("openwakeword"):
            raise ImportError("no module named openwakeword")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    spotter = OpenWakeWordSpotter()
    try:
        spotter._ensure_model()
        assert False, "should have raised"
    except WakeWordModelUnavailableError:
        pass


def test_preload_populates_the_cached_model():
    spotter = OpenWakeWordSpotter()
    fake = _FakeModel()
    with patch.object(OpenWakeWordSpotter, "_ensure_model", return_value=fake) as mocked:
        spotter.preload()
    mocked.assert_called_once()


class _FakeLoadedModel:
    """Stands in for openwakeword.model.Model, recording the paths it was
    constructed with and exposing a `models` dict keyed the way the real
    Model would key it when loaded from an explicit file path (by filename
    stem, which is not necessarily the bare model name)."""

    def __init__(self, wakeword_models, inference_framework, melspec_model_path,
                 embedding_model_path):
        self.wakeword_models = wakeword_models
        self.inference_framework = inference_framework
        self.melspec_model_path = melspec_model_path
        self.embedding_model_path = embedding_model_path
        import os as _os
        stem = _os.path.splitext(_os.path.basename(wakeword_models[0]))[0]
        self.models = {stem: object()}

    def predict(self, pcm16, threshold, debounce_time):
        return {k: 0.0 for k in self.models}


def test_models_are_downloaded_and_loaded_from_the_persistent_data_dir(tmp_path, monkeypatch):
    """The bug this guards: openWakeWord's own default download location is
    a folder inside the installed package. Under PyInstaller that resolves
    inside the bundle's _internal directory and gets wiped on every update
    -- exactly the class of bug settings.user_data_root()'s docstring
    describes for the calendar and wake-word-enrollment data this module
    replaces. Weights must go to (and load from) wherever this spotter was
    told to keep them, not the package's install location.
    """
    import openwakeword
    import openwakeword.utils as oww_utils

    downloads = []
    monkeypatch.setattr(
        oww_utils, "download_models",
        lambda model_names, target_directory: downloads.append((model_names, target_directory)),
    )
    monkeypatch.setattr("openwakeword.model.Model", _FakeLoadedModel)

    models_dir = tmp_path / "wake_word_models"
    spotter = OpenWakeWordSpotter(model_name="hey_jarvis", models_dir=models_dir)
    spotter.preload()

    assert downloads == [(["hey_jarvis"], str(models_dir))]

    model = spotter._model
    for path in (model.wakeword_models[0], model.melspec_model_path, model.embedding_model_path):
        assert str(models_dir) in path, f"{path} was not placed under the configured data dir"
    assert "hey_jarvis" in model.wakeword_models[0]


def test_the_score_key_is_read_back_from_the_loaded_model_not_assumed(tmp_path, monkeypatch):
    """Loading from an explicit file path (required by the test above) keys
    the model by filename stem, e.g. "hey_jarvis_v0.1" -- not necessarily
    the bare "hey_jarvis" this spotter was constructed with. feed() has to
    look up scores under whatever key the model actually used, or a future
    openWakeWord release changing that filename scheme would make feed()
    silently stop detecting anything (looking up a key with no matching
    entry defaults to 0.0, which reads as "never heard").
    """
    import openwakeword.utils as oww_utils

    monkeypatch.setattr(
        oww_utils, "download_models", lambda model_names, target_directory: None
    )
    monkeypatch.setattr("openwakeword.model.Model", _FakeLoadedModel)

    spotter = OpenWakeWordSpotter(model_name="hey_jarvis", models_dir=tmp_path)
    spotter.preload()

    assert spotter._score_key != "hey_jarvis"
    assert spotter._score_key.startswith("hey_jarvis")


def test_without_a_models_dir_it_falls_back_to_the_package_default(monkeypatch):
    """Only meant for quick scripts/tests -- main.py always passes
    settings.data_dir -- but must not crash when omitted."""
    import openwakeword.utils as oww_utils

    calls = []
    monkeypatch.setattr(
        oww_utils, "download_models",
        lambda model_names, target_directory: calls.append(target_directory),
    )
    monkeypatch.setattr("openwakeword.model.Model", _FakeLoadedModel)

    spotter = OpenWakeWordSpotter(model_name="hey_jarvis")
    spotter.preload()

    assert calls and "resources" in calls[0] and "models" in calls[0]
