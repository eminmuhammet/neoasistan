import pytest

from neo.voice.stt import is_hallucination

# Whisper's subtitle-boilerplate artefacts. 'Altyazı M.K.' is not a synthetic
# example: logs/neo.log at 20:36:00 shows it transcribed from ambient audio
# and forwarded to the Claude API as though the user had said it.
HALLUCINATIONS = [
    "Altyazı M.K.",
    "Altyazı M.K",
    "altyazı m.k.",
    "ALTYAZI M.K.",
    "Abone olmayı unutmayın",
    "abone olmayı unutmayın!",
    "İzlediğiniz için teşekkürler",
    "İzlediğiniz için teşekkür ederim.",
    "Kanalıma abone olun",
    "Türkçe altyazı",
]

REAL_COMMANDS = [
    "Bugünkü programımda neler var",
    "Hava durumu nasıl?",
    "Alo",
    "Naber?",
    "Merhaba Neo, nasılsın?",
    "Bilgisayarı kilitle",
    "Yarın saat 15'te dişçi randevusu var, not al",
    "Neo araştırma modu",
    "Spotify'ı aç",
]


@pytest.mark.parametrize("text", HALLUCINATIONS)
def test_known_whisper_artefacts_are_filtered(text):
    assert is_hallucination(text) is True


@pytest.mark.parametrize("text", REAL_COMMANDS)
def test_real_commands_are_not_filtered(text):
    assert is_hallucination(text) is False


def test_empty_transcript_is_filtered():
    assert is_hallucination("") is True
    assert is_hallucination("   ") is True
    assert is_hallucination("...") is True


def test_filter_is_case_and_diacritic_insensitive():
    # Whisper's casing/diacritics vary between runs on the same audio, so
    # matching has to survive both.
    assert is_hallucination("ALTYAZI M.K.") is True
    assert is_hallucination("Altyazi M.K.") is True


def test_word_containing_a_blocked_phrase_is_not_filtered():
    # Substring matching would wrongly kill a legitimate request that merely
    # mentions subtitles; only whole-phrase matches count.
    assert is_hallucination("Altyazıları aç") is False
    assert is_hallucination("Bu videonun altyazısını indirir misin") is False
