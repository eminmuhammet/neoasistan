import pytest

from neo.voice.phrase_match import matches_wake_phrase, normalize, similarity

PHRASE = "Neo uyan"


@pytest.mark.parametrize(
    "transcript",
    [
        "Neo uyan",
        "neo uyan",
        "Neo uyan.",
        "Neo, uyan!",
        "neyo uyan",  # recognizer hears the name loosely
        "neo uyanı",
        "neo uyar",
        "Ne o uyan",
        "NEO UYAN",
    ],
)
def test_plausible_renderings_are_accepted(transcript):
    """Speech-to-text will not reproduce a wake phrase exactly -- live logs
    showed "Neo" alone coming back as "Ne?", "Ne o?" and "No." Matching has
    to tolerate that while still rejecting unrelated speech."""
    assert matches_wake_phrase(transcript, PHRASE) is True


@pytest.mark.parametrize(
    "transcript",
    [
        "bugün hava nasıl",
        "programımda neler var",
        "mükemmel çalışıyor",
        "saat kaç",
        "nasılsın",
        "Altyazı M.K.",
        "bir toplantı ekle",
        "",
    ],
)
def test_unrelated_speech_is_rejected(transcript):
    assert matches_wake_phrase(transcript, PHRASE) is False


def test_phrase_inside_a_longer_utterance_is_found():
    """People run the wake phrase straight into the command."""
    assert matches_wake_phrase("neo uyan hava durumu nasıl", PHRASE) is True
    assert matches_wake_phrase("tamam neo uyan", PHRASE) is True


def test_normalize_folds_turkish_characters_and_punctuation():
    assert normalize("Neo, UYÂN!") == "neo uyan"
    assert normalize("  çğıöşü  ") == "cgiosu"


def test_similarity_is_ordered_sensibly():
    exact = similarity("neo uyan", PHRASE)
    close = similarity("neyo uyan", PHRASE)
    far = similarity("bugün hava nasıl", PHRASE)

    assert exact == pytest.approx(1.0)
    assert exact > close > far


def test_a_different_phrase_can_be_configured():
    """Detection is not hardcoded to one wake phrase."""
    assert matches_wake_phrase("bilgisayar dinle", "bilgisayar dinle") is True
    assert matches_wake_phrase("neo uyan", "bilgisayar dinle") is False
