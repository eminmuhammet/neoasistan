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


@pytest.mark.parametrize(
    "transcript",
    ["Ne o ya?", "Ne oluyor?", "ne o", "neo", "ne oldu", "ne yapıyorsun"],
)
def test_near_misses_that_share_letters_are_rejected(transcript):
    """"Ne o ya?" is ordinary Turkish filler that shares almost all of its
    letters, in order, with "neo uyan" -- it scored 0.80 and woke NEO
    mid-conversation on the old 0.72 threshold."""
    assert matches_wake_phrase(transcript, PHRASE) is False


def test_the_measured_gap_still_holds():
    """Guards the margin the threshold sits in, so a change to normalization
    or scoring can't quietly close it."""
    genuine = min(
        similarity(t, PHRASE) for t in ["Neo uyan", "neyo uyan", "neo uyar", "Ne o uyan"]
    )
    nearest_false = max(
        similarity(t, PHRASE) for t in ["Ne o ya?", "Ne oluyor?", "ne o", "neo", "ne oldu"]
    )

    assert genuine > nearest_false, f"gerçek {genuine:.2f} <= yanlış {nearest_false:.2f}"
    assert nearest_false < 0.85 <= genuine


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
