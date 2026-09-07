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
    ["Ne oluyor?", "Mel oya", "ne o", "neo", "ne oldu", "ne yapıyorsun"],
)
def test_noise_transcripts_from_the_live_log_are_rejected(transcript):
    """These are what the recognizer actually produced from an empty room."""
    assert matches_wake_phrase(transcript, PHRASE) is False


@pytest.mark.parametrize("transcript", ["Ne yok, uyan.", "Ne o ya?"])
def test_real_attempts_from_the_live_log_are_accepted(transcript):
    """What the recognizer produced when the user genuinely said the phrase.

    An earlier threshold of 0.85, derived from tidy strings rather than real
    transcripts, would have rejected both -- i.e. would have stopped NEO
    waking up at all on this microphone.
    """
    assert matches_wake_phrase(transcript, PHRASE) is True


def test_the_phrase_itself_is_the_weak_link():
    """Documents why no threshold fixes this: 'Ne o ya?' (filler) and
    'Ne yok, uyan.' (a real attempt) score within 0.04 of each other, so the
    separation has to come from a more distinctive wake phrase."""
    filler = similarity("Ne o ya?", PHRASE)
    genuine = similarity("Ne yok, uyan.", PHRASE)

    assert abs(genuine - filler) < 0.1

    # A distinctive phrase does separate.
    better = "Neo devrede"
    assert similarity("Neo devrede", better) > 0.9
    assert similarity("Ne o ya?", better) < 0.6
    assert similarity("Ne oluyor?", better) < 0.6


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
