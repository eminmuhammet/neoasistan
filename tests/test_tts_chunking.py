"""Sentences are spoken as they're ready, not after the whole reply has been
synthesized -- a long reply used to be several seconds of silence while the
entire text round-tripped through the network before anything played."""

from neo.voice.tts import split_for_speech


def test_short_reply_is_not_split():
    """Splitting "Saat 14.30." would only add an audible seam for no
    benefit -- there is nothing to overlap synthesis with."""
    assert split_for_speech("Saat 14.30.") == ["Saat 14.30."]


def test_empty_text_produces_no_chunks():
    assert split_for_speech("") == []
    assert split_for_speech("   ") == []


def test_long_reply_is_split_on_sentence_boundaries():
    text = (
        "Bugün hava güneşli ve sıcaklık 24 derece. Yarın için takviminde bir "
        "toplantın var, saat 16.00 da. Ayrıca CPU kullanımın şu anda yüzde "
        "12 civarında görünüyor."
    )
    chunks = split_for_speech(text)

    assert len(chunks) > 1
    # Reassembling the chunks (each one is spoken on its own, so a space is
    # the right join) must not lose or duplicate any sentence.
    assert " ".join(chunks) == text
    for chunk in chunks:
        assert chunk == chunk.strip()


def test_first_chunk_is_available_without_the_rest():
    """The whole point: the first sentence must stand on its own so it can
    start playing before later sentences exist."""
    text = "Kısa bir cümle. " + "Uzun bir devam cümlesi burada geliyor. " * 4
    chunks = split_for_speech(text)

    assert len(chunks) >= 2
    assert chunks[0].strip().startswith("Kısa bir cümle")


def test_short_trailing_fragment_is_not_spoken_alone():
    """A short tail split off on its own would land as an odd clipped
    fragment; it's appended to the previous chunk instead."""
    text = "Bu ilk cümle oldukça uzun ve tek başına bir parça oluşturacak kadar geniş. Son."
    chunks = split_for_speech(text)

    assert chunks[-1].endswith("Son.")
    assert not any(chunk.strip() == "Son." for chunk in chunks)


def test_chunks_never_start_or_end_with_whitespace():
    text = "Birinci cümle burada. İkinci cümle burada da devam ediyor gayet uzun bir şekilde."
    for chunk in split_for_speech(text):
        assert chunk == chunk.strip()
        assert chunk != ""
