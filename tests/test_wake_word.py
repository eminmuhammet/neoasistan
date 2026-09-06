from neo.voice.wake_word import _should_finalize_command


def test_finalizes_on_long_pause_after_enough_audio():
    assert (
        _should_finalize_command(
            command_buffer_len=5000,
            silence_run=30000,
            silence_gap_samples=24000,
            step_samples=1600,
            timed_out=False,
            heard_speech=True,
        )
        is True
    )


def test_does_not_finalize_before_pause_or_timeout():
    assert (
        _should_finalize_command(
            command_buffer_len=5000,
            silence_run=1000,
            silence_gap_samples=24000,
            step_samples=1600,
            timed_out=False,
            heard_speech=True,
        )
        is False
    )


def test_finalizes_on_timeout_even_without_a_pause():
    assert (
        _should_finalize_command(
            command_buffer_len=5000,
            silence_run=0,
            silence_gap_samples=24000,
            step_samples=1600,
            timed_out=True,
            heard_speech=False,
        )
        is True
    )


def test_does_not_finalize_on_pause_alone_if_buffer_too_short():
    assert (
        _should_finalize_command(
            command_buffer_len=100,
            silence_run=30000,
            silence_gap_samples=24000,
            step_samples=1600,
            timed_out=False,
            heard_speech=True,
        )
        is False
    )


def test_does_not_finalize_on_leading_silence_before_speech_starts():
    # The user said "Neo" and hasn't started their actual command yet --
    # this pause must NOT be treated as "finished speaking", or every
    # natural hesitation forces the user to repeat the wake word.
    assert (
        _should_finalize_command(
            command_buffer_len=30000,
            silence_run=30000,
            silence_gap_samples=24000,
            step_samples=1600,
            timed_out=False,
            heard_speech=False,
        )
        is False
    )


def test_finalizes_on_trailing_silence_after_speech_heard():
    assert (
        _should_finalize_command(
            command_buffer_len=30000,
            silence_run=24000,
            silence_gap_samples=24000,
            step_samples=1600,
            timed_out=False,
            heard_speech=True,
        )
        is True
    )
