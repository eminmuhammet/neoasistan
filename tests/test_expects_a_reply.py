"""_expects_a_reply decides when a follow-up listening window opens.

Requested explicitly: not after every reply NEO gives, only when NEO asked
something and a response is the expected next thing -- "her konuşma
bittiğinde değil, benden cevap beklediğinde".
"""

from neo.ui.main_window import _expects_a_reply


def test_a_question_expects_a_reply():
    assert _expects_a_reply("Hangi şehir için bakayım?") is True


def test_a_plain_statement_does_not():
    assert _expects_a_reply("İşte hava durumu: bugün 21 derece, açık.") is False


def test_a_question_behind_trailing_quotes_still_counts():
    assert _expects_a_reply('Devam edeyim mi?"') is True
    assert _expects_a_reply("Emin misin?»") is True


def test_a_multi_sentence_reply_only_looks_at_the_end():
    text = "Toplantın saat 14:00'te. Hatırlatıcı da kurayım mı?"
    assert _expects_a_reply(text) is True

    text2 = "Toplantın saat 14:00'te. Not aldım."
    assert _expects_a_reply(text2) is False


def test_trailing_whitespace_is_ignored():
    assert _expects_a_reply("Devam edeyim mi?  \n") is True


def test_empty_text_does_not_expect_a_reply():
    assert _expects_a_reply("") is False
