import pytest

from neo.core.context import ConversationContext
from neo.core.llm_client import _cacheable_system, _cacheable_tools
from neo.core.local_commands import (
    START_LISTENING,
    STOP_LISTENING,
    match_control_command,
    match_local_command,
)
from neo.voice.tts import strip_speech_noise


# -- prompt caching --------------------------------------------------------
#
# Measured before this was added: the system prompt (1,289 tokens) and 18 tool
# schemas (2,712 tokens) were resent at full price on every call -- 4,001
# fixed tokens, or $1.20 per 100 messages before the user had said anything.


def test_system_prompt_is_marked_cacheable():
    blocks = _cacheable_system("merhaba")
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert blocks[0]["text"] == "merhaba"


def test_last_local_tool_carries_the_cache_breakpoint():
    tools = [{"name": "a"}, {"name": "b"}, {"type": "web_search_20250305", "name": "web_search"}]
    result = _cacheable_tools(tools)

    assert "cache_control" not in result[0]
    assert result[1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # Server tools stay after the breakpoint so their config can't invalidate
    # the expensive part of the cached prefix.
    assert result[-1]["name"] == "web_search"
    assert "cache_control" not in result[-1]


def test_cacheable_tools_handles_empty_and_server_only():
    assert _cacheable_tools([]) == []
    server_only = [{"type": "web_search_20250305", "name": "web_search"}]
    assert _cacheable_tools(server_only) == server_only


def test_cacheable_tools_does_not_mutate_the_caller_list():
    tools = [{"name": "a"}]
    _cacheable_tools(tools)
    assert "cache_control" not in tools[0]


def test_cache_uses_the_one_hour_window():
    """With the 5-minute default, NEO's usage pattern -- short bursts spread
    across a day -- would land most first-messages after expiry, paying the
    write premium instead of the read discount. That is worse than not
    caching at all."""
    from neo.core.llm_client import _CACHE_CONTROL

    assert _CACHE_CONTROL["ttl"] == "1h"


# -- history trimming ------------------------------------------------------


def test_trim_never_leaves_an_orphaned_tool_result():
    """The API rejects a tool_result whose tool_use was trimmed away, so a
    plain slice could turn a long conversation into a hard error the moment
    the window fell between a tool call and its result."""
    context = ConversationContext(max_messages=4)
    for i in range(3):
        context.add_user(f"soru {i}")
        context.add_assistant([{"type": "tool_use", "id": f"t{i}", "name": "x", "input": {}}])
        context.add_tool_result(f"t{i}", "{}")

    assert len(context.messages) <= 4
    first = context.messages[0]
    assert not ConversationContext._is_tool_result(first)


def test_trim_keeps_the_most_recent_exchanges():
    context = ConversationContext(max_messages=4)
    for i in range(6):
        context.add_user(f"soru {i}")

    assert context.messages[-1]["content"] == "soru 5"
    assert len(context.messages) == 4


# -- voice control of listening -------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "dinlemeyi durdur",
        "Dinlemeyi kapat",
        "dinlemeni kes",
        "mikrofonu kapat",
        "sürekli dinlemeyi durdur",
        "beni dinleme",
        "uyku moduna geç",
    ],
)
def test_stop_listening_is_recognized(text):
    assert match_control_command(text) == STOP_LISTENING


@pytest.mark.parametrize(
    "text",
    ["dinlemeye başla", "dinlemeyi aç", "mikrofonu aç", "beni dinle"],
)
def test_start_listening_is_recognized(text):
    assert match_control_command(text) == START_LISTENING


@pytest.mark.parametrize(
    "text",
    ["saat kaç", "hava durumu nasıl", "bugün programımda neler var", "nasılsın"],
)
def test_ordinary_requests_are_not_control_commands(text):
    assert match_control_command(text) is None


# -- action requests must not be answered as status queries ---------------


@pytest.mark.parametrize(
    "text",
    [
        "yarın şu saate program ekle",
        "yarın saat 15'te program ekle",
        "yarına bir toplantı ekle",
        "saat 14:00'e randevu kaydet",
        "yarın saat 9 için hatırlatma oluştur",
        "cuma saat 10'a etkinlik ekler misin",
        "bugünkü saat 15 notunu sil",
    ],
)
def test_scheduling_requests_do_not_hit_the_time_shortcut(text):
    """Observed live: "yarın şu saate program ekle" contained "saat", matched
    none of the specific action phrases the guard listed, and was answered
    with the current time instead of creating the note."""
    assert match_local_command(text) is None


@pytest.mark.parametrize(
    "text",
    ["saat kaç", "saat kaç oldu", "bugün günlerden ne", "cpu kullanımı ne durumda"],
)
def test_genuine_status_questions_still_use_the_fast_path(text):
    assert match_local_command(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "Yarın 16.00'a bir program ekle",
        "bugün programıma toplantı ekle",
        "yarın için bir hatırlatma oluştur",
        "günaydın, yarına diş hekimi randevusu ekle",
    ],
)
def test_action_requests_bypass_every_fast_path_branch(text):
    """The guard used to live only in match_local_command, which runs last.
    "Yarın 16.00'a bir program ekle" matched the free-time pattern ("yarın"
    ... "program") in an earlier branch and came back "Yarın için not aldığın
    bir şey yok" -- the note was silently never created."""
    import asyncio

    from neo.core.local_commands import try_handle_locally

    assert asyncio.run(try_handle_locally(text, registry=None)) is None


@pytest.mark.parametrize(
    "text",
    [
        "Takbimimde sadece saat 13.00'daki buluşma gözüküyor diğerleri neden gözükmüyor",
        "saat 15 civarı müsait miyim yoksa dolu muyum acaba",
        "dün akşam saat kaçta uyuduğumu hatırlıyor musun",
    ],
)
def test_long_sentences_never_hit_the_status_shortcut(text):
    """A keyword guard is only as good as the user's spelling: the first of
    these missed the "takvim" guard because of a b/v typo, so the stray
    "saat" turned a nine-word question about the calendar into "Saat
    22:17:38." Length is a guard that typos can't slip past."""
    assert match_local_command(text) is None


# -- spoken output ---------------------------------------------------------


def test_emoji_are_not_spoken():
    """Edge TTS reads ✅ aloud as its Unicode name, which the user hears as a
    stray phrase in the middle of a sentence."""
    assert strip_speech_noise("Not eklendi ✅") == "Not eklendi"
    assert strip_speech_noise("🔬 Araştırma modu açık") == "Araştırma modu açık"
    assert strip_speech_noise("Tamam 👍🏽, hallettim") == "Tamam , hallettim"


def test_stripping_leaves_ordinary_turkish_untouched():
    text = "Yarın saat 15:00'te diş hekimi randevun var, unutma!"
    assert strip_speech_noise(text) == text


def test_stripping_preserves_punctuation_and_numbers():
    assert strip_speech_noise("Sıcaklık 24°C, nem %60") == "Sıcaklık 24°C, nem %60"
