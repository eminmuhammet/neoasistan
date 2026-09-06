import asyncio
from datetime import date, timedelta

from neo.core.local_commands import match_local_command, try_handle_locally
from neo.memory.calendar_store import CalendarStore
from neo.tools.base import ToolRegistry
from neo.tools.calendar import GetCalendarNotesTool
from neo.tools.time_tools import GetDateTool, GetTimeTool


def test_match_local_command_for_time():
    command = match_local_command("saat kaç")
    assert command is not None
    assert command.tool_name == "get_time"


def test_match_local_command_for_date():
    command = match_local_command("bugün ayın kaçı, tarih ne")
    assert command is not None
    assert command.tool_name == "get_date"


def test_no_match_for_unrelated_chat():
    assert match_local_command("yapay zeka hakkında ne düşünüyorsun") is None


def test_matches_turkish_suffixed_forms():
    # Turkish glues case suffixes onto the noun with no apostrophe or space
    # ("disk" + "te" -> "diskte"), unlike the acronyms (ram/cpu/gpu).
    assert match_local_command("diskte ne kadar yer var").tool_name == "get_disk_usage"
    assert match_local_command("saatte kaç dendi").tool_name == "get_time"


def test_ram_does_not_false_positive_on_unrelated_word():
    # A bare prefix match on "ram" would otherwise also fire on "Ramazan".
    assert match_local_command("ramazan ne zaman") is None


def test_try_handle_locally_returns_formatted_time_without_registry_lookup_failure():
    registry = ToolRegistry()
    registry.register(GetTimeTool())

    reply = asyncio.run(try_handle_locally("saat kaç", registry))

    assert reply is not None
    assert reply.startswith("Saat ")


def test_try_handle_locally_returns_none_when_nothing_matches():
    registry = ToolRegistry()
    reply = asyncio.run(try_handle_locally("bugün moralim bozuk", registry))
    assert reply is None


def test_try_handle_locally_reports_missing_tool_gracefully():
    registry = ToolRegistry()  # get_date intentionally not registered

    reply = asyncio.run(try_handle_locally("tarih ne", registry))

    assert reply is not None
    assert "get_date" in reply


def test_date_formatter_uses_real_tool_output():
    registry = ToolRegistry()
    registry.register(GetDateTool())

    reply = asyncio.run(try_handle_locally("hangi gündeyiz", registry))

    assert reply is not None
    assert reply.startswith("Bugün ")


def test_static_reply_for_how_are_you_needs_no_registry_or_llm():
    from neo.core.local_commands import _STATIC_REPLIES

    registry = ToolRegistry()  # nothing registered -- must not be needed
    how_are_you_replies = _STATIC_REPLIES[0][1]

    reply = asyncio.run(try_handle_locally("naber, nasılsın", registry))

    assert reply in how_are_you_replies


def test_static_reply_for_capabilities_question():
    registry = ToolRegistry()
    reply = asyncio.run(try_handle_locally("neler yapabilirsin", registry))
    assert reply is not None
    assert "CPU" in reply


def test_static_reply_takes_priority_over_llm_but_not_over_tool_commands():
    registry = ToolRegistry()
    registry.register(GetTimeTool())
    # "saat kaç" must still resolve to the time tool, not a static reply.
    reply = asyncio.run(try_handle_locally("saat kaç", registry))
    assert reply is not None
    assert reply.startswith("Saat ")


def test_action_requests_are_not_hijacked_by_the_status_fast_path():
    # "yarın saat 15'te ... not al" used to match the time shortcut because
    # of the word "saat" and answer with the clock instead of taking a note.
    assert match_local_command("yarın saat 15te diş hekimi randevum var, takvime not al") is None
    assert match_local_command("saat 9 için toplantı hatırlat") is None
    assert match_local_command("chrome'u aç") is None


def test_plain_status_questions_still_match():
    assert match_local_command("saat kaç").tool_name == "get_time"
    assert match_local_command("cpu yüzde kaç").tool_name == "get_cpu_usage"


def test_bare_weather_question_matches_local_command():
    command = match_local_command("bugün hava nasıl")
    assert command is not None
    assert command.tool_name == "get_weather"


def _registry_with_calendar(tmp_path):
    store = CalendarStore(tmp_path / "calendar.db")
    registry = ToolRegistry()
    registry.register(GetCalendarNotesTool(store))
    return registry, store


def test_good_morning_with_no_notes(tmp_path):
    registry, _ = _registry_with_calendar(tmp_path)
    reply = asyncio.run(try_handle_locally("günaydın", registry))
    assert reply is not None
    assert "boş" in reply.lower()


def test_good_morning_reads_todays_notes(tmp_path):
    registry, store = _registry_with_calendar(tmp_path)
    store.add_note(date.today().isoformat(), "Diş hekimi 14:00")

    reply = asyncio.run(try_handle_locally("günaydın", registry))

    assert reply is not None
    assert "Diş hekimi 14:00" in reply


def test_free_check_today_when_empty(tmp_path):
    registry, _ = _registry_with_calendar(tmp_path)
    reply = asyncio.run(try_handle_locally("bugün programım boş mu", registry))
    assert reply is not None
    assert reply.lower().startswith("evet")


def test_free_check_today_when_busy(tmp_path):
    registry, store = _registry_with_calendar(tmp_path)
    store.add_note(date.today().isoformat(), "Toplantı 10:00")

    reply = asyncio.run(try_handle_locally("bugün boş muyum", registry))

    assert reply is not None
    assert reply.lower().startswith("hayır")
    assert "Toplantı 10:00" in reply


def test_free_check_tomorrow_uses_tomorrows_date(tmp_path):
    registry, store = _registry_with_calendar(tmp_path)
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    store.add_note(tomorrow, "Seyahat")

    reply = asyncio.run(try_handle_locally("yarın dolu mu", registry))

    assert reply is not None
    assert "Seyahat" in reply


def test_weather_question_with_named_city_is_excluded_from_local_shortcut():
    # Must go through the LLM path so the city argument is actually parsed,
    # instead of silently answering for the wrong place.
    assert match_local_command("Bursa'da hava nasıl") is None
    assert match_local_command("İstanbul'da hava durumu ne") is None
