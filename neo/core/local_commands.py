from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

from ..tools.base import ToolRegistry, ToolResult

ResponseFormatter = Callable[[ToolResult], str]


def _fmt_time(result: ToolResult) -> str:
    if not result.success:
        return result.error or "Saat bilgisine erişemiyorum."
    return f"Saat {result.data['time']}."


def _fmt_date(result: ToolResult) -> str:
    if not result.success:
        return result.error or "Tarih bilgisine erişemiyorum."
    d = result.data
    return f"Bugün {d['day']} {d['month']} {d['year']}, {d['weekday']}."


def _fmt_cpu(result: ToolResult) -> str:
    if not result.success:
        return result.error or "İşlemci kullanımına erişemiyorum."
    return f"İşlemci şu anda yüzde {result.data['cpu_percent']:.0f} kullanılıyor."


def _fmt_ram(result: ToolResult) -> str:
    if not result.success:
        return result.error or "RAM kullanımına erişemiyorum."
    d = result.data
    return f"RAM'in yüzde {d['percent']:.0f}'i kullanılıyor ({d['used_gb']} / {d['total_gb']} GB)."


def _fmt_gpu(result: ToolResult) -> str:
    if not result.success:
        return result.error or "GPU verisine erişemiyorum."
    d = result.data
    parts = [f"{d['name']} şu anda yüzde {d['gpu_percent']} kullanılıyor"]
    if "temperature_c" in d:
        parts.append(f"sıcaklığı {d['temperature_c']} derece")
    parts.append(f"VRAM {d['vram_used_mb']}/{d['vram_total_mb']} MB")
    return ", ".join(parts) + "."


def _fmt_disk(result: ToolResult) -> str:
    if not result.success:
        return result.error or "Disk bilgisine erişemiyorum."
    lines = [
        f"{d['drive']} {d['free_gb']} GB boş / {d['total_gb']} GB (%{d['percent_used']:.0f} dolu)"
        for d in result.data["drives"]
    ]
    return "; ".join(lines) + "."


def _fmt_system_info(result: ToolResult) -> str:
    if not result.success:
        return result.error or "Sistem bilgisine erişemiyorum."
    d = result.data
    return (
        f"{d['computer_name']} - {d['os']}, {d['cpu']} "
        f"({d['cpu_cores_physical']} çekirdek / {d['cpu_cores_logical']} iş parçacığı), "
        f"{d['ram_total_gb']} GB RAM."
    )


def _fmt_network(result: ToolResult) -> str:
    if not result.success:
        return result.error or "Ağ durumuna erişemiyorum."
    return "İnternet bağlantım var." if result.data["connected"] else "İnternete bağlı değilim."


def _fmt_weather(result: ToolResult) -> str:
    if not result.success:
        return result.error or "Hava durumuna erişemiyorum."
    d = result.data
    msg = f"{d['city']}: {d['condition']}, {d['temperature_c']}°C"
    today = d.get("today")
    if today:
        msg += (
            f" (bugün {today['min_c']}-{today['max_c']}°C, "
            f"yağış ihtimali %{today['rain_chance_percent']})"
        )
    return msg + "."


@dataclass
class LocalCommand:
    pattern: re.Pattern
    tool_name: str
    formatter: ResponseFormatter
    exclude: re.Pattern | None = None


# Very common, unambiguous system queries answered directly from local tools --
# no LLM round-trip, so they keep working even with no internet or no API
# credit. Anything that doesn't match one of these still goes through the
# full Agent/LLM tool-use pipeline.
#
# Turkish attaches case suffixes directly onto the word ("disk" + "te" ->
# "diskte", no apostrophe), so plain Turkish nouns below only anchor the
# left edge (\bword, not \bword\b) or "diskte"/"saatte" etc. would never
# match. The short English acronyms (ram/cpu/gpu) keep a trailing boundary
# since they're commonly left undeclined in speech ("ram ne kadar") and a
# bare prefix match on "ram" would otherwise also fire on unrelated words
# like "Ramazan".
_COMMANDS: tuple[LocalCommand, ...] = (
    LocalCommand(re.compile(r"\bsaat"), "get_time", _fmt_time),
    # "günlerden" is spelled out separately: the `\bgün\b` branch below needs
    # a word boundary, so "bugün günlerden ne?" -- a very ordinary way to ask
    # -- fell through to the LLM and cost a call to answer the date.
    LocalCommand(
        re.compile(r"\btarih|hangi g[üu]nde|g[üu]nlerden|bug[üu]n.*\bg[üu]n\b"),
        "get_date",
        _fmt_date,
    ),
    LocalCommand(re.compile(r"\bcpu\b|\bi[şs]lemci"), "get_cpu_usage", _fmt_cpu),
    LocalCommand(re.compile(r"\bram\b|\bbellek"), "get_ram_usage", _fmt_ram),
    LocalCommand(re.compile(r"\bgpu\b|ekran kart"), "get_gpu_status", _fmt_gpu),
    LocalCommand(re.compile(r"\bdisk"), "get_disk_usage", _fmt_disk),
    LocalCommand(
        re.compile(r"bilgisayar.{0,15}[öo]zellik|sistem bilgi"), "get_system_info", _fmt_system_info
    ),
    LocalCommand(re.compile(r"internet.{0,15}ba[ğg]lant|a[ğg] durumu"), "get_network_status", _fmt_network),
    # Only the bare "hava nasıl/durumu" question (no named city) is safe to
    # answer locally with the default city -- a specific city ("Bursa'da
    # hava nasıl?") needs the LLM path so the city argument is actually
    # parsed and used, instead of silently answering for the wrong place.
    LocalCommand(
        re.compile(r"hava\s*durum|hava\s+nas[ıi]l"),
        "get_weather",
        _fmt_weather,
        exclude=re.compile(r"['’](da|de|ta|te|nda|nde)\b"),
    ),
)


# A request that asks NEO to *do* something (take a note, open an app,
# search, remind) must never be swallowed by the status-query fast path.
# "Yarın saat 15'te randevum var, takvime not al" contains "saat", which
# used to hijack the whole sentence into a "what time is it?" answer.
#
# Listing individual phrasings kept letting new ones through: "yarın şu saate
# program ekle" matched none of "not al"/"nota ekle", fell through to the
# status path, and got answered with the current time. So this now covers the
# action *verbs* broadly plus the scheduling nouns, rather than a handful of
# specific phrases.
_ACTION_INTENT_PATTERN = re.compile(
    # scheduling / note nouns
    r"takvim|randevu|program[ıi]?m?a|toplant[ıi]|etkinlik|hat[ıi]rlatma|g[öo]rev|"
    # imperative verbs that mean "do something", not "tell me something"
    r"\bekle(r misin|sene|yiver)?\b|\bnot al|kaydet|kayded|\byaz\b|olu[şs]tur|"
    r"hat[ıi]rlat|planla|ayarla|kur\b|g[üu]ncelle|de[ğg]i[şs]tir|ta[şs][ıi]|"
    r"\bsil\b|iptal|\bkald[ıi]r\b|"
    r"\ba[çc]\b|\ba[çc]ar m[ıi]s[ıi]n\b|\ba[çc]sana\b|ba[şs]lat|"
    r"ara[şs]t[ıi]r|\bara\b|g[öo]nder"
)


# The fast path answers *status questions*, and real ones are short: "saat
# kaç", "hava nasıl", "cpu ne durumda", "bugün günlerden ne". Anything longer
# is a sentence with context the LLM needs to read.
#
# This backstops the keyword guards rather than replacing them. Those guards
# are spelling-sensitive by nature, and a single typo defeated one live:
# "Takbimimde sadece saat 13.00'daki buluşma gözüküyor diğerleri neden
# gözükmüyor" missed the "takvim" guard because of the b/v slip, so the
# stray "saat" hijacked a nine-word question about the calendar and NEO
# answered with the current time. Length doesn't care how a word is spelled.
_MAX_FAST_PATH_WORDS = 6


def match_local_command(text: str) -> LocalCommand | None:
    lowered = text.lower()
    if len(lowered.split()) > _MAX_FAST_PATH_WORDS:
        return None
    if _ACTION_INTENT_PATTERN.search(lowered):
        return None
    for command in _COMMANDS:
        if command.pattern.search(lowered) and not (command.exclude and command.exclude.search(lowered)):
            return command
    return None


# Small talk / meta questions with a fixed, personality-flavoured answer --
# these have no "real" data to fetch, so there's nothing an LLM call would
# add except cost and a delay; answered directly so they work even with no
# API credit. Each pattern has a few phrasings, picked at random, so it
# doesn't feel like a scripted bot repeating the exact same line every time.
_STATIC_REPLIES: tuple[tuple[re.Pattern, tuple[str, ...]], ...] = (
    (
        re.compile(r"nas[ıi]ls[ıi]n|\bnaber\b|ne haber|n['’]?abersin"),
        (
            "İyiyim, teşekkürler! Sen nasılsın?",
            "Gayet iyiyim. Bir şeye mi ihtiyacın var, yoksa sohbet mi edelim?",
            "İyilik sağlık. Sen naber, her şey yolunda mı?",
        ),
    ),
    (
        re.compile(r"neler yapabilirsin|nelere yard[ıi]mc[ıi] olabilirsin|ne i[şs]e yarars[ıi]n"),
        (
            "Saat ve tarih söylerim; bilgisayarının CPU, RAM, GPU ve disk "
            "durumunu okurum; hava durumuna bakarım; uygulama ve web sitesi "
            "açarım; takvimine not alır, günaydın dediğinde günün notlarını "
            "hatırlatırım. Daha genel sohbet, araştırma ve karmaşık "
            "istekler için Claude ile konuşup cevap üretirim.",
        ),
    ),
    (
        re.compile(r"\bsen kimsin\b|ad[ıi]n ne|\bkim bu\b"),
        ("Ben NEO, bu bilgisayarda çalışan kişisel yapay zekâ asistanınım.",),
    ),
)


_GOOD_MORNING_PATTERN = re.compile(r"g[üu]nayd[ıi]n")
_MORNING_GREETINGS = ("Günaydın!", "Günaydın, umarım iyi uyumuşsundur!", "Günaydın, güne başlayalım.")

# Only "bugün"/"yarın" free-time checks are resolved locally (trivial date
# math); anything referencing another day ("cuma", "gelecek hafta") needs the
# LLM path, which can resolve arbitrary date references using the current
# date given in its system prompt.
_FREE_CHECK_PATTERN = re.compile(r"\b(bug[üu]n|yar[ıi]n)\b.{0,25}\b(bo[şs]|dolu|program)")


async def _handle_good_morning(registry: ToolRegistry) -> str:
    greeting = random.choice(_MORNING_GREETINGS)
    result = await registry.execute("get_calendar_notes", {"date": date.today().isoformat()})
    if not result.success:
        return greeting
    notes = result.data.get("notes") or []
    if not notes:
        return f"{greeting} Bugün için not aldığın bir şey yok, programın boş görünüyor."
    return f"{greeting} Bugün için not aldıkların: {'; '.join(notes)}."


# "Boş muyum?" is a yes/no question; "programımda neler var?" is not. Both
# reach this handler, and answering the second one with "Hayır, ..." reads as
# a non sequitur -- as seen live on "Yarın programında neler var?".
_YES_NO_PATTERN = re.compile(r"\b(bo[şs]|dolu|m[üu]sait)\b")


async def _handle_free_check(registry: ToolRegistry, match: re.Match, text: str) -> str:
    if match.group(1).startswith("bug"):
        target_date, label = date.today().isoformat(), "bugün"
    else:
        target_date, label = (date.today() + timedelta(days=1)).isoformat(), "yarın"

    result = await registry.execute("get_calendar_notes", {"date": target_date})
    if not result.success:
        return result.error or "Takvim bilgisine erişemiyorum."

    notes = result.data.get("notes") or []
    yes_no = bool(_YES_NO_PATTERN.search(text.lower()))

    if not notes:
        if yes_no:
            return f"Evet, {label} programın boş görünüyor."
        return f"{label.capitalize()} için not aldığın bir şey yok."
    listed = "; ".join(notes)
    if yes_no:
        return f"Hayır, {label} için şunlar var: {listed}."
    return f"{label.capitalize()} için şunlar var: {listed}."


# Commands aimed at NEO itself rather than at data. Checked before the action
# guard, since "durdur"/"kapat" are exactly the kind of verbs that guard is
# meant to catch -- here they're the whole point, not a false positive.
STOP_LISTENING = "stop_listening"
START_LISTENING = "start_listening"

_CONTROL_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (
        re.compile(
            r"(dinleme(?:yi|ni|mi)?|mikrofonu?|s[üu]rekli dinleme)\s*\S*\s*"
            r"\b(durdur|kapat|kes|b[ıi]rak|bitir)\b"
            r"|\bbeni dinleme\b"
            r"|\buyku moduna ge[çc]"
        ),
        STOP_LISTENING,
    ),
    (
        re.compile(
            r"(dinleme(?:ye|yi)?|mikrofonu?|s[üu]rekli dinleme)\s*\S*\s*"
            r"\b(ba[şs]la|a[çc]|devam)\b"
            r"|\bbeni dinle\b"
        ),
        START_LISTENING,
    ),
)


def match_control_command(text: str) -> str | None:
    lowered = text.lower()
    for pattern, action in _CONTROL_PATTERNS:
        if pattern.search(lowered):
            return action
    return None


async def try_handle_locally(text: str, registry: ToolRegistry) -> str | None:
    """Returns a formatted reply if `text` matches a known local command or
    small-talk pattern, otherwise None (caller should fall back to the
    LLM-driven Agent)."""
    lowered = text.lower()

    if _GOOD_MORNING_PATTERN.search(lowered):
        return await _handle_good_morning(registry)

    free_check_match = _FREE_CHECK_PATTERN.search(lowered)
    if free_check_match:
        return await _handle_free_check(registry, free_check_match, text)

    for pattern, replies in _STATIC_REPLIES:
        if pattern.search(lowered):
            return random.choice(replies)

    command = match_local_command(text)
    if command is None:
        return None
    result = await registry.execute(command.tool_name, {})
    return command.formatter(result)
