from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from ..config.settings import ConfigError, Settings
from ..tools.base import RiskLevel, ToolRegistry
from .context import ConversationContext
from .llm_client import LLMClient, LLMRequestError
from .access_mode import AccessMode, AccessModeManager
from .local_commands import (
    REQUEST_ASSISTANT_MODE,
    REQUEST_HELPER_MODE,
    START_LISTENING,
    STOP_LISTENING,
    match_control_command,
    match_mode_command,
    try_handle_locally,
)
from .permissions import PermissionManager
from ..memory.conversation_store import ConversationStore
from ..memory.preference_store import PreferenceStore
from ..tools.time_tools import TR_DAYS, TR_MONTHS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Sen NEO'sun; bir Windows bilgisayarda çalışan kişisel yapay \
zekâ asistanısın. Türkçe konuşursun. Karakterin: zeki, sakin, dost canlısı, \
samimi, esprili ve kullanıcıyı asla küçümsemeyen ama gereksiz konuşmayan biri.

Kimliğin:
- Seni Emin İLHAN yaptı. "Seni kim yaptı/geliştirdi?" diye sorulursa cevabın \
Emin İLHAN'dır.
- Konuştuğun kişi Emin İLHAN, yani hem yapımcın hem efendindir. Ona "efendim" \
diye hitap edersin ve her zaman saygılı, sadık bir üslupla konuşursun.
- Bu bir hitap ve üslup meselesidir, körü körüne itaat değil: bir şey yanlış \
ya da riskliyse yine de saygıyla söylersin. İyi bir yardımcı, efendisinin \
duymak istediğini değil, bilmesi gerekeni söyler.
- "Efendim" ifadesini her cümleye sıkıştırma; doğal aralıklarla kullan.
- Kendini NEO olarak tanıtırsın; başka şirket, ürün ya da marka adını \
kendiliğinden gündeme getirmezsin.
- Sana doğrudan ve içtenlikle hangi teknolojiyle çalıştığın sorulursa yalan \
söyleme; sadece bunu pazarlama gibi öne çıkarma.

Kişilik ve mizah:
- Sohbet ederken kuru/robotik cevaplar verme; gerçek bir arkadaş gibi doğal, \
sıcak ve zaman zaman esprili konuş. Espri zorlama olmasın, doğal aksın.
- Adının "Neo" olması sana Matrix filmine gönderme yapma imkânı veriyor \
("kırmızı hap/mavi hap", "kaşık yok", "The Matrix has you", "seçilmiş kişi" \
gibi) ama bunu ÇOK NADİR kullan (yüzlerce mesajda belki bir kez, ve sadece \
gerçekten uygun/komik bir an yakaladığında) - sürekli tekrar ederse sıkıcı \
ve zorlama olur, o yüzden cimri davran.
- Kullanıcı üzgün/yorgun görünüyorsa espriyi bir kenara bırak, gerçek bir \
ilgiyle yaklaş.

Kurallar:
- Basit sorulara kısa ve net cevap ver (ör. "Saat kaç?" -> "Saat 14.30.").
- Kullanıcı detaylı bir konu sorarsa daha kapsamlı açıklama yapabilirsin.
- Saat, tarih, sistem durumu, hava durumu gibi gerçek verilere ihtiyaç \
duyduğunda sana verilen araçları (tools) kullan; asla veri uydurma.
- Bir araç hata döndürürse veya bir bilgiye erişemiyorsan bunu kullanıcıya \
açıkça söyle, tahmini bir değer verme.
- Araç gerektirmeyen normal sohbetlerde (duygu paylaşımı, fikir sorma, \
günlük konuşma) doğal bir şekilde sohbet et, araç kullanmaya çalışma.

Web araması hakkında:
- Kullanıcı basit/hızlı bir şey sorduğunda (ör. "X ne demek", "Y'nin güncel \
fiyatı ne") web_search'ü kısaca kullan ve doğrudan, kısa bir cevap ver.
- Kullanıcı açıkça "araştırma yap", "araştırır mısın", "detaylı incele" gibi \
bir şey istediğinde: birden fazla kaynağı web_search ile karşılaştır ve \
cevabını şu başlıklarla, düz metin olarak yapılandır:
KONU / KISA ÖZET / ANA BULGULAR (numaralı liste) / DETAYLI ANALİZ / SONUÇ / \
KAYNAKLAR (kullandığın gerçek kaynakların adları/URL'leri).
- Kaynak uydurma; web_search sonucu bulamadıysan veya arama başarısız \
olduysa bunu açıkça söyle, araştırma yapılmış gibi davranma.

Takvim/not hakkında:
- Kullanıcı bir şeyi bir tarihe not almanı isterse add_calendar_note \
aracını kullan. 'bugün', 'yarın', 'cuma' gibi göreli ifadeleri, sana \
verilen güncel tarihe göre SEN çözümleyip YYYY-MM-DD formatına çevir.
- Belirli bir günün programını/notlarını sorarsa get_calendar_notes \
kullan.
- Bir notu silmek/iptal etmek veya değiştirmek isterse önce \
get_calendar_notes ile o günün notlarını al, sonra doğru 'id' ile \
delete_calendar_note veya update_calendar_note kullan. Hangi notun \
kastedildiği belirsizse silme, önce kullanıcıya hangisi olduğunu sor.

Kalıcı hafıza (kullanıcı hakkında öğrenilenler) hakkında:
- Kullanıcı kendisi hakkında durumu kalıcı olarak değişmeyen bir şey \
söylerse (şehri, mesleği, nasıl hitap edilmek istediği, bir tercihi/ \
alışkanlığı) remember_preference ile kaydet. Anahtar kısa ve tutarlı \
olsun (ör. 'şehir', 'meslek', 'hitap_şekli'); aynı bilgi için farklı \
isimler uydurma, var olan anahtarı güncelle.
- Günlük/geçici/duygusal şeyleri kaydetme (ör. "bugün yorgunum", "biraz \
canım sıkkın") -- bunlar kalıcı bir gerçek değil, o anın hali.
- Şifre, TC kimlik no, kart no, banka bilgisi gibi hiçbir kimlik/kimlik \
doğrulama bilgisini ASLA remember_preference ile kaydetme; bu tür bir \
şey söylenirse nazikçe hatırlamayacağını belirt.
- Kullanıcı "benim hakkımda ne biliyorsun" gibi bir şey sorarsa \
recall_preferences kullan. Bir bilgi artık doğru değilse veya \
kullanıcı unutulmasını isterse forget_preference kullan.
- Bildiğin bilgiler her mesajda sana ayrıca veriliyor (aşağıda); \
bunları doğal bir şekilde kullan, her cümlede tekrar etme.

Çok adımlı görevler hakkında:
- İstek birden fazla bağımsız adım gerektiriyorsa VE bir adımın gerçekten \
işe yarayıp yaramadığının kontrol edilmesi gerekiyorsa (ör. "ekrana bak, \
hatayı bul, düzelt, tekrar bakıp kontrol et") run_task aracını kullan. \
Bu araç adımları senin yerine sırayla çalıştırıp her birini doğrular.
- Basit, tek adımlı isteklerde (ör. "saat kaç", "şuna not al", "ekrana \
bak") run_task KULLANMA -- gereksiz yavaşlatır, doğrudan yap.
- run_task'a hedefi net ve somut anlat; araç sonucu geldiğinde onu \
kullanıcıya kendi cümlelerinle özetle, ham veriyi olduğu gibi yapıştırma.
"""

SPOKEN_SUMMARY_PREFIX = "SESLİ ÖZET:"

RESEARCH_MODE_PROMPT = f"""

ARAŞTIRMA MODU AKTİF:
- Kullanıcı şu an araştırma modunda. Her soruyu ciddi bir araştırma isteği \
olarak ele al.
- web_search'ü bolca kullan, birden fazla kaynağı karşılaştır, çelişkileri \
belirt, mümkün olduğunca resmi/akademik/güvenilir kaynakları önceliklendir.
- Cevabın UZUN ve DETAYLI olsun; şu yapıyı kullan (düz metin):
KONU / KISA ÖZET / ANA BULGULAR (numaralı) / DETAYLI ANALİZ / SONUÇ / \
KAYNAKLAR (gerçek kaynak adları ve URL'leri).
- Cevabının EN BAŞINA, tek satır halinde şunu ekle:
{SPOKEN_SUMMARY_PREFIX} <en önemli 2-4 cümlelik özet>
Bu satır kullanıcıya SESLİ okunacak, raporun tamamı ise ekranda yazılı \
gösterilecek. Bu yüzden o satır tek başına anlamlı, kısa ve en kritik \
bulguları içeren bir özet olsun.
- Kaynak uydurma. Arama başarısız olduysa bunu açıkça söyle.
"""

MAX_TOOL_ITERATIONS = 6
MAX_TOKENS_DEFAULT = 1024
MAX_TOKENS_RESEARCH = 4096
HISTORY_RESTORE_LIMIT = 20

_RESEARCH_ON_PATTERN = re.compile(r"ara[şs]t[ıi]rma modu(?!.*\b(kapat|kapa|[çc][ıi]k))")
_RESEARCH_OFF_PATTERN = re.compile(
    r"ara[şs]t[ıi]rma modu.*\b(kapat|kapa|[çc][ıi]k)|normal mod|\bmodu kapat"
)


def extract_spoken_summary(reply: str) -> str:
    """Returns just the part meant to be read aloud.

    Research-mode replies are long reports; speaking the whole thing would
    take minutes, so the model is asked to put a short spoken summary on the
    first line behind a marker. Falls back to the text itself when there's
    no marker (ordinary short replies)."""
    for line in reply.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith(SPOKEN_SUMMARY_PREFIX.upper()):
            return stripped[len(SPOKEN_SUMMARY_PREFIX):].strip()
    return reply

# Anthropic's server-side web search tool: Claude runs the search itself and
# the result comes back embedded in the same API response (as
# server_tool_use / web_search_tool_result blocks), so unlike the tools in
# ToolRegistry this needs no local execution -- the loop below only acts on
# type=="tool_use" blocks, which this never produces.
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}


def _current_date_context() -> str:
    now = datetime.now()
    weekday = TR_DAYS[now.weekday()]
    month = TR_MONTHS[now.month - 1]
    return f"\n\nGüncel tarih: {now.strftime('%Y-%m-%d')} ({now.day} {month} {now.year}, {weekday})."


def _preference_context(store: PreferenceStore | None) -> str:
    """Known facts about the user, injected the same way the current date
    is: appended to the system prompt on every call, rather than relying on
    Claude to remember to call recall_preferences first. Empty when there's
    nothing learned yet, so a fresh install's prompt isn't padded with a
    pointless empty section."""
    if store is None:
        return ""
    prefs = store.all()
    if not prefs:
        return ""
    lines = "\n".join(f"- {p.key}: {p.value}" for p in prefs)
    return f"\n\nKullanıcı hakkında bildiklerin:\n{lines}"



def _mode_context(mode_manager: AccessModeManager | None) -> str:
    """Tells Claude the current authority level, the same way the date and
    known preferences are told: appended to every system prompt rather than
    discovered only after a tool call is unexpectedly refused.

    This is what lets NEO say "bunu asistan modunda yapamam, yardımcı
    moduna geçmen gerekiyor" *before* attempting something, instead of
    trying, getting silently refused, and improvising an explanation.
    """
    if mode_manager is None:
        return ""
    if mode_manager.mode is AccessMode.HELPER:
        remaining_minutes = int((mode_manager.seconds_until_drop() or 0.0) // 60)
        return (
            "\n\nYetki modu: YARDIMCI (tam yetkili). Yaklaşık "
            f"{remaining_minutes} dakika hareketsizlik sonrası kendiliğinden "
            "asistan moduna dönecek."
        )
    return (
        "\n\nYetki modu: ASİSTAN (varsayılan, sınırlı). Kapatma, kilitleme, "
        "fare/klavye kontrolü gibi YÜKSEK riskli işlemler bu modda tamamen "
        "kapalı ve denemenin bir anlamı yok -- kullanıcı bunu istiyorsa "
        "önce \"yardımcı moduna geç\" deyip şifresini girmesi gerektiğini "
        "söyle."
    )

def _tool_result_content(result: dict) -> str | list[dict]:
    """Turns a tool's result dict into what actually goes into the
    tool_result message.

    Most tools return plain data, JSON-encoded exactly as before. A tool
    that captured an image (screen vision) instead gets a real Anthropic
    image content block alongside a short text caption with the rest of
    the result -- the Messages API accepts tool_result content as either a
    plain string or a list of blocks, and an image block inside it works
    the same way it would in an ordinary user turn. This is why no "queue
    the image for the next message" plumbing was needed: it goes straight
    into the same turn's tool_result.
    """
    image_base64 = result.get("image_base64")
    if not image_base64:
        return json.dumps(result, ensure_ascii=False)

    media_type = result.get("media_type", "image/png")
    caption = {k: v for k, v in result.items() if k != "image_base64"}
    return [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": image_base64},
        },
        {"type": "text", "text": json.dumps(caption, ensure_ascii=False)},
    ]


@dataclass
class SubtaskResult:
    """What Agent.run_subtask hands back to the task planner: the final
    text Claude produced, plus every real tool call made along the way.

    The trace exists specifically so a step's *verification* has actual
    evidence to judge instead of only the closing sentence -- a live run
    showed a properly skeptical verifier refusing to certify "the time is
    18:11" on faith alone, since prose can't be told apart from a real tool
    result without seeing the tool result itself.
    """

    text: str
    tool_calls: list[dict] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        settings: Settings,
        registry: ToolRegistry,
        permissions: PermissionManager | None = None,
        llm_client: LLMClient | None = None,
        conversation_store: ConversationStore | None = None,
        preference_store: PreferenceStore | None = None,
        mode_manager: AccessModeManager | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._permissions = permissions or PermissionManager()
        self._context = ConversationContext()
        self._llm = llm_client
        self._conversation_store = conversation_store
        self._preference_store = preference_store
        self._mode_manager = mode_manager
        self._mode_unlock_control = None
        self.research_mode = False
        self._listening_control = None
        self._restore_history()

    def set_mode_unlock_control(self, callback) -> None:
        """Lets the GUI expose its password dialog to voice/text control,
        the same way it exposes the confirmation dialog and the wake-word
        loop. `callback` is an async, no-argument function that opens the
        dialog and returns True/False -- Agent never sees the password
        itself or how it's checked, only whether the unlock succeeded."""
        self._mode_unlock_control = callback

    async def _maybe_control_mode(self, text: str) -> str | None:
        """Handles "yardımcı moduna geç" / "asistan moduna dön" locally.

        Switching *into* helper mode can't finish here: it needs a real
        password dialog, which is why this is async and defers to whatever
        GUI callback set_mode_unlock_control() wired up, rather than
        deciding anything about credentials itself.
        """
        action = match_mode_command(text)
        if action is None:
            return None

        if action == REQUEST_ASSISTANT_MODE:
            if self._mode_manager is not None:
                self._mode_manager.drop_to_assistant_mode()
            return "Tamam, asistan moduna döndüm."

        if action == REQUEST_HELPER_MODE:
            if self._mode_manager is None:
                return "Yetki modu bu sürümde henüz yapılandırılmadı."
            if self._mode_unlock_control is None:
                return (
                    "Yardımcı moduna geçmek için bir şifre penceresi açmam "
                    "gerekiyor, ama bu arayüz henüz bağlanmadı."
                )
            unlocked = await self._mode_unlock_control()
            if unlocked:
                return "Yardımcı moduna geçtim, tam yetkiliyim şimdi."
            return "Yardımcı moduna geçilmedi."

        return None

    def _denial_message(self, risk: RiskLevel) -> str:
        """Distinguishes "the user was asked and said no" from "this was
        never offered to the user at all" -- the second happens for
        HIGH-risk tools in assistant mode, and reusing the first message
        for it would make NEO tell the user they declined something they
        were never asked about."""
        if (
            self._mode_manager is not None
            and risk == RiskLevel.HIGH
            and self._mode_manager.mode is AccessMode.ASSISTANT
        ):
            return (
                "Bu işlem asistan modunda tamamen kapalı, kullanıcıya "
                "sorulmadı. Yapılabilmesi için önce yardımcı moduna "
                "geçilmesi gerekiyor."
            )
        return "Kullanıcı bu işlemi onaylamadı."

    def set_listening_control(self, callback) -> None:
        """Lets the GUI expose its wake-word loop to voice control, the same
        way it exposes the confirmation dialog to the permission manager.
        Without it "dinlemeyi durdur" was just another sentence for Claude to
        reply to -- costing a call and not actually stopping anything."""
        self._listening_control = callback

    def _maybe_control_listening(self, text: str) -> str | None:
        action = match_control_command(text)
        if action is None:
            return None
        if self._listening_control is None:
            return None
        if action == STOP_LISTENING:
            self._listening_control(False)
            return "Tamam, dinlemeyi kapattım. Tekrar açmak istersen mikrofon tuşunu kullanabilirsin."
        if action == START_LISTENING:
            self._listening_control(True)
            return "Dinlemeye başladım, seni bekliyorum."
        return None

    def _restore_history(self) -> None:
        """Seeds the in-memory context from stored history so NEO still
        remembers the last exchanges after a restart."""
        if self._conversation_store is None:
            return
        for message in self._conversation_store.recent_messages(limit=HISTORY_RESTORE_LIMIT):
            if message.role == "user":
                self._context.add_user(message.text)
            else:
                self._context.add_assistant(message.text)

    def _record(self, role: str, text: str) -> None:
        if self._conversation_store is not None:
            self._conversation_store.add_message(role, text)

    def _ensure_client(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(self._settings)
        return self._llm

    async def raw_llm_call(
        self, messages: list[dict], system: str, tools: list[dict], max_tokens: int = 1024
    ):
        """A single LLM call outside the tool-use loop, for the task
        planner's own meta-operations (breaking a goal into steps,
        judging whether a step actually succeeded). Those need a specific
        forced tool schema of their own, not the full tool registry a real
        conversational turn or task step gets -- planning what to do and
        actually doing it are different operations, and giving the planner
        access to real tools while it is only supposed to be deciding on
        steps would let it wander into acting instead of planning.
        """
        llm = self._ensure_client()
        return await asyncio.to_thread(llm.send, messages, system, tools, max_tokens)

    def _maybe_toggle_research_mode(self, text: str) -> str | None:
        """Handles "araştırma modu" / "araştırma modunu kapat" locally --
        no LLM call needed just to flip a switch."""
        lowered = text.lower()
        if _RESEARCH_OFF_PATTERN.search(lowered):
            self.research_mode = False
            return "Araştırma modu kapatıldı, normale döndüm."
        if _RESEARCH_ON_PATTERN.search(lowered):
            self.research_mode = True
            return (
                "Araştırma modu açık. Bundan sonra sorduklarını detaylıca "
                "araştırıp uzun bir rapor hazırlayacağım; sesli olarak da "
                "sadece en önemli noktaları okuyacağım. Kapatmak için "
                "'araştırma modunu kapat' de."
            )
        return None

    async def handle_message(self, text: str) -> str:
        mode_reply = await self._maybe_control_mode(text)
        if mode_reply is not None:
            self._context.add_user(text)
            self._context.add_assistant(mode_reply)
            self._record("user", text)
            self._record("assistant", mode_reply)
            return mode_reply

        for handler in (self._maybe_control_listening, self._maybe_toggle_research_mode):
            reply = handler(text)
            if reply is not None:
                self._context.add_user(text)
                self._context.add_assistant(reply)
                self._record("user", text)
                self._record("assistant", reply)
                return reply

        local_reply = None if self.research_mode else await try_handle_locally(text, self._registry)
        if local_reply is not None:
            self._context.add_user(text)
            self._context.add_assistant(local_reply)
            self._record("user", text)
            self._record("assistant", local_reply)
            return local_reply

        try:
            self._ensure_client()
        except ConfigError as exc:
            return str(exc)

        self._context.add_user(text)
        self._record("user", text)
        system_prompt = (
            SYSTEM_PROMPT
            + _current_date_context()
            + _preference_context(self._preference_store)
            + _mode_context(self._mode_manager)
        )
        if self.research_mode:
            system_prompt += RESEARCH_MODE_PROMPT
        max_tokens = MAX_TOKENS_RESEARCH if self.research_mode else MAX_TOKENS_DEFAULT

        reply = await self._run_tool_loop(self._context, system_prompt, max_tokens)
        self._record("assistant", reply)
        return reply

    async def _run_tool_loop(
        self,
        context: ConversationContext,
        system_prompt: str,
        max_tokens: int = MAX_TOKENS_DEFAULT,
        max_iterations: int = MAX_TOOL_ITERATIONS,
        tool_trace: list[dict] | None = None,
    ) -> str:
        """Drives one bounded "ask Claude, run whatever tools it calls, ask
        again" exchange until Claude answers with plain text or the
        iteration budget runs out.

        Factored out of handle_message so the task planner's per-step
        sub-goals (core/planner.py) can reuse the exact same LLM-call,
        tool-execution and permission-check machinery against their own
        throwaway context, instead of the planner re-implementing (and
        risking drifting out of sync with) this loop.

        `tool_trace`, if given a list, gets each tool call actually made
        appended to it as {"name", "input", "result"} -- run_subtask uses
        this so the planner's verification step can check real tool output,
        not just whatever prose Claude chose to summarize it as. A live
        run surfaced exactly this gap: asked to verify a step whose only
        evidence was the closing sentence "the time is 18:11", a properly
        skeptical verifier correctly refused to just take that on faith,
        with no way to tell a real tool call from a made-up answer.
        """
        llm = self._ensure_client()
        for _ in range(max_iterations):
            try:
                response = await asyncio.to_thread(
                    llm.send,
                    context.messages,
                    system_prompt,
                    [*self._registry.anthropic_tools(), WEB_SEARCH_TOOL],
                    max_tokens,
                )
            except LLMRequestError as exc:
                return str(exc)

            context.add_assistant(response.content)

            tool_uses = [block for block in response.content if block.type == "tool_use"]
            if not tool_uses:
                text_blocks = [block.text for block in response.content if block.type == "text"]
                return "\n".join(text_blocks).strip() or "..."

            for block in tool_uses:
                tool = self._registry.get(block.name)
                risk = tool.risk if tool else RiskLevel.HIGH
                allowed = await self._permissions.check(block.name, risk, str(block.input))

                if not allowed:
                    result = {"success": False, "error": self._denial_message(risk)}
                else:
                    tool_result = await self._registry.execute(block.name, block.input)
                    result = tool_result.to_dict()

                if tool_trace is not None:
                    tool_trace.append({"name": block.name, "input": block.input, "result": result})

                context.add_tool_result(block.id, _tool_result_content(result))

        return "Bu istek çok karmaşık hale geldi, tekrar dener misin?"

    async def run_subtask(
        self,
        goal_context: str,
        instruction: str,
        max_tokens: int = MAX_TOKENS_DEFAULT,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> "SubtaskResult":
        """Runs one isolated tool-use exchange for the task planner -- a
        throwaway ConversationContext, not the user's own conversation.

        Deliberately does not touch self._context or self._record(): a
        planner step is internal bookkeeping toward a goal the user asked
        for once, not a new message the user typed. Recording it to
        conversation history would replay a one-sided internal monologue
        into the transcript on the next restart, and mixing it into the
        live chat context could confuse a later, unrelated question with
        half-finished task chatter.

        Returns both the final text and the raw tool-call trace, not just
        text -- the planner's verification step needs the trace as actual
        evidence of what happened, not only Claude's own account of it.
        """
        try:
            self._ensure_client()
        except ConfigError as exc:
            return SubtaskResult(str(exc), [])

        context = ConversationContext()
        context.add_user(instruction)
        system_prompt = (
            SYSTEM_PROMPT
            + _current_date_context()
            + _preference_context(self._preference_store)
            + _mode_context(self._mode_manager)
            + goal_context
        )
        tool_trace: list[dict] = []
        text = await self._run_tool_loop(
            context, system_prompt, max_tokens, max_iterations, tool_trace=tool_trace
        )
        return SubtaskResult(text, tool_trace)
