from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime

from ..config.settings import ConfigError, Settings
from ..tools.base import RiskLevel, ToolRegistry
from .context import ConversationContext
from .llm_client import LLMClient, LLMRequestError
from .local_commands import (
    START_LISTENING,
    STOP_LISTENING,
    match_control_command,
    try_handle_locally,
)
from .permissions import PermissionManager
from ..memory.conversation_store import ConversationStore
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


class Agent:
    def __init__(
        self,
        settings: Settings,
        registry: ToolRegistry,
        permissions: PermissionManager | None = None,
        llm_client: LLMClient | None = None,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._permissions = permissions or PermissionManager()
        self._context = ConversationContext()
        self._llm = llm_client
        self._conversation_store = conversation_store
        self.research_mode = False
        self._listening_control = None
        self._restore_history()

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
            llm = self._ensure_client()
        except ConfigError as exc:
            return str(exc)

        self._context.add_user(text)
        self._record("user", text)
        system_prompt = SYSTEM_PROMPT + _current_date_context()
        if self.research_mode:
            system_prompt += RESEARCH_MODE_PROMPT
        max_tokens = MAX_TOKENS_RESEARCH if self.research_mode else MAX_TOKENS_DEFAULT

        for _ in range(MAX_TOOL_ITERATIONS):
            try:
                response = await asyncio.to_thread(
                    llm.send,
                    self._context.messages,
                    system_prompt,
                    [*self._registry.anthropic_tools(), WEB_SEARCH_TOOL],
                    max_tokens,
                )
            except LLMRequestError as exc:
                return str(exc)

            self._context.add_assistant(response.content)

            tool_uses = [block for block in response.content if block.type == "tool_use"]
            if not tool_uses:
                text_blocks = [block.text for block in response.content if block.type == "text"]
                reply = "\n".join(text_blocks).strip() or "..."
                self._record("assistant", reply)
                return reply

            for block in tool_uses:
                tool = self._registry.get(block.name)
                risk = tool.risk if tool else RiskLevel.HIGH
                allowed = await self._permissions.check(block.name, risk, str(block.input))

                if not allowed:
                    result = {"success": False, "error": "Kullanıcı bu işlemi onaylamadı."}
                else:
                    tool_result = await self._registry.execute(block.name, block.input)
                    result = tool_result.to_dict()

                self._context.add_tool_result(block.id, json.dumps(result, ensure_ascii=False))

        return "Bu istek çok karmaşık hale geldi, tekrar dener misin?"
