from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Callable

from .llm_client import LLMRequestError
from ..memory.task_store import Task, TaskStore

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)

# Bounds that keep a task from running away: a step that keeps failing gets
# a bounded number of retries, not infinite ones, and the whole task gives
# up past a wall-clock budget rather than running all afternoon on
# something that was never going to finish.
MAX_ATTEMPTS_PER_STEP = 2
MAX_TASK_SECONDS = 900.0  # 15 minutes
MAX_PLAN_STEPS = 8
# How many times the whole remaining plan may be rewritten after a step
# keeps failing. Bounded independently of MAX_ATTEMPTS_PER_STEP (retries of
# the *same* step) and MAX_TASK_SECONDS (wall clock): without this, a goal
# that's subtly impossible could get replanned forever, each attempt
# plausible-looking but never actually landing.
MAX_REPLANS = 2

ProgressCallback = Callable[[Task], None]

# Structured output via a forced tool call, rather than asking Claude to
# format a numbered list in prose and parsing that: tool-use input is
# already schema-validated and reliably parseable, which is exactly what
# this codebase already leans on elsewhere (remember_preference,
# add_calendar_note, ...) for anything that needs to come back as data
# instead of free text.
_PLAN_TOOL = {
    "name": "submit_plan",
    "description": "Verilen hedefi sıralı, tek tek doğrulanabilir adımlara böler.",
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Bu adımda tam olarak ne yapılacağı.",
                        }
                    },
                    "required": ["description"],
                },
                "minItems": 1,
            }
        },
        "required": ["steps"],
    },
}

_VERIFY_TOOL = {
    "name": "submit_verification",
    "description": "Bir görev adımının gerçekten başarıyla tamamlanıp tamamlanmadığını bildirir.",
    "input_schema": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "reason": {
                "type": "string",
                "description": "Kısa gerekçe -- neden başarılı ya da başarısız.",
            },
        },
        "required": ["success", "reason"],
    },
}

_PLANNING_SYSTEM_PROMPT = (
    "Sen NEO'nun görev planlayıcısısın. Sana bir hedef verilecek; bunu, "
    f"her biri tek başına anlamlı ve sonucu kontrol edilebilir en fazla "
    f"{MAX_PLAN_STEPS} adıma böl. Adımlar mantıklı bir sırada olsun ve her "
    "biri somut, ne yapılacağı açık olsun (\"bir şeyler yap\" gibi belirsiz "
    "adımlar değil). Basit, tek adımlı hedefler için tek adımlık bir plan "
    "vermen yeterli. Cevabını SADECE submit_plan aracıyla ver."
)

_REPLAN_TOOL = {
    "name": "submit_replan",
    "description": (
        "Başarısız olan bir adımdan sonra, hedefe farklı bir yoldan "
        "ulaşmanın mümkün olup olmadığına karar verir."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "give_up": {
                "type": "boolean",
                "description": (
                    "Hedefe bu koşullar altında ulaşmanın gerçekten mümkün "
                    "olmadığı düşünülüyorsa true."
                ),
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"description": {"type": "string"}},
                    "required": ["description"],
                },
                "description": (
                    "give_up false ise, başarısız adımın yerini alacak ve "
                    "hedefe ulaşmayı deneyecek yeni adım(lar)."
                ),
            },
        },
        "required": ["give_up"],
    },
}

_REPLAN_SYSTEM_PROMPT = (
    "Sen NEO'nun görev planlayıcısısın. Bir görev adımı başarısız oldu ve "
    "yeniden denenmesi de işe yaramadı. Sana genel hedefi, şimdiye kadar "
    "tamamlanan adımları, başarısız olan adımı, neden başarısız olduğunu ve "
    "eski plandaki artık geçersiz sayılan kalan adımları vereceğim. Görevi "
    "bırakmadan önce farklı bir yaklaşımla hedefe ulaşmanın bir yolu olup "
    "olmadığını düşün -- aynı hatayı tekrarlayacak bir adım önerme. Varsa, "
    f"en fazla {MAX_PLAN_STEPS} yeni adımdan oluşan bir plan öner. Gerçekten "
    "bir yol yoksa (ör. gerekli bir izin/veri hiç yok, ya da bir araç kalıcı "
    "olarak kullanılamıyor) give_up=true ile bildir; iyimser olma. Cevabını "
    "SADECE submit_replan aracıyla ver."
)

_VERIFY_SYSTEM_PROMPT = (
    "Sen bir görev adımının sonucunu değerlendiren bağımsız bir "
    "denetleyicisin. Sana adımın amacı, o adımda gerçekten çalıştırılan "
    "araçların ham sonuçları ve NEO'nun buna dayanarak söylediği cümle "
    "verilecek. Kararını NEO'nun cümlesine değil, ham araç sonuçlarına "
    "dayandır -- araç sonucu gerçekten adımın amacını karşılıyorsa "
    "başarılı say, NEO'nun sözü tek kanıt gibi görünüyorsa (hiç araç "
    "çalıştırılmamışsa ya da sonuç amaçla uyuşmuyorsa) başarısız say. "
    "submit_verification aracıyla bildir. İyimser olma: bir araç hata "
    "döndürdüyse ya da sonuç belirsizse başarısız say."
)


class TaskPlanner:
    """The "gör → düşün → planla → uygula → doğrula → bildir" loop: breaks a
    goal into steps, runs each one (through Agent.run_subtask, so every real
    action still goes through the exact same tool registry and permission
    checks an ordinary message would), and checks the actual result against
    what the step was supposed to accomplish before moving on.

    A step that still fails after its retries doesn't immediately end the
    task: the remaining plan gets rewritten around the failure (see
    _replan) up to MAX_REPLANS times, so "gör → düşün → planla → uygula →
    doğrula → **planı revize et** → devam et" is the actual loop, not just
    plan-once-and-retry.
    """

    def __init__(
        self,
        agent: "Agent",
        task_store: TaskStore,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._agent = agent
        self._task_store = task_store
        self._on_progress = on_progress

    async def run(self, goal: str) -> tuple[bool, str]:
        task_id = await asyncio.to_thread(self._task_store.create_task, goal)

        try:
            step_descriptions = await self._create_plan(goal)
        except LLMRequestError as exc:
            await asyncio.to_thread(self._task_store.update_task_status, task_id, "failed")
            return False, str(exc)

        if not step_descriptions:
            await asyncio.to_thread(self._task_store.update_task_status, task_id, "failed")
            return False, "Bu hedefi adımlara bölemedim, daha açık ifade eder misin?"

        for position, description in enumerate(step_descriptions):
            await asyncio.to_thread(self._task_store.add_step, task_id, position, description)

        await asyncio.to_thread(self._task_store.update_task_status, task_id, "running")
        await self._notify(task_id)

        completed_summaries: list[str] = []
        deadline = time.monotonic() + MAX_TASK_SECONDS
        replans_used = 0

        while True:
            task = await asyncio.to_thread(self._task_store.get_task, task_id)
            step = next((s for s in task.steps if s.status == "pending"), None)
            if step is None:
                break

            if time.monotonic() > deadline:
                await asyncio.to_thread(
                    self._task_store.update_step, step.id, status="failed", result_summary="Zaman aşımı"
                )
                await asyncio.to_thread(self._task_store.update_task_status, task_id, "failed")
                await self._notify(task_id)
                return False, self._final_summary(
                    goal, completed_summaries, "Görev zaman sınırını aştı."
                )

            await asyncio.to_thread(self._task_store.update_step, step.id, status="running")
            await self._notify(task_id)

            success, summary = await self._execute_step(goal, completed_summaries, step)

            if success:
                await asyncio.to_thread(
                    self._task_store.update_step, step.id, status="done", result_summary=summary
                )
                completed_summaries.append(summary)
                await self._notify(task_id)
                continue

            await asyncio.to_thread(
                self._task_store.update_step, step.id, status="failed", result_summary=summary
            )
            await self._notify(task_id)

            if replans_used >= MAX_REPLANS:
                await asyncio.to_thread(self._task_store.update_task_status, task_id, "failed")
                await self._notify(task_id)
                return False, self._final_summary(goal, completed_summaries, summary)

            remaining = [s for s in task.steps if s.status == "pending" and s.id != step.id]
            try:
                give_up, new_descriptions = await self._replan(
                    goal, completed_summaries, step.description, summary,
                    [s.description for s in remaining],
                )
            except LLMRequestError:
                give_up, new_descriptions = True, []

            if give_up or not new_descriptions:
                await asyncio.to_thread(self._task_store.update_task_status, task_id, "failed")
                await self._notify(task_id)
                return False, self._final_summary(goal, completed_summaries, summary)

            replans_used += 1
            # The old remaining plan is superseded, not merely postponed --
            # left "pending" they'd still be picked up (in their original,
            # now-stale position order) once the new steps finished.
            for old_step in remaining:
                await asyncio.to_thread(self._task_store.update_step, old_step.id, status="skipped")

            next_position = max((s.position for s in task.steps), default=-1) + 1
            for description in new_descriptions:
                await asyncio.to_thread(self._task_store.add_step, task_id, next_position, description)
                next_position += 1
            await self._notify(task_id)

        await asyncio.to_thread(self._task_store.update_task_status, task_id, "done")
        await self._notify(task_id)
        return True, self._final_summary(goal, completed_summaries, None)

    async def _execute_step(
        self, goal: str, prior_summaries: list[str], step
    ) -> tuple[bool, str]:
        failure_reason = ""
        for attempt in range(1, MAX_ATTEMPTS_PER_STEP + 1):
            instruction = step.description
            if failure_reason:
                instruction += (
                    f"\n\n(Önceki denemen başarısız oldu: {failure_reason} "
                    "Farklı bir yaklaşım dene.)"
                )

            goal_context = self._step_context(goal, prior_summaries, step.description)
            result = await self._agent.run_subtask(goal_context, instruction)
            await asyncio.to_thread(self._task_store.update_step, step.id, attempts=attempt)

            try:
                success, reason = await self._verify(goal, step.description, result)
            except LLMRequestError as exc:
                # A broken verification call must not silently look like a
                # successful step -- treat it as this attempt failing and
                # let the retry loop (or the final "failed" outcome) handle it.
                success, reason = False, str(exc)

            if success:
                return True, result.text
            failure_reason = reason

        return False, failure_reason or "Adım doğrulanamadı."

    async def _create_plan(self, goal: str) -> list[str]:
        messages = [{"role": "user", "content": f"Hedef: {goal}"}]
        response = await self._agent.raw_llm_call(
            messages, _PLANNING_SYSTEM_PROMPT, [_PLAN_TOOL], 1024
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_plan":
                steps = block.input.get("steps", [])
                descriptions = [s["description"] for s in steps if s.get("description")]
                return descriptions[:MAX_PLAN_STEPS]
        return []

    async def _verify(self, goal: str, step_description: str, result) -> tuple[bool, str]:
        prompt = (
            f"Genel hedef: {goal}\n\n"
            f"Bu adımın amacı: {step_description}\n\n"
            f"Bu adımda çalıştırılan araçlar ve gerçek sonuçları:\n"
            f"{self._render_tool_trace(result.tool_calls)}\n\n"
            f"NEO'nun adım sonunda söylediği: {result.text}\n\n"
            "Yukarıdaki gerçek araç sonuçlarına bakarak (sadece NEO'nun "
            "söylediğine değil) bu adım gerçekten başarıyla tamamlandı mı?"
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self._agent.raw_llm_call(
            messages, _VERIFY_SYSTEM_PROMPT, [_VERIFY_TOOL], 512
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_verification":
                return bool(block.input.get("success")), str(block.input.get("reason", ""))
        # Claude didn't call the forced tool -- treat this conservatively as
        # a failure rather than silently assuming the step succeeded on
        # something that couldn't actually be parsed.
        logger.warning("Doğrulama yanıtı ayrıştırılamadı, adım başarısız sayılıyor")
        return False, "Doğrulama yanıtı anlaşılamadı."

    async def _replan(
        self,
        goal: str,
        prior_summaries: list[str],
        failed_description: str,
        failure_reason: str,
        remaining_descriptions: list[str],
    ) -> tuple[bool, list[str]]:
        """Asks whether a different approach could still reach the goal
        after a step exhausted its retries. Returns (give_up, new_steps) --
        give_up=True (or an empty step list) both mean "stop the task",
        the caller doesn't need to distinguish which."""
        done = "\n".join(f"- {s}" for s in prior_summaries) if prior_summaries else "(henüz yok)"
        remaining = (
            "\n".join(f"- {d}" for d in remaining_descriptions)
            if remaining_descriptions else "(kalan adım yok)"
        )
        prompt = (
            f"Genel hedef: {goal}\n\n"
            f"Şimdiye kadar tamamlanan adımlar:\n{done}\n\n"
            f"Başarısız olan adım: {failed_description}\n"
            f"Başarısızlık nedeni: {failure_reason}\n\n"
            f"Eski plandaki, artık geçersiz sayılan kalan adımlar:\n{remaining}\n\n"
            "Hedefe farklı bir yoldan ulaşmanın bir yolu var mı?"
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self._agent.raw_llm_call(
            messages, _REPLAN_SYSTEM_PROMPT, [_REPLAN_TOOL], 1024
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_replan":
                if block.input.get("give_up"):
                    return True, []
                steps = block.input.get("steps", [])
                descriptions = [s["description"] for s in steps if s.get("description")]
                return False, descriptions[:MAX_PLAN_STEPS]
        # Claude didn't call the forced tool -- give up conservatively
        # rather than looping on an unparseable response.
        logger.warning("Yeniden planlama yanıtı ayrıştırılamadı, görev durduruluyor")
        return True, []

    def _render_tool_trace(self, tool_calls: list[dict]) -> str:
        """Real evidence for the verifier -- without this, verification was
        only ever given Claude's own closing sentence, and a properly
        skeptical verifier correctly refused to certify a claim ("the time
        is 18:11") it had no way to tell apart from a made-up answer."""
        if not tool_calls:
            return "(bu adımda hiçbir araç çalıştırılmadı)"
        lines = []
        for call in tool_calls:
            lines.append(f"- {call['name']}({call['input']}) -> {call['result']}")
        return "\n".join(lines)

    def _step_context(self, goal: str, prior_summaries: list[str], step_description: str) -> str:
        done = "\n".join(f"- {s}" for s in prior_summaries) if prior_summaries else "(henüz yok)"
        return (
            "\n\nŞU AN BİR GÖREVİN PARÇASI OLARAK ÇALIŞIYORSUN.\n"
            f"Genel hedef: {goal}\n"
            f"Şimdiye kadar tamamlanan adımlar:\n{done}\n"
            f"Şimdi SADECE şu adımı gerçekleştir: {step_description}\n"
            "Gerektiği kadar araç kullan, ama sadece bu adımın kapsamında kal."
        )

    def _final_summary(self, goal: str, summaries: list[str], failure_reason: str | None) -> str:
        bullet_list = "\n".join(f"- {s}" for s in summaries) if summaries else "(hiçbir adım tamamlanamadı)"
        if failure_reason is None:
            return f"Görevi tamamladım: {goal}\n\n{bullet_list}"
        return (
            f"Görevi tamamlayamadım: {goal}\n\n"
            f"Tamamlanan adımlar:\n{bullet_list}\n\n"
            f"Durma nedeni: {failure_reason}"
        )

    async def _notify(self, task_id: int) -> None:
        if self._on_progress is None:
            return
        task = await asyncio.to_thread(self._task_store.get_task, task_id)
        if task is not None:
            self._on_progress(task)
