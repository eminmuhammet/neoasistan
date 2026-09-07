"""TaskPlanner: the "gör → düşün → planla → uygula → doğrula → bildir" loop.

A fake stand-in for Agent (not a real one) drives these tests, since the
planner only needs two things from Agent -- raw_llm_call (forced-tool
meta-calls for planning/verification) and run_subtask (actually running a
step through the real tool loop) -- and controlling both precisely is what
lets these tests exercise retry, failure, and timeout behavior
deterministically without a real LLM.
"""

import asyncio
from dataclasses import dataclass, field

import pytest

from neo.core.agent import SubtaskResult
from neo.core.llm_client import LLMRequestError
from neo.core.planner import MAX_ATTEMPTS_PER_STEP, MAX_TASK_SECONDS, TaskPlanner
from neo.memory.task_store import TaskStore


@dataclass
class FakeBlock:
    type: str
    text: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class FakeMessage:
    content: list


class FakeAgent:
    def __init__(self, plan_steps, verify_results, subtask_results):
        """plan_steps: list[str] or None (None simulates Claude not calling
        the forced plan tool at all). verify_results/subtask_results are
        popped in call order across the whole run."""
        self._plan_steps = plan_steps
        self._verify_results = list(verify_results)
        self._subtask_results = list(subtask_results)
        self.subtask_calls: list[tuple[str, str]] = []
        self.raw_call_count = 0

    async def raw_llm_call(self, messages, system, tools, max_tokens=1024):
        self.raw_call_count += 1
        tool_name = tools[0]["name"] if tools else None

        if tool_name == "submit_plan":
            if self._plan_steps is None:
                return FakeMessage([FakeBlock(type="text", text="düz metin, arac yok")])
            steps_input = [{"description": d} for d in self._plan_steps]
            return FakeMessage(
                [FakeBlock(type="tool_use", name="submit_plan", input={"steps": steps_input})]
            )

        if tool_name == "submit_verification":
            success, reason = self._verify_results.pop(0)
            return FakeMessage(
                [
                    FakeBlock(
                        type="tool_use",
                        name="submit_verification",
                        input={"success": success, "reason": reason},
                    )
                ]
            )

        raise AssertionError(f"beklenmeyen zorlanmis arac: {tool_name}")

    async def run_subtask(self, goal_context, instruction, max_tokens=1024, max_iterations=6):
        self.subtask_calls.append((goal_context, instruction))
        text = self._subtask_results.pop(0)
        # A minimal but real tool call in the trace, so _render_tool_trace
        # (and any verifier prompt built from it) has something concrete to
        # work with -- mirrors what a real run_subtask would produce.
        return SubtaskResult(text, [{"name": "fake_tool", "input": {}, "result": {"success": True}}])


def _planner(tmp_path, agent, on_progress=None):
    store = TaskStore(tmp_path / "tasks.db")
    return TaskPlanner(agent, store, on_progress=on_progress), store


# -- happy path --------------------------------------------------------------


def test_a_two_step_task_completes_successfully(tmp_path):
    agent = FakeAgent(
        plan_steps=["ekrana bak", "hatayı bildir"],
        verify_results=[(True, "iyi"), (True, "iyi")],
        subtask_results=["ekranı gördüm", "hatayı bildirdim"],
    )
    planner, store = _planner(tmp_path, agent)

    success, summary = asyncio.run(planner.run("Ekrandaki hatayı bul ve bildir"))

    assert success is True
    assert "ekranı gördüm" in summary
    assert "hatayı bildirdim" in summary

    task = store.recent_tasks()[0]
    assert task.status == "done"
    assert [s.status for s in task.steps] == ["done", "done"]
    assert [s.attempts for s in task.steps] == [1, 1]


def test_each_step_receives_the_prior_steps_summary_as_context(tmp_path):
    """Later steps need to know what earlier ones actually accomplished --
    otherwise step 2 can't build on step 1's real result."""
    agent = FakeAgent(
        plan_steps=["birinci adım", "ikinci adım"],
        verify_results=[(True, "ok"), (True, "ok")],
        subtask_results=["birinci sonuç", "ikinci sonuç"],
    )
    planner, _ = _planner(tmp_path, agent)

    asyncio.run(planner.run("Hedef"))

    second_call_context = agent.subtask_calls[1][0]
    assert "birinci sonuç" in second_call_context


# -- retry and failure --------------------------------------------------------


def test_a_step_that_fails_once_then_succeeds_completes_the_task(tmp_path):
    agent = FakeAgent(
        plan_steps=["tek adım"],
        verify_results=[(False, "yanlış buton"), (True, "bu sefer doğru")],
        subtask_results=["ilk deneme", "ikinci deneme"],
    )
    planner, store = _planner(tmp_path, agent)

    success, summary = asyncio.run(planner.run("Hedef"))

    assert success is True
    assert "ikinci deneme" in summary  # the successful attempt's result is kept
    step = store.recent_tasks()[0].steps[0]
    assert step.attempts == 2
    assert step.status == "done"


def test_the_retry_instruction_mentions_the_previous_failure_reason(tmp_path):
    """A retry that doesn't know why it failed is just guessing again --
    the planner must actually tell the second attempt what went wrong."""
    agent = FakeAgent(
        plan_steps=["tek adım"],
        verify_results=[(False, "yanlış pencereye tıkladın"), (True, "ok")],
        subtask_results=["ilk deneme", "ikinci deneme"],
    )
    planner, _ = _planner(tmp_path, agent)

    asyncio.run(planner.run("Hedef"))

    second_instruction = agent.subtask_calls[1][1]
    assert "yanlış pencereye tıkladın" in second_instruction


def test_a_step_that_never_succeeds_stops_the_task(tmp_path):
    agent = FakeAgent(
        plan_steps=["imkansız adım", "hiç ulaşılmayacak adım"],
        verify_results=[(False, "olmadı")] * MAX_ATTEMPTS_PER_STEP,
        subtask_results=["deneme"] * MAX_ATTEMPTS_PER_STEP,
    )
    planner, store = _planner(tmp_path, agent)

    success, summary = asyncio.run(planner.run("Hedef"))

    assert success is False
    assert "olmadı" in summary

    task = store.recent_tasks()[0]
    assert task.status == "failed"
    assert task.steps[0].status == "failed"
    assert task.steps[0].attempts == MAX_ATTEMPTS_PER_STEP
    # The second step must never have been attempted at all.
    assert task.steps[1].status == "pending"
    assert len(agent.subtask_calls) == MAX_ATTEMPTS_PER_STEP


def test_retries_never_exceed_the_configured_maximum(tmp_path):
    """More failures queued than MAX_ATTEMPTS_PER_STEP must never be
    consumed -- proves the retry loop is actually bounded, not just
    happening to stop in this particular test's data."""
    agent = FakeAgent(
        plan_steps=["adım"],
        verify_results=[(False, "no")] * 10,
        subtask_results=["deneme"] * 10,
    )
    planner, _ = _planner(tmp_path, agent)

    asyncio.run(planner.run("Hedef"))

    assert len(agent.subtask_calls) == MAX_ATTEMPTS_PER_STEP


# -- planning failures --------------------------------------------------------


def test_empty_plan_fails_gracefully_without_crashing(tmp_path):
    agent = FakeAgent(plan_steps=None, verify_results=[], subtask_results=[])
    planner, store = _planner(tmp_path, agent)

    success, summary = asyncio.run(planner.run("Belirsiz bir istek"))

    assert success is False
    assert "adımlara bölemedim" in summary
    assert store.recent_tasks()[0].status == "failed"


def test_plan_creation_llm_failure_does_not_crash(tmp_path):
    class BrokenPlanAgent(FakeAgent):
        async def raw_llm_call(self, *a, **k):
            raise LLMRequestError("API'ye ulaşılamadı")

    agent = BrokenPlanAgent(plan_steps=[], verify_results=[], subtask_results=[])
    planner, store = _planner(tmp_path, agent)

    success, summary = asyncio.run(planner.run("Hedef"))

    assert success is False
    assert "ulaşılamadı" in summary
    assert store.recent_tasks()[0].status == "failed"


def test_verification_llm_failure_counts_as_a_failed_attempt_not_a_crash(tmp_path):
    class FlakyVerifyAgent(FakeAgent):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._calls = 0

        async def raw_llm_call(self, messages, system, tools, max_tokens=1024):
            if tools and tools[0]["name"] == "submit_verification":
                self._calls += 1
                if self._calls == 1:
                    raise LLMRequestError("gecici hata")
            return await super().raw_llm_call(messages, system, tools, max_tokens)

    agent = FlakyVerifyAgent(
        plan_steps=["adım"],
        verify_results=[(True, "ikinci denemede calisti")],
        subtask_results=["ilk", "ikinci"],
    )
    planner, _ = _planner(tmp_path, agent)

    success, summary = asyncio.run(planner.run("Hedef"))

    assert success is True  # recovered on the retry after the flaky call


# -- timeout ------------------------------------------------------------------


def test_task_stops_when_the_time_budget_is_exceeded(tmp_path, monkeypatch):
    import neo.core.planner as planner_module

    agent = FakeAgent(
        plan_steps=["adım 1", "adım 2"],
        verify_results=[(True, "ok")],
        subtask_results=["sonuç 1"],
    )
    planner, store = _planner(tmp_path, agent)

    clock = [1000.0]
    monkeypatch.setattr(planner_module.time, "monotonic", lambda: clock[0])

    async def run_and_expire():
        # Jump the clock past the deadline right after the first step
        # would have started, simulating a task that ran long.
        real_execute_step = planner._execute_step

        async def slow_execute_step(*args, **kwargs):
            clock[0] += MAX_TASK_SECONDS + 1
            return await real_execute_step(*args, **kwargs)

        planner._execute_step = slow_execute_step
        return await planner.run("Hedef")

    success, summary = asyncio.run(run_and_expire())

    assert success is False
    assert "zaman sınırını aştı" in summary
    task = store.recent_tasks()[0]
    assert task.status == "failed"
    # The clock is advanced *during* the first step, so that step still
    # completes normally; the deadline check runs at the top of the loop,
    # before the *next* step -- so it is the second step that gets cut off,
    # not the first.
    assert task.steps[0].status == "done"
    assert task.steps[1].status == "failed"
    assert "Zaman aşımı" in task.steps[1].result_summary


# -- progress notifications ---------------------------------------------------


def test_progress_callback_fires_as_the_task_advances(tmp_path):
    seen_statuses = []

    def on_progress(task):
        seen_statuses.append((task.status, [s.status for s in task.steps]))

    agent = FakeAgent(
        plan_steps=["tek adım"],
        verify_results=[(True, "ok")],
        subtask_results=["sonuç"],
    )
    planner, _ = _planner(tmp_path, agent, on_progress=on_progress)

    asyncio.run(planner.run("Hedef"))

    assert len(seen_statuses) >= 2  # at least "running" then "done"
    assert seen_statuses[-1][0] == "done"
    assert seen_statuses[-1][1] == ["done"]


def test_no_progress_callback_is_fine(tmp_path):
    """on_progress is optional -- must not be required for the planner to
    run at all."""
    agent = FakeAgent(plan_steps=["adım"], verify_results=[(True, "ok")], subtask_results=["x"])
    planner, _ = _planner(tmp_path, agent, on_progress=None)

    success, _ = asyncio.run(planner.run("Hedef"))
    assert success is True
