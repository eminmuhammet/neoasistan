import asyncio

import pytest

from neo.core.agent import SYSTEM_PROMPT
from neo.core.local_commands import try_handle_locally
from neo.ui.main_window import build_about_text

CREATOR = "Emin İLHAN"


@pytest.mark.parametrize(
    "question",
    [
        "seni kim yaptı",
        "Seni kim geliştirdi?",
        "seni kim kodladı",
        "yapımcın kim",
        "geliştiricin kim",
        "üreticin kim",
    ],
)
def test_creator_question_is_answered_locally(question):
    """Answered without a network round trip, so it stays correct and free."""
    reply = asyncio.run(try_handle_locally(question, registry=None))

    assert reply is not None
    assert CREATOR in reply


def test_system_prompt_names_the_creator():
    assert CREATOR in SYSTEM_PROMPT


def test_about_dialog_credits_the_creator():
    assert CREATOR in build_about_text()


@pytest.mark.parametrize("brand", ["Claude", "Anthropic", "OpenAI", "GPT", "Gemini"])
def test_no_brand_names_in_what_neo_says(brand):
    """NEO presents as NEO. Other brands aren't advertised in its own
    persona, its capability list, or the About dialog."""
    assert brand not in build_about_text()

    capabilities = asyncio.run(try_handle_locally("neler yapabilirsin", registry=None))
    assert brand not in capabilities


def test_prompt_does_not_instruct_neo_to_deny_what_it_runs_on():
    """Not advertising a brand is branding; denying a direct, sincere
    question would be a lie, and the prompt must not ask for that."""
    assert "yalan söyleme" in SYSTEM_PROMPT


def test_system_prompt_establishes_the_master_address():
    """The user asked NEO to address them as "efendim" and treat them as
    its master -- a tone/address instruction, not blind obedience."""
    assert "efendim" in SYSTEM_PROMPT
    assert "efendi" in SYSTEM_PROMPT.lower()


def test_master_address_is_not_framed_as_blind_obedience():
    """A prompt that just says "obey your master" would make NEO agree with
    risky requests instead of flagging them. The instruction must keep the
    honesty/safety framing intact."""
    lowered = SYSTEM_PROMPT.lower()
    assert "körü körüne" in lowered or "yanlış" in lowered
