from neo.core.context import ConversationContext


def test_add_user_and_assistant():
    ctx = ConversationContext()
    ctx.add_user("merhaba")
    ctx.add_assistant("selam")
    assert ctx.messages[0]["role"] == "user"
    assert ctx.messages[1]["role"] == "assistant"


def test_trims_old_messages():
    ctx = ConversationContext(max_messages=4)
    for i in range(10):
        ctx.add_user(f"mesaj {i}")
    assert len(ctx.messages) == 4
    assert ctx.messages[-1]["content"] == "mesaj 9"
