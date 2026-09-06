from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Every turn resends the whole retained history, so this is a direct, ongoing
# cost multiplier -- and tool results (web-search payloads especially) are
# large. 20 still covers a long back-and-forth; anything older is in the
# conversation store if it's ever needed.
MAX_MESSAGES = 20


@dataclass
class ConversationContext:
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_messages: int = MAX_MESSAGES

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, content: Any) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def add_tool_result(self, tool_use_id: str, content: str) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                    }
                ],
            }
        )
        self._trim()

    @staticmethod
    def _is_tool_result(message: dict[str, Any]) -> bool:
        content = message.get("content")
        return isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in content
        )

    def _trim(self) -> None:
        """Drops the oldest messages, but never in a way that leaves a
        tool_result whose tool_use was cut away.

        The API rejects an orphaned tool_result outright, so a plain slice
        could turn a long conversation into a hard error the moment the
        window happened to fall between a tool call and its result.
        """
        if len(self.messages) <= self.max_messages:
            return
        overflow = len(self.messages) - self.max_messages
        trimmed = self.messages[overflow:]
        while trimmed and self._is_tool_result(trimmed[0]):
            trimmed = trimmed[1:]
        self.messages = trimmed

    def clear(self) -> None:
        self.messages.clear()
