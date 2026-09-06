from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class ChatView(QScrollArea):
    """A scrollable column of chat bubbles -- user messages right-aligned,
    NEO's left-aligned, each with distinct styling -- instead of a flat
    QTextEdit transcript, so a conversation reads clearly at a glance."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ChatScrollArea")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        container.setObjectName("ChatContainer")
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)
        self.setWidget(container)

        # Scrolling on a 0ms timer after appending didn't work: the timer
        # fires before the word-wrapped bubble has been laid out, so
        # scrollBar.maximum() is still the pre-insert value and the view
        # stops short of the newest message. rangeChanged fires once the
        # layout has actually grown, which is the moment to follow it.
        self._pinned_to_bottom = True
        self._bubbles: list[QWidget] = []
        bar = self.verticalScrollBar()
        bar.rangeChanged.connect(self._follow_new_content)
        bar.valueChanged.connect(self._remember_scroll_position)

    def _remember_scroll_position(self, value: int) -> None:
        """Stop following the tail while the user is reading further up, and
        start again once they return to the bottom."""
        bar = self.verticalScrollBar()
        self._pinned_to_bottom = value >= bar.maximum() - 4

    def _follow_new_content(self, _minimum: int, maximum: int) -> None:
        if self._pinned_to_bottom:
            self.verticalScrollBar().setValue(maximum)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._apply_bubble_width()

    def append(self, sender: str, text: str) -> None:
        is_user = sender == "Sen"

        bubble = QFrame()
        bubble.setObjectName("BubbleUser" if is_user else "BubbleNeo")
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 8, 12, 8)
        bubble_layout.setSpacing(2)

        if not is_user:
            name_label = QLabel("NEO")
            name_label.setObjectName("BubbleSender")
            bubble_layout.addWidget(name_label)

        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setObjectName("BubbleText")
        text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bubble_layout.addWidget(text_label)

        bubble.setMaximumWidth(self._bubble_width())
        bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self._bubbles.append(bubble)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if is_user:
            row.addStretch(1)
            row.addWidget(bubble)
        else:
            row.addWidget(bubble)
            row.addStretch(1)

        # Insert before the trailing stretch that keeps messages
        # bottom-anchored while the conversation is still short.
        self._layout.insertLayout(self._layout.count() - 1, row)

    def _bubble_width(self) -> int:
        """Bubbles scale with the window instead of being pinned at 360px.

        On a maximized window that fixed cap wrapped a one-line reply across
        four lines with most of the row left empty.
        """
        return max(320, int(self.viewport().width() * 0.62))

    def _apply_bubble_width(self) -> None:
        width = self._bubble_width()
        for bubble in self._bubbles:
            bubble.setMaximumWidth(width)
