DARK_QSS = """
QWidget {
    background-color: #0b0f14;
    color: #d7e3ea;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #0b0f14;
}

QTextEdit {
    background-color: #10161d;
    border: 1px solid #1f2a35;
    border-radius: 10px;
    padding: 10px;
    selection-background-color: #23c9ff55;
}

QLineEdit {
    background-color: #10161d;
    border: 1px solid #1f2a35;
    border-radius: 18px;
    padding: 8px 14px;
    color: #d7e3ea;
}

QLineEdit:focus {
    border: 1px solid #23c9ff;
}

QPushButton {
    background-color: #17222c;
    border: 1px solid #23c9ff;
    border-radius: 18px;
    color: #23c9ff;
    padding: 8px 18px;
}

QPushButton:hover {
    background-color: #1c2b38;
}

QPushButton:pressed {
    background-color: #23c9ff;
    color: #0b0f14;
}

QPushButton:disabled {
    background-color: #12181f;
    border: 1px solid #26313c;
    color: #4c5a66;
}

QPushButton:checked {
    background-color: #113042;
    border: 1px solid #35e08a;
    color: #35e08a;
}

QPushButton#InfoButton {
    border-radius: 16px;
    padding: 4px;
    font-size: 15px;
    min-width: 32px;
    max-width: 32px;
    min-height: 32px;
    max-height: 32px;
}

QPushButton#StopButton {
    border-color: #ff4d4f;
    color: #ff4d4f;
}

QPushButton#StopButton:hover {
    background-color: #3a1416;
}

QLabel#TitleLabel {
    font-size: 24px;
    font-weight: 600;
    color: #23c9ff;
    letter-spacing: 6px;
}

QLabel#StatusLabel {
    color: #9db0bd;
    font-size: 13px;
    font-weight: 600;
}

QLabel#ResearchLabel {
    color: #f2b134;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

QWidget#StatsPanel, QWidget#ControlPanel {
    background-color: #0e1319;
    border: 1px solid #1c2733;
    border-radius: 12px;
}

QScrollArea#ChatScrollArea, QWidget#ChatContainer {
    background-color: transparent;
    border: none;
}

QScrollArea#ChatScrollArea QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}

QScrollArea#ChatScrollArea QScrollBar::handle:vertical {
    background: #1f2a35;
    border-radius: 4px;
    min-height: 24px;
}

QScrollArea#ChatScrollArea QScrollBar::add-line:vertical,
QScrollArea#ChatScrollArea QScrollBar::sub-line:vertical {
    height: 0;
}

QFrame#BubbleUser {
    background-color: #142530;
    border: 1px solid #23c9ff;
    border-radius: 14px;
}

QFrame#BubbleNeo {
    background-color: #10161d;
    border: 1px solid #1f2a35;
    border-radius: 14px;
}

QLabel#BubbleSender {
    color: #23c9ff;
    font-weight: 700;
    font-size: 11px;
}

QLabel#BubbleText {
    color: #d7e3ea;
    font-size: 13px;
    background: transparent;
}
"""
