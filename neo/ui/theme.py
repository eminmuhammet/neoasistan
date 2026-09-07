DARK_QSS = """
QWidget {
    background-color: #060a08;
    color: #d9ece0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #060a08;
}

QTextEdit {
    background-color: #0c1410;
    border: 1px solid #1b2e22;
    border-radius: 10px;
    padding: 10px;
    selection-background-color: #35e08a55;
}

QLineEdit {
    background-color: #0c1410;
    border: 1px solid #1b2e22;
    border-radius: 18px;
    padding: 8px 14px;
    color: #d9ece0;
}

QLineEdit:focus {
    border: 1px solid #2fa8e0;
}

QPushButton {
    background-color: #0f1a14;
    border: 1px solid #35e08a;
    border-radius: 18px;
    color: #35e08a;
    padding: 8px 18px;
}

QPushButton:hover {
    background-color: #14251a;
}

QPushButton:pressed {
    background-color: #35e08a;
    color: #06120c;
}

QPushButton:disabled {
    background-color: #10160f;
    border: 1px solid #223328;
    color: #4c6656;
}

QPushButton:checked {
    background-color: #113042;
    border: 1px solid #2fa8e0;
    color: #2fa8e0;
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
    font-size: 26px;
    font-weight: 600;
    color: #35e08a;
    letter-spacing: 6px;
}

QLabel#SubtitleLabel {
    color: #5c7568;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 3px;
}

QLabel#StatusLabel {
    color: #9db8a8;
    font-size: 13px;
    font-weight: 600;
}

QLabel#ResearchLabel {
    color: #f2b134;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

QWidget#StatsPanel, QWidget#ControlPanel, QWidget#LeftPanel {
    background-color: #0a120d;
    border: 1px solid #182920;
    border-radius: 12px;
}

QWidget#LeftPanel {
    border: none;
    background-color: transparent;
}

QWidget#StatsFooter {
    background-color: #060d09;
    border-top: 1px solid #111e16;
}

QWidget#StatsFooter QWidget#StatsPanel {
    border: none;
    background-color: transparent;
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
    background: #1b2e22;
    border-radius: 4px;
    min-height: 24px;
}

QScrollArea#ChatScrollArea QScrollBar::add-line:vertical,
QScrollArea#ChatScrollArea QScrollBar::sub-line:vertical {
    height: 0;
}

QFrame#BubbleUser {
    background-color: #12222e;
    border: 1px solid #2fa8e0;
    border-radius: 14px;
}

QFrame#BubbleNeo {
    background-color: #0c1410;
    border: 1px solid #1b2e22;
    border-radius: 14px;
}

QLabel#BubbleSender {
    color: #35e08a;
    font-weight: 700;
    font-size: 11px;
}

QLabel#BubbleText {
    color: #d9ece0;
    font-size: 13px;
    background: transparent;
}
"""
