from neo.web.server import build_app, load_or_create_token


class FakeAgent:
    def __init__(self) -> None:
        self.received: list[str] = []

    async def handle_message(self, text: str) -> str:
        self.received.append(text)
        return f"yanıt: {text}"


def _client(token="test-token", agent=None):
    from fastapi.testclient import TestClient

    app = build_app(agent or FakeAgent(), token)
    return TestClient(app)


def test_status_requires_a_valid_token():
    client = _client(token="secret")

    no_token = client.get("/status")
    wrong_token = client.get("/status", headers={"x-neo-token": "wrong"})
    right_token = client.get("/status", headers={"x-neo-token": "secret"})

    assert no_token.status_code == 401
    assert wrong_token.status_code == 401
    assert right_token.status_code == 200
    assert right_token.json() == {"status": "ok"}


def test_message_endpoint_relays_to_agent_and_returns_reply():
    agent = FakeAgent()
    client = _client(token="secret", agent=agent)

    response = client.post(
        "/message", json={"text": "merhaba"}, headers={"x-neo-token": "secret"}
    )

    assert response.status_code == 200
    assert response.json() == {"reply": "yanıt: merhaba"}
    assert agent.received == ["merhaba"]


def test_message_endpoint_requires_token_too():
    client = _client(token="secret")

    response = client.post("/message", json={"text": "merhaba"})

    assert response.status_code == 401


def test_message_endpoint_rejects_empty_text():
    client = _client(token="secret")

    response = client.post(
        "/message", json={"text": "   "}, headers={"x-neo-token": "secret"}
    )

    assert response.status_code == 400


def test_load_or_create_token_persists_across_calls(tmp_path):
    path = tmp_path / "web_token.txt"

    first = load_or_create_token(path)
    second = load_or_create_token(path)

    assert first == second
    assert len(first) > 20


def test_load_or_create_token_generates_different_tokens_for_different_files(tmp_path):
    token_a = load_or_create_token(tmp_path / "a.txt")
    token_b = load_or_create_token(tmp_path / "b.txt")

    assert token_a != token_b


# -- mobile page / PWA (see neo/web/mobile_ui.py) ----------------------------


def test_mobile_page_is_served_without_a_token():
    """A plain Safari navigation can't attach the x-neo-token header, so the
    page shell itself must not require one -- only the actual actions
    (/status, /message) stay gated."""
    client = _client(token="secret")

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "NEO" in response.text


def test_manifest_is_served_for_add_to_home_screen():
    client = _client(token="secret")

    response = client.get("/manifest.json")

    assert response.status_code == 200
    assert response.json()["name"] == "NEO"
    assert response.json()["icons"][0]["src"] == "/icon.png"


def test_service_worker_is_served():
    client = _client(token="secret")

    response = client.get("/sw.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


def test_icon_is_a_real_png():
    client = _client(token="secret")

    response = client.get("/icon.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
