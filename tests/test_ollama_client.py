import httpx
import pytest

from lib.ollama_client import OllamaClient, OllamaError

OriginalClient = httpx.Client


def make_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)

    def mock_client(**kwargs):
        return OriginalClient(
            base_url=kwargs["base_url"],
            timeout=kwargs["timeout"],
            transport=transport,
        )

    monkeypatch.setattr(
        "lib.ollama_client.httpx.Client",
        mock_client,
    )

    return OllamaClient()


def test_health(monkeypatch):
    client = make_client(
        monkeypatch,
        lambda request: httpx.Response(200, text="Ollama is running"),
    )
    try:
        assert client.health()
    finally:
        client.close()


def test_list_models(monkeypatch):
    client = make_client(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"models": [{"name": "qwen3:8b"}]}
        ),
    )
    try:
        assert client.list_models()[0]["name"] == "qwen3:8b"
    finally:
        client.close()


def test_generate(monkeypatch):
    client = make_client(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"response": "CareerPilot AI"}
        ),
    )
    try:
        assert client.generate("Explain AI") == "CareerPilot AI"
    finally:
        client.close()


def test_chat(monkeypatch):
    client = make_client(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"message": {"content": "Hello!"}}
        ),
    )
    try:
        assert client.chat(
            [{"role": "user", "content": "Hi"}]
        ) == "Hello!"
    finally:
        client.close()


def test_empty_prompt():
    with OllamaClient() as client:
        with pytest.raises(ValueError):
            client.generate(" ")


def test_empty_messages():
    with OllamaClient() as client:
        with pytest.raises(ValueError):
            client.chat([])


def test_connection_error(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("Connection failed")

    client = make_client(monkeypatch, handler)
    try:
        with pytest.raises(OllamaError, match="Cannot connect"):
            client.list_models()
    finally:
        client.close()


def test_timeout(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("Timed out")

    client = make_client(monkeypatch, handler)
    try:
        with pytest.raises(OllamaError, match="timed out"):
            client.list_models()
    finally:
        client.close()


def test_http_error(monkeypatch):
    client = make_client(
        monkeypatch,
        lambda request: httpx.Response(500, text="Internal error"),
    )
    try:
        with pytest.raises(OllamaError, match="HTTP 500"):
            client.list_models()
    finally:
        client.close()


def test_invalid_json(monkeypatch):
    client = make_client(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            content=b"invalid-json",
            headers={"Content-Type": "application/json"},
        ),
    )
    try:
        with pytest.raises(OllamaError, match="invalid JSON"):
            client.list_models()
    finally:
        client.close()
