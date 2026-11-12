"""Local Ollama inference client for LinkedIn CareerPilot AI."""

from __future__ import annotations

from typing import Any

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot complete a request."""


class OllamaClient:
    """Synchronous HTTP client for a locally running Ollama server."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a request and translate connection failures into OllamaError."""
        try:
            response = self._client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()

        except httpx.ConnectError as exc:
            raise OllamaError(
                "Cannot connect to Ollama. Check that Ollama is running."
            ) from exc

        except httpx.TimeoutException as exc:
            raise OllamaError(
                "Ollama request timed out. The model may need more time."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise OllamaError(
                f"Ollama returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:500]}"
            ) from exc

        except ValueError as exc:
            raise OllamaError("Ollama returned an invalid JSON response.") from exc

    def health(self) -> bool:
        """Check whether the Ollama server is responding."""
        try:
            response = self._client.get("/")
            response.raise_for_status()
            return True
        except (httpx.HTTPError, OllamaError):
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """Return models currently installed in Ollama."""
        data = self._request("GET", "/api/tags")
        return data.get("models", [])

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> str:
        """Generate text from a prompt using a local Ollama model."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
        }

        if system:
            payload["system"] = system

        if options:
            payload["options"] = options

        if think is not None:
            payload["think"] = think

        data = self._request("POST", "/api/generate", json=payload)
        return data.get("response", "")

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Run a multi-message conversation using Ollama."""
        if not messages:
            raise ValueError("At least one chat message is required.")

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
        }

        if options:
            payload["options"] = options

        data = self._request("POST", "/api/chat", json=payload)
        message = data.get("message", {})
        return message.get("content", "")
