"""LLM client for Ollama API."""

import json
import time
from collections.abc import Generator
from dataclasses import dataclass

import httpx


class LLMError(RuntimeError):
    """Raised when the LLM backend fails after retries."""


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.9
    stop: list[str] | None = None


class LLMClient:
    """Client for Ollama LLM API."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        max_retries: int = 2,
        backoff: float = 0.5,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    def _where(self) -> str:
        """Context string for error messages."""
        return f"model={self.model} base_url={self.base_url}"

    def _post(self, path: str, payload: dict[str, object]) -> httpx.Response:
        """POST with retry on transient errors and 5xx; no retry on 4xx."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(path, json=payload)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RequestError) as exc:
                last_exc = exc
            else:
                if response.status_code < 500:
                    response.raise_for_status()
                    return response
                last_exc = httpx.HTTPStatusError(
                    f"server error {response.status_code}",
                    request=response.request,
                    response=response,
                )
            if attempt < self.max_retries:
                time.sleep(self.backoff * (2**attempt))
        raise LLMError(f"LLM request to {path} failed ({self._where()}): {last_exc}") from last_exc

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
            },
        }

        if system:
            payload["system"] = system

        if stop:
            payload["options"]["stop"] = stop  # type: ignore[index]

        response = self._post("/api/generate", payload)
        return str(response.json()["response"])

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system:
            payload["system"] = system

        # Retrying a partially-consumed stream is unsafe, so we only
        # classify and wrap connection/timeout errors as LLMError.
        try:
            with self.client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RequestError) as exc:
            raise LLMError(f"LLM stream failed ({self._where()}): {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM stream failed ({self._where()}): {exc}") from exc

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = self._post("/api/chat", payload)
        return str(response.json()["message"]["content"])

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = self.client.get("/api/tags")
            if response.status_code != 200:
                return False
            models = [m["name"] for m in response.json().get("models", [])]
            return any(self.model in m for m in models)
        except httpx.RequestError:
            return False

    def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        """Ensure HTTP client is closed on garbage collection."""
        self.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
