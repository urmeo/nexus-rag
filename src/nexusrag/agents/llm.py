"""LLM client for Ollama API."""

import json
from collections.abc import Generator
from dataclasses import dataclass

import httpx


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
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Lazy HTTP client initialization."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt
            system: Optional system message
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Returns:
            Generated text
        """
        payload = {
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

        response = self.client.post("/api/generate", json=payload)
        response.raise_for_status()

        return str(response.json()["response"])

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """
        Stream text completion token by token.

        Yields:
            Generated tokens
        """
        payload = {
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

        with self.client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """
        Chat completion with message history.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Assistant response
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = self.client.post("/api/chat", json=payload)
        response.raise_for_status()

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
