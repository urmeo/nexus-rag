"""LLM client for Ollama API."""

import json
import time
from collections.abc import Generator

import httpx


class LLMError(RuntimeError):
    """LLM backend or response failure."""


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
        """Retry request errors and HTTP 5xx; fail immediately on timeouts and HTTP 4xx."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(path, json=payload)
            except httpx.TimeoutException as exc:
                raise LLMError(f"LLM timed out after {self.timeout}s ({self._where()})") from exc
            except httpx.RequestError as exc:
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
        options: dict[str, object] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.9,
        }
        if stop:
            options["stop"] = stop

        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        if system:
            payload["system"] = system

        response = self._post("/api/generate", payload)
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned non-JSON body ({self._where()})") from exc
        answer = data.get("response") if isinstance(data, dict) else None
        if not isinstance(answer, str):
            raise LLMError(f"LLM response must contain a string 'response' field ({self._where()})")
        if not answer.strip():
            raise LLMError(f"LLM returned an empty response ({self._where()})")
        return answer

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

        has_text = False
        try:
            with self.client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if not isinstance(data, dict):
                            raise LLMError(f"LLM stream expected a JSON object ({self._where()})")
                        if data.get("error"):
                            raise LLMError(f"LLM stream failed ({self._where()}): {data['error']}")
                        done = data.get("done", False)
                        if not isinstance(done, bool):
                            raise LLMError(
                                f"LLM stream 'done' field must be a boolean ({self._where()})"
                            )
                        if "response" in data:
                            token = data["response"]
                            if not isinstance(token, str):
                                raise LLMError(
                                    f"LLM stream 'response' field must be a string ({self._where()})"
                                )
                            has_text = has_text or bool(token.strip())
                            yield token
                        if done:
                            if not has_text:
                                raise LLMError(f"LLM returned an empty response ({self._where()})")
                            return
                raise LLMError(f"LLM stream ended before completion ({self._where()})")
        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM stream failed ({self._where()}): {exc}") from exc

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = self.client.get("/api/tags")
            if response.status_code != 200:
                return False
            data = response.json()
            if not isinstance(data, dict):
                return False
            models = [m["name"] for m in data.get("models", [])]
            model = self.model
            if ":" not in model.rsplit("/", 1)[-1]:
                model += ":latest"
            return self.model in models or model in models
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError):
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
