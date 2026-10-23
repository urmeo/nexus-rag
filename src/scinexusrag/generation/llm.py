"""Ollama client."""

import json
import time
from collections.abc import Generator
from contextlib import suppress
from sys import float_info
from threading import Lock

import httpx


class LLMError(RuntimeError):
    """Backend failure."""


class LLMClient:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        max_retries: int = 2,
        backoff: float = 0.5,
    ):
        self._client: httpx.Client | None = None
        self._client_lock = Lock()
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= float_info.max
        ):
            raise ValueError("timeout must be finite and positive")
        if (
            isinstance(backoff, bool)
            or not isinstance(backoff, (int, float))
            or not 0 <= backoff <= float_info.max
        ):
            raise ValueError("backoff must be finite and nonnegative")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    @property
    def client(self) -> httpx.Client:
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(self.timeout),
                )
            return self._client

    def _where(self) -> str:
        endpoint = httpx.URL(self.base_url).copy_with(
            userinfo=b"", path="", query=None, fragment=None
        )
        return f"model={self.model} base_url={endpoint}"

    def _check_status(self, response: httpx.Response, path: str) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise httpx.HTTPStatusError(
                f"LLM request to {path} returned HTTP {response.status_code} ({self._where()})",
                request=response.request,
                response=response,
            ) from None

    def _post(self, path: str, payload: dict[str, object]) -> httpx.Response:
        failure = "request failed"
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(path, json=payload)
            except httpx.TimeoutException:
                raise LLMError(f"LLM timed out after {self.timeout}s ({self._where()})") from None
            except httpx.RequestError as exc:
                failure = type(exc).__name__
            else:
                if response.status_code < 500:
                    self._check_status(response, path)
                    return response
                failure = f"HTTP {response.status_code}"
            if attempt < self.max_retries:
                time.sleep(self.backoff * (2**attempt))
        raise LLMError(f"LLM request to {path} failed ({self._where()}): {failure}") from None

    def _generation_payload(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
        *,
        stream: bool,
        stop: list[str] | None = None,
    ) -> dict[str, object]:
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
            "stream": stream,
            "options": options,
        }

        if system:
            payload["system"] = system
        return payload

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> str:
        payload = self._generation_payload(
            prompt, system, temperature, max_tokens, stream=False, stop=stop
        )
        response = self._post("/api/generate", payload)
        try:
            data = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise LLMError(f"LLM returned non-JSON body ({self._where()})") from None
        if isinstance(data, dict):
            if data.get("error"):
                raise LLMError(
                    f"LLM generation failed: backend reported an error ({self._where()})"
                )
            if "done" in data:
                if not isinstance(data["done"], bool):
                    raise LLMError(f"LLM response 'done' field must be a boolean ({self._where()})")
                if not data["done"]:
                    raise LLMError(f"LLM response ended before completion ({self._where()})")
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
        payload = self._generation_payload(prompt, system, temperature, max_tokens, stream=True)

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
                            raise LLMError(
                                f"LLM stream failed: backend reported an error ({self._where()})"
                            )
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
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"LLM stream failed with HTTP {exc.response.status_code} ({self._where()})"
            ) from None
        except (httpx.RequestError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM stream failed: {type(exc).__name__} ({self._where()})") from None

    def is_available(self) -> bool:
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
        except (httpx.HTTPError, json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
            return False

    def close(self) -> None:
        with self._client_lock:
            client, self._client = self._client, None
            if client is not None:
                client.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
