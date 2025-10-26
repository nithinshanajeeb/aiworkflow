"""Client wrapper for interacting with an Ollama server."""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Sequence
from urllib import error, request

from openai import OpenAI

try:
    from langfuse.openai import OpenAI as LangfuseOpenAI
except Exception:  # noqa: BLE001
    LangfuseOpenAI = None

DEFAULT_MODEL_NAME = "granite3.2-vision"


class OllamaClient:
    """Lightweight helper for model management and multimodal requests."""

    default_model: str = DEFAULT_MODEL_NAME

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://ollama:11434").rstrip("/")
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY") or "ollama"
        self.default_model = default_model
        self._openai_client: Optional[Any] = None
        self._langfuse_enabled = self._detect_langfuse()

    def ensure_model(self, model_name: Optional[str] = None) -> None:
        """Pull the model if it is not available on the Ollama server."""

        name = model_name or self.default_model
        tags_url = f"{self.base_url}/api/tags"
        try:
            with request.urlopen(tags_url, timeout=10) as response:
                payload = json.load(response)
        except error.URLError as exc:
            raise RuntimeError(f"Unable to reach Ollama at {self.base_url}: {exc}") from exc

        models = {entry.get("name") for entry in payload.get("models", [])}
        if name in models:
            return

        self._stream_post("/api/pull", {"name": name}, timeout=300, error_context=f"Failed to pull model '{name}'")

    def image_to_markdown(
        self,
        image_b64: str,
        *,
        model_name: Optional[str] = None,
        page_number: Optional[int] = None,
    ) -> str:
        """Send a base64 encoded image to the vision model and return Markdown output."""

        model = model_name or self.default_model
        client = self._get_openai_client()
        data_url = f"data:image/png;base64,{image_b64}"

        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that transcribes documents into concise Markdown.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract the text from this page and respond in Markdown."},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    },
                ],
            )
        except Exception as exc:  # noqa: BLE001
            location = f" on page {page_number}" if page_number is not None else ""
            raise RuntimeError(f"Ollama chat completion failed{location}: {exc}") from exc

        choices = getattr(response, "choices", None) or []
        if not choices:
            location = f" for page {page_number}" if page_number is not None else ""
            raise RuntimeError(f"No completion returned{location}")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None
        if not content:
            location = f" for page {page_number}" if page_number is not None else ""
            raise RuntimeError(f"Empty completion content{location}")

        return content

    def chat_completion(
        self,
        messages: Sequence[dict],
        *,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Run a standard chat completion request and return Markdown/plain text."""

        model = model_name or self.default_model
        client = self._get_openai_client()

        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=list(messages),
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Ollama chat completion failed: {exc}") from exc

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("No completion returned")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None
        if not content:
            raise RuntimeError("Empty completion content")

        return content.strip()

    def _stream_post(self, path: str, payload: dict, *, timeout: int, error_context: str) -> List[dict]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        messages: List[dict] = []
        try:
            with request.urlopen(req, timeout=timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    message = json.loads(line)
                    if message.get("error"):
                        raise RuntimeError(f"{error_context}: {message['error']}")
                    messages.append(message)
        except error.URLError as exc:
            raise RuntimeError(f"{error_context}: {exc}") from exc

        return messages

    def _get_openai_client(self) -> OpenAI:
        if self._openai_client is None:
            client_kwargs = {
                "base_url": f"{self.base_url}/v1",
                "api_key": self.api_key,
            }
            if self._langfuse_enabled and LangfuseOpenAI is not None:
                self._openai_client = LangfuseOpenAI(**client_kwargs)
            else:
                self._openai_client = OpenAI(**client_kwargs)
        return self._openai_client

    def _detect_langfuse(self) -> bool:
        """Return True when Langfuse instrumentation is configured."""

        required = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
        return all(os.getenv(var) for var in required)
