"""Utility for handling document uploads in the intake app."""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, List, Optional
from urllib import error, request

from openai import OpenAI

try:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
except ModuleNotFoundError:  # pragma: no cover - used only when streamlit is available
    UploadedFile = object  # type: ignore[assignment]


def _resolve_base_url(base_url: Optional[str]) -> str:
    return (base_url or os.getenv("OLLAMA_BASE_URL") or "http://ollama:11434").rstrip("/")


def _stream_ollama_request(
    *,
    base_url: str,
    path: str,
    payload: dict,
    timeout: int,
) -> List[dict]:
    url = f"{base_url}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
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
                messages.append(json.loads(line))
    except error.URLError as exc:
        raise RuntimeError(f"Failed request to Ollama at {url}: {exc}") from exc

    return messages


def setup_ollama_model(
    base_url: Optional[str] = None,
    model_name: str = "granite3.2-vision",
) -> None:
    """Ensure the requested Ollama model is available by pulling it if missing."""

    resolved_base = _resolve_base_url(base_url)

    tags_url = f"{resolved_base}/api/tags"
    try:
        with request.urlopen(tags_url, timeout=10) as response:
            payload = json.load(response)
    except error.URLError as exc:
        raise RuntimeError(f"Unable to reach Ollama at {resolved_base}: {exc}") from exc

    models = {entry.get("name") for entry in payload.get("models", [])}
    if model_name in models:
        return

    _stream_ollama_request(
        base_url=resolved_base,
        path="/api/pull",
        payload={"name": model_name},
        timeout=300,
    )


def _get_openai_client(base_url: str) -> OpenAI:
    api_key = os.getenv("OLLAMA_API_KEY") or "ollama"
    return OpenAI(base_url=f"{base_url}/v1", api_key=api_key)


class DocumentParser:
    """Save uploaded documents to a temporary workspace directory and parse them."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        base_dir = base_dir or Path(tempfile.gettempdir()) / "aiworkflow_uploads"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, files: Iterable[UploadedFile]) -> List[str]:
        """Persist uploaded files and return their absolute paths."""
        saved_paths: List[str] = []
        for uploaded_file in files:
            safe_name = Path(uploaded_file.name).name
            unique_name = f"{uuid.uuid4().hex}_{safe_name}"
            destination = self.base_dir / unique_name
            with destination.open("wb") as dest_file:
                dest_file.write(uploaded_file.getbuffer())
            saved_paths.append(str(destination))
        return saved_paths

    def pdf_pages_to_markdown(
        self,
        pdf_path: Path | str,
        base_url: Optional[str] = None,
        model_name: str = "granite3.2-vision",
    ) -> List[str]:
        """Split a PDF into pages, use Ollama vision to transcribe each, and return Markdown strings."""

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            from pdf2image import convert_from_path
        except ImportError as exc:  # pragma: no cover - dependency is optional at runtime
            raise RuntimeError("pdf2image is required for PDF to image conversion.") from exc

        images = convert_from_path(str(pdf_path))
        resolved_base = _resolve_base_url(base_url)
        client = _get_openai_client(resolved_base)

        page_markdown: List[str] = []
        for index, image in enumerate(images, start=1):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            markdown = _generate_markdown_from_image(
                client=client,
                model_name=model_name,
                image_b64=image_b64,
                page_number=index,
            )
            page_markdown.append(markdown.strip())

        return page_markdown


def _generate_markdown_from_image(
    *,
    client: OpenAI,
    model_name: str,
    image_b64: str,
    page_number: int,
) -> str:
    data_url = f"data:image/png;base64,{image_b64}"
    try:
        response = client.chat.completions.create(
            model=model_name,
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
        raise RuntimeError(f"Ollama chat completion failed on page {page_number}: {exc}") from exc

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError(f"No completion returned for page {page_number}")

    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message else None
    if not content:
        raise RuntimeError(f"Empty completion content for page {page_number}")

    return content
