"""Utility helpers for saving PDFs and transcribing their pages to Markdown."""

from __future__ import annotations

import base64
import io
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from ollama import OllamaClient

try:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
except ModuleNotFoundError:  # pragma: no cover - used only when streamlit is available
    UploadedFile = object  # type: ignore[assignment]


class DocumentParser:
    """Minimal PDF pipeline: save uploads, rasterize pages, send to Ollama."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        ollama_client: Optional[OllamaClient] = None,
    ) -> None:
        working_dir = base_dir or Path(tempfile.gettempdir()) / "aiworkflow_uploads"
        self.base_dir = Path(working_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_client = ollama_client

    def attach_ollama_client(self, client: OllamaClient) -> None:
        """Set or replace the Ollama client used for transcription."""

        self.ollama_client = client

    def save_pdf(self, uploaded_file: UploadedFile) -> Path:
        """Persist a single uploaded PDF and return the saved path."""

        if not hasattr(uploaded_file, "name") or not hasattr(uploaded_file, "getbuffer"):
            raise TypeError("Uploaded file must provide 'name' and 'getbuffer' attributes")

        original_name = Path(getattr(uploaded_file, "name") or "upload.pdf").name
        suffix = Path(original_name).suffix or ".pdf"
        destination = self.base_dir / f"{uuid.uuid4().hex}{suffix}"

        with destination.open("wb") as dest_file:
            dest_file.write(uploaded_file.getbuffer())

        return destination

    def save(self, files: Iterable[UploadedFile]) -> List[str]:
        """Persist multiple uploaded PDFs; convenience wrapper for Streamlit usage."""

        return [str(self.save_pdf(uploaded_file)) for uploaded_file in files]

    def pdf_to_images(self, pdf_path: Path | str) -> List[bytes]:
        """Convert each PDF page into a PNG byte payload."""

        pdf_location = Path(pdf_path)
        if not pdf_location.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_location}")

        try:
            from pdf2image import convert_from_path
        except ImportError as exc:  # pragma: no cover - dependency is optional at runtime
            raise RuntimeError("pdf2image is required for PDF to image conversion.") from exc

        image_payloads: List[bytes] = []
        for image in convert_from_path(str(pdf_location)):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_payloads.append(buffer.getvalue())

        return image_payloads

    def images_to_markdown(
        self,
        image_payloads: Sequence[bytes],
        *,
        model_name: Optional[str] = None,
    ) -> List[str]:
        """Send image byte payloads to the Ollama client and return Markdown per page."""

        if self.ollama_client is None:
            raise RuntimeError("Ollama client is not configured for DocumentParser")

        page_markdown: List[str] = []
        for index, image_bytes in enumerate(image_payloads, start=1):
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            markdown = self.ollama_client.image_to_markdown(
                image_b64,
                model_name=model_name,
                page_number=index,
            )
            page_markdown.append(markdown.strip())

        return page_markdown

    def pdf_to_markdown(
        self,
        pdf_path: Path | str,
        *,
        model_name: Optional[str] = None,
    ) -> List[str]:
        """High-level helper: rasterize a PDF then return Markdown per page."""

        images = self.pdf_to_images(pdf_path)
        return self.images_to_markdown(images, model_name=model_name)
