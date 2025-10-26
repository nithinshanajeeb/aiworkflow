# eligibility.py
"""Loan eligibility decision logic powered by the trained decision tree model."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Literal, Optional, Sequence

from ollama import OllamaClient

try:  # Python 3.8 compatibility in case functools.cached_property unavailable
    from functools import cached_property
except ImportError:  # pragma: no cover - fallback for older Python
    from functools import lru_cache

    def cached_property(func):  # type: ignore[misc]
        return property(lru_cache(maxsize=None)(func))

Status = Literal["approve", "approve_conditional", "soft_decline", "insufficient"]

PredictFn = Callable[[Optional[float], Optional[float]], Dict[str, Any]]

MODULE_NAME = "eligibility_decision_tree_model"


class EligibilityPredictor:
    """Predict loan eligibility decisions using the trained decision tree model."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model_path = model_path or self._default_model_path()

    def predict(
        self,
        credit_score: Optional[int | float],
        monthly_income_aed: Optional[int | float],
    ) -> Dict[str, Any]:
        """Return decision dict with status, reasons, and a confidence hint."""

        predict_decision = self._predict_callable
        credit_value = self._coerce_optional_float(credit_score)
        income_value = self._coerce_optional_float(monthly_income_aed)
        return predict_decision(credit_value, income_value)

    @cached_property
    def _predict_callable(self) -> PredictFn:
        module = self._model_module
        predict_decision = getattr(module, "predict_decision", None)
        if not callable(predict_decision):
            raise AttributeError(
                "Loaded model module does not expose a callable 'predict_decision'"
            )
        return predict_decision

    @cached_property
    def _model_module(self) -> ModuleType:
        spec = self._create_module_spec()
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Unable to load decision tree model module from {self._model_path}"
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def _create_module_spec(self) -> ModuleSpec | None:
        module_path = self._model_path
        return importlib.util.spec_from_file_location(MODULE_NAME, module_path)

    @staticmethod
    def _coerce_optional_float(value: Optional[int | float]) -> Optional[float]:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _default_model_path() -> Path:
        return Path(__file__).resolve().parent / "models" / "decision-tree" / "model.py"


_DEFAULT_PREDICTOR = EligibilityPredictor()


def decide_eligibility(credit_score: int | None, monthly_income_aed: float | None) -> Dict[str, Any]:
    """Backward-compatible helper for existing code paths."""

    return _DEFAULT_PREDICTOR.predict(credit_score, monthly_income_aed)


def verify_document_consistency(
    ollama_client: OllamaClient | None,
    applicant_profile: Dict[str, Any],
    documents: Sequence[Dict[str, Any]],
    *,
    model_name: Optional[str] = None,
) -> str:
    """Use an LLM to compare applicant-provided fields against statement transcripts."""

    if ollama_client is None:
        raise RuntimeError("Ollama client is not configured for fraud verification")
    if not documents:
        raise ValueError("No document transcripts available for verification")

    safe_documents: list[Dict[str, Any]] = [dict(document) for document in documents]

    def _format_value(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value)
        return str(value)

    profile_lines = []
    for key, value in sorted(applicant_profile.items()):
        label = key.replace("_", " ").strip().title()
        if not label:
            continue
        profile_lines.append(f"- {label}: {_format_value(value)}")
    profile_section = "\n".join(profile_lines) or "- No applicant profile data supplied"

    document_sections: list[str] = []
    for document in safe_documents:
        source = document.get("source") or "uploaded_document.pdf"
        document_name = Path(str(source)).name
        pages = document.get("pages") or []
        if not isinstance(pages, Sequence):
            continue

        page_chunks: list[str] = []
        for index, page in enumerate(pages, start=1):
            if not page:
                continue
            text = str(page).strip()
            if not text:
                continue
            max_chars = 2000
            snippet = text[:max_chars]
            if len(text) > max_chars:
                snippet = f"{snippet}\n[…truncated…]"
            page_chunks.append(f"Page {index}:\n{snippet}")

        if page_chunks:
            document_sections.append(
                f"Document: {document_name}\n" + "\n\n".join(page_chunks)
            )

    document_section = "\n\n".join(document_sections) or "No readable text was extracted from the documents."

    system_prompt = (
        "You are a senior compliance analyst verifying supporting documents for loan applications. "
        "Cross-check applicant declarations against bank statement transcripts to detect fraud or inconsistencies."
    )

    user_prompt = (
        "Verify whether the applicant-provided information matches the supplied bank statement transcripts.\n\n"
        "Applicant Profile:\n"
        f"{profile_section}\n\n"
        "Bank Statement Transcripts:\n"
        f"{document_section}\n\n"
        "Analysis Instructions:\n"
        "1. Compare income amounts, employer details, account holder names, dates, and identification numbers wherever possible.\n"
        "2. Cite the document name and page whenever you confirm or dispute a value.\n"
        "3. Flag any discrepancies, missing evidence, or suspicious anomalies.\n"
        "4. Assign a fraud risk rating (Low, Medium, High) with a concise rationale.\n"
        "5. Recommend follow-up actions if risk is Medium or High.\n"
        "6. If information is missing or unclear, state that explicitly instead of guessing.\n\n"
        "Respond in Markdown using this structure:\n"
        "## Overall Verdict\n"
        "- Determination: <Match / Partial Match / Mismatch / Inconclusive>\n"
        "- Fraud Risk: <Low / Medium / High> — <brief rationale>\n\n"
        "## Confirmed Matches\n"
        "- <Field>: <Applicant value> — cite document and page; use 'None observed' if nothing confirmed.\n\n"
        "## Discrepancies\n"
        "- <Field>: describe the issue and cite evidence; use 'None observed' if none.\n\n"
        "## Recommended Actions\n"
        "- List specific next steps or 'No further action' if appropriate.\n\n"
        "Do not invent data; rely solely on the provided transcripts."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return ollama_client.chat_completion(
        messages,
        model_name=model_name,
        temperature=0.0,
    )
