# eligibility.py
"""Loan eligibility decision logic powered by the trained decision tree model."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Literal, Optional

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
