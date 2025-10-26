"""Eligibility decision tree trained on credit score and monthly income.

This module exposes a small `sklearn.tree.DecisionTreeClassifier` that mimics
the rule-based eligibility logic. Besides providing the `predict_decision`
helper for inference, it also offers a command-line training pipeline capable of
loading tabular data, fitting the model, and persisting the trained artifact.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.tree import DecisionTreeClassifier

try:  # joblib is preferred for persistence but optional for inference.
    from joblib import dump, load
    _JOBLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - joblib provided via requirements.
    dump = load = None
    _JOBLIB_AVAILABLE = False

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:  # Ensure local imports work when run as script.
    sys.path.insert(0, str(MODULE_DIR))

try:  # Prefer package-relative import when available.
    from .data import DEFAULT_THRESHOLDS, Thresholds, generate_dataset
except ImportError:  # pragma: no cover - allows running as a script.
    from data import DEFAULT_THRESHOLDS, Thresholds, generate_dataset

# Thresholds used across the model and metadata.
TRAINING_THRESHOLDS = DEFAULT_THRESHOLDS

FEATURE_NAMES: Sequence[str] = ("credit_score", "monthly_income_aed")
CLASS_NAMES: Sequence[str] = ("approve", "approve_conditional", "soft_decline")
DEFAULT_MODEL_PATH = Path(__file__).with_name("trained_decision_tree.joblib")

LOGGER = logging.getLogger(__name__)

try:  # Optional progress bar support.
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


@dataclass(frozen=True)
class EligibilityDecision:
    """Container for decision outcomes matching the legacy interface."""

    status: str
    reasons: List[str]
    confidence: float

    def as_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "reasons": self.reasons, "confidence": self.confidence}


def train_decision_tree(
    random_state: int = 42,
    max_depth: int | None = 3,
    thresholds: Thresholds = TRAINING_THRESHOLDS,
    *,
    features: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> DecisionTreeClassifier:
    """Train and return a decision tree classifier on provided (or synthetic) data."""

    if features is None or labels is None:
        LOGGER.info("Training with synthetic dataset using thresholds: %s", thresholds)
        features, labels, _ = generate_dataset(thresholds=thresholds)

    LOGGER.info("Fitting decision tree (samples=%d, features=%d)", len(features), features.shape[1])
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, class_weight="balanced")
    clf.fit(features, labels)
    LOGGER.info("Training completed")
    return clf


def load_training_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load training features and labels from a CSV with expected columns."""

    LOGGER.info("Loading training data from %s", path)
    with path.open("r", newline="") as csvfile:
        total_rows = max(sum(1 for _ in csvfile) - 1, 0)

    rows: list[list[float]] = []
    targets: list[str] = []
    with path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset at {path} is missing a header row.")

        required_columns = set(FEATURE_NAMES) | {"status"}
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Dataset at {path} is missing columns: {', '.join(sorted(missing))}")

        iterator: Iterable[dict[str, str]] = reader
        progress_interval = None
        if tqdm is not None and total_rows > 0:
            iterator = tqdm(reader, total=total_rows, desc="Loading data", unit="rows")
        elif total_rows > 0:
            progress_interval = max(total_rows // 10, 1)

        for line_number, row in enumerate(iterator, start=2):
            try:
                feature_row = [float(row[name]) for name in FEATURE_NAMES]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid feature value at line {line_number} in {path}: {exc}") from exc
            status = row["status"]
            rows.append(feature_row)
            targets.append(status)

            if tqdm is None and progress_interval and (line_number - 1) % progress_interval == 0:
                LOGGER.info("Loaded %d/%d rows", line_number - 1, total_rows)

    if not rows:
        raise ValueError(f"Dataset at {path} contains no rows to train on.")

    LOGGER.info("Finished loading %d rows", len(rows))
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=str)


def save_model(model: DecisionTreeClassifier, model_path: Path) -> Path:
    """Persist the fitted model to disk and return the path."""

    if not _JOBLIB_AVAILABLE:
        raise ImportError(
            "joblib is required to save models. Install it with `pip install joblib`."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    LOGGER.info("Model saved to %s", model_path)
    return model_path


def load_model(model_path: Path) -> DecisionTreeClassifier:
    """Load a serialized decision tree classifier from disk."""

    if not _JOBLIB_AVAILABLE:
        raise ImportError(
            "joblib is required to load persisted models. Install it with `pip install joblib`."
        )

    LOGGER.info("Loading model from %s", model_path)
    return load(model_path)


def _initialize_model() -> DecisionTreeClassifier:
    """Load a persisted model if present; otherwise fall back to synthetic training."""

    if DEFAULT_MODEL_PATH.exists():
        if _JOBLIB_AVAILABLE:
            LOGGER.info("Found existing model artifact at %s", DEFAULT_MODEL_PATH)
            return load_model(DEFAULT_MODEL_PATH)
        LOGGER.warning(
            "Model artifact exists at %s but joblib is unavailable; training fresh model instead.",
            DEFAULT_MODEL_PATH,
        )

    LOGGER.info("No persisted model found, training a new one with synthetic data")
    return train_decision_tree()


# Train once at import time so downstream code can simply call `predict_decision`.
MODEL = _initialize_model()


def predict_decision(
    credit_score: float | None,
    monthly_income_aed: float | None,
    model: DecisionTreeClassifier | None = None,
) -> Dict[str, Any]:
    """Replicate the `eligibility.decide_eligibility` response using the tree."""

    if credit_score is None or monthly_income_aed is None:
        return {
            "status": "insufficient",
            "reasons": ["Credit score and monthly income are required for a decision."],
            "confidence": 0.0,
        }

    clf = model or MODEL
    features = np.asarray([[credit_score, monthly_income_aed]], dtype=float)
    status = clf.predict(features)[0]

    # Use class probability as a soft confidence score with a fallback map.
    probable_classes = clf.classes_.tolist()
    proba = clf.predict_proba(features)[0]
    status_index = probable_classes.index(status)
    confidence = float(np.round(proba[status_index], 2))

    strong_credit = int(TRAINING_THRESHOLDS.strong_credit)
    min_credit = int(TRAINING_THRESHOLDS.min_credit)
    strong_income = int(TRAINING_THRESHOLDS.strong_income)
    min_income = int(TRAINING_THRESHOLDS.min_income)

    if status == "approve":
        reasons = [f"Credit score ≥ {strong_credit} and income ≥ AED {strong_income}."]
    elif status == "approve_conditional":
        reasons = [f"Credit score ≥ {min_credit} and income ≥ AED {min_income} (minimum thresholds)."]
    else:
        reasons = []
        if credit_score < TRAINING_THRESHOLDS.min_credit:
            reasons.append(f"Credit score {credit_score:.0f} < {min_credit} minimum.")
        if monthly_income_aed < TRAINING_THRESHOLDS.min_income:
            reasons.append(f"Monthly income AED {monthly_income_aed:.0f} < AED {min_income} minimum.")

    decision = EligibilityDecision(status=status, reasons=reasons, confidence=confidence)
    return decision.as_dict()


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for training and loading the model."""

    parser = argparse.ArgumentParser(
        description="Train the eligibility decision tree or load an existing model.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="CSV file with columns credit_score, monthly_income_aed, status for training.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Where to write the trained model artifact (joblib format).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Existing trained model to load instead of training from data.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Decision tree maximum depth (default: 3).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state seed for training (default: 42).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> DecisionTreeClassifier:
    """Entry point for command-line model training and loading."""

    setup_cli_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.model_path is not None:
        model = load_model(args.model_path)
        LOGGER.info("Model loaded from %s", args.model_path)
        return model

    if args.data_path is None or args.model_output is None:
        parser.error("--data-path and --model-output are required when --model-path is not provided.")

    features, labels = load_training_data(args.data_path)
    model = train_decision_tree(
        random_state=args.random_state,
        max_depth=args.max_depth,
        features=features,
        labels=labels,
    )
    save_path = save_model(model, args.model_output)
    LOGGER.info("Model trained and saved to %s", save_path)
    return model


def setup_cli_logging(level: int = logging.INFO) -> None:
    """Configure logging when the module is used as a script."""

    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


__all__ = [
    "CLASS_NAMES",
    "FEATURE_NAMES",
    "EligibilityDecision",
    "DEFAULT_MODEL_PATH",
    "MODEL",
    "LOGGER",
    "build_parser",
    "load_model",
    "load_training_data",
    "predict_decision",
    "setup_cli_logging",
    "save_model",
    "train_decision_tree",
    "main",
]


if __name__ == "__main__":
    main()
