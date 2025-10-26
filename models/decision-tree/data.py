"""Synthetic dataset utilities for the eligibility decision tree."""

from __future__ import annotations

import csv
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    strong_credit: float
    min_credit: float
    strong_income: float
    min_income: float


DEFAULT_THRESHOLDS = Thresholds(strong_credit=700, min_credit=650, strong_income=5000, min_income=3000)
DATASET_CSV_PATH = Path(__file__).with_name("synthetic_training_data.csv")


def label_status(credit_score: float, monthly_income_aed: float, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> str:
    if credit_score >= thresholds.strong_credit and monthly_income_aed >= thresholds.strong_income:
        return "approve"
    if credit_score >= thresholds.min_credit and monthly_income_aed >= thresholds.min_income:
        return "approve_conditional"
    return "soft_decline"


def generate_feature_grid(
    credit_scores: Iterable[float] | None = None,
    monthly_incomes: Iterable[float] | None = None,
    *,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Tuple[np.ndarray, np.ndarray]:
    if credit_scores is None:
        credit_scores = np.linspace(300, 900, num=25)
    if monthly_incomes is None:
        monthly_incomes = np.linspace(0, 15000, num=25)

    rows: list[list[float]] = []
    labels: list[str] = []
    for cs in credit_scores:
        for income in monthly_incomes:
            rows.append([cs, income])
            labels.append(label_status(cs, income, thresholds))

    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=str)


def generate_dataset(
    *,
    credit_scores: Iterable[float] | None = None,
    monthly_incomes: Iterable[float] | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    path: Path | None = DATASET_CSV_PATH,
) -> Tuple[np.ndarray, np.ndarray, Thresholds]:
    features, labels = generate_feature_grid(
        credit_scores=credit_scores,
        monthly_incomes=monthly_incomes,
        thresholds=thresholds,
    )
    if path is not None:
        _write_dataset_csv(features, labels, path)
    return features, labels, thresholds


def _write_dataset_csv(features: np.ndarray, labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["credit_score", "monthly_income_aed", "status"])
        for (credit_score, income), status in zip(features, labels):
            writer.writerow([f"{credit_score:.2f}", f"{income:.2f}", status])


__all__: Sequence[str] = (
    "Thresholds",
    "DEFAULT_THRESHOLDS",
    "DATASET_CSV_PATH",
    "generate_dataset",
    "generate_feature_grid",
    "label_status",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic eligibility training data CSV.")
    parser.add_argument(
        "--credit-range",
        nargs=3,
        metavar=("MIN", "MAX", "STEPS"),
        type=float,
        help="Override credit score range as min max steps (inclusive linspace).",
    )
    parser.add_argument(
        "--income-range",
        nargs=3,
        metavar=("MIN", "MAX", "STEPS"),
        type=float,
        help="Override monthly income range as min max steps (inclusive linspace).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_CSV_PATH,
        help="Path for the generated CSV (defaults to synthetic_training_data.csv alongside this file).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)

    credit_scores = None
    if args.credit_range:
        cmin, cmax, csteps = args.credit_range
        credit_scores = np.linspace(cmin, cmax, num=int(csteps))

    monthly_incomes = None
    if args.income_range:
        imin, imax, isteps = args.income_range
        monthly_incomes = np.linspace(imin, imax, num=int(isteps))

    output_path: Path = args.output
    generate_dataset(
        credit_scores=credit_scores,
        monthly_incomes=monthly_incomes,
        path=output_path,
    )

    return output_path


if __name__ == "__main__":
    output = main()
    print(f"Synthetic dataset written to {output}")
