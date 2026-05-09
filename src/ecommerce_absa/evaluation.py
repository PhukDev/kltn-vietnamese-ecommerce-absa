from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def evaluate_classification(y_true, y_pred, labels: Sequence[str] | None = None) -> dict:
    try:
        from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required. Install dependencies with: pip install -r requirements.txt") from exc

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "precision_weighted": float(weighted_precision),
        "recall_weighted": float(weighted_recall),
        "f1_weighted": float(weighted_f1),
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
    }


def confusion_matrix_frame(y_true, y_pred, labels: Sequence[str]):
    try:
        import pandas as pd  # type: ignore
        from sklearn.metrics import confusion_matrix
    except ImportError as exc:
        raise RuntimeError("pandas and scikit-learn are required. Install dependencies with: pip install -r requirements.txt") from exc

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])


def write_json(payload: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
