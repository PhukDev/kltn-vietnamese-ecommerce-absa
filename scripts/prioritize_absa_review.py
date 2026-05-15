from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.compat import disable_optional_pyarrow  # noqa: E402
from ecommerce_absa.config import ABSA_ASPECT_COLUMNS, MODELS_DIR  # noqa: E402
from ecommerce_absa.models.baselines import load_model_artifact  # noqa: E402
from ecommerce_absa.preprocessing import TextPreprocessor  # noqa: E402

from flag_sentiment_mismatches import build_flags  # type: ignore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prioritize ABSA rows for manual review")
    parser.add_argument("--input-dir", default="data/prefilled")
    parser.add_argument("--pattern", default="absa_annotation_batch_*.csv")
    parser.add_argument("--aspect-model", default=str(MODELS_DIR / "absa_aspect_baseline.joblib"))
    parser.add_argument("--output-dir", default="artifacts/reports/review_priority")
    args = parser.parse_args()

    disable_optional_pyarrow()
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy and pandas are required. Install dependencies with: pip install -r requirements.txt") from exc

    files = sorted(Path(args.input_dir).glob(args.pattern))
    if not files:
        raise SystemExit("No input files found")

    frames = []
    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["source_file"] = path.name
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)

    preprocessor = TextPreprocessor(use_word_segmentation=False)
    flags = build_flags(merged, preprocessor=preprocessor)[
        ["reviewid", "processed_text", "content_polarity", "mismatch_reason", "flagged"]
    ].copy()
    merged = merged.drop(columns=["content_polarity", "mismatch_reason", "flagged"], errors="ignore")
    merged = merged.merge(flags, on="reviewid", how="left")

    aspect_model, _ = load_model_artifact(args.aspect_model)
    processed_texts = merged["processed_text"].fillna("").astype(str).tolist()
    predictions = aspect_model.predict(processed_texts)
    decision_scores = aspect_model.decision_function(processed_texts)
    if getattr(decision_scores, "ndim", 1) == 1:
        decision_scores = decision_scores.reshape(-1, 1)

    merged["current_label_count"] = merged.apply(count_current_labels, axis=1)
    merged["model_label_count"] = [int(sum(int(value) for value in row)) for row in predictions]
    merged["model_aspects"] = [", ".join(labels_from_vector(row)) or "general" for row in predictions]
    merged["aspect_mismatch_count"] = [
        aspect_mismatch_count(merged.iloc[index], predictions[index]) for index in range(len(merged))
    ]
    merged["min_margin"] = [float(min(abs(value) for value in row)) for row in decision_scores]

    high_uncertainty_threshold = float(np.quantile(merged["min_margin"], 0.2))
    medium_uncertainty_threshold = float(np.quantile(merged["min_margin"], 0.4))

    scored = merged.apply(
        lambda row: score_row(
            row,
            high_uncertainty_threshold=high_uncertainty_threshold,
            medium_uncertainty_threshold=medium_uncertainty_threshold,
        ),
        axis=1,
        result_type="expand",
    )
    scored.columns = ["risk_score", "risk_level", "risk_reasons"]
    merged = pd.concat([merged, scored], axis=1)

    merged = merged.sort_values(
        by=["risk_score", "aspect_mismatch_count", "flagged", "min_margin"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_output = output_dir / "absa_review_priority_all.csv"
    high_output = output_dir / "absa_review_priority_high.csv"
    medium_output = output_dir / "absa_review_priority_medium.csv"
    low_output = output_dir / "absa_review_priority_low.csv"

    output_columns = [
        "source_file",
        "reviewid",
        "content",
        "sentiment",
        "product",
        "price",
        "delivery",
        "service",
        "app",
        "sentiment_original",
        "sentiment_changed",
        "content_polarity",
        "flagged",
        "model_aspects",
        "current_label_count",
        "model_label_count",
        "aspect_mismatch_count",
        "min_margin",
        "risk_score",
        "risk_level",
        "risk_reasons",
        "prefill_review_note",
    ]

    merged[output_columns].to_csv(all_output, index=False, encoding="utf-8-sig")
    merged[merged["risk_level"] == "high"][output_columns].to_csv(high_output, index=False, encoding="utf-8-sig")
    merged[merged["risk_level"] == "medium"][output_columns].to_csv(medium_output, index=False, encoding="utf-8-sig")
    merged[merged["risk_level"] == "low"][output_columns].to_csv(low_output, index=False, encoding="utf-8-sig")

    summary = {
        "rows_total": int(len(merged)),
        "rows_high": int((merged["risk_level"] == "high").sum()),
        "rows_medium": int((merged["risk_level"] == "medium").sum()),
        "rows_low": int((merged["risk_level"] == "low").sum()),
        "high_uncertainty_threshold": high_uncertainty_threshold,
        "medium_uncertainty_threshold": medium_uncertainty_threshold,
        "outputs": {
            "all": str(all_output),
            "high": str(high_output),
            "medium": str(medium_output),
            "low": str(low_output),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def count_current_labels(row) -> int:
    total = 0
    for aspect in ABSA_ASPECT_COLUMNS:
        total += int(str(row.get(aspect, "0")).strip() == "1")
    return total


def labels_from_vector(row) -> list[str]:
    return [aspect for index, aspect in enumerate(ABSA_ASPECT_COLUMNS) if int(row[index]) == 1]


def aspect_mismatch_count(row, predicted_row) -> int:
    mismatches = 0
    for index, aspect in enumerate(ABSA_ASPECT_COLUMNS):
        current = int(str(row.get(aspect, "0")).strip() == "1")
        predicted = int(predicted_row[index])
        if current != predicted:
            mismatches += 1
    return mismatches


def score_row(row, *, high_uncertainty_threshold: float, medium_uncertainty_threshold: float):
    score = 0
    reasons: list[str] = []

    if bool(row.get("flagged")):
        score += 3
        reasons.append("sentiment mismatch was auto-flagged")

    mismatch_count = int(row.get("aspect_mismatch_count", 0))
    if mismatch_count >= 2:
        score += 3
        reasons.append(f"aspect labels disagree with model on {mismatch_count} columns")
    elif mismatch_count == 1:
        score += 2
        reasons.append("aspect labels disagree with model on 1 column")

    min_margin = float(row.get("min_margin", 0.0))
    if min_margin <= high_uncertainty_threshold:
        score += 2
        reasons.append("model confidence is low")
    elif min_margin <= medium_uncertainty_threshold:
        score += 1
        reasons.append("model confidence is moderate")

    current_label_count = int(row.get("current_label_count", 0))
    if current_label_count >= 3:
        score += 1
        reasons.append("multi-aspect review")

    if current_label_count == 0 and str(row.get("content_polarity", "")).strip().lower() != "unclear":
        score += 1
        reasons.append("review has clear sentiment but no aspect labels")

    if str(row.get("service", "0")).strip() == "1" and str(row.get("app", "0")).strip() == "1":
        score += 1
        reasons.append("service and app co-occur; common confusion pair")

    if bool(row.get("sentiment_changed")):
        score += 1
        reasons.append("sentiment was auto-adjusted")

    if score >= 5:
        level = "high"
    elif score >= 3:
        level = "medium"
    else:
        level = "low"
    return score, level, "; ".join(reasons)


if __name__ == "__main__":
    main()
