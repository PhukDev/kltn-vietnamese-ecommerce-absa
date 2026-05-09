from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.config import DATA_PATH, REPORTS_DIR  # noqa: E402
from ecommerce_absa.data import load_reviews, score_to_sentiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight EDA for ecommerce reviews")
    parser.add_argument("--input", default=str(DATA_PATH))
    parser.add_argument("--limit", type=int, default=100000)
    args = parser.parse_args()

    columns = ["reviewid", "content", "score", "thumbsupcount", "reviewcreatedversion", "at", "replycontent", "appid"]
    frame = load_reviews(args.input, columns=columns, limit=args.limit)
    frame["sentiment"] = frame["score"].map(score_to_sentiment)
    frame["content_length"] = frame["content"].fillna("").astype(str).str.len()
    frame["has_reply"] = frame["replycontent"].fillna("").astype(str).str.strip().astype(bool)

    summary = {
        "rows_loaded": int(len(frame)),
        "columns": list(frame.columns),
        "score_distribution": _series_counts(frame["score"]),
        "sentiment_distribution": _series_counts(frame["sentiment"]),
        "appid_distribution": _series_counts(frame["appid"], top=20),
        "missing_values": {column: int(value) for column, value in frame.isna().sum().to_dict().items()},
        "thumbsupcount": {
            "min": float(frame["thumbsupcount"].fillna(0).min()),
            "mean": float(frame["thumbsupcount"].fillna(0).mean()),
            "max": float(frame["thumbsupcount"].fillna(0).max()),
        },
        "content_length": {
            "min": int(frame["content_length"].min()),
            "mean": float(frame["content_length"].mean()),
            "max": int(frame["content_length"].max()),
        },
        "reply_rate": float(frame["has_reply"].mean()),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "eda_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved EDA summary to {output_path}")


def _series_counts(series, top: int | None = None) -> dict[str, int]:
    counts = series.fillna("").astype(str).value_counts()
    if top:
        counts = counts.head(top)
    return {str(key): int(value) for key, value in counts.to_dict().items()}


if __name__ == "__main__":
    main()
