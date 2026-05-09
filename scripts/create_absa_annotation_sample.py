from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.compat import disable_optional_pyarrow  # noqa: E402
from ecommerce_absa.config import (  # noqa: E402
    ABSA_ANNOTATION_COLUMNS,
    ABSA_ANNOTATION_SAMPLE_PATH,
    ABSA_ASPECT_COLUMNS,
    DATA_PATH,
    DEFAULT_RANDOM_STATE,
    SENTIMENT_LABELS,
)
from ecommerce_absa.data import load_reviews, score_to_sentiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced ABSA annotation CSV from raw reviews")
    parser.add_argument("--input", default=str(DATA_PATH))
    parser.add_argument("--output", default=str(ABSA_ANNOTATION_SAMPLE_PATH))
    parser.add_argument("--limit", type=int, default=100000, help="Rows to scan from the raw dataset")
    parser.add_argument("--sample-size", type=int, default=300, help="Rows to export for manual annotation")
    parser.add_argument("--min-content-length", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    disable_optional_pyarrow()
    columns = ["reviewid", "content", "score"]
    raw = load_reviews(args.input, columns=columns, limit=args.limit)
    raw["sentiment"] = raw["score"].map(score_to_sentiment)
    raw["content"] = raw["content"].fillna("").astype(str).str.strip()
    raw = raw.dropna(subset=["sentiment"])
    raw = raw[raw["content"].str.len() >= args.min_content_length]
    raw = raw.drop_duplicates(subset=["reviewid"])

    sample = balanced_sentiment_sample(raw, args.sample_size, args.random_state)
    for aspect in ABSA_ASPECT_COLUMNS:
        sample[aspect] = ""

    output = sample[ABSA_ANNOTATION_COLUMNS].reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "output": str(output_path),
        "rows_scanned": int(len(raw)),
        "rows_exported": int(len(output)),
        "sentiment_distribution": {
            str(label): int(count) for label, count in output["sentiment"].value_counts().to_dict().items()
        },
        "aspect_columns_left_blank": ABSA_ASPECT_COLUMNS,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def balanced_sentiment_sample(frame, sample_size: int, random_state: int):
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    base = sample_size // len(SENTIMENT_LABELS)
    remainder = sample_size % len(SENTIMENT_LABELS)
    requested = {
        label: base + (1 if index < remainder else 0)
        for index, label in enumerate(SENTIMENT_LABELS)
    }

    parts = []
    selected_indexes: set[int] = set()
    for label, count in requested.items():
        group = frame[frame["sentiment"] == label]
        if group.empty:
            continue
        take = min(count, len(group))
        part = group.sample(n=take, random_state=random_state)
        parts.append(part)
        selected_indexes.update(part.index.tolist())

    if parts:
        sample = frame.iloc[0:0].copy()
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pandas is required. Install dependencies with: pip install -r requirements.txt") from exc
        sample = pd.concat(parts)
    else:
        sample = frame.iloc[0:0].copy()

    if len(sample) < min(sample_size, len(frame)):
        remaining = frame.drop(index=list(selected_indexes), errors="ignore")
        fill_count = min(sample_size - len(sample), len(remaining))
        if fill_count > 0:
            fill = remaining.sample(n=fill_count, random_state=random_state)
            try:
                import pandas as pd  # type: ignore
            except ImportError as exc:
                raise RuntimeError("pandas is required. Install dependencies with: pip install -r requirements.txt") from exc
            sample = pd.concat([sample, fill])

    return sample.sample(frac=1, random_state=random_state)


if __name__ == "__main__":
    main()
