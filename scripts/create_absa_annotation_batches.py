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
    ABSA_ASPECT_COLUMNS,
    DATA_PATH,
    DEFAULT_RANDOM_STATE,
)
from ecommerce_absa.data import load_reviews, score_to_sentiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create multiple non-overlapping ABSA annotation batches")
    parser.add_argument("--input", default=str(DATA_PATH))
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--prefix", default="absa_annotation_batch")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--batch-sizes", default="300,300,400")
    parser.add_argument("--min-content-length", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    disable_optional_pyarrow()
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pandas is required. Install dependencies with: pip install -r requirements.txt") from exc

    batch_sizes = parse_batch_sizes(args.batch_sizes)
    columns = ["reviewid", "content", "score"]
    raw = load_reviews(args.input, columns=columns, limit=args.limit)
    raw["sentiment"] = raw["score"].map(score_to_sentiment)
    raw["content"] = raw["content"].fillna("").astype(str).str.strip()
    raw = raw.dropna(subset=["sentiment"])
    raw = raw[raw["content"].str.len() >= args.min_content_length]
    raw = raw.drop_duplicates(subset=["reviewid"]).reset_index(drop=True)

    remaining = raw.copy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for index, batch_size in enumerate(batch_sizes, start=1):
        batch = balanced_sentiment_sample(
            remaining,
            batch_size,
            random_state=args.random_state + index,
        ).copy()
        for aspect in ABSA_ASPECT_COLUMNS:
            batch[aspect] = ""
        output = batch[ABSA_ANNOTATION_COLUMNS].reset_index(drop=True)
        output_path = output_dir / f"{args.prefix}_{index:03d}.csv"
        output.to_csv(output_path, index=False, encoding="utf-8-sig")

        summary.append(
            {
                "file": str(output_path),
                "rows_exported": int(len(output)),
                "sentiment_distribution": {
                    str(label): int(count) for label, count in output["sentiment"].value_counts().to_dict().items()
                },
            }
        )
        remaining = remaining[~remaining["reviewid"].isin(batch["reviewid"])].reset_index(drop=True)

    print(
        json.dumps(
            {
                "rows_scanned": int(len(raw)),
                "rows_remaining_after_export": int(len(remaining)),
                "batches": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_batch_sizes(value: str) -> list[int]:
    batch_sizes = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        size = int(text)
        if size <= 0:
            raise ValueError("Batch sizes must be positive integers")
        batch_sizes.append(size)
    if not batch_sizes:
        raise ValueError("At least one batch size is required")
    return batch_sizes


def balanced_sentiment_sample(frame, sample_size: int, random_state: int):
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if frame.empty:
        return frame.iloc[0:0].copy()

    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pandas is required. Install dependencies with: pip install -r requirements.txt") from exc

    labels = ["negative", "neutral", "positive"]
    base = sample_size // len(labels)
    remainder = sample_size % len(labels)
    requested = {label: base + (1 if idx < remainder else 0) for idx, label in enumerate(labels)}

    parts = []
    selected_reviewids = set()
    for label, count in requested.items():
        group = frame[frame["sentiment"] == label]
        if group.empty:
            continue
        take = min(count, len(group))
        part = group.sample(n=take, random_state=random_state)
        parts.append(part)
        selected_reviewids.update(part["reviewid"].tolist())

    sample = pd.concat(parts) if parts else frame.iloc[0:0].copy()
    if len(sample) < min(sample_size, len(frame)):
        remaining = frame[~frame["reviewid"].isin(selected_reviewids)]
        fill_count = min(sample_size - len(sample), len(remaining))
        if fill_count > 0:
            fill = remaining.sample(n=fill_count, random_state=random_state)
            sample = pd.concat([sample, fill])

    return sample.sample(frac=1, random_state=random_state).drop_duplicates(subset=["reviewid"])


if __name__ == "__main__":
    main()
