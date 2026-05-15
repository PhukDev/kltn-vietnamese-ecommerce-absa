from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.compat import disable_optional_pyarrow  # noqa: E402
from ecommerce_absa.config import ABSA_ASPECT_COLUMNS  # noqa: E402
from ecommerce_absa.race import detect_aspects  # noqa: E402

from flag_sentiment_mismatches import build_flags  # type: ignore  # noqa: E402
from ecommerce_absa.preprocessing import TextPreprocessor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create prefilled ABSA annotation batches with auto sentiment review")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--pattern", default="absa_annotation_batch_*.csv")
    parser.add_argument("--output-dir", default="data/prefilled")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    disable_optional_pyarrow()
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pandas is required. Install dependencies with: pip install -r requirements.txt") from exc

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit("No batch files found")

    preprocessor = TextPreprocessor(use_word_segmentation=False)
    summaries = []

    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        flagged = build_flags(frame, preprocessor=preprocessor)
        flagged = flagged[["reviewid", "content_polarity", "mismatch_reason", "flagged"]].copy()
        merged = frame.merge(flagged, on="reviewid", how="left")

        merged["sentiment_original"] = merged["sentiment"]
        merged["sentiment_auto_fixed"] = merged.apply(auto_fix_sentiment, axis=1)
        merged["sentiment"] = merged["sentiment_auto_fixed"]
        merged["sentiment_changed"] = merged["sentiment_original"] != merged["sentiment"]

        prefilled = merged["content"].fillna("").astype(str).map(prefill_aspects)
        for aspect in ABSA_ASPECT_COLUMNS:
            merged[f"{aspect}_prefill"] = prefilled.map(lambda item, aspect=aspect: item.get(aspect, ""))
            merged[aspect] = merged[f"{aspect}_prefill"]

        merged["prefill_review_note"] = merged.apply(build_review_note, axis=1)

        ordered_columns = [
            "reviewid",
            "content",
            "sentiment",
            *ABSA_ASPECT_COLUMNS,
            "sentiment_original",
            "sentiment_changed",
            "content_polarity",
            "mismatch_reason",
            "flagged",
            *(f"{aspect}_prefill" for aspect in ABSA_ASPECT_COLUMNS),
            "prefill_review_note",
        ]
        output = merged[ordered_columns].copy()

        output_path = output_dir / path.name
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Output file already exists: {output_path}. Use --overwrite to replace it.")
        output.to_csv(output_path, index=False, encoding="utf-8-sig")

        summaries.append(
            {
                "source": str(path),
                "output": str(output_path),
                "rows": int(len(output)),
                "sentiment_changed": int(output["sentiment_changed"].sum()),
                "flagged_rows": int(output["flagged"].fillna(False).astype(bool).sum()),
            }
        )

    print(json.dumps({"outputs": summaries}, ensure_ascii=False, indent=2))


def auto_fix_sentiment(row) -> str:
    current = str(row.get("sentiment", "")).strip().lower()
    content_polarity = str(row.get("content_polarity", "")).strip().lower()
    if current == "neutral" and content_polarity in {"positive", "negative"}:
        return content_polarity
    if current == "positive" and content_polarity == "negative":
        return "negative"
    if current == "negative" and content_polarity == "positive":
        return "positive"
    return current


def prefill_aspects(text: str) -> dict[str, str]:
    aspects = detect_aspects(text)
    result = {aspect: "0" for aspect in ABSA_ASPECT_COLUMNS}
    for aspect in aspects:
        if aspect in result:
            result[aspect] = "1"
    return result


def build_review_note(row) -> str:
    notes = []
    if bool(row.get("sentiment_changed")):
        notes.append("review sentiment auto-adjusted")
    if bool(row.get("flagged")):
        notes.append("manual review recommended")
    if not any(str(row.get(aspect, "")).strip() == "1" for aspect in ABSA_ASPECT_COLUMNS):
        notes.append("all aspects are 0; verify if review is general")
    return "; ".join(notes)


if __name__ == "__main__":
    main()
