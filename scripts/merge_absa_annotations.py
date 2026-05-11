from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.absa import prepare_absa_annotation_frame, require_absa_columns  # noqa: E402
from ecommerce_absa.compat import disable_optional_pyarrow  # noqa: E402
from ecommerce_absa.config import ABSA_ANNOTATION_COLUMNS, ABSA_ANNOTATION_SAMPLE_PATH  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge labeled ABSA annotation CSV files into a master dataset")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--pattern", default="absa_annotation_batch_*.csv")
    parser.add_argument("--seed-file", default=str(ABSA_ANNOTATION_SAMPLE_PATH))
    parser.add_argument("--output", default="data/absa_annotation_master.csv")
    parser.add_argument("--exclude-incomplete", action="store_true")
    args = parser.parse_args()

    disable_optional_pyarrow()
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pandas is required. Install dependencies with: pip install -r requirements.txt") from exc

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    seed_file = Path(args.seed_file)
    if seed_file.exists():
        files.insert(0, seed_file)
    if not files:
        raise SystemExit("No annotation files found")

    frames = []
    file_summaries = []
    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        require_columns(frame, path)
        require_absa_columns(frame)
        frame = frame[ABSA_ANNOTATION_COLUMNS].copy()
        frame["source_file"] = path.name
        frames.append(frame)

        prepared_rows = 0
        try:
            prepared_rows = int(len(prepare_absa_annotation_frame(frame)))
        except Exception:
            prepared_rows = 0
        file_summaries.append(
            {
                "file": str(path),
                "rows_loaded": int(len(frame)),
                "fully_labeled_rows": prepared_rows,
            }
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["reviewid"], keep="first").reset_index(drop=True)

    if args.exclude_incomplete:
        prepared = prepare_absa_annotation_frame(merged)
        source_map = merged[["reviewid", "source_file"]].drop_duplicates(subset=["reviewid"])
        prepared = prepared.merge(source_map, on="reviewid", how="left")
        output_frame = prepared[["reviewid", "content", "sentiment", "product", "price", "delivery", "service", "app", "source_file"]]
    else:
        output_frame = merged

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "output": str(output_path),
        "files": file_summaries,
        "rows_merged_before_dedup": int(sum(item["rows_loaded"] for item in file_summaries)),
        "rows_after_reviewid_dedup": int(len(merged)),
        "rows_written": int(len(output_frame)),
        "exclude_incomplete": bool(args.exclude_incomplete),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def require_columns(frame, path: Path) -> None:
    missing = [column for column in ABSA_ANNOTATION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


if __name__ == "__main__":
    main()
