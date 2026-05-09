from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.api import PredictionService  # noqa: E402
from ecommerce_absa.config import DATA_PATH  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export API-style predictions to CSV")
    parser.add_argument("--input", default=str(DATA_PATH))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "reports" / "predictions.csv"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()

    service = PredictionService(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8", newline="") as input_file, open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output_file:
        reader = csv.DictReader(input_file)
        fieldnames = [
            "reviewid",
            "content",
            "score",
            "thumbsupcount",
            "sentiment",
            "aspects",
            "race_primary_stage",
            "alert_severity",
            "alert_message",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(reader):
            if args.limit and index >= args.limit:
                break
            result = service.predict(row)
            alert = result.get("alert") or {}
            writer.writerow(
                {
                    "reviewid": row.get("reviewid", ""),
                    "content": row.get("content", ""),
                    "score": row.get("score", ""),
                    "thumbsupcount": row.get("thumbsupcount", ""),
                    "sentiment": result.get("sentiment", ""),
                    "aspects": "|".join(result.get("aspects", [])),
                    "race_primary_stage": result.get("race", {}).get("primary_stage", ""),
                    "alert_severity": alert.get("severity", ""),
                    "alert_message": alert.get("message", ""),
                }
            )
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
