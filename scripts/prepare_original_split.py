from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the original 1.3M reviews dataset into 30% train and 70% predict splits")
    parser.add_argument("--input", default="vietnamese_ecommerce_review.csv")
    parser.add_argument("--train-out", default="data/original_train.csv")
    parser.add_argument("--predict-out", default="data/original_predict.csv")
    parser.add_argument("--test-size", type=float, default=0.7)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        import pandas as pd  # type: ignore
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError("pandas and scikit-learn are required. Install dependencies with: pip install -r requirements.txt") from exc

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Original dataset file not found: {input_path}")

    print(f"Reading original dataset: {input_path} (This may take a moment for 1.3M rows)...")
    
    # Read entire file (exactly 1,300,086 rows)
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    
    print(f"Total rows read: {len(df):,}")

    # Fill NA values in score with a placeholder so we don't drop any rows
    df["stratify_score"] = df["score"].fillna(-1).astype(int)

    print("Performing stratified split on all rows to keep exactly 1,300,086 reviews...")

    train_df, predict_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["stratify_score"]
    )

    # Drop temporary column
    train_df = train_df.drop(columns=["stratify_score"])
    predict_df = predict_df.drop(columns=["stratify_score"])

    train_out_path = Path(args.train_out)
    predict_out_path = Path(args.predict_out)
    
    train_out_path.parent.mkdir(parents=True, exist_ok=True)
    predict_out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing train set to {train_out_path} ({len(train_df):,} rows)...")
    train_df.to_csv(train_out_path, index=False, encoding="utf-8-sig")

    print(f"Writing predict set to {predict_out_path} ({len(predict_df):,} rows)...")
    predict_df.to_csv(predict_out_path, index=False, encoding="utf-8-sig")

    # Metrics
    summary = {
        "train_path": str(train_out_path),
        "predict_path": str(predict_out_path),
        "total_rows": len(df),
        "train_rows": len(train_df),
        "predict_rows": len(predict_df),
        "total_sum": len(train_df) + len(predict_df)
    }
    
    print("\nSplit successfully completed!")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
