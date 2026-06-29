from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Add src to Python Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the original 1.3M reviews dataset into 30% labeled train and 70% predict splits")
    parser.add_argument("--input", default="vietnamese_ecommerce_review.csv")
    parser.add_argument("--train-out", default="data/original_train.csv")
    parser.add_argument("--predict-out", default="data/original_predict.csv")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        import pandas as pd  # type: ignore
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError("pandas and scikit-learn are required. Install dependencies with: pip install -r requirements.txt") from exc

    # Load ASPECT_KEYWORDS from config / race
    from ecommerce_absa.race import ASPECT_KEYWORDS

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Original dataset file not found: {input_path}")

    print(f"Reading original dataset: {input_path}...")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"Total rows read: {len(df):,}")

    # Data Cleaning: Drop duplicate review IDs
    if "reviewid" in df.columns:
        df = df.drop_duplicates(subset=["reviewid"]).copy()

    # Data Cleaning: Drop rows with missing content or invalid scores
    df = df[df["content"].notna() & df["score"].isin([1.0, 2.0, 3.0, 4.0, 5.0])].copy()
    print(f"Cleaned dataset rows (unique & valid): {len(df):,}")

    df["stratify_score"] = df["score"].fillna(-1).astype(int)

    print(f"Performing stratified split ({100 - int(args.test_size * 100)}% train, {int(args.test_size * 100)}% predict)...")
    train_df, predict_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["stratify_score"]
    )

    # -------------------------------------------------------------
    # 1. Processing Train Set (Generate ABSA labels automatically)
    # -------------------------------------------------------------
    print(f"Generating ABSA aspect & sentiment labels automatically for {100 - int(args.test_size * 100)}% train set...")
    
    # Map score to sentiment
    def score_to_sentiment(score_val):
        try:
            val = int(score_val)
            if val in [1, 2]:
                return "negative"
            elif val == 3:
                return "neutral"
            else:
                return "positive"
        except (TypeError, ValueError):
            return "positive"

    train_df["sentiment"] = train_df["score"].apply(score_to_sentiment)
    
    # Vectorized fast labeling using aspect keyword regular expressions
    content_lower = train_df["content"].fillna("").astype(str).str.lower()
    for aspect, keywords in ASPECT_KEYWORDS.items():
        pattern = "|".join([re.escape(kw) for kw in keywords])
        train_df[aspect] = content_lower.str.contains(pattern, regex=True).astype(int)

    # Reorder columns to match ABSA format
    train_cols = ["reviewid", "content", "sentiment", "product", "price", "delivery", "service", "app"]
    train_df = train_df[train_cols].copy()

    # -------------------------------------------------------------
    # 2. Processing 70% Predict Set (Keep original columns for inference)
    # -------------------------------------------------------------
    predict_df = predict_df.drop(columns=["stratify_score"]).copy()

    # Save to CSV files
    train_out_path = Path(args.train_out)
    predict_out_path = Path(args.predict_out)
    
    train_out_path.parent.mkdir(parents=True, exist_ok=True)
    predict_out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing ABSA labeled train set to {train_out_path} ({len(train_df):,} rows)...")
    train_df.to_csv(train_out_path, index=False, encoding="utf-8-sig")

    print(f"Writing raw predict set to {predict_out_path} ({len(predict_df):,} rows)...")
    predict_df.to_csv(predict_out_path, index=False, encoding="utf-8-sig")

    summary = {
        "train_path": str(train_out_path),
        "predict_path": str(predict_out_path),
        "total_rows": len(df),
        "train_rows": len(train_df),
        "predict_rows": len(predict_df),
        "train_aspect_counts": {
            aspect: int(train_df[aspect].sum()) for aspect in ASPECT_KEYWORDS.keys()
        }
    }
    
    print("\nSplit and ABSA labeling completed successfully!")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
