from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare working-train (30%) and gold-eval (70%) splits from the master ABSA annotation data")
    parser.add_argument("--master", default="data/absa_annotation_master.csv")
    parser.add_argument("--train-out", default="data/absa_working_train.csv")
    parser.add_argument("--eval-out", default="data/absa_gold_eval.csv")
    parser.add_argument("--test-size", type=float, default=0.7)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        import pandas as pd  # type: ignore
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError("pandas and scikit-learn are required. Install dependencies with: pip install -r requirements.txt") from exc

    master_path = Path(args.master)
    if not master_path.exists():
        raise FileNotFoundError(f"Master annotation file not found: {master_path}")

    master = pd.read_csv(master_path, encoding="utf-8-sig")
    
    # Filter rows with valid sentiment
    valid_sentiment_labels = ["negative", "neutral", "positive"]
    master = master[master["sentiment"].isin(valid_sentiment_labels)].copy()

    train_frame, eval_frame = train_test_split(
        master,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=master["sentiment"]
    )

    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    train_out.parent.mkdir(parents=True, exist_ok=True)

    train_frame.to_csv(train_out, index=False, encoding="utf-8-sig")
    eval_frame.to_csv(eval_out, index=False, encoding="utf-8-sig")

    # Print distributions
    print(
        json.dumps(
            {
                "train_out": str(train_out),
                "eval_out": str(eval_out),
                "rows_train": int(len(train_frame)),
                "rows_eval": int(len(eval_frame)),
                "train_sentiment_dist": train_frame["sentiment"].value_counts().to_dict(),
                "eval_sentiment_dist": eval_frame["sentiment"].value_counts().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
