from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare working-train and gold-eval splits from the current ABSA data")
    parser.add_argument("--master", default="data/absa_annotation_master.csv")
    parser.add_argument("--gold", default="data/absa_gold_subset_candidate.csv")
    parser.add_argument("--train-out", default="data/absa_working_train.csv")
    parser.add_argument("--eval-out", default="data/absa_gold_eval.csv")
    parser.add_argument("--review-out", default="data/absa_gold_eval_review_queue.csv")
    args = parser.parse_args()

    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pandas is required. Install dependencies with: pip install -r requirements.txt") from exc

    master = pd.read_csv(args.master, encoding="utf-8-sig")
    gold = pd.read_csv(args.gold, encoding="utf-8-sig")

    gold_ids = set(gold["reviewid"].astype(str))
    eval_frame = master[master["reviewid"].astype(str).isin(gold_ids)].copy()
    train_frame = master[~master["reviewid"].astype(str).isin(gold_ids)].copy()

    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    review_out = Path(args.review_out)
    train_out.parent.mkdir(parents=True, exist_ok=True)

    train_frame.to_csv(train_out, index=False, encoding="utf-8-sig")
    eval_frame.to_csv(eval_out, index=False, encoding="utf-8-sig")

    review_queue = gold.copy()
    review_queue["needs_manual_review"] = True
    review_queue["review_status"] = "pending"
    review_queue["review_note"] = ""
    review_queue.to_csv(review_out, index=False, encoding="utf-8-sig")

    print(
        json.dumps(
            {
                "train_out": str(train_out),
                "eval_out": str(eval_out),
                "review_out": str(review_out),
                "rows_train": int(len(train_frame)),
                "rows_eval": int(len(eval_frame)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
