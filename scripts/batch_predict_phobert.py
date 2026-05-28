from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.config import ABSA_ASPECT_COLUMNS, SENTIMENT_LABELS  # noqa: E402
from ecommerce_absa.preprocessing import TextPreprocessor  # noqa: E402
from ecommerce_absa.race import analyze_business_signal  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Highly optimized batch prediction for e-commerce reviews using PhoBERT Multitask")
    parser.add_argument("--input", required=True, help="Path to the raw reviews CSV (e.g. vietnamese_ecommerce_review.csv)")
    parser.add_argument("--model", required=True, help="Path to the trained PyTorch PhoBERT model checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Path to export the predicted CSV file")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of reviews to process (0 for all)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for GPU acceleration")
    parser.add_argument("--max-len", type=int, default=128, help="Maximum sequence length")
    args = parser.parse_args()

    # 1. Load PyTorch and Deep Learning Modules
    try:
        import pandas as pd  # type: ignore
        import torch  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore
        from torch.cuda.amp import autocast  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
        from ecommerce_absa.models.deep_learning import build_phobert_multitask_classifier  # noqa: E402
    except ImportError as exc:
        raise RuntimeError(
            "pandas, torch, and transformers are required. "
            "Install dependencies with: pip install -r requirements-dl.txt"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # 2. Load Checkpoint & Model
    print(f"Loading checkpoint from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)
    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "vinai/phobert-base")
    max_len = args.max_len or config.get("max_len", 128)

    print("Building PhoBERT multitask classifier...")
    model = build_phobert_multitask_classifier(
        model_name=model_name,
        num_sentiment_labels=3,
        num_aspect_labels=5,
        dropout=config.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    tokenizer_name = checkpoint.get("tokenizer_name") or model_name
    print(f"Loading tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    # 3. Read Raw CSV
    print(f"Reading raw reviews from {args.input}...")
    columns = ["reviewid", "content", "score", "thumbsupcount", "reviewcreatedversion", "at", "replycontent", "appid"]
    
    if args.limit > 0:
        raw_df = pd.read_csv(args.input, usecols=columns, nrows=args.limit, encoding="utf-8-sig")
    else:
        raw_df = pd.read_csv(args.input, usecols=columns, encoding="utf-8-sig")

    print(f"Loaded {len(raw_df):,} reviews.")

    # 4. Text Preprocessing
    print("Preprocessing texts...")
    preprocessor = TextPreprocessor()
    raw_df["processed_text"] = raw_df["content"].fillna("").astype(str).apply(preprocessor.transform)

    # 5. Create PyTorch Dataset & DataLoader
    class InferenceDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len: int):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, index):
            text = str(self.texts[index])
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "index": index,
            }

    dataset = InferenceDataset(raw_df["processed_text"].tolist(), tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 6. Run GPU Accelerated Batch Prediction
    print("Running batch predictions...")
    aspect_results = [None] * len(dataset)
    sentiment_results = [None] * len(dataset)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            indices = batch["index"].numpy()

            # Autocast mixed precision for faster forward pass on GPU
            with autocast(enabled=(device == "cuda")):
                outputs = model(input_ids, attention_mask=attention_mask)
                aspect_logits = outputs["aspect_logits"]
                sentiment_logits = outputs["sentiment_logits"]

            # Aspect binary vectors
            aspect_preds = (torch.sigmoid(aspect_logits) >= 0.5).int().cpu().numpy()
            # Sentiment indices
            sentiment_preds = sentiment_logits.argmax(dim=-1).cpu().numpy()

            for i, idx in enumerate(indices):
                # aspect labels list
                row_aspects = [
                    aspect
                    for a_idx, aspect in enumerate(ABSA_ASPECT_COLUMNS)
                    if int(aspect_preds[i][a_idx]) == 1
                ]
                aspect_results[idx] = row_aspects or ["general"]
                sentiment_results[idx] = SENTIMENT_LABELS[sentiment_preds[i]]

            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {(batch_idx + 1) * args.batch_size:,} / {len(dataset):,} reviews...")

    print("Batch prediction completed successfully!")

    # 7. Map to RACE Framework & Save Predicted CSV
    print("Mapping predictions to business signals (RACE)...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as out_file:
        fieldnames = [
            "reviewid",
            "content",
            "score",
            "thumbsupcount",
            "reviewcreatedversion",
            "at",
            "replycontent",
            "appid",
            "sentiment",
            "aspects",
            "race_stage",
            "alert",
        ]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in raw_df.iterrows():
            sentiment = sentiment_results[idx]
            aspects = aspect_results[idx]
            
            # Map RACE Business logic
            business = analyze_business_signal(
                content=str(row.get("content", "")),
                sentiment=sentiment,
                aspects=aspects,
                thumbsupcount=int(row.get("thumbsupcount", 0)),
                replycontent=str(row.get("replycontent", "")),
            )

            writer.writerow({
                "reviewid": row.get("reviewid", ""),
                "content": row.get("content", ""),
                "score": row.get("score", ""),
                "thumbsupcount": row.get("thumbsupcount", 0),
                "reviewcreatedversion": row.get("reviewcreatedversion", ""),
                "at": row.get("at", ""),
                "replycontent": row.get("replycontent", ""),
                "appid": row.get("appid", ""),
                "sentiment": sentiment,
                "aspects": ", ".join(aspects),
                "race_stage": business["race"]["primary_stage"],
                "alert": (business["alert"] or {}).get("message", ""),
            })

    print(f"✨ Successfully saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
