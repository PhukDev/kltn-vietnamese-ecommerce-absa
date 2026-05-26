from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.absa import prepare_absa_annotation_frame  # noqa: E402
from ecommerce_absa.compat import disable_optional_pyarrow  # noqa: E402
from ecommerce_absa.config import ABSA_ASPECT_COLUMNS, DATA_DIR, MODELS_DIR, REPORTS_DIR  # noqa: E402
from ecommerce_absa.evaluation import evaluate_classification  # noqa: E402
from ecommerce_absa.models.deep_learning import build_phobert_multitask_classifier, require_torch  # noqa: E402
from ecommerce_absa.preprocessing import TextPreprocessor, WordSegmenter  # noqa: E402

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_TO_ID = {label: index for index, label in enumerate(SENTIMENT_LABELS)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhoBERT multitask ABSA model on working train / gold eval split")
    parser.add_argument("--train", default=str(DATA_DIR / "absa_working_train.csv"))
    parser.add_argument("--eval", default=str(DATA_DIR / "absa_gold_eval.csv"))
    parser.add_argument("--model-name", default="vinai/phobert-base")
    parser.add_argument("--model-output", default=str(MODELS_DIR / "absa_phobert_gold_eval.pt"))
    parser.add_argument("--metrics-output", default=str(REPORTS_DIR / "absa_phobert_gold_eval_metrics.json"))
    parser.add_argument("--segmenter", choices=["auto", "vncorenlp", "none"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    disable_optional_pyarrow()
    try:
        import pandas as pd  # type: ignore
        from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "pandas, scikit-learn and transformers are required. Install dependencies with: pip install -r requirements-dl.txt"
        ) from exc

    torch = require_torch()
    torch.manual_seed(args.seed)

    preprocessor = TextPreprocessor(
        use_word_segmentation=True,
        segmenter=WordSegmenter(backend=args.segmenter),
    )

    train_raw = pd.read_csv(args.train, encoding="utf-8-sig")
    eval_raw = pd.read_csv(args.eval, encoding="utf-8-sig")
    train_frame = prepare_absa_annotation_frame(train_raw, preprocessor=preprocessor)
    eval_frame = prepare_absa_annotation_frame(eval_raw, preprocessor=preprocessor)
    train_frame = train_frame[train_frame["sentiment"].isin(SENTIMENT_TO_ID)]
    eval_frame = eval_frame[eval_frame["sentiment"].isin(SENTIMENT_TO_ID)]

    if len(train_frame) < 5 or len(eval_frame) < 5:
        raise SystemExit("Both train and eval sets need at least 5 labeled rows with valid sentiment labels.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    train_dataset = PhoBertDataset(train_frame, tokenizer=tokenizer, max_len=args.max_len)
    eval_dataset = PhoBertDataset(eval_frame, tokenizer=tokenizer, max_len=args.max_len)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_phobert_multitask_classifier(
        model_name=args.model_name,
        num_sentiment_labels=len(SENTIMENT_LABELS),
        num_aspect_labels=len(ABSA_ASPECT_COLUMNS),
        dropout=args.dropout,
    )

    aspect_pos_weight = build_aspect_pos_weight(train_frame[ABSA_ASPECT_COLUMNS].to_numpy(dtype="float32"), torch)
    sentiment_class_weight = build_sentiment_class_weight(train_frame["sentiment"].tolist(), torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_score = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            torch,
            aspect_pos_weight=aspect_pos_weight,
            sentiment_class_weight=sentiment_class_weight,
        )
        metrics = evaluate_model(model, eval_loader, torch)
        combined_score = (metrics["aspect"]["f1_micro"] + metrics["sentiment"]["f1_macro"]) / 2.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "aspect_f1_micro": metrics["aspect"]["f1_micro"],
                "sentiment_f1_macro": metrics["sentiment"]["f1_macro"],
                "combined_score": combined_score,
            }
        )
        if combined_score > best_score:
            best_score = combined_score
            best_state = {
                "model_state": model.state_dict(),
                "tokenizer_name": args.model_name,
                "config": {
                    "model_name": args.model_name,
                    "segmenter": args.segmenter,
                    "max_len": args.max_len,
                    "dropout": args.dropout,
                },
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")

    model.load_state_dict(best_state["model_state"])
    final_metrics = evaluate_model(model, eval_loader, torch)
    final_metrics.update(
        {
            "rows_train_loaded": int(len(train_raw)),
            "rows_train_labeled": int(len(train_frame)),
            "rows_eval_loaded": int(len(eval_raw)),
            "rows_eval_labeled": int(len(eval_frame)),
            "aspect_columns": ABSA_ASPECT_COLUMNS,
            "sentiment_labels": SENTIMENT_LABELS,
            "history": history,
        }
    )

    model_output = Path(args.model_output)
    metrics_output = Path(args.metrics_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_output)
    metrics_output.write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(model_output), "metrics": final_metrics}, ensure_ascii=False, indent=2))


class PhoBertDataset:
    def __init__(self, frame, *, tokenizer, max_len: int):
        self.texts = frame["processed_text"].tolist()
        self.aspect_labels = frame[ABSA_ASPECT_COLUMNS].to_numpy(dtype="float32")
        self.sentiment_labels = [SENTIMENT_TO_ID[item] for item in frame["sentiment"].tolist()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        import torch

        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "aspect_labels": torch.tensor(self.aspect_labels[index], dtype=torch.float32),
            "sentiment_labels": torch.tensor(self.sentiment_labels[index], dtype=torch.long),
        }


def build_aspect_pos_weight(train_y, torch):
    positives = train_y.sum(axis=0)
    negatives = len(train_y) - positives
    weights = []
    for pos, neg in zip(positives, negatives):
        if pos <= 0:
            weights.append(1.0)
        else:
            weights.append(float(neg / pos))
    return torch.tensor(weights, dtype=torch.float32)


def build_sentiment_class_weight(sentiments: list[str], torch):
    counts = {label: 0 for label in SENTIMENT_LABELS}
    for item in sentiments:
        counts[str(item)] = counts.get(str(item), 0) + 1
    total = sum(counts.values())
    weights = []
    for label in SENTIMENT_LABELS:
        count = counts.get(label, 0)
        if count <= 0:
            weights.append(1.0)
        else:
            weights.append(float(total / (len(SENTIMENT_LABELS) * count)))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, optimizer, torch, *, aspect_pos_weight, sentiment_class_weight) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        optimizer.zero_grad()
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            aspect_labels=batch["aspect_labels"],
            sentiment_labels=batch["sentiment_labels"],
            aspect_pos_weight=aspect_pos_weight,
            sentiment_class_weight=sentiment_class_weight,
        )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def evaluate_model(model, loader, torch) -> dict:
    model.eval()
    aspect_true = []
    aspect_pred = []
    sentiment_true = []
    sentiment_pred = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            aspect_logits = output["aspect_logits"]
            sentiment_logits = output["sentiment_logits"]
            aspect_predictions = (torch.sigmoid(aspect_logits) >= 0.5).int().cpu().numpy()
            sentiment_predictions = sentiment_logits.argmax(dim=-1).cpu().numpy()
            aspect_pred.extend(aspect_predictions.tolist())
            aspect_true.extend(batch["aspect_labels"].int().cpu().numpy().tolist())
            sentiment_pred.extend(sentiment_predictions.tolist())
            sentiment_true.extend(batch["sentiment_labels"].cpu().numpy().tolist())

    aspect_metrics = evaluate_multilabel(aspect_true, aspect_pred)
    sentiment_true_labels = [SENTIMENT_LABELS[index] for index in sentiment_true]
    sentiment_pred_labels = [SENTIMENT_LABELS[index] for index in sentiment_pred]
    sentiment_metrics = evaluate_classification(sentiment_true_labels, sentiment_pred_labels, labels=SENTIMENT_LABELS)
    return {"aspect": aspect_metrics, "sentiment": sentiment_metrics}


def evaluate_multilabel(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support

    result = {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }
    for average in ("micro", "macro", "samples"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )
        result[f"precision_{average}"] = float(precision)
        result[f"recall_{average}"] = float(recall)
        result[f"f1_{average}"] = float(f1)
    per_label = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    result["per_aspect"] = {
        aspect: {
            "precision": float(per_label[0][index]),
            "recall": float(per_label[1][index]),
            "f1": float(per_label[2][index]),
            "support": int(per_label[3][index]),
        }
        for index, aspect in enumerate(ABSA_ASPECT_COLUMNS)
    }
    return result


if __name__ == "__main__":
    main()
