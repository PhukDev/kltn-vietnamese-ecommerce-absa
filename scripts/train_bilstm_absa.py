from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.absa import prepare_absa_annotation_frame  # noqa: E402
from ecommerce_absa.compat import disable_optional_pyarrow  # noqa: E402
from ecommerce_absa.config import ABSA_ASPECT_COLUMNS, DATA_DIR, MODELS_DIR, REPORTS_DIR  # noqa: E402
from ecommerce_absa.models.deep_learning import build_bilstm_classifier, require_torch  # noqa: E402
from ecommerce_absa.preprocessing import TextPreprocessor, WordSegmenter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Bi-LSTM + POS style ABSA model on working train / gold eval split")
    parser.add_argument("--train", default=str(DATA_DIR / "absa_working_train.csv"))
    parser.add_argument("--eval", default=str(DATA_DIR / "absa_gold_eval.csv"))
    parser.add_argument("--model-output", default=str(MODELS_DIR / "absa_bilstm_gold_eval.pt"))
    parser.add_argument("--metrics-output", default=str(REPORTS_DIR / "absa_bilstm_gold_eval_metrics.json"))
    parser.add_argument("--segmenter", choices=["auto", "vncorenlp", "none"], default="none")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    disable_optional_pyarrow()
    try:
        import pandas as pd  # type: ignore
        from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support
    except ImportError as exc:
        raise RuntimeError("pandas and scikit-learn are required. Install dependencies with: pip install -r requirements.txt") from exc

    torch = require_torch()
    torch.manual_seed(args.seed)

    preprocessor = TextPreprocessor(
        use_word_segmentation=args.segmenter != "none",
        segmenter=WordSegmenter(backend=args.segmenter),
    )

    train_raw = pd.read_csv(args.train, encoding="utf-8-sig")
    eval_raw = pd.read_csv(args.eval, encoding="utf-8-sig")
    train_frame = prepare_absa_annotation_frame(train_raw, preprocessor=preprocessor)
    eval_frame = prepare_absa_annotation_frame(eval_raw, preprocessor=preprocessor)

    train_texts = train_frame["processed_text"].tolist()
    eval_texts = eval_frame["processed_text"].tolist()
    train_y = train_frame[ABSA_ASPECT_COLUMNS].to_numpy(dtype="float32")
    eval_y = eval_frame[ABSA_ASPECT_COLUMNS].to_numpy(dtype="float32")

    vocab = build_vocab(train_texts)
    pos_vocab = build_pos_vocab(train_texts)

    train_dataset = TextTensorDataset(
        train_texts,
        train_y,
        vocab=vocab,
        pos_vocab=pos_vocab,
        max_len=args.max_len,
    )
    eval_dataset = TextTensorDataset(
        eval_texts,
        eval_y,
        vocab=vocab,
        pos_vocab=pos_vocab,
        max_len=args.max_len,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_bilstm_classifier(
        vocab_size=len(vocab),
        num_classes=len(ABSA_ASPECT_COLUMNS),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pos_vocab_size=len(pos_vocab),
        pos_embedding_dim=32,
    )

    pos_weight = build_pos_weight(train_y, torch)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state = None
    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, torch)
        metrics = evaluate_model(model, eval_loader, torch)
        history.append({"epoch": epoch, "train_loss": train_loss, "eval_f1_micro": metrics["f1_micro"]})
        if metrics["f1_micro"] > best_f1:
            best_f1 = metrics["f1_micro"]
            best_state = {
                "model_state": model.state_dict(),
                "vocab": vocab,
                "pos_vocab": pos_vocab,
                "config": {
                    "embedding_dim": args.embedding_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "max_len": args.max_len,
                    "segmenter": args.segmenter,
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


class TextTensorDataset:
    def __init__(self, texts, labels, *, vocab, pos_vocab, max_len: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.pos_vocab = pos_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        import torch

        tokens = self.texts[index].split()
        input_ids = encode_tokens(tokens, self.vocab, self.max_len)
        pos_ids = encode_tokens(guess_pos_tags(tokens), self.pos_vocab, self.max_len)
        attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "pos_ids": torch.tensor(pos_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def build_vocab(texts) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def build_pos_vocab(texts) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(guess_pos_tags(text.split()))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def guess_pos_tags(tokens: list[str]) -> list[str]:
    tags = []
    for token in tokens:
        if token.isdigit():
            tags.append("NUM")
        elif "_" in token:
            tags.append("NOUN")
        elif token in {"không", "ko", "chua", "chưa"}:
            tags.append("PART")
        elif token in {"rất", "qua", "quá"}:
            tags.append("ADV")
        else:
            tags.append("WORD")
    return tags


def encode_tokens(tokens: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    encoded = [vocab.get(token, vocab.get("<unk>", 1)) for token in tokens[:max_len]]
    if len(encoded) < max_len:
        encoded.extend([0] * (max_len - len(encoded)))
    return encoded


def build_pos_weight(train_y, torch):
    positives = train_y.sum(axis=0)
    negatives = len(train_y) - positives
    weights = []
    for pos, neg in zip(positives, negatives):
        if pos <= 0:
            weights.append(1.0)
        else:
            weights.append(float(neg / pos))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, optimizer, criterion, torch) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pos_ids=batch["pos_ids"],
        )
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def evaluate_model(model, loader, torch) -> dict:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pos_ids=batch["pos_ids"],
            )
            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(batch["labels"].int().cpu().numpy().tolist())
    return evaluate_multilabel(y_true, y_pred)


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
