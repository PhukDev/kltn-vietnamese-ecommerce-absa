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
from ecommerce_absa.features import make_vectorizer  # noqa: E402
from ecommerce_absa.models.baselines import save_model_artifact  # noqa: E402
from ecommerce_absa.preprocessing import TextPreprocessor, WordSegmenter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ABSA baseline on working train set and evaluate on gold eval set")
    parser.add_argument("--train", default=str(DATA_DIR / "absa_working_train.csv"))
    parser.add_argument("--eval", default=str(DATA_DIR / "absa_gold_eval.csv"))
    parser.add_argument("--model-output", default=str(MODELS_DIR / "absa_aspect_baseline_gold_eval.joblib"))
    parser.add_argument("--metrics-output", default=str(REPORTS_DIR / "absa_baseline_gold_eval_metrics.json"))
    parser.add_argument("--confusion-output", default=str(REPORTS_DIR / "confusion_matrix_absa_gold_eval.csv"))
    parser.add_argument("--vectorizer", choices=["tfidf", "bow"], default="tfidf")
    parser.add_argument("--segmenter", choices=["auto", "vncorenlp", "none"], default="none")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-df", type=int, default=1)
    args = parser.parse_args()

    disable_optional_pyarrow()
    try:
        import pandas as pd  # type: ignore
        from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise RuntimeError("pandas and scikit-learn are required. Install dependencies with: pip install -r requirements.txt") from exc

    preprocessor = TextPreprocessor(
        use_word_segmentation=args.segmenter != "none",
        segmenter=WordSegmenter(backend=args.segmenter),
    )

    train_raw = pd.read_csv(args.train, encoding="utf-8-sig")
    eval_raw = pd.read_csv(args.eval, encoding="utf-8-sig")
    train_frame = prepare_absa_annotation_frame(train_raw, preprocessor=preprocessor)
    eval_frame = prepare_absa_annotation_frame(eval_raw, preprocessor=preprocessor)

    if len(train_frame) < 5 or len(eval_frame) < 5:
        raise SystemExit("Both train and eval sets need at least 5 completely labeled rows.")

    train_x = train_frame["processed_text"]
    train_y = train_frame[ABSA_ASPECT_COLUMNS].to_numpy(dtype=int)
    eval_x = eval_frame["processed_text"]
    eval_y = eval_frame[ABSA_ASPECT_COLUMNS].to_numpy(dtype=int)

    model = Pipeline(
        steps=[
            (
                "vectorizer",
                make_vectorizer(
                    args.vectorizer,
                    max_features=args.max_features,
                    min_df=args.min_df,
                ),
            ),
            ("classifier", OneVsRestClassifier(LinearSVC(class_weight="balanced"))),
        ]
    )
    model.fit(train_x, train_y)
    predictions = model.predict(eval_x)

    metrics = evaluate_multilabel(
        eval_y,
        predictions,
        aspect_columns=ABSA_ASPECT_COLUMNS,
        precision_recall_fscore_support=precision_recall_fscore_support,
        accuracy_score=accuracy_score,
        hamming_loss=hamming_loss,
    )
    metrics.update(
        {
            "rows_train_loaded": int(len(train_raw)),
            "rows_train_labeled": int(len(train_frame)),
            "rows_eval_loaded": int(len(eval_raw)),
            "rows_eval_labeled": int(len(eval_frame)),
            "vectorizer": args.vectorizer,
            "segmenter": args.segmenter,
            "aspect_columns": ABSA_ASPECT_COLUMNS,
        }
    )

    model_output = Path(args.model_output)
    metrics_output = Path(args.metrics_output)
    confusion_output = Path(args.confusion_output)
    save_model_artifact(
        model,
        model_output,
        metadata={
            "model_name": "absa_aspect_baseline_gold_eval",
            "task": "multi_label_aspect_detection",
            "vectorizer": args.vectorizer,
            "segmenter": args.segmenter,
            "aspect_columns": ABSA_ASPECT_COLUMNS,
            "rows_train_labeled": int(len(train_frame)),
            "rows_eval_labeled": int(len(eval_frame)),
        },
    )
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    confusion_frame(eval_y, predictions, ABSA_ASPECT_COLUMNS).to_csv(confusion_output, index=False, encoding="utf-8-sig")
    print(json.dumps({"model": str(model_output), "metrics": metrics}, ensure_ascii=False, indent=2))


def evaluate_multilabel(
    y_true,
    y_pred,
    *,
    aspect_columns: list[str],
    precision_recall_fscore_support,
    accuracy_score,
    hamming_loss,
) -> dict:
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
        for index, aspect in enumerate(aspect_columns)
    }
    return result


def confusion_frame(y_true, y_pred, aspect_columns: list[str]):
    try:
        import pandas as pd  # type: ignore
        from sklearn.metrics import confusion_matrix
    except ImportError as exc:
        raise RuntimeError("pandas and scikit-learn are required. Install dependencies with: pip install -r requirements.txt") from exc

    rows = []
    for index, aspect in enumerate(aspect_columns):
        tn, fp, fn, tp = confusion_matrix(y_true[:, index], y_pred[:, index], labels=[0, 1]).ravel()
        rows.append(
            {
                "aspect": aspect,
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
