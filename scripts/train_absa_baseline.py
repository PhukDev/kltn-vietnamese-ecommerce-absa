from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.absa import prepare_absa_annotation_frame  # noqa: E402
from ecommerce_absa.compat import disable_optional_pyarrow  # noqa: E402
from ecommerce_absa.config import (  # noqa: E402
    ABSA_ANNOTATION_SAMPLE_PATH,
    ABSA_ASPECT_COLUMNS,
    DEFAULT_RANDOM_STATE,
    MODELS_DIR,
    REPORTS_DIR,
)
from ecommerce_absa.features import make_vectorizer  # noqa: E402
from ecommerce_absa.models.baselines import save_model_artifact  # noqa: E402
from ecommerce_absa.preprocessing import TextPreprocessor, WordSegmenter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Scikit-Learn baseline for multi-label ABSA aspect detection")
    parser.add_argument("--input", default=str(ABSA_ANNOTATION_SAMPLE_PATH))
    parser.add_argument("--model-output", default=str(MODELS_DIR / "absa_aspect_baseline.joblib"))
    parser.add_argument("--metrics-output", default=str(REPORTS_DIR / "absa_baseline_metrics.json"))
    parser.add_argument("--confusion-output", default=str(REPORTS_DIR / "confusion_matrix_absa_aspects.csv"))
    parser.add_argument("--vectorizer", choices=["tfidf", "bow"], default="tfidf")
    parser.add_argument("--segmenter", choices=["auto", "vncorenlp", "none"], default="auto")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    disable_optional_pyarrow()
    try:
        import pandas as pd  # type: ignore
        from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support
        from sklearn.model_selection import train_test_split
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise RuntimeError("pandas and scikit-learn are required. Install dependencies with: pip install -r requirements.txt") from exc

    raw = pd.read_csv(args.input)
    preprocessor = TextPreprocessor(
        use_word_segmentation=args.segmenter != "none",
        segmenter=WordSegmenter(backend=args.segmenter),
    )
    frame = prepare_absa_annotation_frame(raw, preprocessor=preprocessor)
    if len(frame) < 5:
        raise SystemExit(
            "Need at least 5 completely labeled rows. Fill product/price/delivery/service/app with 0 or 1 before training."
        )

    x = frame["processed_text"]
    y = frame[ABSA_ASPECT_COLUMNS].to_numpy(dtype=int)
    train_x, test_x, train_y, test_y = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

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
    predictions = model.predict(test_x)

    metrics = evaluate_multilabel(
        test_y,
        predictions,
        aspect_columns=ABSA_ASPECT_COLUMNS,
        precision_recall_fscore_support=precision_recall_fscore_support,
        accuracy_score=accuracy_score,
        hamming_loss=hamming_loss,
    )
    metrics.update(
        {
            "rows_loaded": int(len(raw)),
            "rows_labeled": int(len(frame)),
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
            "model_name": "absa_aspect_baseline",
            "task": "multi_label_aspect_detection",
            "vectorizer": args.vectorizer,
            "segmenter": args.segmenter,
            "aspect_columns": ABSA_ASPECT_COLUMNS,
            "rows_labeled": int(len(frame)),
        },
    )
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    confusion_frame(test_y, predictions, ABSA_ASPECT_COLUMNS).to_csv(confusion_output, index=False, encoding="utf-8-sig")
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
