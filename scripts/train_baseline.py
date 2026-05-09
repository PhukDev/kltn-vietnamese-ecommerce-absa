from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.config import (  # noqa: E402
    DATA_PATH,
    DEFAULT_RANDOM_STATE,
    MODELS_DIR,
    REPORTS_DIR,
    SENTIMENT_LABELS,
)
from ecommerce_absa.compat import disable_optional_pyarrow  # noqa: E402
from ecommerce_absa.data import load_reviews, prepare_sentiment_frame  # noqa: E402
from ecommerce_absa.evaluation import confusion_matrix_frame, evaluate_classification, write_json  # noqa: E402
from ecommerce_absa.models.baselines import build_baseline_pipelines, save_model_artifact  # noqa: E402
from ecommerce_absa.preprocessing import TextPreprocessor, WordSegmenter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Naive Bayes and SVM sentiment baselines")
    parser.add_argument("--input", default=str(DATA_PATH))
    parser.add_argument("--limit", type=int, default=100000, help="Rows to load from CSV")
    parser.add_argument("--vectorizer", choices=["tfidf", "bow"], default="tfidf")
    parser.add_argument("--segmenter", choices=["auto", "vncorenlp", "none"], default="auto")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    disable_optional_pyarrow()
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required. Install dependencies with: pip install -r requirements.txt") from exc

    columns = ["content", "score", "thumbsupcount", "replycontent", "appid"]
    raw = load_reviews(args.input, columns=columns, limit=args.limit)
    preprocessor = TextPreprocessor(
        use_word_segmentation=args.segmenter != "none",
        segmenter=WordSegmenter(backend=args.segmenter),
    )
    frame = prepare_sentiment_frame(raw, preprocessor=preprocessor)

    train_x, test_x, train_y, test_y = train_test_split(
        frame["processed_text"],
        frame["sentiment"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=frame["sentiment"],
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    best_model_name = ""
    best_f1 = -1.0

    for model_name, model in build_baseline_pipelines(args.vectorizer).items():
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        metrics = evaluate_classification(test_y, predictions, labels=SENTIMENT_LABELS)
        metrics["rows_loaded"] = int(len(raw))
        metrics["rows_used"] = int(len(frame))
        metrics["vectorizer"] = args.vectorizer
        metrics["segmenter"] = args.segmenter
        all_metrics[model_name] = metrics

        confusion = confusion_matrix_frame(test_y, predictions, labels=SENTIMENT_LABELS)
        confusion.to_csv(REPORTS_DIR / f"confusion_matrix_{model_name}.csv", encoding="utf-8-sig")
        artifact_path = MODELS_DIR / f"{model_name}.joblib"
        save_model_artifact(
            model,
            artifact_path,
            metadata={
                "model_name": model_name,
                "vectorizer": args.vectorizer,
                "segmenter": args.segmenter,
                "labels": SENTIMENT_LABELS,
                "rows_used": int(len(frame)),
            },
        )

        f1 = metrics["f1_macro"]
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name

    best_path = MODELS_DIR / f"{best_model_name}.joblib"
    shutil.copy2(best_path, MODELS_DIR / "best_model.joblib")
    write_json(all_metrics, REPORTS_DIR / "baseline_metrics.json")
    print(json.dumps({"best_model": best_model_name, "metrics": all_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
