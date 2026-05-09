from __future__ import annotations

from pathlib import Path

from ecommerce_absa.features import make_vectorizer


def build_naive_bayes_pipeline(vectorizer_kind: str = "tfidf"):
    try:
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required. Install dependencies with: pip install -r requirements.txt") from exc

    return Pipeline(
        steps=[
            ("vectorizer", make_vectorizer(vectorizer_kind)),
            ("classifier", MultinomialNB()),
        ]
    )


def build_svm_pipeline(vectorizer_kind: str = "tfidf"):
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required. Install dependencies with: pip install -r requirements.txt") from exc

    return Pipeline(
        steps=[
            ("vectorizer", make_vectorizer(vectorizer_kind)),
            ("classifier", LinearSVC(class_weight="balanced")),
        ]
    )


def build_baseline_pipelines(vectorizer_kind: str = "tfidf") -> dict[str, object]:
    return {
        "naive_bayes": build_naive_bayes_pipeline(vectorizer_kind),
        "svm": build_svm_pipeline(vectorizer_kind),
    }


def save_model_artifact(model, path: str | Path, metadata: dict) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required. Install dependencies with: pip install -r requirements.txt") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, output_path)


def load_model_artifact(path: str | Path) -> tuple[object, dict]:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required. Install dependencies with: pip install -r requirements.txt") from exc

    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], artifact.get("metadata", {})
    return artifact, {}
