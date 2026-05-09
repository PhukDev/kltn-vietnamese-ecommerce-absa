from __future__ import annotations

VIETNAMESE_STOP_WORDS = [
    "a",
    "ai",
    "anh",
    "bị",
    "bởi",
    "các",
    "cái",
    "cần",
    "chị",
    "cho",
    "có",
    "của",
    "đã",
    "đang",
    "đây",
    "để",
    "đến",
    "đi",
    "đó",
    "được",
    "em",
    "gì",
    "hay",
    "khi",
    "là",
    "lại",
    "lên",
    "mà",
    "mình",
    "một",
    "này",
    "nên",
    "nếu",
    "nhé",
    "như",
    "nhưng",
    "những",
    "nó",
    "ở",
    "qua",
    "ra",
    "rằng",
    "rất",
    "rồi",
    "sẽ",
    "thì",
    "trên",
    "và",
    "vào",
    "về",
    "vì",
    "với",
]


def make_vectorizer(
    kind: str = "tfidf",
    *,
    max_features: int = 50000,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: tuple[int, int] = (1, 2),
):
    try:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required. Install dependencies with: pip install -r requirements.txt") from exc

    common_kwargs = {
        "max_features": max_features,
        "min_df": min_df,
        "max_df": max_df,
        "ngram_range": ngram_range,
        "stop_words": VIETNAMESE_STOP_WORDS,
    }
    if kind == "bow":
        return CountVectorizer(**common_kwargs)
    if kind == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, norm="l2", **common_kwargs)
    raise ValueError("kind must be one of: bow, tfidf")
