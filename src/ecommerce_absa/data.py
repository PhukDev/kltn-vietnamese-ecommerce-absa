from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import SCORE_COLUMN, TEXT_COLUMN
from .preprocessing import TextPreprocessor


def score_to_sentiment(score: object) -> str | None:
    try:
        value = int(float(str(score).strip()))
    except (TypeError, ValueError):
        return None
    if value in {1, 2}:
        return "negative"
    if value == 3:
        return "neutral"
    if value in {4, 5}:
        return "positive"
    return None


def require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pandas is required. Install dependencies with: pip install -r requirements.txt") from exc
    return pd


def load_reviews(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    limit: int | None = None,
):
    pd = require_pandas()
    return pd.read_csv(path, usecols=list(columns) if columns else None, nrows=limit)


def prepare_sentiment_frame(
    frame,
    *,
    preprocessor: TextPreprocessor | None = None,
    text_column: str = TEXT_COLUMN,
    score_column: str = SCORE_COLUMN,
):
    preprocessor = preprocessor or TextPreprocessor()
    prepared = frame.copy()
    prepared[text_column] = prepared[text_column].fillna("").astype(str)
    prepared["sentiment"] = prepared[score_column].map(score_to_sentiment)
    prepared = prepared.dropna(subset=["sentiment"])
    prepared = prepared[prepared[text_column].str.strip().astype(bool)]
    prepared["processed_text"] = prepared[text_column].map(preprocessor.transform)
    prepared = prepared[prepared["processed_text"].str.strip().astype(bool)]
    return prepared
