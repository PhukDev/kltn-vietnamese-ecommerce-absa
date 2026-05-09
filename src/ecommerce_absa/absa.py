from __future__ import annotations

from math import isnan
from typing import Iterable

from .config import ABSA_ASPECT_COLUMNS, TEXT_COLUMN
from .preprocessing import TextPreprocessor

POSITIVE_ASPECT_VALUES = {"1", "true", "yes", "y", "x"}
NEGATIVE_ASPECT_VALUES = {"0", "false", "no", "n"}


def normalize_aspect_label(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and isnan(value):
            return None
        if value in {0, 1}:
            return int(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in POSITIVE_ASPECT_VALUES:
        return 1
    if text in NEGATIVE_ASPECT_VALUES:
        return 0
    return None


def require_absa_columns(
    frame,
    aspect_columns: Iterable[str] = ABSA_ASPECT_COLUMNS,
    text_column: str = TEXT_COLUMN,
) -> None:
    required = [text_column, *aspect_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required ABSA annotation columns: {', '.join(missing)}")


def prepare_absa_annotation_frame(
    frame,
    *,
    preprocessor: TextPreprocessor | None = None,
    text_column: str = TEXT_COLUMN,
    aspect_columns: Iterable[str] = ABSA_ASPECT_COLUMNS,
):
    aspect_columns = list(aspect_columns)
    require_absa_columns(frame, aspect_columns, text_column)

    preprocessor = preprocessor or TextPreprocessor()
    prepared = frame.copy()
    prepared[text_column] = prepared[text_column].fillna("").astype(str)
    valid_mask = prepared[text_column].str.strip().astype(bool)

    for column in aspect_columns:
        normalized = prepared[column].map(normalize_aspect_label)
        prepared[column] = normalized
        valid_mask = valid_mask & normalized.notna()

    prepared = prepared[valid_mask].copy()
    for column in aspect_columns:
        prepared[column] = prepared[column].astype(int)

    prepared["processed_text"] = prepared[text_column].map(preprocessor.transform)
    prepared = prepared[prepared["processed_text"].str.strip().astype(bool)]
    return prepared
