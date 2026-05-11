from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_absa.compat import disable_optional_pyarrow  # noqa: E402
from ecommerce_absa.preprocessing import TextPreprocessor  # noqa: E402

POSITIVE_HINTS = {
    "tot",
    "tuyet_voi",
    "hai_long",
    "rat_thich",
    "thich",
    "ok",
    "ngon",
    "de_su_dung",
    "nhanh",
    "tien_loi",
    "uy_tin",
    "yeu_thich",
    "vui_ve",
}

NEGATIVE_HINTS = {
    "te",
    "qua_te",
    "loi",
    "lag",
    "chan",
    "lua_dao",
    "that_vong",
    "khong_hai_long",
    "khong_mo_duoc",
    "khong_giao",
    "cham",
    "tre",
    "lau",
    "vo_trach_nhiem",
    "kho_chiu",
    "xao",
    "gia_cao",
    "khong_hien",
    "khong_bat_may",
    "khong_xu_ly",
    "roi_mat",
}

NEGATIVE_PHRASES = (
    "qua te",
    "rat te",
    "that vong",
    "lua dao",
    "khong mo duoc",
    "khong giao",
    "giao cham",
    "giao tre",
    "khong hien",
    "kho chiu",
    "vo trach nhiem",
    "khong bat may",
    "khong xu ly",
)

POSITIVE_PHRASES = (
    "rat thich",
    "rat tot",
    "rat hai long",
    "tuyet voi",
    "uy tin",
    "chat luong",
    "de su dung",
    "giao hang nhanh",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flag likely sentiment mismatches in ABSA annotation batches")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--pattern", default="absa_annotation_batch_*.csv")
    parser.add_argument("--output", default="artifacts/reports/sentiment_mismatch_flags.csv")
    args = parser.parse_args()

    disable_optional_pyarrow()
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pandas is required. Install dependencies with: pip install -r requirements.txt") from exc

    files = sorted(Path(args.input_dir).glob(args.pattern))
    if not files:
        raise SystemExit("No batch files found")

    preprocessor = TextPreprocessor(use_word_segmentation=False)
    flagged_frames = []
    batch_summary = []

    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "content" not in frame.columns or "sentiment" not in frame.columns:
            raise ValueError(f"{path} is missing required columns: content, sentiment")
        batch_flags = build_flags(frame, preprocessor=preprocessor)
        batch_flags["source_file"] = path.name
        batch_flags = batch_flags[output_columns()]
        flagged_frames.append(batch_flags[batch_flags["flagged"]].copy())
        batch_summary.append(
            {
                "file": str(path),
                "rows_loaded": int(len(frame)),
                "rows_flagged": int(batch_flags["flagged"].sum()),
            }
        )

    result = pd.concat(flagged_frames, ignore_index=True) if flagged_frames else pd.DataFrame()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not result.empty:
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=output_columns()).to_csv(output_path, index=False, encoding="utf-8-sig")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "rows_flagged_total": int(len(result)),
                "batches": batch_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_flags(frame, *, preprocessor: TextPreprocessor):
    processed = frame["content"].fillna("").astype(str).map(preprocessor.transform)
    result = frame.copy()
    result["processed_text"] = processed
    scores = processed.map(score_text)
    result["positive_hits"] = scores.map(lambda item: item["positive_hits"])
    result["negative_hits"] = scores.map(lambda item: item["negative_hits"])
    result["content_polarity"] = scores.map(lambda item: item["content_polarity"])
    result["mismatch_reason"] = result.apply(build_reason, axis=1)
    result["flagged"] = result["mismatch_reason"].astype(bool)
    return result


def score_text(text: str) -> dict:
    normalized = ascii_fold(text)
    token_set = set(normalized.split())
    positive_hits = sorted(hit for hit in POSITIVE_HINTS if hit in token_set)
    negative_hits = sorted(hit for hit in NEGATIVE_HINTS if hit in token_set)

    positive_phrase_hits = [phrase for phrase in POSITIVE_PHRASES if phrase in normalized]
    negative_phrase_hits = [phrase for phrase in NEGATIVE_PHRASES if phrase in normalized]

    positive_score = len(positive_hits) + 2 * len(positive_phrase_hits)
    negative_score = len(negative_hits) + 2 * len(negative_phrase_hits)

    if negative_score >= positive_score + 1 and negative_score >= 2:
        content_polarity = "negative"
    elif positive_score >= negative_score + 1 and positive_score >= 2:
        content_polarity = "positive"
    else:
        content_polarity = "unclear"

    return {
        "positive_hits": ", ".join(positive_hits + positive_phrase_hits),
        "negative_hits": ", ".join(negative_hits + negative_phrase_hits),
        "content_polarity": content_polarity,
    }


def build_reason(row) -> str:
    sentiment = str(row.get("sentiment", "")).strip().lower()
    content_polarity = str(row.get("content_polarity", "")).strip().lower()
    if not sentiment or content_polarity == "unclear":
        return ""
    if sentiment == "positive" and content_polarity == "negative":
        return "score-based sentiment is positive but content looks negative"
    if sentiment == "neutral" and content_polarity in {"positive", "negative"}:
        return f"score-based sentiment is neutral but content looks {content_polarity}"
    if sentiment == "negative" and content_polarity == "positive":
        return "score-based sentiment is negative but content looks positive"
    return ""


def ascii_fold(text: str) -> str:
    replacements = {
        "á": "a",
        "à": "a",
        "ả": "a",
        "ã": "a",
        "ạ": "a",
        "ă": "a",
        "ắ": "a",
        "ằ": "a",
        "ẳ": "a",
        "ẵ": "a",
        "ặ": "a",
        "â": "a",
        "ấ": "a",
        "ầ": "a",
        "ẩ": "a",
        "ẫ": "a",
        "ậ": "a",
        "é": "e",
        "è": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ẹ": "e",
        "ê": "e",
        "ế": "e",
        "ề": "e",
        "ể": "e",
        "ễ": "e",
        "ệ": "e",
        "í": "i",
        "ì": "i",
        "ỉ": "i",
        "ĩ": "i",
        "ị": "i",
        "ó": "o",
        "ò": "o",
        "ỏ": "o",
        "õ": "o",
        "ọ": "o",
        "ô": "o",
        "ố": "o",
        "ồ": "o",
        "ổ": "o",
        "ỗ": "o",
        "ộ": "o",
        "ơ": "o",
        "ớ": "o",
        "ờ": "o",
        "ở": "o",
        "ỡ": "o",
        "ợ": "o",
        "ú": "u",
        "ù": "u",
        "ủ": "u",
        "ũ": "u",
        "ụ": "u",
        "ư": "u",
        "ứ": "u",
        "ừ": "u",
        "ử": "u",
        "ữ": "u",
        "ự": "u",
        "ý": "y",
        "ỳ": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ỵ": "y",
        "đ": "d",
    }
    folded = text.lower()
    for src, dst in replacements.items():
        folded = folded.replace(src, dst)
    return " ".join(folded.split())


def output_columns() -> list[str]:
    return [
        "source_file",
        "reviewid",
        "content",
        "sentiment",
        "processed_text",
        "content_polarity",
        "positive_hits",
        "negative_hits",
        "mismatch_reason",
        "flagged",
    ]


if __name__ == "__main__":
    main()
