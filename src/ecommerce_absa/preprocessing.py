from __future__ import annotations

import html
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Mapping

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
SPECIAL_RE = re.compile(r"[^\w\s%]+", re.UNICODE)

DEFAULT_ABBREVIATIONS: dict[str, str] = {
    "sp": "sản phẩm",
    "san pham": "sản phẩm",
    "nv": "nhân viên",
    "dc": "được",
    "đc": "được",
    "ko": "không",
    "k": "không",
    "khong": "không",
    "hok": "không",
    "hong": "không",
    "hông": "không",
    "mn": "mọi người",
    "mk": "mình",
    "m": "mình",
    "ntn": "như thế nào",
    "bt": "bình thường",
    "okie": "ok",
    "shop": "cửa hàng",
    "sh": "cửa hàng",
    "ship": "giao hàng",
    "shipper": "người giao hàng",
    "app": "ứng dụng",
    "ad": "quản trị viên",
    "cskh": "chăm sóc khách hàng",
    "ib": "nhắn tin",
    "rep": "trả lời",
    "feedback": "phản hồi",
}

TEXT_EMOJI_MAP: dict[str, str] = {
    ":)": " vui_vẻ ",
    ":-)": " vui_vẻ ",
    ":d": " vui_vẻ ",
    ":-d": " vui_vẻ ",
    ":(": " buồn ",
    ":-(": " buồn ",
    "<3": " yêu_thích ",
    ":v": " vui_vẻ ",
}

UNICODE_EMOJI_MAP: dict[str, str] = {
    "😀": " vui_vẻ ",
    "😃": " vui_vẻ ",
    "😄": " vui_vẻ ",
    "😁": " vui_vẻ ",
    "😊": " vui_vẻ ",
    "😍": " yêu_thích ",
    "❤": " yêu_thích ",
    "❤️": " yêu_thích ",
    "👍": " hài_lòng ",
    "👎": " không_hài_lòng ",
    "😢": " buồn ",
    "😭": " buồn ",
    "😡": " tức_giận ",
    "😠": " tức_giận ",
}

DEFAULT_MULTIWORD_TERMS = (
    "sản phẩm",
    "nhân viên",
    "dịch vụ",
    "chăm sóc khách hàng",
    "giao hàng",
    "vận chuyển",
    "đóng gói",
    "hoàn tiền",
    "đổi trả",
    "thanh toán",
    "mã giảm giá",
    "khuyến mãi",
    "ứng dụng",
    "đơn hàng",
    "cửa hàng",
    "người giao hàng",
    "hài lòng",
    "không hài lòng",
)


def normalize_unicode(text: object) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFC", html.unescape(str(text)))


def normalize_spaces(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def strip_html_urls(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    return text


def emoji_to_text(text: str) -> str:
    lowered = text.lower()
    for src, dst in TEXT_EMOJI_MAP.items():
        lowered = lowered.replace(src, dst)
    for src, dst in UNICODE_EMOJI_MAP.items():
        lowered = lowered.replace(src, dst)
    return lowered


def standardize_vocabulary(
    text: str,
    mapping: Mapping[str, str] = DEFAULT_ABBREVIATIONS,
) -> str:
    normalized = text
    for source, target in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.IGNORECASE)
        normalized = pattern.sub(target, normalized)
    return normalized


def remove_special_chars(text: str) -> str:
    return SPECIAL_RE.sub(" ", text)


@dataclass
class WordSegmenter:
    """Vietnamese word segmentation wrapper.

    Uses py_vncorenlp when available and configured. The fallback only joins a
    curated list of project terms so baseline scripts can still run on machines
    without the Java/VnCoreNLP model files.
    """

    backend: str = "auto"
    model_dir: str | None = None
    multiword_terms: tuple[str, ...] = DEFAULT_MULTIWORD_TERMS
    _segmenter: object | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "vncorenlp", "none"}:
            raise ValueError("backend must be one of: auto, vncorenlp, none")
        if self.backend == "none":
            return

        model_dir = self.model_dir or os.getenv("VNCORENLP_MODEL_DIR")
        if not model_dir:
            if self.backend == "vncorenlp":
                raise RuntimeError("VNCORENLP_MODEL_DIR is required for backend='vncorenlp'")
            return

        try:
            import py_vncorenlp  # type: ignore

            self._segmenter = py_vncorenlp.VnCoreNLP(
                annotators=["wseg", "pos"],
                save_dir=model_dir,
            )
        except Exception:
            if self.backend == "vncorenlp":
                raise
            self._segmenter = None

    def segment(self, text: str) -> str:
        if not text:
            return ""
        if self._segmenter is not None:
            segmented = self._segmenter.word_segment(text)
            return " ".join(segmented)
        return self._fallback_segment(text)

    def _fallback_segment(self, text: str) -> str:
        segmented = text
        for term in sorted(self.multiword_terms, key=len, reverse=True):
            segmented = re.sub(
                rf"(?<!\w){re.escape(term)}(?!\w)",
                term.replace(" ", "_"),
                segmented,
                flags=re.IGNORECASE,
            )
        return segmented


@dataclass
class TextPreprocessor:
    lowercase: bool = True
    use_word_segmentation: bool = True
    abbreviations: Mapping[str, str] = field(default_factory=lambda: DEFAULT_ABBREVIATIONS)
    segmenter: WordSegmenter = field(default_factory=WordSegmenter)

    def transform(self, text: object) -> str:
        text = normalize_unicode(text)
        text = strip_html_urls(text)
        text = emoji_to_text(text)
        if self.lowercase:
            text = text.lower()
        text = standardize_vocabulary(text, self.abbreviations)
        text = remove_special_chars(text)
        text = normalize_spaces(text)
        if self.use_word_segmentation:
            text = self.segmenter.segment(text)
            text = normalize_spaces(text)
        return text

    def transform_many(self, texts: list[object]) -> list[str]:
        return [self.transform(text) for text in texts]
