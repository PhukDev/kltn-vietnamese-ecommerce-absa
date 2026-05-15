from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_ALERT_THUMBS_THRESHOLD
from .preprocessing import normalize_unicode

ASPECT_KEYWORDS: dict[str, list[str]] = {
    "product": [
        "sản phẩm",
        "sản_phẩm",
        "hàng",
        "chất lượng",
        "chất_lượng",
        "mẫu",
        "size",
        "màu",
        "chính hãng",
        "chính_hãng",
    ],
    "price": [
        "giá",
        "rẻ",
        "đắt",
        "tiền",
        "phí",
        "sale",
        "voucher",
        "mã giảm giá",
        "mã_giảm_giá",
        "khuyến mãi",
        "khuyến_mãi",
    ],
    "delivery": [
        "giao hàng",
        "giao_hàng",
        "ship",
        "vận chuyển",
        "vận_chuyển",
        "đóng gói",
        "đóng_gói",
        "người giao hàng",
        "người_giao_hàng",
    ],
    "service": [
        "dịch vụ",
        "dịch_vụ",
        "nhân viên",
        "nhân_viên",
        "hỗ trợ",
        "hỗ_trợ",
        "chăm sóc khách hàng",
        "chăm_sóc_khách_hàng",
        "phản hồi",
        "phản_hồi",
        "đổi trả",
        "đổi_trả",
        "hoàn tiền",
        "hoàn_tiền",
    ],
    "app": [
        "app",
        "ứng dụng",
        "ứng_dụng",
        "lỗi",
        "lag",
        "đăng nhập",
        "đăng_nhập",
        "thanh toán",
        "thanh_toán",
        "đơn hàng",
        "đơn_hàng",
        "cập nhật",
        "cập_nhật",
    ],
}


@dataclass(frozen=True)
class RaceSignal:
    primary_stage: str
    stages: list[str]
    reasons: list[str]
    alert: dict | None


def _text_for_matching(text: object) -> str:
    return normalize_unicode(text).lower()


def detect_aspects(text: object) -> list[str]:
    normalized = _text_for_matching(text)
    found = [
        aspect
        for aspect, keywords in ASPECT_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    ]
    return found or ["general"]


def map_to_race(
    *,
    sentiment: str | None,
    aspects: list[str],
    thumbsupcount: int | float | str | None = None,
    replycontent: object | None = None,
    content: object | None = None,
    alert_threshold: int = DEFAULT_ALERT_THUMBS_THRESHOLD,
) -> RaceSignal:
    thumbs = _safe_int(thumbsupcount)
    reply = _text_for_matching(replycontent)
    text = _text_for_matching(content)
    stages: list[str] = []
    reasons: list[str] = []

    if thumbs >= alert_threshold:
        stages.append("Reach")
        reasons.append(f"review has high thumbsupcount ({thumbs})")
    if any(token in text for token in ["tìm", "xem", "voucher", "sale", "khuyến mãi", "mã giảm giá"]):
        stages.append("Act")
        reasons.append("review mentions discovery, promotion, or app interaction")
    if any(aspect in aspects for aspect in ["product", "price", "delivery", "app"]):
        stages.append("Convert")
        reasons.append("review reflects purchase, product, delivery, price, or checkout experience")
    if sentiment == "negative" or reply or "service" in aspects:
        stages.append("Engage")
        reasons.append("review needs retention, support, reply, or eWOM management")

    if not stages:
        stages.append("Act")
        reasons.append("general customer interaction")

    alert = build_alert(
        sentiment=sentiment,
        aspects=aspects,
        thumbsupcount=thumbs,
        alert_threshold=alert_threshold,
    )
    primary_stage = "Engage" if alert else stages[-1]
    return RaceSignal(primary_stage=primary_stage, stages=_dedupe(stages), reasons=_dedupe(reasons), alert=alert)


def build_alert(
    *,
    sentiment: str | None,
    aspects: list[str],
    thumbsupcount: int,
    alert_threshold: int = DEFAULT_ALERT_THUMBS_THRESHOLD,
) -> dict | None:
    if sentiment != "negative":
        return None
    high_visibility = thumbsupcount >= alert_threshold
    if "service" in aspects and high_visibility:
        return {
            "severity": "high",
            "stage": "Engage",
            "message": "Negative service review has high thumbsupcount; CSKH should intervene.",
        }
    if high_visibility:
        return {
            "severity": "medium",
            "stage": "Engage",
            "message": "Negative high-visibility review should be monitored.",
        }
    return None


def analyze_business_signal(
    *,
    content: object,
    sentiment: str | None,
    aspects: list[str] | None = None,
    thumbsupcount: int | float | str | None = None,
    replycontent: object | None = None,
) -> dict:
    aspects = aspects or detect_aspects(content)
    race = map_to_race(
        sentiment=sentiment,
        aspects=aspects,
        thumbsupcount=thumbsupcount,
        replycontent=replycontent,
        content=content,
    )
    return {
        "aspects": aspects,
        "race": {
            "primary_stage": race.primary_stage,
            "stages": race.stages,
            "reasons": race.reasons,
        },
        "alert": race.alert,
    }


def _safe_int(value: int | float | str | None) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
