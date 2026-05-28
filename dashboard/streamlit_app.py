from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # type: ignore  # noqa: E402

from ecommerce_absa.api import PredictionService  # noqa: E402
from ecommerce_absa.config import DATA_PATH, MODELS_DIR  # noqa: E402
from ecommerce_absa.data import load_reviews  # noqa: E402


@st.cache_resource(show_spinner=False)
def get_prediction_service(model_path: str | None, aspect_model_path: str | None):
    return PredictionService(model_path, aspect_model_path)


@st.cache_data(show_spinner=False)
def load_dashboard_frame(path: str, limit: int, model_path: str | None, aspect_model_path: str | None):
    # Check if the CSV is already pre-predicted to skip model inference for instant loading
    import pandas as pd
    try:
        header_check = pd.read_csv(path, nrows=2, encoding="utf-8-sig")
        is_prepredicted = "sentiment" in header_check.columns and "aspects" in header_check.columns
    except Exception:
        is_prepredicted = False

    if is_prepredicted:
        frame = pd.read_csv(path, nrows=limit, encoding="utf-8-sig")
        frame["aspect_source"] = "prepredicted"
        frame["alert"] = frame["alert"].fillna("")
        frame["replycontent"] = frame["replycontent"].fillna("")
        return frame

    columns = ["reviewid", "content", "score", "thumbsupcount", "replycontent", "appid", "at"]
    frame = load_reviews(path, columns=columns, limit=limit)

    service = get_prediction_service(model_path, aspect_model_path)

    results = []
    for idx, row in frame.iterrows():
        payload = {
            "content": row.get("content", ""),
            "score": row.get("score"),
            "thumbsupcount": row.get("thumbsupcount", 0),
            "replycontent": row.get("replycontent", ""),
        }
        res = service.predict(payload)
        results.append(res)

    frame["sentiment"] = [r["sentiment"] for r in results]
    frame["aspects"] = [", ".join(r["aspects"]) for r in results]
    frame["race_stage"] = [r["race"]["primary_stage"] for r in results]
    frame["alert"] = [r["alert"].get("message", "") if r["alert"] else "" for r in results]
    frame["aspect_source"] = [r["metadata"]["aspect_source"] for r in results]

    return frame


def main() -> None:
    st.set_page_config(page_title="Ecommerce ABSA Dashboard", layout="wide")
    st.title("Ecommerce ABSA Dashboard")

    with st.sidebar:
        st.subheader("Cấu hình Dữ liệu & Mô hình")
        data_path = st.text_input("Đường dẫn file CSV", value=str(DATA_PATH))

        st.write("---")
        st.markdown("**Lựa chọn Mô hình ABSA**")
        st.caption("Mẹo: Nhập file PyTorch .pt (ví dụ: `artifacts/models/absa_phobert_gold_eval.pt`) để dùng PhoBERT, hoặc .joblib cho Baseline.")

        aspect_model_path = st.text_input(
            "Mô hình Khía cạnh (Aspect Model)",
            value=str(MODELS_DIR / "absa_aspect_baseline.joblib"),
        )
        sentiment_model_path = st.text_input(
            "Mô hình Cảm xúc (Sentiment Model - Tùy chọn)",
            value="",
        )

        st.write("---")
        limit = st.number_input("Số lượng dòng nạp vào", min_value=1000, max_value=300000, value=50000, step=1000)

    frame = load_dashboard_frame(
        data_path,
        int(limit),
        sentiment_model_path or None,
        aspect_model_path or None,
    )
    total_reviews = len(frame)
    alert_count = int(frame["alert"].astype(bool).sum())
    negative_rate = float((frame["sentiment"] == "negative").mean()) if total_reviews else 0.0
    reply_rate = float(frame["replycontent"].fillna("").astype(str).str.strip().astype(bool).mean()) if total_reviews else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reviews", f"{total_reviews:,}")
    col2.metric("Negative rate", f"{negative_rate:.1%}")
    col3.metric("Reply rate", f"{reply_rate:.1%}")
    col4.metric("Alerts", f"{alert_count:,}")
    st.caption(f"Aspect source: {frame['aspect_source'].iloc[0] if total_reviews else 'n/a'}")

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment distribution")
        st.bar_chart(frame["sentiment"].value_counts())
    with right:
        st.subheader("RACE stage distribution")
        st.bar_chart(frame["race_stage"].value_counts())

    st.subheader("High priority alerts")
    alerts = frame[frame["alert"].astype(bool)].copy()
    if alerts.empty:
        st.info("No alert in loaded sample.")
    else:
        st.dataframe(
            alerts[["reviewid", "content", "score", "thumbsupcount", "aspects", "race_stage", "alert"]]
            .sort_values("thumbsupcount", ascending=False)
            .head(100),
            use_container_width=True,
        )

    st.subheader("Review explorer")
    selected_sentiment = st.multiselect(
        "Sentiment",
        options=sorted(frame["sentiment"].dropna().unique()),
        default=sorted(frame["sentiment"].dropna().unique()),
    )
    filtered = frame[frame["sentiment"].isin(selected_sentiment)]
    st.dataframe(
        filtered[
            ["reviewid", "content", "score", "thumbsupcount", "sentiment", "aspects", "aspect_source", "race_stage", "appid"]
        ].head(500),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
