from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # type: ignore  # noqa: E402

from ecommerce_absa.config import DATA_PATH  # noqa: E402
from ecommerce_absa.data import load_reviews, score_to_sentiment  # noqa: E402
from ecommerce_absa.race import analyze_business_signal  # noqa: E402


@st.cache_data(show_spinner=False)
def load_dashboard_frame(path: str, limit: int):
    columns = ["reviewid", "content", "score", "thumbsupcount", "replycontent", "appid", "at"]
    frame = load_reviews(path, columns=columns, limit=limit)
    frame["sentiment"] = frame["score"].map(score_to_sentiment)
    signals = frame.apply(
        lambda row: analyze_business_signal(
            content=row.get("content", ""),
            sentiment=row.get("sentiment"),
            thumbsupcount=row.get("thumbsupcount", 0),
            replycontent=row.get("replycontent", ""),
        ),
        axis=1,
    )
    frame["aspects"] = signals.map(lambda item: ", ".join(item["aspects"]))
    frame["race_stage"] = signals.map(lambda item: item["race"]["primary_stage"])
    frame["alert"] = signals.map(lambda item: (item["alert"] or {}).get("message", ""))
    return frame


def main() -> None:
    st.set_page_config(page_title="Ecommerce ABSA Dashboard", layout="wide")
    st.title("Ecommerce ABSA Dashboard")

    with st.sidebar:
        data_path = st.text_input("CSV path", value=str(DATA_PATH))
        limit = st.number_input("Rows to load", min_value=1000, max_value=300000, value=50000, step=1000)

    frame = load_dashboard_frame(data_path, int(limit))
    total_reviews = len(frame)
    alert_count = int(frame["alert"].astype(bool).sum())
    negative_rate = float((frame["sentiment"] == "negative").mean()) if total_reviews else 0.0
    reply_rate = float(frame["replycontent"].fillna("").astype(str).str.strip().astype(bool).mean()) if total_reviews else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reviews", f"{total_reviews:,}")
    col2.metric("Negative rate", f"{negative_rate:.1%}")
    col3.metric("Reply rate", f"{reply_rate:.1%}")
    col4.metric("Alerts", f"{alert_count:,}")

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
        filtered[["reviewid", "content", "score", "thumbsupcount", "sentiment", "aspects", "race_stage", "appid"]].head(500),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
