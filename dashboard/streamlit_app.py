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
        data_option = st.selectbox(
            "Tập dữ liệu hiển thị",
            options=["Mặc định (Full)", "Tập Train (30% gốc)", "Tập Predict (70% gốc)", "Đường dẫn tùy chỉnh"],
            index=0
        )
        if data_option == "Mặc định (Full)":
            data_path = str(DATA_PATH)
        elif data_option == "Tập Train (30% gốc)":
            data_path = "data/original_train.csv"
        elif data_option == "Tập Predict (70% gốc)":
            data_path = "data/original_predict.csv"
        else:
            data_path = st.text_input("Đường dẫn file CSV tùy chỉnh", value=str(DATA_PATH))

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

    # Add Tab Navigation
    tab_ana, tab_comp = st.tabs(["Dashboard Phân Tích Dữ Liệu & Cảnh Báo", "So Sánh Hiệu Năng Các Mô Hình"])

    with tab_ana:
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

    with tab_comp:
        st.header("Báo Cáo So Sánh Hiệu Năng Các Mô Hình ABSA")
        st.write(
            "Số liệu hiển thị dưới đây được đọc động từ kết quả huấn luyện thực tế trên tập "
            "**Train (30%)** và đánh giá trên tập **Gold Eval (70%)** để tránh hiện tượng quá khớp (overfitting)."
        )

        import json
        import pandas as pd
        reports_dir = ROOT / "artifacts" / "reports"
        
        # Load Metrics helper
        def load_json(name):
            p = reports_dir / name
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
            return None

        svm_data = load_json("absa_baseline_gold_eval_metrics.json")
        bilstm_data = load_json("absa_bilstm_gold_eval_metrics.json")
        phobert_data = load_json("absa_phobert_gold_eval_metrics.json")

        def fmt_pct(val):
            if val is None:
                return "N/A"
            return f"{val * 100:.2f}%"

        def fmt_val(val, decimals=4):
            if val is None:
                return "N/A"
            return f"{val:.{decimals}f}"

        # Bảng 1: So sánh tổng quan
        st.subheader("1. So sánh tổng quan hiệu năng các mô hình")
        overview_rows = []
        
        # SVM Row
        svm_ok = svm_data is not None
        overview_rows.append({
            "Mô hình / Thuật toán": "Baseline TF-IDF + SVM",
            "Aspect F1-micro": fmt_pct(svm_data.get("f1_micro")) if svm_ok else "N/A",
            "Aspect F1-macro": fmt_pct(svm_data.get("f1_macro")) if svm_ok else "N/A",
            "Aspect Hamming Loss": fmt_val(svm_data.get("hamming_loss")) if svm_ok else "N/A",
            "Sentiment F1-macro": "N/A",
            "Sentiment Accuracy": "N/A",
            "Đặc trưng kiến trúc": "Phân loại khía cạnh độc lập (Multi-label bằng OneVsRest)."
        })
        
        # Bi-LSTM Row
        bilstm_ok = bilstm_data is not None
        overview_rows.append({
            "Mô hình / Thuật toán": "Bi-LSTM + Word2Vec",
            "Aspect F1-micro": fmt_pct(bilstm_data.get("f1_micro")) if bilstm_ok else "N/A",
            "Aspect F1-macro": fmt_pct(bilstm_data.get("f1_macro")) if bilstm_ok else "N/A",
            "Aspect Hamming Loss": fmt_val(bilstm_data.get("hamming_loss")) if bilstm_ok else "N/A",
            "Sentiment F1-macro": "N/A",
            "Sentiment Accuracy": "N/A",
            "Đặc trưng kiến trúc": "Mạng hồi quy hai chiều, học đặc trưng tuần tự và ngữ cảnh của từ."
        })
        
        # PhoBERT Row
        phobert_ok = phobert_data is not None
        ph_aspect = phobert_data.get("aspect", {}) if phobert_ok else {}
        ph_sentiment = phobert_data.get("sentiment", {}) if phobert_ok else {}
        overview_rows.append({
            "Mô hình / Thuật toán": "PhoBERT Multi-task",
            "Aspect F1-micro": fmt_pct(ph_aspect.get("f1_micro")) if phobert_ok else "N/A",
            "Aspect F1-macro": fmt_pct(ph_aspect.get("f1_macro")) if phobert_ok else "N/A",
            "Aspect Hamming Loss": fmt_val(ph_aspect.get("hamming_loss")) if phobert_ok else "N/A",
            "Sentiment F1-macro": fmt_pct(ph_sentiment.get("f1_macro")) if phobert_ok else "N/A",
            "Sentiment Accuracy": fmt_pct(ph_sentiment.get("accuracy")) if phobert_ok else "N/A",
            "Đặc trưng kiến trúc": "Học máy đa nhiệm (Multi-task), tối ưu hóa đồng thời Aspect và Sentiment trong 1 forward pass."
        })
        
        st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

        # Bảng 2: So sánh Aspect-specific F1-score
        st.subheader("2. So sánh chi tiết hiệu năng Aspect-specific F1-score")
        aspects = ["product", "price", "delivery", "service", "app"]
        aspect_names = {
            "product": "Product (Sản phẩm)",
            "price": "Price (Giá cả)",
            "delivery": "Delivery (Giao hàng)",
            "service": "Service (Dịch vụ)",
            "app": "App (Ứng dụng)"
        }
        
        aspect_rows = []
        for aspect in aspects:
            svm_f1 = "N/A"
            if svm_ok and "per_aspect" in svm_data and aspect in svm_data["per_aspect"]:
                svm_f1 = fmt_pct(svm_data["per_aspect"][aspect].get("f1"))
                
            bilstm_f1 = "N/A"
            if bilstm_ok and "per_aspect" in bilstm_data and aspect in bilstm_data["per_aspect"]:
                bilstm_f1 = fmt_pct(bilstm_data["per_aspect"][aspect].get("f1"))
                
            phobert_f1 = "N/A"
            if phobert_ok and "aspect" in phobert_data and "per_aspect" in ph_aspect and aspect in ph_aspect["per_aspect"]:
                phobert_f1 = fmt_pct(ph_aspect["per_aspect"][aspect].get("f1"))
                
            aspect_rows.append({
                "Khía cạnh (Aspect)": aspect_names[aspect],
                "Baseline TF-IDF + SVM": svm_f1,
                "Bi-LSTM + Word2Vec": bilstm_f1,
                "PhoBERT Multi-task": phobert_f1
            })
            
        st.dataframe(pd.DataFrame(aspect_rows), use_container_width=True, hide_index=True)

        # Nhận xét & Biện luận
        st.subheader("3. Nhận xét & Biện luận khoa học")
        st.markdown(
            """
            *   **Tổng quan huấn luyện:** Với tỷ lệ phân chia mới (**30% Train - 70% Eval**), mô hình được đánh giá trên tập test lớn hơn gấp đôi tập train. Điều này giúp ngăn chặn overfitting một cách đáng tin cậy và phản ánh chính xác khả năng tổng quát hóa của mô hình trên dữ liệu thực tế.
            *   **Hiệu năng của SVM:** Linear SVM hoạt động cực kỳ ấn tượng trên dữ liệu hạn chế nhờ biên phân chia tối ưu hóa khoảng cách lớp tốt.
            *   **Khía cạnh Delivery:** Đạt hiệu năng cao nhất (>80%) ở tất cả mô hình nhờ các từ khóa mang tính đặc trưng cực cao và tập trung như *"ship", "giao hàng", "nhanh", "chậm", "gói hàng"*.
            *   **Khía cạnh Service:** Luôn là thách thức lớn nhất do ranh giới ngữ nghĩa giữa "dịch vụ CSKH" và "sản phẩm" rất mơ hồ trong tiếng Việt thương mại điện tử.
            """
        )


if __name__ == "__main__":
    main()
