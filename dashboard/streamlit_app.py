from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # type: ignore

from ecommerce_absa.api import PredictionService
from ecommerce_absa.config import DATA_PATH, MODELS_DIR
from ecommerce_absa.data import load_reviews

# Define Translation Dictionary
TRANSLATIONS = {
    "vi": {
        "title": "Ecommerce ABSA Dashboard",
        "sidebar_title": "Cấu hình Dữ liệu & Mô hình",
        "dataset_label": "Tập dữ liệu hiển thị",
        "dataset_options": ["Mặc định (Full)", "Tập Train (70% gốc)", "Tập Predict (30% gốc)", "Đường dẫn tùy chỉnh"],
        "custom_path_label": "Đường dẫn file CSV tùy chỉnh",
        "model_section": "Lựa chọn Mô hình ABSA",
        "model_tip": "Mẹo: Nhập file PyTorch .pt (ví dụ: `artifacts/models/absa_phobert_gold_eval.pt`) để dùng PhoBERT, hoặc .joblib cho Baseline.",
        "aspect_model_label": "Mô hình Khía cạnh (Aspect Model)",
        "sentiment_model_label": "Mô hình Cảm xúc (Sentiment Model - Tùy chọn)",
        "limit_label": "Số lượng dòng nạp vào",
        "tab_analysis": "Dashboard Phân Tích Dữ Liệu & Cảnh Báo",
        "tab_comparison": "So Sánh Hiệu Năng Các Mô Hình",
        "metric_reviews": "Đánh giá",
        "metric_neg_rate": "Tỷ lệ tiêu cực",
        "metric_reply_rate": "Tỷ lệ phản hồi",
        "metric_alerts": "Cảnh báo",
        "aspect_source": "Nguồn nhãn khía cạnh",
        "sentiment_dist": "Phân phối cảm xúc",
        "race_dist": "Phân phối giai đoạn RACE",
        "high_priority_alerts": "Cảnh báo ưu tiên cao",
        "no_alerts": "Không có cảnh báo nào trong dữ liệu được nạp.",
        "review_explorer": "Trình khám phá đánh giá",
        "sentiment_label": "Cảm xúc",
        "comparison_title": "Báo Cáo So Sánh Hiệu Năng Các Mô Hình ABSA",
        "comparison_desc": "Số liệu hiển thị dưới đây được đọc động từ kết quả huấn luyện thực tế trên tập **Train (70%)** và đánh giá trên tập độc lập **Gold Eval** để tránh hiện tượng quá khớp (overfitting).",
        "sub_overview": "1. So sánh tổng quan hiệu năng các mô hình",
        "sub_aspect_detail": "2. So sánh chi tiết hiệu năng Aspect-specific F1-score",
        "sub_comments": "3. Nhận xét & Biện luận khoa học",
        "comments_text": """
            *   **Tổng quan huấn luyện:** Với tỷ lệ phân chia mới (**70% Train - 30% Predict**), mô hình được tối ưu hóa trên tập train lớn chiếm 70% dữ liệu sạch. Điều này giúp cải thiện đáng kể độ chính xác và khả năng học sâu của mô hình PhoBERT trên tập dữ liệu tiếng Việt thương mại điện tử lớn.
            *   **Hiệu năng của SVM:** Linear SVM hoạt động cực kỳ ấn tượng trên dữ liệu hạn chế nhờ biên phân chia tối ưu hóa khoảng cách lớp tốt.
            *   **Khía cạnh Delivery:** Đạt hiệu năng cao nhất (>80%) ở tất cả mô hình nhờ các từ khóa mang tính đặc trưng cực cao và tập trung như *"ship", "giao hàng", "nhanh", "chậm", "gói hàng"*.
            *   **Khía cạnh Service:** Luôn là thách thức lớn nhất do ranh giới ngữ nghĩa giữa "dịch vụ CSKH" và "sản phẩm" rất mơ hồ trong tiếng Việt thương mại điện tử.
            """,
        "architecture_svm": "Phân loại khía cạnh độc lập (Multi-label bằng OneVsRest).",
        "architecture_bilstm": "Mạng hồi quy hai chiều, học đặc trưng tuần tự và ngữ cảnh của từ.",
        "architecture_phobert": "Học máy đa nhiệm (Multi-task), tối ưu hóa đồng thời Aspect và Sentiment trong 1 forward pass.",
        "col_model": "Mô hình / Thuật toán",
        "col_architecture": "Đặc trưng kiến trúc",
        "col_aspect": "Khía cạnh (Aspect)",
    },
    "en": {
        "title": "Ecommerce ABSA Dashboard",
        "sidebar_title": "Data & Model Configuration",
        "dataset_label": "Display Dataset",
        "dataset_options": ["Default (Full)", "Train Set (70% original)", "Predict Set (30% original)", "Custom Path"],
        "custom_path_label": "Custom CSV File Path",
        "model_section": "ABSA Model Selection",
        "model_tip": "Tip: Enter PyTorch .pt (e.g. `artifacts/models/absa_phobert_gold_eval.pt`) for PhoBERT, or .joblib for Baseline.",
        "aspect_model_label": "Aspect Model Path",
        "sentiment_model_label": "Sentiment Model Path (Optional)",
        "limit_label": "Row Limit",
        "tab_analysis": "Data Analysis & Alerts Dashboard",
        "tab_comparison": "Models Performance Comparison",
        "metric_reviews": "Reviews",
        "metric_neg_rate": "Negative rate",
        "metric_reply_rate": "Reply rate",
        "metric_alerts": "Alerts",
        "aspect_source": "Aspect source",
        "sentiment_dist": "Sentiment distribution",
        "race_dist": "RACE stage distribution",
        "high_priority_alerts": "High priority alerts",
        "no_alerts": "No alert in loaded sample.",
        "review_explorer": "Review explorer",
        "sentiment_label": "Sentiment",
        "comparison_title": "ABSA Models Performance Comparison Report",
        "comparison_desc": "The metrics shown below are dynamically read from actual training results on the **Train (70%)** and evaluated on the independent **Gold Eval** set to avoid overfitting.",
        "sub_overview": "1. Overview Model Performance Comparison",
        "sub_aspect_detail": "2. Detailed Aspect-specific F1-score Comparison",
        "sub_comments": "3. Scientific Comments & Analysis",
        "comments_text": """
            *   **Training Overview:** With the new split ratio (**70% Train - 30% Predict**), the model is optimized on a large training set containing 70% of the clean data. This significantly improves accuracy and PhoBERT's representation learning on a large Vietnamese e-commerce review dataset.
            *   **SVM Performance:** Linear SVM performs exceptionally well on limited data due to its optimized class separation margins.
            *   **Delivery Aspect:** Achieved the highest performance (>80%) across all models thanks to highly specific and concentrated keywords like *"ship", "delivery", "fast", "slow", "packaging"*.
            *   **Service Aspect:** Consistently remains the biggest challenge due to the vague semantic boundaries between "customer support service" and "product quality" in Vietnamese e-commerce.
            """,
        "architecture_svm": "Independent aspect classification (Multi-label via OneVsRest).",
        "architecture_bilstm": "Bidirectional LSTM, learning sequential and contextual word representations.",
        "architecture_phobert": "Multi-task learning, optimizing Aspect and Sentiment simultaneously in a single forward pass.",
        "col_model": "Model / Algorithm",
        "col_architecture": "Architecture Features",
        "col_aspect": "Aspect",
    }
}


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

    # Add Language Switcher to the top of Sidebar
    with st.sidebar:
        st.subheader("Language / Ngôn ngữ")
        lang_option = st.selectbox(
            "Select Dashboard Language",
            options=["Tiếng Việt", "English"],
            index=0,
            label_visibility="collapsed"
        )
        lang = "vi" if lang_option == "Tiếng Việt" else "en"
        st.write("---")

    st.title(TRANSLATIONS[lang]["title"])

    with st.sidebar:
        st.subheader(TRANSLATIONS[lang]["sidebar_title"])
        data_option = st.selectbox(
            TRANSLATIONS[lang]["dataset_label"],
            options=TRANSLATIONS[lang]["dataset_options"],
            index=2
        )
        
        if data_option == TRANSLATIONS[lang]["dataset_options"][0]:
            data_path = str(DATA_PATH)
        elif data_option == TRANSLATIONS[lang]["dataset_options"][1]:
            data_path = "data/original_train.csv"
        elif data_option == TRANSLATIONS[lang]["dataset_options"][2]:
            data_path = "data/original_predict.csv"
        else:
            data_path = st.text_input(TRANSLATIONS[lang]["custom_path_label"], value=str(DATA_PATH))

        st.write("---")
        st.markdown(f"**{TRANSLATIONS[lang]['model_section']}**")
        st.caption(TRANSLATIONS[lang]["model_tip"])

        aspect_model_path = st.text_input(
            TRANSLATIONS[lang]["aspect_model_label"],
            value=str(MODELS_DIR / "absa_aspect_baseline.joblib"),
        )
        default_phobert_path = MODELS_DIR / "absa_phobert_gold_eval.pt"
        sentiment_model_path = st.text_input(
            TRANSLATIONS[lang]["sentiment_model_label"],
            value=str(default_phobert_path) if default_phobert_path.exists() else "",
        )

        st.write("---")
        limit = st.number_input(TRANSLATIONS[lang]["limit_label"], min_value=1000, max_value=300000, value=50000, step=1000)

    # Add Tab Navigation with dynamic translation
    tab_ana, tab_comp = st.tabs([TRANSLATIONS[lang]["tab_analysis"], TRANSLATIONS[lang]["tab_comparison"]])

    # Define dataframe column renaming mapping
    display_cols = {
        "reviewid": "Review ID",
        "content": "Content / Nội dung" if lang == "vi" else "Content",
        "score": "Score / Điểm" if lang == "vi" else "Score",
        "thumbsupcount": "Likes / Thích" if lang == "vi" else "Likes",
        "sentiment": "Sentiment / Cảm xúc" if lang == "vi" else "Sentiment",
        "aspects": "Aspects / Khía cạnh" if lang == "vi" else "Aspects",
        "aspect_source": "Source / Nguồn" if lang == "vi" else "Source",
        "race_stage": "RACE Stage",
        "alert": "Alert / Cảnh báo" if lang == "vi" else "Alert",
        "appid": "App ID"
    }

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
        col1.metric(TRANSLATIONS[lang]["metric_reviews"], f"{total_reviews:,}")
        col2.metric(TRANSLATIONS[lang]["metric_neg_rate"], f"{negative_rate:.1%}")
        col3.metric(TRANSLATIONS[lang]["metric_reply_rate"], f"{reply_rate:.1%}")
        col4.metric(TRANSLATIONS[lang]["metric_alerts"], f"{alert_count:,}")
        st.caption(f"{TRANSLATIONS[lang]['aspect_source']}: {frame['aspect_source'].iloc[0] if total_reviews else 'n/a'}")

        left, right = st.columns(2)
        with left:
            st.subheader(TRANSLATIONS[lang]["sentiment_dist"])
            st.bar_chart(frame["sentiment"].value_counts())
        with right:
            st.subheader(TRANSLATIONS[lang]["race_dist"])
            st.bar_chart(frame["race_stage"].value_counts())

        st.subheader(TRANSLATIONS[lang]["high_priority_alerts"])
        alerts = frame[frame["alert"].astype(bool)].copy()
        if alerts.empty:
            st.info(TRANSLATIONS[lang]["no_alerts"])
        else:
            alerts_display = alerts[["reviewid", "content", "score", "thumbsupcount", "aspects", "race_stage", "alert"]].copy()
            alerts_display = alerts_display.rename(columns=display_cols)
            st.dataframe(
                alerts_display.sort_values(display_cols["thumbsupcount"], ascending=False).head(100),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader(TRANSLATIONS[lang]["review_explorer"])
        selected_sentiment = st.multiselect(
            TRANSLATIONS[lang]["sentiment_label"],
            options=sorted(frame["sentiment"].dropna().unique()),
            default=sorted(frame["sentiment"].dropna().unique()),
        )
        filtered = frame[frame["sentiment"].isin(selected_sentiment)]
        
        filtered_display = filtered[
            ["reviewid", "content", "score", "thumbsupcount", "sentiment", "aspects", "aspect_source", "race_stage", "appid"]
        ].copy()
        filtered_display = filtered_display.rename(columns=display_cols)
        st.dataframe(
            filtered_display.head(500),
            use_container_width=True,
            hide_index=True,
        )

    with tab_comp:
        st.header(TRANSLATIONS[lang]["comparison_title"])
        st.write(TRANSLATIONS[lang]["comparison_desc"])

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
        st.subheader(TRANSLATIONS[lang]["sub_overview"])
        overview_rows = []
        
        # SVM Row
        svm_ok = svm_data is not None
        overview_rows.append({
            TRANSLATIONS[lang]["col_model"]: "Baseline TF-IDF + SVM",
            "Aspect F1-micro": fmt_pct(svm_data.get("f1_micro")) if svm_ok else "N/A",
            "Aspect F1-macro": fmt_pct(svm_data.get("f1_macro")) if svm_ok else "N/A",
            "Aspect Hamming Loss": fmt_val(svm_data.get("hamming_loss")) if svm_ok else "N/A",
            "Sentiment F1-macro": "N/A",
            "Sentiment Accuracy": "N/A",
            TRANSLATIONS[lang]["col_architecture"]: TRANSLATIONS[lang]["architecture_svm"]
        })
        
        # Bi-LSTM Row
        bilstm_ok = bilstm_data is not None
        overview_rows.append({
            TRANSLATIONS[lang]["col_model"]: "Bi-LSTM + Word2Vec",
            "Aspect F1-micro": fmt_pct(bilstm_data.get("f1_micro")) if bilstm_ok else "N/A",
            "Aspect F1-macro": fmt_pct(bilstm_data.get("f1_macro")) if bilstm_ok else "N/A",
            "Aspect Hamming Loss": fmt_val(bilstm_data.get("hamming_loss")) if bilstm_ok else "N/A",
            "Sentiment F1-macro": "N/A",
            "Sentiment Accuracy": "N/A",
            TRANSLATIONS[lang]["col_architecture"]: TRANSLATIONS[lang]["architecture_bilstm"]
        })
        
        # PhoBERT Row
        phobert_ok = phobert_data is not None
        ph_aspect = phobert_data.get("aspect", {}) if phobert_ok else {}
        ph_sentiment = phobert_data.get("sentiment", {}) if phobert_ok else {}
        overview_rows.append({
            TRANSLATIONS[lang]["col_model"]: "PhoBERT Multi-task",
            "Aspect F1-micro": fmt_pct(ph_aspect.get("f1_micro")) if phobert_ok else "N/A",
            "Aspect F1-macro": fmt_pct(ph_aspect.get("f1_macro")) if phobert_ok else "N/A",
            "Aspect Hamming Loss": fmt_val(ph_aspect.get("hamming_loss")) if phobert_ok else "N/A",
            "Sentiment F1-macro": fmt_pct(ph_sentiment.get("f1_macro")) if phobert_ok else "N/A",
            "Sentiment Accuracy": fmt_pct(ph_sentiment.get("accuracy")) if phobert_ok else "N/A",
            TRANSLATIONS[lang]["col_architecture"]: TRANSLATIONS[lang]["architecture_phobert"]
        })
        
        st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

        # Bảng 2: So sánh Aspect-specific F1-score
        st.subheader(TRANSLATIONS[lang]["sub_aspect_detail"])
        aspects = ["product", "price", "delivery", "service", "app"]
        aspect_names = {
            "product": "Product (Sản phẩm)" if lang == "vi" else "Product",
            "price": "Price (Giá cả)" if lang == "vi" else "Price",
            "delivery": "Delivery (Giao hàng)" if lang == "vi" else "Delivery",
            "service": "Service (Dịch vụ)" if lang == "vi" else "Service",
            "app": "App (Ứng dụng)" if lang == "vi" else "App"
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
                TRANSLATIONS[lang]["col_aspect"]: aspect_names[aspect],
                "Baseline TF-IDF + SVM": svm_f1,
                "Bi-LSTM + Word2Vec": bilstm_f1,
                "PhoBERT Multi-task": phobert_f1
            })
            
        st.dataframe(pd.DataFrame(aspect_rows), use_container_width=True, hide_index=True)

        # Nhận xét & Biện luận
        st.subheader(TRANSLATIONS[lang]["sub_comments"])
        st.markdown(TRANSLATIONS[lang]["comments_text"])


if __name__ == "__main__":
    main()
