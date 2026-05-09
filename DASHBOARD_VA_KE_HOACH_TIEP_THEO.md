# Dashboard và kế hoạch tiếp theo

Tài liệu này dùng để mở lại dashboard sau khi tắt VS Code, đồng thời ghi rõ các bước tiếp theo theo đúng đề cương trong `agent_skills.md` và `project_architecture..md`.

Đề cương chính dùng để đối chiếu: `C:\Users\PHUK\Documents\KLTN 2025-2026\DCKL_KsorPhuk_V1.docx`.

## 1. Trạng thái hiện tại của project

Project hiện đã hoàn thành các phần sau:

- Đã đọc dataset `vietnamese_ecommerce_review.csv`.
- Đã chạy EDA và tạo `artifacts/reports/eda_summary.json`.
- Đã tiền xử lý text bằng Regex, chuẩn hóa viết tắt, emoji và word segmentation fallback.
- Đã train baseline sentiment với Naive Bayes và SVM.
- Đã tạo metrics và confusion matrix:
  - `artifacts/reports/baseline_metrics.json`
  - `artifacts/reports/confusion_matrix_naive_bayes.csv`
  - `artifacts/reports/confusion_matrix_svm.csv`
- Đã lưu model tốt nhất:
  - `artifacts/models/best_model.joblib`
- Đã chạy được API JSON.
- Đã chạy được Streamlit dashboard.

Kết luận: project đang hoàn thành mức baseline + dashboard demo. Chưa hoàn thành ABSA có nhãn khía cạnh thật, Bi-LSTM, PhoBERT và so sánh deep learning.

## 2. Đối chiếu với đề cương `DCKL_KsorPhuk_V1.docx`

Hướng triển khai hiện tại là đúng với đề cương ở các điểm sau:

- Đúng tên đề tài: trích xuất thông tin và phân tích quan điểm khách hàng trên nền tảng thương mại điện tử để tối ưu hóa chiến lược kinh doanh.
- Đúng nguồn dữ liệu: sử dụng dataset `vietnamese-ecommerce-review` của HienBM/Kaggle.
- Đúng quy trình KDD gồm 5 giai đoạn:
  - Thu thập/lựa chọn dữ liệu.
  - Tiền xử lý dữ liệu.
  - Chuyển đổi dữ liệu.
  - Khai phá dữ liệu/mô hình hóa.
  - Triển khai và đánh giá.
- Đúng kỹ thuật tiền xử lý:
  - Regex để xóa HTML, URL, ký tự nhiễu.
  - Dictionary để chuẩn hóa teencode, từ viết tắt.
  - Emoji handling.
  - Word segmentation.
- Đúng baseline:
  - Bag of Words/TF-IDF.
  - Naive Bayes.
  - SVM.
- Đúng đánh giá:
  - Confusion Matrix.
  - Accuracy.
  - Precision.
  - Recall.
  - F1-score.
- Đúng hướng ứng dụng:
  - API JSON.
  - Dashboard.
  - RACE.
  - Cảnh báo dựa trên `thumbsupcount`, `replycontent`, sentiment và aspect.

Các điểm hiện tại chưa hoàn thiện so với đề cương:

- Chưa tải dữ liệu trực tiếp bằng Kaggle API; hiện đang dùng file CSV local đã có sẵn.
- Word segmentation hiện có fallback; chưa cấu hình VnCoreNLP/vnTokenizer thật.
- Dashboard hiện là demo phân tích từ dataset, chưa phải dashboard thời gian thực.
- `aspects` và `RACE` hiện là rule-based/lexicon, chưa phải kết quả từ model ABSA đã train.
- Chưa có tập nhãn khía cạnh thật cho ABSA.
- Chưa train Bi-LSTM kết hợp POS Tagging.
- Chưa fine-tune PhoBERT bằng Hugging Face.
- Chưa có bảng so sánh đầy đủ Naive Bayes, SVM, Bi-LSTM và PhoBERT.
- Chưa hoàn thiện báo cáo khóa luận và slide.

Kết luận đối chiếu: project hiện đúng hướng đề cương, nhưng mới đạt phần tiền xử lý, baseline, đánh giá cơ sở, API và dashboard demo. Chưa được xem là hoàn thiện toàn bộ đề cương.

## 3. Cách vào lại dashboard sau khi tắt VS Code

Mở VS Code vào đúng folder:

```text
C:\Users\PHUK\Documents\KLTN 2025-2026\DataCenter
```

Mở Terminal trong VS Code, sau đó chạy từng dòng:

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
$env:PYTHONPATH="$PWD\src"; streamlit run dashboard\streamlit_app.py
```

Nếu chạy đúng, Streamlit sẽ hiện link dạng:

```text
http://localhost:8501
```

Mở link này trên trình duyệt để vào dashboard.

## 4. Những lệnh không cần chạy lại

Nếu thư mục `.venv` và `artifacts/` vẫn còn, không cần chạy lại các lệnh sau:

```powershell
python -m venv .venv
pip install -r requirements.txt
python scripts\train_baseline.py --limit 100000 --vectorizer tfidf
```

Chỉ cần train lại khi:

- Xóa thư mục `.venv`.
- Xóa thư mục `artifacts/models`.
- Sửa code tiền xử lý/model.
- Muốn train với nhiều dòng dữ liệu hơn.

## 5. Cách chạy lại API nếu cần

API không bắt buộc để xem dashboard, nhưng dùng để test model qua JSON.

Chạy terminal riêng:

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
$env:PYTHONPATH="$PWD\src"; python -m ecommerce_absa.api --model artifacts\models\best_model.joblib --port 8000
```

Mở terminal khác để test:

```powershell
$body = @{ content = "Dịch vụ hỗ trợ quá tệ, app lỗi liên tục"; thumbsupcount = 25 } | ConvertTo-Json; Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json; charset=utf-8" -Body $body
```

Nếu API đang chạy và muốn tắt, quay lại terminal API và nhấn:

```text
Ctrl + C
```

## 6. Giải thích nhanh dashboard

Dashboard hiện tại đọc một phần dataset và hiển thị:

- `Reviews`: số review đang load.
- `Negative rate`: tỷ lệ review tiêu cực, suy ra từ `score`.
- `Reply rate`: tỷ lệ review có `replycontent`.
- `Alerts`: số review cần cảnh báo ưu tiên.
- `Sentiment distribution`: phân bố negative, neutral, positive.
- `RACE stage distribution`: phân bố review theo Reach, Act, Convert, Engage.
- `High priority alerts`: các review tiêu cực có `thumbsupcount` cao hoặc liên quan dịch vụ.
- `Review explorer`: bảng lọc và xem từng review.

Lưu ý quan trọng: `aspects` và `RACE` hiện tại là rule-based/lexicon để demo nghiệp vụ. Đây chưa phải kết quả ABSA deep learning đã train bằng nhãn khía cạnh thật.

## 7. Kế hoạch tiếp theo theo đúng đề cương

### Bước 1: Hoàn thiện bằng chứng baseline

Mục tiêu: đóng gói kết quả đã có để đưa vào chương thực nghiệm.

Cần làm:

- Ghi lại thông số Naive Bayes và SVM từ `baseline_metrics.json`.
- Ghi lại confusion matrix của từng model.
- Chụp ảnh dashboard làm minh chứng demo.
- Ghi rõ dataset đã train với 100,000 dòng và dùng 99,656 dòng hợp lệ.

Tiêu chí xong:

- Có bảng so sánh Naive Bayes và SVM.
- Có nhận xét vì sao chọn SVM theo F1 macro.

### Bước 2: Tạo tập nhãn ABSA thật

Mục tiêu: chuyển từ dashboard rule-based sang ABSA đúng nghĩa.

Lưu ý: đề cương không viết riêng một bước "annotate dữ liệu", nhưng đây là bước kỹ thuật cần có để đánh giá ABSA đúng nghĩa. Dataset hiện tại chỉ có `score`, chưa có nhãn khía cạnh như `product`, `price`, `delivery`, `service`, `app`.

Cần làm:

- Tạo file annotate riêng, ví dụ `data/absa_annotation_sample.csv`.
- Lấy mẫu review từ dataset, ưu tiên đủ negative, neutral, positive.
- Gán nhãn các khía cạnh:
  - `product`
  - `price`
  - `delivery`
  - `service`
  - `app`
- Gán nhãn sentiment cho từng review hoặc từng khía cạnh.

Tiêu chí xong:

- Có ít nhất 1,000 review được gán nhãn.
- Mỗi dòng có review text, aspect labels và sentiment label.

### Bước 3: Train ABSA baseline bằng Scikit-Learn

Mục tiêu: dùng công cụ đã được phê duyệt để có model ABSA baseline.

Cần làm:

- Dùng TF-IDF cho text đã tiền xử lý.
- Train classifier cho multi-label aspect detection.
- Train classifier cho sentiment.
- Đánh giá bằng Accuracy, Precision, Recall, F1 và confusion matrix.

Tiêu chí xong:

- Có metrics riêng cho aspect detection.
- Có metrics riêng cho sentiment classification.
- Có bảng so sánh với sentiment baseline hiện tại.

### Bước 4: Triển khai Bi-LSTM kết hợp POS Tagging

Mục tiêu: thực hiện phần học sâu trong đề cương.

Cần làm:

- Cài `requirements-dl.txt`.
- Cấu hình VnCoreNLP/POS Tagging.
- Tạo vocabulary và POS vocabulary.
- Train Bi-LSTM trên tập ABSA đã gán nhãn.
- Dùng class weights hoặc Focal Loss nếu dữ liệu lệch lớp.

Tiêu chí xong:

- Có model Bi-LSTM.
- Có metrics trên test set.
- Có nhận xét so sánh với baseline Scikit-Learn.

### Bước 5: Fine-tune PhoBERT

Mục tiêu: thực hiện transformer theo đề cương.

Cần làm:

- Dùng Hugging Face `transformers`.
- Tokenize bằng PhoBERT tokenizer.
- Dùng token đặc biệt `<s>`/CLS embedding.
- Tạo model multi-task:
  - Sigmoid cho multi-label aspect detection.
  - Softmax cho sentiment classification.
- Train trên GPU nếu có.

Tiêu chí xong:

- Có kết quả PhoBERT trên test set.
- Có bảng so sánh PhoBERT với Bi-LSTM, SVM, Naive Bayes.

### Bước 6: Đánh giá chuyên sâu và so sánh thuật toán

Mục tiêu: hoàn thiện phần thực nghiệm của khóa luận.

Cần làm:

- Lập bảng tổng hợp:
  - Naive Bayes
  - SVM
  - Bi-LSTM
  - PhoBERT
- So sánh theo:
  - Accuracy
  - Precision
  - Recall
  - F1
  - Confusion Matrix
- Phân tích lỗi: neutral khó nhận diện, dữ liệu lệch lớp, review ngắn, teencode.

Tiêu chí xong:

- Có bảng so sánh cuối cùng.
- Có nhận xét model nào phù hợp nhất và vì sao.

### Bước 7: Hoàn thiện RACE và cảnh báo kinh doanh

Mục tiêu: biến kết quả model thành insight kinh doanh.

Cần làm:

- Dùng aspect/sentiment từ model ABSA thay cho rule-based lexicon.
- Ánh xạ kết quả vào RACE:
  - Reach
  - Act
  - Convert
  - Engage
- Hoàn thiện logic cảnh báo:
  - review tiêu cực
  - khía cạnh `service`
  - `thumbsupcount` cao
  - có/không có `replycontent`

Tiêu chí xong:

- Dashboard dùng kết quả model thay vì chỉ dùng rule.
- Có case-study minh họa cảnh báo CSKH/eWOM.

### Bước 8: Viết báo cáo và slide

Mục tiêu: hoàn thiện sản phẩm khóa luận.

Cần làm:

- Chương lý thuyết: KDD, sentiment analysis, ABSA, RACE.
- Chương phương pháp: tiền xử lý, TF-IDF, Naive Bayes, SVM, Bi-LSTM, PhoBERT.
- Chương thực nghiệm: dataset, metrics, kết quả, so sánh.
- Chương ứng dụng: API, dashboard, RACE, alert.
- Slide tóm tắt quy trình và kết quả.

Tiêu chí xong:

- Báo cáo có đầy đủ bảng metrics và hình dashboard.
- Slide có pipeline, kết quả model, dashboard demo và định hướng kinh doanh.

## 8. Thứ tự ưu tiên ngay tiếp theo

Nếu tiếp tục làm project từ trạng thái hiện tại, nên làm theo thứ tự:

1. Tạo file mẫu annotate ABSA.
2. Gán nhãn 1,000 review đầu tiên.
3. Viết script train ABSA baseline từ file annotate.
4. Cập nhật dashboard dùng kết quả ABSA baseline.
5. Sau đó mới làm Bi-LSTM và PhoBERT.

Lý do: đề cương yêu cầu ABSA, nhưng dataset hiện tại chưa có nhãn khía cạnh. Nếu không có bước annotate, phần Bi-LSTM/PhoBERT sẽ không có ground truth để đánh giá đúng.

## 9. Cập nhật phase ABSA baseline

Đã bổ sung nền cho bước 2 và bước 3:

- `data/absa_annotation_sample.csv`: seed annotation 30 dòng, cân bằng 10 negative, 10 neutral, 10 positive, dùng để kiểm thử pipeline.
- `scripts/create_absa_annotation_sample.py`: sinh batch annotation từ dataset gốc, để trống nhãn aspect cho gán nhãn thủ công.
- `docs/absa_annotation_guide.md`: quy tắc gán nhãn `product`, `price`, `delivery`, `service`, `app`.
- `scripts/train_absa_baseline.py`: train baseline multi-label aspect detection bằng TF-IDF/Bag of Words + Linear SVM theo Scikit-Learn.
- `artifacts/reports/absa_baseline_metrics.json` và `artifacts/reports/confusion_matrix_absa_aspects.csv`: output kiểm thử sau khi train.

Lưu ý: metrics hiện tại chỉ dùng để chứng minh pipeline chạy được, chưa đại diện cho hiệu năng ABSA thật. Việc cần làm trước khi chuyển sang Bi-LSTM/POS Tagging và PhoBERT là mở rộng annotation lên tối thiểu 1,000 review và kiểm tra chéo chất lượng nhãn.
