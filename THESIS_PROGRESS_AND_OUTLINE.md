# Tiến Độ Khóa Luận Và Đề Cương

## Mục đích

File này dùng làm nơi theo dõi tập trung cho:

- tiến độ thực tế của project
- mức độ bám sát đề cương khóa luận
- các mốc đã hoàn thành
- các việc cần làm tiếp theo

Từ giờ chỉ cần cập nhật file này là có thể nắm được toàn bộ trạng thái dự án.

## Phạm vi đề tài

Tên đề tài:

- Trích xuất thông tin và phân tích quan điểm khách hàng trên nền tảng thương mại điện tử nhằm tối ưu hóa chiến lược kinh doanh

Các yêu cầu chính theo đề cương:

1. Thực hiện đúng quy trình KDD:
   - thu thập/lựa chọn dữ liệu
   - tiền xử lý
   - chuyển đổi dữ liệu
   - mô hình hóa
   - triển khai và đánh giá
2. Xây dựng các mô hình cơ sở:
   - Bag of Words / TF-IDF
   - Naive Bayes
   - SVM
3. Xây dựng luồng ABSA:
   - tạo dữ liệu gán nhãn thủ công
   - dự đoán đa nhãn theo khía cạnh
   - đánh giá bằng Confusion Matrix, Accuracy, Precision, Recall, F1
4. Xây dựng giai đoạn học sâu:
   - Bi-LSTM + POS Tagging
   - PhoBERT fine-tuning
5. Xây dựng lớp phần mềm:
   - API JSON
   - dashboard
6. Ánh xạ kết quả mô hình sang insight kinh doanh:
   - RACE
   - logic cảnh báo

## Trạng thái hiện tại của project

Đã hoàn thành:

- Đã có dataset gốc lưu local.
- Đã có pipeline EDA.
- Đã có pipeline tiền xử lý tiếng Việt:
  - regex cleaning
  - chuẩn hóa viết tắt
  - xử lý emoji
  - word segmentation fallback
- Đã train baseline sentiment:
  - Naive Bayes
  - SVM
- Đã có Flask API.
- Đã có Streamlit dashboard.
- Đã có workflow gán nhãn ABSA.
- Đã có script train ABSA baseline.
- Đã tích hợp model aspect vào API/dashboard, có fallback rule-based.

Đã làm một phần:

- Đã mở rộng tập dữ liệu ABSA, nhưng phần lớn vẫn đi qua prefill và chỉnh nhanh.
- Đã retrain ABSA baseline trên tập dữ liệu mở rộng.
- Đã có workflow ưu tiên rà các dòng rủi ro cao.

Chưa hoàn thành:

- Huấn luyện Bi-LSTM + POS
- Fine-tune PhoBERT
- Bảng so sánh cuối giữa baseline và deep learning
- Tập đánh giá học thuật sạch hoàn chỉnh
- Báo cáo khóa luận hoàn chỉnh
- Slide hoàn chỉnh

## Kết quả baseline hiện tại

### Baseline sentiment

Nguồn: `artifacts/reports/baseline_metrics.json`

- Naive Bayes:
  - accuracy: `0.8866`
  - macro F1: `0.5750`
- SVM:
  - accuracy: `0.8748`
  - macro F1: `0.6210`

Nhận xét:

- SVM là baseline sentiment tốt hơn nếu xét theo macro F1.
- Đây là mốc baseline để so sánh với các mô hình sau.

### Baseline ABSA

Nguồn: `artifacts/reports/absa_baseline_metrics.json`

- số dòng dùng: `1029`
- vectorizer: `tfidf`
- subset accuracy: `0.8414`
- hamming loss: `0.0408`
- micro F1: `0.9147`
- macro F1: `0.9101`

Theo từng khía cạnh:

- product: `0.8990`
- price: `0.9213`
- delivery: `0.9296`
- service: `0.8571`
- app: `0.9434`

Nhận xét:

- Pipeline ABSA đã chạy được end-to-end.
- Các chỉ số này dùng tốt cho mốc nội bộ.
- Chưa nên xem là bằng chứng học thuật cuối cùng, vì nhiều nhãn vẫn xuất phát từ prefill và chỉnh nhanh.

### Baseline ABSA trên gold eval đã rà

Nguồn: `artifacts/reports/absa_baseline_gold_eval_metrics.json`

- train set: `779` dòng
- eval set: `250` dòng đã rà trong `absa_gold_eval_review_queue.csv`
- micro F1: `0.6238`
- macro F1: `0.6279`

Theo từng khía cạnh:

- product: `0.5199`
- price: `0.7327`
- delivery: `0.8063`
- service: `0.4138`
- app: `0.6667`

Nhận xét:

- Kết quả này thấp hơn rõ rệt so với đánh giá trên tập pha trộn trước đó.
- Đây là tín hiệu tốt về mặt học thuật, vì nó cho thấy gold eval đang phản ánh khó khăn thật của bài toán.
- Điểm yếu lớn nhất hiện tại nằm ở:
  - product
  - service
  - app

### Bi-LSTM + POS style baseline trên gold eval

Nguồn: `artifacts/reports/absa_bilstm_gold_eval_metrics.json`

- train set: `779` dòng
- eval set: `250` dòng
- micro F1: `0.6018`
- macro F1: `0.6010`

Theo từng khía cạnh:

- product: `0.4929`
- price: `0.5811`
- delivery: `0.8020`
- service: `0.4409`
- app: `0.6879`

Nhận xét:

- Bi-LSTM đầu tiên chưa vượt baseline ABSA hiện tại trên cùng tập gold eval.
- Đây vẫn là một mốc quan trọng vì nó hoàn thành yêu cầu triển khai mô hình học sâu đầu tiên trong đề cương.
- Bước deep learning tiếp theo nên chuyển sang PhoBERT để có cơ hội cải thiện rõ hơn.

## Các file dữ liệu đang dùng

Các file chính:

- `data/absa_annotation_master.csv`
- `data/prefilled/`
- `data/absa_working_train.csv`
- `data/absa_gold_subset_candidate.csv`
- `data/absa_gold_eval.csv`
- `data/absa_gold_eval_review_queue.csv`
- `scripts/prepare_gold_subset_split.py`
- `scripts/train_absa_with_gold_eval.py`

Phân chia hiện tại:

- working train set: `779` dòng
- gold eval candidate: `250` dòng

Mục đích sử dụng:

- `absa_working_train.csv`: tập train mở rộng
- `absa_gold_eval_review_queue.csv`: file để rà nhãn thủ công
- `absa_gold_eval.csv`: snapshot của tập eval sau khi tách ra

## Khoảng cách so với đề cương

Những phần đã bám đúng đề cương:

- Đã có KDD flow trong code và tài liệu.
- Đã có tiền xử lý.
- Đã có baseline truyền thống.
- Đã có API và dashboard.
- Đã có RACE và alert ở mức phần mềm.

Những phần còn thiếu để đúng nghĩa khóa luận hoàn chỉnh:

1. Một tập đánh giá ABSA sạch và đáng tin cậy
2. Thực nghiệm Bi-LSTM + POS ổn định hơn hoặc đủ để chốt mốc so sánh
3. Thực nghiệm PhoBERT
4. Bảng so sánh cuối giữa các mô hình
5. Case-study kinh doanh hoàn chỉnh
6. Báo cáo và slide hoàn chỉnh

## Đánh giá tổng thể hiện tại

Mức độ trưởng thành của project:

- phần mềm demo: tốt
- baseline modeling: tốt
- workflow ABSA: đã chạy được
- chất lượng đánh giá học thuật: chưa xong
- giai đoạn deep learning: chưa bắt đầu

Kết luận thực tế:

- Project hiện không còn bị nghẽn ở phần kỹ thuật triển khai.
- Điểm nghẽn chính bây giờ là chất lượng tập đánh giá và giai đoạn so sánh mô hình học sâu.
- Gold eval đầu tiên đã xác nhận rằng baseline hiện tại chưa đủ mạnh để dùng làm kết quả cuối cùng.

## Bước tiếp theo nên làm

Bước tiếp theo đúng nhất theo đề cương là:

- làm sạch `gold eval review queue`
- chốt một tập evaluation đáng tin
- đánh giá lại baseline trên cách chia train/eval mới
- sau đó mới sang Bi-LSTM và PhoBERT

Lý do:

- Đề cương yêu cầu so sánh mô hình theo hướng học thuật, không chỉ làm demo chạy được.
- Nếu tập eval còn phụ thuộc mạnh vào prefill/rule-based, kết quả so sánh sau này sẽ yếu và khó bảo vệ.

## Kế hoạch hành động ngắn hạn

1. Rà file `data/absa_gold_eval_review_queue.csv`
2. Chỉnh 6 cột nhãn:
   - `sentiment`
   - `product`
   - `price`
   - `delivery`
   - `service`
   - `app`
3. Sau mỗi dòng đã chắc:
   - đổi `review_status = done`
4. Nếu có sửa đáng chú ý:
   - ghi ngắn vào `review_note`
5. Khi đã đủ số dòng sạch:
   - khóa lại `gold eval`
   - đánh giá lại ABSA baseline bằng split:
     - train: `data/absa_working_train.csv`
     - eval: `data/absa_gold_eval.csv`
6. Sau đó mới chuyển sang:
   - Bi-LSTM + POS
   - PhoBERT

## Bước tiếp theo sau khi hoàn tất 250 dòng gold eval

Sau khi đã có `250` dòng gold eval sạch hơn, workflow kế tiếp là:

1. Giữ:
   - `data/absa_working_train.csv` làm tập train
   - `data/absa_gold_eval.csv` làm tập eval
2. Dùng baseline hiện tại làm mốc tham chiếu
3. Chuyển sang giai đoạn deep learning:
   - `scripts/train_bilstm_absa.py`
4. Sau Bi-LSTM mới sang PhoBERT

Lưu ý môi trường:

- Muốn chạy Bi-LSTM hoặc PhoBERT thì cần cài `requirements-dl.txt`
- Hiện tại repo đã có script nhưng môi trường `.venv` chưa có `torch` và `transformers`

## Ghi chú vận hành

Tập `1029` dòng hiện tại nên xem là:

- tập train/working set mở rộng

Tập `250` dòng nên hướng tới việc trở thành:

- trusted evaluation set

Nếu làm đúng hướng này, phần thực nghiệm sau sẽ sạch hơn nhiều:

- baseline ABSA
- Bi-LSTM
- PhoBERT
- bảng so sánh cuối

## Workflow hiện tại

Khi `gold eval review queue` đã được rà đủ sạch, lệnh đánh giá chuẩn mới là:

```powershell
.\.venv\Scripts\python.exe scripts\train_absa_with_gold_eval.py
```

Ý nghĩa:

- train trên `data/absa_working_train.csv`
- evaluate trên `data/absa_gold_eval.csv`

Đây là workflow nên dùng cho giai đoạn so sánh mô hình tiếp theo, thay vì tiếp tục random split trên toàn bộ tập 1,029 dòng.

## Mốc PhoBERT hiện tại

Đã có script:

- `scripts/train_phobert_absa.py`

Đã chạy thử PhoBERT trên split hiện tại với điều kiện:

- CPU
- 1 epoch
- train: `779` dòng
- eval: `250` dòng

Kết quả aspect:

- micro F1: `0.4325`
- macro F1: `0.4193`

Kết quả sentiment:

- accuracy: `0.4880`
- macro F1: `0.4297`

Kết luận tạm thời:

- PhoBERT đã chạy được end-to-end trong môi trường local.
- Tuy nhiên kết quả hiện tại chưa thể dùng làm mốc so sánh cuối cùng vì mới chỉ là lượt chạy thử trên CPU với 1 epoch.
- Muốn đánh giá PhoBERT công bằng hơn cần tăng số epoch và tốt nhất là dùng GPU.
