# DataCenter ABSA Pipeline

Triển khai theo `agent_skills.md` và `project_architecture..md` cho đề tài phân tích quan điểm khách hàng thương mại điện tử.

## Phạm vi đã scaffold

- KDD pipeline: selection, preprocessing, transformation, modeling, evaluation, deployment.
- Tiền xử lý tiếng Việt: Regex clean HTML/URL/ký tự nhiễu, chuẩn hóa từ viết tắt/teencode, emoji mapping, wrapper VnCoreNLP/vnTokenizer với fallback phrase segmentation.
- Baseline ML: Bag of Words, TF-IDF, Naive Bayes, Linear SVM bằng Scikit-Learn.
- Đánh giá: confusion matrix, Accuracy, Precision, Recall, F1.
- Deep learning optional: Bi-LSTM + POS embedding hook, PhoBERT multitask head, Focal Loss/class weights.
- Business layer: nhận diện khía cạnh bằng lexicon, ánh xạ RACE, cảnh báo eWOM.
- Deployment: Flask JSON API và Streamlit dashboard.

## Giả định và tiêu chí thành công

Triển khai này bám đúng 2 file yêu cầu `agent_skills.md` và `project_architecture..md`, đồng thời áp dụng nguyên tắc trong `CLAUDE.md`: làm tối thiểu, nêu rõ giả định, và chỉ thêm phần có thể truy vết về yêu cầu.

Giả định hiện tại:

- Dataset chỉ có nhãn sentiment gián tiếp qua `score`; chưa có nhãn khía cạnh thủ công.
- Baseline Naive Bayes/SVM là luồng train chính có thể chạy ngay sau khi cài `requirements.txt`.
- Bi-LSTM/PhoBERT là module kiến trúc tùy chọn để phục vụ giai đoạn GPU/fine-tuning, chưa tự chạy nếu chưa cài `requirements-dl.txt`.
- Nhận diện khía cạnh/RACE hiện dùng lexicon weak-label để phục vụ dashboard và cảnh báo, cần thay bằng nhãn ABSA thật khi có dữ liệu annotate.

Tiêu chí verify:

- `python -m unittest discover -s tests` phải pass.
- `python scripts\export_predictions.py --limit 5 --output artifacts\reports\predictions_smoke.csv` phải xuất được JSON-style predictions sang CSV.
- Sau khi cài dependency, `python scripts\train_baseline.py --limit 100000 --vectorizer tfidf` phải tạo model, metrics và confusion matrix trong `artifacts/`.

## Mapping yêu cầu

| Yêu cầu trong `.md` | File triển khai |
| --- | --- |
| Regex clean HTML/URL/ký tự nhiễu, chuẩn hóa từ viết tắt, emoji | `src/ecommerce_absa/preprocessing.py` |
| VnCoreNLP/vnTokenizer hoặc fallback word segmentation | `src/ecommerce_absa/preprocessing.py` |
| Bag of Words, TF-IDF | `src/ecommerce_absa/features.py` |
| Naive Bayes, SVM | `src/ecommerce_absa/models/baselines.py`, `scripts/train_baseline.py` |
| Accuracy, Precision, Recall, F1, Confusion Matrix | `src/ecommerce_absa/evaluation.py` |
| Bi-LSTM/POS, PhoBERT, Sigmoid/Softmax/Cross-Entropy/Focal Loss | `src/ecommerce_absa/models/deep_learning.py` |
| API JSON | `src/ecommerce_absa/api.py` |
| Dashboard | `dashboard/streamlit_app.py` |
| RACE và cảnh báo eWOM | `src/ecommerce_absa/race.py` |

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu chạy Bi-LSTM/PhoBERT:

```powershell
pip install -r requirements-dl.txt
```

## Chạy EDA

```powershell
$env:PYTHONPATH="$PWD\src"
python scripts\eda.py --limit 50000
```

Kết quả được ghi vào `artifacts/reports/eda_summary.json`.

## Huấn luyện baseline

```powershell
$env:PYTHONPATH="$PWD\src"
python scripts\train_baseline.py --limit 100000 --vectorizer tfidf
```

Artifacts chính:

- `artifacts/models/naive_bayes.joblib`
- `artifacts/models/svm.joblib`
- `artifacts/models/best_model.joblib`
- `artifacts/reports/baseline_metrics.json`
- `artifacts/reports/confusion_matrix_*.csv`

## Chạy API JSON

```powershell
$env:PYTHONPATH="$PWD\src"
$env:PYTHONPATH="$PWD\src"; python -m ecommerce_absa.api --model artifacts\models\best_model.joblib --port 8000
```

Ví dụ request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"content":"Dịch vụ hỗ trợ quá tệ, app lỗi liên tục", "thumbsupcount": 25}'
```

## Chạy dashboard

```powershell
$env:PYTHONPATH="$PWD\src"
streamlit run dashboard\streamlit_app.py
```

## Ghi chú dữ liệu

Dataset hiện có `vietnamese_ecommerce_review.csv` gồm khoảng 1.3 triệu review với các cột `reviewid`, `username`, `content`, `score`, `thumbsupcount`, `reviewcreatedversion`, `at`, `replycontent`, `appid`. Vì dataset chưa có nhãn khía cạnh thủ công, module ABSA hiện dùng weak labeling bằng lexicon để phục vụ dashboard/RACE và có thể thay bằng nhãn annotate thật khi có dữ liệu.

## ABSA annotation và baseline

Phase ABSA hiện thêm luồng annotate thật, chưa thay dashboard rule-based.

Tạo batch annotate mới từ dataset gốc:

```powershell
.\.venv\Scripts\python.exe scripts\create_absa_annotation_sample.py --limit 100000 --sample-size 300 --output data\absa_annotation_batch_001.csv
```

Train baseline multi-label aspect detection bằng Scikit-Learn:

```powershell
.\.venv\Scripts\python.exe scripts\train_absa_baseline.py --input data\absa_annotation_sample.csv --segmenter none
```

Hướng dẫn gán nhãn nằm ở `docs/absa_annotation_guide.md`. File `data/absa_annotation_sample.csv` là seed 30 dòng để kiểm thử end-to-end; cần annotate tối thiểu khoảng 1,000 review trước khi dùng metrics cho báo cáo chính.
