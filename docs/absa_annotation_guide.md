# Hướng dẫn gán nhãn ABSA

Mục tiêu của phase này là tạo ground truth cho bài toán ABSA trước khi huấn luyện Bi-LSTM/POS Tagging và PhoBERT. File annotate dùng định dạng CSV, mỗi dòng là một review.

## Schema

File mẫu: `data/absa_annotation_sample.csv`

Các cột bắt buộc:

- `reviewid`: ID review từ dataset gốc.
- `content`: nội dung review gốc.
- `sentiment`: nhãn cảm xúc cấp review, gồm `negative`, `neutral`, `positive`. Cột này được sinh từ `score`, nhưng người annotate có thể sửa nếu nội dung và điểm sao mâu thuẫn rõ ràng.
- `product`: 1 nếu review nói về sản phẩm/hàng hóa/chất lượng/mẫu mã/size/màu/chính hãng.
- `price`: 1 nếu review nói về giá/phí/voucher/sale/khuyến mãi/rẻ/đắt.
- `delivery`: 1 nếu review nói về giao hàng/vận chuyển/shipper/đóng gói/thời gian giao.
- `service`: 1 nếu review nói về hỗ trợ/CSKH/nhân viên/phản hồi/đổi trả/hoàn tiền.
- `app`: 1 nếu review nói về app/web/tính năng/đăng nhập/giỏ hàng/thanh toán/đơn hàng/lỗi/cập nhật.

## Quy tắc nhãn aspect

- Dùng `1` nếu aspect được nhắc đến hoặc được đánh giá.
- Dùng `0` nếu aspect không xuất hiện trong review.
- Một review có thể có nhiều aspect cùng bằng `1`.
- Nếu review chỉ thể hiện cảm xúc chung như "rất tốt", "quá tệ", "bình thường" nhưng không nói rõ khía cạnh nào, đặt tất cả aspect bằng `0`.
- Không để trống khi đã annotate xong. Dòng còn trống được xem là chưa gán nhãn và script train sẽ bỏ qua.
- Không dùng nhãn rule-based từ dashboard để thay cho nhãn thật. Có thể dùng keyword để chọn review cần annotate, nhưng quyết định 0/1 phải dựa trên đọc nội dung.

## Lệnh sinh file annotate

```powershell
.\.venv\Scripts\python.exe scripts\create_absa_annotation_sample.py --limit 100000 --sample-size 300 --output data\absa_annotation_batch_001.csv
```

Script sẽ cố gắng cân bằng `negative`, `neutral`, `positive` dựa trên `score` và để trống các cột aspect cho người annotate.

## Lệnh train baseline ABSA

```powershell
.\.venv\Scripts\python.exe scripts\train_absa_baseline.py --input data\absa_annotation_sample.csv --segmenter none
```

Output chính:

- `artifacts/models/absa_aspect_baseline.joblib`
- `artifacts/reports/absa_baseline_metrics.json`
- `artifacts/reports/confusion_matrix_absa_aspects.csv`

## Mốc dữ liệu khuyến nghị

- `data/absa_annotation_sample.csv` hiện chỉ là seed 30 dòng để kiểm thử pipeline.
- Trước khi báo cáo kết quả học thuật, cần annotate tối thiểu 1,000 review, ưu tiên phân bố đều theo `negative`, `neutral`, `positive`.
- Nên kiểm tra chéo ít nhất 10% số dòng đã gán nhãn để giảm sai lệch giữa các lần annotate.
