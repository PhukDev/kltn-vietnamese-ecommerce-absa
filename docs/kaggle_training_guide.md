# Hướng Dẫn Huấn Luyện Mô Hình PhoBERT Trên Kaggle GPU

Tài liệu này hướng dẫn chi tiết cách tải mã nguồn lên Kaggle Notebook, kích hoạt cấu hình GPU (T4 x2 hoặc P100) để huấn luyện mô hình PhoBERT Multi-task đạt hiệu năng tối ưu và nhanh chóng nhất.

---

## 💡 Tại sao nên chọn Kaggle thay vì huấn luyện cục bộ (CPU)?
- **Tốc độ vượt trội**: Huấn luyện PhoBERT (5 epochs) trên CPU cá nhân có thể mất **hơn 6 giờ**. Trên Kaggle GPU T4 hoặc P100, thời gian train chỉ khoảng **3 - 5 phút**.
- **Miễn phí**: Kaggle cung cấp khoảng 30 giờ sử dụng GPU miễn phí mỗi tuần.

---

## 🛠️ Bước 1: Chuẩn bị mã nguồn dự án

Do Kaggle cho phép kết nối Internet, phương pháp tốt nhất và nhanh nhất là clone trực tiếp từ GitHub. Hãy đảm bảo bạn đã đẩy những thay đổi mới nhất (đặc biệt là tệp dữ liệu chia theo tỷ lệ 30/70 mới) lên GitHub:
- Git Repo URL của bạn: `https://github.com/PhukDev/kltn-vietnamese-ecommerce-absa.git`

> [!TIP]
> Nếu bạn không muốn sử dụng GitHub hoặc dự án của bạn đang ở chế độ riêng tư (Private):
> 1. Nén thư mục dự án cục bộ thành tệp `DataCenter.zip` (loại bỏ thư mục `.venv`, `.git` và file dữ liệu thô lớn `vietnamese_ecommerce_review.csv` để file zip chỉ nặng ~200KB).
> 2. Lên Kaggle chọn **New Dataset**, tải tệp `DataCenter.zip` này lên. Sau đó liên kết Dataset này vào Notebook.

---

## 🚀 Bước 2: Tạo Kaggle Notebook & Cấu hình GPU

1. Truy cập trang web [Kaggle](https://www.kaggle.com/) và đăng nhập tài khoản.
2. Tại thanh điều hướng bên trái, chọn **Create** -> **New Notebook**.
3. **Kích hoạt GPU & Internet (Quan trọng)**:
   - Ở thanh công cụ bên phải (Notebook options), mở mục **Accelerator**.
   - Chọn **GPU T4 x2** (được khuyên dùng vì có 2 GPU song song) hoặc **GPU P100**.
   - Kiểm tra xem mục **Internet** đã được chuyển sang trạng thái **On** (màu xanh lá) hay chưa. Đây là điều kiện bắt buộc để tải thư viện PyTorch/Transformers và pre-trained weights từ Hugging Face.

---

## 🏋️‍♂️ Bước 3: Đoạn code chạy thực nghiệm trên Kaggle

Hãy tạo một ô mã nguồn (Code cell) mới trong Kaggle Notebook, copy toàn bộ đoạn mã bên dưới vào và bấm **Run** (Ctrl + Enter):

```python
# 1. Kiểm tra trạng thái GPU trên máy ảo Kaggle
print("--- 1. Kiểm tra thông tin GPU ---")
!nvidia-smi

# 2. Tải mã nguồn từ GitHub của bạn
print("\n--- 2. Tải mã nguồn dự án ---")
!git clone https://github.com/PhukDev/kltn-vietnamese-ecommerce-absa.git

# 3. Di chuyển thư mục làm việc vào project
%cd kltn-vietnamese-ecommerce-absa

# 4. Cài đặt các thư viện phục vụ học sâu
print("\n--- 3. Cài đặt các thư viện Deep Learning ---")
!pip install -r requirements-dl.txt

# 5. Huấn luyện mô hình PhoBERT với split dữ liệu mới (30% train, 70% eval)
print("\n--- 4. Bắt đầu huấn luyện PhoBERT Multi-task ---")
!python scripts/train_phobert_absa.py \
    --train data/absa_working_train.csv \
    --eval data/absa_gold_eval.csv \
    --epochs 5 \
    --batch-size 16 \
    --lr 2e-5 \
    --segmenter none

# 6. Hiển thị kết quả huấn luyện cuối cùng
print("\n--- 5. Kết quả huấn luyện ---")
!cat artifacts/reports/absa_phobert_gold_eval_metrics.json
```

---

## 📥 Bước 4: Tải tệp mô hình (.pt) và kết quả (.json) về máy tính

Khi script trên chạy hoàn tất:
1. Nhìn sang thanh **Output** ở góc phải màn hình Kaggle.
2. Tìm đến đường dẫn thư mục `kltn-vietnamese-ecommerce-absa/artifacts/models/` và `artifacts/reports/`.
3. Nhấp vào nút ba chấm bên cạnh tệp **`absa_phobert_gold_eval.pt`** và **`absa_phobert_gold_eval_metrics.json`** -> chọn **Download** để tải về máy tính của bạn.
4. Di chuyển hai tệp này vào đúng vị trí tương ứng trong dự án cục bộ của bạn:
   - File model đặt tại: `artifacts/models/absa_phobert_gold_eval.pt`
   - File metrics đặt tại: `artifacts/reports/absa_phobert_gold_eval_metrics.json`

Sau khi sao chép thành công, hãy khởi chạy Streamlit Dashboard cục bộ, hệ thống sẽ tự động cập nhật số liệu mới trên Tab **So Sánh Hiệu Năng Các Mô Hình** và bạn có thể chọn mô hình PhoBERT mới chạy thử nghiệm!
