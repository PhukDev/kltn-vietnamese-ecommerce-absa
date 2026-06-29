# Hướng Dẫn Train Dữ Liệu Bằng Google Colab Cho Dự Án KLTN ABSA

Tài liệu này hướng dẫn chi tiết cách cấu hình và chạy các tập lệnh huấn luyện mô hình học sâu (**PhoBERT Multitask** và **Bi-LSTM**) trong dự án của bạn trên môi trường Google Colab để tận dụng GPU miễn phí (hoặc trả phí) của Google giúp tăng tốc độ huấn luyện.

---

## 💡 Tại sao nên dùng Google Colab cho dự án này?
1. **PhoBERT** (VinAI) dựa trên kiến trúc RoBERTa, rất nặng và **bắt buộc phải chạy trên GPU** để đạt tốc độ train hợp lý. Nếu chạy trên CPU cá nhân, mỗi epoch có thể mất hàng tiếng đồng hồ.
2. Google Colab cung cấp GPU miễn phí (thường là **NVIDIA T4** với 15GB VRAM) hoặc các GPU mạnh hơn (L4, A100 ở gói Pro).
3. Bạn **không cần** upload file dữ liệu lớn `vietnamese_ecommerce_review.csv` (~643MB). Các script train deep learning của bạn (`train_phobert_absa.py` và `train_bilstm_absa.py`) chỉ sử dụng các tập dữ liệu đã gán nhãn khía cạnh nhỏ nằm trong thư mục `data/` (`absa_working_train.csv` và `absa_gold_eval.csv` ~ dưới 200KB). Do đó việc chuẩn bị và upload dự án lên Colab diễn ra cực kỳ nhanh chóng!

---

## 🛠️ Chuẩn Bị Dự Án Trước Khi Upload

Trước khi đưa dự án lên Colab, hãy đảm bảo cấu trúc thư mục của bạn gọn gàng:
- **Không tải lên** thư mục ảo `.venv/` (thư mục này chứa hàng gigabyte thư viện cục bộ và không tương thích với HĐH Linux của Colab).
- **Không cần tải lên** file dữ liệu thô khổng lồ `vietnamese_ecommerce_review.csv` trừ khi bạn muốn chạy lại toàn bộ quy trình tiền xử lý từ đầu trên Colab.

---

## 🚀 Phương Pháp 1: Sử dụng GitHub (Khuyên Dùng - Sạch Sẽ & Chuyên Nghiệp)

Nếu bạn đã đẩy code dự án lên một kho lưu trữ GitHub (ví dụ: `https://github.com/PhukDev/kltn-vietnamese-ecommerce-absa`), đây là cách tối ưu nhất.

### Bước 1: Khởi động Google Colab & Bật GPU
1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Tạo một Notebook mới (**New Notebook**).
3. Trên thanh công cụ, chọn **Runtime** -> **Change runtime type** (Thay đổi loại trình chạy).
4. Tại mục **Hardware accelerator** (Bộ tăng tốc phần cứng), chọn **T4 GPU** (hoặc GPU bất kỳ có sẵn) -> Nhấn **Save**.

### Bước 2: Clone dự án và cài đặt môi trường
Tạo một ô mã nguồn (Code cell) mới và chạy các lệnh sau:

```python
# 1. Kiểm tra thông tin GPU để chắc chắn GPU đã hoạt động
!nvidia-smi

# 2. Clone dự án từ GitHub của bạn
# (Nếu là repo riêng tư, bạn có thể dùng Personal Access Token dạng: https://<token>@github.com/...)
!git clone https://github.com/PhukDev/kltn-vietnamese-ecommerce-absa.git

# 3. Di chuyển vào thư mục dự án
%cd kltn-vietnamese-ecommerce-absa

# 4. Cài đặt các thư viện Deep Learning bắt buộc
!pip install -r requirements-dl.txt
```

---

## 📦 Phương Pháp 2: Upload trực tiếp file Zip (Nếu không dùng GitHub)

### Bước 1: Nén dự án thành file `.zip`
Nén toàn bộ thư mục dự án của bạn lại thành một file (ví dụ: `DataCenter.zip`). 
> **Lưu ý quan trọng**: Hãy loại bỏ các thư mục `.venv`, `.git` và file `vietnamese_ecommerce_review.csv` trước khi nén để giảm dung lượng file zip xuống chỉ còn vài MB.

### Bước 2: Upload và giải nén trên Colab
1. Mở Google Colab, bật **T4 GPU** giống như Phương pháp 1.
2. Nhìn sang bảng điều khiển bên trái, chọn biểu tượng **Thư mục** (Files).
3. Click vào biểu tượng **Upload to session storage** (Tải lên bộ lưu trữ phiên) và chọn file `DataCenter.zip` của bạn.
4. Chờ file upload xong, chạy các lệnh giải nén và di chuyển thư mục:

```python
# 1. Giải nén file project
!unzip -q DataCenter.zip

# 2. Di chuyển vào thư mục dự án (đổi tên theo tên thư mục của bạn sau khi giải nén)
%cd DataCenter

# 3. Cài đặt thư viện
!pip install -r requirements-dl.txt
```

---

## 💾 Kết nối Google Drive để lưu Model lâu dài (Cực Kỳ Quan Trọng!)

Colab là môi trường tạm thời (Session-based). Khi bạn đóng tab hoặc sau vài tiếng không tương tác, toàn bộ dữ liệu trên Colab sẽ bị xóa sạch, bao gồm cả file trọng số mô hình `.pt` vừa train xong. 

Để lưu trữ mô hình huấn luyện vĩnh viễn, bạn nên lưu chúng trực tiếp vào **Google Drive** cá nhân.

Chạy đoạn code sau để kết nối Colab với Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Tạo thư mục lưu trữ model trên Google Drive của bạn (nếu chưa có)
import os
os.makedirs('/content/drive/MyDrive/KLTN_Models', exist_ok=True)
```

---

## 🏋️‍♂️ Thực Hiện Huấn Luyện (Training)

Tất cả các script của bạn đã được thiết kế rất thông minh: chúng tự động nhận diện thư mục gốc của dự án (`PROJECT_ROOT`) và tự động thêm thư mục `src/` vào biến môi trường Python (`PYTHONPATH`). Bạn có thể chạy trực tiếp bằng dòng lệnh:

### 1. Huấn luyện mô hình PhoBERT ABSA
Mô hình PhoBERT sử dụng học sâu đa tác vụ (Multitask Learning) cho cả khía cạnh (Aspect) và cảm xúc (Sentiment). Chạy lệnh sau để huấn luyện:

```python
!python scripts/train_phobert_absa.py \
    --train data/absa_working_train.csv \
    --eval data/absa_gold_eval.csv \
    --model-name vinai/phobert-base \
    --epochs 10 \
    --batch-size 8 \
    --lr 2e-5 \
    --max-len 128 \
    --segmenter none
```
*💡 **Mẹo**:*
- Do dữ liệu huấn luyện của bạn tương đối nhỏ, nên đặt `--batch-size 8` hoặc `16` để tránh quá tải bộ nhớ và tối ưu hóa hội tụ.
- Đối với VnCoreNLP: Nếu bạn để `--segmenter auto` hoặc `vncorenlp`, hệ thống sẽ tự tải gói VnCoreNLP về. Colab đã có sẵn Java nên bạn không cần cài thêm Java, tuy nhiên để tránh phát sinh lỗi tải file trên Colab, tùy chọn `--segmenter none` (phân tách theo khoảng trắng mặc định) thường chạy ổn định nhất để kiểm thử nhanh.

### 2. Huấn luyện mô hình Bi-LSTM ABSA
Mô hình Bi-LSTM kết hợp nhúng từ và thẻ từ loại (POS embedding). Chạy lệnh sau:

```python
!python scripts/train_bilstm_absa.py \
    --train data/absa_working_train.csv \
    --eval data/absa_gold_eval.csv \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-3 \
    --max-len 128 \
    --segmenter none
```

---

## 📥 Tải Kết Quả Và Mô Hình Đã Huấn Luyện Về Máy

Sau khi script chạy xong, các mô hình tốt nhất cùng báo cáo chi tiết sẽ được tự động lưu vào thư mục `artifacts/models/` và `artifacts/reports/` của dự án trên Colab.

Hãy copy chúng sang Google Drive đã mount ở trên để lưu trữ vĩnh viễn:

```python
# Sao chép model PhoBERT đã train thành công sang Google Drive
!cp artifacts/models/absa_phobert_gold_eval.pt /content/drive/MyDrive/KLTN_Models/
!cp artifacts/reports/absa_phobert_gold_eval_metrics.json /content/drive/MyDrive/KLTN_Models/

# Sao chép model Bi-LSTM đã train thành công sang Google Drive
!cp artifacts/models/absa_bilstm_gold_eval.pt /content/drive/MyDrive/KLTN_Models/
!cp artifacts/reports/absa_bilstm_gold_eval_metrics.json /content/drive/MyDrive/KLTN_Models/

print("Đã sao lưu thành công các mô hình sang Google Drive của bạn tại thư mục 'KLTN_Models'!")
```

Sau đó, bạn chỉ cần mở Google Drive trên máy tính cá nhân, tải các file `.pt` và `.json` này về thư mục `artifacts/models/` và `artifacts/reports/` tương ứng của dự án cục bộ để chạy thử nghiệm hoặc kết nối vào Dashboard Streamlit/API Flask!

---

## 📝 Mẫu Notebook Hoàn Chỉnh Để Copy-Paste Vào Colab

Dưới đây là mã nguồn toàn bộ một Notebook mẫu bạn có thể copy trực tiếp vào một ô Code duy nhất trên Colab để thực hiện trọn vẹn quy trình (sử dụng phương thức GitHub làm ví dụ):

```python
# =====================================================================
# 1. THIẾT LẬP MÔI TRƯỜNG & TẢI MÃ NGUỒN
# =====================================================================
print("--- 1. Kiểm tra GPU ---")
!nvidia-smi

print("\n--- 2. Tải mã nguồn dự án ---")
# Thay URL bên dưới bằng repo GitHub của bạn
!git clone https://github.com/PhukDev/kltn-vietnamese-ecommerce-absa.git
%cd kltn-vietnamese-ecommerce-absa

print("\n--- 3. Cài đặt các thư viện yêu cầu ---")
!pip install -r requirements-dl.txt

# =====================================================================
# 2. LIÊN KẾT GOOGLE DRIVE ĐỂ LƯU MODEL
# =====================================================================
print("\n--- 4. Kết nối Google Drive để sao lưu mô hình ---")
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/KLTN_Models', exist_ok=True)

# =====================================================================
# 3. HUẤN LUYỆN MÔ HÌNH (TRAINING)
# =====================================================================
print("\n--- 5. Bắt đầu huấn luyện PhoBERT ABSA ---")
!python scripts/train_phobert_absa.py \
    --train data/absa_working_train.csv \
    --eval data/absa_gold_eval.csv \
    --epochs 10 \
    --batch-size 8 \
    --lr 2e-5 \
    --segmenter none

# =====================================================================
# 4. SAO LƯU KẾT QUẢ VÀO GOOGLE DRIVE
# =====================================================================
print("\n--- 6. Sao lưu kết quả train sang Google Drive ---")
!cp artifacts/models/absa_phobert_gold_eval.pt /content/drive/MyDrive/KLTN_Models/
!cp artifacts/reports/absa_phobert_gold_eval_metrics.json /content/drive/MyDrive/KLTN_Models/
print("=> ĐÃ HOÀN THÀNH HUẤN LUYỆN VÀ SAO LƯU!")
```
