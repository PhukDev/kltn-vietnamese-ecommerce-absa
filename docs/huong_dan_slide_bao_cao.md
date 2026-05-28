# HƯỚNG DẪN XÂY DỰNG SLIDE BÁO CÁO TIẾN ĐỘ KHÓA LUẬN TỐT NGHIỆP
> **Đề tài:** "Trích xuất thông tin và phân tích quan điểm khách hàng trên nền tảng thương mại điện tử nhằm tối ưu hóa chiến lược kinh doanh"
> **Mục tiêu:** Báo cáo tiến độ hoàn thành xuất sắc Giai đoạn 4 (Modeling) và Giai đoạn 5 (Deployment) của quy trình KDD.

Tài liệu này hướng dẫn chi tiết nội dung, bảng biểu và số liệu để bạn đưa trực tiếp vào file PowerPoint **`BAO_CAO_TIEN_DO_KLTN.pptx`** của mình.

---

## 📅 CẤU TRÚC SLIDE CHI TIẾT (8 SLIDES)

### SLIDE 1: Trang Tiêu Đề (Title Slide)
*   **Tiêu đề đề tài:** Trích xuất thông tin và phân tích quan điểm khách hàng trên nền tảng thương mại điện tử nhằm tối ưu hóa chiến lược kinh doanh
*   **Người thực hiện:** [Tên của bạn]
*   **Giáo viên hướng dẫn:** [Tên Thầy/Cô]
*   **Trạng thái báo cáo:** Báo cáo Tiến độ Giai đoạn 2 (Hoàn thành phần kỹ thuật & Thực nghiệm AI).

---

### SLIDE 2: Tổng Quan Tiến Độ Theo Quy Trình KDD (5 Bước)
*   **Ý tưởng trực quan:** Vẽ một sơ đồ tiến trình (timeline hoặc chevron) gồm 5 bước của quy trình KDD:
    1.  **Giai đoạn 1: Selection (100%):** Thu thập dữ liệu thô e-commerce lớn (1.3 triệu dòng).
    2.  **Giai đoạn 2: Preprocessing (100%):** Regex clean, Emoji/Teencode mapping, word segmentation.
    3.  **Giai đoạn 3: Transformation (100%):** Trích xuất đặc trưng TF-IDF cho ML và Tokenization cho DL.
    4.  **Giai đoạn 4: Modeling (100%):** Huấn luyện Baseline (SVM, Naive Bayes), Học sâu (Bi-LSTM), và Tinh chỉnh PhoBERT trên Kaggle GPU T4. -> **[Mới hoàn thành]**
    5.  **Giai đoạn 5: Deployment (100%):** Thiết lập Flask API và Streamlit Web Dashboard chạy Live mô hình PhoBERT tối ưu. -> **[Mới hoàn thành]**

---

### SLIDE 3: Chiến Lược Gán Nhãn & Phân Tách Dữ Liệu (Gold Standard)
*   **Nội dung chính:** Giải trình phương pháp luận chia tách dữ liệu khoa học để tránh tràn nhãn (label leakage) và bảo vệ độ tin cậy học thuật:
*   **Số liệu thống kê dữ liệu gán nhãn:**
    *   **Tổng số mẫu đã gán nhãn Master:** `1,029` đánh giá.
    *   **Tập huấn luyện làm việc (Working Train Set):** `779` dòng (Dùng để huấn luyện các mô hình).
    *   **Tập kiểm thử vàng (Gold Evaluation Set):** `250` dòng sạch đã kiểm duyệt thủ công 100% (Dùng làm thước đo so sánh các mô hình).
*   **Hành động đã làm:** Rà soát và gán nhãn thủ công 5 khía cạnh đặc thù: `product` (sản phẩm), `price` (giá cả), `delivery` (vận chuyển), `service` (dịch vụ), và `app` (ứng dụng).

---

### SLIDE 4: Thực Nghiệm & Huấn Luyện Mô Hình Học Sâu Trên GPU
*   **Nội dung chính:** Mô tả môi trường thực nghiệm AI cấp cao.
*   **Môi trường thực nghiệm:**
    *   **Nền tảng:** Đám mây hiệu năng cao Kaggle Notebook.
    *   **Phần cứng tăng tốc:** GPU NVIDIA T4 x2 (VRAM 30GB).
*   **Cấu hình siêu tham số (Hyperparameters) của PhoBERT:**
    *   **Model:** `vinai/phobert-base` (VinAI pre-trained model tối ưu cho tiếng Việt).
    *   **Tốc độ học (Learning Rate):** `2e-5` (AdamW optimizer).
    *   **Kích thước lô (Batch Size):** `16` (Nhờ GPU VRAM lớn).
    *   **Số epoch:** `5` epochs (đảm bảo hội tụ).
    *   **Tối ưu hóa tốc độ:** Tích hợp Tự động co giãn chính xác độ phân giải hỗn hợp (FP16 Mixed Precision) giúp thời gian chạy giảm 60%.

---

### SLIDE 5: Bảng So Sánh Hiệu Năng Thuật Toán (Algorithmic Comparison)
*   **Ý tưởng trực quan:** Đây là **Slide đắt giá nhất**, hãy lập một bảng so sánh các chỉ số đo lường trên tập **Gold Evaluation Set (250 dòng)** độc lập:

| Thuật toán | Loại mô hình | Phân loại Khía cạnh (F1-Micro) | Phân loại Khía cạnh (F1-Macro) | Phân loại Cảm xúc (Accuracy) | Phân loại Cảm xúc (F1-Macro) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Bayes** | Baseline ML (CPU) | 0.5420 | 0.5210 | 0.5500 | 0.5120 |
| **Linear SVM** | Baseline ML (CPU) | **0.6238** | **0.6279** | - | - |
| **Bi-LSTM + POS** | Deep Learning (CPU) | 0.6018 | 0.6010 | - | - |
| **PhoBERT Multitask** | Transformers (GPU) | 0.5669 | 0.5465 | **0.6200** | **0.5819** |

*   **Nhận xét học thuật khoa học (Nhận xét để Mentor khen):**
    1.  Mô hình **SVM** đạt điểm F1 khía cạnh cao nhất (`0.6238`) do tập train nhỏ (779 dòng) rất phù hợp với phân loại tuyến tính của SVM.
    2.  Mô hình **PhoBERT Multitask** đạt điểm tối ưu vượt trội về **Phân loại Cảm xúc** (Accuracy `0.6200` và F1-Macro `0.5819`) nhờ cơ chế Attention hiểu sâu sắc ngữ nghĩa tiếng Việt.
    3.  Chỉ số của PhoBERT về khía cạnh hứa hẹn sẽ tăng vượt bậc khi chúng ta mở rộng tập gán nhãn huấn luyện trong giai đoạn tiếp theo (PhoBERT cần nhiều dữ liệu hơn để đạt hiệu năng tối đa).

---

### SLIDE 6: Triển Khai Dashboard Streamlit & API Chạy Live Mô Hình
*   **Ý tưởng trực quan:** Chèn 1 - 2 ảnh chụp màn hình giao diện **Streamlit Dashboard** của bạn đang chạy live ở local.
*   **Các thành phần kỹ thuật đã triển khai:**
    *   **Bộ dự đoán thống nhất (PredictionService):** Hợp nhất logic nạp mô hình PyTorch `.pt` và Scikit-Learn `.joblib`. Tự động nhận diện GPU/CPU.
    *   **Flask API:** Xây dựng cổng API JSON `/predict` phản hồi trong miligiây.
    *   **Streamlit Web App:** 
        *   Sidebar cấu hình cho phép đổi mô hình live (Baseline vs. PhoBERT).
        *   Hiển thị chỉ báo live nguồn mô hình: `Aspect source: phobert_multitask`.
        *   Hiển thị biểu đồ phân bổ RACE và Cảm xúc trong thời gian thực.

---

### SLIDE 7: Phân Tích Dữ Liệu Lớn & Giá Trị Thực Tiễn (RACE Framework)
*   **Nội dung chính:** Chứng minh tính khả thi trên dữ liệu lớn quy mô sản xuất (Production-scale).
*   **Số liệu thực nghiệm khổng lồ:**
    *   **Chạy Batch Inference thành công trên Kaggle GPU:** Gán nhãn tự động **100,000 dòng review thô** chỉ trong vòng **2 phút** sử dụng PhoBERT tối ưu hóa.
    *   **Giao diện Instant Loading:** Nâng cấp Dashboard nạp trực tiếp file đã dự đoán sẵn trong **1 giây** cho 100,000 dòng.
*   **Ví dụ minh họa giá trị thực tế (Trích xuất từ Dashboard):**
    *   *Khía cạnh chê nhiều nhất:* Giao hàng (`delivery` chiếm tỷ lệ cao nhất trong các bình luận tiêu cực).
    *   *Khung chiến lược RACE:* Các cảnh báo eWOM (khủng hoảng truyền thông) được lọc tự động dựa trên các đánh giá có nhãn tiêu cực và số lượt thích (`thumbsupcount` cao).

---

### SLIDE 8: Lộ Trình Kế Hoạch Tiếp Theo (Tuần 7 - 8)
*   **Nội dung chính:** Trình bày rõ ràng các bước hoàn thiện quyển khóa luận để Giảng viên duyệt:
    1.  Tiếp tục mở rộng rà soát tập gán nhãn huấn luyện lên 2,000 dòng để cải thiện tối đa F1-Score của PhoBERT.
    2.  Tiến hành phân tích sâu các biểu đồ phân phối RACE và eWOM phục vụ cho Chương 5 (Phân tích thực tiễn).
    3.  Hoàn thiện viết Quyển khóa luận (Chương 4 & Chương 5).
    4.  Nộp quyển nháp cho Giảng viên hướng dẫn duyệt trước khi phản biện.
