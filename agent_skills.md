YÊU CẦU KỸ NĂNG KỸ THUẬT CHO AGENT (AGENT SKILLS)
Dự án khóa luận: "Trích xuất thông tin và phân tích quan điểm khách hàng trên nền tảng thương mại điện tử nhằm tối ưu hóa chiến lược kinh doanh". 
Agent chỉ được phép sử dụng các phương pháp, thuật toán và công cụ đã được phê duyệt trong đề cương dưới đây:

1. Kỹ thuật Tiền xử lý dữ liệu (Data Preprocessing)
Công cụ được chỉ định: Biểu thức chính quy (Regex), VnCoreNLP hoặc vnTokenizer. 

Kỹ năng bắt buộc:

Làm sạch văn bản cấp thấp: Dùng Regex để loại bỏ mã HTML, siêu liên kết (URLs) và các ký tự đặc biệt vô nghĩa. 

Chuẩn hóa từ vựng (Vocabulary Standardization): Xây dựng từ điển ánh xạ (hash map/dictionary) dạng Key-Value để dịch từ viết tắt (vd: "sp" -> "sản phẩm", "nv" -> "nhân viên", "đc" -> "được"). 

Xử lý biểu tượng cảm xúc (Emoji Handling): Chuyển đổi emoji thành văn bản thuần (vd: ":)" hoặc ":D" -> "vui_vẻ", "<3" -> "yêu_thích"). 

Phân mảnh từ (Word Segmentation): Dùng VnCoreNLP/vnTokenizer để nối âm tiết từ ghép tiếng Việt bằng dấu gạch dưới (vd: "giày thể thao" -> "giày_thể_thao"). 

2. Học máy truyền thống (Machine Learning Baselines)
Công cụ được chỉ định: Scikit-Learn. 

Kỹ năng bắt buộc:

Chuyển đổi dữ liệu (Data Transformation): Xây dựng ma trận thưa bằng phương pháp Túi từ (Bag of Words) và TF-IDF. Cần áp dụng đúng công thức hạ trọng số từ hư từ và tăng trọng số từ đặc trưng. 

Khởi tạo và huấn luyện 2 mô hình cơ sở: Naive Bayes và Support Vector Machine (SVM). 

3. Học sâu & Transformer (Deep Learning)
Công cụ được chỉ định: Thư viện Hugging Face (transformers), môi trường GPU (Google Colab hoặc máy tính cá nhân). 

Kỹ năng bắt buộc:

Bi-LSTM: Xây dựng mạng nơ-ron hồi quy hai chiều kết hợp kỹ thuật gán nhãn từ loại (POS Tagging) để so sánh hiệu năng. 

PhoBERT: Tải và tinh chỉnh (Fine-tuning) kiến trúc RoBERTa (huấn luyện trước trên 20GB văn bản). Sử dụng lớp nhúng (embedding layer) với token đặc biệt <s> kết nối vào Mạng nơ-ron kết nối đầy đủ (Dense Layer). 

Hàm kích hoạt (Activation Function): Dùng Sigmoid cho bài toán phân loại khía cạnh (Multi-label) và Softmax cho bài toán phân loại cảm xúc. 

Hàm mất mát (Loss Function): Sử dụng Cross-Entropy hoặc Focal Loss có tích hợp cơ chế gán trọng số lớp (class weights) để phạt nặng mô hình khi đoán sai nhãn thiểu số (xử lý mất cân bằng dữ liệu). 

4. Đánh giá thuật toán (Evaluation)
Kỹ năng bắt buộc: Tính toán dựa trên Tập kiểm thử (Test Set) bằng Ma trận nhầm lẫn (Confusion Matrix) và 4 chỉ số: Accuracy, Precision, Recall, F1_Score. 

5. Phát triển phần mềm & Tích hợp kinh doanh (Software & Business)
Công cụ được chỉ định: Đầu ra định dạng API (JSON format), Framework Web (Streamlit, Flask hoặc ReactJS). 

Kỹ năng bắt buộc:

Ánh xạ kết quả định lượng (quan điểm đa khía cạnh, số lượt thumbsupcount, replycontent) vào 4 giai đoạn của mô hình RACE (Reach, Act, Convert, Engage). 

Lập trình hệ thống cảnh báo tự động trên Dashboard (Ví dụ: Nếu phát hiện bình luận Tiêu cực về "Dịch vụ" có số lượt thumbsupcount tăng vọt, phát cảnh báo để bộ phận CSKH can thiệp ở giai đoạn "Engage" nhằm quản trị eWOM).