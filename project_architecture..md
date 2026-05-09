KIẾN TRÚC DỰ ÁN & QUY TRÌNH KDD: PHÂN TÍCH QUAN ĐIỂM ĐA KHÍA CẠNH (ABSA)
Dự án phải tuân thủ nghiêm ngặt 5 bước của quy trình Khám phá Tri thức trong Cơ sở dữ liệu (KDD Pipeline) và lộ trình 8 tuần đã được xác định. 

PHẦN 1: QUY TRÌNH KDD (5 BƯỚC)
Giai đoạn 1: Thu thập dữ liệu (Data Selection)
Nguồn: Trích xuất tự động qua API của Kaggle (tập vietnamese-ecommerce-review do HienBM thu thập từ UEL Store). 

Input Database: Bản ghi từ cuối 2021 đến giữa 2022, gồm các trường: reviewid, username, content, score, thumbsupcount, reviewcreatedversion, replycontent. 

Giai đoạn 2: Tiền xử lý (Data Preprocessing)
Chạy module Python làm sạch văn bản cấp thấp bằng Regex (xóa HTML, URLs, ký tự đặc biệt). 

Ánh xạ từ điển (Hash map/Dictionary) chuẩn hóa teencode, từ viết tắt và Emoji. 

Phân mảnh từ (Word Segmentation) bằng VnCoreNLP hoặc vnTokenizer. 

Giai đoạn 3: Chuyển đổi dữ liệu (Data Transformation)
Tạo ma trận Bag of Words và TF-IDF cho Machine Learning. 

Mã hóa ngữ cảnh câu với token <s> của PhoBERT cho Deep Learning. 

Giai đoạn 4: Khai phá dữ liệu (Data Mining / Modeling)
Baseline: Khởi chạy thuật toán Naive Bayes và Support Vector Machine (SVM) bằng Scikit-Learn. 

Deep Learning: Huấn luyện Bi-LSTM (kết hợp POS Tagging) và Tinh chỉnh PhoBERT trên GPU (Mạng đa tác vụ sử dụng Sigmoid, Softmax và Cross-Entropy/Focal Loss). 

Giai đoạn 5: Triển khai & Đánh giá (Deployment & Evaluation)
Đo lường mô hình bằng Confusion Matrix, Accuracy, Precision, Recall, F1_Score. 

Thiết kế API (JSON format) đổ dữ liệu lên Web Dashboard (Streamlit, Flask hoặc ReactJS). 

Phân tích Case-study kinh doanh theo mô hình RACE. 

PHẦN 2: TIẾN ĐỘ THỰC HIỆN ĐỀ TÀI (TIMELINE 8 TUẦN)
Tiến độ các task yêu cầu Agent tuân thủ nghiêm ngặt theo thời gian sau :

Tuần 1 - Tuần 2: Lập đề cương, thu thập tài liệu và khảo sát lý thuyết Khai phá dữ liệu. 

Tuần 2 - Tuần 4: Tải dữ liệu Kaggle, tiến hành EDA và Tiền xử lý NLP. 

Tuần 3 - Tuần 5: Huấn luyện và đánh giá các mô hình cơ sở (SVM, Naive Bayes). 

Tuần 4 - Tuần 8: Huấn luyện và tinh chỉnh mô hình học sâu (Bi-LSTM, PhoBERT). 

Tuần 5 - Tuần 7: Đánh giá hiệu năng chuyên sâu (ABSA) và so sánh thuật toán. 

Tuần 6 - Tuần 8: Ánh xạ insight dữ liệu vào khung chiến lược kinh doanh (RACE). 

Tuần 7 - Tuần 8: Viết báo cáo khóa luận các chương lý thuyết, phương pháp và thực nghiệm. 

Tuần 8: Hoàn thiện quyển khóa luận, chuẩn bị slide.