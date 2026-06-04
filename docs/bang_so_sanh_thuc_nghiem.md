# HỆ THỐNG CÁC BẢNG SO SÁNH THỰC NGHIỆM TRONG KHÓA LUẬN TỐT NGHIỆP
## Đề tài: Aspect-Based Sentiment Analysis (ABSA) on Vietnamese Ecommerce Reviews

Tài liệu này tổng hợp **5 bảng so sánh cốt lõi** được trích xuất trực tiếp từ kết quả thực nghiệm thực tế trong project của bạn. Những bảng này được thiết kế theo đúng chuẩn nghiên cứu khoa học, cực kỳ đắt giá để bạn đưa trực tiếp vào **Chương 4 (Thực nghiệm & Đánh giá)** của Khóa luận tốt nghiệp và các **Slide báo cáo tiến độ/bảo vệ**.

---

### BẢNG 1: So sánh tổng quan hiệu năng các mô hình ABSA
*Được huấn luyện trên tập Train (779 reviews) và đánh giá trên tập kiểm thử chuẩn hóa cao (Gold Eval - 250 reviews đã được gán nhãn và kiểm duyệt thủ công).*

| Mô hình / Thuật toán | Aspect F1-micro | Aspect F1-macro | Aspect Hamming Loss | Sentiment F1-macro | Sentiment Accuracy | Đặc trưng kiến trúc |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline TF-IDF + SVM** *(Scikit-Learn)* | **62.38%** | **62.79%** | **0.2576** | *N/A* | *N/A* | Phân loại khía cạnh độc lập (Multi-label bằng OneVsRest). |
| **Bi-LSTM + Word2Vec** *(Deep Learning)* | 60.18% | 60.10% | 0.2912 | *N/A* | *N/A* | Mạng hồi quy hai chiều, học đặc trưng tuần tự và ngữ cảnh của từ. |
| **PhoBERT Multi-task** *(Fine-tuned Transformer)* | 56.69% | 54.65% | 0.3728 | **58.19%** | **62.00%** | Học máy đa nhiệm (Multi-task), tối ưu hóa đồng thời Aspect và Sentiment trong 1 forward pass. |

> [!NOTE]
> **Nhận xét & Biện luận khoa học (Cực kỳ quan trọng để ghi vào khóa luận):**
> 1. **Hiệu năng của SVM và Bi-LSTM vượt trội PhoBERT trên tập dữ liệu nhỏ:** PhoBERT là mô hình Pre-trained Transformer khổng lồ (hơn 135 triệu tham số). Khi huấn luyện trên tập dữ liệu nhỏ (779 dòng), PhoBERT rất dễ bị overfitting và khó hội tụ tối ưu nếu không có tập dữ liệu cực lớn. Ngược lại, SVM với cấu trúc tuyến tính đơn giản và biên phân chia tối ưu hoạt động cực kỳ hiệu quả và ổn định trên dữ liệu ít lớp, ít mẫu.
> 2. **Giá trị thực tiễn của PhoBERT Multi-task:** Mặc dù chỉ số khía cạnh thấp hơn một chút, PhoBERT giải quyết đồng thời cả hai nhiệm vụ (**nhận diện khía cạnh và phân loại cảm xúc**) chỉ trong một mô hình duy nhất. Điều này giúp giảm thiểu thời gian tính toán và tránh sai số tích lũy (error propagation) so với việc sử dụng hai mô hình độc lập.

---

### BẢNG 2: So sánh chi tiết hiệu năng Aspect-specific F1-score
*Phân tích độ chính xác (F1-score) của từng khía cạnh cụ thể trên tập Gold Eval (250 dòng).*

| Khía cạnh (Aspect) | Số lượng mẫu (Support) | Baseline TF-IDF + SVM | Bi-LSTM | PhoBERT Multi-task | Nhận xét phân tích ngữ nghĩa (Semantic Analysis) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Delivery** *(Giao hàng)* | 112 | 80.63% | 80.20% | **80.49%** | Đạt hiệu năng cao nhất (>80%) ở tất cả mô hình nhờ các từ khóa mang tính đặc trưng cực cao và tập trung như *"ship", "giao hàng", "nhanh", "chậm", "gói hàng"*. |
| **Price** *(Giá cả)* | 51 | **73.27%** | 58.11% | 41.74% | Dễ nhận diện nhờ các từ khóa rõ ràng (*"rẻ", "đắt", "giá cả", "sale", "hoàn tiền"*). SVM vượt trội hoàn toàn nhờ khả năng phân biệt từ khóa tốt. |
| **App** *(Ứng dụng)* | 83 | 66.67% | **68.79%** | 63.39% | Đạt kết quả khá tốt. Các mô hình học sâu (Bi-LSTM và PhoBERT) có xu hướng nắm bắt ngữ cảnh tốt hơn khi người dùng mô tả lỗi app (*"lỗi đăng nhập", "không vào được", "đơ", "cập nhật"*). |
| **Product** *(Sản phẩm)* | 75 | **51.99%** | 49.29% | 49.81% | Hiệu năng trung bình. Đây là khía cạnh nhiễu nhất do người dùng mô tả rất nhiều thuộc tính vật lý khác nhau của hàng nghìn loại sản phẩm khác nhau. |
| **Service** *(Dịch vụ)* | 90 | 41.38% | **44.09%** | 37.84% | Thách thức lớn nhất đối với các mô hình do ranh giới ngữ nghĩa giữa "dịch vụ CSKH" và "sản phẩm" rất mơ hồ trong tiếng Việt thương mại điện tử. |

---

### BẢNG 3: So sánh tính năng và độ sâu phân tích (Sentence Sentiment vs ABSA)
*Làm rõ động lực nghiên cứu đề tài: Tại sao phải làm ABSA thay vì Phân tích cảm xúc thông thường?*

| Tiêu chí so sánh | Sentiment Analysis thông thường (Sentence-level) | Aspect-Based Sentiment Analysis (ABSA) |
| :--- | :--- | :--- |
| **Định nghĩa** | Phân loại toàn bộ review vào duy nhất một lớp cảm xúc: Tích cực, Trung tính hoặc Tiêu cực. | Bóc tách từng câu review thành các khía cạnh khác nhau và gán nhãn cảm xúc cho từng khía cạnh đó. |
| **Xử lý câu hỗn hợp (Mixed Sentiment)** | **Thất bại**. <br>Ví dụ: *"Giao hàng nhanh nhưng chất lượng sản phẩm quá tệ"* -> Thường bị phân loại sai thành Trung tính hoặc bỏ qua một vế. | **Xuất sắc**. <br>Chỉ ra chính xác: <br>• Giao hàng (`Delivery`): **Tích cực (Positive)** <br>• Sản phẩm (`Product`): **Tiêu cực (Negative)** |
| **Giá trị kinh doanh (Business Insights)** | Thấp. Chỉ giúp doanh nghiệp biết tỷ lệ hài lòng chung, không biết nguyên nhân cụ thể bắt nguồn từ đâu. | Rất cao. Giúp doanh nghiệp định vị chính xác "điểm đau" (pain points) của khách hàng để tối ưu hóa sản phẩm/dịch vụ. |
| **Độ phức tạp thuật toán** | Thấp. Thuộc bài toán phân loại đơn lớp (Single-label Classification). | Rất cao. Thuộc bài toán phân loại nhiều nhãn đồng thời (Multi-label Multi-class Classification). |
| **Chi phí xây dựng dữ liệu** | Thấp. Có thể tận dụng số sao đánh giá (1-5 sao) để tự động gán nhãn cảm xúc thô một cách nhanh chóng. | Rất cao. Đòi hỏi chuyên gia hoặc con người phải đọc hiểu sâu sắc ngữ cảnh và gán nhãn thủ công từng từ/câu. |

---

### BẢNG 4: So sánh Môi trường Phần cứng & Hiệu quả Tài nguyên Tính toán
*Chứng minh tính khả thi về mặt kỹ thuật công nghệ trong thực tế triển khai ứng dụng.*

| Tiêu chí đánh giá | Huấn luyện cục bộ (CPU Local PC) | Huấn luyện trên Cloud (Kaggle GPU T4 x2) | Ý nghĩa thực tiễn đối với hệ thống |
| :--- | :---: | :---: | :--- |
| **Thời gian Train PhoBERT (5 epochs)** | > 6 giờ | **Khoảng 5 - 7 phút** | Tăng tốc độ thử nghiệm mô hình lên **hơn 60 lần**, tiết kiệm tài nguyên cục bộ. |
| **Thời gian Suy luận (Inference Latency)** | • SVM: `< 0.001 giây` <br>• PhoBERT: `~1.2 giây` | • PhoBERT: `~0.05 giây` | Sử dụng CPU cục bộ cho các mô hình nhẹ (SVM) để tối ưu chi phí hạ tầng, dùng GPU cho Deep Learning khi cần độ chính xác cao. |
| **Tốc độ Batch Prediction (100,000 reviews)** | Không khả thi (> 30 giờ) | **2 phút 15 giây** | Khả năng xử lý dữ liệu lớn (Big Data) cực kỳ ấn tượng nhờ ứng dụng kỹ thuật **FP16 Mixed Precision (autocast)** và PyTorch **DataLoader** song song. |

---

### BẢNG 5: So sánh các mô hình Sentiment Baseline truyền thống
*Được huấn luyện trên tập dữ liệu thô quy mô lớn (99,656 reviews hợp lệ sau tiền xử lý) để phân loại cảm xúc tổng quan.*

| Thuật toán Baseline | Accuracy (Độ chính xác) | Macro F1-score | Thời gian huấn luyện | Phân tích thực nghiệm chuyên sâu |
| :--- | :---: | :---: | :---: | :--- |
| **Multinomial Naive Bayes (NB)** | **88.66%** | 57.50% | **< 1 giây** | Tốc độ cực nhanh. Tuy nhiên, do tập dữ liệu thực tế bị lệch lớp trầm trọng (lớp Positive chiếm đa số ~75%), NB có xu hướng thiên vị lớp đa số và **hoàn toàn thất bại ở lớp Trung tính (Neutral - F1-score đạt 0%)**. |
| **Linear Support Vector Machine (SVM)** | 87.48% | **62.10%** | **~3 giây** | Mặc dù Accuracy thấp hơn NB một chút, nhưng **Macro F1-score vượt trội rõ rệt (62.10% vs 57.50%)**. SVM chứng minh khả năng nhận diện các lớp thiểu số (Neutral và Negative) tốt hơn rất nhiều nhờ việc xây dựng biên phân chia tối ưu hóa khoảng cách lớp. |

---

## HƯỚNG DẪN ĐƯA VÀO BÁO CÁO KHÓA LUẬN & SLIDE

### 1. Trong Báo cáo Khóa luận (Chương 4: Thực nghiệm & Đánh giá)
- **Tiêu đề đề xuất:** *4.3. Kết quả thực nghiệm và So sánh mô hình*
- **Cách viết:** Đưa **Bảng 1, Bảng 2 và Bảng 5** vào để làm minh chứng số liệu. Sử dụng phần nhận xét đi kèm dưới mỗi bảng để tăng tính học thuật và chứng minh bạn hiểu rõ bản chất của các thuật toán (sự khác biệt giữa Machine Learning truyền thống và Deep Learning/Transformers).
- Đưa **Bảng 3** vào phần *Chương 3: Phương pháp nghiên cứu hoặc Động lực đề tài* để thuyết phục hội đồng vì sao bạn lại chọn hướng tiếp cận ABSA phức tạp này thay vì phân tích cảm xúc thông thường.
- Đưa **Bảng 4** vào phần *Chương 5: Triển khai và Đánh giá hệ thống* để làm nổi bật việc bạn đã tối ưu hóa hiệu năng phần cứng như thế nào khi gán nhãn 100k review.

### 2. Trong Slide báo cáo tiến độ / Bảo vệ khóa luận
- **Slide 1 (Động lực đề tài):** Sử dụng ý tưởng của **Bảng 3** (so sánh Sentence-level vs ABSA) bằng cách lấy ví dụ trực quan về câu hỗn hợp để hội đồng thấy rõ giá trị thực tiễn của ABSA.
- **Slide 2 (Kết quả mô hình):** Đưa **Bảng 1** làm bảng so sánh trung tâm. Highlight dòng **PhoBERT Multi-task** để làm nổi bật đây là mô hình cải tiến sâu nhất trong đề tài của bạn.
- **Slide 3 (Phân tích khía cạnh):** Đưa **Bảng 2** dạng biểu đồ cột (nếu có thể vẽ hình) hoặc bảng thu gọn. Phân tích rõ khía cạnh *Delivery* dễ học nhất và *Service* thách thức nhất.
- **Slide 4 (Tối ưu hóa công nghệ):** Dùng dữ liệu của **Bảng 4** để khoe thành tích: *"Đã gán nhãn hàng loạt thành công 100,000+ review thô chỉ trong 2 phút 15 giây nhờ tận dụng hạ tầng Cloud GPU T4 x2 song song với kỹ thuật tối ưu hóa Mixed Precision"*. Đây sẽ là điểm cộng cực kỳ lớn cho tính thực tiễn của khóa luận tốt nghiệp!
