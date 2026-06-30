*Đầu trang (Header): Chương 5: Kết luận và hướng phát triển*
*Chân trang (Footer): Sinh viên thực hiện: Ksor Phuk*

# CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 5.1. KẾT LUẬN
Đề tài *"Trích xuất thông tin và phân tích quan điểm khách hàng trên nền tảng thương mại điện tử nhằm tối ưu hóa chiến lược kinh doanh"* đã được thực hiện và hoàn thành đầy đủ các nội dung nghiên cứu lý thuyết, thực nghiệm xây dựng mô hình và triển khai ứng dụng phần mềm. Dưới đây là những kết quả chính đạt được cùng một số hạn chế còn tồn tại của đề tài:

### 5.1.1. Những kết quả đạt được
- **Về mặt Lý thuyết:** Nghiên cứu và hệ thống hóa toàn diện cơ sở khoa học của Text Mining, các phương pháp biểu diễn nhúng từ (TF-IDF, Word2Vec), các thuật toán học máy truyền thống (Naive Bayes, Linear SVM), học sâu tuần tự (Bi-LSTM) và kiến trúc Transformer hiện đại (PhoBERT). Đề tài cũng đã kết nối thành công lý thuyết kỹ thuật AI với lý thuyết quản trị kinh doanh thông qua khung chiến lược RACE và eWOM.
- **Về mặt Thực nghiệm và Mô hình hóa:** Lập trình và huấn luyện thành công các mô hình ABSA. Qua quá trình đối sánh hiệu năng khách quan trên tập Gold Eval gồm 721 dòng, mô hình **PhoBERT Multi-task** đã chứng minh sự vượt trội hoàn toàn với **Aspect F1-micro đạt 97.08%**, **F1-macro đạt 97.34%** và **Hamming Loss đạt 0.0139**, vượt xa mô hình Baseline SVM (F1-micro 85.25%) và Bi-LSTM (F1-micro 74.87%). Đồng thời, mô hình đa nhiệm cũng giải quyết hiệu quả tác vụ phân loại cảm xúc khía cạnh đạt **76.01% Accuracy**. Đề tài đã đề xuất và thực nghiệm thành công kỹ thuật **Pre-tokenization** trên GPU T4 x2, giúp tối ưu hóa 100% tài nguyên tính toán và hoàn thành huấn luyện chỉ trong **25 phút**.
- **Về mặt Ứng dụng Phần mềm:** Thiết kế và phát triển hoàn chỉnh ứng dụng Web Dashboard tương tác thời gian thực bằng Streamlit và Flask API. Hệ thống phần mềm có khả năng nạp dữ liệu lớn, trực quan hóa phân phối cảm xúc và phân phối RACE, đồng thời tích hợp logic cảnh báo eWOM thông minh dựa trên tương tác (`thumbsupcount`) và khía cạnh nhạy cảm thương hiệu (`Service`).
- **Về mặt Nghiệp vụ Doanh nghiệp:** Ánh xạ thành công kết quả trích xuất của hơn 389.964 reviews thực tế từ hệ thống bán lẻ UEL Store vào khung RACE. Đề xuất hệ thống giải pháp định lượng giúp doanh nghiệp cải thiện tỷ lệ chuyển đổi, tối ưu hóa quy trình giao vận (logistics), nâng cao chất lượng dịch vụ khách hàng và quản trị hiệu quả khủng hoảng truyền miệng tiêu cực trên không gian mạng.

### 5.1.2. Những hạn chế còn tồn tại
Mặc dù đạt được những kết quả khả quan và mang tính thực tiễn cao, đề tài vẫn còn một số điểm hạn chế khoa học cần được phân tích và khắc phục trong tương lai:
1. **Sự mất cân bằng dữ liệu tự nhiên và độ nhạy cảm của lớp Trung tính:** Bộ dữ liệu TMĐT thực tế bị lệch lớp cực kỳ nghiêm trọng, trong đó nhãn cảm xúc Tích cực chiếm khoảng 75%. Mặc dù đã áp dụng trọng số lớp (class weights) và các hàm mất mát tùy biến để phạt lỗi, mô hình phân loại cảm xúc vẫn gặp khó khăn lớn khi nhận diện các bình luận mang sắc thái Trung tính (`neutral`). Nhãn trung tính thường đại diện cho các câu văn chỉ mang tính chất mô tả sự thật vật lý (ví dụ: *"vải dày bình thường"*, *"hộp màu vàng nhạt"*), rất dễ bị mô hình nhầm lẫn sang nhãn Tích cực hoặc Tiêu cực do thiếu các từ khóa thể hiện thái độ rõ ràng.
2. **Thách thức xử lý sắc thái châm biếm (Sarcasm) và ẩn ý ngược nghĩa:** Độ chính xác phân loại cảm xúc khía cạnh tổng thể dừng lại ở mức 76.01% do tiếng Việt trực tuyến cực kỳ phong phú các câu văn mang sắc thái mỉa mai, ẩn ý hoặc sử dụng các từ ngữ mang tính tương phản cao. 
   - *Ví dụ châm biếm:* *"Giao hàng nhanh quá, đặt hàng có 10 ngày là nhận được rồi!"* hay *"Sản phẩm tốt cực kỳ, dùng được đúng 2 tiếng là hỏng luôn."* 
   Trong các ví dụ này, PhoBERT dễ bị các từ bổ nghĩa mạnh mang tính tích cực như *"nhanh quá"*, *"tốt cực kỳ"* đánh lừa và gán nhãn Tích cực cho khía cạnh `Delivery` và `Product`. Việc nhận diện châm biếm đòi hỏi mô hình phải có tri thức nền (commonsense knowledge) để đối chiếu thông tin khách quan (10 ngày đối với giao hàng là rất chậm, 2 tiếng sử dụng đối với một sản phẩm thông thường là rất tệ).
3. **Hiện tượng từ mới nằm ngoài từ điển (OOV) và teencode tiến hóa:** Dù đã xây dựng từ điển chuẩn hóa teencode lớn với 55 từ viết tắt, ngôn ngữ mạng của người tiêu dùng trẻ (nhất là Gen Z) luôn biến đổi không ngừng theo thời gian với sự xuất hiện của các từ lóng, teencode mới hàng ngày (ví dụ: *"keo lỳ"*, *"tái châu"*, *"báo"*). Mô hình PhoBERT và các Baseline ML vốn được huấn luyện trên các corpus tĩnh sẽ dần bị lỗi thời khi gặp phải các từ mới này, làm suy giảm hiệu năng theo thời gian nếu không có cơ chế cập nhật liên tục.
4. **Tính động và khả năng tích hợp thời gian thực:** Hệ thống Dashboard hiện tại hoạt động dựa trên các mô hình đã được huấn luyện tĩnh trên tập dữ liệu lịch sử, chưa tích hợp cơ chế tự động thu thập và tự động cập nhật huấn luyện lại (online learning) khi xuất hiện các luồng đánh giá mới của người tiêu dùng trực tiếp từ các sàn TMĐT thông qua API thời gian thực.

---

## 5.2. HƯỚNG PHÁT TRIỂN CỦA ĐỀ TÀI
Từ những kết quả đạt được và các hạn chế nêu trên, các hướng nghiên cứu tiếp theo được đề xuất nhằm phát triển đề tài lên tầm cao mới, hướng tới một hệ sinh thái AI toàn diện cho quản trị doanh nghiệp:

### 5.2.1. Nâng cấp kiến trúc mô hình ngôn ngữ lớn (LLM) và Fine-tuning
- **Mục tiêu:** Giải quyết triệt để bài toán châm biếm, ẩn ý và từ lóng OOV mới.
- **Giải pháp:** Nghiên cứu áp dụng các mô hình Transformer có quy mô tham số lớn hơn (như PhoBERT-large với 370 triệu tham số) hoặc tinh chỉnh các mô hình ngôn ngữ lớn (LLM) mã nguồn mở thế hệ mới đã được bản địa hóa cho tiếng Việt (như LLaMA-3-Vietnamese, ViGPT, GPT-4o-mini) bằng phương pháp tinh chỉnh hiệu quả tham số PEFT (Parameter-Efficient Fine-Tuning) thông qua kỹ thuật LoRA (Low-Rank Adaptation) hoặc QLoRA. Các LLM với hàng tỷ tham số sở hữu lượng tri thức nền khổng lồ và khả năng lập luận ngữ cảnh (contextual reasoning) vượt trội sẽ dễ dàng phát hiện các vế mâu thuẫn mang tính châm biếm trong câu đánh giá.

### 5.2.2. Xây dựng Trợ lý ảo phản hồi khách hàng tự động (RAG CSKH Agent)
- **Mục tiêu:** Tự động hóa hoàn toàn quy trình xử lý phản hồi, rút ngắn thời gian CSKH và cá nhân hóa trải nghiệm.
- **Giải pháp:** Thiết kế hệ thống AI Agent tích hợp kỹ thuật RAG (Retrieval-Augmented Generation) kết hợp với mô hình ABSA PhoBERT. Quy trình hoạt động của hệ thống bao gồm:
  1. *Bước 1: Phân tích ABSA:* Mô hình PhoBERT trích xuất chính xác khía cạnh lỗi (ví dụ: `Product` - cảm xúc `Negative` do sản phẩm bị nứt, hoặc `Delivery` - cảm xúc `Negative` do giao chậm).
  2. *Bước 2: Truy xuất Tri thức (Retrieval):* Dựa trên nhãn khía cạnh lỗi, hệ thống tự động truy vấn vào Cơ sở dữ liệu Vector (Vector Database như ChromaDB, Milvus) chứa các tài liệu chính sách của doanh nghiệp (ví dụ: Chính sách bảo hành đổi trả trong 7 ngày, Chính sách đền bù voucher 20k cho đơn hàng chậm trên 3 ngày).
  3. *Bước 3: Tạo phản hồi bằng LLM (Generation):* Đưa câu đánh giá của khách hàng, kết quả nhãn ABSA và nội dung chính sách đã truy xuất làm ngữ cảnh (Prompt Context) vào LLM. LLM sẽ tự động soạn thảo một phản hồi đồng cảm, chuyên nghiệp, cá nhân hóa:
     $$\text{Prompt} = \{\text{Đánh giá khách hàng} + \text{Nhãn ABSA} + \text{Chính sách truy xuất}\} \xrightarrow{\text{LLM}} \text{Phản hồi CSKH cá nhân hóa}$$
     *Ví dụ đầu ra:* *"Chào bạn, chúng tôi vô cùng xin lỗi vì sự cố sản phẩm bị nứt như bạn đã phản hồi. Theo chính sách bảo hành đổi trả của gian hàng, chúng tôi xin hỗ trợ gửi sản phẩm mới thay thế miễn phí cho bạn ngay lập tức. Nhân viên CSKH sẽ liên hệ hỗ trợ bạn làm thủ tục..."*
  4. *Bước 4: Duyệt và Gửi (Human-in-the-loop):* Câu trả lời được hiển thị lên màn hình duyệt của nhân viên CSKH để họ kiểm tra trước khi tự động gửi lên Shopee/Lazada API.

### 5.2.3. Tích hợp cơ chế tự học và tự cập nhật (Active Learning Loop)
- **Mục tiêu:** Khắc phục hiện tượng suy giảm hiệu năng mô hình do teencode mới và giảm thiểu chi phí gán nhãn thủ công.
- **Giải pháp:** Xây dựng quy trình học chủ động (Active Learning):
  - Hệ thống tự động theo dõi xác suất dự báo (prediction confidence) của mô hình PhoBERT trên các luồng dữ liệu đánh giá mới cào về hàng ngày.
  - Lọc ra các đánh giá có độ không chắc chắn cao nhất (entropy lớn nhất, ví dụ mô hình phân vân 50% Positive và 50% Negative) và đẩy về hàng đợi duyệt của chuyên gia (Human Annotators).
  - Chuyên gia thực hiện gán nhãn chuẩn hóa cho các mẫu khó này. Dữ liệu mới được tự động đưa vào kho huấn luyện để chạy tinh chỉnh lại mô hình PhoBERT định kỳ hàng tháng, giúp mô hình tự học các teencode mới mà không cần gán nhãn lại toàn bộ kho dữ liệu.

### 5.2.4. Phát triển Module Business Intelligence (BI) mở rộng và phân tích đối thủ
- **Mục tiêu:** Mở rộng các chiều phân tích sâu cho ban lãnh đạo doanh nghiệp.
- **Giải pháp:** Tích hợp thêm các module phân tích nâng cao trên Dashboard:
  - *Phân tích ý định mua hàng (Purchase Intent Analysis):* Nhận diện các khách hàng tiềm năng hỏi thăm về sản phẩm để chủ động gửi ưu đãi kích cầu.
  - *Phân tích so sánh đối thủ cạnh tranh (Competitor Analysis):* Áp dụng mô hình ABSA để cào và phân tích trực tiếp các đánh giá tại các gian hàng đối thủ cùng ngành, giúp doanh nghiệp phát hiện điểm yếu của đối thủ (ví dụ: đối thủ hay bị chê giao chậm) để định vị chiến dịch marketing của mình tốt hơn.

---
*Đầu trang (Header): Chương 5: Kết luận và hướng phát triển*
*Chân trang (Footer): Sinh viên thực hiện: Ksor Phuk*
