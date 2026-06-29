# HƯỚNG DẪN GIẢI TRÌNH ĐỀ TÀI KHÓA LUẬN TỐT NGHIỆP
> **Tên đề tài:** "Trích xuất thông tin và phân tích quan điểm khách hàng trên nền tảng thương mại điện tử nhằm tối ưu hóa chiến lược kinh doanh"
> **Tài liệu tham chiếu chính:** Bài báo khoa học *"UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis"* và bộ dữ liệu *vietnamese_ecommerce_review.csv*.

---

## MỤC LỤC
1. [Phần 1: Bài Toán Cốt Lõi & Tính Cấp Thiết (Business & ML Problem)](#phan-1-bai-toan-cot-loi--tinh-cap-thiet)
2. [Phần 2: Giá Trị Thực Tiễn Của 5 Nhãn Khía Cạnh (Business Value)](#phan-2-gia-tri-thuc-tien-cua-5-nhan-khia-canh)
3. [Phần 3: Đối Chiếu Khoa Học: Kế Thừa & Đóng Góp Mới So Với Bài Báo UIT-VSFC](#phan-3-doi-chieu-khoa-hoc-ke-thua--dong-gop-moi)
4. [Phần 4: Kịch Bản Đối Thoại Trực Tiếp Với Giảng Viên Hướng Dẫn (Mentor Pitching)](#phan-4-kich-ban-doi-thoai-truc-tiep)
5. [Phần 5: Lộ Trình Các Bước Triển Khai Kỹ Thuật Tiếp Theo](#phan-5-lo-trinh-trien-khai)

---

<a name="phan-1-bai-toan-cot-loi--tinh-cap-thiet"></a>
## PHẦN 1: BÀI TOÁN CỐT LÕI & TÍNH CẤP THIẾT

### 1. Tại sao phân tích cảm xúc chung chung (Sentiment Analysis) lại vô ích với doanh nghiệp?
Nếu chỉ gán nhãn và dự đoán một bình luận là **Tích cực (Positive)** hay **Tiêu cực (Negative)**, doanh nghiệp sẽ rơi vào trạng thái "mù thông tin":
-   Biết khách hàng đang giận dữ nhưng **không biết họ giận vì lý do gì** (do hàng lỗi, shipper thái độ, hay ứng dụng bị lag).
-   Không thể đưa ra quyết định sửa đổi ở cấp phòng ban hay quy trách nhiệm vận hành.

### 2. Sự vượt trội của Phân tích quan điểm theo khía cạnh (Aspect-Based Sentiment Analysis - ABSA)
Đề tài này giải quyết triệt để vấn đề trên bằng cách phân tách bình luận thô thành các cặp **(Khía cạnh - Sắc thái cảm xúc tương ứng)** trên **5 khía cạnh cốt lõi** của Thương mại điện tử:

```
                  ┌─────────────── Đánh giá của khách hàng ──────────────┐
                  │ "Giao hàng rất nhanh nhưng sản phẩm bị rách đế..."  │
                  └──────────────────────────────────┬───────────────────┘
                                                     ▼
                                     [ BỘ NÃO PHÂN TÍCH ABSA ]
                                                     │
                             ┌───────────────────────┴───────────────────────┐
                             ▼                                               ▼
              Khía cạnh: DELIVERY (Vận chuyển)                Khía cạnh: PRODUCT (Sản phẩm)
              Sắc thái:  POSITIVE (Tích cực)                  Sắc thái:  NEGATIVE (Tiêu cực)
```

-   **`product` (Sản phẩm)**: Chất lượng hàng hóa, tính chính hãng, đóng gói sản phẩm.
-   **`price` (Giá cả)**: Mức giá đắt/rẻ, tính trung thực của mã giảm giá, khuyến mãi.
-   **`delivery` (Vận chuyển)**: Tốc độ giao hàng, thái độ của nhân viên giao hàng (shipper).
-   **`service` (Dịch vụ)**: Tốc độ và thái độ phản hồi của đội ngũ CSKH/admin, giải quyết khiếu nại.
-   **`app` (Ứng dụng)**: Trải nghiệm người dùng trên app di động, lỗi kỹ thuật (bugs), lag, lỗi thanh toán.

---

<a name="phan-2-gia-tri-thuc-tien-cua-5-nhan-khia-canh"></a>
## PHẦN 2: GIÁ TRỊ THỰC TIỄN CỦA 5 NHÃN KHÍA CẠNH

Khi giảng viên hỏi *"Doanh nghiệp cần 5 nhãn này làm gì?"*, đây là câu trả lời mang tính quyết định: **5 nhãn này là nguyên liệu đầu vào để tự động hóa 4 quy trình quản trị doanh nghiệp quy mô lớn (Big Data) mà con người không thể làm thủ công:**

### 1. Tự động hóa KPI phòng ban & Đánh giá đối tác vận chuyển thứ ba
-   **Thực tế**: Các sàn TMĐT thuê các bên vận chuyển (Giao Hàng Nhanh, J&T, Viettel Post...) và quản trị hàng ngàn nhà cung cấp (Merchants).
-   **Ứng dụng**: Nhãn `delivery` và `product` được dùng để tính toán tự động **Chỉ số hài lòng vận chuyển (Logistics Satisfaction Index)** và **Chỉ số chất lượng sản phẩm (Product Quality Index)** theo thời gian thực.
-   **Giá trị**: Cuối tháng, hệ thống tự động xuất báo cáo đề xuất phạt hợp đồng hoặc cắt giảm sản lượng đối với các đối tác giao nhận có tỷ lệ tiêu cực cao.

### 2. Phân tích nguyên nhân gốc rễ (Root Cause Analysis - RCA)
-   **Thực tế**: Một sản phẩm nhận đánh giá 1 sao (Cảm xúc tiêu cực), chủ cửa hàng cần biết vấn đề nằm ở đâu để sửa đổi.
-   **Ứng dụng**: Hệ thống phân tích tổ hợp nhãn. Ví dụ:
    -   *Sản phẩm chất lượng tốt (`product` = Tích cực)* nhưng *giao hàng quá chậm (`delivery` = Tiêu cực)* khiến khách hàng đánh giá 1 sao.
-   **Giá trị**: Chủ cửa hàng biết ngay vấn đề không nằm ở khâu sản xuất hay nguồn hàng, từ đó tiết kiệm hàng trăm triệu đồng chi phí cải tiến sản phẩm không cần thiết, thay vào đó tập trung tối ưu khâu kho vận.

### 3. Khoanh vùng lỗi kỹ thuật và Lập lộ trình phát triển phần mềm (Bug Prioritization)
-   **Thực tế**: Hàng ngàn review chê ứng dụng lag, lỗi thanh toán trôi đi mỗi ngày trên App Store/Google Play.
-   **Ứng dụng**: Hệ thống tự động lọc riêng các đánh giá có nhãn `app` = Tiêu cực, sau đó chạy thuật toán gom cụm chủ đề (Topic Modeling) hoặc trích xuất từ khóa trên tập dữ liệu này.
-   **Giá trị**: Đội ngũ lập trình phát hiện ngay lập tức lỗi hệ thống (ví dụ: *"90% review tiêu cực về app liên quan đến từ khóa 'lỗi ví ShopeePay'"*) để vá lỗi lập tức trong vài giờ, thay vì mất hàng tuần đọc thủ công.

### 4. Hệ thống cảnh báo sớm khủng hoảng eWOM (electronic Word of Mouth)
-   **Thực tế**: Một đánh giá tiêu cực nghiêm trọng (ví dụ lỗi trừ tiền nhưng không tạo đơn) nếu nhận nhiều lượt thích (`thumbsupcount` cao) sẽ dễ dàng lan truyền tạo khủng hoảng truyền thông.
-   **Ứng dụng**: Hệ thống giám sát liên tục. Nếu phát hiện một câu có nhãn `app` = Tiêu cực + `thumbsupcount` > 10, hệ thống tự động bắn cảnh báo khẩn cấp (Alert) qua Telegram/Slack cho đội PR và kỹ thuật.
-   **Giá trị**: Dập tắt hoàn toàn nguy cơ khủng hoảng thương hiệu từ trong trứng nước.

---

<a name="phan-3-doi-chieu-khoa-hoc-ke-thua--dong-gop-moi"></a>
## PHẦN 3: ĐỐI CHIẾU KHOA HỌC: KẾ THƯA & ĐỒNG GÓP MỚI SO VỚI BÀI BÁO UIT-VSFC

Đề tài khóa luận của bạn **kế thừa nền tảng khoa học vững chắc** từ bài báo khoa học uy tín cấp quốc gia **UIT-VSFC** nhưng **cải tiến vượt bậc** để giải quyết bài toán thực tế của doanh nghiệp:

### Bảng So Sánh Đối Chiếu Học Thuật

| Tiêu chí | Bài báo khoa học gốc (UIT-VSFC) | Đề tài khóa luận của bạn | Ý nghĩa học thuật & Thực tiễn |
| :--- | :--- | :--- | :--- |
| **Lĩnh vực ứng dụng** | Giáo dục (Phản hồi của sinh viên về môn học/trường) | **Thương mại điện tử** (Phản hồi của khách hàng về mua sắm trực tuyến) | Chuyển dịch từ nghiên cứu môi trường học thuật sang **tối ưu hóa chiến lược kinh doanh thực tế**. |
| **Quy mô dữ liệu** | ~16,000 dòng sạch được gán nhãn | **~2.000.000 dòng** dữ liệu thô lớn | Giải quyết bài toán dữ liệu lớn thực tế (**Big Data**) với độ nhiễu cao. |
| **Tính chất bài toán** | Phân loại đơn nhãn (chọn 1 trong 4 chủ đề: *Lecturer, Curriculum, Facility, Others*) | **Phân loại đa nhãn (Multi-label Classification)** trên 5 khía cạnh. | Phản ánh đúng thực tế: Một câu đánh giá của khách hàng thường chứa nhiều khía cạnh cùng lúc. |
| **Công nghệ / Thuật toán** | Thuật toán truyền thống (Naive Bayes, Maximum Entropy) trên CPU | **Naive Bayes, SVM** (Làm mốc baseline) + **Bi-LSTM (POS Tagging)** + **PhoBERT Multitask Fine-tuning (GPU)** | Cập nhật những công nghệ học sâu và mô hình ngôn ngữ lớn (Transformers) tối tân nhất hiện nay cho tiếng Việt. |
| **Đầu ra thực tiễn** | Đánh giá học thuật trên giấy tờ | **Flask JSON API** + **Streamlit Web Dashboard** + **Ánh xạ chiến lược RACE & cảnh báo eWOM** | Biến mô hình toán học thành **sản phẩm phần mềm thực tế** có thể đưa vào vận hành ngay lập tức. |

---

<a name="phan-4-kich-ban-doi-thoai-truc-tiep"></a>
## PHẦN 4: KỊCH BẢN ĐỐI THOẠI TRỰC TIẾP VỚI GIẢNG VIÊN HƯỚNG DẪN

### Kịch bản 1: Nhắn tin/Email giải trình nhanh để xin duyệt hướng đi
> *"Kính gửi Thầy/Cô,*
>
> *Em xin phép được báo cáo tóm tắt hướng đi và giá trị thực tiễn của đề tài Khóa luận tốt nghiệp của em để nhận thêm định hướng từ Thầy/Cô ạ.*
>
> *Đề tài của em không đi theo lối mòn phân tích cảm xúc chung chung (chỉ phân loại tốt/xấu - vốn ít giá trị thực tế đối với doanh nghiệp). Đề tài tập trung giải quyết bài toán **Phân tích quan điểm đa khía cạnh (Aspect-Based Sentiment Analysis - ABSA)** trên dữ liệu lớn Thương mại điện tử với 2 trụ cột chính:*
>
> 1. **Về mặt Khoa học (Tính học thuật):**
> * *Em kế thừa quy trình KDD và tiền xử lý tiếng Việt chuẩn mực từ bài báo khoa học uy tín **UIT-VSFC**.*
> * *Em nâng cấp bài toán từ đơn nhãn lên **phân loại đa nhãn (Multi-label)** trên 5 khía cạnh đặc thù (`product`, `price`, `delivery`, `service`, `app`).*
> * *Em thực hiện thực nghiệm đối chiếu sâu giữa học máy truyền thống (**Naive Bayes, SVM**) với học sâu (**Bi-LSTM + POS Tagging**) và Transformers tiên tiến (**PhoBERT Multitask Fine-tuning trên GPU Colab**).*
>
> 2. **Về mặt ứng dụng thực tiễn (Giá trị doanh nghiệp):**
> * *5 nhãn khía cạnh này sẽ tự động hóa việc **chấm điểm KPI đối tác vận chuyển và nhà cung cấp**.*
> * *Hệ thống tích hợp chức năng **quản trị khủng hoảng truyền thông (eWOM Alert)**: Tự động phát hiện các lỗi kỹ thuật hệ thống nghiêm trọng (lỗi `app`) có số lượt tương tác (`thumbsupcount`) cao để cảnh báo lập tức qua Slack/Telegram cho đội ngũ kỹ thuật.*
> * *Toàn bộ kết quả được hiển thị trên **Streamlit Dashboard** và kết nối qua **Flask API**.*
>
> *Em rất mong nhận được ý kiến chỉ dẫn của Thầy/Cô để em hoàn thiện đề cương chi tiết ạ. Em cảm ơn Thầy/Cô nhiều!"*

---

### Kịch bản 2: Trả lời trực tiếp khi gặp mặt hoặc bảo vệ đề cương
> *"Dạ thưa Thầy/Cô, em rất cảm ơn câu hỏi cực kỳ thực tế của Thầy/Cô ạ. Đúng như Thầy/Cô đã chỉ dạy, nếu hệ thống chỉ gán nhãn rồi phán một câu là 'được' hay 'không được' thì doanh nghiệp họ không cần vì nó quá mơ hồ.*
>
> *Tuy nhiên, điểm khác biệt cốt lõi ở đề tài của em là **sử dụng 5 nhãn khía cạnh này làm 'bộ lọc thông minh' để cấu trúc hóa tự động 2 triệu dòng dữ liệu thô lớn.** *
>
> *Khi dữ liệu đã được cấu trúc hóa tự động thành các cặp khía cạnh - sắc thái chi tiết, doanh nghiệp sẽ ứng dụng trực tiếp vào vận hành:*
> * * **Thứ nhất**: Ban giám đốc sử dụng nhãn `delivery` để chấm điểm tự động các đơn vị giao hàng thứ ba. Cuối tháng, bên nào bị chê quá nhiều sẽ bị phạt hợp đồng.*
> * * **Thứ hai**: Đội ngũ kỹ thuật sử dụng nhãn `app` để khoanh vùng nhanh các lỗi ứng dụng nghiêm trọng (ví dụ lỗi thanh toán) bằng cách lọc các câu tiêu cực về `app` rồi chạy gom cụm từ khóa, giúp rút ngắn thời gian sửa lỗi từ vài tuần xuống còn vài giờ.*
> * * **Thứ ba**: Đội ngũ CSKH sử dụng nhãn `service` tiêu cực để tự động kích hoạt chiến dịch chăm sóc, tặng voucher đền bù khách hàng chủ động, giảm tỷ lệ khách hàng rời bỏ sàn.*
>
> *Vì thế, việc huấn luyện mô hình 5 nhãn khía cạnh này chính là chìa khóa để **tự động hóa vận hành cấp phòng ban** cho các doanh nghiệp TMĐT quy mô lớn ạ."*

---

<a name="phan-5-lo-trinh-trien-khai"></a>
## PHẦN 5: LỘ TRÌNH CÁC BƯỚC TRIỂN KHAI KỸ THUẬT TIẾP THEO

Để biến hướng đi này thành hiện thực và có số liệu báo cáo thuyết phục, bạn cần bám sát roadmap kỹ thuật dưới đây:

### Bước 1: Hoàn thiện Tập Kiểm Thử Vàng (Gold Evaluation Set)
-   **Mục tiêu**: Rà soát thủ công **250 dòng** trong file `data/absa_gold_eval_review_queue.csv`.
-   **Ý nghĩa**: Đây là tập dữ liệu "chuẩn mực không tì vết" để làm thước đo đánh giá chính xác hiệu năng của mọi mô hình. Khi có tập Gold Eval sạch, kết quả so sánh mô hình mới được hội đồng công nhận.

### Bước 2: Huấn luyện mô hình Deep Learning trên Google Colab (GPU)
-   **Mục tiêu**: Sử dụng GPU của Colab để huấn luyện mô hình **PhoBERT Multitask** (`scripts/train_phobert_absa.py`) và **Bi-LSTM** (`scripts/train_bilstm_absa.py`).
-   **Hành động**: Sửa đổi nhẹ mã nguồn để tích hợp chuyển đổi sang GPU (`.to('cuda')`) nhằm tận dụng tối đa sức mạnh của Colab, nâng số lượng epoch lên từ 5 - 10 để cải thiện vượt trội chỉ số F1-score so với mốc baseline truyền thống.

### Bước 3: Đóng gói API & Streamlit Dashboard
-   **Mục tiêu**: Chạy thử nghiệm Flask API (`src/ecommerce_absa/api.py`) và giao diện trực quan Streamlit Dashboard (`dashboard/streamlit_app.py`).
-   **Hành động**: Tích hợp mô hình PhoBERT tốt nhất sau khi train từ Colab vào Dashboard, thiết lập logic phát cảnh báo eWOM khi phát hiện review tiêu cực nghiêm trọng có lượt like cao.

### Bước 4: Hoàn thành Quyển Khóa Luận & Slide Bảo Vệ
-   **Mục tiêu**: Viết báo cáo khoa học bám sát 5 chương tiêu chuẩn của KLTN ngành Công nghệ thông tin / Hệ thống thông tin.
-   **Hành động**: Vẽ các biểu đồ so sánh F1-score giữa các mô hình (Naive Bayes vs SVM vs Bi-LSTM vs PhoBERT) trên tập Gold Eval để chứng minh mô hình học sâu PhoBERT mang lại hiệu quả vượt trội như thế nào.
