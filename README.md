# RETECO SemEval-2027 — Track 1a

Đồ án xây dựng hệ thống truy hồi thông tin cho **Track 1a: Temporal Grounded
Retrieval** của cuộc thi RETECO (SemEval 2027).

## Bài toán

Cho một câu hỏi, tìm trong kho văn bản những tài liệu trả lời được câu hỏi đó,
và xếp những tài liệu đúng lên đầu danh sách.

Điểm đặc thù: câu hỏi mang **ràng buộc về thời gian**. Một tài liệu đúng chủ đề
nhưng mô tả tình trạng ở thời điểm khác vẫn bị tính là không liên quan.

| Hạng mục | Nội dung |
| --- | --- |
| Số nhóm chủ đề | 13 |
| Kho văn bản | 1.654.055 tài liệu |
| Câu hỏi | 1.211 (train) + 519 (dev) |
| Chỉ số đánh giá | nDCG@10, trung bình theo nhóm |
| Mốc cơ sở (BM25) | 0,0879 trên `train` |
| Mục tiêu | Khoảng 0,30 |

Mốc cơ sở lấy từ `RETECO/starter_kit/BASELINE_RESULTS.md`.

## Cấu trúc thư mục

```text
notebooks/
    00_problemStatement.ipynb    Phát biểu bài toán, khảo sát định dạng dữ liệu
    01a_eda_corpus.ipynb         EDA phần 1 — kho văn bản
    01b_eda_queries.ipynb        EDA phần 2 — câu hỏi và đáp án
reteco_data/                     Dữ liệu (không commit, xem phần Cài đặt)
RETECO/                          Bộ công cụ ban tổ chức (không commit)
```

## Cài đặt

Dữ liệu và bộ công cụ không nằm trong repo vì dung lượng lớn (4,5 GB). Tải như sau:

```bash
pip install huggingface_hub
hf download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset \
    --local-dir reteco_data --include "track1_tempo/*"

git clone https://github.com/DataScienceUIBK/RETECO.git
```

Thư viện Python cần có: `numpy`, `matplotlib`.

Mỗi notebook có biến `DATA` ở ô code đầu tiên — sửa cho khớp đường dẫn trên máy
đang chạy.

## Các vấn đề phát hiện ở bước EDA

| # | Vấn đề | Mức độ | Hướng xử lý |
| --- | --- | --- | --- |
| 1 | Khoảng 30% văn bản trong kho là bản sao, có nhóm trên 60% | Cao | Khử trùng lặp có ánh xạ ngược |
| 2 | Mã tài liệu để lộ đâu là đáp án | Nghiêm trọng | Chỉ ghi nhận, **không sử dụng** |
| 3 | Số câu hỏi giữa các nhóm chênh khoảng 80 lần | Cao | Kiểm định thống kê phải tính theo nhóm |
| 4 | Câu hỏi chứa HTML, văn bản trong kho thì không | Trung bình | Loại thẻ HTML khi tiền xử lý |
| 5 | Khử trùng lặp ngây thơ làm mất 1–3% đáp án | Thấp nhưng âm thầm | Ánh xạ sang mọi mã cùng nội dung |
| 6 | Câu hỏi không có kết quả bị loại khỏi trung bình, không tính 0 | Cao | Luôn ghi kèm `num_topics` |

Chi tiết và số liệu đầy đủ nằm trong hai notebook EDA.

### Về vấn đề 2

Toàn bộ tài liệu là đáp án đều mang mã theo một khuôn dạng riêng, trong khi
khuôn dạng đó chỉ chiếm 11–26% kho văn bản. Lọc theo khuôn dạng mã sẽ loại được
75–89% kho mà không mất một đáp án nào.

Đây là dấu vết của quy trình xây dựng bộ dữ liệu, không phải đặc tính của bài
toán. Đặc điểm này **không được đưa vào hệ thống dự thi**, vì ba lý do: nó không
giải quyết bài toán, không có gì bảo đảm còn đúng trên tập test ẩn, và kết quả
thu được sẽ không phản ánh chất lượng thực của hệ thống.

## Nguyên tắc thực nghiệm

1. **Phát triển trên `train`.** Tập `dev` chỉ chạy một lần duy nhất ở bước cuối,
   sau đó không điều chỉnh hệ thống nữa.
2. **So sánh theo nhóm.** Điểm cuối là trung bình của 13 nhóm, nên phép kiểm
   định thống kê cũng phải tính theo nhóm, không gộp chung toàn bộ câu hỏi.
3. **Luôn báo cáo `num_topics`.** Hai con số điểm chỉ so sánh được khi số câu
   hỏi được chấm bằng nhau.
4. **Không kết luận từ một nhóm đơn lẻ.** Độ khó giữa các nhóm chênh tới 14 lần.
5. **Chỉ truy hồi trong kho đã cho.** Không bổ sung dữ liệu từ nguồn ngoài.
6. **Công khai mô hình đã dùng.** Khai báo đầy đủ mô hình, API và phiên bản
   trong báo cáo cuối.

## Tiến độ

- [x] Notebook 00 — phát biểu bài toán
- [x] Notebook 01a — EDA kho văn bản
- [x] Notebook 01b — EDA câu hỏi và đáp án
- [ ] Tiền xử lý và hệ thống cơ sở
- [ ] Thực nghiệm cải tiến
- [ ] Chạy `dev` và viết báo cáo

## Nguồn

- Trang cuộc thi: https://datascienceuibk.github.io/RETECO/
- Bộ dữ liệu: https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027
- Bộ công cụ: https://github.com/DataScienceUIBK/RETECO
