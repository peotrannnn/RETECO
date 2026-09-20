# RETECO SemEval-2027 — Track 1a

Đồ án xây dựng hệ thống truy hồi thông tin cho **Track 1a: Temporal Grounded
Retrieval**, cuộc thi RETECO (SemEval 2027).

---

## Bài toán là gì

Cho một câu hỏi, tìm trong kho văn bản những tài liệu trả lời được câu hỏi đó,
và xếp tài liệu đúng lên đầu danh sách.

Giống một công cụ tìm kiếm thu nhỏ. Khác ở một điểm: câu hỏi mang **ràng buộc
về thời gian**.

Ví dụ: *"Tính đến năm 2017 thì sao?"*, *"Sau lần cập nhật gần nhất thì sao?"*

Một tài liệu đúng chủ đề nhưng mô tả tình trạng ở thời điểm khác vẫn bị tính là
**sai**. Đây là phần khó nhất của bài toán.

---

## Số liệu cơ bản

| Hạng mục | Giá trị |
| --- | --- |
| Số nhóm chủ đề | 13 |
| Kho văn bản | 1.654.055 tài liệu |
| Câu hỏi | 1.211 (train) + 519 (dev) |
| Chỉ số đánh giá | nDCG@10, trung bình theo nhóm |

### Các mốc điểm

| Mốc | nDCG@10 | Ghi chú |
| --- | --- | --- |
| Hệ thống cơ sở | 0,0719 | Điểm xuất phát của đồ án |
| BM25 chính thức | 0,0879 | Mốc ban tổ chức công bố, trên `train` |
| **Hiện tại** | **0,1463** | Vượt mốc chính thức 66% |
| Trần đo được | 0,4584 | Nếu xếp hạng lại độ sâu 100 hoàn hảo |
| Mục tiêu | ~0,30 | Mức của nhóm dẫn đầu |

Mốc 0,0879 lấy từ `RETECO/starter_kit/BASELINE_RESULTS.md`.

---

## Đối chiếu yêu cầu môn học

| # | Yêu cầu | Notebook | Trạng thái |
| --- | --- | --- | --- |
| 1 | Bài toán tìm kiếm tài liệu | `00`, `01a`, `01b` | Xong |
| 2 | Chọn term | `03a` | Xong |
| 3 | Mô hình truy xuất tài liệu | `04` | Đang làm |
| 4 | Công thức xếp hạng | `05` | Chưa |
| 5 | Lập chỉ mục và xử lý câu truy vấn | `06` | Chưa |
| 6 | Thử nghiệm và đánh giá | `02`, `07`, `08` | Đang làm |

Lộ trình đầy đủ và thứ tự thực hiện: xem [`PIPELINE.md`](PIPELINE.md).

---

## Cấu trúc thư mục

```text
src/
    reteco.py                     Code dùng chung cho mọi notebook
                                  (RetrievalIndex, BM25, chấm điểm, thống kê)

notebooks/
    00_problemStatement.ipynb     Phát biểu bài toán
    01a_eda_corpus.ipynb          Khảo sát kho văn bản
    01b_eda_queries.ipynb         Khảo sát câu hỏi và đáp án
    02_baseline.ipynb             Hệ thống cơ sở
    03a_preprocessing.ipynb       Chọn term và dạng câu hỏi
    03b_diagnostics.ipynb         Chia tập, đo trần
    04_retrieval_model.ipynb      So năm mô hình truy xuất
    results/                      Kết quả JSON của mọi notebook

reteco_data/                      Dữ liệu (không commit)
RETECO/                           Bộ công cụ ban tổ chức (không commit)
```

---

## Cài đặt

Dữ liệu 4,5 GB nên không nằm trong repo. Tải riêng:

```bash
pip install huggingface_hub
hf download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset \
    --local-dir reteco_data --include "track1_tempo/*"

git clone https://github.com/DataScienceUIBK/RETECO.git
```

Thư viện cần có: `numpy`, `scipy`, `matplotlib`.

Mỗi notebook có biến `DATA` ở ô code đầu tiên — sửa cho khớp máy đang chạy.

---

## Những gì đã phát hiện

### Sáu vấn đề của bộ dữ liệu

| # | Vấn đề | Mức độ | Hướng xử lý |
| --- | --- | --- | --- |
| 1 | ~30% kho là bản sao, có nhóm trên 60% | Cao | Khử trùng lặp có ánh xạ ngược |
| 2 | Mã tài liệu để lộ đâu là đáp án | Nghiêm trọng | Ghi nhận, **không sử dụng** |
| 3 | Số câu hỏi giữa các nhóm chênh ~80 lần | Cao | Kiểm định phải tính theo nhóm |
| 4 | Câu hỏi có HTML, tài liệu thì không | Trung bình | Loại thẻ HTML khi tiền xử lý |
| 5 | Khử trùng lặp ngây thơ mất 1–3% đáp án | Thấp nhưng âm thầm | Ánh xạ sang mọi mã cùng nội dung |
| 6 | Câu hỏi không có kết quả bị *loại* khỏi trung bình | Cao | Luôn ghi kèm `num_topics` |

Chi tiết và số liệu đầy đủ: hai notebook EDA.

### Về vấn đề 2 — rò rỉ mã tài liệu

Toàn bộ tài liệu là đáp án đều mang mã theo một khuôn dạng riêng, trong khi
khuôn dạng đó chỉ chiếm 11–26% kho. Lọc theo khuôn dạng mã sẽ loại được 75–89%
kho mà **không mất một đáp án nào**.

Đây là dấu vết của quy trình xây dựng bộ dữ liệu, không phải đặc tính của bài
toán truy hồi theo thời gian.

**Không đưa đặc điểm này vào hệ thống dự thi**, vì ba lý do:

1. Nó không giải quyết bài toán, chỉ khai thác cách dữ liệu được tạo ra.
2. Không có gì bảo đảm nó còn đúng trên tập test ẩn.
3. Kết quả thu được sẽ không phản ánh chất lượng thực của hệ thống.

### Bốn giả thuyết đã bị số liệu bác bỏ

| Giả thuyết | Thực tế | Phát hiện ở |
| --- | --- | --- |
| Stopword + stemming kiểu Anserini sẽ giúp | Làm **tệ hơn** baseline | `03a` |
| Mô phỏng Lucene analyzer sẽ đóng khoảng cách tới 0,0879 | Không đóng được chút nào | `03a` |
| Loại thẻ HTML là phần thắng lớn nhất | Chỉ đáng +0,0003 khi đã có stop list | `03a` |
| BM25 bị chặn, phải cần mô hình hiểu ngữ nghĩa | 99,3% đáp án **vẫn trong tầm với** | `03b` |

Ghi lại cả những giả thuyết sai là **chủ ý**: nó cho thấy quy trình có tự kiểm
tra, chứ không phải chỉ báo cáo phần thành công.

---

## Trần đã đo được

Notebook `03b` đo xem mỗi hướng cải tiến có thể đạt tối đa bao nhiêu, thay vì
đoán.

| Phép đo | Kết quả |
| --- | --- |
| Điểm hiện tại | 0,1463 |
| Trần nếu xếp hạng lại độ sâu 100 | **0,4584** (gấp 3,1 lần) |
| Trần nếu xếp hạng lại độ sâu 1000 | **0,7296** (gấp 5,0 lần) |
| Đáp án BM25 với tới được | **99,3%** (4.351 / 4.381) |

Con số 99,3% nói rằng đáp án gần như luôn nằm trong danh sách BM25 trả về, chỉ
bị **xếp sai chỗ**. Vấn đề là xếp hạng chứ không phải tìm kiếm — nên bước xếp
hạng lại (`07b`) được ưu tiên trước mô hình hiểu ngữ nghĩa (`07c`), dù kế hoạch
ban đầu định làm ngược lại.

Mục tiêu 0,30 nằm trong trần 0,4584, nên không cần đổi kiến trúc để chạm tới.

---

## Một chỉ mục, năm mô hình

Notebook `04` so năm mô hình truy xuất kinh điển. Cả năm dùng **chung một chỉ
mục**: kho văn bản chỉ được đọc và đếm từ một lần, sau đó mỗi mô hình là một
cách tính trọng số khác nhau trên cùng dãy số đếm đó.

| Mô hình | Trọng số một từ trong một tài liệu |
| --- | --- |
| `boolean_and` | 1 nếu có, và phải có **đủ** mọi từ |
| `boolean_or` | 1 nếu có |
| `tf` | `1 + log(f)` |
| `tfidf` | `(1 + log(f)) × log(N/df)`, rồi chuẩn hoá độ dài |
| `bm25` | công thức Okapi, có `k1` và `b` |

Làm như vậy vì hai lý do. Thứ nhất, nhanh hơn khoảng năm lần. Thứ hai, và quan
trọng hơn: chênh lệch đo được khi đó chắc chắn là chênh lệch **của mô hình**,
không phải của hai chỉ mục khác nhau.

Việc viết lại `reteco.py` cho mục đích này được bảo vệ bằng năm cổng kiểm tra
chạy ở đầu notebook `04`, trong đó có cổng quan trọng nhất: `BM25` vẫn phải cho
kết quả **giống hệt** bản tham chiếu của ban tổ chức.

---

## Nguyên tắc thực nghiệm

1. **Phát triển trên `train`.** Tập `dev` chỉ chạy một lần ở bước cuối, sau đó
   không điều chỉnh hệ thống nữa.
2. **Chỉ đổi một thứ mỗi lần.** Có vậy chênh lệch đo được mới quy được cho
   nguyên nhân.
3. **So sánh theo nhóm.** Điểm cuối là trung bình của 13 nhóm, nên kiểm định
   thống kê cũng phải tính theo nhóm.
4. **Luôn báo cáo `num_topics`.** Hai con số điểm chỉ so được khi số câu hỏi
   được chấm bằng nhau.
5. **Hoà thì chọn cái đơn giản hơn.** Ít thành phần hơn là ít rủi ro khớp nhiễu.
6. **Không kết luận từ một nhóm.** Độ khó giữa các nhóm chênh tới 14 lần.
7. **Chỉ truy hồi trong kho đã cho.** Không bổ sung dữ liệu ngoài.
8. **Công khai mô hình đã dùng** trong báo cáo cuối.

---

## Nguồn

- Trang cuộc thi: https://datascienceuibk.github.io/RETECO/
- Bộ dữ liệu: https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027
- Bộ công cụ: https://github.com/DataScienceUIBK/RETECO
