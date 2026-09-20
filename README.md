# RETECO SemEval-2027 — Track 1a

Dự án xây dựng công cụ tìm tài liệu cho **Track 1a: Temporal Grounded
Retrieval**, cuộc thi RETECO (SemEval 2027).

---

## 1. Bài toán

Cho tập văn bản `D = {d1, d2, ..., dn}`. Với mỗi câu truy vấn `q`, tìm tập tài
liệu `D' = {di | di ∈ D, di thoả q}` và xếp tài liệu đúng lên đầu danh sách.

Khác một công cụ tìm kiếm thông thường ở một điểm: câu truy vấn mang **ràng
buộc về thời gian**.

Ví dụ: *"Tính đến năm 2017 thì sao?"*, *"Sau lần cập nhật gần nhất thì sao?"*

Một tài liệu đúng chủ đề nhưng mô tả tình trạng ở thời điểm khác vẫn bị tính là
**sai**.

### Miền tri thức

**Nhiều lĩnh vực** — 13 nhóm chủ đề: bitcoin, cardano, economics, genealogy,
history, hsm, iota, law, monero, politics, quant, travel, workplace.

### Dạng bài toán đã chọn

Đề cương liệt kê năm dạng bài toán. Dự án làm bốn:

| Dạng | Làm | Ở đâu |
| --- | :-: | --- |
| Chọn term có hiệu quả tốt nhất | có | `03a` |
| So sánh hai cách tiếp cận | có | `04` |
| Chọn công thức xếp hạng tốt nhất | có | `05` |
| Tìm hiểu tổ chức chỉ mục | có | `06a` |
| Xây dựng ngữ liệu thử nghiệm | không | ngữ liệu đã có sẵn |

---

## 2. Tiến độ theo sáu hạng mục

Ký hiệu trong cột **Trạng thái**:

| Giá trị | Nghĩa |
| --- | --- |
| xong | Đã viết, đã chạy, số liệu nằm sẵn trong notebook |
| chưa làm | Còn lại trong kế hoạch |
| bỏ qua | Cố tình không làm, lý do ghi ở cột bên cạnh |

### Hạng mục 1 — Bài toán tìm kiếm tài liệu

| Kết quả cần có | Trạng thái | Ở đâu |
| --- | --- | --- |
| Phát biểu bài toán, tập tài liệu, câu truy vấn | xong | `00` |
| Quy trình lập chỉ mục và xử lý câu truy vấn | xong | `06a` mục 1, 5 |
| Xác định miền tri thức và tập tài liệu | xong | `01a` |

### Hạng mục 2 — Chọn term

| Kết quả cần có | Trạng thái | Ở đâu |
| --- | --- | --- |
| 1. Loại term được chọn và lý giải | xong | `03a` |
| 2. Phương pháp chọn term | xong | `03a` |
| 3. Thuật toán chọn term | xong | `make_tokenizer` trong `src/reteco.py` |
| 4. Danh sách term được chọn và tỷ lệ trên tổng số term | xong | `03a` mục 8 |

Loại term được chọn: **từ (word)**, cắt theo `[A-Za-z0-9]+`. Khái niệm và
n-gram không dùng, lý do ghi ở `03a`.

### Hạng mục 3 — Mô hình truy xuất tài liệu

| Kết quả cần có | Trạng thái | Ở đâu |
| --- | --- | --- |
| 1. Biểu diễn tài liệu và câu truy vấn | xong | `04` mục 1 |
| 2. Độ đo ước tính độ tương đồng | xong | `04` mục 1 |

Ba hướng mô hình mà đề cương nêu:

| Hướng | Nhánh | Trạng thái | Mô hình đã cài |
| --- | --- | --- | --- |
| Tập hợp | Boolean | xong | `boolean_and`, `boolean_or` |
| Tập hợp | Boolean mở rộng | bỏ qua | Ngoài phạm vi dự án |
| Không gian vector | — | xong | `tfidf`, độ đo cosine |
| Không gian vector | Ngữ nghĩa tiềm ẩn | bỏ qua | Ngoài phạm vi dự án |
| Xác suất | BIM → 2-Poisson | xong | `bm25` |
| Xác suất | Uni-gram, Query Likelihood | xong | `qlm`, làm trơn Dirichlet |

### Hạng mục 4 — Công thức xếp hạng

| Kết quả cần có | Trạng thái | Ở đâu |
| --- | --- | --- |
| 1. Công thức cơ sở tính độ liên quan | xong | `05` mục 1 |
| 2. Công thức tính trọng số term | xong | `05` mục 1, 6 |
| 3. Công thức xếp hạng tài liệu | xong | `05` mục 1 |

Họ công thức BM mà đề cương nhắc tới nằm trong cùng một lưới tham số:

| Tên | Tham số | Đã đo ở `05` |
| --- | --- | --- |
| BM15 | `b = 0` | xong (dòng `no_length`) |
| BM25 | `0 < b < 1` | xong (cả lưới) |
| BM11 | `b = 1` | xong (cột `b = 1,0`) |

### Hạng mục 5 — Lập chỉ mục và xử lý câu truy vấn

| Kết quả cần có | Trạng thái | Ở đâu |
| --- | --- | --- |
| Cấu trúc dữ liệu cho chỉ mục | xong | `06a` mục 1, 3 |
| Thuật toán lập chỉ mục: từ điển và posting | xong | `06a` mục 3, 4 |
| Số liệu: TF, IDF, chiều dài tài liệu | xong | `06a` mục 3 |
| Số liệu: xác suất term | xong | `04` mục 11 — `p(t|C)` của Dirichlet |
| Thuật toán xử lý câu truy vấn | xong | `06a` mục 5 |
| Xử lý câu truy vấn boolean | xong | `04` mục 1, 6 |
| Vị trí term | bỏ qua | Lý do ghi ở `06a` mục 8 |
| Nén chỉ mục | bỏ qua | Lý do ghi ở `06a` mục 8 |
| Cụm term liên tiếp | bỏ qua | Lý do ghi ở `06a` mục 8 |

### Hạng mục 6 — Thử nghiệm và đánh giá

| Kết quả cần có | Trạng thái | Ở đâu |
| --- | --- | --- |
| Mô tả ngữ liệu thử nghiệm | xong | `01a`, `01b` |
| Có mức độ liên quan phân cấp hay không | xong | **Không** — mọi dòng qrels đều bằng 1 |
| Bảng kết quả theo NDCG | xong | `02`, `03a`, `04`, `05` |
| Bảng kết quả theo P, R, F1 | xong | `06b` mục 9 |
| Bảng kết quả theo P@k, MAP | xong | `06b` mục 9 |
| Phân tích ca có kết quả cao và thấp | xong | `06b` mục 9 |

Đánh giá chính dùng **nDCG@10 trung bình theo nhóm**, theo đúng quy định của
ban tổ chức. Qrels chỉ có một mức liên quan, nên nDCG ở đây chạy với mức liên
quan nhị phân.

Sáu hạng mục đã đủ trên tập `train`. Tập `dev` còn để dành cho lần chạy cuối.

| Notebook | Mục | Thời gian thực tế |
| --- | --- | --- |
| `03a_preprocessing` | 8 — danh sách term và tỷ lệ chọn | 7,7 phút |
| `04_retrieval_model` | 11 — Query Likelihood | 10,2 phút |
| `06b_pipeline` | 9 — đủ bộ độ đo, phân tích ca | vài giây |

Cả ba mục đều có bộ nhớ đệm riêng, nên chạy lần thứ hai chỉ mất vài giây.

---

## 3. Mô tả ngữ liệu

| Hạng mục | Giá trị |
| --- | --- |
| Ngôn ngữ | Tiếng Anh |
| Lĩnh vực | 13 nhóm chủ đề, nhiều lĩnh vực |
| Nguồn | Sẵn có, do ban tổ chức phát hành |
| Số tài liệu | 1.654.055 |
| Kích thước | 4,12 GB |
| Độ dài tài liệu trung vị | 241 token |
| Số câu truy vấn | 1.211 (`train`) + 519 (`dev`) |
| Độ dài câu truy vấn trung vị | 119 từ (`fit`) / 137 từ (`check`) |
| Số đáp án trung bình mỗi câu | 3,57 |
| Mức độ liên quan | Nhị phân, chỉ có giá trị 1 |
| Chỉ số đánh giá chính thức | nDCG@10, trung bình theo nhóm |

---

## 4. Các mốc điểm

| Mốc | nDCG@10 | Ghi chú |
| --- | --- | --- |
| Hệ thống cơ sở | 0,0719 | Điểm xuất phát |
| BM25 chính thức | 0,0879 | Mốc ban tổ chức công bố, trên `train` |
| Sau chọn term (`03a`) | 0,1463 | |
| Query Likelihood (`04` mục 11) | 0,1839 | Một tham số, chọn trên `fit` |
| **Hiện tại** (`05`) | **0,1890** | Gấp 2,15 lần mốc chính thức |
| Trần nếu xếp hạng lại độ sâu 100 | 0,5718 | |
| Trần nếu xếp hạng lại độ sâu 200 | 0,6539 | |
| Mục tiêu | ~0,30 | Mức nhóm dẫn đầu |

Mốc 0,0879 lấy từ `RETECO/starter_kit/BASELINE_RESULTS.md`.

---

## 5. Cấu trúc thư mục

```text
src/
    reteco.py                     Nạp dữ liệu, tách từ, chỉ mục, chấm điểm
    pipeline.py                   Khung ghép và so sánh các hệ thống

notebooks/
    00_problemStatement.ipynb     Phát biểu bài toán              [HM 1]
    01a_eda_corpus.ipynb          Khảo sát kho văn bản            [HM 1, 6]
    01b_eda_queries.ipynb         Khảo sát câu truy vấn, đáp án   [HM 1, 6]
    02_baseline.ipynb             Hệ thống cơ sở, khung đánh giá  [HM 6]
    03a_preprocessing.ipynb       Chọn term, dạng câu truy vấn    [HM 2]
    03b_diagnostics.ipynb         Chia tập, đo trần
    04_retrieval_model.ipynb      Sáu mô hình truy xuất           [HM 3]
    05_ranking_formula.ipynb      Công thức xếp hạng, k1 và b     [HM 4]
    06a_indexing.ipynb            Chỉ mục ngược, xử lý truy vấn   [HM 5]
    06b_pipeline.ipynb            Khung mô-đun, bảng đánh giá     [HM 6]
    results/                      Kết quả JSON của mọi notebook

systems/                          Định nghĩa từng hệ thống (JSON)
submissions/                      File nộp định dạng TREC
reteco_data/                      Dữ liệu (không commit)
RETECO/                           Bộ công cụ ban tổ chức (không commit)
```

Lộ trình và việc còn lại: xem [`PIPELINE.md`](PIPELINE.md).

---

## 6. Cài đặt

Dữ liệu 4,12 GB nên không nằm trong repo:

```bash
pip install huggingface_hub
hf download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset \
    --local-dir reteco_data --include "track1_tempo/*"

git clone https://github.com/DataScienceUIBK/RETECO.git
```

Thư viện cần có: `numpy`, `scipy`, `matplotlib`.

Mỗi notebook có biến `DATA` ở ô code đầu tiên — sửa cho khớp máy đang chạy.

---

## 7. Kết quả đã đo

### Chọn term (`03a`)

Lưới 4 bộ tách từ × 4 dạng câu truy vấn, nDCG@10:

| | raw | stripped | title | title_weighted |
| --- | --- | --- | --- | --- |
| simple | 0,0719 | 0,0905 | 0,1316 | 0,1139 |
| nohtml | 0,0986 | 0,0951 | 0,1314 | 0,1190 |
| nohtml_stop | 0,1271 | 0,1267 | 0,1367 | **0,1463** |
| lucene_like | 0,0706 | 0,0937 | 0,1330 | 0,1143 |

### Tỷ lệ term được chọn (`03a` mục 8)

Quét một lượt toàn bộ 1.654.055 tài liệu, 7,7 phút:

| Đếm theo | Phân tích được | Được chọn | Tỷ lệ |
| --- | --- | --- | --- |
| Lượt xuất hiện | 640.024.414 | 414.168.500 | **64,7%** |
| Term khác nhau | 3.173.933 | 3.173.760 | **99,99%** |

Danh sách loại trừ có 173 từ. Chúng chiếm **35,3% tổng số lượt xuất hiện** mà
chỉ là 173 trong hơn 3,17 triệu term khác nhau. Riêng từ `the` đã chiếm 5,27%
kho.

Tỷ lệ giữ lại của từng nhóm nằm trong khoảng 60,6% (`workplace`) tới 68,8%
(`genealogy`), nên một quy tắc viết chung dùng được cho cả 13 nhóm.

Danh sách đầy đủ ở `notebooks/results/terms_selected.txt` (42 MB, không
commit). Thống kê và 200 term đứng đầu ở `term_stats.json` (có commit).

### Mô hình truy xuất (`04`)

| Mô hình | Hướng | nDCG@10 | recall@100 | trần@100 |
| --- | --- | --- | --- | --- |
| `qlm` (`mu` = 1000) | Xác suất, uni-gram | **0,1839** | 0,4996 | 0,5571 |
| `bm25` | Xác suất, BIM | 0,1463 | 0,4075 | 0,4584 |
| `tfidf` | Không gian vector | 0,1407 | **0,5084** | **0,5680** |
| `tf` | Không gian vector, bỏ idf | 0,0021 | 0,0197 | 0,0235 |
| `boolean_or` | Tập hợp | 0,0014 | 0,0103 | |
| `boolean_and` | Tập hợp | trả lời 141/1.211 câu | | |

`bm25` và `tfidf` **hoà nhau**: chênh lệch +0,0055, khoảng tin cậy
[−0,0183, +0,0271].

`qlm` thì thắng cả hai một cách rõ ràng:

| So sánh | Chênh lệch | Khoảng tin cậy 95% | Kết luận |
| --- | --- | --- | --- |
| `qlm` vs `bm25` | +0,0376 | [+0,0234, +0,0530] | Thật |
| `qlm` vs `tfidf` | +0,0432 | [+0,0266, +0,0594] | Thật |

Bảng trên so với `bm25` **chưa tinh chỉnh** (`k1` = 0,9, `b` = 0,4), vì đó là
cấu hình notebook `04` dùng. So với `bm25` đã tinh chỉnh ở `05` (0,1867) thì
`qlm` (0,1839) chênh 0,0028 — quá nhỏ để kết luận, và **chưa chạy kiểm định**
cho cặp này.

Quét `mu` trên bốn giá trị, chọn theo phần `fit`:

| `mu` | fit | check | cả `train` | recall@100 | trần@100 |
| --- | --- | --- | --- | --- | --- |
| 500 | 0,1781 | 0,2104 | 0,1832 | 0,4895 | 0,5480 |
| **1000** | **0,1791** | 0,2087 | **0,1839** | **0,4996** | **0,5571** |
| 2000 | 0,1666 | 0,1911 | 0,1703 | 0,4940 | 0,5480 |
| 4000 | 0,1464 | 0,1724 | 0,1504 | 0,4765 | 0,5298 |

`mu` = 1000 nằm giữa dải đã quét, nên dải này đã bao được điểm tối ưu.

### Công thức xếp hạng (`05`)

Lưới `k1` × `b`, cột trần@100:

```text
k1 | b      0,4     0,6    0,75    0,9     1,0
0,9               0,5154  0,5427  0,5597
1,2      0,4756   0,5293  0,5531  0,5615
1,5      0,4900   0,5402  0,5580  0,5644  0,5703
2,0                              0,5718  0,5629
2,5                              0,5725  0,5581
```

| | `k1` | `b` | nDCG@10 | trần@100 |
| --- | --- | --- | --- | --- |
| Mặc định | 0,9 | 0,4 | 0,1463 | 0,4584 |
| Điểm cao nhất | 1,5 | 0,75 | **0,1890** | 0,5580 |
| **Đã chốt** | **2,0** | **0,9** | 0,1867 | **0,5718** |

Chênh lệch so với mặc định **+0,0427**, khoảng tin cậy [+0,0287, +0,0579].

Giá trị mỗi thành phần trong công thức, đo bằng cách gỡ từng cái ra:

| Gỡ cái gì | Cách gỡ | nDCG@10 | Mất bao nhiêu |
| --- | --- | --- | --- |
| (nguyên bản) | — | 0,1463 | — |
| Trừng phạt tài liệu dài | `b = 0` | 0,0039 | **−0,1424** |
| Vai trò số lần lặp | `k1 → 0` | 0,0053 | −0,1410 |
| Giới hạn của việc lặp | `k1 → ∞` | 0,0881 | −0,0582 |
| Độ hiếm của từ | bỏ `idf` | 0,1066 | −0,0397 |

### Chỉ mục (`06a`)

| Hạng mục | Giá trị |
| --- | --- |
| Tổng số ô khác 0 | 216.320.198 |
| Kích thước chỉ mục | 2,51 GB (59,5% dữ liệu gốc) |
| Thời gian dựng 13 nhóm | 13,6 phút |
| Từ vựng tăng theo | `(số tài liệu)^0,44` |
| Nhanh hơn quét toàn bộ | **2.158 lần** mỗi câu truy vấn |
| Quét toàn bộ cho cả `train` | 43 giờ, so với 14 phút dùng chỉ mục |

### Đủ bộ độ đo (`06b` mục 9)

Hệ thống đã chốt, `bm25` với `k1` = 2,0 và `b` = 0,9, trung bình theo 13 nhóm:

| Hệ thống | P@10 | R@10 | F1@10 | MAP | nDCG@10 | R@100 |
| --- | --- | --- | --- | --- | --- | --- |
| Đã tinh chỉnh | **0,0632** | **0,2371** | **0,0957** | **0,1469** | **0,1867** | **0,5125** |
| Mặc định | 0,0496 | 0,1923 | 0,0754 | 0,1150 | 0,1463 | 0,4075 |

Cả sáu độ đo xếp cùng một thứ tự. Khi các độ đo đồng thuận như vậy thì kết
luận không phụ thuộc vào việc chọn độ đo nào.

`P@10` thấp là chuyện bình thường ở bộ dữ liệu này: mỗi câu chỉ có trung bình
3,57 đáp án, nên trong 10 ô của danh sách nhiều nhất cũng chỉ 3–4 ô có thể
đúng. Trần của `P@10` do đó vào khoảng 0,36 ngay cả với hệ thống hoàn hảo.

Chênh lệch giữa các nhóm còn lớn: `iota` 0,2656 so với `monero` 0,1078, gấp
2,46 lần.

### Phân tích ca cao và thấp (`06b` mục 9)

Chia 1.211 câu truy vấn theo việc *cần gì để sửa*:

| Nhóm | Số câu | Tỷ lệ | Ai sửa được |
| --- | --- | --- | --- |
| Đã trúng trong top 10 | 571 | 47,2% | — |
| Đáp án nằm trong 100 ứng viên, chưa lên top | 361 | **29,8%** | Bước xếp hạng lại |
| Đáp án không có trong 100 ứng viên | 279 | **23,0%** | Chỉ tầng một, hoặc lấy sâu hơn |

Năm câu điểm cao nhất đều đạt nDCG@10 = 1,0000: ít đáp án (1–3), và đáp án
nằm ngay hạng 1.

Năm câu điểm thấp nhất chia làm hai kiểu. Ba câu có đáp án **không nằm trong
100 ứng viên**; hai câu còn lại có đáp án ở hạng 22 và 31, tức là đã tìm ra
nhưng xếp chưa đủ cao.

---

## 8. Những gì đã phát hiện

### Sáu vấn đề của bộ dữ liệu

| # | Vấn đề | Mức độ | Hướng xử lý |
| --- | --- | --- | --- |
| 1 | ~30% kho là bản sao, có nhóm trên 60% | Cao | Khử trùng lặp có ánh xạ ngược |
| 2 | Mã tài liệu để lộ đâu là đáp án | Nghiêm trọng | Ghi nhận, **không sử dụng** |
| 3 | Số câu truy vấn giữa các nhóm chênh ~80 lần | Cao | Kiểm định phải tính theo nhóm |
| 4 | Câu truy vấn có HTML, tài liệu thì không | Trung bình | Loại thẻ HTML khi tiền xử lý |
| 5 | Khử trùng lặp ngây thơ mất 1–3% đáp án | Thấp nhưng âm thầm | Ánh xạ sang mọi mã cùng nội dung |
| 6 | Câu truy vấn không có kết quả bị *loại* khỏi trung bình | Cao | Luôn ghi kèm `num_topics` |

### Về vấn đề 2 — rò rỉ mã tài liệu

Toàn bộ tài liệu là đáp án đều mang mã theo một khuôn dạng riêng, trong khi
khuôn dạng đó chỉ chiếm 11–26% kho. Lọc theo khuôn dạng mã sẽ loại được 75–89%
kho mà không mất một đáp án nào.

Đây là dấu vết của quy trình xây dựng bộ dữ liệu. **Không đưa vào hệ thống dự
thi**, vì ba lý do:

1. Nó khai thác cách dữ liệu được tạo ra, không giải bài toán.
2. Không có gì bảo đảm nó còn đúng trên tập test ẩn.
3. Kết quả thu được sẽ không phản ánh chất lượng thực của hệ thống.

### Tám giả thuyết đã bị số liệu bác bỏ

| Giả thuyết | Thực tế | Phát hiện ở |
| --- | --- | --- |
| Stopword + stemming kiểu Anserini sẽ giúp | Làm **tệ hơn** baseline | `03a` |
| Mô phỏng Lucene analyzer sẽ đóng khoảng cách tới 0,0879 | Không đóng được chút nào | `03a` |
| Loại thẻ HTML là phần thắng lớn nhất | Chỉ đáng +0,0003 khi đã có stop list | `03a` |
| BM25 bị chặn, phải cần mô hình hiểu ngữ nghĩa | 99,3% đáp án **vẫn trong tầm với** | `03b` |
| BM25 hơn hẳn TF-IDF | **Hoà**, và TF-IDF thắng nếu đổi bộ tách từ | `04` |
| `idf` là thành phần quan trọng nhất | Chuẩn hoá độ dài đáng gấp **5 lần** `idf` | `05` |
| Gộp hai mô hình hơn hẳn lấy sâu gấp đôi | Chênh nhau **0,0022**, coi như hoà | `05` |
| `bm25` và `tfidf` đã là hai mô hình cổ điển mạnh nhất | Query Likelihood **thắng cả hai**, chênh lệch thật | `04` |

Ghi lại cả những giả thuyết sai là chủ ý. Nó cho thấy quy trình có tự kiểm tra,
và báo cáo cả phần thất bại.

### Danh sách loại trừ bỏ sót mảnh vụn của từ

Mục 8 của `03a` xếp hạng các term *sống sót* sau khi lọc. Hai mươi term đứng
đầu gần như không có từ nào mang nghĩa:

| Hạng | Term | Số lượt | Đó là gì |
| --- | --- | --- | --- |
| 1 | `s` | 4.432.973 | Mảnh còn lại của sở hữu cách và số nhiều |
| 2 | `1` | 1.452.925 | Chữ số đứng một mình |
| 4 | `url` | 1.247.057 | Dấu vết của đánh dấu liên kết |
| 12 | `t` | 792.532 | Mảnh còn lại của dạng rút gọn (`don't`) |
| 18 | `n` | 619.449 | Mảnh vụn |
| 20 | `c` | 579.525 | Mảnh vụn |

Riêng `s` chiếm 0,69% toàn kho, gấp ba lần term có nghĩa đứng đầu. Cách cắt
`[A-Za-z0-9]+` tách ở dấu nháy, nên `Tesla's` thành `tesla` và `s`. Danh sách
loại trừ là một danh sách cố định nên không chạm tới chúng.

`make_tokenizer` đã có sẵn công tắc `min_length`, đặt `min_length = 2` sẽ bỏ
`s`, `t`, `n`, `c` và chữ số đứng một mình. Ghi lại làm giả thuyết cho một thí
nghiệm sau, chưa đo nên chưa kết luận.

### Một kết luận của `04` đã bị `05` sửa

`04` kết luận `idf` chiếm gần như toàn bộ điểm số, dựa trên `tf` = 0,0021 so
với `tfidf` = 0,1407. Phép so đó đổi hai thứ cùng lúc: cấu hình `tf` vừa không
có `idf`, vừa không có chuẩn hoá độ dài.

`05` tách ra được:

| Cấu hình | `idf` | Chuẩn hoá độ dài | nDCG@10 |
| --- | --- | --- | --- |
| `tf` | không | không | 0,0021 |
| `tfidf_no_idf` | không | có | 0,1186 |
| `tfidf` | có | có | 0,1407 |

Chuẩn hoá độ dài đáng **+0,1165**, `idf` đáng **+0,0221**. Gấp 5 lần.

Sai sót ở `04` vi phạm nguyên tắc "chỉ đổi một thứ mỗi lần" của chính dự án.
Ghi lại ở đây thay vì xoá đi.

---

## 9. Nguyên tắc thực nghiệm

1. **Phát triển trên `train`.** Tập `dev` chỉ chạy một lần ở bước cuối.
2. **Chỉ đổi một thứ mỗi lần.** Có vậy chênh lệch đo được mới quy được cho
   nguyên nhân.
3. **Kiểm định theo nhóm.** Điểm cuối là trung bình 13 nhóm, nên kiểm định
   thống kê cũng phải tính theo nhóm.
4. **Luôn báo cáo `num_topics`.** Hai điểm số chỉ so được khi số câu truy vấn
   được chấm bằng nhau.
5. **Hoà thì chọn cái đơn giản hơn.**
6. **Không kết luận từ một nhóm.** Trần giữa các nhóm chênh 3,1 lần.
7. **Ghi lại cả giả thuyết sai.**
8. **Không dùng rò rỉ mã tài liệu.**
9. **Đo trần trước khi đầu tư.**
10. **Chỉ truy hồi trong kho đã cho**, và công khai mọi mô hình đã dùng.

---

## 10. Nguồn

- Trang cuộc thi: https://datascienceuibk.github.io/RETECO/
- Bộ dữ liệu: https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027
- Bộ công cụ: https://github.com/DataScienceUIBK/RETECO
