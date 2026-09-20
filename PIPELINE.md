# Quy trình làm việc

Tài liệu này trả lời ba câu hỏi:

1. Notebook nào làm gì, theo thứ tự nào
2. Làm sao biết một cải tiến là thật hay chỉ là may mắn
3. Khi kết quả chưa đạt thì làm gì tiếp

---

## Phần 1 — Hai mục tiêu

Đồ án phục vụ hai thứ khác nhau:

| Mục tiêu | Nội dung | Được đánh giá bởi |
| --- | --- | --- |
| A | Sáu yêu cầu về truy hồi thông tin cổ điển | Thầy hướng dẫn |
| B | Điểm cao trên bảng xếp hạng (~0,30) | Ban tổ chức cuộc thi |

Hai mục tiêu này **kéo về hai hướng khác nhau**:

- Mục tiêu A cần: mô hình cổ điển, công thức, chỉ mục. Chi phí thấp, chạy CPU.
- Mục tiêu B cần: mô hình nơ-ron. Chi phí cao, cần GPU.

**Thứ tự đã chọn: làm A trước, B sau.** Lý do: A là phần được chấm điểm, chi
phí thấp, và một phần của A (tinh chỉnh tham số) còn có thể cải thiện điểm số
cho B luôn.

---

## Phần 2 — Danh sách notebook

| # | Notebook | Nội dung | YC | Máy | Trạng thái |
| :-: | --- | --- | :-: | :-: | --- |
| | `00_problemStatement` | Phát biểu bài toán | 1 | CPU | Xong |
| | `01a_eda_corpus` | Khảo sát kho văn bản | 1 | CPU | Xong |
| | `01b_eda_queries` | Khảo sát câu hỏi, đáp án | 1 | CPU | Xong |
| | `02_baseline` | Hệ thống cơ sở, khung đánh giá | 6 | CPU | Xong |
| | `03a_preprocessing` | Chọn term, dạng câu hỏi | 2 | CPU | Xong |
| | `03b_diagnostics` | Chia tập, đo trần | — | CPU | Xong |
| 1 | `04_retrieval_model` | So năm mô hình truy xuất | 3 | CPU | **Tiếp theo** |
| 2 | `05_ranking_formula` | Mổ xẻ công thức, quét `k1`×`b` | 4 | CPU | Chưa |
| 3 | `06_indexing` | Chỉ mục ngược, xử lý truy vấn | 5 | CPU | Chưa |
| 4 | `07a_dedup` | Khử trùng lặp | 6 | CPU | Chưa |
| 5 | `07b_rerank` | Xếp hạng lại từ độ sâu 1000 | 6 | GPU nhẹ | Chưa |
| 6 | `07c_dense` | Mô hình hiểu ngữ nghĩa | 6 | GPU nặng | Chưa |
| 7 | `07d_hybrid` | Kết hợp | 6 | CPU | Chưa |
| 8 | `08_final` | Chạy `dev`, tổng hợp | 6 | CPU | Chưa |

Cột **#** là thứ tự làm. Cột **YC** là số thứ tự yêu cầu của thầy.

### Sơ đồ

```text
  03a (xong, 0,1463)
     │
     ▼
   03b   chia tập + đo trần  (xong)
     │
     ├──── PHẦN YÊU CẦU MÔN HỌC ────────
     ▼
    04   so năm mô hình truy xuất
     ▼
    05   công thức + tinh chỉnh k1, b
     ▼
    06   chỉ mục
     │
     ├──── PHẦN ĐẨY THỨ HẠNG ───────────
     ▼
   07a   khử trùng lặp
     ▼
   07b   xếp hạng lại        ← bước tăng điểm lớn nhất
     ▼
   07c   mô hình hiểu ngữ nghĩa
     ▼
   07d   kết hợp
     │
     ▼
    08   chạy dev một lần + tổng hợp
```

### Vì sao thứ tự này

| Ràng buộc | Lý do |
| --- | --- |
| `03b` trước tất cả | Nó chia tập dữ liệu mà mọi thí nghiệm sau đều dùng |
| `04` trước `05` | `k1` và `b` chỉ tồn tại nếu mô hình xác suất thắng ở `04` |
| Mọi mô hình dùng chung một chỉ mục | Chênh lệch đo được khi đó là chênh lệch của mô hình, không phải của chỉ mục |
| `07a` trước `07b` | Khử trùng lặp giảm chi phí GPU và dọn kết quả trùng khỏi top-k |
| `07b` trước `07c` | Rẻ hơn một bậc độ lớn, mà trần đã đo được là cao hơn |
| `08` sau cùng | `dev` chỉ chạy một lần |

### Dừng ở đâu cũng được

Nếu hết thời gian, dừng sau bất kỳ bước nào rồi nhảy thẳng tới `08`.

| Làm tới | Được gì |
| --- | --- |
| `06` | Đủ toàn bộ yêu cầu của thầy |
| `07b` | Đồ án tốt cả về yêu cầu lẫn kết quả |
| `07d` | Đủ điều kiện nhắm top 3 |

---

## Phần 3 — Chia tập dữ liệu

### Vấn đề

Sắp tới sẽ chạy hàng chục thí nghiệm, mỗi lần giữ cấu hình điểm cao nhất.

Thử đủ nhiều thì **sẽ có cấu hình điểm cao chỉ vì may mắn** trên đúng tập dữ
liệu này. Gọi là **khớp nhiễu**.

Cách phát hiện: thử lại trên dữ liệu **chưa từng dùng để chọn**. Nếu vẫn tốt thì
là thật; nếu rớt thì là may mắn.

Mà `dev` chỉ chạy được **một lần** ở cuối, nên không dùng làm chỗ kiểm tra dọc
đường được.

### Cách giải quyết

```text
train (1.211 câu hỏi)
   ├── fit     969 câu    chạy thí nghiệm, chọn cấu hình
   └── check   242 câu    kiểm tra lại, KHÔNG dùng để chọn

dev (519 câu hỏi)         chạy đúng một lần, ở notebook 08
```

Thắng trên `fit` nhưng rớt trên `check` → khớp nhiễu → bỏ.

### Ba quy tắc bắt buộc

1. **Cắt theo từng nhóm.** Cắt ngẫu nhiên toàn bộ sẽ có nhóm không còn câu hỏi
   nào ở phần `check`.
2. **Một cách chia duy nhất.** File `results/split.json` không được tạo lại sau
   khi đã chạy thí nghiệm. Đổi cách chia thì mọi so sánh trước đó mất hiệu lực.
3. **Chỉ đọc `check` ở mức macro 13 nhóm.** Nhóm nhỏ nhất có 7 câu hỏi, nên phần
   `check` của nó chỉ còn 1 câu — vô nghĩa nếu đọc riêng.

### Cách dùng

Chạy thí nghiệm **một lần trên toàn bộ `train`**, rồi đọc ra hai con số từ cùng
lần chạy đó:

```python
R.restrict(results, split, "fit")     # để chọn
R.restrict(results, split, "check")   # để kiểm tra lại
```

Không chạy hai lần — vừa tốn thời gian, vừa có nguy cơ hai lần chạy khác nhau ở
chỗ nào đó ngoài ý muốn.

---

## Phần 4 — Vòng lặp khi kết quả chưa đạt

### Nguyên tắc

**Không đoán bước tiếp theo. Đo trần của từng lựa chọn trước.**

Trần = mức tối đa một hướng có thể đạt nếu làm hoàn hảo. Biết trần thì biết
hướng nào đáng đầu tư, thay vì thử mò.

### Sơ đồ

```text
        đo điểm hiện tại
               ↓
       chạy phép đo chẩn đoán
               ↓
     chọn hướng có TRẦN cao nhất
               ↓
          chạy thí nghiệm
               ↓
      kiểm định trên phần fit
           ┌────┴────┐
      có ý nghĩa     hoà
           ↓          ↓
    thử trên check    BỎ
        ┌──┴──┐
     giữ được  rớt
        ↓        ↓
      CHỐT     BỎ
```

### Ba phép đo chẩn đoán

Notebook `03b` chạy cả ba. Chúng trả lời **ba câu hỏi khác nhau**, không phải ba
cách đo cùng một thứ.

| Phép đo | Câu hỏi | Đơn vị |
| --- | --- | --- |
| Trần xếp hạng lại | Sắp xếp lại các kết quả hiện có, được thêm bao nhiêu điểm? | nDCG@10 |
| Đường cong recall | Lấy sâu hơn 100, bắt thêm được đáp án không? | recall |
| Trần từ vựng | Bao nhiêu đáp án BM25 **không bao giờ** với tới? | % |

---

## Phần 5 — Kết quả chẩn đoán (đã đo trên 13 nhóm)

### Số liệu

| Phép đo | Kết quả |
| --- | --- |
| Điểm hiện tại | 0,1463 |
| Trần xếp hạng lại ở độ sâu 100 | **0,4584** (gấp 3,1 lần) |
| Trần xếp hạng lại ở độ sâu 1000 | **0,7296** (gấp 5,0 lần) |
| recall@100 | 0,4075 |
| recall@1000 | 0,6839 |
| Đáp án BM25 với tới được | **99,3%** (4.351 / 4.381) |

### Ba kết luận

**1. BM25 không bị chặn về cấu trúc.**

99,3% tài liệu đáp án chia sẻ ít nhất một từ với câu hỏi của nó. Chỉ 30 tài
liệu trên tổng số 4.381 là ngoài tầm với của mọi phương pháp dựa trên từ khoá.

Nghĩa là: đáp án gần như luôn nằm đâu đó trong danh sách BM25 trả về — chỉ bị
**xếp sai chỗ**. Vấn đề là xếp hạng, không phải tìm kiếm.

**Điều này đảo ngược dự đoán ban đầu.** Trước khi đo, hướng ưu tiên là mô hình
hiểu ngữ nghĩa (`07c`). Số liệu nói ngược lại: xếp hạng lại (`07b`) rẻ hơn một
bậc độ lớn mà trần lại cao hơn. Đổi thứ tự.

**2. Mục tiêu 0,30 nằm trong trần đã đo.**

```text
hiện tại    0,1463
mục tiêu    0,30      ← cần thu được 47% khoảng trống của độ sâu 100
trần@100    0,4584
trần@1000   0,7296
```

Không cần đổi kiến trúc để chạm mục tiêu. Một bộ xếp hạng lại thu được
khoảng một nửa khoảng trống là đủ.

**3. Trần lệch rất mạnh giữa các nhóm.**

| Nhóm | Hiện tại | Trần@100 | | Nhóm | Hiện tại | Trần@100 |
| --- | --- | --- | --- | --- | --- | --- |
| monero | 0,0622 | 0,2316 | | politics | 0,3333 | 0,7242 |
| bitcoin | 0,0872 | 0,2468 | | cardano | 0,2245 | 0,6985 |
| quant | 0,0367 | 0,2795 | | hsm | 0,2980 | 0,6812 |
| travel | 0,0813 | 0,2978 | | workplace | 0,2040 | 0,5150 |

Điểm cuối là trung bình **13 nhóm bằng nhau**, nên bốn nhóm trần thấp kéo tụt
trần chung. Ngay cả một bộ xếp hạng lại hoàn hảo trên độ sâu 100 cũng chỉ dừng
ở 0,4584.

→ **Hệ quả cho `07b`:** lấy về sâu **1000** rồi mới xếp hạng lại, không phải
100. Đó là cách rẻ nhất nâng trần từ 0,4584 lên 0,7296. Làm hai tầng: một bộ
nhẹ lọc 1000 → 100, một bộ nặng xếp 100 → 10.

### Cách đọc bảng recall cho đúng

Bảng ở mục 4 của `03b` dễ đọc nhầm. Mức tăng recall từ 100 lên 200 (+0,0779)
*lớn hơn* mức tăng từ 50 lên 100 (+0,0638), trông như lợi ích đang tăng lên.

Không phải. Các khoảng có độ rộng khác nhau. Tính trên **mỗi tài liệu thêm
vào**:

| Khoảng | recall thêm được / tài liệu |
| --- | --- |
| 10 → 50 | 0,00379 |
| 50 → 100 | 0,00128 |
| 100 → 200 | 0,00078 |
| 200 → 500 | 0,00034 |
| 500 → 1000 | 0,00019 |

Giảm đều, mỗi bậc rẻ đi khoảng một nửa. Lấy sâu hơn vẫn có ích nhưng ngày càng
đắt — đúng như kỳ vọng.

---

## Phần 6 — Một chỉ mục, năm mô hình

Notebook `04` so năm mô hình truy xuất kinh điển. Nếu làm ngây thơ — dựng lại
chỉ mục riêng cho từng mô hình — thời gian sẽ gấp khoảng năm lần, và tệ hơn, sẽ
không còn chắc rằng chênh lệch đo được là do mô hình chứ không phải do chỉ mục
khác nhau.

Nên `reteco.py` được viết lại quanh một lớp duy nhất, `RetrievalIndex`:

```text
đọc kho văn bản  ──►  đếm từ  ──►  ma trận số đếm   (làm MỘT lần, tốn thời gian)
                                        │
            ┌───────────┬───────────┬───┴────────┬────────────┐
            ▼           ▼           ▼            ▼            ▼
       boolean_and  boolean_or     tf         tfidf         bm25
            └───────────┴───────────┴────────────┴────────────┘
                     đổi trọng số  (rất nhanh)
```

| Mô hình | Trọng số một từ trong một tài liệu |
| --- | --- |
| `boolean_and` | 1 nếu có, và phải có **đủ** mọi từ |
| `boolean_or` | 1 nếu có |
| `tf` | `1 + log(f)` |
| `tfidf` | `(1 + log(f)) × log(N/df)`, rồi chuẩn hoá độ dài |
| `bm25` | công thức Okapi, có `k1` và `b` |

Lớp `BM25` cũ giờ là một trường hợp riêng của lớp này, nên các notebook `02`,
`03a`, `03b` không phải sửa gì.

### Viết lại code đang chạy đúng là việc có rủi ro

Nên notebook `04` mở đầu bằng **năm cổng kiểm tra**, chạy trước mọi kết luận:

| # | Kiểm tra |
| --- | --- |
| 1 | `BM25` vẫn cho kết quả giống hệt bản tham chiếu của ban tổ chức |
| 2 | `RetrievalIndex(weighting="bm25")` giống hệt `BM25` |
| 3 | Đổi qua cả năm mô hình rồi quay về `bm25` thì kết quả không đổi |
| 4 | `boolean_and` luôn là tập con của `boolean_or` |
| 5 | Vectơ `tfidf` của mỗi tài liệu có độ dài bằng 1 |

Thêm một cổng nữa ngay sau khi chạy xong 13 nhóm: cấu hình
`nohtml_stop + bm25` phải cho lại đúng 0,1463 như notebook `03a`. Không đúng
thì dừng.

---

## Phần 7 — Hiện trạng và mục tiêu

| Mốc | nDCG@10 |
| --- | --- |
| Hệ thống cơ sở | 0,0719 |
| BM25 chính thức | 0,0879 |
| **Hiện tại** | **0,1463** |
| Trần nếu xếp hạng lại độ sâu 100 | 0,4584 |
| Trần nếu xếp hạng lại độ sâu 1000 | 0,7296 |
| Mục tiêu top 3 | ~0,30 |

### Ước lượng lộ trình

| Sau bước | nDCG@10 | Căn cứ |
| --- | --- | --- |
| Hiện tại | 0,146 | đã đo |
| + tinh chỉnh `k1`, `b` | 0,15–0,17 | kinh nghiệm chung, chưa đo |
| + xếp hạng lại độ sâu 100 | 0,22–0,28 | trần 0,458, thu được 25–45% |
| + xếp hạng lại độ sâu 1000 | 0,26–0,34 | trần 0,730, thu được 20–32% |
| + kết hợp mô hình ngữ nghĩa | 0,28–0,36 | bù phần 0,7% ngoài tầm với |

**Ước lượng, không phải cam kết.** Phần trong ngoặc là tỉ lệ khoảng trống thu
được — đó mới là ẩn số thật.

### Điều đã thay đổi so với kế hoạch cũ

Kế hoạch cũ cho rằng mô hình hiểu ngữ nghĩa (`07c`) là bắt buộc để chạm 0,30.
Số liệu chẩn đoán bác bỏ điều đó: 99,3% đáp án đã nằm trong tầm với của BM25.

`07c` vẫn có giá trị, nhưng vai trò đổi: không còn là **cứu cánh chính** mà là
**thành phần bổ sung** cho bước kết hợp `07d`, và là cách nâng trần của tầng
truy hồi ở bốn nhóm trần thấp.

---

## Phần 8 — Khi người thứ hai tham gia

Hiện một người làm toàn bộ. Người thứ hai tham gia muộn, nên phần giao lại phải
là những khối **tự chứa, không chặn việc đang chạy**.

| Khối | Gồm gì | Vì sao tự chứa |
| --- | --- | --- |
| Nhánh GPU | `07c`, `07d` | Chạy máy khác, file code riêng, chỉ cần đầu ra của `07a` |
| Nhánh chỉ mục | `06` | Không phụ thuộc kết quả notebook nào |
| Báo cáo | Viết từ `results/*.json` | Mọi số liệu đã nằm sẵn trong file |

### Quy tắc khi có hai người

1. **Một notebook có một chủ.** Không sửa notebook của người kia.
2. **`reteco.py` do người làm chính sở hữu.** Người thứ hai viết vào file riêng.
3. **Mọi kết quả ghi vào `results/<tên>.json`** theo cùng khuôn.
4. **Commit riêng từng notebook.**

---

## Phần 9 — Nguyên tắc chung

Rút ra từ những gì đã đo được ở các notebook trước, không phải quy tắc lý thuyết.

| # | Nguyên tắc | Bằng chứng |
| --- | --- | --- |
| 1 | Chỉ đổi một thứ mỗi lần | Có vậy mới quy được chênh lệch cho nguyên nhân |
| 2 | Kiểm định theo nhóm, không gộp chung | `history` chiếm 46% câu hỏi nhưng 7,7% trọng số điểm |
| 3 | Luôn ghi kèm `num_topics` | Câu hỏi không có kết quả bị *loại* chứ không tính 0 |
| 4 | Hoà thì chọn cái đơn giản hơn | Ít thành phần hơn là ít rủi ro khớp nhiễu |
| 5 | Không kết luận từ một nhóm | Trần giữa các nhóm chênh 3,1 lần (0,232 đến 0,724) |
| 6 | Ghi lại cả giả thuyết sai | Ba giả thuyết bị bác ở `03a`, một ở `03b` |
| 7 | Không dùng rò rỉ mã tài liệu | Phát hiện ở `01a` |
| 8 | `dev` chạy đúng một lần | Luật thi |
| 9 | Đo trần trước khi đầu tư | `03b` đã đổi thứ tự `07b`/`07c` nhờ đo |

---

## Phần 10 — Việc tiếp theo

| # | Việc | Chi phí | Được gì |
| :-: | --- | --- | --- |
| 1 | `04` — so năm mô hình truy xuất | 25–35 phút | Yêu cầu 3 |
| 2 | `05` — công thức + `k1`, `b` | 1 giờ | Yêu cầu 4, có thể tăng điểm |
| 3 | `06` — chỉ mục | 30 phút | Yêu cầu 5 |
| 4 | `07a` — khử trùng lặp | 20 phút | Giảm chi phí bước sau |
| 5 | `07b` — xếp hạng lại từ độ sâu 1000 | GPU 40 phút | Bước tăng điểm lớn nhất |

Sau việc 3 là đã đủ toàn bộ yêu cầu của thầy.
