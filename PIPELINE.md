# Quy trình làm việc

Tài liệu này trả lời bốn câu hỏi:

1. Notebook nào làm gì, theo thứ tự nào
2. Còn thiếu gì so với đề cương
3. Làm sao biết một cải tiến là thật hay chỉ là may mắn
4. Khi kết quả chưa đạt thì làm gì tiếp

---

## Phần 1 — Hai mục tiêu

| Mục tiêu | Nội dung | Đo bằng |
| --- | --- | --- |
| A | Đủ sáu hạng mục của đề cương | Bảng đối chiếu ở Phần 3 |
| B | Điểm cao trên bảng xếp hạng (~0,30) | nDCG@10 |

Hai mục tiêu kéo về hai hướng khác nhau:

- Mục tiêu A cần mô hình cổ điển, công thức, chỉ mục. Chi phí thấp, chạy CPU.
- Mục tiêu B cần mô hình nơ-ron. Chi phí cao, cần GPU.

**Thứ tự đã chọn: A trước, B sau.** A là phần nền tảng, chi phí thấp, và một
phần của A (tinh chỉnh tham số) còn nâng điểm cho B luôn — đã xảy ra đúng như
vậy ở `05`, khi chỉnh `k1` và `b` nâng điểm 29%.

---

## Phần 2 — Danh sách notebook

| Notebook | Nội dung | HM | Máy | Trạng thái |
| --- | --- | :-: | :-: | --- |
| `00_problemStatement` | Phát biểu bài toán | 1 | CPU | xong |
| `01a_eda_corpus` | Khảo sát kho văn bản | 1, 6 | CPU | xong |
| `01b_eda_queries` | Khảo sát câu truy vấn, đáp án | 1, 6 | CPU | xong |
| `02_baseline` | Hệ thống cơ sở, khung đánh giá | 6 | CPU | xong |
| `03a_preprocessing` | Chọn term, dạng câu truy vấn | 2 | CPU | xong |
| `03a` mục 8 | Danh sách term và tỷ lệ chọn | 2 | CPU | xong |
| `03b_diagnostics` | Chia tập, đo trần | — | CPU | xong |
| `04_retrieval_model` | Sáu mô hình truy xuất | 3 | CPU | xong |
| `04` mục 11 | Query Likelihood, nhánh uni-gram | 3 | CPU | xong |
| `05_ranking_formula` | Công thức xếp hạng, `k1` và `b` | 4 | CPU | xong |
| `06a_indexing` | Chỉ mục ngược, xử lý truy vấn | 5 | CPU | xong |
| `06b_pipeline` | Khung mô-đun | — | CPU | xong |
| `06b` mục 9 | Đủ bộ độ đo, phân tích ca | 6 | CPU | xong |
| `07a_dedup` | Khử trùng lặp | 6 | CPU | chưa làm |
| `07b_rerank` | Xếp hạng lại từ độ sâu 1000 | 6 | GPU | chưa làm |
| `07c_dense` | Mô hình hiểu ngữ nghĩa | 6 | GPU | chưa làm |
| `07d_hybrid` | Kết hợp | 6 | CPU | chưa làm |
| `08_final` | Chạy `dev`, đủ bộ độ đo, tổng hợp | 6 | CPU | chưa làm |

Cột **HM** là số thứ tự hạng mục trong đề cương. Cột **Trạng thái** nhận hai
giá trị: `xong` là đã chạy, `chưa làm` là còn lại trong kế hoạch.

Ba dòng ghi "mục" là các phần thêm vào cuối notebook có sẵn, không phải
notebook riêng. Sáu hạng mục của đề cương đã đủ trên tập `train`.

Danh sách này mở. Thí nghiệm mới nhận chữ cái tiếp theo trong nhóm `07`.

### Sơ đồ

```text
  00 ─► 01a ─► 01b ─► 02 ─► 03a ─► 03b ─► 04 ─► 05 ─► 06a ─► 06b   (xong)
                            │                  │                │
                            ▼                  ▼                ▼
                       03a muc 8          04 muc 11        06b muc 9
                       term va ty le      query likelihood do do + phan tich ca
                            └─────────── ba muc da chay ─────────────┘
                                                                 │
                                       ┌──── PHAN DAY THU HANG ───┤
                                       ▼                          │
                                  07a  khu trung lap              │
                                       ▼                          │
                                  07b  xep hang lai ← buoc lon    │
                                       ▼                          │
                                  07c  mo hinh ngu nghia          │
                                       ▼                          │
                                  07d  ket hop                    │
                                       └───────────┬──────────────┘
                                                   ▼
                                      08   dev mot lan + tong hop
```

### Vì sao thứ tự này

| Ràng buộc | Lý do |
| --- | --- |
| `03b` trước mọi thí nghiệm | Nó chia tập dữ liệu mà mọi so sánh sau đều dùng |
| `04` trước `05` | `k1` và `b` chỉ tồn tại nếu mô hình xác suất thắng ở `04` |
| `06b` trước `07*` | `07*` sinh nhiều hệ thống, cần cache theo chặng |
| `07a` trước `07b` | Khử trùng lặp giảm chi phí GPU và dọn kết quả trùng khỏi top-k |
| `07b` trước `07c` | Rẻ hơn một bậc độ lớn, mà trần đo được lại cao hơn |
| `08` sau cùng | `dev` chỉ chạy một lần |

Mục 9 của `06b` cần `results/runs/` đã có sẵn, tức là cần mục 4 của chính
`06b` đã chạy trước.

### Dừng ở đâu cũng được

| Làm tới | Được gì |
| --- | --- |
| Hiện tại | Đủ hạng mục 1 tới 6 trên tập `train` |
| `08` | Thêm kết quả trên `dev`, đủ đề cương |
| `07b` | Đủ đề cương và có kết quả tốt |
| `07d` | Đủ điều kiện nhắm top 3 |

---

## Phần 3 — Ba chỗ hổng so với đề cương, đã bù

Ba việc, đều chạy CPU, đều đã chạy xong. Code nằm ở cuối ba notebook có sẵn,
không có ô lệnh cũ nào bị sửa, nên phần đã chạy trước đó giữ nguyên kết quả.

### 1. Danh sách term và tỷ lệ chọn — hạng mục 2, kết quả cần có số 4

Đề cương yêu cầu *"danh sách các term được chọn cùng với tỷ lệ term được chọn
trong tổng số term phân tích được từ tất cả tài liệu"*.

Cách đo: quét một lượt toàn bộ kho bằng `tokenize_simple` — đó là bước phân
tích tài liệu, chưa bỏ gì cả, nên kết quả của nó là **mẫu số**. Bộ lọc
(`HTML_WORDS` + `EXTENDED_STOPWORDS`) áp lên sau, cho ra **tử số**. Một lần
tách từ phục vụ cả hai con số.

Tỷ lệ được tính hai cách, vì hai cách trả lời hai câu hỏi khác nhau:

| Đếm theo | Câu hỏi nó trả lời |
| --- | --- |
| Lượt xuất hiện | Chỉ mục nhỏ đi bao nhiêu |
| Term khác nhau | Từ vựng hẹp lại bao nhiêu |

Kết quả trên toàn bộ 1.654.055 tài liệu, quét mất 7,7 phút:

| Đếm theo | Phân tích được | Được chọn | Tỷ lệ |
| --- | --- | --- | --- |
| Lượt xuất hiện | 640.024.414 | 414.168.500 | **64,7%** |
| Term khác nhau | 3.173.933 | 3.173.760 | **99,99%** |

Hai con số cách nhau rất xa, và đó là điều quy tắc chọn term nhắm tới. Danh
sách loại trừ chỉ có 173 từ, nên nó gần như không đụng tới từ vựng. Nhưng 173
từ đó lặp lại trong hầu hết mọi câu, nên chúng gom về 35,3% tổng số lượt xuất
hiện. Riêng `the` chiếm 5,27% kho.

Tỷ lệ giữ lại của 13 nhóm nằm trong khoảng 60,6% (`workplace`) tới 68,8%
(`genealogy`). Một quy tắc viết chung cho kết quả gần như nhau ở mọi nhóm, nên
không nhóm nào cần danh sách riêng.

Đầu ra: `results/terms_selected.txt` (42 MB, không commit) và
`results/term_stats.json` (tỷ lệ, thống kê từng nhóm, 200 term đứng đầu, có
commit).

Một điều đáng chú ý lọt ra từ lượt quét: 20 term *sống sót* đứng đầu gần như
không có từ nào mang nghĩa. Dẫn đầu là `s` với 4.432.973 lượt, sau đó là chữ
số đứng một mình, rồi `url`, `t`, `n`, `c`. Đó là mảnh vụn do cách cắt
`[A-Za-z0-9]+` tách ở dấu nháy. Công tắc `min_length` trong `make_tokenizer`
xử lý được, nhưng chưa đo nên chưa đưa vào.

### 2. Query Likelihood — hạng mục 3, nhánh uni-gram

Đề cương chia mô hình xác suất thành hai nhánh: BIM → 2-Poisson, và uni-gram
(Query Likelihood). Nhánh thứ nhất đã có qua BM25.

Query Likelihood xếp hạng theo `p(q | Md)` với làm trơn Dirichlet. Công thức
tách được thành hai phần, và đó là điều cho phép dùng lại nguyên chỉ mục cũ:

```text
score = SUM(t in q, tf>0) qtf * log(1 + tf / (mu * p(t|C)))   <- ma trận thưa
        - |q| * log(|d| + mu)                                  <- một số/tài liệu
```

Phần thứ nhất là ma trận thưa giống hệt `bm25` và `tfidf`. Phần thứ hai cộng
vào sau khi nhân ma trận, trong `search`.

Hai điều đi kèm:

- Bổ sung số liệu **xác suất term** mà hạng mục 5 liệt kê và trước đó chưa có.
- Query Likelihood cho mọi tài liệu một điểm, kể cả tài liệu không chứa chữ nào
  của câu truy vấn, nên nó không bao giờ trả về danh sách rỗng.

Cài đặt: thêm `qlm` vào `RetrievalIndex.WEIGHTINGS`, cùng chỗ với `bm25` và
`tfidf`. Đã kiểm chứng bằng cách so với một bản cài đặt tay trên kho 300 tài
liệu — thứ tự trùng khớp và chênh lệch điểm là 1,78e-15.

Kết quả sau 10,2 phút, quét 4 giá trị `mu`, chọn `mu` = 1000 trên phần `fit`:

| Mô hình | nDCG@10 | recall@100 | trần@100 | Số câu trả lời |
| --- | --- | --- | --- | --- |
| `qlm` (`mu` = 1000) | **0,1839** | 0,4996 | 0,5571 | 1.211 |
| `bm25` (`k1` = 0,9, `b` = 0,4) | 0,1463 | 0,4075 | 0,4584 | 1.211 |
| `tfidf` | 0,1407 | 0,5084 | **0,5680** | 1.211 |

| So sánh | Chênh lệch | Khoảng tin cậy 95% | Kết luận |
| --- | --- | --- | --- |
| `qlm` vs `bm25` | +0,0376 | [+0,0234, +0,0530] | Thật |
| `qlm` vs `tfidf` | +0,0432 | [+0,0266, +0,0594] | Thật |

`qlm` là mô hình cao điểm nhất trong sáu mô hình của notebook `04`, và là kết
quả duy nhất tới giờ vượt hai mô hình kia một cách có ý nghĩa thống kê.

Cần nói rõ phạm vi: bảng trên so với `bm25` **chưa tinh chỉnh**, vì đó là cấu
hình notebook `04` dùng. `bm25` đã tinh chỉnh ở `05` đạt 0,1867, chênh `qlm`
0,0028 — quá nhỏ để kết luận, và cặp này **chưa chạy kiểm định**. Việc cần làm
để kết luận ghi ở Phần 11.

→ `04_retrieval_model`, mục 11.

### 3. Đủ bộ độ đo và phân tích ca — hạng mục 6

Đề cương yêu cầu ba nhóm độ đo và một phần phân tích:

| Yêu cầu | Trạng thái |
| --- | --- |
| P, R, F1 cho trường hợp không xếp hạng | xong |
| P@k, MAP cho trường hợp có xếp hạng | xong |
| NDCG cho trường hợp có mức độ liên quan | xong |
| Phân tích ca có kết quả cao và thấp | xong |

Qrels chỉ có một mức liên quan, nên nDCG ở đây chạy với mức nhị phân. Điều đó
phải ghi rõ trong báo cáo, và đó cũng là lý do báo cáo thêm MAP: MAP được định
nghĩa sẵn cho nhãn nhị phân, nên hai con số kiểm chứng lẫn nhau.

Việc chấm lại không cần truy xuất lại. Danh sách kết quả của từng hệ thống đã
nằm sẵn ở `results/runs/`, nên các độ đo mới chỉ là chấm lại cùng một danh sách
theo công thức khác.

Phần phân tích ca dựa trên một cách chia mới, `P.failure_breakdown`. Nó xếp mọi
câu truy vấn vào một trong ba nhóm:

| Nhóm | Nghĩa | Số câu | Tỷ lệ | Ai sửa được |
| --- | --- | --- | --- | --- |
| `scored` | Đã có đáp án trong top 10 | 571 | 47,2% | — |
| `recoverable` | Đáp án nằm trong 100 ứng viên, chưa lên top 10 | 361 | **29,8%** | Bước xếp hạng lại |
| `unreachable` | Đáp án không có trong 100 ứng viên | 279 | **23,0%** | Chỉ tầng một, hoặc lấy sâu hơn |

Đo trên `bm25_tuned` (`k1` = 2,0, `b` = 0,9), độ sâu 100, cả 1.211 câu.

Con số 29,8% là trần của `07b` nếu giữ nguyên độ sâu 100: xếp hạng lại chỉ sắp
xếp lại những gì tầng một đưa cho, nên nhiều nhất nó chạm tới được chừng đó câu.

Con số 23,0% mới là điều đổi kế hoạch. Gần một phần tư số câu có đáp án **không
nằm trong 100 ứng viên**, và không bộ xếp hạng lại nào cứu được. Nâng độ sâu
lên 1000 là việc phải làm *trước* khi xếp hạng lại, chứ không phải tùy chọn.

Bảng đủ bộ độ đo của hệ thống đã chốt, trung bình theo 13 nhóm:

| Hệ thống | P@10 | R@10 | F1@10 | MAP | nDCG@10 | R@100 |
| --- | --- | --- | --- | --- | --- | --- |
| `bm25_tuned` | **0,0632** | **0,2371** | **0,0957** | **0,1469** | **0,1867** | **0,5125** |
| `bm25_default` | 0,0496 | 0,1923 | 0,0754 | 0,1150 | 0,1463 | 0,4075 |

Sáu độ đo xếp cùng một thứ tự, nên kết luận không phụ thuộc vào việc chọn độ đo
nào.

Năm câu điểm cao nhất đều đạt nDCG@10 = 1,0000, đều có 1–3 đáp án và đều có
đáp án nằm ở hạng 1. Năm câu điểm thấp nhất chia làm hai kiểu: ba câu đáp án
không nằm trong 100 ứng viên, hai câu đáp án ở hạng 22 và 31.

→ `06b_pipeline`, mục 9. Vài giây.

---

## Phần 4 — Chia tập dữ liệu

### Vấn đề

Thử nhiều phiên bản rồi giữ cái điểm cao nhất thì **cái thắng luôn có vẻ tốt
hơn thực lực**. Mô phỏng với 20 phiên bản có thực lực y hệt nhau:

| Số phiên bản đã thử | Điểm bản thắng cao hơn thực lực |
| --- | --- |
| 5 | +0,0175 |
| 10 | +0,0230 |
| 20 | +0,0280 |
| 40 | +0,0324 |

### Cách giải quyết

```text
train (1.211 câu)
   ├── fit     969 câu    chạy thí nghiệm, CHỌN cấu hình
   └── check   242 câu    xác nhận, KHÔNG dùng để chọn

dev (519 câu)             chạy đúng một lần, ở notebook 08
```

Tỷ lệ 80/20, cắt theo từng nhóm, `seed = 0`.

### Quy tắc chọn

| Bước | Dùng | Được làm gì |
| --- | --- | --- |
| 1. Chọn ứng viên | `fit` | Sắp xếp, chọn bản cao nhất |
| 2. Xác nhận | `check` | Trả lời có/không: bản đó giữ được phong độ chứ? |
| 3. Báo cáo | `dev` | Chạy một lần cho bản đã chốt |

**Không bao giờ sắp xếp theo `check`.** Đề đã dùng để chọn thì không dùng để
chấm được nữa. Sắp theo điểm `train` gộp cũng hỏng, vì `train` chứa `check`.

### Ba quy tắc bắt buộc

1. **Cắt theo từng nhóm.** Cắt ngẫu nhiên toàn bộ sẽ có nhóm không còn câu nào
   ở phần `check`.
2. **Một cách chia duy nhất.** `results/split.json` không được tạo lại sau khi
   đã chạy thí nghiệm.
3. **Chỉ đọc `check` ở mức trung bình 13 nhóm.** `iota` chỉ còn 1 câu ở phần
   `check`.

### Hai phần có giống nhau không

`06b` mục 6 đo, không đọc điểm số nào:

| Tính chất | `fit` | `check` | Chênh |
| --- | --- | --- | --- |
| Số câu | 969 | 242 | |
| Số từ trung vị | 119 | 137,5 | 15,5% |
| Số đáp án trung bình | 3,57 | 3,79 | 6,1% |
| Có hơn một đáp án | 87,0% | 89,3% | 2,6% |
| **Có nhắc tới năm** | **33,5%** | **41,3%** | **23,2%** |

Dòng cuối đáng chú ý. Phần `check` có nhiều câu nhắc tới một năm cụ thể hơn
hẳn. Với bài toán truy hồi theo thời gian, đó là khác biệt có liên quan.

Nó cũng khớp với quan sát khác: điểm trên `check` luôn cao hơn `fit`
(0,2215 so với 0,1797 ở cấu hình đã chốt). Một con số năm là token hiếm và rất
dễ phân biệt, nên BM25 xử lý câu có năm dễ hơn.

**Hệ quả:** điểm `check` cao hơn `fit` một phần vì lý do này, không hẳn vì hệ
thống tốt hơn trên câu khó. Báo cáo phải ghi rõ. Cách kiểm chứng: so điểm giữa
câu có năm và câu không có năm, **chỉ trong phần `fit`** — làm vậy không tiêu
`check`.

---

## Phần 5 — Vòng lặp khi kết quả chưa đạt

**Không đoán bước tiếp theo. Đo trần của từng lựa chọn trước.**

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

Ba phép đo chẩn đoán ở `03b` trả lời ba câu hỏi khác nhau:

| Phép đo | Câu hỏi | Đơn vị |
| --- | --- | --- |
| Trần xếp hạng lại | Sắp xếp lại kết quả hiện có được thêm bao nhiêu? | nDCG@10 |
| Đường cong recall | Lấy sâu hơn bắt thêm được đáp án không? | recall |
| Trần từ vựng | Bao nhiêu đáp án BM25 không bao giờ với tới? | % |

---

## Phần 6 — Kết quả chẩn đoán

| Phép đo | Kết quả |
| --- | --- |
| Đáp án BM25 với tới được | **99,3%** (4.351 / 4.381) |
| Trần xếp hạng lại, độ sâu 100 | 0,5718 |
| Trần xếp hạng lại, độ sâu 200 | 0,6539 |
| Trần xếp hạng lại, độ sâu 1000 | ≥ 0,7296 |
| recall@100 | 0,5125 |

Con số 0,7296 đo trên BM25 chưa chỉnh tham số, nên giờ là **cận dưới**.

### Ba kết luận

**1. BM25 không bị chặn về cấu trúc.** 99,3% tài liệu đáp án chia sẻ ít nhất
một từ với câu truy vấn của nó. Chỉ 30 tài liệu trên 4.381 là ngoài tầm với của
mọi phương pháp dựa trên từ khoá.

Kết quả này đảo ngược dự đoán ban đầu. Trước khi đo, hướng ưu tiên là mô hình
hiểu ngữ nghĩa. Số liệu cho thấy xếp hạng lại rẻ hơn một bậc độ lớn mà trần lại
cao hơn.

**2. Mục tiêu 0,30 nằm trong trần đã đo.**

```text
hiện tại     0,1867
mục tiêu     0,30      ← cần thu 29,4% khoảng trống ở độ sâu 100
trần@100     0,5718                24,4% ở độ sâu 200
trần@200     0,6539
```

**3. Gộp hai mô hình không hơn lấy sâu gấp đôi.**

```text
một mô hình đã chỉnh, độ sâu 200     0,6539
hai mô hình, mỗi bên độ sâu 100      0,6517
```

Chênh 0,0022, trong mức nhiễu. Theo nguyên tắc hoà thì chọn cái đơn giản hơn:
`07b` dùng **một** mô hình BM25 đã chỉnh, lấy sâu hơn.

---

## Phần 7 — Hiện trạng và mục tiêu

| Mốc | nDCG@10 |
| --- | --- |
| Hệ thống cơ sở | 0,0719 |
| BM25 chính thức | 0,0879 |
| Sau `03a` | 0,1463 |
| `qlm` (`mu` = 1000) | 0,1839 |
| **Hiện tại** (`k1=2,0  b=0,9`) | **0,1867** |
| Mục tiêu top 3 | ~0,30 |

Hai dòng cuối chênh nhau 0,0028 và chưa chạy kiểm định, nên tạm coi là hoà.
Việc chốt tầng một ghi ở Phần 11.

### Ước lượng lộ trình

| Sau bước | nDCG@10 | Căn cứ |
| --- | --- | --- |
| Hiện tại | 0,187 | đã đo |
| + xếp hạng lại độ sâu 100 | 0,28–0,33 | trần 0,572, thu 25–40% |
| + xếp hạng lại độ sâu 1000 | 0,30–0,37 | trần ≥ 0,730, thu 20–30% |
| + mô hình ngữ nghĩa | 0,32–0,39 | bù phần 0,7% ngoài tầm với |

Ước lượng, không phải cam kết. Phần trong ngoặc là tỷ lệ khoảng trống thu được,
và đó mới là ẩn số thật.

---

## Phần 8 — Khung mô-đun

Mỗi hệ thống là một file JSON trong `systems/`. Bốn chặng chạy theo thứ tự:

```text
retrieve  ──►  dedup  ──►  rerank  ──►  fuse
 └─ đắt ─┘     └ rẻ ┘     └─ GPU ─┘    └ rẻ ┘
```

Mỗi chặng đánh dấu bằng mã băm của cấu hình nó cộng mọi chặng phía trên. Đổi
chặng cuối thì các chặng trước lấy từ cache.

```python
import pipeline as P
P.configure(DATA, "results")

h = P.run("systems/01_bm25_tuned.json", split="train")
P.score(h)
P.table()                       # sắp theo cột fit
P.compare(h_a, h_b, part="fit") # bootstrap phân tầng theo nhóm
P.split_report()                # fit và check có giống nhau không
P.submit(system, out, tag, split="dev", checker=CHECKER)
```

Bốn hàm dùng cho hạng mục 6, thêm vào cùng với mục 9 của `06b`:

```python
P.metrics()                     # P@10, R@10, F1@10, MAP, nDCG@10, R@100
high, low = P.cases(h, n=5)     # 5 câu điểm cao nhất, 5 câu thấp nhất
P.explain(h, low)               # từng câu: mấy đáp án, nằm ở hạng nào
P.failure_breakdown(h)          # scored / recoverable / unreachable
```

`P.score` giờ ghi thêm các độ đo mới vào file điểm, và đánh số phiên bản bằng
`P.SCORE_VERSION`. File điểm cũ ghi trước khi có các độ đo đó sẽ tự được chấm
lại — chấm lại đọc từ `results/runs/`, không truy xuất lại lần nào.

Ba thư mục hai người cùng ghi:

```text
systems/<tên>.json            định nghĩa hệ thống
results/runs/<hash>.jsonl     danh sách trả về      (không commit)
results/scores/<hash>.json    điểm                  (commit)
results/index.json            tra ngược hash        (commit)
```

Cổng kiểm tra ở `06b` mục 3 đã chạy: khung cho lại đúng con số của `05`, sai số
**0,00e+00**.

---

## Phần 9 — Khi hai người cùng làm

| # | Quy tắc | Vì sao |
| --- | --- | --- |
| 1 | `reteco.py` và `pipeline.py` do một người sở hữu | Hai người sửa cùng lúc thì git không trộn được |
| 2 | Chặng mới viết vào file riêng, đăng ký tên mới | Không ai phải sửa file của người kia |
| 3 | Notebook có một chủ duy nhất | Notebook là JSON nhúng cả ảnh, git trộn ra rác |
| 4 | Mọi kết quả ghi theo khuôn của `pipeline.py` | Ghép được thành một bảng |

Việc giao được ngay, không chặn việc đang chạy:

| Khối | Gồm gì |
| --- | --- |
| `03c` và `04b` | Hai việc còn thiếu của hạng mục 2 và 3, độc lập hoàn toàn |
| Nhánh GPU | `07c`, `07d` — chạy máy khác, chỉ cần đầu ra của `07a` |
| Bộ độ đo | P, R, F1, P@k, MAP thêm vào `reteco.py` |

---

## Phần 10 — Nguyên tắc chung

| # | Nguyên tắc | Bằng chứng |
| --- | --- | --- |
| 1 | Chỉ đổi một thứ mỗi lần | `04` vi phạm một lần và ra kết luận sai về `idf` |
| 2 | Kiểm định theo nhóm | `history` chiếm 46% câu hỏi nhưng 7,7% trọng số điểm |
| 3 | Luôn ghi kèm `num_topics` | `boolean_and` trả lời 141/1.211 câu mà điểm báo cao gấp 3,7 lần |
| 4 | Hoà thì chọn cái đơn giản hơn | Áp dụng khi chốt tầng một ở `05` |
| 5 | Không kết luận từ một nhóm | Trần giữa các nhóm chênh 3,1 lần |
| 6 | Ghi lại cả giả thuyết sai | Bảy giả thuyết đã bị bác |
| 7 | Không dùng rò rỉ mã tài liệu | Phát hiện ở `01a` |
| 8 | Đo trần trước khi đầu tư | `03b` đã đổi thứ tự `07b`/`07c` nhờ đo |
| 9 | Viết lại code đang chạy đúng thì phải có cổng kiểm tra | `04`, `05`, `06b` đều có |
| 10 | `dev` chạy đúng một lần | Luật thi |

---

## Phần 11 — Việc tiếp theo

Sáu hạng mục của đề cương đã đủ trên tập `train`. Từ đây là phần đẩy thứ hạng.

| # | Việc | Chi phí | Được gì |
| :-: | --- | --- | --- |
| 1 | `qlm` vs `bm25` đã tinh chỉnh — chạy kiểm định | 30 phút | Chốt tầng một |
| 2 | `07a` — khử trùng lặp | 20 phút | Giảm chi phí GPU, dọn top-k |
| 3 | `07b` — xếp hạng lại từ độ sâu 1000 | GPU 40 phút | Bước tăng điểm lớn nhất |
| 4 | `07c`, `07d` | GPU | Đẩy thứ hạng |
| 5 | `08` — chạy `dev` một lần, tổng hợp | 1 giờ | Số liệu cuối cùng |

### Về việc 1

Mục 11 của `04` cho `qlm` 0,1839, còn `05` cho `bm25` đã tinh chỉnh 0,1867.
Hai con số chênh nhau 0,0028, nhỏ hơn nhiều so với độ rộng khoảng tin cậy của
mọi phép so đã chạy, nên nhìn bằng mắt thì không kết luận được.

Việc cần làm: đưa `qlm` thành một file trong `systems/`, chạy qua khung ở
`06b`, rồi `P.compare` với `bm25_tuned` trên phần `fit`. Theo nguyên tắc số 5,
hoà thì chọn cái đơn giản hơn — và `qlm` có một tham số trong khi `bm25` có
hai.

Cũng đáng quét `mu` mịn hơn trong khoảng 500–2000, vì `mu` = 500 và `mu` = 1000
chỉ chênh nhau 0,0010 trên `fit`, tức là đỉnh của đường cong khá phẳng.

### Về việc 3

Tầng một chọn theo cột `ceiling`, không theo cột `nDCG@10`. Cấu hình đang chốt:
`k1 = 2,0`, `b = 0,9`, lấy về độ sâu 1000. Nếu việc 1 kết luận `qlm` thắng thì
cấu hình này đổi theo.

Lý do phải nâng độ sâu trước khi xếp hạng lại nằm ở Phần 3 mục 3: ở độ sâu 100
có 23,0% số câu mà đáp án không hề nằm trong danh sách ứng viên.
