# Nhật ký thí nghiệm / Experiment log

File này ghi lại **mọi thí nghiệm đã chạy**, theo thứ tự thời gian.

Vì sao cần: các con số trong `README.md` là *kết luận hiện tại*. File này là
*đường đi đến kết luận đó* — gồm cả những lần đi sai. Khi viết báo cáo cuối kỳ,
đây là nơi lấy dữ liệu.

**Quy ước:** mỗi thí nghiệm mới thêm một mục vào cuối. Không sửa mục cũ. Nếu
một kết luận cũ bị chứng minh là sai, thêm mục mới nói nó sai — đừng xoá mục cũ.

---

## Điểm số qua thời gian

| Thời điểm | Cấu hình | macro nDCG@10 | macro R@100 |
| --- | --- | --- | --- |
| Khởi điểm | `baseline` + `raw` | 0,0719 | 0,1995 |
| Sau NB02 | `aggressive` + `title-weighted` | 0,1434 | 0,3848 |
| Hiện tại | `extstop+minlen` + `title-weighted` | **0,1497** | **0,4193** |

Mốc đối chiếu chính thức (bảng xếp hạng TEMPO, 1.730 truy vấn train+dev):

| Hạng | Hệ thống | nDCG@10 |
| --- | --- | --- |
| 1 | DiVeR | 0,320 |
| 2 | E5 | 0,304 |
| 3 | SFR | 0,300 |
| 10 | BGE | 0,220 |
| 12 (cuối) | BM25 | 0,108 |

Hệ thống hiện tại (0,1497) **đã vượt BM25 chính thức** nhưng còn cách top 3
khoảng **2 lần**.

---

## Giai đoạn 1 — Dựng baseline (notebook 01)

### 1.1 Làm BM25 chạy đủ nhanh

**Vấn đề:** BM25 của starter kit duyệt toàn bộ corpus bằng vòng lặp Python cho
mỗi truy vấn. Chạy được với `iota` (10k tài liệu), không chạy nổi với `history`
(356k tài liệu).

**Làm gì:** tính sẵn điểm BM25 cho từng cặp (tài liệu, từ) vào ma trận thưa
scipy. Mỗi truy vấn chỉ chạm posting list của chính các từ trong nó.

**Kết quả:** nhanh hơn **~233 lần** ở bước search. Xếp hạng **giống hệt** bản
gốc, điểm lệch ~1e-6 do float32.

**Kết luận:** đây là điều kiện bắt buộc để chạy được 13 domain.

### 1.2 Sửa lỗi hết RAM trên Kaggle

**Vấn đề:** session bị SIGKILL, không có traceback — dấu hiệu hết RAM chứ không
phải lỗi Python.

**Làm gì:** ba thay đổi.
- Encode corpus theo chunk 20.000 tài liệu thay vì một lần.
- Giải phóng text corpus sau khi dựng xong retriever.
- Dùng `array` thay `list` khi dựng index (4 byte/phần tử thay vì ~28).

**Kết quả:** đỉnh bộ nhớ giảm khoảng 2,1 lần khi encode.

### 1.3 Thử đổi dạng truy vấn (3 domain nhỏ)

**Câu hỏi:** truy vấn là *nguyên một bài đăng Stack Exchange* — tiêu đề ngắn +
thân bài dài gấp 4,8–18 lần. Phần thân có làm loãng nhu cầu thông tin không?

**Làm gì:** so 4 dạng — `raw`, `stripped` (gỡ HTML), `title` (chỉ tiêu đề),
`title-weighted` (tiêu đề × 3 + nguyên bài).

**Kết quả:** dùng **riêng tiêu đề** làm recall@100 tăng **2–3 lần**
(11,9 → 26,4 | 24,2 → 43,9 | 10,8 → 33,5). Chỉ gỡ HTML thì gần như không đổi.

**Kết luận:** thủ phạm là **thân bài**, không phải thẻ HTML. Nhưng mới đo trên
3 domain nhỏ — cần xác nhận trên đủ 13 domain.

### 1.4 Phát hiện rò rỉ cấu trúc ID

Cấu trúc ID của bộ dữ liệu để lộ thông tin về đáp án.

**Chỉ ghi nhận trong báo cáo. TUYỆT ĐỐI KHÔNG đưa vào hệ thống dự thi.**

---

## Giai đoạn 2 — Đo trên đủ 13 domain (notebook 02)

Notebook 02 **cố ý không import** `src/track1a/`. Nó tự viết lại BM25 và metric
từ đầu. Lý do: nếu dùng chính code sản phẩm để kiểm tra code sản phẩm thì không
chứng minh được gì.

### 2.1 Cổng kiểm tra (Part B) — BẮT BUỘC QUA

**Câu hỏi:** bản tự viết trong notebook có tương đương code sản phẩm không?

**Kết quả:** macro nDCG@10 = **0,0719**, khớp **chính xác** con số sản phẩm.
1.211 truy vấn, 1.654.055 tài liệu, 13 domain, 0 truy vấn rỗng, 7,2 phút.

**Kết luận:** QUA. Mọi kết quả phía sau đáng tin.

### 2.2 So 6 tokenizer (Part C)

**Câu hỏi:** stopword + stemming kiểu Anserini có giúp không? (Giả thuyết ban
đầu của tui là **có**.)

**Kết quả:**

| Tokenizer | macro nDCG@10 | vs baseline |
| --- | --- | --- |
| `stem` | 0,0654 | **−0,0090 (TỆ HƠN)** |
| `stop+stem` (mặc định Anserini) | 0,0708 | **−0,0041 (TỆ HƠN)** |
| `stop_lucene` | 0,0763 | +0,0056 |
| `stop+stem+html` | 0,0937 | +0,0220 |
| `aggressive` | 0,1154 | +0,0435 (13/13 domain) |

**Kết luận:** **giả thuyết của tui SAI.** Stemming làm tệ đi. Cách sửa trực
giác nhất lại phản tác dụng.

Kiểm tra chéo: độ phủ thời gian TC@10 cũng tăng (0,1283 → 0,2266), số truy vấn
phủ đủ mốc thời gian tăng 99 → 174. Vậy không phải ăn may trên một chỉ số.

### 2.3 So 4 dạng truy vấn trên đủ 13 domain (Part D)

| Dạng | nDCG@10 | R@100 |
| --- | --- | --- |
| `raw` | 0,0719 | 0,1995 |
| `stripped` | 0,0905 | 0,2402 |
| `title-weighted` | 0,1139 | 0,2876 |
| `title` | 0,1316 | **0,3382** |

**Kết luận:** **hai người thắng khác nhau.** `title` cho recall cao nhất
(thắng 10/13 domain), `title-weighted` cho nDCG@10 cao nhất.

Chọn cái nào phụ thuộc bước sau: recall là **trần** của mọi reranker, nên nếu
có rerank thì dùng `title`.

### 2.4 Kết hợp tokenizer × dạng truy vấn (Part D.1)

| Cấu hình | nDCG@10 | R@100 |
| --- | --- | --- |
| `aggressive` + `title` | 0,1389 | **0,3910** |
| `aggressive` + `title-weighted` | **0,1434** | 0,3799 |

**Kết luận:** hai hiệu ứng **không cộng tuyến tính**. Cộng ngây thơ dự đoán
0,1751, thực tế chỉ 0,1389–0,1434 — chồng lấn khoảng 35%, vì cả hai đều tấn
công cùng một thứ: nhiễu trong truy vấn.

### 2.5 Khử trùng lặp (Part E)

**Phát hiện:** corpus có **29,4% tài liệu trùng lặp** (1.654.055 → 1.167.099
văn bản phân biệt). `bitcoin` trùng 70,4%, `cardano` 67,7%.

**Kết quả lạ:** hai chính sách `keep_first` (giữ bản đầu) và `keep_gold_aware`
(ưu tiên giữ bản là đáp án) cho kết quả **giống hệt nhau** trên cả 1.211 truy
vấn. Điều này đáng ngờ nên tui đã đi tìm nguyên nhân thay vì chấp nhận.

**Nguyên nhân:** một đoạn văn bản trùng lặp có thể là đáp án của **nhiều truy
vấn khác nhau** dưới nhiều id khác nhau cùng lúc. Không có cách chọn đại diện
nào cứu được tất cả.

**Kết luận:** đây là **tính chất của dữ liệu**, không phải lỗi code. Khử trùng
lặp về bản chất là mất mát. Vẫn còn 83/4.381 cặp đáp án bị phá, ảnh hưởng
69/1.211 truy vấn.

### 2.6 Trần của rerank

R@100 = 0,3910 ở cấu hình tốt nhất. Vì DCG lõm theo số đáp án lấy được, một
reranker hoàn hảo trên shortlist đó đạt **ít nhất 0,391** — tức **≥2,8 lần**
mức hiện tại.

**Kết luận:** rerank là khoản đầu tư còn lại có lợi nhất, và nó **không tốn
GPU encode**.

---

## Giai đoạn 3 — Sửa mốc đối chiếu

**Vấn đề:** dự án đang dùng con số 0,0879 làm "baseline BM25 chính thức".

**Làm gì:** tìm lại nguồn gốc con số đó.

**Kết quả:** **0,0879 không tồn tại trong bất kỳ tài liệu TEMPO/RETECO chính
thức nào.** Con số thật trên bảng xếp hạng là **0,108**.

**Ba hệ quả:**
1. Hệ thống hiện tại (0,1434) **đã vượt** BM25 chính thức — điều mà trước đó
   tưởng là chưa đạt.
2. Đích top 3 thật là **~0,30**, không phải ~0,09. Khoảng cách xa hơn nhiều.
3. `bge-base-en-v1.5` — model dense đang đặt mặc định trong code — xếp
   **hạng 10/12** trên bảng này (0,220). E5 xếp hạng 2 (0,304). Cần cân nhắc
   đổi **trước khi** tốn GPU encode, vì tên model nằm trong cache key.

---

## Giai đoạn 4 — Tách `aggressive` ra từng thành phần

`aggressive` gộp 4 thay đổi: stopword mở rộng + stemming + loại HTML +
`min_len=2`. Nó được chọn là tốt nhất trong 6 biến thể trên tập train — mà
"tốt nhất trong N cái trên tập train" chính là nơi hiện tượng overfit ẩn nấp.

### 4.1 Ablation đầy đủ, 13 domain (28,6 phút)

| Biến thể | macro nDCG@10 | macro R@100 |
| --- | --- | --- |
| `baseline` | 0,1139 | 0,3078 |
| `stem` | 0,1063 | 0,2945 |
| `html+stem` | 0,1127 | 0,3005 |
| `html_only` | 0,1189 | 0,3203 |
| `minlen_only` | 0,1209 | 0,3257 |
| `stop+stem+html` | 0,1213 | 0,3286 |
| `html+stem+extstop` | 0,1402 | 0,3772 |
| `aggressive` | 0,1434 | 0,3848 |
| `extstop_only` | **0,1459** | **0,4129** |

**Kết quả bất ngờ:** `extstop_only` — chỉ **một** thay đổi — **vượt** cả bundle
`aggressive` gồm 4 thay đổi. Thắng 10/13 domain trên nDCG và 10/13 trên recall.

**Nguyên nhân:** stemming kéo cả bundle xuống. Nó gây hại ở **mọi** nơi xuất
hiện:

| Thêm stem vào | Hiệu ứng |
| --- | --- |
| `baseline` | −0,0076 |
| `html_only` | −0,0062 |
| `extstop_only` | −0,0056 |

### 4.2 Phát hiện lỗ hổng trong chính thiết kế thí nghiệm

**Vấn đề:** tui dựng toàn bộ decomposition theo kiểu "`aggressive` **trừ** một
thành phần". Nhưng `aggressive` **luôn chứa** `stem`. Nên mọi tổ hợp được đo
đều kéo theo stemming.

**Hệ quả:** tổ hợp của các thành phần *có ích* mà **không có stem** chưa bao
giờ được đo.

**Sửa:** thêm 3 preset `extstop+html`, `extstop+minlen`, `extstop+html+minlen`.

### 4.3 Chạy tập trung 6 biến thể (19,1 phút)

| Biến thể | macro nDCG@10 | macro R@100 | Số thành phần |
| --- | --- | --- | --- |
| `baseline` | 0,1139 | 0,3078 | 0 |
| `aggressive` | 0,1434 | 0,3848 | 4 |
| `extstop_only` | 0,1459 | 0,4129 | 1 |
| `extstop+html` | 0,1462 | 0,4133 | 2 |
| `extstop+html+minlen` | 0,1483 | 0,4190 | 3 |
| **`extstop+minlen`** | **0,1497** | **0,4193** | **2** |

### 4.4 Kiểm định ghép cặp để chốt

**Vấn đề:** script chỉ tính khoảng tin cậy **so với baseline**. Cả
`extstop_only` và `extstop+minlen` đều "có ý nghĩa" so với baseline — điều đó
**không nói gì** về việc chúng có khác nhau thật không.

**Làm gì:** viết `pairwise_ci.py` — so từng cặp biến thể trực tiếp, đọc lại
file JSON đã lưu nên chỉ mất vài giây.

**Kết quả:**

| `extstop+minlen` so với | Δ nDCG | Δ R@100 | Ý nghĩa TK |
| --- | --- | --- | --- |
| `baseline` | +0,0358 | +0,1115 | **có** |
| `extstop_only` | +0,0038 | +0,0064 | **có** |
| `extstop+html` | +0,0035 | +0,0060 | **có** |
| `aggressive` | +0,0063 | +0,0345 | **có** |
| `extstop+html+minlen` | +0,0014 | +0,0003 | **không (hoà)** |

**Chốt:** `extstop+minlen`. Hơn mọi biến thể đơn giản hơn một cách có ý nghĩa
thống kê, hoà với biến thể phức tạp hơn → chọn cái đơn giản hơn, bỏ `drop_html`.

### 4.5 Tìm hiểu vì sao `min_len=2` có tác dụng

Trước đó tui nói `min_len=2` "không có cơ chế lý thuyết nào" nên đáng nghi.
Khi nó hoá ra có ý nghĩa thống kê, tui đi kiểm tra xem nó thực sự xoá cái gì.

**Nó không phải "bỏ từ ngắn vì ít thông tin".** Regex `[A-Za-z0-9]+` cắt ở dấu
nháy và dấu câu, để lại **mảnh vụn**:

| Mảnh | Sinh ra từ | Có mặt trong corpus `law` |
| --- | --- | --- |
| `s` | `Tesla's` → `tesla`+`s`, `1980s` → `1980`+`s` | **56,7% tài liệu** |
| `1`, `2` | `January 2, 2017` → `january`+`2`+`2017` | 22–25% tài liệu |
| `u`, `t`, `e` | `don't` → `don`+`t` | 11–15% tài liệu |

Một từ có mặt trong 57% corpus cộng một lượng điểm nhỏ cho hơn nửa số tài liệu.
Và vì BM25 ở đây **có tính tần suất từ trong truy vấn**, một truy vấn chứa 28
chữ `s` nhân lượng nhiễu đó lên 28 lần.

**Vì sao không trùng với `drop_html`:** `HTML_TOKENS` chỉ có 4 token một ký tự
(`a`, `b`, `i`, `p`). Nên `drop_html` để nguyên `s`, `t`, `u`, `e` và mọi chữ
số đơn.

**Đáng viết vào báo cáo:** mảnh `1`, `2` là **tàn dư ngày tháng**. Trong một
task truy hồi *theo thời gian*, một chữ số trần khớp 25% corpus là nhiễu đội
lốt tín hiệu thời gian.

**Ghi chú:** có thể còn cách sạch hơn — xử lý dấu nháy ngay trong tokenizer
thay vì dọn mảnh vụn sau. **Chưa đo.** Đây là giả thuyết, không phải kết quả.

---

## Những kết luận đã phải sửa

Phần này quan trọng cho báo cáo: nó cho thấy quy trình có tự sửa sai.

| Kết luận cũ | Thực tế | Phát hiện nhờ |
| --- | --- | --- |
| Stopword + stemming kiểu Anserini sẽ giúp | Làm **tệ hơn** baseline | Đo 13 domain (2.2) |
| Baseline BM25 chính thức là 0,0879 | Là **0,108**; 0,0879 không tồn tại | Tra nguồn gốc (GĐ 3) |
| Loại token HTML là phần thắng lớn nhất | Chỉ đáng **+0,0003** khi đã có stop list | Ablation (4.1, 4.3) |
| `aggressive` là tokenizer tốt nhất | Thua `extstop+minlen` ở cả 2 chỉ số | Ablation (4.1) |
| `min_len=2` không có cơ chế, đáng nghi | Có cơ chế thật: dọn mảnh vụn tokenizer | Đếm tần suất (4.5) |

**Bài học chung:** mọi giả thuyết "trực giác" trong danh sách trên đều sai. Chỉ
có đo mới ra kết luận đúng.

---

## Bug đã tìm ra và sửa

| Bug | Hậu quả nếu không sửa |
| --- | --- |
| Tokenizer không nằm trong BM25 cache key | Hai tokenizer dùng chung một index → **sai âm thầm**, run vẫn hợp lệ, điểm vẫn trông hợp lý |
| `run_domain.py --help` crash (`TypeError: %o`) | Dấu `%` trong chuỗi help bị argparse hiểu là mã định dạng. Không xem được help |
| `dense_retriever.search()` crash trên corpus rỗng | Gặp ngay khi bắt đầu lọc/khử trùng lặp corpus |
| Notebook sinh ra JSON hỏng (mất ký tự xuống dòng) | Notebook mở lên không chạy được |
| BM25 chiếm đỉnh 3,3GB RAM | Hết RAM trên domain lớn. Sửa còn 1,47GB |
| Notebook tokenize lại corpus cho mỗi cấu hình | Lãng phí ~45 phút mỗi lần chạy |
| Script in 2 thống kê khác nhau dưới cùng nhãn "gain" | Đọc nhầm bảng → quyết định sai |

Ngoài ra có vài lỗi **của chính tui** khi viết dữ liệu kiểm thử (sai giá trị kỳ
vọng của Porter stemmer, sai số đếm trong fixture). Đều phát hiện bằng cách đối
chiếu với tài liệu gốc — và trong cả hai trường hợp, **code đúng, test sai**.

---

## Chưa làm

| Việc | Vì sao chưa | Tốn gì |
| --- | --- | --- |
| Đo `extstop+minlen` + `--query-form title` | Việc kế tiếp | ~15 phút CPU |
| Đo rerank trên 13 domain | Chờ chốt query-form | CPU, ~121.000 cặp |
| Đổi model dense BGE → E5 | Quyết định, chưa sửa code | 0 phút |
| Pilot dense trên `history` | Chờ có GPU | GPU, nhỏ |
| Encode toàn bộ corpus | Chờ chốt model + dedup | GPU, lớn |
| Sweep `--sparse-weight` cho RRF | Chờ có embedding | CPU |
| Chạy trên `dev` | **Chỉ 1 lần, cuối cùng, theo luật thi** | — |

---

## Hai thứ phải nhớ khi đọc mọi con số

**1. Ngưỡng nhiễu.** Gộp 1.211 truy vấn thì khoảng tin cậy bootstrap là
**±0,045**. Một domain 58 truy vấn là **±0,207**. Đừng bao giờ kết luận từ một
domain lẻ.

**2. Truy vấn rỗng bị LOẠI, không tính 0 điểm.** Scorer chỉ chấm truy vấn có
mặt trong **cả** đáp án lẫn file run. Một hệ thống trả về rỗng cho nửa số truy
vấn sẽ có điểm *cao hơn* một hệ thống trả lời hết. Vì vậy **mọi bảng kết quả
phải ghi `num_topics` bên cạnh điểm số.**
