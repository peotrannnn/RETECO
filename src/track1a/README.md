# Custom RETECO Track 1a system

Hệ thống Track 1a: retrieval hai tầng (first-stage + reranking), giữ nguyên
cách tách trách nhiệm của starter kit nhưng bỏ Track 1b và Track 2.

## Kiến trúc / Architecture

```text
run_release.py            điều phối nhiều domain, validate, chấm điểm, ghi summary
  └── run_domain.py       chạy 1 domain, ghi TREC run file
        └── query form:   raw | stripped | title | title-weighted
        └── first stage:  retriever.py | dense_retriever.py | hybrid_retriever.py
        └── second stage: reranker.py  (tùy chọn, bật bằng --rerank)
```

### Các thành phần / Components

- **`retriever.py`** — BM25 trên inverted index (scipy sparse), **tokenizer cấu hình được** (xem mục *Tokenization*).
  Công thức y hệt `bm25.py` của starter kit (k1=0.9, b=0.4, idf không âm, có
  tính tần suất từ trong truy vấn) — đã kiểm chứng cho ra **xếp hạng giống
  hệt**, sai khác điểm chỉ ~1e-6 do float32. Khác biệt duy nhất là cách tính:
  điểm BM25 cho từng cặp (tài liệu, từ) được tính sẵn vào ma trận thưa, nên
  mỗi truy vấn chỉ chạm vào posting list của chính các từ trong nó thay vì
  duyệt toàn bộ corpus bằng vòng lặp Python. Nhanh hơn ~233 lần ở bước search
  trên corpus 30k tài liệu — đây là điều kiện bắt buộc để chạy được `history`
  (356k tài liệu). Index được cache ra đĩa.

- **`dense_retriever.py`** — bi-encoder (sentence-transformers) + faiss.
  Encode corpus theo **chunk 20.000 tài liệu**, ghi thẳng vào mảng float16 đã
  cấp phát sẵn. Cắt ngắn **bất đối xứng**: tài liệu 256 token, truy vấn 512.
  Xem mục *Bộ nhớ* và *Cache* bên dưới — hai điểm này là lý do bản trước bị
  OOM trên Kaggle.

- **`hybrid_retriever.py`** — Reciprocal Rank Fusion **có trọng số** giữa
  BM25 và dense. Mặc định `sparse_weight=0.3`, `dense_weight=1.0`: RRF cân
  bằng 50/50 đối xử hai retriever như nhau, nhưng BM25 yếu hơn dense rõ rệt
  trên task này — cho BM25 phiếu ngang hàng là tiêm nhiễu vào một xếp hạng tốt
  hơn. Đây là **giả thuyết, chưa phải kết quả đã tinh chỉnh** — hãy sweep
  `--sparse-weight` trên tập train.

- **`reranker.py`** — cross-encoder rerank trên top-`--candidate-k` của
  first stage. Bi-encoder phải nén cả tài liệu thành một vector *trước khi*
  thấy câu hỏi, nên chỉ đo được độ tương đồng chủ đề thô; cross-encoder đọc
  câu hỏi và tài liệu cùng lúc nên phán đoán được tài liệu này có trả lời
  đúng câu hỏi này không — kể cả điều kiện thời gian mà Track 1a xoay quanh
  ("tính đến 2017", "trước khi fork", "phiên bản mới nhất"). Chi phí là
  `candidate_k` lượt forward mỗi truy vấn, chỉ chạy trên shortlist.

  Dùng `transformers` trực tiếp, **không** dùng `sentence_transformers.CrossEncoder`
  (bản 5.x truyền batch vào model theo vị trí → `AttributeError` trên
  `input_ids.device`). `--rerank-fp16` là **opt-in**: kiến trúc XLM-RoBERTa dễ
  tràn số ở half precision, logits thành `inf`, mọi ứng viên cùng điểm, xếp
  hạng sụp về tie-break theo doc id — mà run vẫn validate và vẫn ra một con số
  nDCG trông hợp lý. Code phát hiện trạng thái đó và tự hạ xuống fp32.

## Tokenization — `--tokenizer`

Đo trên đủ 13 domain (1.211 truy vấn train), macro nDCG@10:

| Preset | macro nDCG@10 | vs baseline | |
| --- | --- | --- | --- |
| `baseline` | 0,0719 | — | chỉ lowercase + `[A-Za-z0-9]+` |
| `stem` | 0,0654 | **−0,0090** | **tệ hơn**, có ý nghĩa thống kê |
| `stop+stem` | 0,0708 | **−0,0041** | **tệ hơn** — đây là mặc định của Anserini |
| `stop_lucene` | 0,0763 | +0,0056 | |
| `stop+stem+html` | 0,0937 | +0,0220 | |
| **`aggressive`** (mặc định) | **0,1154** | **+0,0435** | tốt hơn ở **13/13 domain** |

**Hai điều cần nhớ trước khi đổi:**

1. **Rút gốc từ đơn thuần làm TỆ đi**, và mặc định stopword+stemming kiểu Lucene/Anserini cũng vậy. Cách sửa trực giác nhất lại phản tác dụng. Thứ thực sự có tác dụng là **loại token cấu trúc HTML**: 100% truy vấn mang thẻ HTML còn tài liệu thì không, nên mọi `p`, `href`, `li`, `code` trong truy vấn là một từ khoá chỉ có thể khớp nhiễu. Rút gốc chỉ giúp *sau khi* đã dọn nhiễu đó.
2. **Không phải ăn may trên một chỉ số.** Độ phủ thời gian TC@10 đi cùng chiều: 0,1283 → 0,2266, và số truy vấn phủ đủ mốc thời gian tăng từ 99 lên 174 trên 1.211.

> **`aggressive` gộp 3 thay đổi và được chọn là tốt nhất trong 6 biến thể trên tập train.** Chạy `tokenizer_experiment.py` để tách xem thành phần nào thực sự gánh phần lợi ích — đó là nơi hiện tượng overfit dễ ẩn nhất.

**Tokenizer nằm trong BM25 cache key.** Hai tokenizer khác nhau tạo ra hai index khác nhau; tái sử dụng nhầm là lỗi *trả sai âm thầm* — file run vẫn hợp lệ, điểm vẫn trông hợp lý.

## Query form — `--query-form`

Truy vấn trong bộ dữ liệu này là **nguyên một bài đăng Stack Exchange**: tiêu
đề ngắn + phần thân dài hơn nhiều. Đo trên release: thân dài gấp **4,8–18,0
lần** tiêu đề, và **100% truy vấn có thẻ HTML** trong khi tài liệu gần như
không có thẻ nào.

| Giá trị | Nội dung đưa vào retrieval |
| --- | --- |
| `raw` (mặc định) | nguyên bài đăng, y như starter kit |
| `stripped` | nguyên bài, đã gỡ thẻ HTML và entity |
| `title` | **chỉ tiêu đề** (phần trước `<p>` đầu tiên), đã gỡ thẻ |
| `title-weighted` | tiêu đề lặp 3 lần + nguyên bài |

Đo trên 3 domain nhỏ: dùng **riêng tiêu đề** làm recall@100 tăng **2–3 lần**
(11,9 → 26,4 | 24,2 → 43,9 | 10,8 → 33,5), còn chỉ gỡ HTML thì gần như không
đổi. Tức là **phần thân bài, chứ không phải thẻ HTML, mới là thứ làm loãng
nhu cầu thông tin**. Chưa xác nhận trên đủ 13 domain — đó là việc của notebook 02.

Query form được áp dụng **một lần duy nhất** ngay sau khi load, nên mọi tầng
phía sau (lexical, dense, fusion, rerank) đều thấy cùng một đoạn text.

## Bộ nhớ / Memory

Đây là phần đã làm chết session Kaggle trước đó (SIGKILL, không có traceback —
dấu hiệu của OOM chứ không phải lỗi Python).

- **Encode theo chunk.** Encode cả corpus trong một lệnh `model.encode()` sẽ
  giữ đồng thời: danh sách corpus, một bản sao thứ hai có gắn prefix, list kết
  quả từng batch của sentence-transformers, và mảng output cuối. Trên `history`
  (356k tài liệu, ~1,3GB text) với RAM ~13GB thì đó là OOM. Chia chunk 20k
  giữ đỉnh bộ nhớ ở mức một chunk text + một chunk vector + mảng đích: đo được
  **392MB → 184MB (2,1 lần)** trên corpus tổng hợp, ước tính ~3,5GB → ~1,6GB
  trên `history`.
- **Giải phóng text corpus.** `run_domain.py` đặt `doc_texts = None` ngay sau
  khi dựng xong retriever, trừ khi bật `--rerank` (chỉ reranker cần tra lại
  text). Giải phóng hơn 1GB trên các domain lớn nhất.
- **`array` thay vì `list`** khi dựng BM25 index: 4 byte/phần tử thay vì ~28.
  Trên `history` đây là khác biệt giữa vài trăm MB và vài GB.

## Cache — cái gì làm cache hết hạn

Cache nằm ở `--cache-dir` (mặc định `cache/embeddings`), dùng chung cho mọi
method và **dùng chung cho cả `train` lẫn `dev`** — vì corpus của hai split là
một. Chạy `dev` sau `train` trên cùng domain không phải encode lại.

**Embedding cache key = (model, độ dài cắt tài liệu, corpus, passage prefix).**
Đổi bất kỳ thành phần nào trong bốn thứ đó → phải encode lại từ đầu (cần GPU):

| Thay đổi | Phải encode lại? |
| --- | --- |
| `--model` | **Có** |
| `--max-seq-length` (tài liệu) | **Có** |
| corpus thay đổi (vd. khử trùng lặp làm đổi số tài liệu) | **Có** |
| `--passage-prefix` | **Có** |
| `--query-form` | Không |
| `--query-max-seq-length` | Không |
| `--sparse-weight` / `--dense-weight` / `--rrf-k` / `--pool-k` | Không |
| `--candidate-k`, `--top-k`, mọi tham số rerank | Không |
| chạy `--split dev` sau khi đã chạy `train` | Không |

Hệ quả thực tế: **GPU dùng để mua cái embedding cache, không phải để chạy thí
nghiệm.** Bốn quyết định ở cột "Có" phải chốt *trước* khi tốn quota GPU, vì cả
bốn đều nằm trong cache key. Mọi thứ ở cột "Không" đều sweep được trên CPU sau
đó mà không cần GPU lần nữa.

BM25 index cũng được cache, key theo `(corpus, k1, b)`.

## Setup

```powershell
pip install -r src\track1a\requirements-dense.txt
```

`--method bm25` chỉ cần numpy + scipy. `dense`/`hybrid`/`--rerank` cần thêm
sentence-transformers + torch (+ faiss-cpu, khuyến nghị).

## Chạy / Running

```powershell
# BM25 baseline
python src\track1a\run_release.py --split train --track1 iota --method bm25

# Weighted hybrid
python src\track1a\run_release.py --split train --track1 iota --method hybrid

# Chỉ dùng tiêu đề làm truy vấn
python src\track1a\run_release.py --split train --method bm25 --query-form title

# Pipeline đầy đủ: hybrid + cross-encoder rerank, đủ 13 domain, có thể tiếp tục
python src\track1a\run_release.py --split train --method hybrid --rerank --skip-existing
```

Kết quả ghi ra `runs/track1a/results_<split>_<pipeline>.json`, trong đó
`<pipeline>` ghép từ method + query form + rerank (vd. `bm25`,
`hybrid_title_rerank`) — **mỗi cấu hình một file riêng, không bao giờ ghi đè
lên nhau.**

> **Luật cuộc thi:** chỉ tinh chỉnh trên `--split train`. Chạy `--split dev`
> **đúng một lần**, ở cuối cùng, như một lần kiểm tra held-out.

### Chạy dài không sợ mất tiến độ

Summary được ghi lại **sau mỗi domain**, không phải chờ đến cuối, và **gộp**
vào file cũ thay vì ghi đè. Nếu session Kaggle chết giữa chừng ở domain thứ
9/13, 8 domain đã xong vẫn còn nguyên. Chạy lại đúng lệnh cũ kèm
`--skip-existing` để tiếp tục từ chỗ dừng.

### Test nhanh, không đụng kết quả chính thức

```powershell
python src\track1a\run_release.py --split train --track1 iota --method hybrid --out runs\scratch
```

`runs/scratch/` bị gitignore — dùng cho mọi thử nghiệm tạm. `runs/track1a/`
chỉ chứa kết quả chính thức.

## Tham số đáng chỉnh / Flags worth tuning

| Cờ | Mặc định | Ý nghĩa | Cần GPU lại? |
| --- | --- | --- | --- |
| `--tokenizer` | `aggressive` | biến thể tokenizer (xem bảng trên) | Không |
| `--query-form` | `title-weighted` | dạng truy vấn; dùng `title` khi có rerank | Không |
| `--sparse-weight` | 0.3 | phiếu của BM25 trong RRF (chưa tinh chỉnh) | Không |
| `--dense-weight` | 1.0 | phiếu của dense trong RRF | Không |
| `--pool-k` | 200 | số ứng viên lấy từ *mỗi* retriever trước khi fuse | Không |
| `--candidate-k` | 100 | số ứng viên đưa vào rerank | Không |
| `--query-max-seq-length` | 512 | cắt ngắn **truy vấn** | Không |
| `--max-seq-length` | 256 | cắt ngắn **tài liệu** khi encode | **Có** |
| `--model` | bge-base-en-v1.5 | bi-encoder | **Có** |
| `--passage-prefix` | `""` | prefix gắn vào tài liệu trước khi encode | **Có** |
| `--batch-size` | 256 | batch encode (chỉ ảnh hưởng tốc độ) | Không |
| `--rerank-model` | bge-reranker-base | cross-encoder | Không |
| `--rerank-fp16` | tắt | half precision cho cross-encoder (có kiểm tra tràn số) | Không |
| `--no-fp16` | tắt | fp32 cho bi-encoder (chậm hơn trên GPU) | Không |

## Retriever contract

Mọi retriever đều thoả:

```python
search(query: str, top_k: int) -> list[tuple[doc_id: str, score: float]]
# sắp xếp giảm dần theo score, tie-break xác định theo doc_id
```

Nhờ vậy `reranker.py` bọc được bất kỳ first stage nào, và phương pháp mới
chỉ cần thêm một module + một nhánh trong `build_retriever()` của
`run_domain.py`, không đụng gì đến phần đọc/ghi file. Corpus rỗng trả về `[]`
ở cả ba retriever (gặp khi truyền vào corpus đã lọc/khử trùng lặp).

## Hiện trạng đo được / Measured state

Đo trên đủ **13 domain Track 1** (1.654.055 tài liệu, 1.211 truy vấn train):

| Cấu hình | macro nDCG@10 | R@100 |
| --- | --- | --- |
| `baseline` + `raw` (khởi điểm) | 0,0719 | 0,1995 |
| `aggressive` + `title` | 0,1389 | **0,3910** |
| **`aggressive` + `title-weighted`** | **0,1434** | 0,3799 |

**Mốc đối chiếu — đã sửa.** Con số 0,0879 dùng trước đây **không phải** baseline chính thức. Bảng xếp hạng TEMPO (chính bộ dữ liệu này, 1.730 truy vấn = train + dev, macro trên 13 domain) ghi:

| Hạng | Hệ thống | nDCG@10 |
| --- | --- | --- |
| 1 | DiVeR | **0,320** |
| 2 | E5 | 0,304 |
| 3 | SFR | 0,300 |
| 4–5 | GritLM / ReasonIR | 0,272 |
| 6 | SBERT | 0,249 |
| 10 | **BGE** | **0,220** |
| 11 | Contriever | 0,214 |
| 12 | **BM25** | **0,108** |

Ba hệ quả:

1. Cấu hình hiện tại (0,1434) đã **vượt BM25 của TEMPO** (0,108).
2. **Mục tiêu top 3 là ~0,30**, không phải ~0,09. Còn cách khoảng 2,1 lần.
3. **Mọi hệ thống trên BM25 đều là dense.** Câu hỏi "dense có hiệu quả không" đã được trả lời sẵn trong tài liệu — có, dứt khoát. Đáng chú ý: **BGE xếp hạng 10/12 (0,220)**, mà `bge-base-en-v1.5` chính là model đang đặt mặc định trong repo này. Model nằm trong cache key, nên đây là quyết định phải cân nhắc lại **trước khi** encode.

Các số khác:

- **Trần recall:** R@100 = 0,3910 ở cấu hình tốt nhất. Vì DCG lõm theo số gold lấy được, một reranker hoàn hảo trên shortlist đó đạt **ít nhất 0,391** — tức ≥2,8 lần mức hiện tại. Rerank giờ là khoản đầu tư còn lại có lợi nhất.
- **Trùng lặp corpus:** 1.654.055 → 1.167.099 văn bản phân biệt (**−29,4%**). `bitcoin` 70,4%, `cardano` 67,7%.
- **Khử trùng lặp mất mát về bản chất.** Cùng một đoạn văn bản là gold của **nhiều truy vấn khác nhau** dưới nhiều id khác nhau, nên không có cách chọn đại diện nào cứu được cả. Đo được: chọn-bản-gold cho kết quả **giống hệt** chọn-bản-đầu trên cả 1.211 truy vấn, mà vẫn còn 83/4.381 cặp gold bị phá.
- **Ngưỡng nhiễu:** gộp 1.211 truy vấn thì khoảng tin cậy bootstrap là ±0,045. Một domain 58 truy vấn là ±0,207 — **không kết luận gì từ một domain lẻ.**

## Bước tiếp theo / Next steps

1. **Chạy `tokenizer_experiment.py`** (~20 phút, CPU) để tách `aggressive` thành các thành phần. Nếu một thành phần đơn lẻ đạt gần hết mức lợi ích thì dùng riêng nó — đơn giản hơn, dễ biện minh hơn trong báo cáo, và ít khả năng đang khớp nhiễu của tập train.
2. **Đo rerank** trên shortlist `aggressive` + `title` (~121.000 cặp, rẻ hơn encode dense khoảng một bậc độ lớn). Lợi ích đã đo: ≥2,8 lần. Không đụng tới embedding cache, không chốt gì không đảo ngược được.
3. **Xem lại lựa chọn embedding model trước khi encode.** BGE xếp hạng 10/12 trên bảng TEMPO. Model nằm trong cache key — chọn sai là encode lại toàn bộ 1,17 triệu tài liệu.
4. **Khi encode: chỉ encode 1.167.099 văn bản phân biệt** (giảm 29,4% GPU). Chính sách xuất kết quả cho nhóm trùng lặp là lựa chọn CPU, sweep sau lúc nào cũng được.
5. **`dev` vẫn chưa đụng tới.** Giữ nguyên cho một lần chạy cuối.

> **Nhắc lại:** cấu trúc ID của bộ dữ liệu có rò rỉ thông tin. Chỉ dùng để ghi nhận trong báo cáo, **không** đưa vào hệ thống dự thi.
