# Custom RETECO Track 1a system

Hệ thống Track 1a: retrieval hai tầng (first-stage + reranking), giữ nguyên
cách tách trách nhiệm của starter kit nhưng bỏ Track 1b và Track 2.

> **Lịch sử thí nghiệm nằm ở [`EXPERIMENTS.md`](EXPERIMENTS.md).** File README
> này chỉ ghi *kết luận hiện tại* và cách dùng. Muốn biết vì sao lại chọn như
> vậy, đã thử những gì, và những kết luận nào đã phải sửa — đọc file kia.

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

Đo trên đủ 13 domain (1.211 truy vấn train, `title-weighted`) — ablation đầy đủ, kết quả thô ở `runs/tokenizer_ablation.json`:

| Preset | macro nDCG@10 | macro R@100 | |
| --- | --- | --- | --- |
| `baseline` | 0,1139 | 0,3078 | chỉ lowercase + `[A-Za-z0-9]+` |
| `stem` | 0,1063 | 0,2945 | **tệ hơn**, có ý nghĩa thống kê |
| `html+stem` | 0,1127 | 0,3005 | |
| `html_only` | 0,1189 | 0,3203 | |
| `minlen_only` | 0,1209 | 0,3257 | |
| `stop+stem+html` | 0,1213 | 0,3286 | mặc định kiểu Anserini |
| `html+stem+extstop` | 0,1402 | 0,3772 | |
| `aggressive` | 0,1434 | 0,3848 | mặc định **cũ**, 4 thành phần |
| `extstop_only` | 0,1459 | 0,4129 | |
| `extstop+html` | 0,1462 | 0,4133 | |
| `extstop+html+minlen` | 0,1483 | 0,4190 | hoà với dòng dưới → chọn cái đơn giản hơn |
| **`extstop+minlen`** (mặc định) | **0,1497** | **0,4193** | 2 thành phần, **không** stem, **không** drop_html |

Quyết định chốt bằng bootstrap ghép cặp giữa các biến thể (`pairwise_ci.py`), không phải chỉ so với baseline:

| `extstop+minlen` so với | Δ macro nDCG | Δ macro R@100 | Ý nghĩa TK |
| --- | --- | --- | --- |
| `baseline` | +0,0358 | +0,1115 | **có** |
| `extstop_only` | +0,0038 | +0,0064 | **có** |
| `extstop+html` | +0,0035 | +0,0060 | **có** |
| `aggressive` | +0,0063 | +0,0345 | **có** |
| `extstop+html+minlen` | +0,0014 | +0,0003 | **không (hoà)** → bỏ `drop_html` |

**Bốn điều cần nhớ:**

1. **Stemming gây hại ở mọi nơi nó xuất hiện**: −0,0076 khi thêm vào baseline, −0,0062 khi thêm vào `html_only`, −0,0056 khi thêm vào `extstop_only`. Mặc định stopword+stemming kiểu Anserini chỉ nhỉnh hơn baseline chút ít. Cách sửa trực giác nhất lại phản tác dụng.
2. **Loại token HTML KHÔNG phải phần thắng lớn** — trái với điều README này từng khẳng định. Khi đã có stop list mở rộng, `drop_html` chỉ đáng +0,0003 nDCG@10, không phân biệt được với con số không. Lý do khẳng định cũ *trông có vẻ* đúng: token HTML phần lớn **không tồn tại trong corpus**, mà BM25 chỉ cộng điểm trên các từ tài liệu *thực sự chứa* — nên chúng gần như trơ, chứ không gây hại.
3. **`min_len=2` có tác dụng, nhưng không vì lý do người ta tưởng.** Nó không phải "bỏ từ ngắn vì ít thông tin". Regex `[A-Za-z0-9]+` cắt ở dấu nháy và dấu câu, để lại mảnh vụn: `Tesla's` → `tesla`+`s`, `1980s` → `1980`+`s`, `don't` → `don`+`t`, `January 2, 2017` → `january`+`2`+`2017`. Đo trên `law`: mảnh `s` có mặt trong **56,7% tài liệu**, `1` và `2` trong 22–25%. Một từ phổ biến đến vậy cộng một lượng điểm nhỏ cho hơn nửa corpus, và vì BM25 ở đây có tính tần suất từ trong truy vấn, một query chứa 28 chữ `s` nhân lượng nhiễu đó lên 28 lần. Đây cũng là lý do nó **không trùng** với `drop_html`: `HTML_TOKENS` chỉ có 4 token một ký tự (`a`, `b`, `i`, `p`), nên `drop_html` để nguyên `s`, `t`, `u`, `e` và mọi chữ số đơn. Riêng với task **thời gian**, các mảnh `1`/`2` là tàn dư ngày tháng — nhiễu đội lốt tín hiệu thời gian.
4. **Không phải ăn may trên một chỉ số.** Độ phủ thời gian TC@10 đi cùng chiều với nDCG.

> Có thể còn cách sạch hơn `min_len=2`: xử lý dấu nháy ngay trong tokenizer thay vì dọn mảnh vụn sau đó. Chưa đo — đây là giả thuyết, không phải kết quả.

> **Lỗ hổng đã phát hiện trong thiết kế ablation, đã bịt:** các preset decomposition ban đầu đều dựng theo kiểu "`aggressive` trừ một thành phần", mà `aggressive` luôn chứa `stem` — nên **mọi** tổ hợp được đo đều kéo theo stemming, và tổ hợp của các thành phần *có ích* mà **không có** stem chưa từng được đo. Ba preset `extstop+html`, `extstop+minlen`, `extstop+html+minlen` bịt lỗ hổng đó; chạy `tokenizer_experiment.py --focus` (~15 phút) để đo.

**Đọc bảng kết quả của `tokenizer_experiment.py`:** script in **hai** thống kê khác nhau. Cột `macro nDCG` / `macro R@100` là trung bình theo **domain** — đây là metric chấm điểm của cuộc thi, mọi quyết định đi theo nó. Cột `diff/q` và khoảng tin cậy là trung bình theo **truy vấn**, nên domain nhiều query (history: 561) kéo mạnh hơn domain ít query (iota: 12); chỉ dùng nó để biết "có phải nhiễu không", không dùng để đọc độ lớn.

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
| `--tokenizer` | `extstop+minlen` | biến thể tokenizer (xem bảng trên) | Không |
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
| `aggressive` + `title` | 0,1389 | 0,3910 |
| `aggressive` + `title-weighted` | 0,1434 | 0,3848 |
| **`extstop+minlen` + `title-weighted`** | **0,1497** | **0,4193** |

> Chưa đo `extstop+minlen` + `title` — `title` là dạng truy vấn cho recall cao nhất ở bundle cũ, nên tổ hợp đó nhiều khả năng còn cao hơn 0,4193. Đo trước khi chạy rerank.

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

1. ~~Chạy `tokenizer_experiment.py` để tách `aggressive`~~ — **đã xong.** Kết quả: stemming gây hại, `extstop_only` một mình vượt cả bundle. Mặc định đã đổi sang `extstop_only`.
2. ~~Chạy `tokenizer_experiment.py --focus`~~ — **đã xong.** Chốt `extstop+minlen`: hơn mọi biến thể đơn giản hơn một cách có ý nghĩa thống kê, hoà với `extstop+html+minlen` nên bỏ `drop_html`. Còn lại: một lượt `--focus --query-form title` để tìm tổ hợp tokenizer × query-form cho recall cao nhất.
3. **Đo rerank** trên shortlist recall cao nhất (~121.000 cặp, rẻ hơn encode dense khoảng một bậc độ lớn). Không đụng tới embedding cache, không chốt gì không đảo ngược được.
4. **Xem lại lựa chọn embedding model trước khi encode.** BGE xếp hạng 10/12 trên bảng TEMPO. Model nằm trong cache key — chọn sai là encode lại toàn bộ 1,17 triệu tài liệu.
5. **Khi encode: chỉ encode 1.167.099 văn bản phân biệt** (giảm 29,4% GPU). Chính sách xuất kết quả cho nhóm trùng lặp là lựa chọn CPU, sweep sau lúc nào cũng được.
6. **`dev` vẫn chưa đụng tới.** Giữ nguyên cho một lần chạy cuối.

> **Nhắc lại:** cấu trúc ID của bộ dữ liệu có rò rỉ thông tin. Chỉ dùng để ghi nhận trong báo cáo, **không** đưa vào hệ thống dự thi.
