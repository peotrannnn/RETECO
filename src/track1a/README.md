# Custom RETECO Track 1a system

Hệ thống Track 1a: retrieval hai tầng (first-stage + reranking), giữ nguyên
cách tách trách nhiệm của starter kit nhưng bỏ Track 1b và Track 2.

## Kiến trúc

```text
run_release.py            điều phối nhiều domain, validate, chấm điểm, ghi summary
  └── run_domain.py       chạy 1 domain, ghi TREC run file
        └── first stage:  retriever.py | dense_retriever.py | hybrid_retriever.py
        └── second stage: reranker.py  (tùy chọn, bật bằng --rerank)
```

### Các thành phần

- **`retriever.py`** — BM25 trên inverted index (scipy sparse).
  Công thức y hệt `bm25.py` của starter kit (k1=0.9, b=0.4, idf không âm, có
  tính tần suất từ trong truy vấn) — đã kiểm chứng cho ra **xếp hạng giống
  hệt**, sai khác điểm chỉ ~1e-6 do float32. Khác biệt duy nhất là cách tính:
  điểm BM25 cho từng cặp (tài liệu, từ) được tính sẵn vào ma trận thưa, nên
  mỗi truy vấn chỉ chạm vào posting list của chính các từ trong nó thay vì
  duyệt toàn bộ corpus bằng vòng lặp Python. Nhanh hơn ~233 lần ở bước search
  trên corpus 30k tài liệu — đây là điều kiện bắt buộc để chạy được `history`
  (corpus 1.28GB). Index được cache ra đĩa.

- **`dense_retriever.py`** — bi-encoder (sentence-transformers) + faiss.
  Hỗ trợ fp16, `--max-seq-length`, `--batch-size`. Embedding cache lưu dạng
  float16 (giảm nửa dung lượng đĩa), key theo (model, corpus, max_seq_length)
  chứ không theo split — nên chạy `dev` sau `train` trên cùng domain không
  phải encode lại.

- **`hybrid_retriever.py`** — Reciprocal Rank Fusion **có trọng số** giữa
  BM25 và dense. Mặc định `sparse_weight=0.3`, `dense_weight=1.0`: RRF cân
  bằng 50/50 đối xử hai retriever như nhau, nhưng trên domain pilot `iota`
  BM25 chỉ đạt nDCG@10 0.0558 so với 0.2131 của dense — cho BM25 phiếu ngang
  hàng là tiêm nhiễu vào một xếp hạng tốt hơn nhiều. Đây là **giả thuyết,
  chưa phải kết quả đã tinh chỉnh** — hãy sweep `--sparse-weight` trên tập
  train sau khi có số liệu đủ 13 domain.

- **`reranker.py`** — cross-encoder rerank trên top-`--candidate-k` của
  first stage. Bi-encoder phải nén cả tài liệu thành một vector *trước khi*
  thấy câu hỏi, nên chỉ đo được độ tương đồng chủ đề thô; cross-encoder đọc
  câu hỏi và tài liệu cùng lúc nên phán đoán được tài liệu này có trả lời
  đúng câu hỏi này không — kể cả điều kiện thời gian mà Track 1a xoay quanh
  ("tính đến 2017", "trước khi fork", "phiên bản mới nhất"). Chi phí là
  `candidate_k` lượt forward mỗi truy vấn, chỉ chạy trên shortlist.

## Setup

```powershell
pip install -r src\track1a\requirements-dense.txt
```

`--method bm25` chỉ cần numpy + scipy. `dense`/`hybrid`/`--rerank` cần thêm
sentence-transformers + torch (+ faiss-cpu, khuyến nghị).

## Chạy

```powershell
# BM25 baseline
python src\track1a\run_release.py --split train --track1 iota --method bm25

# Weighted hybrid
python src\track1a\run_release.py --split train --track1 iota --method hybrid

# Pipeline đầy đủ: hybrid + cross-encoder rerank, đủ 13 domain, có thể tiếp tục
python src\track1a\run_release.py --split train --method hybrid --rerank --skip-existing
```

Kết quả ghi ra `runs/track1a/results_<split>_<pipeline>.json`, trong đó
`<pipeline>` là `bm25`, `dense`, `hybrid`, hoặc `hybrid_rerank` — mỗi cấu
hình một file riêng, không bao giờ ghi đè lên nhau.

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

## Tham số đáng chỉnh

| Cờ | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--sparse-weight` | 0.3 | phiếu của BM25 trong RRF (chưa tinh chỉnh) |
| `--dense-weight` | 1.0 | phiếu của dense trong RRF |
| `--max-seq-length` | 256 | cắt ngắn tài liệu khi encode; 512 chậm gấp đôi, không mất đuôi bài dài |
| `--batch-size` | 256 | batch encode |
| `--candidate-k` | 100 | số ứng viên đưa vào rerank |
| `--model` | bge-base-en-v1.5 | bi-encoder |
| `--rerank-model` | bge-reranker-base | cross-encoder |
| `--no-fp16` | tắt | dùng fp32 (chậm hơn trên GPU) |

## Retriever contract

Mọi retriever đều thoả:

```python
search(query: str, top_k: int) -> list[tuple[doc_id: str, score: float]]
# sắp xếp giảm dần theo score
```

Nhờ vậy `reranker.py` bọc được bất kỳ first stage nào, và phương pháp mới
chỉ cần thêm một module + một nhánh trong `build_retriever()` của
`run_domain.py`, không đụng gì đến phần đọc/ghi file.

## Bước tiếp theo

- Sweep `--sparse-weight` sau khi có số liệu 13 domain.
- Thử `bge-large-en-v1.5` / `bge-reranker-v2-m3` nếu còn quota GPU.
- Khai thác `guidance_train.jsonl` (metadata thời gian) để xây bước viết lại
  truy vấn trước retrieval — phần riêng của track temporal.
- Cân nhắc chia tài liệu dài thành passage rồi max-pool, thay vì cắt cứng ở
  `--max-seq-length`.
