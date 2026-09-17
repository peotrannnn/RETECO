# RETECO Track 1a

Hệ thống retrieval cho **SemEval-2027 Task 1, Track 1a (Temporal Grounded
Retrieval)** — RETECO shared task.

- Task website: https://datascienceuibk.github.io/RETECO/participate.html
- Dataset: https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027
- Starter kit (ban tổ chức): https://github.com/DataScienceUIBK/RETECO/tree/main/starter_kit

Mục tiêu: top 5 trở lên trên bảng xếp hạng Track 1a. Metric chính thức:
**nDCG@10** (qua `pytrec_eval`), macro-average trên 13 domain của Track 1.

## Cấu trúc repo

```text
reteco-track1a/
├── README.md                  # file này
├── .gitignore
├── src/
│   └── track1a/
│       ├── __init__.py
│       ├── retriever.py          # BM25 thuần Python (baseline lexical)
│       ├── dense_retriever.py    # Bi-encoder + faiss/numpy (semantic)
│       ├── hybrid_retriever.py   # BM25 + dense, fuse bằng Reciprocal Rank Fusion
│       ├── run_domain.py         # chạy 1 domain, chọn method qua --method
│       ├── run_release.py        # chạy nhiều domain + validate + chấm điểm
│       ├── requirements-dense.txt
│       └── README.md             # chi tiết kỹ thuật từng file/contract
├── notebooks/
│   └── reteco_track1a_kaggle.ipynb   # notebook chạy trên Kaggle (có GPU free)
├── runs/
│   └── track1a/
│       └── results_<split>_<method>.json   # summary điểm số (nhẹ, có commit)
└── docs/
    └── (ghi chú nhóm, phân công, ...)
```

`reteco_data/` (dataset gốc), `RETECO/` (clone starter kit), `cache/`
(embedding cache) và các file run chi tiết (`.txt`/`.trec`) **không** nằm
trong repo — xem `.gitignore`. Tất cả tái tạo được bằng các lệnh ở phần
Setup bên dưới.

## Setup

Từ thư mục gốc dự án (không phải trong `src/`):

```bash
git clone https://github.com/<username>/reteco-track1a.git
cd reteco-track1a

# 1. Clone starter kit của ban tổ chức (scorer + format checker chính thức)
git clone https://github.com/DataScienceUIBK/RETECO.git

# 2. Tải dataset (4.5GB đầy đủ — có thể lọc bằng allow_patterns khi test nhanh)
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='DataScience-UIBK/RETECO-SemEval2027',
    repo_type='dataset',
    local_dir='reteco_data',
)
"

# 3. Cài dependency cho retriever
pip install -r src/track1a/requirements-dense.txt   # dense/hybrid
# method bm25 không cần cài gì thêm ngoài chuẩn Python
```

Layout sau setup phải khớp đúng những gì `run_release.py` kỳ vọng:

```text
reteco-track1a/
├── RETECO/starter_kit/     (scorer.py, ir_metrics.py, format_checker.py)
├── reteco_data/track1_tempo/<domain>/
├── src/track1a/
├── runs/
└── cache/embeddings/       (tự tạo khi chạy dense/hybrid)
```

## Cách chạy

```bash
# BM25 baseline, 1 domain, tập train
python src/track1a/run_release.py --split train --track1 iota --method bm25

# Dense retriever (cần model tải lần đầu từ HuggingFace)
python src/track1a/run_release.py --split train --track1 iota --method dense

# Hybrid (BM25 + dense, khuyến nghị dùng)
python src/track1a/run_release.py --split train --track1 iota --method hybrid

# Toàn bộ 13 domain Track 1
python src/track1a/run_release.py --split train --method hybrid

# Chỉ chạy --split dev SAU CÙNG, như bước kiểm tra giữ lại
# (không tinh chỉnh hyperparameter dựa trên dev — luật thi cấm)
python src/track1a/run_release.py --split dev --method hybrid
```

Trên Windows PowerShell dùng cú pháp giống hệt, chỉ đổi `\` cho path nếu
cần. Chạy trên Kaggle: dùng `notebooks/reteco_track1a_kaggle.ipynb` (có GPU
miễn phí, khuyên dùng cho `dense`/`hybrid` vì encode corpus nhanh hơn CPU
nhiều lần).

Kết quả ghi ra `runs/track1a/results_<split>_<method>.json` — mỗi method
một file riêng, không ghi đè nhau, dễ so sánh.

## Kết quả hiện tại

Macro nDCG@10 trên tập **train** (số liệu đang cập nhật dần, xem ngày ghi
kèm — domain chưa test để trống):

| Method | iota | law | ... 11 domain còn lại | Macro (toàn Track 1a) |
| --- | ---: | ---: | --- | --- |
| BM25 (baseline) | 0.0558 | — | chưa chạy | chưa chạy |
| Dense | — | — | chưa chạy | chưa chạy |
| Hybrid | 0.2458 | — | chưa chạy | chưa chạy |

*(cập nhật: 2026-09-17, mới test trên 2/13 domain — số liệu chưa đại diện
cho toàn bộ Track 1a, cần chạy hết trước khi kết luận)*

Baseline tham chiếu từ ban tổ chức (macro toàn bộ tập, official Lucene
BM25): train 0.0879 / dev 0.0967.

## Retriever contract

Mọi retriever (bm25 / dense / hybrid / thứ bạn thêm sau này) phải thoả:

```python
search(query: str, top_k: int) -> list[tuple[doc_id: str, score: float]]
# sắp xếp giảm dần theo score
```

Nhờ vậy có thể thêm phương pháp mới (cross-encoder reranker, query
rewriting...) thành module riêng trong `src/track1a/` mà không cần sửa
`run_domain.py` — chỉ thêm 1 nhánh `if args.method == "..."` trong
`build_retriever()`.

## Quy trình làm việc nhóm

- Mỗi người làm 1 nhánh (`git checkout -b feature/reranker`) khi thử
  phương pháp mới, merge vào `main` qua Pull Request.
- Code sống ở GitHub, sửa ở đâu cũng đồng bộ lại GitHub trước — không sửa
  song song trên Kaggle và local mà không `git pull`/`push`.
- Trên Kaggle, clone thẳng từ GitHub thay vì upload lại code thành Kaggle
  Dataset mỗi lần sửa:
  ```python
  !git clone https://github.com/<username>/reteco-track1a.git
  ```
- Chỉ tinh chỉnh hyperparameter/method dựa trên `--split train`. Chạy
  `--split dev` một lần cuối để kiểm tra, không dùng để chọn cấu hình.

## Bước tiếp theo

- Chạy hybrid trên đủ 13 domain Track 1, cả train lẫn dev, để có con số
  đại diện thật.
- Thử model embedding mạnh hơn (`BAAI/bge-large-en-v1.5`,
  `intfloat/e5-large-v2`) nếu base model đã chứng minh có lợi so với BM25.
- Thêm cross-encoder reranker trên top-100 của hybrid.
- Khai thác `guidance_train.jsonl` (metadata temporal) để thiết kế bước
  query rewriting trước khi retrieval.
