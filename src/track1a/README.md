# Custom RETECO Track 1a system

This directory is the user's Track 1a system. It mirrors the organizer starter
kit's separation of responsibilities while removing Track 1b and Track 2.

## Files

- `retriever.py`
  - BM25 retriever (pure-Python, same formulation as the starter kit).

- `dense_retriever.py`
  - Bi-encoder (sentence-transformers) retriever with an on-disk corpus
    embedding cache. Needs `requirements-dense.txt`.

- `hybrid_retriever.py`
  - Reciprocal Rank Fusion of `retriever.py` (sparse) + `dense_retriever.py`
    (dense). Needs `requirements-dense.txt`.

- `run_domain.py`
  - Runs one Track 1a domain with a chosen `--method` (`bm25` / `dense` /
    `hybrid`).
  - Reads `documents.jsonl` and `examples_<split>.jsonl`.
  - Writes a six-column TREC run.

- `run_release.py`
  - Driver over one or more Track 1 domains.
  - Calls `run_domain.py`, validates with `format_checker.py`, scores with
    `scorer.py`.
  - Writes `results_<split>_<method>.json` (one file per method, so bm25 /
    dense / hybrid stay comparable side by side).

- `__init__.py`
  - Marks the directory as a Python package.

## Expected project layout

```text
D:\RETECO-project\
├── RETECO\
│   └── starter_kit\
│       ├── scorer.py
│       ├── ir_metrics.py
│       └── format_checker.py
├── cache\
│   └── embeddings\        # dense corpus embeddings, auto-created
├── reteco_data\
│   └── track1_tempo\
├── runs\
└── src\
    └── track1a\
        ├── __init__.py
        ├── retriever.py
        ├── dense_retriever.py
        ├── hybrid_retriever.py
        ├── run_domain.py
        ├── run_release.py
        ├── requirements-dense.txt
        └── README.md
```

## Setup

The `bm25` method needs nothing beyond the standard library (same as before).
For `dense` / `hybrid`, from `D:\RETECO-project`:

```powershell
pip install -r src\track1a\requirements-dense.txt
```

If you have an NVIDIA GPU, install the matching CUDA build of `torch` first
(see pytorch.org) — encoding the largest domains (history, hsm, politics,
monero, genealogy, travel) on CPU is slow.

## Run IOTA train

From `D:\RETECO-project`:

```powershell
# BM25 baseline (unchanged behavior)
python src\track1a\run_release.py --split train --track1 iota

# Dense retriever
python src\track1a\run_release.py --split train --track1 iota --method dense

# Hybrid (BM25 + dense, RRF-fused)
python src\track1a\run_release.py --split train --track1 iota --method hybrid
```

## Run IOTA dev

```powershell
python src\track1a\run_release.py --split dev --track1 iota --method hybrid
```

Corpus embeddings are cached under `cache\embeddings\`, keyed by model name +
corpus fingerprint. Since `documents.jsonl` is identical for train and dev,
running dev right after train for the same domain re-uses the cached
embeddings instead of re-encoding the corpus.

## Run selected domains / all domains

```powershell
python src\track1a\run_release.py --split train --track1 iota law --method hybrid
python src\track1a\run_release.py --split train --method hybrid   # all 13 domains
```

## Comparing methods

Each method writes its own summary file, so you can diff macro nDCG@10
directly:

```text
runs\track1a\results_train_bm25.json
runs\track1a\results_train_dense.json
runs\track1a\results_train_hybrid.json
```

Tune everything against `--split train`. Only run `--split dev` at the end,
as a held-out check — the competition rules say dev is not meant to be tuned
against.

## Design contract

The rest of the system should stay stable while retrieval methods change.

Input contract:

```text
documents.jsonl
examples_<split>.jsonl
```

Output contract:

```text
qid  Q0  docid  rank  score  tag
```

Evaluation contract:

```text
organizer scorer.py -> nDCG@10
```

Retriever contract (what `run_domain.py` expects from any retriever object):

```text
search(query, top_k) -> [(doc_id, score), ...]   # sorted by score descending
```

`retriever.py`, `dense_retriever.py` and `hybrid_retriever.py` all satisfy
this, so a new retrieval method can be dropped in as another module without
touching `run_domain.py`'s I/O logic.

## What's next (see conversation for the full roadmap)

- Try a stronger / larger embedding model (`BAAI/bge-large-en-v1.5`,
  `intfloat/e5-large-v2`) once the base model's gain over BM25 alone is
  confirmed on train.
- Add a cross-encoder reranker over the hybrid method's top-100 candidates.
- Mine `guidance_train.jsonl` for temporal query patterns to build a
  query-rewriting step before retrieval (extract/normalize dates or
  before/after/latest phrasing).
