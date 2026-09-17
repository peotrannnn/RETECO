#!/usr/bin/env python3
"""Run the current RETECO Track 1a retriever for one domain.

Input:
- documents.jsonl
- examples_train.jsonl or examples_dev.jsonl

Output:
- TREC run file:
  qid  Q0  docid  rank  score  tag

Methods (--method):
  bm25    (default) pure-Python BM25, identical formulation to the starter
          kit's bm25.py.
  dense   sentence-transformers bi-encoder + faiss/numpy nearest-neighbor
          search. Needs requirements-dense.txt installed.
  hybrid  Reciprocal Rank Fusion of bm25 + dense. Needs requirements-dense.txt.

Examples:
    python src/track1a/run_domain.py \
        --corpus reteco_data/track1_tempo/iota/documents.jsonl \
        --queries reteco_data/track1_tempo/iota/examples_train.jsonl \
        --out runs/track1a/iota/1a_train.txt

    python src/track1a/run_domain.py --method dense \
        --model BAAI/bge-base-en-v1.5 --cache-dir cache/embeddings \
        --corpus reteco_data/track1_tempo/iota/documents.jsonl \
        --queries reteco_data/track1_tempo/iota/examples_train.jsonl \
        --out runs/track1a/iota/1a_train_dense.txt

    python src/track1a/run_domain.py --method hybrid \
        --model BAAI/bge-base-en-v1.5 --cache-dir cache/embeddings \
        --corpus reteco_data/track1_tempo/iota/documents.jsonl \
        --queries reteco_data/track1_tempo/iota/examples_train.jsonl \
        --out runs/track1a/iota/1a_train_hybrid.txt
"""

import argparse
import json
from pathlib import Path

from retriever import Retriever


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(path):
    docs = read_jsonl(path)
    return [doc["id"] for doc in docs], [doc["content"] for doc in docs]


def load_queries(path):
    items = read_jsonl(path)
    # Track 1a query construction follows the starter kit exactly:
    # topic id = item["id"], query text = item["query"].
    return [(item["id"], item["query"]) for item in items]


def write_run(path, rows, tag):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked in rows:
            for rank, (docid, score) in enumerate(ranked, start=1):
                f.write(
                    f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\t{tag}\n"
                )


def build_retriever(args, doc_ids, doc_texts):
    if args.method == "bm25":
        return Retriever(doc_ids, doc_texts, k1=args.k1, b=args.b)

    if args.method == "dense":
        from dense_retriever import DenseRetriever
        return DenseRetriever(
            doc_ids, doc_texts,
            model_name=args.model,
            cache_dir=args.cache_dir,
            device=args.device,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
        )

    if args.method == "hybrid":
        from dense_retriever import DenseRetriever
        from hybrid_retriever import HybridRetriever
        sparse = Retriever(doc_ids, doc_texts, k1=args.k1, b=args.b)
        dense = DenseRetriever(
            doc_ids, doc_texts,
            model_name=args.model,
            cache_dir=args.cache_dir,
            device=args.device,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
        )
        return HybridRetriever(sparse, dense, k=args.rrf_k, pool_k=args.pool_k)

    raise ValueError(f"unknown method: {args.method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=100)

    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"],
                         default="bm25")
    parser.add_argument("--tag", default=None,
                         help="run tag; defaults to track1a_<method>")

    # bm25 params (also used as the sparse half of hybrid)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)

    # dense / hybrid params
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--cache-dir", default=None,
                         help="cache corpus embeddings here across runs "
                              "(train and dev share the same corpus, so "
                              "this makes dev free after train)")
    parser.add_argument("--device", default=None,
                         help="cpu | cuda | cuda:0 ... (auto-detected if omitted)")
    parser.add_argument(
        "--query-prefix",
        default="Represent this sentence for searching relevant passages: ",
        help="prepended to the query text before encoding (bge-style "
             "instruction prefix); pass '' to disable")
    parser.add_argument("--passage-prefix", default="")
    parser.add_argument("--rrf-k", type=int, default=60,
                         help="RRF constant (hybrid only)")
    parser.add_argument("--pool-k", type=int, default=200,
                         help="candidates pulled from each retriever before "
                              "fusion (hybrid only)")

    args = parser.parse_args()
    tag = args.tag or f"track1a_{args.method}"

    doc_ids, doc_texts = load_corpus(args.corpus)
    queries = load_queries(args.queries)

    retriever = build_retriever(args, doc_ids, doc_texts)

    rows = [
        (qid, retriever.search(query, top_k=args.top_k))
        for qid, query in queries
    ]

    write_run(args.out, rows, tag)

    print(
        f"wrote {args.out}: "
        f"{len(rows)} topics over {len(doc_ids)} docs "
        f"(Track 1a, method={args.method}, top_k={args.top_k})"
    )


if __name__ == "__main__":
    main()
