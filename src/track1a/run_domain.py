#!/usr/bin/env python3
"""Run the RETECO Track 1a retriever for one domain.

Input:
- documents.jsonl
- examples_train.jsonl or examples_dev.jsonl

Output:
- TREC run file:  qid  Q0  docid  rank  score  tag

First stage (--method):
  bm25    BM25 over a scipy inverted index (same formula as the starter kit)
  dense   sentence-transformers bi-encoder + faiss/numpy search
  hybrid  weighted Reciprocal Rank Fusion of bm25 + dense

Second stage (--rerank):
  adds a cross-encoder reranker over the first stage's top --candidate-k
  documents. Combines with any --method.

Examples:
    python run_domain.py \
        --corpus  reteco_data/track1_tempo/iota/documents.jsonl \
        --queries reteco_data/track1_tempo/iota/examples_train.jsonl \
        --out     runs/track1a/iota/1a_train.txt

    python run_domain.py --method hybrid --rerank \
        --cache-dir cache/embeddings \
        --corpus  reteco_data/track1_tempo/iota/documents.jsonl \
        --queries reteco_data/track1_tempo/iota/examples_train.jsonl \
        --out     runs/track1a/iota/1a_train.txt
"""

import argparse
import json
from pathlib import Path

from retriever import Retriever


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(path):
    ids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                ids.append(d["id"])
                texts.append(d["content"])
    return ids, texts


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
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\t{tag}\n")


def method_label(args):
    """Name that identifies the full pipeline, used for run tags and for the
    summary filename, so e.g. hybrid and hybrid+rerank never overwrite each
    other's results."""
    return args.method + ("_rerank" if args.rerank else "")


def build_retriever(args, doc_ids, doc_texts):
    def make_sparse():
        return Retriever(doc_ids, doc_texts, k1=args.k1, b=args.b,
                         cache_dir=args.cache_dir)

    def make_dense():
        from dense_retriever import DenseRetriever
        return DenseRetriever(
            doc_ids, doc_texts,
            model_name=args.model,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            device=args.device,
            fp16=not args.no_fp16,
            max_seq_length=args.max_seq_length,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
        )

    if args.method == "bm25":
        base = make_sparse()
    elif args.method == "dense":
        base = make_dense()
    elif args.method == "hybrid":
        from hybrid_retriever import HybridRetriever
        base = HybridRetriever(
            make_sparse(), make_dense(),
            k=args.rrf_k, pool_k=args.pool_k,
            sparse_weight=args.sparse_weight,
            dense_weight=args.dense_weight,
        )
    else:
        raise ValueError(f"unknown method: {args.method}")

    if args.rerank:
        from reranker import RerankRetriever
        base = RerankRetriever(
            base, doc_ids, doc_texts,
            model_name=args.rerank_model,
            candidate_k=args.candidate_k,
            batch_size=args.rerank_batch_size,
            device=args.device,
            max_length=args.rerank_max_length,
            # Opt-in, and separate from the bi-encoder's fp16: the reranker
            # architecture overflows in half precision where the bi-encoder
            # does not. See reranker.py.
            fp16=args.rerank_fp16,
        )

    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=100)

    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"],
                        default="bm25")
    parser.add_argument("--tag", default=None,
                        help="run tag; defaults to track1a_<method label>")
    parser.add_argument("--cache-dir", default=None,
                        help="cache BM25 indexes and corpus embeddings here; "
                             "reused across methods and across train/dev, "
                             "since the corpus is the same for both")
    parser.add_argument("--device", default=None,
                        help="cpu | cuda | cuda:0 ... (auto-detected if omitted)")
    parser.add_argument("--no-fp16", action="store_true",
                        help="disable fp16 for the encoders (slower on GPU)")

    # bm25 params (also the sparse half of hybrid)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)

    # dense params
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=256,
                        help="encoder truncation length; 256 roughly doubles "
                             "throughput vs the model's 512 limit")
    parser.add_argument(
        "--query-prefix",
        default="Represent this sentence for searching relevant passages: ",
        help="prepended to the query before encoding (bge-style instruction "
             "prefix); pass '' to disable")
    parser.add_argument("--passage-prefix", default="")

    # hybrid params
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--pool-k", type=int, default=200,
                        help="candidates pulled from each retriever before fusion")
    parser.add_argument("--sparse-weight", type=float, default=0.3,
                        help="RRF vote weight for BM25")
    parser.add_argument("--dense-weight", type=float, default=1.0,
                        help="RRF vote weight for the bi-encoder")

    # reranking
    parser.add_argument("--rerank", action="store_true",
                        help="add a cross-encoder reranker over the first stage")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--candidate-k", type=int, default=100,
                        help="how many first-stage candidates to rerank")
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument("--rerank-max-length", type=int, default=512)
    parser.add_argument("--rerank-fp16", action="store_true",
                        help="half precision for the cross-encoder; off by "
                             "default because this architecture can overflow "
                             "to inf. Guarded at runtime regardless.")

    args = parser.parse_args()
    tag = args.tag or f"track1a_{method_label(args)}"

    doc_ids, doc_texts = load_corpus(args.corpus)
    queries = load_queries(args.queries)

    retriever = build_retriever(args, doc_ids, doc_texts)

    rows = [
        (qid, retriever.search(query, top_k=args.top_k))
        for qid, query in queries
    ]

    write_run(args.out, rows, tag)

    print(f"wrote {args.out}: {len(rows)} topics over {len(doc_ids)} docs "
          f"(Track 1a, {method_label(args)}, top_k={args.top_k})")


if __name__ == "__main__":
    main()
