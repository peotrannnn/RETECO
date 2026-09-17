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
import re
from pathlib import Path

from retriever import Retriever, TOKENIZERS, DEFAULT_TOKENIZER


# --------------------------------------------------------------------------
# Query forms
#
# Queries in this dataset are whole Stack Exchange posts: a short title
# followed by a much longer body. Measured on the release, the body runs
# 4.8-18.0x the length of the title, and every query carries HTML markup that
# no document in the corpus contains.
#
# Measured over all 13 domains (1,211 train queries), macro nDCG@10 / R@100:
#
#     raw              0.0719 / 0.1995   <- the starter kit's construction
#     stripped         0.0905 / 0.2402
#     title-weighted   0.1139 / 0.2876
#     title            0.1316 / 0.3382   <- best recall, on 10 of 13 domains
#
# Two different winners, and which one is right depends on what comes next:
# `title` maximises RECALL, which is the hard cap on any reranker, while
# `title-weighted` maximises nDCG@10 on its own. Combined with the tokenizer
# (see below) the ordering flips again, which is why the combination was
# measured rather than assumed.
#
# Retrieving with the title alone was measured to raise recall@100 by a factor
# alone changed little -- the body, not the markup, is what dilutes the
# information need.
#
# The chosen form is applied ONCE, right after loading, so every stage
# downstream (lexical, dense, fusion, reranking) sees the same query text.
# --------------------------------------------------------------------------
_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY = re.compile(r"&(quot|amp|lt|gt|nbsp|#\d+);")


def strip_markup(text):
    text = _HTML_TAG.sub(" ", text)
    text = _HTML_ENTITY.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_of(query):
    """The title is everything before the first <p>; the rest is the body."""
    m = re.search(r"<p>", query)
    return query[:m.start()] if m else query


QUERY_FORMS = {
    # the starter kit's construction: the post exactly as published
    "raw":            lambda q: q,
    # same text, markup removed
    "stripped":       lambda q: strip_markup(q),
    # the condensed information need, nothing else
    "title":          lambda q: strip_markup(title_of(q)),
    # title up-weighted but the body retained as context
    "title-weighted": lambda q: strip_markup((title_of(q) + " ") * 3 + q),
}


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


def load_queries(path, form="raw"):
    """Topic id is item["id"]; the query text is item["query"], reshaped by
    the chosen query form (see QUERY_FORMS). The topic id is never altered --
    it must match the qrels exactly."""
    transform = QUERY_FORMS[form]
    items = read_jsonl(path)
    return [(item["id"], transform(item["query"])) for item in items]


def write_run(path, rows, tag):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked in rows:
            for rank, (docid, score) in enumerate(ranked, start=1):
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\t{tag}\n")


def method_label(args):
    """Name identifying the full pipeline: first stage, tokenizer, query form
    and whether reranking is on. Used for run tags and summary filenames, so
    no two configurations overwrite each other's results.

    The tokenizer is part of the label because it changes the index, not just
    the ranking -- two runs that differ only in tokenizer are different
    systems and must not share a results file."""
    parts = [args.method]
    if args.tokenizer != "baseline":
        parts.append(args.tokenizer.replace("+", "-"))
    if args.query_form != "raw":
        parts.append(args.query_form)
    if args.rerank:
        parts.append("rerank")
    return "_".join(parts)


def build_retriever(args, doc_ids, doc_texts):
    def make_sparse():
        return Retriever(doc_ids, doc_texts, k1=args.k1, b=args.b,
                         tokenizer=args.tokenizer, cache_dir=args.cache_dir)

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
            query_max_seq_length=args.query_max_seq_length,
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
    parser.add_argument("--tokenizer", choices=sorted(TOKENIZERS),
                        default=DEFAULT_TOKENIZER,
                        help="lexical tokenizer preset (see retriever.py). "
                             "'aggressive' measured macro nDCG@10 0.1154 vs "
                             "0.0719 for 'baseline', better on 13/13 domains. "
                             "Note that 'stem' and 'stop+stem' measured WORSE "
                             "than baseline.")
    parser.add_argument("--query-form", choices=list(QUERY_FORMS),
                        default="title-weighted",
                        help="how the query text is reshaped before retrieval. "
                             "'title-weighted' measured the best macro nDCG@10 "
                             "and 'title' the best recall@100 -- use 'title' "
                             "when a reranker follows, since recall is the cap "
                             "on what reranking can reach.")
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
                        help="DOCUMENT truncation length; the dominant cost "
                             "knob, since the corpus is ~99.9% of the encoding")
    parser.add_argument("--query-max-seq-length", type=int, default=512,
                        help="QUERY truncation length, separate from documents: "
                             "queries are cheap to encode and cutting one can "
                             "remove the temporal condition being scored")
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

    # Recall, not this stage's own nDCG, is what bounds a reranker. Say so once
    # rather than silently letting a lower-recall shortlist cap the second
    # stage -- it is a cheap flag to change and an expensive one to notice.
    if args.rerank and args.query_form != "title":
        print(f"  [note] --query-form title measured the highest recall@100 "
              f"(0.3382 vs {'0.2876' if args.query_form == 'title-weighted' else 'less'} "
              f"for {args.query_form}); recall is the ceiling on reranking.",
              flush=True)

    doc_ids, doc_texts = load_corpus(args.corpus)
    queries = load_queries(args.queries, form=args.query_form)

    retriever = build_retriever(args, doc_ids, doc_texts)

    # Release the corpus text once the retrievers hold what they need. Only the
    # reranker looks documents up again; for every other pipeline this frees
    # well over a gigabyte on the largest domains, which is the difference
    # between finishing and being killed by the OOM reaper.
    if not args.rerank:
        doc_texts = None

    rows = [
        (qid, retriever.search(query, top_k=args.top_k))
        for qid, query in queries
    ]

    write_run(args.out, rows, tag)

    print(f"wrote {args.out}: {len(rows)} topics over {len(doc_ids)} docs "
          f"(Track 1a, {method_label(args)}, top_k={args.top_k})")


if __name__ == "__main__":
    main()
