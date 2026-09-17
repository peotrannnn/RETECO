#!/usr/bin/env python3
"""Driver for the RETECO Track 1a system.

For each selected Track 1 domain it runs the retriever, validates the run
file with the organizer's `format_checker.py`, scores it with the
organizer's `scorer.py`, and records the result.

Two properties that matter for long multi-domain runs:

1. The summary is rewritten after EVERY domain, not once at the end, and it
   MERGES into any existing summary for the same (split, pipeline). A
   13-domain run that dies halfway (Kaggle session timeout, OOM) keeps
   everything it had already finished.
2. `--skip-existing` skips domains already recorded, so a killed run is
   resumed by re-issuing the same command.

Examples:
    # BM25 baseline, one domain
    python run_release.py --split train --track1 iota --method bm25

    # full pipeline (weighted hybrid + cross-encoder rerank), all 13 domains,
    # resumable
    python run_release.py --split train --method hybrid --rerank --skip-existing

Tune against --split train. Run --split dev once, at the very end, as a
held-out check -- the competition rules say dev is not for tuning.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA = PROJECT_ROOT / "reteco_data"
DEFAULT_OUT = PROJECT_ROOT / "runs" / "track1a"
DEFAULT_CACHE = PROJECT_ROOT / "cache" / "embeddings"
STARTER_KIT = PROJECT_ROOT / "RETECO" / "starter_kit"

RUN_DOMAIN = HERE / "run_domain.py"
SCORER = STARTER_KIT / "scorer.py"
FORMAT_CHECKER = STARTER_KIT / "format_checker.py"


def utf8_env():
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return env


def method_label(args):
    return args.method + ("_rerank" if args.rerank else "")


def run_retrieval(corpus, queries, out, args, tag):
    cmd = [
        sys.executable, str(RUN_DOMAIN),
        "--corpus", str(corpus),
        "--queries", str(queries),
        "--out", str(out),
        "--top-k", str(args.top_k),
        "--method", args.method,
        "--tag", tag,
        "--k1", str(args.k1),
        "--b", str(args.b),
    ]

    if args.cache_dir:
        cmd += ["--cache-dir", str(args.cache_dir)]
    if args.device:
        cmd += ["--device", args.device]
    if args.no_fp16:
        cmd += ["--no-fp16"]

    if args.method in ("dense", "hybrid"):
        cmd += [
            "--model", args.model,
            "--batch-size", str(args.batch_size),
            "--max-seq-length", str(args.max_seq_length),
            "--query-max-seq-length", str(args.query_max_seq_length),
            "--query-prefix", args.query_prefix,
            "--passage-prefix", args.passage_prefix,
        ]

    if args.method == "hybrid":
        cmd += [
            "--rrf-k", str(args.rrf_k),
            "--pool-k", str(args.pool_k),
            "--sparse-weight", str(args.sparse_weight),
            "--dense-weight", str(args.dense_weight),
        ]

    if args.rerank:
        cmd += [
            "--rerank",
            "--rerank-model", args.rerank_model,
            "--candidate-k", str(args.candidate_k),
            "--rerank-batch-size", str(args.rerank_batch_size),
            "--rerank-max-length", str(args.rerank_max_length),
        ]
        if args.rerank_fp16:
            cmd += ["--rerank-fp16"]

    subprocess.run(cmd, check=True, cwd=str(HERE), env=utf8_env())


def validate_run(run_path, qrels_path, corpus_path):
    cmd = [
        sys.executable, str(FORMAT_CHECKER), str(run_path),
        "--qrels", str(qrels_path),
        "--corpus", str(corpus_path),
        "--doc-key", "id",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False,
                               cwd=str(STARTER_KIT), env=utf8_env())
    if completed.returncode != 0:
        raise RuntimeError(
            "Generated run failed RETECO format validation:\n"
            + completed.stdout + completed.stderr
        )
    return completed.stdout.strip()


def score_run(run_path, qrels_path, steps_path):
    cmd = [
        sys.executable, str(SCORER),
        "--track", "1",
        "--run", str(run_path),
        "--qrels", str(qrels_path),
        "--steps", str(steps_path),
        "--json",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True,
                               cwd=str(STARTER_KIT), env=utf8_env())
    return json.loads(completed.stdout)


def write_summary(summary_path, results):
    """Recompute the macro average over every domain recorded so far and
    persist. Called after each domain so progress is never lost."""
    scored = [e["metrics"]["nDCG@10"] for e in results["per_domain"].values()]
    results["macro_nDCG@10"] = round(sum(scored) / len(scored), 4) if scored else 0.0
    results["num_domains_recorded"] = len(results["per_domain"])

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="RETECO release root")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="output root for Track 1a runs")
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--track1", nargs="*",
                        help="domain names; omit to run all Track 1 domains")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip domains already present in the summary file "
                             "(use to resume an interrupted run)")

    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"],
                        default="bm25")
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--cache-dir", default=str(DEFAULT_CACHE),
        help="cache BM25 indexes and corpus embeddings here; shared across "
             "methods and across train/dev. Pass '' to disable.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-fp16", action="store_true")

    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)

    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--query-max-seq-length", type=int, default=512)
    parser.add_argument(
        "--query-prefix",
        default="Represent this sentence for searching relevant passages: ")
    parser.add_argument("--passage-prefix", default="")

    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--pool-k", type=int, default=200)
    parser.add_argument("--sparse-weight", type=float, default=0.3)
    parser.add_argument("--dense-weight", type=float, default=1.0)

    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument("--rerank-max-length", type=int, default=512)
    parser.add_argument("--rerank-fp16", action="store_true")

    args = parser.parse_args()
    if args.cache_dir == "":
        args.cache_dir = None

    label = method_label(args)
    tag = args.tag or f"track1a_{label}"

    data_root = Path(args.data).resolve()
    output_root = Path(args.out).resolve()
    track1_root = data_root / "track1_tempo"

    if not track1_root.is_dir():
        raise FileNotFoundError(f"Track 1 data directory not found: {track1_root}")
    if not SCORER.is_file():
        raise FileNotFoundError(f"Organizer scorer not found: {SCORER}")
    if not FORMAT_CHECKER.is_file():
        raise FileNotFoundError(f"Organizer format checker not found: {FORMAT_CHECKER}")

    domains = (
        args.track1
        if args.track1 is not None
        else sorted(p.name for p in track1_root.iterdir() if p.is_dir())
    )

    split = args.split
    summary_path = output_root / f"results_{split}_{label}.json"

    results = {
        "track": "1a",
        "split": split,
        "pipeline": label,
        "method": args.method,
        "rerank": bool(args.rerank),
        "retriever": tag,
        "top_k": args.top_k,
        "k1": args.k1,
        "b": args.b,
        "model": args.model if args.method in ("dense", "hybrid") else None,
        "max_seq_length": args.max_seq_length if args.method in ("dense", "hybrid") else None,
        "sparse_weight": args.sparse_weight if args.method == "hybrid" else None,
        "dense_weight": args.dense_weight if args.method == "hybrid" else None,
        "rerank_model": args.rerank_model if args.rerank else None,
        "candidate_k": args.candidate_k if args.rerank else None,
        "per_domain": {},
    }

    # Merge into an existing summary instead of overwriting it, so running
    # extra domains later ADDS to the record rather than destroying it.
    if summary_path.is_file():
        with open(summary_path, "r", encoding="utf-8") as f:
            results["per_domain"] = json.load(f).get("per_domain", {})

    for domain in domains:
        if args.skip_existing and domain in results["per_domain"]:
            print(f"[skip] {domain} already recorded in {summary_path.name}", flush=True)
            continue

        domain_dir = track1_root / domain
        corpus = domain_dir / "documents.jsonl"
        queries = domain_dir / f"examples_{split}.jsonl"
        qrels = domain_dir / f"qrels_{split}.txt"
        steps = domain_dir / f"steps_{split}.jsonl"

        for required in (corpus, queries, qrels, steps):
            if not required.is_file():
                raise FileNotFoundError(f"Missing required file for {domain}: {required}")

        domain_out = output_root / domain
        domain_out.mkdir(parents=True, exist_ok=True)
        run_path = domain_out / f"1a_{split}_{label}.txt"

        print(f"\n=== track1a/{domain} ({split}, {label}) ===", flush=True)

        run_retrieval(corpus=corpus, queries=queries, out=run_path,
                      args=args, tag=tag)
        validation = validate_run(run_path=run_path, qrels_path=qrels,
                                  corpus_path=corpus)
        metrics = score_run(run_path=run_path, qrels_path=qrels, steps_path=steps)

        results["per_domain"][domain] = {"run": str(run_path), "metrics": metrics}
        write_summary(summary_path, results)   # persist after EVERY domain

        print(validation, flush=True)
        print(f"track1a/{domain:<12} nDCG@10 {metrics['nDCG@10']:.4f}  "
              f"({metrics['num_topics']} topics)", flush=True)

    write_summary(summary_path, results)

    print("\n" + "=" * 70)
    print(f"Track 1a [{label}] macro nDCG@10: {results['macro_nDCG@10']:.4f}  "
          f"over {results['num_domains_recorded']} domain(s)")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
