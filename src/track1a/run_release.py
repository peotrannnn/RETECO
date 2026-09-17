#!/usr/bin/env python3
"""Driver for the custom RETECO Track 1a system.

This mirrors the starter kit's `run_release.py`, but is intentionally limited
to Track 1a. It:
1. runs the current retriever for each selected Track 1 domain,
2. writes one TREC run per domain,
3. scores it with the organizer's official pure-Python scorer,
4. writes a JSON summary (one file per method, so bm25/dense/hybrid runs
   don't clobber each other and stay easy to compare).

Examples:
    # BM25 baseline, IOTA train only
    python src/track1a/run_release.py --split train --track1 iota

    # Dense retriever, all Track 1 domains, train
    python src/track1a/run_release.py --split train --method dense \
        --model BAAI/bge-base-en-v1.5

    # Hybrid (BM25 + dense via RRF), one domain, dev check
    python src/track1a/run_release.py --split dev --track1 iota --method hybrid

Recommended workflow: tune everything against --split train. Only run
--split dev once, at the end, as a held-out sanity check -- per the
competition rules, dev is not meant to be tuned against.
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


def run_retrieval(corpus, queries, out, args, tag):
    cmd = [
        sys.executable,
        str(RUN_DOMAIN),
        "--corpus", str(corpus),
        "--queries", str(queries),
        "--out", str(out),
        "--top-k", str(args.top_k),
        "--method", args.method,
        "--tag", tag,
        "--k1", str(args.k1),
        "--b", str(args.b),
    ]

    if args.method in ("dense", "hybrid"):
        cmd += ["--model", args.model]
        if args.cache_dir:
            cmd += ["--cache-dir", str(args.cache_dir)]
        if args.device:
            cmd += ["--device", args.device]
        cmd += ["--query-prefix", args.query_prefix,
                "--passage-prefix", args.passage_prefix]

    if args.method == "hybrid":
        cmd += ["--rrf-k", str(args.rrf_k), "--pool-k", str(args.pool_k)]

    subprocess.run(
        cmd,
        check=True,
        cwd=str(HERE),
        env=utf8_env(),
    )


def validate_run(run_path, qrels_path, corpus_path):
    cmd = [
        sys.executable,
        str(FORMAT_CHECKER),
        str(run_path),
        "--qrels", str(qrels_path),
        "--corpus", str(corpus_path),
        "--doc-key", "id",
    ]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(STARTER_KIT),
        env=utf8_env(),
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Generated run failed RETECO format validation:\n"
            + completed.stdout
            + completed.stderr
        )

    return completed.stdout.strip()


def score_run(run_path, qrels_path, steps_path):
    cmd = [
        sys.executable,
        str(SCORER),
        "--track", "1",
        "--run", str(run_path),
        "--qrels", str(qrels_path),
        "--steps", str(steps_path),
        "--json",
    ]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=str(STARTER_KIT),
        env=utf8_env(),
    )

    return json.loads(completed.stdout)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default=str(DEFAULT_DATA),
                         help="RETECO release root")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                         help="output root for Track 1a runs")
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--track1", nargs="*",
                         help="domain names; omit this option to run all Track 1 domains")
    parser.add_argument("--top-k", type=int, default=100)

    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"],
                         default="bm25")
    parser.add_argument("--tag", default=None)

    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)

    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument(
        "--cache-dir", default=str(DEFAULT_CACHE),
        help="cache corpus embeddings here; shared across train/dev since "
             "the corpus is identical for both splits. Pass '' to disable.")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--query-prefix",
        default="Represent this sentence for searching relevant passages: ")
    parser.add_argument("--passage-prefix", default="")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--pool-k", type=int, default=200)

    args = parser.parse_args()
    if args.cache_dir == "":
        args.cache_dir = None

    tag = args.tag or f"track1a_{args.method}"

    data_root = Path(args.data).resolve()
    output_root = Path(args.out).resolve()
    track1_root = data_root / "track1_tempo"

    if not track1_root.is_dir():
        raise FileNotFoundError(
            f"Track 1 data directory not found: {track1_root}"
        )

    if not SCORER.is_file():
        raise FileNotFoundError(
            f"Organizer scorer not found: {SCORER}"
        )

    if not FORMAT_CHECKER.is_file():
        raise FileNotFoundError(
            f"Organizer format checker not found: {FORMAT_CHECKER}"
        )

    domains = (
        args.track1
        if args.track1 is not None
        else sorted(
            path.name
            for path in track1_root.iterdir()
            if path.is_dir()
        )
    )

    split = args.split
    results = {
        "track": "1a",
        "split": split,
        "method": args.method,
        "retriever": tag,
        "top_k": args.top_k,
        "k1": args.k1,
        "b": args.b,
        "model": args.model if args.method in ("dense", "hybrid") else None,
        "per_domain": {},
    }

    for domain in domains:
        domain_dir = track1_root / domain

        corpus = domain_dir / "documents.jsonl"
        queries = domain_dir / f"examples_{split}.jsonl"
        qrels = domain_dir / f"qrels_{split}.txt"
        steps = domain_dir / f"steps_{split}.jsonl"

        for required in (corpus, queries, qrels, steps):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Missing required file for {domain}: {required}"
                )

        domain_out = output_root / domain
        domain_out.mkdir(parents=True, exist_ok=True)

        run_path = domain_out / f"1a_{split}.txt"

        print(f"\n=== track1a/{domain} ({split}, method={args.method}) ===",
              flush=True)

        run_retrieval(
            corpus=corpus,
            queries=queries,
            out=run_path,
            args=args,
            tag=tag,
        )

        validation = validate_run(
            run_path=run_path,
            qrels_path=qrels,
            corpus_path=corpus,
        )

        metrics = score_run(
            run_path=run_path,
            qrels_path=qrels,
            steps_path=steps,
        )

        results["per_domain"][domain] = {
            "run": str(run_path),
            "metrics": metrics,
        }

        print(validation, flush=True)
        print(
            f"track1a/{domain:<10} "
            f"nDCG@10 {metrics['nDCG@10']:.4f}",
            flush=True,
        )

    # Macro-average exactly at the domain level, matching the official
    # leaderboard aggregation.
    scored = [
        entry["metrics"]["nDCG@10"]
        for entry in results["per_domain"].values()
    ]
    results["macro_nDCG@10"] = (
        round(sum(scored) / len(scored), 4)
        if scored
        else 0.0
    )

    output_root.mkdir(parents=True, exist_ok=True)
    # One summary file per method, so bm25 / dense / hybrid results can be
    # compared side by side instead of overwriting each other.
    summary_path = output_root / f"results_{split}_{args.method}.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"Track 1a [{args.method}] macro nDCG@10: {results['macro_nDCG@10']:.4f}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
