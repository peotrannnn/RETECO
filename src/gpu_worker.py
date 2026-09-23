"""Run the GPU part of the pipeline on a machine that has a GPU.

This file is deliberately standalone. It imports nothing from this project,
so the only things that have to travel to the GPU machine are this script and
the job folder written by `pipeline.export_gpu_job`.

    # on the CPU machine
    >>> P.export_gpu_job(digest, "gpu_job", "job_rerank_01",
    ...                  task="rerank", depth=1000)

    # copy gpu_job/ across, then on the GPU machine
    $ pip install torch sentence-transformers
    $ python gpu_worker.py gpu_job --data /path/to/track1_tempo

    # copy gpu_job/scores.jsonl back, then on the CPU machine
    >>> P.import_gpu_scores("gpu_job/scores.jsonl")

Two tasks:

    rerank  score every (query, candidate) pair with a cross-encoder and
            return a new score per pair. Cost is proportional to the number
            of pairs, so `depth` at export time decides the bill.

    dense   embed every document once, embed every query, and return the
            nearest documents. Cost is proportional to the CORPUS, not to the
            queries, and it is the expensive one.

Both write `scores.jsonl` into the job folder, in the format
`pipeline.import_gpu_scores` expects:

    {"job_id": ..., "d": <domain>, "q": <query id>,
     "i": [doc ids], "s": [scores]}

Progress is written to stdout and to `progress.txt`, and finished domains are
written out as they complete, so a session that gets cut off resumes from
where it stopped rather than starting over.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# ----------------------------------------------------------------- defaults --
# Small, fast and widely available. Both fit in the memory of a free-tier GPU.
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Documents in this corpus have a median of 241 tokens and a long tail into
# the hundreds of thousands. Truncating is not optional.
MAX_DOC_CHARS = 2000
MAX_QUERY_CHARS = 1200


# ------------------------------------------------------------------- input --
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_job(folder):
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    queries = {}
    for row in read_jsonl(folder / "queries.jsonl"):
        queries[row["q"]] = row["t"]
    return job, queries


def load_documents(folder, job, data_dir, wanted_by_domain):
    """Document text, either from the job folder or from the corpus itself."""
    texts = {}
    if job.get("with_text") and (folder / "documents.jsonl").exists():
        for row in read_jsonl(folder / "documents.jsonl"):
            texts[row["i"]] = row["t"]
        return texts

    if data_dir is None:
        raise SystemExit(
            "This job does not carry document text, so --data must point at "
            "the track1_tempo folder on this machine.\n"
            "Either pass --data, or re-export the job with with_text=True.")

    for domain, wanted in wanted_by_domain.items():
        path = Path(data_dir) / domain / "documents.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} not found")
        for record in read_jsonl(path):
            if wanted is None or record["id"] in wanted:
                texts[record["id"]] = record["content"] or ""
    return texts


def strip_markup(text):
    """The queries carry HTML; the documents do not. Same shaping as the
    sparse stage, so the two stages see comparable text."""
    import re
    text = re.sub(r"&(?:quot|amp|lt|gt|nbsp|apos|#\d+);", " ", text or "")
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


# ------------------------------------------------------------------ output --
class Sink:
    """Append-only writer that remembers which domains are already done."""

    def __init__(self, path, job_id):
        self.path = Path(path)
        self.job_id = job_id
        self.done = set()
        if self.path.exists():
            for row in read_jsonl(self.path):
                self.done.add(row["d"])
            print(f"resuming; already finished: {sorted(self.done) or 'nothing'}")
        self.handle = open(self.path, "a", encoding="utf-8")

    def write(self, domain, qid, ids, scores):
        self.handle.write(json.dumps({
            "job_id": self.job_id, "d": domain, "q": qid,
            "i": list(ids), "s": [round(float(s), 6) for s in scores]}) + "\n")

    def flush(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())


def note(folder, message):
    print(message, flush=True)
    with open(folder / "progress.txt", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')}  {message}\n")


# ------------------------------------------------------------------ rerank --
def run_rerank(folder, job, data_dir, model_name, batch_size, device):
    from sentence_transformers import CrossEncoder

    candidates = list(read_jsonl(folder / "candidates.jsonl"))
    _, queries = load_job(folder)

    wanted = {}
    for row in candidates:
        wanted.setdefault(row["d"], set()).update(row["i"])

    sink = Sink(folder / "scores.jsonl", job["job_id"])
    todo = [row for row in candidates if row["d"] not in sink.done]
    if not todo:
        note(folder, "every domain already done")
        return

    note(folder, f"loading {model_name} on {device}")
    model = CrossEncoder(model_name, max_length=512, device=device)

    texts = load_documents(folder, job, data_dir,
                           {d: wanted[d] for d in wanted})
    note(folder, f"{len(texts):,} document texts in memory")

    by_domain = {}
    for row in todo:
        by_domain.setdefault(row["d"], []).append(row)

    total_pairs = sum(len(row["i"]) for row in todo)
    done_pairs, started = 0, time.time()

    for domain, rows in by_domain.items():
        t0 = time.time()
        for row in rows:
            query = strip_markup(queries[row["q"]])[:MAX_QUERY_CHARS]
            ids = [i for i in row["i"] if i in texts]
            if not ids:
                continue
            pairs = [(query, texts[i][:MAX_DOC_CHARS]) for i in ids]
            scores = model.predict(pairs, batch_size=batch_size,
                                   show_progress_bar=False)
            sink.write(domain, row["q"], ids, scores)
            done_pairs += len(ids)
        sink.flush()
        rate = done_pairs / max(time.time() - started, 1e-9)
        left = (total_pairs - done_pairs) / max(rate, 1e-9)
        note(folder, f"{domain:<12} {len(rows):>5} queries  "
                     f"{time.time() - t0:>6.0f}s  "
                     f"{done_pairs:,}/{total_pairs:,} pairs  "
                     f"{rate:,.0f} pairs/s  ~{left / 60:.0f} min left")
    note(folder, f"rerank finished in {(time.time() - started) / 60:.1f} min")


# ------------------------------------------------------------------- dense --
def run_dense(folder, job, data_dir, model_name, batch_size, device, depth):
    """Embed the corpus once per domain, then search it with the queries.

    Held one domain at a time on purpose. The whole corpus at 384 dimensions
    in float32 is about 2.5 GB, which fits, but one domain at a time keeps the
    peak low enough for a free-tier GPU and makes the job resumable.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    _, queries = load_job(folder)
    queries_by_domain = {}
    for row in read_jsonl(folder / "queries.jsonl"):
        queries_by_domain.setdefault(row["d"], []).append((row["q"], row["t"]))

    sink = Sink(folder / "scores.jsonl", job["job_id"])
    note(folder, f"loading {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    started = time.time()
    for domain in job["domains"]:
        if domain in sink.done:
            continue
        t0 = time.time()
        path = Path(data_dir) / domain / "documents.jsonl"
        ids, texts = [], []
        for record in read_jsonl(path):
            ids.append(record["id"])
            texts.append((record["content"] or "")[:MAX_DOC_CHARS])

        doc_vectors = model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True)
        del texts

        pairs = queries_by_domain.get(domain, [])
        query_vectors = model.encode(
            [strip_markup(t)[:MAX_QUERY_CHARS] for _, t in pairs],
            batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False)

        # Vectors are unit length, so the dot product IS cosine similarity.
        for (qid, _), vector in zip(pairs, query_vectors):
            similarity = doc_vectors @ vector
            k = min(depth, similarity.shape[0])
            top = np.argpartition(-similarity, k - 1)[:k]
            top = top[np.argsort(-similarity[top], kind="stable")]
            sink.write(domain, qid, [ids[i] for i in top],
                       [similarity[i] for i in top])
        sink.flush()
        del doc_vectors, ids
        note(folder, f"{domain:<12} {len(pairs):>5} queries  "
                     f"{time.time() - t0:>6.0f}s")
    note(folder, f"dense finished in {(time.time() - started) / 60:.1f} min")


# -------------------------------------------------------------------- main --
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder", help="the job folder written by export_gpu_job")
    parser.add_argument("--data", default=None,
                        help="path to track1_tempo on this machine; needed "
                             "unless the job carries document text")
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None,
                        help="cuda, cpu, or mps; auto-detected when omitted")
    parser.add_argument("--depth", type=int, default=1000,
                        help="how many documents a dense run returns per query")
    args = parser.parse_args()

    folder = Path(args.folder)
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            device = "cpu"
    if device == "cpu":
        print("WARNING: running on CPU. This is the step the GPU is for; "
              "expect it to take many hours.")

    task = job.get("task", "rerank")
    note(folder, f"job {job['job_id']}  task {task}  "
                 f"{job.get('pairs', 0):,} pairs")

    if task == "rerank":
        run_rerank(folder, job, args.data,
                   args.model or job.get("model") or DEFAULT_RERANK_MODEL,
                   args.batch_size, device)
    elif task == "dense":
        run_dense(folder, job, args.data,
                  args.model or job.get("model") or DEFAULT_DENSE_MODEL,
                  args.batch_size, device, args.depth)
    else:
        raise SystemExit(f"unknown task {task!r}")

    print(f"\nDone. Copy {folder / 'scores.jsonl'} back and run:")
    print(f"    P.import_gpu_scores('scores.jsonl', '{job['job_id']}')")


if __name__ == "__main__":
    main()
