"""Compose, run, cache and compare retrieval systems.

A *system* is a dictionary, usually loaded from a JSON file. It names an
ordered list of stages; each stage names an implementation that is registered
under a `kind`:

    {
      "name": "bm25_tuned",
      "retrieve": {"kind": "sparse", "query_form": "title_weighted",
                   "tokenizer": "nohtml_stop", "model": "bm25",
                   "k1": 2.0, "b": 0.9, "depth": 1000},
      "dedup":  null,
      "rerank": null,
      "fuse":   null
    }

Why this file exists
--------------------
Notebooks 07a to 07d multiply out: two ways of de-duplicating, three rerankers,
two depths, two ways of fusing. That is two dozen systems. Two things then go
wrong without a framework:

  1. Changing the reranker would rebuild the index, which takes 14 minutes per
     pass over the 13 domains.
  2. Two people cannot compare results unless both write the same files.

Both are solved by keying every stage on a hash of its own configuration plus
everything upstream of it. Change the reranker and the retrieval hash does not
move, so its cached output is reused.

Files written
-------------
    results/runs/<hash>.jsonl     the ranked lists that stage produced
    results/scores/<hash>.json    scores, including the per-query numbers
    results/index.json            hash -> name, system, when it was run

Nothing here recomputes a metric. Scoring goes through `reteco.evaluate`, the
same function checked against the organisers' reference implementation in
notebook 02.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import reteco as R

__all__ = [
    "STAGES", "register", "load_system", "system_hash",
    "run", "score", "table", "compare", "submit", "listing",
    "configure", "split_report",
]

# Stages always run in this order. A stage whose configuration is null is the
# identity, and is skipped entirely -- including in the hash, so adding
# "rerank": null to a system does not invalidate its cache.
STAGES = ("retrieve", "dedup", "rerank", "fuse")

_REGISTRY: dict[str, dict] = {stage: {} for stage in STAGES}

# Set by configure(); every path below hangs off these.
_DATA: Path | None = None
_RESULTS: Path | None = None
_SPLIT: dict | None = None


def configure(data_dir, results_dir="results"):
    """Point the pipeline at the dataset and the output folder."""
    global _DATA, _RESULTS, _SPLIT
    _DATA = Path(data_dir)
    _RESULTS = Path(results_dir)
    for sub in ("runs", "scores"):
        (_RESULTS / sub).mkdir(parents=True, exist_ok=True)

    split_file = _RESULTS / "split.json"
    if not split_file.exists():
        raise FileNotFoundError(
            f"{split_file} is missing. It is created by notebook 03b and must "
            f"never be regenerated once experiments have been run against it.")
    _SPLIT = json.loads(split_file.read_text(encoding="utf-8"))
    return _DATA


def _require():
    if _DATA is None:
        raise RuntimeError("call pipeline.configure(DATA) first")


def domains():
    _require()
    return R.list_domains(_DATA)


def register(stage, kind):
    """Register an implementation for one stage.

    The second person on the project adds stages by registering new kinds from
    their own file. Nothing in this module needs editing for that.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")

    def decorate(function):
        _REGISTRY[stage][kind] = function
        return function

    return decorate


# ------------------------------------------------------------- the systems --
def load_system(path):
    system = json.loads(Path(path).read_text(encoding="utf-8"))
    system.setdefault("name", Path(path).stem)
    for stage in STAGES:
        system.setdefault(stage, None)
    unknown = set(system) - set(STAGES) - {"name", "note"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    return system


def system_hash(system, upto=STAGES[-1], split="train"):
    """Hash of everything from the first stage up to and including `upto`.

    Stages configured as null are left out, so they cannot change the hash.
    The split is part of the key because the same system run on `train` and on
    `dev` produces different lists.
    """
    cut = STAGES.index(upto)
    active = [[stage, system[stage]] for stage in STAGES[:cut + 1]
              if system.get(stage)]
    blob = json.dumps({"split": split, "stages": active}, sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# -------------------------------------------------------------- run files --
def _run_path(digest):
    return _RESULTS / "runs" / f"{digest}.jsonl"


def _save_run(digest, ranked):
    with open(_run_path(digest), "w", encoding="utf-8") as f:
        for domain, per_query in ranked.items():
            for qid, hits in per_query.items():
                f.write(json.dumps({"d": domain, "q": qid,
                                    "i": [h[0] for h in hits],
                                    "s": [round(float(h[1]), 6) for h in hits]},
                                   separators=(",", ":")) + "\n")


def _load_run(digest):
    ranked = {}
    with open(_run_path(digest), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ranked.setdefault(row["d"], {})[row["q"]] = list(
                zip(row["i"], row["s"]))
    return ranked


# ----------------------------------------------------------------- running --
def run(system, split="train", force=False, verbose=True):
    """Run a system, reusing every cached stage. Returns the final hash."""
    _require()
    if isinstance(system, (str, Path)):
        system = load_system(system)

    ranked, digest = None, None
    for stage in STAGES:
        config = system.get(stage)
        if not config:
            continue                      # a null stage is the identity
        digest = system_hash(system, stage, split)

        if _run_path(digest).exists() and not force:
            if verbose:
                print(f"  {stage:<9} {digest}  cached")
            ranked = _load_run(digest)
            continue

        kind = config.get("kind")
        if kind not in _REGISTRY[stage]:
            raise KeyError(f"no implementation registered for "
                           f"{stage}.kind = {kind!r}; "
                           f"known: {sorted(_REGISTRY[stage])}")
        started = time.time()
        ranked = _REGISTRY[stage][kind](ranked, config, split)
        _save_run(digest, ranked)
        if verbose:
            print(f"  {stage:<9} {digest}  computed in "
                  f"{time.time() - started:.0f}s")

    if digest is None:
        raise ValueError("this system has no active stage")

    _note(digest, system, split)
    return digest


def _note(digest, system, split):
    """Record hash -> system, so a hash can always be traced back."""
    path = _RESULTS / "index.json"
    entries = (json.loads(path.read_text(encoding="utf-8"))
               if path.exists() else {})
    name = system.get("name", "unnamed")
    existing = entries.get(digest)
    if existing:
        # Two system files that hash the same ARE the same system. Keep the
        # first name and record the other, which is a useful warning that two
        # configurations differ only in ways that do not matter.
        aliases = sorted(set(existing.get("aliases", [])) | {name}
                         - {existing["name"]})
        entries[digest] = {**existing, "aliases": aliases,
                           "when": datetime.now(timezone.utc)
                                   .isoformat(timespec="seconds")}
    else:
        entries[digest] = {
            "name": name,
            "split": split,
            "system": {s: system.get(s) for s in STAGES},
            "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    path.write_text(json.dumps(entries, indent=1), encoding="utf-8")


def listing():
    path = _RESULTS / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ------------------------------------------------------------------ stages --
_TOKENIZERS = {
    "simple": lambda: R.tokenize_simple,
    "lucene_like": lambda: R.tokenize_lucene,
    "nohtml": lambda: R.make_tokenizer(drop_html=True),
    "nohtml_stop": lambda: R.make_tokenizer(drop_html=True,
                                            stopwords=R.EXTENDED_STOPWORDS),
}

import re as _re

_HTML_TAG = _re.compile(r"<[^>]+>")
_HTML_ENTITY = _re.compile(r"&(?:quot|amp|lt|gt|nbsp|apos|#\d+);")


def _strip_markup(text):
    text = _HTML_ENTITY.sub(" ", _HTML_TAG.sub(" ", text))
    return _re.sub(r"\s+", " ", text).strip()


def _title_of(text):
    match = _re.search(r"<p>", text)
    return text[:match.start()] if match else text


_QUERY_FORMS = {
    "raw": lambda t: t,
    "stripped": _strip_markup,
    "title": lambda t: _strip_markup(_title_of(t)),
    "title_weighted": lambda t: _strip_markup((_title_of(t) + " ") * 3 + t),
}


def _queries_file(split):
    return {"train": "examples_train.jsonl", "dev": "examples_dev.jsonl"}[split]


def _qrels_file(split):
    return {"train": "qrels_train.txt", "dev": "qrels_dev.txt"}[split]


@register("retrieve", "sparse")
def _retrieve_sparse(incoming, config, split):
    """Build one index per domain and retrieve. This is the expensive stage."""
    shape = _QUERY_FORMS[config.get("query_form", "title_weighted")]
    tokenizer = _TOKENIZERS[config.get("tokenizer", "nohtml_stop")]()
    model = config.get("model", "bm25")
    depth = int(config.get("depth", 100))

    ranked = {}
    for i, domain in enumerate(domains(), start=1):
        t0 = time.time()
        doc_ids, doc_texts = R.load_corpus(_DATA / domain / "documents.jsonl")
        queries = R.load_queries(_DATA / domain / _queries_file(split))

        index = R.RetrievalIndex(
            doc_ids, doc_texts, tokenizer=tokenizer, weighting=model,
            k1=float(config.get("k1", 0.9)), b=float(config.get("b", 0.4)),
            mu=float(config.get("mu", 2000.0)), keep_counts=False)

        per_query = {}
        for qid, text in queries:
            hits = index.search(shape(text), top_k=depth)
            if hits:                      # a query with no hits is left OUT,
                per_query[qid] = hits     # matching the official scorer
        ranked[domain] = per_query

        del index, doc_ids, doc_texts
        print(f"    [{i:>2}/{len(domains())}] {domain:<12}"
              f"{time.time() - t0:>6.0f}s", flush=True)
    return ranked


@register("dedup", "none")
@register("rerank", "none")
@register("fuse", "none")
def _identity(incoming, config, split):
    return incoming


# ------------------------------------------------- duplicate text groups --
def _groups_dir():
    return _RESULTS / "dup_groups"


def build_dup_groups(force=False):
    """One pass over the corpus, writing {doc_id: canonical_id} per domain.

    Only documents that actually have a twin are written, so the files hold
    roughly a third of the corpus rather than all of it. Built once; every
    later dedup run reads these files instead of re-reading 4 GB of text.
    """
    _require()
    _groups_dir().mkdir(parents=True, exist_ok=True)
    summary = {}
    for i, domain in enumerate(domains(), start=1):
        path = _groups_dir() / f"{domain}.json"
        if path.exists() and not force:
            mapping = json.loads(path.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            doc_ids, doc_texts = R.load_corpus(_DATA / domain / "documents.jsonl")
            mapping = R.duplicate_groups(doc_ids, doc_texts)
            path.write_text(json.dumps(mapping), encoding="utf-8")
            print(f"    [{i:>2}/{len(domains())}] {domain:<12}"
                  f"{len(doc_ids):>9,} docs{time.time() - t0:>6.0f}s", flush=True)
            del doc_ids, doc_texts
        canonicals = set(mapping.values())
        summary[domain] = {"in_groups": len(mapping), "groups": len(canonicals)}
    return summary


def _load_groups(domain):
    path = _groups_dir() / f"{domain}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run pipeline.build_dup_groups() once "
            f"(notebook 07a does this) before using a dedup stage.")
    return json.loads(path.read_text(encoding="utf-8"))


@register("dedup", "content_hash")
def _dedup_content_hash(incoming, config, split):
    """Collapse copies of the same text, then optionally put some back.

    Two settings, and they pull in opposite directions:

      keep   how many candidates to carry forward after collapsing. The
             collapse frees slots, and this decides whether the freed slots
             are handed back (a deeper list of distinct texts) or dropped.
      expand how many ids of one text may appear in the final list. 1 trusts
             the collapse. More than 1 hedges against the qrels marking a
             copy that is not the one kept.
    """
    keep = config.get("keep")
    expand = int(config.get("expand", 1))
    out = {}
    for domain, per_query in incoming.items():
        canonical_of = _load_groups(domain)
        result = {}
        for qid, hits in per_query.items():
            ids = [doc for doc, _ in hits]
            score_of = dict(hits)
            kept, twins = R.collapse(ids, canonical_of)
            final = R.expand_groups(kept, twins, max_per_group=expand,
                                    limit=keep)
            # A twin inherits the score of the id it was ranked behind, so the
            # list stays sorted and the scores stay meaningful downstream.
            result[qid] = [(doc, score_of.get(doc, 0.0)) for doc in final]
        out[domain] = result
    return out


# ------------------------------------------------------- work sent to GPU --
def _gpu_dir():
    return _RESULTS / "gpu"


def export_gpu_job(digest, out_dir, job_id, task="rerank", split="train",
                   depth=None, with_text=False, model=None):
    """Write a self-standing job folder for a machine with a GPU.

    The GPU machine does not need this project. It needs the worker script,
    the folder this writes, and either a copy of the corpus or `with_text`.

    with_text=False  the worker reads the corpus itself. The folder stays
                     around 50 MB, which is a file the two machines can pass
                     around easily.
    with_text=True   the document text travels too. Self-contained, but the
                     folder runs to hundreds of megabytes.
    """
    _require()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ranked = _load_run(digest)

    n_pairs = 0
    needed = {d: set() for d in ranked}
    with open(out / "candidates.jsonl", "w", encoding="utf-8") as f:
        for domain, per_query in ranked.items():
            for qid, hits in per_query.items():
                ids = [doc for doc, _ in hits][:depth] if depth else [
                    doc for doc, _ in hits]
                needed[domain].update(ids)
                n_pairs += len(ids)
                f.write(json.dumps({"d": domain, "q": qid, "i": ids}) + "\n")

    with open(out / "queries.jsonl", "w", encoding="utf-8") as f:
        for domain in ranked:
            for qid, text in R.load_queries(
                    _DATA / domain / _queries_file(split)):
                f.write(json.dumps({"d": domain, "q": qid, "t": text}) + "\n")

    if with_text:
        with open(out / "documents.jsonl", "w", encoding="utf-8") as f:
            for domain, wanted in needed.items():
                for record in R._read_jsonl(
                        _DATA / domain / "documents.jsonl"):
                    if record["id"] in wanted:
                        f.write(json.dumps({"d": domain, "i": record["id"],
                                            "t": record["content"] or ""})
                                + "\n")

    job = {"job_id": job_id, "task": task, "split": split,
           "from_hash": digest, "depth": depth, "model": model,
           "pairs": n_pairs, "with_text": with_text,
           "domains": sorted(ranked)}
    (out / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")

    print(f"wrote {out}")
    print(f"  job_id  {job_id}")
    print(f"  pairs   {n_pairs:,}")
    print(f"  corpus  {'included' if with_text else 'read by the worker'}")
    return job


def import_gpu_scores(path, job_id=None):
    """Take `scores.jsonl` back from the GPU machine.

    The file is stored under its job id. A rerank or dense stage naming that
    job id then finds it, which is what lets a system definition describe a
    run that was produced on another machine.
    """
    _require()
    _gpu_dir().mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if job_id is None:
        job_id = rows[0].get("job_id") if rows else None
    if not job_id:
        raise ValueError("job_id not given and not present in the file")

    target = _gpu_dir() / f"{job_id}.jsonl"
    with open(target, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    queries = len({r["q"] for r in rows})
    print(f"imported {queries:,} queries into {target}")
    return job_id


def _load_gpu_scores(job_id):
    path = _gpu_dir() / f"{job_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run the worker on the GPU machine, then "
            f"pipeline.import_gpu_scores(<its scores.jsonl>, '{job_id}').")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out.setdefault(row["d"], {})[row["q"]] = list(
                zip(row["i"], row["s"]))
    return out


@register("rerank", "gpu_job")
def _rerank_gpu_job(incoming, config, split):
    """Reorder each candidate list using scores computed on the GPU machine.

    Only documents the GPU actually scored are reordered. A candidate the
    worker skipped keeps its place at the bottom rather than vanishing, so a
    partial job degrades instead of silently shortening every list.
    """
    job_id = config["job"]
    depth = config.get("depth")
    scored = _load_gpu_scores(job_id)

    out = {}
    for domain, per_query in incoming.items():
        domain_scores = scored.get(domain, {})
        result = {}
        for qid, hits in per_query.items():
            new = domain_scores.get(qid)
            if not new:
                result[qid] = hits
                continue
            ordered = sorted(new, key=lambda pair: (-pair[1], pair[0]))
            covered = {doc for doc, _ in new}
            tail = [(doc, s) for doc, s in hits if doc not in covered]
            merged = ordered + tail
            result[qid] = merged[:depth] if depth else merged
        out[domain] = result
    return out


@register("retrieve", "gpu_job")
def _retrieve_gpu_job(incoming, config, split):
    """A candidate list produced entirely on the GPU machine.

    Used for dense retrieval, where the expensive part is embedding the whole
    corpus and belongs wherever the GPU is.
    """
    depth = config.get("depth")
    scored = _load_gpu_scores(config["job"])
    return {domain: {qid: (hits[:depth] if depth else hits)
                     for qid, hits in per_query.items()}
            for domain, per_query in scored.items()}


# ---------------------------------------------------------------- fusion --
@register("fuse", "rrf")
def _fuse_rrf(incoming, config, split):
    """Reciprocal rank fusion of this system's list with other saved runs.

    `sources` names other runs by hash or by the name they were registered
    under. They must already exist in results/runs/, which means they were
    produced by their own system definition and can be inspected on their own.
    """
    k = float(config.get("k", 60))
    depth = config.get("depth")
    names = {entry["name"]: h for h, entry in listing().items()}
    others = [_load_run(names.get(source, source))
              for source in config.get("sources", [])]

    out = {}
    for domain, per_query in incoming.items():
        result = {}
        for qid, hits in per_query.items():
            lists = [[doc for doc, _ in hits]]
            for other in others:
                extra = other.get(domain, {}).get(qid)
                if extra:
                    lists.append([doc for doc, _ in extra])
            fused = R.reciprocal_rank_fusion(lists, k=k, top_k=depth)
            # RRF scores are tiny and on their own scale; the rank is what
            # carries meaning, so the position is written back as the score.
            result[qid] = [(doc, 1.0 / (k + rank))
                           for rank, doc in enumerate(fused, start=1)]
        out[domain] = result
    return out


# ----------------------------------------------------------------- scoring --
def _gold(split):
    return {d: R.load_qrels(_DATA / d / _qrels_file(split)) for d in domains()}


# Metrics recorded per system. `ndcg` decides; the rest describe.
METRICS = ("ndcg", "recall", "ceiling", "precision", "f1", "recall_at_cut", "map")

# Bumped whenever the contents of a score record change, so that records
# written by an older version are recomputed instead of silently reused.
SCORE_VERSION = 2


def score(digest, split="train", force=False):
    """Score a run and cache the result. Returns the score record."""
    _require()
    path = _RESULTS / "scores" / f"{digest}.json"
    if path.exists() and not force:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("score_version") == SCORE_VERSION:
            return cached

    ranked = _load_run(digest)
    gold_all = _gold(split)

    per_domain, per_query_ndcg, per_query_ap = {}, {}, {}
    for domain, per_query in ranked.items():
        gold = gold_all[domain]
        lists = {qid: [doc for doc, _ in hits] for qid, hits in per_query.items()}
        result = R.evaluate(lists, gold)
        reachable = [R.rerank_ceiling(lists[q], gold[q])
                     for q in lists if q in gold]
        entry = {m: result[m] for m in METRICS if m in result}
        entry["ceiling"] = float(np.mean(reachable)) if reachable else 0.0
        entry.update({
            "n_topics": result["n_topics"],
            "n_gold_queries": result["n_gold_queries"],
            "per_query_ndcg": result["per_query_ndcg"],
            "per_query_recall": result["per_query_recall"],
            "per_query_ap": result["per_query_ap"],
        })
        per_domain[domain] = entry
        per_query_ndcg.update(result["per_query_ndcg"])
        per_query_ap.update(result["per_query_ap"])

    record = {
        "hash": digest,
        "split": split,
        "score_version": SCORE_VERSION,
        "name": listing().get(digest, {}).get("name", "unnamed"),
        "macro": {m: R.macro(per_domain, m) for m in METRICS},
        "n_topics": sum(d["n_topics"] for d in per_domain.values()),
        "n_gold_queries": sum(d["n_gold_queries"] for d in per_domain.values()),
        "per_domain": {d: {k: v for k, v in s.items()
                           if not k.startswith("per_query")}
                       for d, s in per_domain.items()},
        "per_query_ndcg": per_query_ndcg,
        "per_query_ap": per_query_ap,
    }
    # The two halves of train are read off the same run, never rerun.
    if split == "train":
        for part in ("fit", "check"):
            restricted = R.restrict(per_domain, _SPLIT, part)
            record[part] = {m: R.macro(restricted, m) for m in ("ndcg", "recall")}
            record[f"{part}_n_topics"] = sum(r["n_topics"]
                                             for r in restricted.values())

    path.write_text(json.dumps(record), encoding="utf-8")
    return record


def _per_domain_results(digest):
    """Rebuild the shape `bootstrap_macro_diff` and `restrict` expect."""
    record = json.loads((_RESULTS / "scores" / f"{digest}.json")
                        .read_text(encoding="utf-8"))
    ranked = _load_run(digest)
    gold_all = _gold(record["split"])
    out = {}
    for domain, per_query in ranked.items():
        gold = gold_all[domain]
        lists = {qid: [doc for doc, _ in hits] for qid, hits in per_query.items()}
        out[domain] = R.evaluate(lists, gold)
    return out


# --------------------------------------------------------------- comparing --
def table(digests=None, part="fit"):
    """Print one row per system. Sorted by `part`, which defaults to fit.

    Sorting by fit is deliberate. Sorting by the whole-train score would let
    `check` influence the choice, and `check` only means anything while no
    choice has been made with it.
    """
    _require()
    records = []
    for digest in (digests or sorted(listing())):
        path = _RESULTS / "scores" / f"{digest}.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        print("no scored systems yet")
        return []

    key = (lambda r: -r.get(part, {}).get("ndcg", r["macro"]["ndcg"]))
    records.sort(key=key)

    print(f"{'name':<30}{'hash':<14}{'fit':>9}{'check':>9}"
          f"{'ceiling':>10}{'topics':>9}")
    print("-" * 81)
    for r in records:
        fit = r.get("fit", {}).get("ndcg")
        check = r.get("check", {}).get("ndcg")
        flag = "" if r["n_topics"] == r["n_gold_queries"] else "  <- INCOMPLETE"
        print(f"{r['name'][:29]:<30}{r['hash']:<14}"
              f"{(f'{fit:.4f}' if fit is not None else '-'):>9}"
              f"{(f'{check:.4f}' if check is not None else '-'):>9}"
              f"{r['macro']['ceiling']:>10.4f}{r['n_topics']:>9}{flag}")
    print("-" * 81)
    print("Sorted by fit. `check` is read only after a choice is made, never "
          "sorted on.")
    return records


def metrics(digests=None):
    """Print every metric for every scored system, one row per system.

    `table` shows the one number that decides. This shows the full set, which
    is what the evaluation write-up reports: P/R/F1 for the unranked view,
    P@10 and MAP for the ranked view, and nDCG@10 on top of them.
    """
    _require()
    records = []
    for digest in (digests or sorted(listing())):
        path = _RESULTS / "scores" / f"{digest}.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        print("no scored systems yet")
        return []
    records.sort(key=lambda r: -r["macro"]["ndcg"])

    columns = [("P@10", "precision"), ("R@10", "recall_at_cut"),
               ("F1@10", "f1"), ("MAP", "map"),
               ("nDCG@10", "ndcg"), ("R@100", "recall")]
    header = f"{'name':<30}" + "".join(f"{label:>10}" for label, _ in columns)
    print(header)
    print("-" * len(header))
    stale = []
    for r in records:
        row = f"{r['name'][:29]:<30}"
        for _, key in columns:
            value = r["macro"].get(key)
            row += (f"{value:>10.4f}" if value is not None else f"{'-':>10}")
        if r.get("score_version") != SCORE_VERSION:
            stale.append(r["hash"])
            row += "   <- scored before these metrics existed"
        print(row)
    print("-" * len(header))
    if stale:
        print("Rescore those rows to fill the gaps; it reads the saved run and")
        print("retrieves nothing:")
        print(f"    for h in {stale!r}: P.score(h, force=True)")
    print(f"Every number is a macro average over the {len(domains())} domains.")
    print("P@10 / R@10 / F1@10 ignore the order inside the top 10;")
    print("MAP and nDCG@10 do not.")
    return records


def cases(digest, n=5, metric="ndcg"):
    """Show the queries this system answers best and worst.

    A table of averages says how well a system does. It never says on what.
    These two lists are where a failure becomes readable: the worst rows are
    the ones worth opening one by one.
    """
    _require()
    record = json.loads((_RESULTS / "scores" / f"{digest}.json")
                        .read_text(encoding="utf-8"))
    key = "per_query_ndcg" if metric == "ndcg" else "per_query_ap"
    scores = {q: v for q, v in record[key].items() if v is not None}
    if not scores:
        print("nothing scored")
        return [], []

    # Which domain a query belongs to, so each case can be read in context.
    domain_of = {}
    for domain in domains():
        for qid, _ in R.load_queries(_DATA / domain / _queries_file(
                record["split"])):
            domain_of[qid] = domain

    # Ties broken by query id both ways, so the lists do not move between runs.
    high = sorted(scores, key=lambda q: (-scores[q], q))[:n]
    low = sorted(scores, key=lambda q: (scores[q], q))[:n]
    label = metric.upper() if metric == "ap" else "nDCG@10"
    for title, group in (("HIGHEST", high), ("LOWEST", low)):
        print(f"{title} {label}")
        for qid in group:
            print(f"  {scores[qid]:.4f}  {domain_of.get(qid, '?'):<12}{qid}")
        print()
    return high, low


def _query_texts(split):
    texts, domain_of = {}, {}
    for domain in domains():
        for qid, text in R.load_queries(_DATA / domain / _queries_file(split)):
            texts[qid] = text
            domain_of[qid] = domain
    return texts, domain_of


def explain(digest, qids, n_chars=70):
    """Per-query detail: how many gold documents exist, and where they landed.

    A score of 0 has two very different causes, and only this tells them
    apart: the first stage never retrieved the gold document (nothing a
    reranker could fix), or it retrieved it and ranked it below 10 (exactly
    what a reranker fixes). The `found` and `best rank` columns separate them.
    """
    _require()
    record = json.loads((_RESULTS / "scores" / f"{digest}.json")
                        .read_text(encoding="utf-8"))
    ranked = _load_run(digest)
    gold_all = _gold(record["split"])
    texts, domain_of = _query_texts(record["split"])

    print(f"{'query':<22}{'domain':<12}{'nDCG':>7}{'gold':>6}{'found':>7}"
          f"{'best rank':>11}")
    print("-" * 65)
    rows = []
    for qid in qids:
        domain = domain_of.get(qid)
        gold = gold_all.get(domain, {}).get(qid, set())
        hits = [doc for doc, _ in ranked.get(domain, {}).get(qid, [])]
        positions = [hits.index(doc) + 1 for doc in gold if doc in hits]
        best = min(positions) if positions else None
        value = record["per_query_ndcg"].get(qid)
        print(f"{qid[:21]:<22}{str(domain)[:11]:<12}"
              f"{(f'{value:.4f}' if value is not None else '-'):>7}"
              f"{len(gold):>6}{len(positions):>7}"
              f"{(str(best) if best else 'not found'):>11}")
        clean = " ".join(_strip_markup(texts.get(qid, "")).split())
        print(f"    {clean[:n_chars]}{'...' if len(clean) > n_chars else ''}")
        rows.append({"qid": qid, "domain": domain, "ndcg": value,
                     "n_gold": len(gold), "n_found": len(positions),
                     "best_rank": best})
    print("-" * 65)
    return rows


def failure_breakdown(digest):
    """Split every query into what it would take to fix it.

    Three buckets, and each one points at different work:

      scored        something relevant is already in the top 10
      recoverable   nothing relevant in the top 10, but the candidate list
                    holds one -- a reranker can lift it
      unreachable   the candidate list holds nothing relevant at all, so no
                    reranker can help and only the first stage can

    The split matters before building a reranker: its whole gain is bounded by
    the recoverable bucket.
    """
    _require()
    record = json.loads((_RESULTS / "scores" / f"{digest}.json")
                        .read_text(encoding="utf-8"))
    ranked = _load_run(digest)
    gold_all = _gold(record["split"])

    buckets = {"scored": [], "recoverable": [], "unreachable": []}
    for domain, per_query in ranked.items():
        gold_domain = gold_all.get(domain, {})
        for qid, hits in per_query.items():
            gold = gold_domain.get(qid)
            if not gold:
                continue
            if (record["per_query_ndcg"].get(qid) or 0.0) > 0:
                buckets["scored"].append(qid)
            elif any(doc in gold for doc, _ in hits):
                buckets["recoverable"].append(qid)
            else:
                buckets["unreachable"].append(qid)

    total = sum(len(v) for v in buckets.values())
    print(f"{'bucket':<16}{'queries':>10}{'share':>10}")
    print("-" * 36)
    for name in ("scored", "recoverable", "unreachable"):
        n = len(buckets[name])
        print(f"{name:<16}{n:>10}{(n / total if total else 0):>10.1%}")
    print("-" * 36)
    print(f"{'total':<16}{total:>10}")
    return buckets


def compare(digest_a, digest_b, part="fit", n_resamples=10_000):
    """Paired bootstrap on the difference, stratified by domain."""
    _require()
    a = _per_domain_results(digest_a)
    b = _per_domain_results(digest_b)
    if part in ("fit", "check"):
        a, b = R.restrict(a, _SPLIT, part), R.restrict(b, _SPLIT, part)
    diff, low, high, real = R.bootstrap_macro_diff(
        a, b, n_resamples=n_resamples, seed=0)

    names = listing()
    name_a = names.get(digest_a, {}).get("name", digest_a)
    name_b = names.get(digest_b, {}).get("name", digest_b)
    print(f"{name_a} vs {name_b}   on {part}")
    print(f"  difference    {diff:+.4f}")
    print(f"  95% interval  [{low:+.4f}, {high:+.4f}]")
    print(f"  verdict       {'REAL' if real else 'tie'}")

    changed = _config_diff(names.get(digest_a, {}).get("system", {}),
                           names.get(digest_b, {}).get("system", {}))
    print(f"  changed       {', '.join(changed) if changed else 'nothing'}")
    if len(changed) > 1:
        print("  more than one thing changed, so the difference cannot be")
        print("  attributed to any single one of them")
    return {"diff": diff, "ci_low": low, "ci_high": high,
            "significant": bool(real), "changed": changed}


def _config_diff(system_a, system_b, prefix=""):
    changed = []
    for key in sorted(set(system_a) | set(system_b)):
        left, right = system_a.get(key), system_b.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            changed += _config_diff(left, right, f"{prefix}{key}.")
        elif left != right:
            changed.append(f"{prefix}{key}: {left} -> {right}")
    return changed


# ------------------------------------------------------- the split itself --
def split_report():
    """Compare fit and check on properties that do not involve any score.

    Looking at what is IN check is allowed; looking at how well a system does
    on check is what has to be rationed. This checks that the two halves are
    alike, so a choice made on fit is not being made on a different kind of
    question from the one check verifies.
    """
    _require()
    rows = {}
    for part in ("fit", "check"):
        lengths, gold_counts, with_year = [], [], 0
        for domain in domains():
            wanted = set(_SPLIT[domain][part])
            gold = R.load_qrels(_DATA / domain / _qrels_file("train"))
            for qid, text in R.load_queries(_DATA / domain /
                                            _queries_file("train")):
                if qid not in wanted:
                    continue
                lengths.append(len(_strip_markup(text).split()))
                gold_counts.append(len(gold.get(qid, ())))
                with_year += bool(_re.search(r"\b(19|20)\d{2}\b", text))
        gold_counts = np.asarray(gold_counts)
        # Shares and means, never a median: a median over small integers moves
        # in whole steps, so a 2 against a 3 reads as a 50% gap that means
        # nothing.
        rows[part] = {
            "queries": len(lengths),
            "median_words": float(np.median(lengths)),
            "mean_gold": float(gold_counts.mean()),
            "share_multi_gold": float((gold_counts > 1).mean()),
            "year_mentioned": with_year / len(lengths),
        }

    print(f"{'property':<26}{'fit':>12}{'check':>12}{'difference':>14}")
    print("-" * 64)
    labels = {"queries": "queries",
              "median_words": "median query words",
              "mean_gold": "mean gold documents",
              "share_multi_gold": "more than one gold doc",
              "year_mentioned": "mentions a year"}
    as_share = {"year_mentioned", "share_multi_gold"}
    worst = 0.0
    for key, label in labels.items():
        f, c = rows["fit"][key], rows["check"][key]
        if key == "queries":
            print(f"{label:<26}{f:>12.0f}{c:>12.0f}{'':>14}")
            continue
        relative = abs(c - f) / f if f else 0.0
        worst = max(worst, relative)
        shown = f"{f:.1%}" if key in as_share else f"{f:.2f}"
        shown_c = f"{c:.1%}" if key in as_share else f"{c:.2f}"
        print(f"{label:<26}{shown:>12}{shown_c:>12}{relative:>13.1%}")
    print("-" * 64)
    print("No score is read here, so running this does not spend `check`.")
    print()
    if worst < 0.15:
        print(f"The two halves look alike (largest gap {worst:.1%}). A choice "
              f"made on fit\nis being checked against the same kind of query.")
    else:
        print(f"One property differs by {worst:.1%}. Say so in the report: "
              f"`check` is\nverifying a slightly different mix of queries.")
    return rows


# ------------------------------------------------------------- submitting --
def submit(system, out_path, tag, split="dev", checker=None, top_k=10):
    """Write a TREC run file and validate it before it can be sent.

        qid   Q0   docid   rank   score   tag

    The organisers' README is explicit: a submission that fails validation
    should never reach the competition platform. So the check runs here, and a
    file that fails is deleted rather than left lying around.
    """
    _require()
    if isinstance(system, (str, Path)):
        system = load_system(system)

    digest = run(system, split=split)
    ranked = _load_run(digest)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for domain in sorted(ranked):
            for qid in sorted(ranked[domain]):
                for rank, (doc_id, score_value) in enumerate(
                        ranked[domain][qid][:top_k], start=1):
                    f.write(f"{qid} Q0 {doc_id} {rank} "
                            f"{score_value:.6f} {tag}\n")
                    lines += 1
    print(f"wrote {lines:,} lines to {out_path}")

    checker = Path(checker) if checker else None
    if checker and checker.exists():
        done = subprocess.run([sys.executable, str(checker), str(out_path)],
                              capture_output=True, text=True)
        print(done.stdout.strip() or done.stderr.strip())
        if done.returncode != 0:
            out_path.unlink()
            raise RuntimeError("the format checker rejected this run; "
                               "the file has been deleted")
        print("format checker: PASSED")
    else:
        print("format checker not found -- run it by hand before submitting")
    return out_path
