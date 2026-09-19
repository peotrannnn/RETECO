#!/usr/bin/env python3
"""Decompose the `aggressive` tokenizer into its parts.

WHY THIS EXISTS
---------------
`aggressive` measured macro nDCG@10 0.1154 against the production baseline's
0.0719, better on 13 of 13 domains. But it bundles THREE changes at once:

    drop_html          remove HTML structural tokens (p, href, li, code, ...)
    stem               Porter stemming
    EXTENDED_STOPWORDS a ~120-word stop list instead of Lucene's 33
    min_len=2          discard single-character tokens

and it was picked as the best of six variants on the train split. Two reasons
not to ship that as-is:

  1. Best-of-N on train is where overfitting hides. A bundle that wins by
     +0.0435 might be one component winning by +0.045 and three doing nothing,
     or four each contributing a little. Those call for different write-ups
     and carry different risk on the unseen test split.
  2. The bundle contains a component with no mechanism behind it. Removing
     HTML has an obvious one (100% of queries carry markup, no document does).
     Stemming has a standard one. But there is no reason to expect discarding
     single-character tokens to help TEMPORAL retrieval, and if that turns out
     to be carrying the gain, the result deserves suspicion rather than a
     paragraph in the system paper.

DESIGN
------
Each row is `aggressive` with exactly one component removed, plus each
component on its own. The difference between `aggressive` and
`aggressive minus X` is X's marginal contribution in the presence of the rest.

This imports the tokenizer from retriever.py, so it tests the SHIPPED code
rather than a copy of it. Indexing uses the tokenize-once / remap-many route,
which notebook 02 verified produces rankings identical to the production
Retriever on all six presets.

USAGE
-----
    python src/track1a/tokenizer_experiment.py
    python src/track1a/tokenizer_experiment.py --domains iota law quant
    python src/track1a/tokenizer_experiment.py --max-docs 5000   # smoke test

Runtime: about 20 minutes for all 13 domains (one tokenization pass, then the
variants are near-free). Peak memory about 2.9 GB, on the largest domain.
"""
import argparse
import gc
import json
import re
import sys
import time
from array import array
from math import log2
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retriever import (TOKENIZERS, HTML_TOKENS, LUCENE_STOPWORDS,      # noqa: E402
                       EXTENDED_STOPWORDS, STEMMER, tokenize)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# The ablation. Order is meaningful: control, then components alone, then
# aggressive-minus-one, then the full bundle.
# --------------------------------------------------------------------------
ABLATION = [
    ("baseline",            "control: lowercase + [A-Za-z0-9]+"),
    ("html_only",           "drop_html alone"),
    ("extstop_only",        "extended stopwords alone"),
    ("minlen_only",         "min_len=2 alone"),
    ("stem",                "stemming alone"),
    ("html+stem",           "drop_html + stem"),
    ("html+stem+extstop",   "aggressive MINUS min_len"),
    ("html+stem+minlen",    "aggressive MINUS extended stopwords"),
    ("stop+stem+html",      "aggressive with Lucene's 33-word stop list"),
    ("aggressive",          "the full bundle"),
    # The rows above are all "aggressive minus one component", and aggressive
    # contains stem, so every one of them carries stem. The 13-domain run
    # measured stem as the only component that HURTS, which leaves the
    # stem-free combination of the components that help unmeasured. These
    # three rows close that gap.
    ("extstop+html",        "extstop + drop_html, NO stem"),
    ("extstop+minlen",      "extstop + min_len, NO stem"),
    ("extstop+html+minlen", "all three helpful components, NO stem"),
]

# A focused follow-up: the control, the current best, the full bundle for
# reference, and the three stem-free combinations. Runs in roughly half the
# time of the full ablation because six variants are skipped.
FOCUS = ["baseline", "extstop_only", "aggressive",
         "extstop+html", "extstop+minlen", "extstop+html+minlen"]

QUERY_FORM_DEFAULT = "title-weighted"

_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY = re.compile(r"&(quot|amp|lt|gt|nbsp|#\d+);")


def strip_markup(t):
    t = _HTML_ENTITY.sub(" ", _HTML_TAG.sub(" ", t))
    return re.sub(r"\s+", " ", t).strip()


def title_of(q):
    m = re.search(r"<p>", q)
    return q[:m.start()] if m else q


QUERY_FORMS = {
    "raw":            lambda q: q,
    "stripped":       strip_markup,
    "title":          lambda q: strip_markup(title_of(q)),
    "title-weighted": lambda q: strip_markup((title_of(q) + " ") * 3 + q),
}


# ------------------------------------------------------------------ metrics
def _dcg(rels):
    return sum(r / log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked, gold, k=10):
    idcg = _dcg([1.0] * min(len(gold), k))
    if idcg <= 0:
        return 0.0
    return _dcg([1.0 if d in gold else 0.0 for d in ranked[:k]]) / idcg


def recall_at_k(ranked, gold, k):
    return len(set(ranked[:k]) & set(gold)) / len(gold) if gold else None


def tc_at_k(ranked, periods, k=10):
    if not periods:
        return None
    top = set(ranked[:k])
    return sum(1 for g in periods if top & set(g)) / len(periods)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(vals, n=4000, seed=0):
    v = np.asarray([x for x in vals if x is not None], float)
    if not v.size:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, v.size)).mean(axis=1)
    return float(v.mean()), *(float(x) for x in np.percentile(m, [2.5, 97.5]))


# ------------------------------------------------------------------ loading
def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_corpus(p, limit=None):
    ids, texts = [], []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                ids.append(d["id"])
                texts.append(d["content"])
                if limit and len(ids) >= limit:
                    break
    return ids, texts


def load_qrels(p):
    gold = {}
    with open(p, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                q, _, d, rel = ln.split()
                if int(rel) > 0:
                    gold.setdefault(q, set()).add(d)
    return gold


def load_steps(p):
    return {r["id"]: [set(s.get("gold_ids", [])) for s in r.get("steps", [])]
            for r in read_jsonl(p)}


# ------------------------------------------------- tokenize once, remap many
def tokenize_corpus(texts):
    """Raw tokenization, paid once per domain."""
    vocab, indptr, indices, data = {}, array("q", [0]), array("i"), array("f")
    for t in texts:
        c = {}
        for tok in re.findall(r"[A-Za-z0-9]+", (t or "").lower()):
            c[tok] = c.get(tok, 0) + 1
        for tok, f in c.items():
            j = vocab.get(tok)
            if j is None:
                j = len(vocab)
                vocab[tok] = j
            indices.append(j)
            data.append(f)
        indptr.append(len(indices))
    m = sparse.csr_matrix(
        (np.frombuffer(data, dtype=np.float32).copy(),
         np.frombuffer(indices, dtype=np.int32).copy(),
         np.frombuffer(indptr, dtype=np.int64).copy()),
        shape=(len(indptr) - 1, max(len(vocab), 1)))
    return m, sorted(vocab, key=vocab.get)


def build_vocab_map(terms, opts):
    """Relabel the raw vocabulary under a tokenizer preset.

    The preset is applied through the PRODUCTION tokenize(), one term at a
    time, so this experiment cannot drift from what the system actually ships.
    """
    col = np.full(len(terms), -1, dtype=np.int64)
    ids, out = {}, []
    for i, t in enumerate(terms):
        u = tokenize(t, **opts)
        if not u:
            continue
        j = ids.get(u[0])
        if j is None:
            j = len(out)
            ids[u[0]] = j
            out.append(u[0])
        col[i] = j
    return col, out


def remap_counts(counts, col_map, n_new):
    new_col = col_map[counts.indices]
    keep = new_col >= 0
    indptr = np.concatenate(([0], np.cumsum(keep, dtype=np.int64)))[counts.indptr]
    m = sparse.csr_matrix(
        (counts.data[keep], new_col[keep].astype(np.int32, copy=False), indptr),
        shape=(counts.shape[0], max(n_new, 1)))
    m.sum_duplicates()          # terms sharing a stem have counts ADDED
    return m


class BM25:
    """Identical in formula to retriever.py; consumes `counts` in place."""

    def __init__(self, doc_ids, counts, terms, k1=0.9, b=0.4, row_block=50_000):
        self.doc_ids, self.N = list(doc_ids), len(doc_ids)
        self.vocab = {t: j for j, t in enumerate(terms)}
        tf = counts.tocsr()
        if tf.dtype != np.float32:
            tf = tf.astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel().astype(np.float32)
        df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
        idf = np.log(1.0 + (self.N - df + 0.5) / (df + 0.5)).astype(np.float32)
        del df
        avgdl = float(dl.mean()) if self.N else 0.0
        norm = (1.0 - b + b * dl / (avgdl or 1.0)).astype(np.float32)
        del dl
        ip = tf.indptr
        for r0 in range(0, self.N, row_block):
            r1 = min(r0 + row_block, self.N)
            s, e = ip[r0], ip[r1]
            if s == e:
                continue
            blk = np.repeat(norm[r0:r1], np.diff(ip[r0:r1 + 1]))
            f = tf.data[s:e]
            tf.data[s:e] = idf[tf.indices[s:e]] * f * (k1 + 1.0) / (f + k1 * blk)
            del blk, f
        del idf, norm
        post = tf.T.tocsr()
        del tf
        self._d, self._i, self._p = post.data, post.indices, post.indptr

    def search(self, terms, top_k=1000):
        c = {}
        for t in terms:
            j = self.vocab.get(t)
            if j is not None:
                c[j] = c.get(j, 0) + 1
        if not c or not self.N:
            return []
        sc = np.zeros(self.N, dtype=np.float32)
        for j, n in c.items():
            s, e = self._p[j], self._p[j + 1]
            if s != e:
                sc[self._i[s:e]] += n * self._d[s:e]
        k = min(top_k, self.N)
        idx = np.argpartition(-sc, k - 1)[:k] if k < self.N else np.arange(self.N)
        idx = idx[sc[idx] > 0]
        if not idx.size:
            return []
        pairs = [(self.doc_ids[i], float(sc[i])) for i in idx]
        pairs.sort(key=lambda x: (-x[1], x[0]))
        return pairs[:top_k]


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT_ROOT / "reteco_data"))
    ap.add_argument("--split", default="train", choices=["train", "dev"])
    ap.add_argument("--domains", nargs="*")
    ap.add_argument("--query-form", default=QUERY_FORM_DEFAULT,
                    choices=list(QUERY_FORMS))
    ap.add_argument("--max-docs", type=int, default=None,
                    help="truncate each corpus; smoke test only")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="measure only these variants (space separated). "
                         "'baseline' is always included as the control.")
    ap.add_argument("--focus", action="store_true",
                    help=f"shorthand for --variants {' '.join(FOCUS)}")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "runs" / "tokenizer_ablation.json"))
    args = ap.parse_args()

    if args.split != "train":
        print("[!] Competition rule: tune on train. dev is a single held-out "
              "check at the very end.\n")

    wanted = FOCUS if args.focus else args.variants
    if wanted:
        unknown = [v for v in wanted if v not in dict(ABLATION)]
        if unknown:
            raise SystemExit(f"unknown variant(s): {', '.join(unknown)}\n"
                             f"available: {', '.join(n for n, _ in ABLATION)}")
        keep = set(wanted) | {"baseline"}   # baseline is the control for every diff
        ablation = [(n, d) for n, d in ABLATION if n in keep]
    else:
        ablation = list(ABLATION)

    root = Path(args.data) / "track1_tempo"
    if not root.is_dir():
        raise SystemExit(f"not found: {root}")
    domains = args.domains or sorted(p.name for p in root.iterdir() if p.is_dir())
    form = QUERY_FORMS[args.query_form]

    print(f"tokenizer ablation | split={args.split} | query_form={args.query_form}")
    print(f"{len(ablation)} variants x {len(domains)} domains\n")

    results = {name: {} for name, _ in ablation}
    t_all = time.time()

    for d in domains:
        dd = root / d
        t0 = time.time()
        doc_ids, texts = load_corpus(dd / "documents.jsonl", args.max_docs)
        counts, terms = tokenize_corpus(texts)
        del texts
        gc.collect()

        queries = [(r["id"], r["query"]) for r in read_jsonl(dd / f"examples_{args.split}.jsonl")]
        gold = load_qrels(dd / f"qrels_{args.split}.txt")
        steps_p = dd / f"steps_{args.split}.jsonl"
        periods = load_steps(steps_p) if steps_p.is_file() else {}
        print(f"  {d:<12} {len(doc_ids):>7,} docs  |V|={len(terms):>8,}  "
              f"tokenized in {time.time()-t0:.0f}s", flush=True)

        for name, _desc in ablation:
            opts = TOKENIZERS[name]
            cm, nt = build_vocab_map(terms, opts)
            bm = BM25(doc_ids, remap_counts(counts, cm, len(nt)), nt)
            del cm
            gc.collect()

            run, empty = {}, 0
            for qid, q in queries:
                hits = bm.search(tokenize(form(q), **opts))
                if hits:
                    run[qid] = [x for x, _ in hits]
                else:
                    empty += 1
            qids = [q for q in gold if q in run]
            results[name][d] = {
                "nDCG@10": mean([ndcg_at_k(run[q], gold[q]) for q in qids]),
                "R@100": mean([recall_at_k(run[q], gold[q], 100) for q in qids]),
                "TC@10": mean([tc_at_k(run[q], periods.get(q, [])) for q in qids]),
                "num_topics": len(qids),
                "vocab": len(nt),
                "empty_results": empty,
                "per_query_ndcg": {q: ndcg_at_k(run[q], gold[q]) for q in qids},
                # Stored so pairwise_ci.py can test recall differences for
                # significance too. Recall is what caps a reranker, so a
                # recall difference between two variants is often the one
                # that actually decides which to ship.
                "per_query_recall": {q: recall_at_k(run[q], gold[q], 100)
                                     for q in qids},
            }
            print(f"      {name:<20} nDCG@10 {results[name][d]['nDCG@10']:.4f}  "
                  f"R@100 {results[name][d]['R@100']:.4f}  |V|={len(nt):>8,}"
                  + (f"  !! {empty} empty" if empty else ""), flush=True)
            del bm
            gc.collect()

        del counts, terms, doc_ids
        gc.collect()

    # ------------------------------------------------------------- report --
    #
    # Two different statistics get reported below, and an earlier version of
    # this script printed both while calling both "gain", which is a good way
    # to read a table wrong:
    #
    #   macro nDCG@10  - mean over DOMAINS, each domain weighted equally.
    #                    This is the competition's metric. Decisions follow it.
    #   mean per-query - mean over QUERIES, so a domain with 561 queries moves
    #                    it far more than one with 12. Only the bootstrap CI
    #                    is computed on this, because a CI needs the paired
    #                    per-query differences.
    #
    # They disagree by design. The columns are labelled so they cannot be
    # compared by accident: macro columns say "macro", per-query ones say "/q".
    base = results["baseline"]

    def macro_of(name, metric="nDCG@10"):
        per = results[name]
        return float(np.mean([r[metric] for r in per.values()]))

    print(f"\n{'='*112}")
    print(f"{'variant':<22}{'description':<40}"
          f"{'macro nDCG':>11}{'macro R@100':>12}{'diff/q':>9}{'CI of diff/q':>18}{'sig':>5}")
    print("-" * 112)
    table = []
    for name, desc in ablation:
        per = results[name]
        macro = macro_of(name)
        rec = macro_of(name, "R@100")
        diffs = [per[d]["per_query_ndcg"][q] - base[d]["per_query_ndcg"][q]
                 for d in per for q in per[d]["per_query_ndcg"]
                 if q in base[d]["per_query_ndcg"]]
        dm, lo, hi = bootstrap_ci(diffs)
        sig = lo > 0 or hi < 0
        emp = sum(r["empty_results"] for r in per.values())
        table.append((name, desc, macro, rec, dm, lo, hi, sig, emp))
        print(f"{name:<22}{desc:<40}{macro:>11.4f}{rec:>12.4f}{dm:>+9.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>18}{('YES' if sig else '-'):>5}"
              + (f"  !! {emp} empty" if emp else ""))

    macro_ndcg = {t[0]: t[2] for t in table}
    macro_rec = {t[0]: t[3] for t in table}
    b_n, b_r = macro_ndcg["baseline"], macro_rec["baseline"]

    # ---- what each component contributes, all in macro terms ---------------
    print(f"\n{'='*112}")
    print("COMPONENT CONTRIBUTIONS (macro nDCG@10 / macro R@100, vs baseline "
          f"{b_n:.4f} / {b_r:.4f})")
    print("-" * 112)

    def show(label, name, against=None):
        """One component's effect. `against` makes it a marginal contribution
        (this variant minus the one without the component); otherwise it is
        the variant's effect measured from baseline."""
        if name not in macro_ndcg or (against and against not in macro_ndcg):
            return
        ref_n = macro_ndcg[against] if against else b_n
        ref_r = macro_rec[against] if against else b_r
        print(f"  {label:<38}{macro_ndcg[name]-ref_n:>+9.4f}{macro_rec[name]-ref_r:>+11.4f}"
              + (f"   ({name} - {against})" if against else ""))

    print("  ALONE, from baseline:")
    show("drop_html", "html_only")
    show("extended stopwords", "extstop_only")
    show("min_len=2", "minlen_only")
    show("stem", "stem")

    print("\n  MARGINAL, on top of the rest of `aggressive`:")
    show("min_len=2", "aggressive", "html+stem+extstop")
    show("extended stopwords", "aggressive", "html+stem+minlen")
    show("extended vs Lucene stop list", "aggressive", "stop+stem+html")

    print("\n  MARGINAL, on top of extended stopwords (stem-free):")
    show("+ drop_html", "extstop+html", "extstop_only")
    show("+ min_len=2", "extstop+minlen", "extstop_only")
    show("+ both", "extstop+html+minlen", "extstop_only")

    # ---- the decision ------------------------------------------------------
    print(f"\n{'='*112}")
    best_n = max(macro_ndcg, key=macro_ndcg.get)
    best_r = max(macro_rec, key=macro_rec.get)
    print(f"  best macro nDCG@10 : {best_n:<24}{macro_ndcg[best_n]:.4f}  "
          f"({macro_ndcg[best_n]-b_n:+.4f} vs baseline)")
    print(f"  best macro R@100   : {best_r:<24}{macro_rec[best_r]:.4f}  "
          f"({macro_rec[best_r]-b_r:+.4f} vs baseline)")
    if best_n != best_r:
        print(f"\n  The two metrics disagree. Recall is the ceiling on any reranker,")
        print(f"  so a pipeline that ends in reranking should follow '{best_r}';")
        print(f"  a first-stage-only submission should follow '{best_n}'.")
    print(f"\n  Prefer the SIMPLEST variant whose figures are not measurably worse")
    print(f"  than the best -- fewer components is less train-split overfitting,")
    print(f"  and a shorter paragraph to defend in the system paper.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "query_form": args.query_form,
                   "domains": domains, "results": results}, f, indent=2)
    print(f"\nwrote {out}")
    print(f"total {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
