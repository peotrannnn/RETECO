#!/usr/bin/env python3
"""Paired significance test between any two tokenizer variants.

WHY THIS EXISTS
---------------
`tokenizer_experiment.py` reports a bootstrap CI for every variant against
`baseline`, which answers "is this variant better than doing nothing". That is
not the question that decides what to ship. The shipping decision is a contest
between the two or three best variants, and two CIs that both exclude zero tell
you nothing about whether they differ FROM EACH OTHER -- overlapping intervals
are routinely reported for differences that are significant, and vice versa.

Concretely, from the 13-domain focus run:

    extstop_only     macro 0.1459   vs baseline [+0.0279,+0.0410]
    extstop+minlen   macro 0.1497   vs baseline [+0.0322,+0.0463]

Both are significant against baseline. Neither figure says whether the +0.0038
between them is real. That +0.0038 is exactly what decides whether `min_len=2`
-- a component with no stated mechanism for helping temporal retrieval -- earns
a place in the shipped tokenizer or gets dropped as train-split noise.

This reads the JSON that `tokenizer_experiment.py` already wrote, so it costs
seconds rather than re-running the 19-minute experiment.

METHOD
------
The test is PAIRED: for every query scored under both variants, take the
difference in nDCG@10, then bootstrap the mean of those differences. Pairing
matters because query difficulty varies enormously here (per-query nDCG@10
ranges from 0 to 1 across a corpus where the macro figure is ~0.15); an
unpaired test would drown the effect in between-query variance.

Two numbers are reported and they answer different questions:

  macro diff   difference of the per-DOMAIN means. This is the competition's
               metric, so this is the EFFECT SIZE that decisions follow.
  CI of diff/q bootstrap over per-QUERY differences. This is the SIGNIFICANCE
               test. It is computed on a different weighting (history has 561
               queries, iota has 12), so its centre will not match the macro
               difference and is not meant to.

Read them together: the macro difference says how much, the CI says whether
the sign is trustworthy.

USAGE
-----
    python pairwise_ci.py runs/tokenizer_focus.json
    python pairwise_ci.py runs/tokenizer_focus.json --pairs extstop_only extstop+minlen
    python pairwise_ci.py runs/tokenizer_focus.json --all
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def bootstrap_ci(diffs, n=10000, seed=0):
    v = np.asarray(diffs, float)
    if not v.size:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, v.size)).mean(axis=1)
    return float(v.mean()), *(float(x) for x in np.percentile(means, [2.5, 97.5]))


def macro(results, name, metric="nDCG@10"):
    per = results[name]
    return float(np.mean([per[d][metric] for d in per]))


def paired_diffs(results, a, b, field="per_query_ndcg"):
    """Per-query differences (a - b), over queries scored under both."""
    out = []
    for d in results[a]:
        if d not in results[b]:
            continue
        pa = results[a][d].get(field)
        pb = results[b][d].get(field)
        if pa is None or pb is None:
            return None          # field absent: file written before it was stored
        out += [pa[q] - pb[q] for q in pa
                if q in pb and pa[q] is not None and pb[q] is not None]
    return out


def compare(results, a, b):
    diffs = paired_diffs(results, a, b)
    dm, lo, hi = bootstrap_ci(diffs)
    sig = lo > 0 or hi < 0
    rdiffs = paired_diffs(results, a, b, "per_query_recall")
    if rdiffs is None:
        r_sig = None            # older result file; recall cannot be tested
    else:
        _, rlo, rhi = bootstrap_ci(rdiffs)
        r_sig = rlo > 0 or rhi < 0
    return {
        "recall_significant": r_sig,
        "a": a, "b": b, "n_queries": len(diffs),
        "macro_ndcg_a": macro(results, a), "macro_ndcg_b": macro(results, b),
        "macro_ndcg_diff": macro(results, a) - macro(results, b),
        "macro_recall_a": macro(results, a, "R@100"),
        "macro_recall_b": macro(results, b, "R@100"),
        "macro_recall_diff": macro(results, a, "R@100") - macro(results, b, "R@100"),
        "per_query_diff": dm, "ci_lo": lo, "ci_hi": hi, "significant": sig,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="output of tokenizer_experiment.py")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="two variant names to compare; repeat the flag for "
                         "more pairs. Default: every variant against the one "
                         "with the best macro nDCG@10.")
    ap.add_argument("--all", action="store_true",
                    help="compare every variant against every other")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    results = data["results"]
    names = list(results)

    print(f"{Path(args.json).name} | split={data.get('split')} | "
          f"query_form={data.get('query_form')} | {len(data.get('domains', []))} domains")

    if args.pairs:
        unknown = [v for v in args.pairs if v not in results]
        if unknown:
            raise SystemExit(f"not in this file: {', '.join(unknown)}\n"
                             f"available: {', '.join(names)}")
        if len(args.pairs) % 2:
            raise SystemExit("--pairs takes variant names two at a time")
        pairs = list(zip(args.pairs[::2], args.pairs[1::2]))
    elif args.all:
        pairs = list(combinations(names, 2))
    else:
        best = max(names, key=lambda n: macro(results, n))
        pairs = [(best, n) for n in names if n != best]
        print(f"comparing everything against the best variant: {best}")

    print(f"\n{'A':<22}{'B':<22}{'macro dNDCG':>12}{'macro dR@100':>13}"
          f"{'diff/q':>9}{'CI of diff/q':>20}{'nDCG sig':>10}{'R sig':>8}")
    print("-" * 116)
    rows = []
    for a, b in pairs:
        r = compare(results, a, b)
        rows.append(r)
        ci = f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
        rs = {True: "YES", False: "-", None: "n/a"}[r["recall_significant"]]
        print(f"{a:<22}{b:<22}{r['macro_ndcg_diff']:>+12.4f}"
              f"{r['macro_recall_diff']:>+13.4f}{r['per_query_diff']:>+9.4f}"
              f"{ci:>20}{('YES' if r['significant'] else '-'):>10}{rs:>8}")

    print("\nHOW TO DECIDE")
    print("-" * 116)
    print("  A pair marked '-' is a TIE: the data cannot tell the two apart.")
    print("  Ship the SIMPLER variant of a tied pair -- fewer components means")
    print("  less train-split overfitting and a shorter claim to defend.")
    print("  Only pay for a component when its row says YES.")

    undecided = [r for r in rows if not r["significant"]]
    if undecided:
        print(f"\n  Ties in this file ({len(undecided)}):")
        for r in undecided:
            print(f"    {r['a']} vs {r['b']}   "
                  f"macro {r['macro_ndcg_diff']:+.4f} nDCG, "
                  f"{r['macro_recall_diff']:+.4f} R@100")


if __name__ == "__main__":
    main()
