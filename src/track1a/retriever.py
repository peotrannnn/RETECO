"""BM25 retriever for RETECO Track 1a -- vectorized with a scipy inverted index.

Scoring is the same Okapi BM25 the starter kit's `bm25.py` implements
(k1=0.9, b=0.4, non-negative idf, query-term-frequency weighted). The only
change is *how* it is computed: the per-(document, term) BM25 weights are
precomputed into a sparse term x document matrix, so a query only touches
the posting lists of its own terms instead of looping over every document
in Python.

Why this matters: the starter kit scores every document in a Python loop per
query. That is fine for `iota` (10k documents) and unusable for `history`
(hundreds of thousands), which running all 13 Track 1 domains requires.

Same output contract as before:
    search(query, top_k) -> [(doc_id, score), ...] sorted by score descending
"""
import hashlib
import os
import re
from array import array

import numpy as np
from scipy import sparse

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text):
    return _TOKEN.findall((text or "").lower())


def _cache_key(doc_ids, k1, b):
    """Fingerprint a (corpus, BM25 params) pair. Domain corpora are frozen
    release files, so count + first id + last id is enough to detect change.
    """
    h = hashlib.sha1()
    h.update(f"bm25|{k1}|{b}|{len(doc_ids)}".encode("utf-8"))
    if doc_ids:
        h.update(doc_ids[0].encode("utf-8"))
        h.update(doc_ids[-1].encode("utf-8"))
    return h.hexdigest()[:16]


class Retriever:
    """BM25 over an inverted index, with an on-disk index cache.

    The cache is keyed by (corpus fingerprint, k1, b), so the index built for
    a `--method bm25` run is reused by the later `--method hybrid` run on the
    same domain instead of being rebuilt.
    """

    def __init__(self, doc_ids, doc_texts, k1=0.9, b=0.4, cache_dir=None,
                 verbose=True):
        self.k1, self.b = k1, b
        self.doc_ids = list(doc_ids)
        self.N = len(self.doc_ids)

        cache_stem = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_stem = os.path.join(cache_dir, f"bm25_{_cache_key(self.doc_ids, k1, b)}")

        if cache_stem and os.path.isfile(cache_stem + ".npz") \
                and os.path.isfile(cache_stem + ".vocab.txt"):
            if verbose:
                print(f"  [bm25] loading cached index: {os.path.basename(cache_stem)}",
                      flush=True)
            self._load_cache(cache_stem)
            return

        if verbose:
            print(f"  [bm25] building index over {self.N} documents", flush=True)
        self._build(doc_texts)

        if cache_stem:
            self._save_cache(cache_stem)

    # ------------------------------------------------------------- build --
    def _build(self, doc_texts):
        vocab = {}
        # `array` rather than list: the largest domain (history) produces tens
        # of millions of nonzeros, and a Python list costs ~28 bytes per entry
        # against 4 here. This is the difference between a few hundred MB and
        # several GB of peak memory while indexing.
        indptr = array("q", [0])
        indices = array("i")
        data = array("f")
        doc_len = np.zeros(self.N, dtype=np.float32)

        for i, text in enumerate(doc_texts):
            toks = tokenize(text)
            doc_len[i] = len(toks)
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, f in counts.items():
                j = vocab.get(t)
                if j is None:
                    j = len(vocab)
                    vocab[t] = j
                indices.append(j)
                data.append(f)
            indptr.append(len(indices))

        V = len(vocab)
        tf = sparse.csr_matrix(
            (np.frombuffer(data, dtype=np.float32).copy(),
             np.frombuffer(indices, dtype=np.int32).copy(),
             np.frombuffer(indptr, dtype=np.int64).copy()),
            shape=(self.N, max(V, 1)),
        )
        del data, indices, indptr

        # df[t] = number of documents containing t. Each (doc, term) pair
        # appears exactly once in the CSR, so a bincount over the column
        # indices gives document frequency directly.
        df = np.bincount(tf.indices, minlength=max(V, 1)).astype(np.float32)
        # Same non-negative BM25 idf as the starter kit.
        idf = np.log(1.0 + (self.N - df + 0.5) / (df + 0.5)).astype(np.float32)

        avgdl = float(doc_len.mean()) if self.N else 0.0
        rows = np.repeat(np.arange(self.N, dtype=np.int64), np.diff(tf.indptr))
        f = tf.data
        denom = f + self.k1 * (1.0 - self.b + self.b * doc_len[rows] / (avgdl or 1.0))
        tf.data = (idf[tf.indices] * f * (self.k1 + 1.0) / denom).astype(np.float32)

        # Term-major layout: postings[t] is the list of documents containing t
        # together with their precomputed BM25 weight.
        postings = tf.T.tocsr()
        del tf
        self.vocab = vocab
        self.terms = sorted(vocab, key=vocab.get)
        self._p_data = postings.data
        self._p_indices = postings.indices
        self._p_indptr = postings.indptr

    # ------------------------------------------------------------- cache --
    def _save_cache(self, stem):
        np.savez(
            stem + ".npz",
            data=self._p_data,
            indices=self._p_indices,
            indptr=self._p_indptr,
            n_docs=np.int64(self.N),
        )
        # Tokens are [A-Za-z0-9]+ so they can never contain a newline.
        with open(stem + ".vocab.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.terms))

    def _load_cache(self, stem):
        z = np.load(stem + ".npz")
        self._p_data = z["data"]
        self._p_indices = z["indices"]
        self._p_indptr = z["indptr"]
        with open(stem + ".vocab.txt", "r", encoding="utf-8") as fh:
            self.terms = fh.read().split("\n") if os.path.getsize(stem + ".vocab.txt") else []
        self.vocab = {t: j for j, t in enumerate(self.terms)}

    # ------------------------------------------------------------ search --
    def search(self, query, top_k=100):
        counts = {}
        for t in tokenize(query):
            j = self.vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1

        if not counts:
            return []

        scores = np.zeros(self.N, dtype=np.float32)
        for j, c in counts.items():
            start, end = self._p_indptr[j], self._p_indptr[j + 1]
            if start == end:
                continue
            # A document appears at most once in a given term's posting list,
            # so the indices within one slice are unique and plain fancy
            # indexing accumulates correctly (and far faster than np.add.at).
            idx = self._p_indices[start:end]
            scores[idx] += c * self._p_data[start:end]

        k = min(top_k, self.N)
        if k <= 0:
            return []
        top_idx = np.argpartition(-scores, k - 1)[:k] if k < self.N else np.arange(self.N)
        top_idx = top_idx[scores[top_idx] > 0]
        if top_idx.size == 0:
            return []

        pairs = [(self.doc_ids[i], float(scores[i])) for i in top_idx]
        # Deterministic tie-break by doc id, same convention as before.
        pairs.sort(key=lambda x: (-x[1], x[0]))
        return pairs[:top_k]
