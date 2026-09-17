"""BM25 retriever for RETECO Track 1a -- vectorized with a scipy inverted index.

Scoring is the same Okapi BM25 the starter kit's `bm25.py` implements
(k1=0.9, b=0.4, non-negative idf, query-term-frequency weighted). The only
change is *how* it is computed: the per-(document, term) BM25 weights are
precomputed into a sparse term x document matrix, so a query only touches
the posting lists of its own terms instead of looping over every document
in Python.

Why this matters: the starter kit scores every document in a Python loop per
query. That is fine for `iota` (10k documents) and unusable for `history`
(356k), which running all 13 Track 1 domains requires.

**Tokenization is configurable, and it is the single largest lexical win
measured so far.** Notebook 02 swept six tokenizers over all 13 domains
(1,211 queries). Macro nDCG@10:

    baseline (lowercase + [A-Za-z0-9]+)   0.0719   <- the original
    stem            (Porter only)         0.0654   significantly WORSE
    stop+stem       (Anserini default)    0.0708   significantly WORSE
    stop_lucene                           0.0763
    stop+stem+html                        0.0937
    aggressive                            0.1154   better on 13/13 domains

Two things in that table are worth keeping in mind before changing it:

  * Stemming ON ITS OWN HURTS, and so does the Lucene/Anserini default of
    stopwords+stemming. The intuitive fix made things worse. What actually
    pays is removing HTML structural tokens: 100% of queries carry markup
    and the documents carry none, so every `p`, `href`, `li` and `code` in a
    query is a term that can only ever match noise. Stemming only helps
    *after* that noise is gone -- before it, conflation makes the noise match
    more documents, not fewer.
  * The gain is not a metric artifact. Temporal coverage (TC@10) moves with
    it, 0.1283 -> 0.2266, and the number of queries covering all of their
    required time periods goes 99 -> 174 out of 1,211.

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


# ---------------------------------------------------------------------------
# Porter stemmer
#
# Self-contained because nltk is not a dependency of this project. Validated
# against Martin Porter's own published step examples (75 word pairs covering
# all seven steps); see notebooks/02_experiments.ipynb, which runs that check
# as an assertion.
# ---------------------------------------------------------------------------
class PorterStemmer:
    """Porter (1980) suffix stripping, memoised.

    The cache matters: stemming is applied to the VOCABULARY (~10^5 terms),
    not to the token stream (~10^8 tokens), so the same word is looked up
    thousands of times.
    """

    def __init__(self):
        self._cache = {}

    @staticmethod
    def _is_cons(w, i):
        c = w[i]
        if c in "aeiou":
            return False
        if c == "y":
            return True if i == 0 else not PorterStemmer._is_cons(w, i - 1)
        return True

    @classmethod
    def _m(cls, w):
        """Measure: the number of vowel-consonant sequences in w."""
        n, i, L = 0, 0, len(w)
        while i < L and cls._is_cons(w, i):
            i += 1
        while i < L:
            while i < L and not cls._is_cons(w, i):
                i += 1
            if i >= L:
                break
            n += 1
            while i < L and cls._is_cons(w, i):
                i += 1
        return n

    @classmethod
    def _has_vowel(cls, w):
        return any(not cls._is_cons(w, i) for i in range(len(w)))

    @classmethod
    def _double_cons_end(cls, w):
        return len(w) >= 2 and w[-1] == w[-2] and cls._is_cons(w, len(w) - 1)

    @classmethod
    def _cvc(cls, w):
        if len(w) < 3:
            return False
        if not (cls._is_cons(w, len(w) - 1) and not cls._is_cons(w, len(w) - 2)
                and cls._is_cons(w, len(w) - 3)):
            return False
        return w[-1] not in "wxy"

    def _step1ab(self, w):
        if w.endswith("sses"):
            w = w[:-2]
        elif w.endswith("ies"):
            w = w[:-2]
        elif w.endswith("ss"):
            pass
        elif w.endswith("s"):
            w = w[:-1]

        if w.endswith("eed"):
            if self._m(w[:-3]) > 0:
                w = w[:-1]
        elif w.endswith("ed") and self._has_vowel(w[:-2]):
            w = self._step1b_tail(w[:-2])
        elif w.endswith("ing") and self._has_vowel(w[:-3]):
            w = self._step1b_tail(w[:-3])
        return w

    def _step1b_tail(self, s):
        if s.endswith(("at", "bl", "iz")):
            return s + "e"
        if self._double_cons_end(s) and not s.endswith(("l", "s", "z")):
            return s[:-1]
        if self._m(s) == 1 and self._cvc(s):
            return s + "e"
        return s

    def _step1c(self, w):
        return w[:-1] + "i" if w.endswith("y") and self._has_vowel(w[:-1]) else w

    _STEP2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
              ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
              ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
              ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
              ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
              ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
              ("biliti", "ble")]
    _STEP3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
              ("ical", "ic"), ("ful", ""), ("ness", "")]
    _STEP4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant",
              "ement", "ment", "ent", "ou", "ism", "ate", "iti", "ous",
              "ive", "ize"]

    def _apply(self, w, table):
        for suf, rep in table:
            if w.endswith(suf):
                stem = w[:-len(suf)]
                return stem + rep if self._m(stem) > 0 else w
        return w

    def _step2(self, w):
        return self._apply(w, self._STEP2)

    def _step3(self, w):
        return self._apply(w, self._STEP3)

    def _step4(self, w):
        for suf in sorted(self._STEP4, key=len, reverse=True):
            if w.endswith(suf):
                stem = w[:-len(suf)]
                return stem if self._m(stem) > 1 else w
        if w.endswith("ion"):
            stem = w[:-3]
            if self._m(stem) > 1 and stem.endswith(("s", "t")):
                return stem
        return w

    def _step5(self, w):
        if w.endswith("e"):
            m = self._m(w[:-1])
            if m > 1 or (m == 1 and not self._cvc(w[:-1])):
                w = w[:-1]
        if self._m(w) > 1 and self._double_cons_end(w) and w.endswith("l"):
            w = w[:-1]
        return w

    def stem(self, word):
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        w = word
        if len(w) > 2:
            for step in (self._step1ab, self._step1c, self._step2,
                         self._step3, self._step4, self._step5):
                w = step(w)
        self._cache[word] = w
        return w


STEMMER = PorterStemmer()


# ---------------------------------------------------------------------------
# Token filters
# ---------------------------------------------------------------------------

# Lucene's default English stop set (33 words), i.e. what Anserini/Pyserini
# strips by default.
LUCENE_STOPWORDS = frozenset("""
a an and are as at be but by for if in into is it no not of on or such
that the their then there these they this to was will with
""".split())

EXTENDED_STOPWORDS = LUCENE_STOPWORDS | frozenset("""
about above after again against all am been before being below between both
cannot did do does doing down during each few from further had has have
having he her here hers him his how i me more most my nor now only other
our out over own same she should so some than them themselves through too
under until up very we were what when where which while who whom why would
you your
""".split())

# HTML structural tokens. No English stop list removes these, but every query
# in this dataset carries markup and no document does, so each one is a query
# term that can only match noise. Removing them was the single biggest
# component of the measured lexical gain.
HTML_TOKENS = frozenset("""
p br div span li ul ol a href http https www img src alt h1 h2 h3 h4 h5 h6
b i em strong code pre blockquote table tr td th hr sup sub nbsp quot amp lt
gt class id style rel nofollow target blank
""".split())


# Named presets. These are exactly the variants notebook 02 measured, so a
# name here and a row in that notebook's table mean the same thing.
TOKENIZERS = {
    "baseline":       dict(),
    "stop_lucene":    dict(stopwords=LUCENE_STOPWORDS),
    "stem":           dict(stem=True),
    "stop+stem":      dict(stopwords=LUCENE_STOPWORDS, stem=True),
    "stop+stem+html": dict(stopwords=LUCENE_STOPWORDS, stem=True, drop_html=True),
    "aggressive":     dict(stopwords=EXTENDED_STOPWORDS, stem=True,
                           drop_html=True, min_len=2),
    # --- decomposition of `aggressive`, one component at a time -----------
    # `aggressive` bundles three changes and was picked as best-of-six on the
    # train split. These isolate the components so the bundle can be replaced
    # by whichever part actually carries the gain.
    "html_only":          dict(drop_html=True),
    "html+stem":          dict(drop_html=True, stem=True),
    "html+stem+extstop":  dict(drop_html=True, stem=True,
                               stopwords=EXTENDED_STOPWORDS),
    "html+stem+minlen":   dict(drop_html=True, stem=True, min_len=2),
    "extstop_only":       dict(stopwords=EXTENDED_STOPWORDS),
    "minlen_only":        dict(min_len=2),
}

DEFAULT_TOKENIZER = "aggressive"


def tokenize(text, stopwords=frozenset(), stem=False, drop_html=False,
             min_len=1, drop_pure_digits=False):
    """Tokenize with a named variant's options.

    Filter order matters and matches Lucene's: stopwords are removed BEFORE
    stemming. The other way round, `this` stems to `thi` and slips past the
    stop list.
    """
    out = []
    for t in _TOKEN.findall((text or "").lower()):
        if len(t) < min_len:
            continue
        if drop_pure_digits and t.isdigit():
            continue
        if drop_html and t in HTML_TOKENS:
            continue
        if t in stopwords:
            continue
        u = STEMMER.stem(t) if stem else t
        if u:
            out.append(u)
    return out


def _cache_key(doc_ids, k1, b, tokenizer):
    """Fingerprint a (corpus, BM25 params, tokenizer) triple.

    The tokenizer MUST be part of this. Two tokenizers produce different
    inverted indexes over the same corpus, and reusing one for the other is a
    silently wrong answer that raises no error -- the run file still validates
    and still reports a plausible score.

    Domain corpora are frozen release files, so count + first id + last id is
    enough to detect a corpus change.
    """
    h = hashlib.sha1()
    h.update(f"bm25|{k1}|{b}|{tokenizer}|{len(doc_ids)}".encode("utf-8"))
    if doc_ids:
        h.update(doc_ids[0].encode("utf-8"))
        h.update(doc_ids[-1].encode("utf-8"))
    return h.hexdigest()[:16]


class Retriever:
    """BM25 over an inverted index, with an on-disk index cache.

    The cache is keyed by (corpus fingerprint, k1, b, tokenizer), so the index
    built for a `--method bm25` run is reused by a later `--method hybrid` run
    on the same domain with the same tokenizer, and is NOT reused across
    different tokenizers.
    """

    def __init__(self, doc_ids, doc_texts, k1=0.9, b=0.4,
                 tokenizer=DEFAULT_TOKENIZER, cache_dir=None, verbose=True):
        if tokenizer not in TOKENIZERS:
            raise ValueError(
                f"unknown tokenizer {tokenizer!r}; "
                f"choose one of {sorted(TOKENIZERS)}"
            )
        self.k1, self.b = k1, b
        self.tokenizer_name = tokenizer
        self.tokenizer_opts = TOKENIZERS[tokenizer]
        self.doc_ids = list(doc_ids)
        self.N = len(self.doc_ids)

        cache_stem = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            key = _cache_key(self.doc_ids, k1, b, tokenizer)
            cache_stem = os.path.join(cache_dir, f"bm25_{key}")

        if cache_stem and os.path.isfile(cache_stem + ".npz") \
                and os.path.isfile(cache_stem + ".vocab.txt"):
            if verbose:
                print(f"  [bm25] loading cached index: "
                      f"{os.path.basename(cache_stem)} (tokenizer={tokenizer})",
                      flush=True)
            self._load_cache(cache_stem)
            return

        if verbose:
            print(f"  [bm25] building index over {self.N} documents "
                  f"(tokenizer={tokenizer})", flush=True)
        self._build(doc_texts)

        if cache_stem:
            self._save_cache(cache_stem)

    def _tokenize(self, text):
        return tokenize(text, **self.tokenizer_opts)

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
            toks = self._tokenize(text)
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
        # Tokens are [A-Za-z0-9]+ (possibly stemmed) so they can never contain
        # a newline.
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
        # The query goes through the SAME tokenizer as the corpus. This is the
        # only place queries are tokenized, so the two cannot drift apart.
        counts = {}
        for t in self._tokenize(query):
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
