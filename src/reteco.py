"""Core code for the RETECO Track 1a project.

One file, imported by every notebook, so a fix lands in one place instead of
being copy-pasted across notebooks.

Contents
--------
    Loading      load_corpus, load_queries, load_qrels, list_domains
    Tokenising   tokenize_simple, tokenize_lucene, make_tokenizer
    Retrieval    RetrievalIndex, BM25
    Scoring      ndcg_at_k, recall_at_k, precision_at_k, f1_at_k,
                 average_precision, evaluate, macro
    Splitting    make_split, restrict
    Ceilings     rerank_ceiling
    Statistics   bootstrap_macro_diff
    Duplicates   content_key, duplicate_groups, collapse, expand_groups
    Fusion       reciprocal_rank_fusion

The BM25 here is a fast reimplementation of the starter kit's
`RETECO/starter_kit/bm25.py`. It must produce identical rankings; the notebook
verifies that rather than assuming it (see `reference_bm25_search`).
"""
from __future__ import annotations

import hashlib
import json
import re
from array import array
from math import log, log2
from pathlib import Path

import numpy as np
from scipy import sparse

__all__ = [
    "list_domains", "load_corpus", "load_queries", "load_qrels",
    "tokenize_simple", "tokenize_lucene", "make_tokenizer", "PorterStemmer",
    "LUCENE_STOPWORDS", "EXTENDED_STOPWORDS", "HTML_WORDS",
    "RetrievalIndex", "BM25", "reference_bm25_search",
    "ndcg_at_k", "recall_at_k", "precision_at_k", "f1_at_k",
    "average_precision", "evaluate", "macro",
    "make_split", "restrict", "rerank_ceiling",
    "bootstrap_macro_diff",
]


# ---------------------------------------------------------------- loading --
def list_domains(data_dir):
    """Domain names found under the Track 1 data directory."""
    return sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_corpus(path):
    """-> (doc_ids, doc_texts), in file order."""
    ids, texts = [], []
    for record in _read_jsonl(path):
        ids.append(record["id"])
        texts.append(record["content"] or "")
    return ids, texts


def load_queries(path):
    """-> [(query_id, query_text), ...].

    The query text is used verbatim, exactly as the official baseline does.
    Any reshaping of it is an experiment and belongs in the notebook, not here.
    """
    return [(r["id"], r["query"]) for r in _read_jsonl(path)]


def load_qrels(path):
    """-> {query_id: set(relevant_doc_ids)}. Only rows with relevance > 0."""
    gold = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, _, doc_id, relevance = line.split()
            if int(relevance) > 0:
                gold.setdefault(qid, set()).add(doc_id)
    return gold


# ------------------------------------------------------------ tokenising --
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize_simple(text):
    """Lowercase, then keep runs of letters and digits.

    Byte-for-byte the starter kit's `bm25.tokenize`. This is the tokeniser the
    pure-Python reference baseline uses, so it is what our BM25 is verified
    against.
    """
    return _TOKEN.findall((text or "").lower())


# Lucene's EnglishAnalyzer stop list (33 words), as used by the official
# baseline through pyserini.
LUCENE_STOPWORDS = frozenset("""
a an and are as at be but by for if in into is it no not of on or such
that the their then there these they this to was will with
""".split())


# Words that only exist because the query still carries HTML markup. The EDA
# found that ~100% of queries contain tags while ~0% of corpus documents do,
# so after `[A-Za-z0-9]+` splitting these appear on one side of the match only.
HTML_WORDS = frozenset("""
a b i p br div span li ul ol dl dt dd em strong pre code blockquote img src
alt href http https www com org rel nofollow noreferrer target blank title
table tr td th thead tbody caption h1 h2 h3 h4 h5 h6 hr sup sub del ins
nbsp quot amp lt gt apos font style class id width height align
""".split())

# A broader English stop list than Lucene's 33 words.
EXTENDED_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by cannot did do does doing down during
each few for from further had has have having he her here hers him his how i
if in into is it its me more most my no nor not of on once only or other our
out over own same she should so some such than that the their them then there
these they this those through to too under until up very was we were what
when where which while who whom why will with would you your
""".split())


class PorterStemmer:
    """Porter (1980) suffix stripping, memoised.

    Included so `tokenize_lucene` can approximate Lucene's PorterStemFilter
    without pulling in nltk. The cache matters: stemming is applied to the
    vocabulary (~10^5 terms), not to the token stream (~10^8 tokens).
    """

    def __init__(self):
        self._cache = {}

    @staticmethod
    def _is_consonant(word, i):
        c = word[i]
        if c in "aeiou":
            return False
        if c == "y":
            return True if i == 0 else not PorterStemmer._is_consonant(word, i - 1)
        return True

    @classmethod
    def _measure(cls, word):
        """Number of vowel-consonant sequences."""
        n, i, length = 0, 0, len(word)
        while i < length and cls._is_consonant(word, i):
            i += 1
        while i < length:
            while i < length and not cls._is_consonant(word, i):
                i += 1
            if i >= length:
                break
            n += 1
            while i < length and cls._is_consonant(word, i):
                i += 1
        return n

    @classmethod
    def _has_vowel(cls, word):
        return any(not cls._is_consonant(word, i) for i in range(len(word)))

    @classmethod
    def _double_consonant_end(cls, word):
        return (len(word) >= 2 and word[-1] == word[-2]
                and cls._is_consonant(word, len(word) - 1))

    @classmethod
    def _cvc(cls, word):
        if len(word) < 3:
            return False
        if not (cls._is_consonant(word, len(word) - 1)
                and not cls._is_consonant(word, len(word) - 2)
                and cls._is_consonant(word, len(word) - 3)):
            return False
        return word[-1] not in "wxy"

    def _step1ab(self, w):
        if w.endswith("sses") or w.endswith("ies"):
            w = w[:-2]
        elif w.endswith("ss"):
            pass
        elif w.endswith("s"):
            w = w[:-1]

        if w.endswith("eed"):
            if self._measure(w[:-3]) > 0:
                w = w[:-1]
        elif w.endswith("ed") and self._has_vowel(w[:-2]):
            w = self._step1b_tail(w[:-2])
        elif w.endswith("ing") and self._has_vowel(w[:-3]):
            w = self._step1b_tail(w[:-3])
        return w

    def _step1b_tail(self, s):
        if s.endswith(("at", "bl", "iz")):
            return s + "e"
        if self._double_consonant_end(s) and not s.endswith(("l", "s", "z")):
            return s[:-1]
        if self._measure(s) == 1 and self._cvc(s):
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
        for suffix, replacement in table:
            if w.endswith(suffix):
                stem = w[:-len(suffix)]
                return stem + replacement if self._measure(stem) > 0 else w
        return w

    def _step4(self, w):
        for suffix in sorted(self._STEP4, key=len, reverse=True):
            if w.endswith(suffix):
                stem = w[:-len(suffix)]
                return stem if self._measure(stem) > 1 else w
        if w.endswith("ion"):
            stem = w[:-3]
            if self._measure(stem) > 1 and stem.endswith(("s", "t")):
                return stem
        return w

    def _step5(self, w):
        if w.endswith("e"):
            m = self._measure(w[:-1])
            if m > 1 or (m == 1 and not self._cvc(w[:-1])):
                w = w[:-1]
        if (self._measure(w) > 1 and self._double_consonant_end(w)
                and w.endswith("l")):
            w = w[:-1]
        return w

    def stem(self, word):
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        w = word
        if len(w) > 2:
            w = self._step1ab(w)
            w = self._step1c(w)
            w = self._apply(w, self._STEP2)
            w = self._apply(w, self._STEP3)
            w = self._step4(w)
            w = self._step5(w)
        self._cache[word] = w
        return w


STEMMER = PorterStemmer()

_POSSESSIVE = re.compile(r"'s\b|'s\b", re.IGNORECASE)


def tokenize_lucene(text):
    """Approximate Lucene's EnglishAnalyzer, which the official baseline uses.

    Lucene's chain is: StandardTokenizer, EnglishPossessiveFilter,
    LowerCaseFilter, StopFilter (the 33-word list), PorterStemFilter.

    This is an APPROXIMATION, not a reimplementation: StandardTokenizer
    follows the Unicode segmentation spec, which this regex does not. Whether
    the approximation is close enough is a question to measure, not assume —
    notebook 03 measures it against the published per-domain figures.
    """
    text = _POSSESSIVE.sub("", (text or "").lower())
    return [STEMMER.stem(t) for t in _TOKEN.findall(text)
            if t not in LUCENE_STOPWORDS]


def make_tokenizer(stopwords=frozenset(), drop_html=False, stem=False,
                   min_length=1):
    """Build a tokeniser from switches, for the preprocessing experiments.

    Every variant starts from `tokenize_simple` so the only thing that changes
    between two runs is the switch being tested.

    min_length is worth understanding before using it. It does not mean "short
    words carry little meaning". The `[A-Za-z0-9]+` split breaks on apostrophes
    and punctuation, and min_length=2 discards the fragments that leaves
    behind: `s` out of possessives and plurals, `t` out of contractions, and
    bare digits out of dates. Those fragments are very common in the corpus, so
    they add a small score to a large share of documents.
    """
    def tokenizer(text):
        tokens = _TOKEN.findall((text or "").lower())
        if drop_html:
            tokens = [t for t in tokens if t not in HTML_WORDS]
        if stopwords:
            tokens = [t for t in tokens if t not in stopwords]
        if min_length > 1:
            tokens = [t for t in tokens if len(t) >= min_length]
        if stem:
            tokens = [STEMMER.stem(t) for t in tokens]
        return tokens

    return tokenizer


# ------------------------------------------------------------- retrieval --
class RetrievalIndex:
    """One inverted index, five classical ways of scoring a document with it.

    Every classical retrieval model compared in this project reduces to the
    same two steps:

        1. count how often each term occurs in each document  -> the index
        2. turn those counts into weights and add up the weights of the
           query's terms                                      -> the model

    Step 1 is identical for all of them and is by far the expensive one, so it
    runs once per corpus. Step 2 is a transform of the count matrix's `data`
    array and is cheap. `set_weighting` therefore switches between models
    without re-reading the corpus, which is what makes comparing five models
    over thirteen domains practical.

    Weightings
    ----------
    boolean_and  A document qualifies only if it contains EVERY query term.
                 Returns a SET, not a ranking: qualifying documents come back
                 in document-id order because the model gives no reason to
                 prefer one over another. A query term that occurs nowhere in
                 the corpus makes the whole conjunction empty.
    boolean_or   Score = how many distinct query terms the document contains.
                 A ranking, but a coarse one: scores are small integers, so
                 large blocks of documents tie.
    tf           Score = sum over query terms of 1 + log(term frequency).
                 Rewards repetition and nothing else: a term occurring in every
                 document counts exactly as much as a rare one.
    tfidf        Classical vector space model, cosine similarity. Document
                 weight = (1 + log f) * log(N / df), then each document vector
                 is scaled to unit length. Rare terms now count more, and long
                 documents stop winning merely by being long.
    bm25         Okapi BM25. Like tfidf, but term frequency saturates (k1) and
                 the length correction is tunable (b).
    qlm          Query likelihood with Dirichlet smoothing. Each document is a
                 language model; the score is the probability that model would
                 generate the query. A different family from the three above:
                 it estimates probabilities rather than weighting terms.

    All five share the tie-breaking rule in `_top_k`, so a measured difference
    between two of them is a difference in the model and not in how ties
    happened to fall.

    Weights are held in float64, not float32. float32 agrees with the reference
    to about 1e-5, which sounds harmless and is not: documents whose true
    scores are equal or nearly equal get ordered differently by the rounding,
    and on a 10k-document corpus that was enough to change the ranking of every
    query tested.
    """

    WEIGHTINGS = ("boolean_and", "boolean_or", "tf", "tfidf", "bm25", "qlm")
    BLOCK = 50_000            # rows per block when row indices are needed

    def __init__(self, doc_ids, doc_texts, tokenizer=tokenize_simple,
                 weighting="bm25", k1=0.9, b=0.4, mu=2000.0,
                 keep_counts=True):
        self.doc_ids = list(doc_ids)
        self.tokenizer = tokenizer
        self.k1, self.b, self.mu = k1, b, float(mu)
        self.N = len(self.doc_ids)
        self.weighting = None
        self._weight_key = None
        self._idf = None
        self._qlm_bias = None

        # --- one pass: raw term frequencies into a document x term CSR matrix
        vocab = {}
        indptr = array("q", [0])
        indices = array("i")
        data = array("d")          # float64: see the note on precision above
        doc_len = array("i")

        for text in doc_texts:
            counts = {}
            n_tokens = 0
            for token in tokenizer(text):
                counts[token] = counts.get(token, 0) + 1
                n_tokens += 1
            doc_len.append(n_tokens)
            for token, freq in counts.items():
                j = vocab.get(token)
                if j is None:
                    j = len(vocab)
                    vocab[token] = j
                indices.append(j)
                data.append(freq)
            indptr.append(len(indices))

        self.vocab = vocab
        self._counts = sparse.csr_matrix(
            (np.frombuffer(data, dtype=np.float64),
             np.frombuffer(indices, dtype=np.int32),
             np.frombuffer(indptr, dtype=np.int64)),
            shape=(self.N, len(vocab)))

        self.doc_len = np.frombuffer(doc_len, dtype=np.int32).astype(np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0

        # Document frequency. Each (document, term) pair is stored exactly once,
        # so counting how often a column index appears IS the df. Identical to
        # converting the matrix to CSC and differencing its indptr, but without
        # building a second copy of the whole matrix.
        self.df = np.bincount(self._counts.indices,
                              minlength=len(vocab)).astype(np.float64)

        # Rank of each document in doc-id order, used to break score ties the
        # same way the reference implementation does (it sorts by
        # (-score, doc_id)).
        self._id_rank = np.empty(self.N, dtype=np.int64)
        self._id_rank[np.argsort(np.array(self.doc_ids, dtype=object),
                                 kind="stable")] = np.arange(self.N)

        self.set_weighting(weighting)
        if not keep_counts:
            self._counts = None

    # ------------------------------------------------------------- building --
    def _row_blocks(self, matrix):
        """Yield (lo, hi, row_index_array) over blocks of rows.

        Several weightings need the row (document) index of every stored value.
        Materialising that array for the whole matrix at once peaks in the
        gigabytes on the largest domain, so it is built a block at a time.
        """
        for start in range(0, self.N, self.BLOCK):
            stop = min(start + self.BLOCK, self.N)
            lo, hi = matrix.indptr[start], matrix.indptr[stop]
            if lo == hi:
                continue
            rows = np.repeat(np.arange(start, stop, dtype=np.int64),
                             np.diff(matrix.indptr[start:stop + 1]))
            yield lo, hi, rows

    def _normalise_rows(self, matrix):
        """Scale every document vector to unit length, for cosine similarity.

        Without this the vector space model degenerates into "longest document
        wins", because a longer document simply has more non-zero weights to
        add up.
        """
        data = matrix.data
        row_norm = np.zeros(self.N, dtype=np.float64)
        for start in range(0, self.N, self.BLOCK):
            stop = min(start + self.BLOCK, self.N)
            lo, hi = matrix.indptr[start], matrix.indptr[stop]
            if lo == hi:
                continue
            cumulative = np.concatenate(([0.0], np.cumsum(data[lo:hi] ** 2)))
            bounds = matrix.indptr[start:stop + 1] - lo
            row_norm[start:stop] = cumulative[bounds[1:]] - cumulative[bounds[:-1]]

        row_norm = np.sqrt(np.maximum(row_norm, 0.0))
        safe = np.where(row_norm > 0, row_norm, 1.0)
        inverse = np.where(row_norm > 0, 1.0 / safe, 0.0)
        for lo, hi, rows in self._row_blocks(matrix):
            data[lo:hi] *= inverse[rows]

    def set_weighting(self, weighting, k1=None, b=None, use_idf=True,
                      mu=None):
        """Rebuild the weight matrix for another model. Returns self.

        `k1` and `b` override the values given to the constructor and affect
        only bm25. `use_idf=False` drops the idf factor from bm25 or tfidf --
        not a sensible retrieval model, but the way to measure what idf alone
        is worth by taking it out and rerunning.

        Two limits of `k1` are worth knowing, because they turn bm25 into
        other models without changing any other code:

            k1 -> 0    term frequency stops mattering entirely; the weight
                       becomes idf alone, i.e. a weighted boolean model
            k1 -> inf  saturation disappears; the weight becomes proportional
                       to raw term frequency, i.e. length-normalised tf-idf

        So sweeping `k1` is not only tuning: it moves along a line between
        three classical models, and the best point on that line is an answer
        about this corpus rather than about bm25.
        """
        if weighting not in self.WEIGHTINGS:
            raise ValueError(f"unknown weighting {weighting!r}; "
                             f"expected one of {self.WEIGHTINGS}")
        k1 = self.k1 if k1 is None else float(k1)
        b = self.b if b is None else float(b)
        mu = self.mu if mu is None else float(mu)
        key = (weighting, k1, b, bool(use_idf), mu)
        if key == self._weight_key:
            return self
        if self._counts is None:
            raise RuntimeError("term counts were released (keep_counts=False); "
                               "rebuild the index to change weighting")

        self.k1, self.b, self.mu = k1, b, mu
        weights = self._counts.copy()
        f = weights.data

        if weighting in ("boolean_and", "boolean_or"):
            f[:] = 1.0
        elif weighting == "tf":
            f[:] = 1.0 + np.log(f)
        elif weighting == "tfidf":
            # Classical vector space idf. Deliberately NOT BM25's idf: they are
            # different formulas and the point of this class is to compare the
            # models as they are defined, not a hybrid of them.
            self._idf = (np.log(self.N / np.maximum(self.df, 1.0)) if use_idf
                         else np.ones(len(self.df), dtype=np.float64))
            f[:] = (1.0 + np.log(f)) * self._idf[weights.indices]
            self._normalise_rows(weights)
        elif weighting == "qlm":
            # log P(q|d) with Dirichlet smoothing splits into a part that only
            # touches terms the document actually contains, and a part that is
            # the same for every query term:
            #
            #   score = SUM(t in q, tf>0) qtf * log(1 + tf / (mu * p(t|C)))
            #           - |q| * log(|d| + mu)
            #
            # The first line is a sparse matrix exactly like the others. The
            # second is one number per document, applied in `search`, because
            # it scales with the query length rather than with any term.
            collection = np.bincount(
                self._counts.indices,
                weights=self._counts.data,
                minlength=len(self.df)).astype(np.float64)
            total = collection.sum() or 1.0
            probability = np.maximum(collection / total, 1e-12)
            f[:] = np.log1p(f / (self.mu * probability[weights.indices]))
            self._qlm_bias = -np.log(self.doc_len + self.mu)
        else:                                            # bm25
            idf = (np.log(1.0 + (self.N - self.df + 0.5) / (self.df + 0.5))
                   if use_idf else np.ones(len(self.df), dtype=np.float64))
            norm = k1 * (1 - b + b * self.doc_len / (self.avgdl or 1.0))
            for lo, hi, rows in self._row_blocks(weights):
                block = f[lo:hi]
                f[lo:hi] = (idf[weights.indices[lo:hi]] * block
                            * (k1 + 1.0) / (block + norm[rows]))

        self.weighting = weighting
        self._weight_key = key
        self.index = weights.T.tocsr()      # term-major, so a query slices rows
        return self

    # ------------------------------------------------------------ searching --
    def search(self, query, top_k=100):
        """-> [(doc_id, score), ...] best first, ties broken by document id."""
        terms = self.tokenizer(query)
        if not terms or not self.N:
            return []

        if self.weighting in ("boolean_and", "boolean_or"):
            # Boolean models see a query as a set of terms; repetition carries
            # no meaning in them.
            distinct = sorted(set(terms))
            columns = [self.vocab[t] for t in distinct if t in self.vocab]
            if not columns:
                return []
            if self.weighting == "boolean_and" and len(columns) < len(distinct):
                # A term occurring nowhere in the corpus makes the conjunction
                # empty. That is the model's real behaviour, not a gap to patch.
                return []
            rows = np.array(columns, dtype=np.int64)
            multipliers = np.ones(rows.size, dtype=np.float64)
        else:
            # Query term frequency is counted: a term appearing twice in the
            # query contributes twice. The reference loops over the token list,
            # not a set.
            counts = {}
            for term in terms:
                j = self.vocab.get(term)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            if not counts:
                return []
            rows = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
            raw = np.fromiter(counts.values(), dtype=np.float64,
                              count=len(counts))
            if self.weighting == "tfidf":
                multipliers = (1.0 + np.log(raw)) * self._idf[rows]
                length = float(np.linalg.norm(multipliers))
                if length > 0:
                    multipliers = multipliers / length
            else:
                multipliers = raw

        scores = np.asarray(
            (sparse.csr_matrix(multipliers[None, :]) @ self.index[rows]).todense()
        ).ravel()

        if self.weighting == "qlm":
            # The -|q| * log(|d| + mu) part. It depends on the document and on
            # how many query terms there are, so it cannot live in the matrix.
            scores = scores + float(multipliers.sum()) * self._qlm_bias

        if self.weighting == "boolean_and":
            # Keep only documents matching every query term, then return them
            # in document-id order. Boolean retrieval produces a set; any order
            # imposed on it is arbitrary, and saying so is more honest than
            # inventing a score.
            required = float(rows.size)
            keep = np.flatnonzero(scores >= required - 0.5)
            if keep.size == 0:
                return []
            order = keep[np.argsort(self._id_rank[keep], kind="stable")]
            return [(self.doc_ids[i], float(scores[i])) for i in order[:top_k]]

        return self._top_k(scores, top_k)

    def _top_k(self, scores, top_k):
        """Top-k by (score desc, doc id asc) -- ties resolved exactly.

        The tie handling is not a detail. Duplicate documents score
        identically, and roughly 30% of this corpus is duplicated text, so the
        top-k cut routinely lands in the middle of a block of equal scores: on
        `iota`, one query had 92 documents tied at the cut. A plain
        `argpartition` keeps an arbitrary subset of that block, which silently
        disagrees with the reference implementation even when every score
        matches to 1e-13.

        Boolean OR makes this sharper still: its scores are small integers, so
        almost every cut lands inside a tie.
        """
        k = min(top_k, self.N)
        if k <= 0:
            return []

        # Everything strictly above the k-th score is in; the tied block at
        # the k-th score is then filled by smallest doc id, as the reference
        # does by sorting the whole list.
        candidates = (np.argpartition(-scores, k - 1)[:k] if k < self.N
                      else np.arange(self.N))
        threshold = scores[candidates].min()

        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        remaining = k - above.size
        if remaining <= 0:
            chosen = above
        elif tied.size > remaining:
            keep = np.argpartition(self._id_rank[tied], remaining - 1)[:remaining]
            chosen = np.concatenate([above, tied[keep]])
        else:
            chosen = np.concatenate([above, tied])

        order = np.lexsort((self._id_rank[chosen], -scores[chosen]))
        chosen = chosen[order][:k]
        return [(self.doc_ids[i], float(scores[i])) for i in chosen]


class BM25(RetrievalIndex):
    """Okapi BM25 over a sparse inverted index.

    Same formula as the starter kit's `bm25.py` (k1=0.9, b=0.4, non-negative
    idf, query term frequency counted), but the per-(document, term) weights
    are precomputed into a term x document sparse matrix. A query then touches
    only the posting lists of its own terms instead of looping over the whole
    corpus in Python.

    That difference is what makes 13 domains feasible: the reference
    implementation scores every document for every query, which is fine for
    `iota` (10k documents) and impractical for `history` (356k).

    Kept as a class of its own so the notebooks written before `RetrievalIndex`
    existed still read naturally. It releases the raw counts once the weights
    are built, which is why its weighting cannot be switched afterwards.
    """

    def __init__(self, doc_ids, doc_texts, tokenizer=tokenize_simple,
                 k1=0.9, b=0.4):
        super().__init__(doc_ids, doc_texts, tokenizer=tokenizer,
                         weighting="bm25", k1=k1, b=b, keep_counts=False)


def reference_bm25_search(doc_ids, doc_texts, query, top_k=100, k1=0.9, b=0.4):
    """Literal transcription of the starter kit's `bm25.py`, for verification.

    Deliberately slow and unoptimised: its only job is to be obviously the same
    algorithm as the reference, so the fast BM25 above can be checked against
    something that has not been rewritten for speed.
    """
    docs = [tokenize_simple(t) for t in doc_texts]
    n_docs = len(docs)
    lengths = [len(d) for d in docs]
    avgdl = (sum(lengths) / n_docs) if n_docs else 0.0

    term_freqs, df = [], {}
    for doc in docs:
        counts = {}
        for word in doc:
            counts[word] = counts.get(word, 0) + 1
        term_freqs.append(counts)
        for word in counts:
            df[word] = df.get(word, 0) + 1
    idf = {w: log(1 + (n_docs - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    q_terms = tokenize_simple(query)
    scored = []
    for i in range(n_docs):
        score, dl = 0.0, lengths[i]
        counts = term_freqs[i]
        for word in q_terms:
            if word not in counts:
                continue
            f = counts[word]
            denominator = f + k1 * (1 - b + b * dl / (avgdl or 1))
            score += idf.get(word, 0.0) * (f * (k1 + 1)) / denominator
        scored.append((doc_ids[i], score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:top_k]


# --------------------------------------------------------------- scoring --
def ndcg_at_k(ranked_ids, gold_ids, k=10):
    """nDCG@k with binary relevance, matching the official metric."""
    ideal = sum(1 / log2(i + 2) for i in range(min(len(gold_ids), k)))
    if ideal <= 0:
        return 0.0
    actual = sum(1 / log2(i + 2)
                 for i, doc_id in enumerate(ranked_ids[:k]) if doc_id in gold_ids)
    return actual / ideal


def recall_at_k(ranked_ids, gold_ids, k=100):
    if not gold_ids:
        return None
    return len(set(ranked_ids[:k]) & set(gold_ids)) / len(gold_ids)


def precision_at_k(ranked_ids, gold_ids, k=10):
    """Share of the first k results that are relevant."""
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return len(set(top) & set(gold_ids)) / len(top)


def f1_at_k(ranked_ids, gold_ids, k=10):
    """Harmonic mean of precision and recall at k.

    This is the set-based view: the first k results are treated as one
    retrieved set and their order inside it is ignored.
    """
    precision = precision_at_k(ranked_ids, gold_ids, k)
    recall = recall_at_k(ranked_ids, gold_ids, k)
    if not recall or precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_precision(ranked_ids, gold_ids, k=None):
    """Average precision for one query; the mean over queries is MAP.

    Precision is read off at every rank that holds a relevant document, and
    those values are averaged over the number of relevant documents. A relevant
    document that is never retrieved contributes 0, which is why the divisor is
    the gold count and not the number found.
    """
    if not gold_ids:
        return None
    gold = set(gold_ids)
    ranked = ranked_ids[:k] if k else ranked_ids
    hits, total = 0, 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in gold:
            hits += 1
            total += hits / rank
    return total / len(gold)


def evaluate(run, gold, k=10, recall_k=100):
    """Score one domain.

    `run` is {query_id: [doc_id, ...]}, `gold` is {query_id: set(doc_id)}.

    Only queries present in BOTH are scored. That is the official rule
    (`scorer.py`, `qids = [q for q in gold if q in run]`), and it means a query
    the system returned nothing for is DROPPED from the mean rather than scored
    zero — so a system that answers fewer queries can score higher. `n_topics`
    is returned alongside the scores for exactly that reason; a score without
    it cannot be compared to another score.
    """
    qids = [q for q in gold if q in run]
    per_query_ndcg = {q: ndcg_at_k(run[q], gold[q], k) for q in qids}
    per_query_recall = {q: recall_at_k(run[q], gold[q], recall_k) for q in qids}
    per_query_ap = {q: average_precision(run[q], gold[q]) for q in qids}
    valid_recall = [v for v in per_query_recall.values() if v is not None]
    valid_ap = [v for v in per_query_ap.values() if v is not None]

    def mean_of(function, **kwargs):
        values = [function(run[q], gold[q], **kwargs) for q in qids]
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else 0.0

    return {
        # The official metric, and the one every earlier notebook reports.
        "ndcg": float(np.mean(list(per_query_ndcg.values()))) if qids else 0.0,
        "recall": float(np.mean(valid_recall)) if valid_recall else 0.0,
        # Set-based view of the first k results: order inside them ignored.
        "precision": mean_of(precision_at_k, k=k),
        "f1": mean_of(f1_at_k, k=k),
        "recall_at_cut": mean_of(recall_at_k, k=k),
        # Rank-aware, over the whole returned list.
        "map": float(np.mean(valid_ap)) if valid_ap else 0.0,
        "n_topics": len(qids),
        "n_gold_queries": len(gold),
        "per_query_ndcg": per_query_ndcg,
        "per_query_recall": per_query_recall,
        "per_query_ap": per_query_ap,
    }


def macro(results, metric="ndcg"):
    """Mean over domains, each weighted equally.

    This is the competition's aggregation. `results` is {domain: evaluate(...)}.
    Note what it is NOT: a mean over queries. `history` holds 561 of the 1,211
    train queries but, as one domain of thirteen, 7.7% of this number.
    """
    return float(np.mean([r[metric] for r in results.values()]))


# ------------------------------------------------------- train/check split --
def make_split(queries_by_domain, fit_fraction=0.8, seed=0):
    """Split the train queries into a fitting part and a checking part.

    Why this exists: running many experiments on one set and keeping the best
    result fits the noise of that set. `dev` is a single held-out check at the
    very end and cannot be used along the way, so the safety net has to live
    inside `train`.

    The split is stratified by domain, so both halves contain all 13 domains.
    It is also deterministic: the same seed gives the same split, which matters
    because every later notebook has to use the identical partition or the
    comparisons stop being paired.

    -> {domain: {"fit": [query_id, ...], "check": [query_id, ...]}}
    """
    rng = np.random.default_rng(seed)
    split = {}
    for domain, query_ids in queries_by_domain.items():
        ids = sorted(query_ids)                 # sorted first: dict order must not leak in
        order = rng.permutation(len(ids))
        cut = max(1, int(round(len(ids) * fit_fraction)))
        # A domain with very few queries still needs at least one in each part
        cut = min(cut, len(ids) - 1) if len(ids) > 1 else len(ids)
        split[domain] = {
            "fit": sorted(ids[i] for i in order[:cut]),
            "check": sorted(ids[i] for i in order[cut:]),
        }
    return split


def restrict(results, split, part):
    """Re-score an existing result over only one part of the split.

    An experiment is run once over all of train; this then reads off the
    fitting score and the checking score from that single run. Running the
    experiment twice would be wasteful and, worse, would leave room for the
    two runs to differ in some other way.
    """
    out = {}
    for domain, scored in results.items():
        keep = set(split.get(domain, {}).get(part, []))
        ndcg = {q: v for q, v in scored["per_query_ndcg"].items() if q in keep}
        recall = {q: v for q, v in scored["per_query_recall"].items() if q in keep}
        valid_recall = [v for v in recall.values() if v is not None]
        out[domain] = {
            "ndcg": float(np.mean(list(ndcg.values()))) if ndcg else 0.0,
            "recall": float(np.mean(valid_recall)) if valid_recall else 0.0,
            "n_topics": len(ndcg),
            "per_query_ndcg": ndcg,
            "per_query_recall": recall,
        }
    return out


def rerank_ceiling(ranked_ids, gold_ids, k=10):
    """nDCG@k a perfect reranker would reach over this candidate list.

    Reranking reorders a fixed shortlist; it cannot add a document the first
    stage never retrieved. So the best any reranker can do is lift every gold
    document that IS in the list up to the top. This computes that number
    exactly -- it is a bound, not an estimate.
    """
    found = sum(1 for doc_id in ranked_ids if doc_id in gold_ids)
    ideal = sum(1 / log2(i + 2) for i in range(min(len(gold_ids), k)))
    if ideal <= 0:
        return 0.0
    best = sum(1 / log2(i + 2) for i in range(min(found, k)))
    return best / ideal


# ------------------------------------------------------------ statistics --
def bootstrap_macro_diff(results_a, results_b, metric="per_query_ndcg",
                         n_resamples=10_000, seed=0):
    """Paired bootstrap confidence interval on the difference of macro scores.

    Queries are resampled WITHIN each domain and the domain set is held fixed,
    because the competition has exactly these 13 domains and is not sampling
    them from a population. Every draw recomputes each domain's mean and then
    averages those means, which is how the score is actually formed.

    Resampling all queries in one undifferentiated pool would answer a
    different question, and the two can disagree: a change that helps `history`
    and hurts ten small domains looks good pooled and bad under the real
    metric.

    -> (point_estimate, ci_low, ci_high, significant)
    """
    rng = np.random.default_rng(seed)
    per_domain = []
    for domain, a in results_a.items():
        b = results_b.get(domain)
        if b is None:
            continue
        shared = [q for q in a[metric] if q in b[metric]]
        diffs = np.asarray(
            [a[metric][q] - b[metric][q] for q in shared
             if a[metric][q] is not None and b[metric][q] is not None],
            dtype=float)
        if diffs.size:
            per_domain.append(diffs)

    if not per_domain:
        return 0.0, 0.0, 0.0, False

    draws = np.empty((n_resamples, len(per_domain)))
    for i, diffs in enumerate(per_domain):
        draws[:, i] = rng.choice(diffs, size=(n_resamples, diffs.size)).mean(axis=1)
    macro_draws = draws.mean(axis=1)

    point = float(np.mean([d.mean() for d in per_domain]))
    low, high = (float(x) for x in np.percentile(macro_draws, [2.5, 97.5]))
    return point, low, high, (low > 0 or high < 0)


# ------------------------------------------------- duplicate text groups --
def content_key(text):
    """Stable key for "these two documents are the same text".

    Whitespace is collapsed and case is folded first, so a copy that differs
    only in line wrapping still lands in the same group. The EDA in notebook
    01a used a raw byte hash; this is slightly looser and finds a few more
    copies, which is the safer direction for a stage whose job is to stop
    copies from eating slots.
    """
    return hashlib.sha1(
        re.sub(r"\s+", " ", (text or "")).strip().lower().encode("utf-8")
    ).digest()


def duplicate_groups(doc_ids, doc_texts):
    """-> {doc_id: canonical_id} for documents that have at least one twin.

    A document with no twin is absent from the map; callers treat "absent"
    as "its own canonical". Storing only the duplicated documents keeps the
    map at roughly a third of the corpus rather than all of it.

    The canonical is the first id in corpus order. Which member is canonical
    is arbitrary and, importantly, is NOT a guess at which member the qrels
    marked relevant -- see `expand_groups` for why that matters.
    """
    first = {}
    members = {}
    for doc_id, text in zip(doc_ids, doc_texts):
        key = content_key(text)
        if key in first:
            members.setdefault(first[key], [first[key]]).append(doc_id)
        else:
            first[key] = doc_id
    return {member: canonical
            for canonical, group in members.items()
            for member in group}


def collapse(ranked_ids, canonical_of):
    """Keep the first id of each text, in rank order.

    Returns (kept, twins): `kept` is the shortened list, `twins` maps each
    kept id to the ids that were dropped behind it, in the order they were
    ranked. `twins` is what makes the collapse reversible.
    """
    kept, twins, seen = [], {}, {}
    for doc_id in ranked_ids:
        canonical = canonical_of.get(doc_id, doc_id)
        if canonical in seen:
            twins.setdefault(seen[canonical], []).append(doc_id)
        else:
            seen[canonical] = doc_id
            kept.append(doc_id)
    return kept, twins


def expand_groups(ranked_ids, twins, max_per_group=1, limit=None):
    """Put dropped twins back, at most `max_per_group` ids per text.

    This exists because the qrels do not always mark every copy of a text.
    Measured on `quant`, five duplicate groups have some copies marked
    relevant and some not, so returning the wrong member of a group scores
    zero on a text that is word-for-word the answer.

    max_per_group = 1   trust the collapse; every slot holds a different text
    max_per_group = n   hedge; spend up to n slots covering one text's ids

    Which is better is an empirical question about this corpus, and notebook
    07a measures it rather than assuming.
    """
    out = []
    for doc_id in ranked_ids:
        out.append(doc_id)
        if max_per_group > 1:
            out.extend(twins.get(doc_id, [])[:max_per_group - 1])
        if limit and len(out) >= limit:
            return out[:limit]
    return out[:limit] if limit else out


# ------------------------------------------------------------- fusion --
def reciprocal_rank_fusion(runs, k=60, top_k=None):
    """Merge several ranked lists of the same query into one.

    score(d) = SUM over runs of 1 / (k + rank(d))

    Rank, not score, is what gets combined. Two retrieval models can put
    their scores on completely different scales -- BM25 sums unbounded term
    weights while Query Likelihood sums log-probabilities that are always
    negative -- so adding or averaging their scores compares nothing. Ranks
    are on the same scale by construction.

    `k` damps the top of each list. With k = 60 the gap between rank 1 and
    rank 2 is small enough that a document needs support from more than one
    run to reach the top, which is the entire point of fusing.
    """
    totals = {}
    for ranked_ids in runs:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            totals[doc_id] = totals.get(doc_id, 0.0) + 1.0 / (k + rank)
    order = sorted(totals, key=lambda d: (-totals[d], d))
    return order[:top_k] if top_k else order
