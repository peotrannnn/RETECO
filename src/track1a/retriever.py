"""Pure-Python BM25 retriever for RETECO Track 1a.

This intentionally mirrors the starter kit's `bm25.py` behavior:
- simple [A-Za-z0-9]+ tokenization
- Okapi BM25
- k1=0.9, b=0.4 by default
- stable score-descending ranking

Later retrieval improvements should primarily be implemented in this module
(or added as new retriever modules) while keeping the runner/output contract
unchanged.
"""

import re
from math import log

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text):
    return _TOKEN.findall((text or "").lower())


class Retriever:
    """Current Track 1a retriever: pure-Python BM25."""

    def __init__(self, doc_ids, doc_texts, k1=0.9, b=0.4):
        self.k1 = k1
        self.b = b
        self.doc_ids = list(doc_ids)
        self.docs = [tokenize(text) for text in doc_texts]

        self.N = len(self.docs)
        self.doc_len = [len(doc) for doc in self.docs]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

        self.tf = []
        df = {}

        for doc in self.docs:
            counts = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1

            self.tf.append(counts)

            for term in counts:
                df[term] = df.get(term, 0) + 1

        # Same non-negative BM25-style IDF used by the starter kit.
        self.idf = {
            term: log(1 + (self.N - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def _score(self, query_terms, doc_index):
        score = 0.0
        dl = self.doc_len[doc_index]
        tf_doc = self.tf[doc_index]

        for term in query_terms:
            if term not in tf_doc:
                continue

            f = tf_doc[term]
            denom = f + self.k1 * (
                1 - self.b + self.b * dl / (self.avgdl or 1)
            )

            score += (
                self.idf.get(term, 0.0)
                * (f * (self.k1 + 1))
                / denom
            )

        return score

    def search(self, query, top_k=100):
        """Return [(doc_id, score), ...] sorted by score descending."""
        query_terms = tokenize(query)

        scored = [
            (self.doc_ids[i], self._score(query_terms, i))
            for i in range(self.N)
        ]

        # Deterministic tie-break by doc id.
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]
