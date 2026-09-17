"""Hybrid retriever for RETECO Track 1a: weighted Reciprocal Rank Fusion
(RRF) of a sparse (BM25) and a dense (bi-encoder) ranking.

RRF combines rankings using rank position only, not raw scores, which
sidesteps the score-scale mismatch between BM25 and cosine similarity:

    rrf_score(doc) = w_sparse / (k + rank_sparse(doc))
                   + w_dense  / (k + rank_dense(doc))

k=60 is the constant from the original RRF paper (Cormack, Clarke &
Buettcher, 2009).

**Why the weights are not equal by default.** Plain RRF treats both
retrievers as equally trustworthy. On this task they are not: on the `iota`
pilot domain BM25 scored nDCG@10 0.0558 against the bi-encoder's 0.2131, so
giving BM25 an equal vote injects close to noise into a much better ranking
and can push genuinely relevant documents out of the top 10. The defaults
below therefore let the dense ranking dominate while keeping BM25 as a
minority vote -- it still contributes exact lexical matches (identifiers,
version numbers, rare technical terms) that embeddings tend to blur.

These weights are a hypothesis, not a tuned result. Sweep them on the train
split (`--sparse-weight`) once the full 13-domain numbers are in.

    search(query, top_k) -> [(doc_id, score), ...] sorted by score descending
"""


class HybridRetriever:
    def __init__(self, sparse, dense, k=60, pool_k=200,
                 sparse_weight=0.3, dense_weight=1.0):
        """
        sparse, dense: objects exposing .search(query, top_k) -> [(doc_id, score), ...]
        k:             RRF constant.
        pool_k:        candidates pulled from EACH retriever before fusing.
                       Should be >= the top_k you request from .search().
        sparse_weight: vote weight for the BM25 ranking.
        dense_weight:  vote weight for the bi-encoder ranking.
        """
        self.sparse = sparse
        self.dense = dense
        self.k = k
        self.pool_k = pool_k
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

    def search(self, query, top_k=100):
        rrf = {}

        for weight, retriever in ((self.sparse_weight, self.sparse),
                                  (self.dense_weight, self.dense)):
            if weight == 0:
                continue
            ranked = retriever.search(query, self.pool_k)
            for rank, (doc_id, _score) in enumerate(ranked, start=1):
                rrf[doc_id] = rrf.get(doc_id, 0.0) + weight / (self.k + rank)

        ranked = sorted(rrf.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k]
