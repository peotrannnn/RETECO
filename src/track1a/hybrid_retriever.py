"""Hybrid retriever for RETECO Track 1a: Reciprocal Rank Fusion (RRF) of a
sparse (BM25) and a dense (bi-encoder) ranking.

RRF combines two rankings using only rank position, not raw scores, which
sidesteps the score-scale mismatch between BM25 and cosine similarity
(no normalization tuning required). Formula (Cormack, Clarke & Buettcher,
2009 -- the standard RRF reference):

    rrf_score(doc) = sum over retrievers r of  1 / (k + rank_r(doc))

k=60 is the value from that paper and is a reasonable default; it mostly
controls how much weight early ranks get relative to later ones.

Keeps the same output contract as retriever.Retriever:
    search(query, top_k) -> [(doc_id, score), ...] sorted by score descending
"""


class HybridRetriever:
    def __init__(self, sparse, dense, k=60, pool_k=200):
        """
        sparse, dense: objects exposing .search(query, top_k) -> [(doc_id, score), ...]
        k:       RRF constant.
        pool_k:  how many candidates to pull from EACH retriever before
                 fusing. Should be >= top_k you plan to request from
                 .search(); 200 gives both retrievers room to disagree and
                 still fuse into a good top-100 for nDCG@10 scoring.
        """
        self.sparse = sparse
        self.dense = dense
        self.k = k
        self.pool_k = pool_k

    def search(self, query, top_k=100):
        sparse_ranked = [doc_id for doc_id, _ in self.sparse.search(query, self.pool_k)]
        dense_ranked = [doc_id for doc_id, _ in self.dense.search(query, self.pool_k)]

        rrf = {}
        for rank, doc_id in enumerate(sparse_ranked, start=1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (self.k + rank)
        for rank, doc_id in enumerate(dense_ranked, start=1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (self.k + rank)

        ranked = sorted(rrf.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k]
