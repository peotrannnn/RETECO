"""Dense (embedding-based) retriever for RETECO Track 1a.

Uses a sentence-transformers bi-encoder to embed the corpus once per domain
and caches the embeddings to disk, then does nearest-neighbor search per
query. Keeps the same output contract as retriever.Retriever:

    search(query, top_k) -> [(doc_id, score), ...] sorted by score descending

Why this exists: the RETECO dataset card states that lexical matching (BM25)
is weak on Track 1 because temporal grounding is not a keyword problem. A
dense bi-encoder captures semantic similarity that BM25 misses, and is meant
to be combined with BM25 (see hybrid_retriever.py) rather than replace it.

Requires: sentence-transformers, torch, numpy.
Optional: faiss-cpu, for fast exact search on the larger domains (history,
hsm, politics, monero, genealogy, travel all have 100k+ documents). Without
faiss this falls back to a plain numpy matmul, which is fine for the smaller
domains (iota, cardano, law, quant, ...) but noticeably slower on the big
ones.
"""
import hashlib
import os

import numpy as np


def _cache_key(model_name, doc_ids):
    """Fingerprint a (model, corpus) pair so cached embeddings are only
    reused when both match. Hashing the full id list would be safer but is
    unnecessary here: domain corpora are frozen release files, so first id +
    last id + count is enough to catch the corpus changing.
    """
    h = hashlib.sha1()
    h.update(model_name.encode("utf-8"))
    h.update(str(len(doc_ids)).encode("utf-8"))
    if doc_ids:
        h.update(doc_ids[0].encode("utf-8"))
        h.update(doc_ids[-1].encode("utf-8"))
    return h.hexdigest()[:16]


class DenseRetriever:
    """Bi-encoder retriever with an on-disk embedding cache.

    The cache is keyed by (model_name, corpus fingerprint), not by split, so
    running train then dev for the same domain re-uses the same corpus
    embeddings for free -- the corpus is identical across splits in RETECO.
    """

    def __init__(self, doc_ids, doc_texts, model_name="BAAI/bge-base-en-v1.5",
                 cache_dir=None, batch_size=64, device=None,
                 query_prefix="Represent this sentence for searching relevant passages: ",
                 passage_prefix=""):
        from sentence_transformers import SentenceTransformer

        self.doc_ids = list(doc_ids)
        self.model_name = model_name
        self.query_prefix = query_prefix or ""
        self.passage_prefix = passage_prefix or ""
        self.model = SentenceTransformer(model_name, device=device)

        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            key = _cache_key(model_name, self.doc_ids)
            cache_path = os.path.join(cache_dir, f"corpus_{key}.npy")

        if cache_path and os.path.isfile(cache_path):
            self.embeddings = np.load(cache_path)
        else:
            texts = [f"{self.passage_prefix}{t or ''}" for t in doc_texts]
            self.embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
            if cache_path:
                np.save(cache_path, self.embeddings)

        self._index = None
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self._index.add(self.embeddings)
        except ImportError:
            pass  # numpy fallback in search()

    def search(self, query, top_k=100):
        q_text = f"{self.query_prefix}{query or ''}"
        q_emb = self.model.encode(
            [q_text], normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")

        if self._index is not None:
            scores, idx = self._index.search(q_emb, top_k)
            pairs = [
                (self.doc_ids[i], float(s))
                for i, s in zip(idx[0], scores[0]) if i != -1
            ]
        else:
            sims = self.embeddings @ q_emb[0]
            k = min(top_k, len(sims))
            top_idx = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            pairs = [(self.doc_ids[i], float(sims[i])) for i in top_idx]

        # Deterministic tie-break by doc id, same convention as retriever.py.
        pairs.sort(key=lambda x: (-x[1], x[0]))
        return pairs[:top_k]
