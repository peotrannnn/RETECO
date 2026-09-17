"""Dense (embedding-based) retriever for RETECO Track 1a.

Uses a sentence-transformers bi-encoder to embed the corpus once per domain,
caches the embeddings to disk, then does nearest-neighbor search per query.

    search(query, top_k) -> [(doc_id, score), ...] sorted by score descending

Why this exists: the RETECO dataset card states that lexical matching (BM25)
is weak on Track 1 because temporal grounding is not a keyword problem. A
dense bi-encoder captures semantic similarity BM25 misses.

Throughput notes -- these matter a lot when encoding all 13 Track 1 domains
(1.65M documents total):
  * fp16 halves encode time on GPU with no measurable retrieval-quality cost.
  * `max_seq_length` is the dominant cost factor. The default here (256) is
    half the model's 512 limit: it roughly doubles throughput, at the cost of
    truncating long documents earlier. Stack Exchange posts front-load their
    content, so the loss is usually small -- but raise it to 512 if you have
    the GPU budget and want to rule that out.
  * Embeddings are cached as float16 on disk (halves cache size: the full
    Track 1 corpus is ~2.5GB cached at fp16 vs ~5GB at fp32) and converted
    back to float32 at load time, which is what faiss needs.

Requires: sentence-transformers, torch, numpy.
Optional: faiss-cpu, for fast exact search. Without it this falls back to a
numpy matmul, which is correct but slower on the larger domains.
"""
import hashlib
import os

import numpy as np


def _cache_key(model_name, doc_ids, max_seq_length):
    """Fingerprint a (model, corpus, truncation) triple so cached embeddings
    are only reused when all three match. Domain corpora are frozen release
    files, so count + first id + last id is enough to detect corpus change.
    """
    h = hashlib.sha1()
    h.update(f"{model_name}|{max_seq_length}|{len(doc_ids)}".encode("utf-8"))
    if doc_ids:
        h.update(doc_ids[0].encode("utf-8"))
        h.update(doc_ids[-1].encode("utf-8"))
    return h.hexdigest()[:16]


class DenseRetriever:
    """Bi-encoder retriever with an on-disk embedding cache.

    The cache is keyed by (model, corpus, max_seq_length) and NOT by split,
    so running train then dev for the same domain re-uses the same corpus
    embeddings for free -- the corpus is identical across splits in RETECO.
    """

    def __init__(self, doc_ids, doc_texts, model_name="BAAI/bge-base-en-v1.5",
                 cache_dir=None, batch_size=256, device=None, fp16=True,
                 max_seq_length=256,
                 query_prefix="Represent this sentence for searching relevant passages: ",
                 passage_prefix="", verbose=True):
        from sentence_transformers import SentenceTransformer
        import torch

        self.doc_ids = list(doc_ids)
        self.model_name = model_name
        self.query_prefix = query_prefix or ""
        self.passage_prefix = passage_prefix or ""

        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length:
            self.model.max_seq_length = int(max_seq_length)

        # fp16 only helps on GPU; on CPU it is usually slower and less stable.
        self.use_fp16 = bool(fp16) and torch.cuda.is_available()
        if self.use_fp16:
            self.model = self.model.half()

        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            key = _cache_key(model_name, self.doc_ids, self.model.max_seq_length)
            cache_path = os.path.join(cache_dir, f"corpus_{key}.npy")

        if cache_path and os.path.isfile(cache_path):
            if verbose:
                print(f"  [dense] loading cached embeddings: {os.path.basename(cache_path)}",
                      flush=True)
            self.embeddings = np.load(cache_path).astype("float32")
        else:
            if verbose:
                print(f"  [dense] encoding {len(self.doc_ids)} documents "
                      f"(fp16={self.use_fp16}, max_seq_length={self.model.max_seq_length}, "
                      f"batch_size={batch_size})", flush=True)
            texts = [f"{self.passage_prefix}{t or ''}" for t in doc_texts]
            emb = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=verbose,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            if cache_path:
                np.save(cache_path, emb.astype("float16"))
            self.embeddings = emb.astype("float32")

        self._index = None
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self._index.add(self.embeddings)
        except ImportError:
            pass  # numpy fallback in search()

    def encode_query(self, query):
        q_text = f"{self.query_prefix}{query or ''}"
        return self.model.encode(
            [q_text], normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")

    def search(self, query, top_k=100):
        q_emb = self.encode_query(query)

        if self._index is not None:
            scores, idx = self._index.search(q_emb, min(top_k, len(self.doc_ids)))
            pairs = [
                (self.doc_ids[i], float(s))
                for i, s in zip(idx[0], scores[0]) if i != -1
            ]
        else:
            sims = self.embeddings @ q_emb[0]
            k = min(top_k, len(sims))
            top_idx = (np.argpartition(-sims, k - 1)[:k] if k < len(sims)
                       else np.arange(len(sims)))
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            pairs = [(self.doc_ids[i], float(sims[i])) for i in top_idx]

        # Deterministic tie-break by doc id, same convention as retriever.py.
        pairs.sort(key=lambda x: (-x[1], x[0]))
        return pairs[:top_k]
