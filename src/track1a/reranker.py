"""Cross-encoder reranking for RETECO Track 1a.

Wraps any first-stage retriever: takes its top-N candidates, scores each
(query, document) pair jointly with a cross-encoder, and reorders them.

    search(query, top_k) -> [(doc_id, score), ...] sorted by score descending

Why a cross-encoder helps where a bi-encoder cannot: a bi-encoder must
compress the whole document into one vector *before* it has seen the query,
so it can only measure coarse topical similarity. A cross-encoder reads the
query and the document together and can judge whether this specific document
answers this specific question -- including the temporal qualifier that
Track 1a turns on ("as of 2017", "before the fork", "the latest version").
That is exactly the distinction the dataset card says lexical and embedding
retrieval both struggle with, which is why this is the highest-leverage
component to add after first-stage retrieval.

Cost: reranking is O(candidate_k) forward passes per query, so it only runs
over the shortlist, never the corpus. With candidate_k=100 and a base-size
reranker on a T4, a full 13-domain train run is on the order of minutes, not
hours.

Requires: sentence-transformers (CrossEncoder), torch.
"""


class RerankRetriever:
    """First-stage retriever + cross-encoder reranker."""

    def __init__(self, base, doc_ids, doc_texts,
                 model_name="BAAI/bge-reranker-base",
                 candidate_k=100, batch_size=64, device=None,
                 max_length=512, fp16=True, doc_char_limit=4000,
                 verbose=True):
        """
        base:          first-stage retriever exposing .search(query, top_k)
        doc_ids/texts: the domain corpus, used to look up candidate text
        candidate_k:   how many candidates to pull from `base` and rerank
        max_length:    cross-encoder truncation length (query + document)
        doc_char_limit: cheap pre-truncation of document text before
                       tokenisation; purely a speed guard, the tokeniser
                       would discard the tail anyway at `max_length`
        """
        from sentence_transformers import CrossEncoder
        import torch

        self.base = base
        self.doc_text = dict(zip(doc_ids, doc_texts))
        self.candidate_k = candidate_k
        self.batch_size = batch_size
        self.doc_char_limit = doc_char_limit

        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

        self.use_fp16 = bool(fp16) and torch.cuda.is_available()
        if self.use_fp16:
            try:
                self.model.model = self.model.model.half()
            except Exception as exc:  # pragma: no cover - version dependent
                self.use_fp16 = False
                if verbose:
                    print(f"  [rerank] fp16 unavailable, staying in fp32: {exc}",
                          flush=True)

        if verbose:
            print(f"  [rerank] {model_name} (candidate_k={candidate_k}, "
                  f"max_length={max_length}, fp16={self.use_fp16})", flush=True)

    def search(self, query, top_k=100):
        candidates = self.base.search(query, self.candidate_k)
        if not candidates:
            return []

        doc_ids = [d for d, _ in candidates]
        pairs = [
            (query, (self.doc_text.get(d) or "")[:self.doc_char_limit])
            for d in doc_ids
        ]

        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False,
        )

        ranked = sorted(
            zip(doc_ids, (float(s) for s in scores)),
            key=lambda x: (-x[1], x[0]),   # deterministic tie-break by doc id
        )
        return ranked[:top_k]
