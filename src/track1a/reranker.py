"""Cross-encoder reranking for RETECO Track 1a.

Wraps any first-stage retriever: takes its top-N candidates, scores each
(query, document) pair jointly with a cross-encoder, and reorders them.

    search(query, top_k) -> [(doc_id, score), ...] sorted by score descending

Why a cross-encoder helps where a bi-encoder cannot: a bi-encoder must
compress the whole document into one vector *before* it has seen the query,
so it can only measure coarse topical similarity. A cross-encoder reads the
query and the document together and can judge whether this specific document
answers this specific question -- including the temporal qualifier Track 1a
turns on ("as of 2017", "before the fork", "the latest version"). That is
the distinction the dataset card says lexical and embedding retrieval both
struggle with, which is why this is the highest-leverage stage to add after
first-stage retrieval.

Cost: O(candidate_k) forward passes per query, over the shortlist only,
never the corpus.

Implementation note -- this deliberately uses `transformers` directly rather
than `sentence_transformers.CrossEncoder`. On sentence-transformers 5.x,
CrossEncoder.predict feeds the tokenised batch into the model positionally,
which reaches XLMRoberta's forward as `input_ids=<BatchEncoding>` and dies on
`input_ids.device`. Driving the tokenizer and model ourselves is the usage
BAAI documents for bge-reranker, keeps fp16/batching/truncation explicit, and
does not depend on sentence-transformers internals that change between
versions.

Requires: transformers, torch.
"""


class RerankRetriever:
    """First-stage retriever + cross-encoder reranker."""

    def __init__(self, base, doc_ids, doc_texts,
                 model_name="BAAI/bge-reranker-base",
                 candidate_k=100, batch_size=64, device=None,
                 max_length=512, fp16=True, doc_char_limit=4000,
                 verbose=True):
        """
        base:           first-stage retriever exposing .search(query, top_k)
        doc_ids/texts:  the domain corpus, used to look up candidate text
        candidate_k:    how many first-stage candidates to rerank
        max_length:     tokeniser truncation for the (query, document) pair
        doc_char_limit: cheap pre-truncation of document text before
                        tokenisation; a speed guard only, the tokeniser would
                        discard the tail anyway at `max_length`
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self._torch = torch
        self.base = base
        self.doc_text = dict(zip(doc_ids, doc_texts))
        self.candidate_k = candidate_k
        self.batch_size = batch_size
        self.max_length = max_length
        self.doc_char_limit = doc_char_limit

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # fp16 only helps on GPU; on CPU it is slower and less numerically safe.
        self.use_fp16 = bool(fp16) and self.device.type == "cuda"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model = model.to(self.device)
        if self.use_fp16:
            model = model.half()
        model.eval()
        self.model = model

        if verbose:
            print(f"  [rerank] {model_name} (candidate_k={candidate_k}, "
                  f"max_length={max_length}, batch_size={batch_size}, "
                  f"device={self.device}, fp16={self.use_fp16})", flush=True)

    def _score_pairs(self, queries, passages):
        torch = self._torch
        scores = []
        with torch.no_grad():
            for i in range(0, len(queries), self.batch_size):
                q_batch = queries[i:i + self.batch_size]
                p_batch = passages[i:i + self.batch_size]
                # Two parallel lists is the unambiguous form of the pair API:
                # `text` and `text_pair`. Passing a list of [q, p] lists instead
                # can be read as pre-tokenised words by some tokenizers.
                inputs = self.tokenizer(
                    q_batch, p_batch,
                    padding=True, truncation=True, max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs, return_dict=True).logits
                scores.extend(logits.view(-1).float().cpu().tolist())
        return scores

    def search(self, query, top_k=100):
        candidates = self.base.search(query, self.candidate_k)
        if not candidates:
            return []

        doc_ids = [d for d, _ in candidates]
        queries = [query] * len(doc_ids)
        passages = [(self.doc_text.get(d) or "")[:self.doc_char_limit]
                    for d in doc_ids]

        scores = self._score_pairs(queries, passages)

        ranked = sorted(
            zip(doc_ids, (float(s) for s in scores)),
            key=lambda x: (-x[1], x[0]),   # deterministic tie-break by doc id
        )
        return ranked[:top_k]
