"""Production-grade hybrid retrieval with RRF, reranking, and MMR."""

import logging
import threading
from collections import defaultdict

from nexusrag.retrieval.dense import DenseSearcher, RetrievalResult
from nexusrag.retrieval.sparse import BM25Retriever
from nexusrag.retrieval.stopwords import STOP_WORDS

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Production-grade hybrid retrieval combining dense and sparse methods.

    Pipeline:
    1. Dense retrieval (semantic similarity)
    2. Sparse retrieval (BM25 keyword matching)
    3. Reciprocal Rank Fusion (RRF)
    4. Optional reranking
    5. Optional MMR diversification
    """

    def __init__(
        self,
        dense_retriever: DenseSearcher,
        sparse_retriever: BM25Retriever,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
        use_keyword_boost: bool = True,
    ):
        """
        Initialize hybrid retriever.

        Args:
            dense_retriever: Semantic retriever
            sparse_retriever: BM25 retriever
            dense_weight: Weight for dense results in fusion
            sparse_weight: Weight for sparse results in fusion
            rrf_k: RRF smoothing constant (default 60)
            use_mmr: Whether to apply MMR diversification
            mmr_lambda: MMR lambda (0=max diversity, 1=max relevance)
            use_keyword_boost: Whether to boost chunks containing query keywords
        """
        if not (0 <= dense_weight <= 1 and 0 <= sparse_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")

        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.use_keyword_boost = use_keyword_boost

        logger.info(
            f"Initialized HybridRetriever: dense={dense_weight}, "
            f"sparse={sparse_weight}, mmr={use_mmr}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        apply_mmr: bool | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve using hybrid approach with RRF fusion and query expansion.

        Args:
            query: Search query
            top_k: Final number of results
            dense_top_k: Override dense retrieval count
            sparse_top_k: Override sparse retrieval count
            apply_mmr: Override MMR setting

        Returns:
            Fused results sorted by combined score
        """
        logger.info(f"Hybrid retrieval for: {query[:50]}...")

        # Expand query for better coverage
        expanded_queries = self._expand_query(query)
        logger.info(f"Using {len(expanded_queries)} query variations")

        # overfetch for fusion
        fetch_k = max(top_k * 3, 15)
        dense_k = dense_top_k or fetch_k
        sparse_k = sparse_top_k or fetch_k

        # Get candidates from both retrievers using all query variations
        all_dense: list[RetrievalResult] = []
        all_sparse: list[RetrievalResult] = []

        for q in expanded_queries:
            all_dense.extend(self.dense.retrieve(q, dense_k))
            all_sparse.extend(self.sparse.retrieve(q, sparse_k))

        # Deduplicate while keeping highest scores
        dense_results = self._deduplicate_results(all_dense)
        sparse_results = self._deduplicate_results(all_sparse)

        logger.info(f"Retrieved: {len(dense_results)} dense, {len(sparse_results)} sparse")

        # Fuse with RRF
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results, fetch_k)

        # Apply keyword boost
        if self.use_keyword_boost:
            fused_results = self._apply_keyword_boost(query, fused_results)

        # MMR diversify
        use_mmr = apply_mmr if apply_mmr is not None else self.use_mmr
        if use_mmr and len(fused_results) > top_k:
            fused_results = self._maximal_marginal_relevance(query, fused_results, top_k)
        else:
            fused_results = fused_results[:top_k]

        logger.info(f"Final results: {len(fused_results)}")
        return fused_results

    def _expand_query(self, query: str) -> list[str]:
        """Generate multiple search queries for better retrieval."""
        queries = [query]  # Original query always first

        # Extract keywords
        keywords = self._extract_keywords(query)
        if keywords and len(keywords) >= 2:
            queries.append(" ".join(keywords))

        # Add question variations
        q_lower = query.lower()
        if q_lower.startswith("what"):
            alt = query.replace("What", "Describe").replace("what", "describe")
            if alt != query:
                queries.append(alt)
        elif q_lower.startswith("how"):
            alt = query.replace("How", "Method for").replace("how", "method for")
            if alt != query:
                queries.append(alt)
        elif q_lower.startswith("who"):
            alt = query.replace("Who", "Authors").replace("who", "authors")
            if alt != query:
                queries.append(alt)
        elif q_lower.startswith("why"):
            alt = query.replace("Why", "Reason for").replace("why", "reason for")
            if alt != query:
                queries.append(alt)

        return queries[:3]  # Limit to 3 variations

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract important keywords from query, filtering stop words."""
        words = query.lower().split()
        return [w for w in words if len(w) > 2 and w.isalpha() and w not in STOP_WORDS]

    def _deduplicate_results(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Keep highest scoring result for each chunk."""
        seen: dict[str, RetrievalResult] = {}
        for result in results:
            chunk_id = result.chunk.id
            if chunk_id not in seen or result.score > seen[chunk_id].score:
                seen[chunk_id] = result
        return list(seen.values())

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = sum(weight / (k + rank)) for each result list
        """
        chunk_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievalResult] = {}

        # Score dense results
        for rank, result in enumerate(dense_results, start=1):
            rrf_score = self.dense_weight / (self.rrf_k + rank)
            chunk_scores[result.chunk.id] += rrf_score
            chunk_map[result.chunk.id] = result

        # Score sparse results
        for rank, result in enumerate(sparse_results, start=1):
            rrf_score = self.sparse_weight / (self.rrf_k + rank)
            chunk_scores[result.chunk.id] += rrf_score
            if result.chunk.id not in chunk_map:
                chunk_map[result.chunk.id] = result

        # Sort by fused score
        sorted_ids = sorted(
            chunk_scores.keys(),
            key=lambda cid: chunk_scores[cid],
            reverse=True,
        )[:top_k]

        # Normalize scores to 0-1 range
        max_rrf = max(chunk_scores.values()) if chunk_scores else 1.0

        # Create results with normalized hybrid scores
        return [
            RetrievalResult(
                chunk=chunk_map[cid].chunk,
                score=chunk_scores[cid] / max_rrf if max_rrf > 0 else 0,
                source="hybrid",
            )
            for cid in sorted_ids
        ]

    def _apply_keyword_boost(
        self, query: str, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Boost scores for chunks containing query keywords."""
        # Extract keywords from query
        keywords = self._extract_keywords(query)

        if not keywords:
            return results

        keyword_set = set(keywords)
        boosted = []

        # Pre-compute ordered query keywords for phrase matching
        query_words = [w for w in query.lower().split() if w in keyword_set]

        for result in results:
            content_lower = result.chunk.content.lower()

            # Count keyword matches (more weight for exact matches)
            matches = sum(1 for kw in keyword_set if kw in content_lower)

            # Calculate boost (up to 25% boost for matching keywords)
            boost = min(0.25, matches * 0.05)

            # Extra boost if multiple keywords appear close together
            if matches >= 2:
                boost += 0.05

            # Phrase match bonus: consecutive query keywords in content
            for j in range(len(query_words) - 1):
                phrase = f"{query_words[j]} {query_words[j + 1]}"
                if phrase in content_lower:
                    boost += 0.05
                    break  # One phrase bonus is enough

            # Hard cap at 0.30 total boost
            boost = min(0.30, boost)

            boosted_score = min(1.0, result.score + boost)

            boosted.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=boosted_score,
                    source=result.source,
                )
            )

        # Re-sort by boosted score
        boosted.sort(key=lambda r: r.score, reverse=True)
        return boosted

    def _maximal_marginal_relevance(
        self,
        _query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Diversify in embedding space.

        MMR = lambda * relevance - (1 - lambda) * max similarity to selected,
        with cosine similarity over the chunk embeddings (normalized, so dot
        product equals cosine).
        """
        if len(results) <= top_k:
            return results[:top_k]

        import numpy as np

        try:
            embs = np.asarray(
                self.dense.embedder.embed([r.chunk.content for r in results]), dtype=np.float64
            )
        except Exception:
            logger.debug("MMR embedding failed; falling back to top-k", exc_info=True)
            return results[:top_k]

        if embs.ndim != 2 or embs.shape[0] != len(results):
            return results[:top_k]

        relevance = np.asarray([r.score for r in results], dtype=np.float64)
        first = int(relevance.argmax())
        selected = [first]
        chosen = {first}
        max_sim = embs @ embs[first]

        while len(selected) < top_k and len(selected) < len(results):
            mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim
            mmr[list(chosen)] = -np.inf
            nxt = int(mmr.argmax())
            selected.append(nxt)
            chosen.add(nxt)
            max_sim = np.maximum(max_sim, embs @ embs[nxt])

        return [results[i] for i in selected]

    def retrieve_dense_only(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Bypass fusion, use only dense retrieval."""
        return self.dense.retrieve(query, top_k)

    def retrieve_sparse_only(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Bypass fusion, use only sparse retrieval."""
        return self.sparse.retrieve(query, top_k)


class AdaptiveHybridRetriever(HybridRetriever):
    """Hybrid retriever that adjusts weights based on query characteristics."""

    def __init__(
        self,
        dense_retriever: DenseSearcher,
        sparse_retriever: BM25Retriever,
        base_dense_weight: float = 0.6,
        base_sparse_weight: float = 0.4,
        rrf_k: int = 60,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
        use_keyword_boost: bool = True,
    ):
        super().__init__(
            dense_retriever,
            sparse_retriever,
            base_dense_weight,
            base_sparse_weight,
            rrf_k,
            use_mmr,
            mmr_lambda,
            use_keyword_boost,
        )
        self.base_dense_weight = base_dense_weight
        self.base_sparse_weight = base_sparse_weight
        self._weight_lock = threading.Lock()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        apply_mmr: bool | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve with query-adaptive weights."""
        dense, sparse = self._adapt_weights(query)
        with self._weight_lock:
            self.dense_weight, self.sparse_weight = dense, sparse
            return super().retrieve(query, top_k, dense_top_k, sparse_top_k, apply_mmr)

    def _adapt_weights(self, query: str) -> tuple[float, float]:
        """Compute query-adaptive weights (thread-safe, no mutation)."""
        words = query.split()
        word_count = len(words)

        # Check for technical terms (likely need exact match)
        has_technical = any(
            w.isupper() or "_" in w or w.endswith("()") or len(w) > 12 for w in words
        )

        # Short keyword queries favor sparse (exact match)
        if word_count <= 3 or has_technical:
            sparse = min(0.6, self.base_sparse_weight + 0.2)
            dense = 1 - sparse
            logger.debug(f"Adapted weights: sparse-heavy ({sparse})")

        # Long natural language queries favor dense (semantic)
        elif word_count >= 10:
            dense = min(0.8, self.base_dense_weight + 0.2)
            sparse = 1 - dense
            logger.debug(f"Adapted weights: dense-heavy ({dense})")

        # Default balanced
        else:
            dense = self.base_dense_weight
            sparse = self.base_sparse_weight

        return dense, sparse
