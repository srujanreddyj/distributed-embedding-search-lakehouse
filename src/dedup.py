import numpy as np
import faiss


class EmbeddingDeduplicator:
    """Tier 3: Near-duplicate detection via embedding similarity.

    After preprocessing, we have embeddings. We build a FAISS index, then for each item query "is there anything within 0.92 cosine similarity?" If yes → it's a near-duplicate → drop it.

    FAISS vs brute force: With 10K–50K items, brute force is fine.
    First occurence is kept, later near-duplicates are dropped.
    """

    def __init__(self, dim: int, similarity_threshold: float = 0.92) -> None:
        self.dim = dim
        self.threshold = similarity_threshold
        # Cosine similarity -> convert to L2 distance
        # inner product = cosine one normalized vectors
        self.index = faiss.IndexFlatIP(dim)

    def deduplicate(self, ids: list[str], embeddings: list[list[float]]) -> list[str]:
        """Return IDs that are NOT near-duplicates of anything earlier in the list."""
        if not embeddings:
            return []

        vectors = np.array(embeddings, dtype=np.float32)
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)

        keep_ids = []
        for i, (id_, vec) in enumerate(zip(ids, vectors)):
            vec = vec.reshape(1, -1)
            # Search: is anything similar already in index?
            if self.index.ntotal > 0:
                distance, _ = self.index.search(vec, 1)
                if distance[0][0] > self.threshold:
                    continue  # skil too simialr to existing item

            self.index.add(vec)
            keep_ids.append(id_)

        return keep_ids
