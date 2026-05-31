"""Search the local Ray-built LanceDB text table.

This script validates that the Ray Data embedding pipeline produced a usable
LanceDB table. It deliberately does not rebuild embeddings; it only embeds a
query and searches the existing `text_documents` table.

This separates ingestion correctness from retrieval correctness.
"""

from pathlib import Path

import lancedb
from sentence_transformers import SentenceTransformer

DB_PATH = Path("data/lancedb_ray")
TABLE_NAME = "text_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main(query: str = "What is machine learning?", k: int = 5) -> None:
    """Run a semantic search query against the Ray-generated text table.

    Args:
        query: Natural-language search query.
        k: Number of nearest neighbors to return.
    """
    print("Opening Ray-built LanceDB table...")
    db = lancedb.connect(str(DB_PATH))
    table = db.open_table(TABLE_NAME)

    print("Loading query embedding model...")
    # The query must use the same embedding model as the documents. Otherwise,
    # query and document vectors would live in incompatible vector spaces.
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding query: {query}")
    query_vector = (
        model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        .astype("float32")
        .tolist()
    )

    print("Searching...")
    results = (
        table.search(query_vector, vector_column_name="text_vector")
        .limit(k)
        .to_pandas()
    )

    for _, row in results.iterrows():
        print("\n---")
        print("distance:", row.get("_distance"))
        print("url:", row.get("url"))
        print("text:", row.get("text", "")[:500])


if __name__ == "__main__":
    main()
