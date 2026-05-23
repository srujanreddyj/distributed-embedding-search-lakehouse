"""
This script validaes the text-search part of the lakehouse before adding Ray, Modal or image data. 
It reads the FineWeb-Edu sample parquet file, embeds as small number of text rows, stores them in LanceDB, and runs one semantic query.

Keeping this step local and small makes failures easier to reason about. 
"""

from pathlib import Path

import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/fineweb-edu-sample.parquet")
DB_PATH = Path("data/lancedb")
TABLE_NAME = "text_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main(limit: int = 500) -> None:
    """ Embed a small text sample, write it to lancedb, and run a test query.

    Args:
        limit: Number of text rows to embed. We keep this small for the first 
        local smoke test so the feedback loop stays fast.
    """

    print("Loading text sample...")
    df = pd.read_parquet(INPUT_PATH).head(limit)

    print("loading text embedding model....")

    # all-minilm-l6-v2 is small and cpu-friendly, which makes it a good first model for proving the local text pipeline. 
    model = SentenceTransformer(MODEL_NAME)

    print("Generating text embeddings..")

    # Normalized embeddings are useful for semantic search because vector magnitude no longer dominates similarity

    vectors = model.encode(
        df['text'].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    #lancedb expects vectors as python lists, not numpy arrays.
    df['text_vector'] = vectors.astype('float32').tolist()

    print("Writing LanceDb table....")
    db = lancedb.connect(str(DB_PATH))
    table = db.create_table(
        TABLE_NAME,
        data=df.to_dict("records"),
        mode="overwrite"
    )

    query = "What is machine learning?"

    # The query must be embedded with the same model as the documents so both
    # vectors live in the same embedding space.
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
    )[0].astype("float32").tolist()

    print(f"\nSearch query: {query}")
    results = (
        table.search(query_vector, vector_column_name="text_vector")
        .limit(5)
        .to_pandas()
    )

    for _, row in results.iterrows():
        print("\n---")
        print("distance:", row.get("_distance"))
        print("url:", row.get("url"))
        print("text:", row.get("text", "")[:500])


if __name__ == "__main__":
    main()