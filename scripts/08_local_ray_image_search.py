"""Search the local Ray-built LanceDb image table.

This script validates that the Ray image embedding pipeline produced a usable lanceDB table.
It embeds a text query with the same CLIP-style model and searches against the `image_vector` column to retrieve matching images.

This i the local version of the future Modal search endpoint.
"""

from pathlib import Path

import lancedb
from pydantic import NonNegativeFloat
from sentence_transformers import SentenceTransformer

DB_PATH = Path("data/lancedb_ray_images")
TABLE_NAME = "image_documents"
MODEL_NAME = "clip-ViT-B-32"


def main(query: str = "a dog sleeping under a blanket", k: int = 5) -> None:
    """Run text-to-image search against the Ray-generated image table.

    Args:
        query: Natural-language query to embed with CLIP text encoder.
        k: Number of nearest image records to return.
    """

    print("Opening Ray-built image LanceDb table.....")
    db = lancedb.connect(str(DB_PATH))
    table = db.open_table(TABLE_NAME)

    print("Loading CLIP-style query model....")
    # CLIP maps text and images into a shared embedding space.
    # We embed the text query and compare it against stored image vectors.
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

    print("Searching image vectors...")
    results = (
        table.search(query_vector, vector_column_name="image_vector")
        .limit(k)
        .to_pandas()
    )

    for _, row in results.iterrows():
        print("\n---")
        print("distance:", row.get("_distance"))
        print("image_id:", row.get("image_id"))
        print("image_path:", row.get("image_path"))
        print("caption:", row.get("caption"))


if __name__ == "__main__":
    main()
