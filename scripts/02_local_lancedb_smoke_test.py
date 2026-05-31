from pathlib import Path

import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/fineweb-edu-sample.parquet")
DB_PATH = Path("data/lancedb")
TABLE_NAME = "fineweb_demo"


def main(limit: int = 500) -> None:
    print("Loading sample data")

    df = pd.read_parquet(INPUT_PATH).head(limit)

    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Generating embeddings....")

    vectors = model.encode(
        df["text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    df["vector"] = vectors.astype("float32").tolist()

    print("Writing LanceDb table.....")

    db = lancedb.connect(str(DB_PATH))

    table = db.create_table(TABLE_NAME, data=df.to_dict("records"), mode="overwrite")

    query = "What is machine Learning?"

    query_vector = (
        model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        .astype("float32")
        .tolist()
    )

    print(f"\nSearch Query: {query}")

    results = table.search(query_vector).limit(5).to_pandas()

    for _, row in results.iterrows():
        print("\n---")
        print("distance:", row.get("_distance"))
        print("url:", row.get("url"))
        print("text:", row.get("text")[:500])


if __name__ == "__main__":
    main()
