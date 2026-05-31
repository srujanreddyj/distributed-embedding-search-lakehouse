"""
Local Ray Data text embedding pipeline.

This script proves the distributed batch inference pattern locally before we move the same idea to Modal.
It reads the FineWeb-Edu sample parquet file with Ray Data , embeds text in batches using a stateful actor,
write vectors to LanceDb, and records simple throughput metrics

The key design choice is using a callable class with Ray Data `map_batches`.
Ray executes callable classes as stateful actors, which lets each actor load the embedding load the embedding model once and reuse it across many batches

"""

import json
import time
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import ray
import ray.data
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/fineweb-edu-sample.parquet"
DB_PATH = "data/lancedb_ray"
TABLE_NAME = "text_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
METRICS_PATH = Path("data/metrics_local_ray_text.json")


class TextEmbedderActor:
    """
    Stateful Ray Actor that owns one embedding model instance.

    Ray creates actor instances from this callable class when it is passed to `map_batches`.
    The model is loaded in `__init__`, so it is reused across many batches instead of being reloaded for every batch
    """

    def __init__(self, model_name, str=MODEL_NAME) -> None:
        """
        Load the sentence-transformer model once per actor
        Args:
            model_name: Hugging Face model ID for the text embedding model.
        """

        self.model = SentenceTransformer(model_name)

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Embed one Ray Data Batch.
        Args:
            batch: A batch of records in NumPy format. Each column is represented as a NumPy array.

        Returns:
            the original batch with a new `text_vector` column containing float32 embeding vectors.
        """

        # keep the same text cap as sample creation so local and remote runs are
        # comparable and very long documents do not dominate runtime.
        texts = [str(value)[:2_000] for value in batch["text"]]

        vectors = self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        batch["text_vector"] = vectors.astype("float32").tolist()
        return batch


def main(actor_count: int = 2, batch_size: int = 128) -> None:
    """
    Ray local Text embedding and write results to lancedb

    Args:
        actor_count: Number of Ray actors to use locally. On a laptop this should stay small
            because each actor loads its own model copy.
        batch_size: Number of records Ray sends to the actor per call.
    """

    ray.init(ignore_reinit_error=True)

    start = time.time()

    print("Reading parquet with Ray Data....")
    dataset = ray.data.read_parquet(INPUT_PATH)

    print("Embedding text with Ray stateful actors....")
    embedded = dataset.map_batches(
        TextEmbedderActor,
        fn_constructor_args=(MODEL_NAME,),
        batch_format="numpy",
        batch_size=batch_size,
        compute=ray.data.ActorPoolStrategy(size=actor_count),
    )

    print("collecting embedding rows....")

    df = embedded.to_pandas()

    print("Writing Ray output to LanceDb...")
    db = lancedb.connect(DB_PATH)

    db.create_table(TABLE_NAME, data=df.to_dict("records"), mode="overwrite")

    elapsed = time.time() - start

    metrics = {
        "rows": len(df),
        "seconds": round(elapsed, 2),
        "rows_per_second": round(len(df) / elapsed, 2),
        "actor_count": actor_count,
        "batch_size": batch_size,
        "model": MODEL_NAME,
        "storage": "local LanceDB",
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    ray.shutdown()


if __name__ == "__main__":
    main()
