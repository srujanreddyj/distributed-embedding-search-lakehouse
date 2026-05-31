"""
coco_dog_sample.parquet
    -> Ray Data
    -> stateful CLIP actor
    -> image_vector + caption_vector
    -> LanceDB image_documents
    -> metrics JSON
"""

"""Local Ray Data image embedding pipeline.

This script proves the image side of the distributed batch inference pattern.
It reads a local COCO dog image-caption manifest, uses Ray Data to batch records,
embeds images and captions with a stateful CLIP actor, writes the result to
LanceDB, and records throughput metrics.

The key architectural point is the same as the text Ray pipeline: use a callable
class with `map_batches` so Ray creates stateful actors that load the model once
and reuse it across many batches.
"""

import json
import time
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import ray
import ray.data
from PIL import Image
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/coco_dog_sample.parquet"
DB_PATH = "data/lancedb_ray_images"
TABLE_NAME = "image_documents"
MODEL_NAME = "clip-ViT-B-32"
METRICS_PATH = Path("data/metrics_local_ray_image.json")


class ImageEmbedderActor:
    """Stateful Ray actor that owns one CLIP-style embedding model.

    Each actor loads the model once in `__init__`, then processes many image
    batches. This is especially useful for image models because model loading
    and preprocessing setup are more expensive than simple text transforms.
    """

    def __init__(self, model_name: str) -> None:
        """Load the CLIP-style model once per actor.

        Args:
            model_name: SentenceTransformers model name for multimodal
                image/text embedding.
        """
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def load_rgb_image(image_path: str) -> Image.Image:
        """Load one image file as RGB for CLIP-style embedding.

        Args:
            image_path: Local path to an image file.

        Returns:
            A PIL RGB image.
        """
        return Image.open(image_path).convert("RGB")

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, Any]:
        """Embed one Ray Data batch of image-caption records.

        Args:
            batch: Ray Data batch in NumPy format.

        Returns:
            The original batch with `image_vector` and `caption_vector` columns.
        """
        image_paths = [str(path) for path in batch["image_path"]]
        captions = [str(caption) for caption in batch["caption"]]

        # Load images inside the actor so each worker does its own local image
        # decoding close to the model inference step.
        images = [self.load_rgb_image(path) for path in image_paths]

        image_vectors = self.model.encode(
            images,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        caption_vectors = self.model.encode(
            captions,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        batch["image_vector"] = image_vectors.astype("float32").tolist()
        batch["caption_vector"] = caption_vectors.astype("float32").tolist()
        return batch


def main(actor_count: int = 1, batch_size: int = 32) -> None:
    """Run local Ray image embedding and write results to LanceDB.

    Args:
        actor_count: Number of Ray actors. Keep this at 1 locally at first
            because each actor loads a full CLIP model.
        batch_size: Number of image-caption records per Ray batch.
    """
    ray.init(ignore_reinit_error=True)

    start = time.time()

    print("Reading image manifest with Ray Data...")
    dataset = ray.data.read_parquet(INPUT_PATH)

    print("Embedding images and captions with Ray stateful actors...")
    embedded = dataset.map_batches(
        ImageEmbedderActor,
        fn_constructor_args=(MODEL_NAME,),
        batch_format="numpy",
        batch_size=batch_size,
        compute=ray.data.ActorPoolStrategy(size=actor_count),
    )

    print("Collecting embedded image rows...")
    df = embedded.to_pandas()

    print("Writing Ray image output to LanceDB...")
    db = lancedb.connect(DB_PATH)
    db.create_table(
        TABLE_NAME,
        data=df.to_dict("records"),
        mode="overwrite",
    )

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
