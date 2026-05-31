"""
local image files
    -> CLIP-style model
    -> image embeddings
    -> caption embeddings
    -> LanceDB image_documents table
    -> text-to-image search
"""

"""Local CLIP image embedding and LanceDB smoke test.

This script validates the image-search side of the multimodal lakehouse before
adding Ray or Modal. It reads the local COCO dog image-caption manifest, embeds
images and captions with a CLIP-style sentence-transformers model, stores both
vectors in LanceDB, and runs one text-to-image search query.

Keeping this local first makes failures easier to isolate: image loading,
model inference, and LanceDB search are tested before distributed execution.
"""

from pathlib import Path

import lancedb
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/coco_dog_sample.parquet")
DB_PATH = Path("data/lancedb_images")
TABLE_NAME = "image_documents"
MODEL_NAME = "clip-ViT-B-32"


def load_rgb_image(image_path: str) -> Image.Image:
    """Load one image file as RGB.

    Args:
        image_path: Local path to an image file.

    Returns:
        A PIL RGB image suitable for CLIP-style embedding.
    """
    return Image.open(image_path).convert("RGB")


def main(limit: int = 100) -> None:
    """Embed a small image-caption sample and run text-to-image search.

    Args:
        limit: Number of image-caption rows to embed. We start small because
            image models are heavier than text models on a laptop.
    """
    print("Loading image-caption sample...")
    df = pd.read_parquet(INPUT_PATH).head(limit)

    print("Loading CLIP-style embedding model...")
    # clip-ViT-B-32 maps both images and text into a shared embedding space.
    # That lets a text query retrieve images by comparing query text vectors
    # against image vectors.
    model = SentenceTransformer(MODEL_NAME)

    print("Loading images from disk...")
    images = [load_rgb_image(path) for path in df["image_path"].tolist()]

    print("Generating image embeddings...")
    image_vectors = model.encode(
        images,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Generating caption embeddings...")
    caption_vectors = model.encode(
        df["caption"].tolist(),
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # LanceDB stores vectors as Python lists. We store both image and caption
    # vectors so the table can support text-to-image and caption-to-caption
    # experiments later.
    df["image_vector"] = image_vectors.astype("float32").tolist()
    df["caption_vector"] = caption_vectors.astype("float32").tolist()

    print("Writing LanceDB image table...")
    db = lancedb.connect(str(DB_PATH))
    table = db.create_table(
        TABLE_NAME,
        data=df.to_dict("records"),
        mode="overwrite",
    )

    query = "a dog sleeping under a blanket"

    # For text-to-image search, embed the text query with the same CLIP model
    # and search against the image vector column.
    query_vector = (
        model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        .astype("float32")
        .tolist()
    )

    print(f"\nText-to-image query: {query}")
    results = (
        table.search(query_vector, vector_column_name="image_vector")
        .limit(5)
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
