"""Create a small local COCO dog image-caption sample.

This script prepares the image side of the multimodal lakehouse demo. It loads
a small number of image-caption records from a COCO-derived dog dataset and
writes two local artifacts:

1. Image files under `data/coco_dog_images/`
2. A parquet manifest at `data/coco_dog_sample.parquet`

We separate image sampling from CLIP embedding so dataset loading problems are
isolated from model inference problems.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "ArkaMukherjee/coco_dog_images_with_captions"
SPLIT = "train"
IMAGE_DIR = Path("data/coco_dog_images")
OUTPUT_PATH = Path("data/coco_dog_sample.parquet")


def clean_caption(caption: str) -> str:
    """Normalize a caption string for storage and embedding.

    Args:
        caption: Raw caption from the dataset.

    Returns:
        A whitespace-normalized caption string.
    """
    return " ".join(str(caption or "").split())


def main(limit: int = 500) -> None:
    """Save a small COCO dog image-caption sample locally.

    Args:
        limit: Number of records to keep. Start small because image datasets
            are heavier than text datasets.
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {limit} records from {DATASET_NAME}...")

    dataset = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

    rows = []

    for idx, row in enumerate(dataset):
        if idx >= limit:
            break

        caption = clean_caption(row.get("captions", ""))

        if not caption:
            continue

        image = row["image"].convert("RGB")
        image_id = f"coco_dog_{idx:06d}"
        image_path = IMAGE_DIR / f"{image_id}.jpg"

        # Store the image locally so later steps can load it without depending
        # on the streaming dataset iterator.
        image.save(image_path, format="JPEG", quality=90)

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "caption": caption,
                "source": DATASET_NAME,
                "split": SPLIT,
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} image-caption rows to {OUTPUT_PATH}")
    print(f"Saved images to {IMAGE_DIR}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
