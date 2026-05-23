"""
Modal GPU function
    -> stream small COCO dog sample
    -> save images to Modal Volume
    -> Ray Data image embedding
    -> LanceDB image_documents table on Volume
    -> metrics JSON

Modal app for the multimodal lakehouse search demo.

This file will eventually contain:
1. A remote batch job that builds LanceDB text and image tables.
2. Search endpoints for text, images, and combined multimodal search.

We start with a lightweight health check so Modal image builds, dependency
imports, and Volume persistence are validated before running GPU workloads.
"""

from pathlib import Path

import modal

APP_NAME = "multimodal-lakehouse-search"
VOLUME_NAME = "multimodal-lakehouse-volume"
DATA_DIR = "/data"

APP_NAME = "multimodal-lakehouse-search"
VOLUME_NAME = "multimodal-lakehouse-volume"
DATA_DIR = "/data"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "ray[data]>=2.40.0",
        "datasets>=2.20.0",
        "sentence-transformers>=3.0.0",
        "lancedb>=0.17.0",
        "pyarrow>=15.0.0",
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "fastapi[standard]",
        "Pillow>=10.0.0",
        "torch",
    )
)

app = modal.App(APP_NAME, image=image)

@app.function(
    cpu=2,
    memory=4096,
    timeout=300,
    volumes={DATA_DIR: volume},
)
def health_check():
    """Verify the Modal runtiome can import core dependencies and write vcolume data.

    Returns:
        a small dicitionary describing the remote runtime environment.
    """
    import platform
    import time

    import datasets
    import lancedb
    import numpy
    import pandas
    import ray
    import sentence_transformers
    from PIL import Image
    from torch import nn

    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    marker_path = data_dir / "health_check.txt"
    marker_path.write_text(f"Modal health check at {time.time()}\n")
    volume.commit()

    return {
        "python": platform.python_version(),
        "ray": ray.__version__,
        "datasets": datasets.__version__,
        "lancedb": lancedb.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "pillow": Image.__version__,
        "volume_marker": str(marker_path),
    }


@app.function(
    gpu='L4',
    cpu=2,
    memory=8192,
    timeout=600,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
def gpu_smoke_test():
    """Verify the Modal GPU environment can import core dependencies and write volume data.

    Returns:
        a small dicitionary describing the remote runtime environment.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    prompts = [
        "a dog sleeping under a blanket",
        "a dog playing outside",
        "a person walking a dog",
    ]

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    model = SentenceTransformer("clip-ViT-B-32", device=device)

    vectors = model.encode(prompts, batch_size=3, normalize_embeddings=True, show_progress_bar=True)

    return {
        "cuda_available": cuda_available,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else "CPU",
        "prompt_count": len(prompts),
        "embedding_shape": list(vectors.shape),
    }

@app.function(
    gpu='L4',
    cpu=2,
    memory=16_384,
    timeout=60 * 30,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
def build_image_table(limit: int=100, batch_size: int = 32) -> dict:
    """Build the image LanceDB table on MOdal using Ray and a GPU.

    This function streams a small COCO image-caption sample, saves images to the Modal Volume, 
    embedss images and captions with a CLIP-style model through Ray Data, and writes the results to LanceDB.

    Args:
        limit: Number of image-caption records to process.
        batch_size: Number of records per Ray batch.
    """
    import json
    import time
    import shutil
    from pathlib import Path
    from typing import Any
    
    import lancedb
    import numpy as np
    import pandas as pd

    import ray
    import ray.data
    from datasets import load_dataset
    from PIL import Image
    import torch
    from sentence_transformers import SentenceTransformer

    dataset_name = "Multimodal-Fatima/COCO_captions_train"
    split = "train"
    
    model_name = "sentence-transformers/clip-ViT-B-32"

    base_dir = Path(DATA_DIR)
    local_db_path = Path("/tmp/lancedb_build")
    volume_db_path = base_dir / "lancedb"
    image_dir = base_dir / "coco_images"
    manifest_path = base_dir / "metrics_modal_ray_image.json"

    db_path = base_dir / "lancedb"
    metrics_path = base_dir / "metrics_modal_ray_image.json"

    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {limit} image-caption records from {dataset_name}...")

    dataset = load_dataset(dataset_name, split=split, streaming=True)

    rows = []

    for idx, row in enumerate(dataset):
        if idx >= limit:
            break

        captions = [str(c).strip() for c in row.get("sentences_raw", []) if str(c).strip()]

        if not captions:
            continue

        caption = captions[0]

        cocoid = int(row.get("cocoid") or idx)
        image_id = f"coco_{cocoid:012d}"
        image_path = image_dir / f"{image_id}.jpg"

        row["image"].convert("RGB").save(image_path, format="JPEG", quality=90)

        rows.append(
            {
                "image_id": image_id,
                "cocoid": cocoid,
                "imgid": int(row.get("imgid") or -1),
                "filename": row.get("filename", ""),
                "image_path": str(image_path),
                "caption": caption,
                "captions_json": json.dumps(captions),
                "source": dataset_name,
                "split": row.get("split", split),
            }
        )

    manifest_df = pd.DataFrame(rows)
    manifest_path = base_dir / "coco_dog_sample.parquet"
    manifest_df.to_parquet(manifest_path, index=False)

    if ray.is_initialized():
        ray.shutdown()

    ray.init(num_cpus=4)

    class ImageEmbedderActor:
        """Stateful Ray actor that owns one GPU-backed CLIP model."""
        
        def __init__(self, model_name: str) -> None:
            """Load the CLIP-style model once per actor on CUDA."""
            self.model = SentenceTransformer(model_name, device="cuda")

        @staticmethod
        def load_rgb_image(image_path: str) -> Image.Image:
            """Load one image from the modal volume as RGB."""
            return Image.open(image_path).convert("RGB")
            
        def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, Any]:
            """embed one batch of image-caption rows."""
            image_paths = [str(path) for path in batch["image_path"]]
            captions = [str(caption) for caption in batch['caption']]

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

    start = time.time()

    ray_dataset = ray.data.read_parquet(str(manifest_path))

    embedded = ray_dataset.map_batches(
        ImageEmbedderActor,
        fn_constructor_args=(model_name,),
        batch_format="numpy",
        batch_size=batch_size,
        compute=ray.data.ActorPoolStrategy(size=1),
        num_gpus=1,
    )

    embedded_df = embedded.to_pandas()

    print("Writing image_documents table to temporary local LanceDB path...")

    # Lancedb uses filesystem commit operations such as atomic rename. 
    # Modal Volume is persistent, but it does not behave exactly like noraml local disk for every filesystem operation.
    # Build the table on ephemeral container disk first. 

    if local_db_path.exists():
        shutil.rmtree(local_db_path)

    db = lancedb.connect(str(local_db_path))
    db.create_table(
        "image_documents",
        data=embedded_df.to_dict("records"),
        mode="overwrite"
    )

    print("Copying finished LanceDB directory to Modal Volume...")

    # Copy the completed lancedb artifact into the persistent modal volume. 
    # The search endpoint will read from this Volume path later.

    if volume_db_path.exists():
        shutil.rmtree(volume_db_path)

    shutil.copytree(local_db_path, volume_db_path)

    elapsed = time.time() - start

    metrics = {
        "rows_requested": limit,
        "rows_written": len(embedded_df),
        "seconds": round(elapsed, 2),
        "rows_per_second": round(len(embedded_df) / elapsed, 2),
        "gpu": "L4",
        "ray_actor_count": 1,
        "batch_size": batch_size,
        "model": model_name,
        "storage": "Modal Volume + LanceDB",
    }

    metrics_path.write_text(json.dumps(metrics, indent=2))

    ray.shutdown()
    volume.commit()

    print(json.dumps(metrics, indent=2))
    return metrics

@app.local_entrypoint()
def main() -> None:
    """Run a remote Modal health checks from the local CLI."""

    health = health_check.remote()
    print("Health check results:", health)

    gpu = gpu_smoke_test.remote()
    print("GPU smoke test results:", gpu)

    """Run the first tiny Modal GPU image batch job."""
    result = build_image_table.remote(limit=100, batch_size=32)
    print(result)