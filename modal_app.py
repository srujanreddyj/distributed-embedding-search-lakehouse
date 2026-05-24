"""Modal app for the multimodal lakehouse search demo.

The app has three responsibilities:
1. Batch-build text and image embedding tables with Ray Data.
2. Persist LanceDB artifacts and raw images on a Modal Volume.
3. Serve JSON search endpoints and a small HTML demo UI.

The UI is intentionally Modal-hosted so the portfolio demo can be opened in a
browser without running a separate frontend service.
"""

from pathlib import Path

import modal

APP_NAME = "multimodal-lakehouse-search"
VOLUME_NAME = "multimodal-lakehouse-volume"
DATA_DIR = "/data"
STATIC_DIR = Path(__file__).parent / "static"
HTML_PATH = STATIC_DIR / "index.html"

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
    .add_local_dir("static", remote_path="/root/static")
)

app = modal.App(APP_NAME, image=image)


TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

@app.function(
    cpu=2,
    memory=4096,
    timeout=300,
    volumes={DATA_DIR: volume},
)
def health_check():
    """Verify the Modal runtime can import dependencies and write Volume data.

    Returns:
        A small dictionary describing the remote runtime environment.
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
    """Verify Modal GPU allocation and CLIP embedding.

    Returns:
        A small dictionary describing GPU availability and embedding shape.
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

    # Use the fully qualified Hugging Face repo ID to avoid alias/casing
    # differences between local and Modal environments.
    model = SentenceTransformer(IMAGE_MODEL_NAME, device=device)

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
def build_image_table(limit: int=1000, batch_size: int = 32) -> dict:
    """Build the image LanceDB table on Modal using Ray and a GPU.

    This function streams COCO image-caption rows, saves raw images to the
    Modal Volume, embeds images/captions with a CLIP-style model through Ray
    Data, and writes the result to LanceDB.

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
    from sentence_transformers import SentenceTransformer

    dataset_name = "Multimodal-Fatima/COCO_captions_train"
    split = "train"
    
    model_name = IMAGE_MODEL_NAME

    base_dir = Path(DATA_DIR)
    local_db_path = Path("/tmp/lancedb_build")
    volume_db_path = base_dir / "lancedb"
    image_dir = base_dir / "coco_images"
    manifest_path = base_dir / "coco_sample.parquet"
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

        # Store the raw image beside the vector table. The browser cannot read
        # this internal path directly, so the UI exposes a small image-serving
        # route later.
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
            """Embed one batch of image-caption rows."""
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


@app.function(
    cpu=2,
    memory=8192,
    timeout=300,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")]
)
@modal.fastapi_endpoint(method="POST")
def search_images(item: dict) -> dict:
    """Search persisted image embeddings with a text query

    This endpoint is the serving counterpart to `build_image_table`.
    The batch job write `image_documents` into LanceDB on the Modal volume; this function 
    reloads the Volume, embeds the user's text query with the same CLIP model,
    and searches against the stored `image_vector` column. 
    Args:
        item: JSON request body with `query` and optional `k`.

    Returns:
        JSON-serializable search results containing image metadata, captions,
        distances, and local Volume image paths.
    """

    from pathlib import Path

    import lancedb
    from sentence_transformers import SentenceTransformer

    # Pull the latest committed Volume state so this endpoint can see the table created by the batch function.
    volume.reload()
    query = str(item.get("query", "")).strip()
    k = int(item.get('k', 5))

    if not query:
        return {"error": "Please provide a non-empty query."}

    db_path = Path(DATA_DIR) / "lancedb"
    model_name = "sentence-transformers/clip-ViT-B-32"

    db = lancedb.connect(str(db_path))
    table = db.open_table("image_documents")

    # CLIP maps text and images into a shared vector space. For text-to-image
    # search, embed the query text and compare it to stored image vectors.
    model = SentenceTransformer(model_name)

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
    )[0].astype("float32").tolist()

    results = (
        table.search(query_vector, vector_column_name="image_vector")
        .limit(k)
        .to_pandas()
    )

    matches = []

    for _, row in results.iterrows():
        matches.append(
            {
                "image_id": row.get("image_id", ""),
                "cocoid": int(row.get("cocoid", -1)),
                "filename": row.get("filename", ""),
                "image_path": row.get("image_path", ""),
                "caption": row.get("caption", ""),
                "source": row.get("source", ""),
                "split": row.get("split", ""),
                "distance": float(row.get("_distance", 0.0)),
            }
        )

    return {
        "query": query,
        "k": k,
        "matches": matches,
    }


@app.function(
    gpu='L4',
    cpu=2,
    memory=16_384,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
def build_text_table(limit: int = 5000, batch_size: int = 128) -> dict:
    """Build the FineWeb-Edu text LanceDb table on Modal using Ray

    This function mirrors the image table build path, but uses a text embedding model and stores results
    in a separate LanceDB root. Keeping image and text DB roots separate avoids accidental overwrites
    while the demo is evolving.

    Args:
        limit (int, optional): _description_. Defaults to 500.
        batch_size (int, optional): _description_. Defaults to 128.

    Returns:
        dict: _description_
    """

    import json
    import shutil
    import time
    from itertools import islice
    from pathlib import Path
    from typing import Any

    import lancedb
    import numpy as np
    import pandas as pd
    import ray
    import ray.data
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset_config = "sample-10BT"
    split = "train"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    base_dir = Path(DATA_DIR)
    manifest_path = base_dir / "fineweb_edu_sample.parquet"
    metrics_path = base_dir / "metrics_modal_ray_text.json"

    local_db_path = Path("/tmp/lancedb_text_build")
    volume_db_path = base_dir / "lancedb_text"


    def clean_text(text: str, max_chars: int = 2_000) -> str:
        """Normalize whitespace and cap long documents for stable demo runtime

        Args:
            text (str): _description_
            max_chars (int, optional): _description_. Defaults to 2_000.

        Returns:
            str: _description_
        """
        return " ".join(str(text or "").split())[:max_chars]

    print(f"Streaming {limit} text records from FineWeb-Edu....")

    dataset = load_dataset(
        dataset_name, 
        name=dataset_config,
        split=split,
        streaming=True,
    )
    rows = []

    for idx, row in enumerate(islice(dataset, limit)):
        text = clean_text(row.get("text", ""))

        if len(text.split()) < 30:
            continue

        rows.append(
            {
                "id": str(idx),
                "text": text,
                "url": row.get("url", ""),
                "token_count": int(row.get("token_count") or 0),
                "source": f"{dataset_name}/{dataset_config}",
            }
        )

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_parquet(manifest_path, index=False)

    if ray.is_initialized():
        ray.shutdown()

    ray.init(num_cpus=4)

    class TextEmbedderActor:
        """Stateful Ray actor that owns one text embedding model."""

        def __init__(self, model_name: str) -> None:
            """Load the text embedding model once per actor."""
            self.model = SentenceTransformer(model_name, device="cuda")

        def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, Any]:
            """Embed one batch of text records."""
            texts = [str(text)[:2_000] for text in batch["text"]]

            vectors = self.model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            batch["text_vector"] = vectors.astype("float32").tolist()
            return batch

    start = time.time()

    ray_dataset = ray.data.read_parquet(str(manifest_path))

    embedded = ray_dataset.map_batches(
        TextEmbedderActor,
        fn_constructor_args=(model_name,),
        batch_format="numpy",
        batch_size=batch_size,
        compute=ray.data.ActorPoolStrategy(size=1),
        num_gpus=1,
    )

    embedded_df = embedded.to_pandas()

    print("Writing text_documents table to temporary local LanceDB path...")

    if local_db_path.exists():
        shutil.rmtree(local_db_path)

    db = lancedb.connect(str(local_db_path))
    db.create_table(
        "text_documents",
        data=embedded_df.to_dict("records"),
        mode="overwrite",
    )

    print("Copying finished text LanceDB directory to Modal Volume...")

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


@app.function(
    cpu=2,
    memory=8192,
    timeout=300,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
@modal.fastapi_endpoint(method="POST")
def search_text(item: dict) -> dict:
    """Search persisted text embeddings with a text query.

    This endpoint is the serving counterpart to `build_text_table`. The batch
    job writes `text_documents` into LanceDB on the Modal Volume; this function
    reloads the Volume, embeds the user's text query with the same text model,
    and searches against the stored `text_vector` column.

    Args:
        item: JSON request body with `query` and optional `k`.

    Returns:
        JSON-serializable text search results with document snippets, source
        metadata, and distances.
    """
    from pathlib import Path

    import lancedb
    from sentence_transformers import SentenceTransformer

    # The text table is written by a separate batch function, so reload the
    # Volume before opening LanceDB to see the latest committed files.
    volume.reload()

    query = str(item.get("query", "")).strip()
    k = int(item.get("k", 5))

    if not query:
        return {"error": "Please provide a non-empty query."}

    db_path = Path(DATA_DIR) / "lancedb_text"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    db = lancedb.connect(str(db_path))
    table = db.open_table("text_documents")

    # Query and documents must be embedded with the same text model so they live
    # in the same vector space.
    model = SentenceTransformer(model_name)

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
    )[0].astype("float32").tolist()

    results = (
        table.search(query_vector, vector_column_name="text_vector")
        .limit(k)
        .to_pandas()
    )

    matches = []

    for _, row in results.iterrows():
        matches.append(
            {
                "id": row.get("id", ""),
                "text": row.get("text", "")[:700],
                "url": row.get("url", ""),
                "source": row.get("source", ""),
                "token_count": int(row.get("token_count", 0)),
                "distance": float(row.get("_distance", 0.0)),
            }
        )

    return {
        "query": query,
        "k": k,
        "matches": matches,
    }


@app.function(
    cpu=2,
    memory=12_288,
    timeout=300,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
@modal.asgi_app()
def web_ui():
    """Serve a tiny browser UI plus API routes for the multimodal demo.

    The UI is deliberately colocated with the Modal backend. It calls an
    internal `/api/search_all` route and renders text matches beside image
    matches. Images are served through `/image/{image_id}` because `/data/...`
    paths are private to the Modal container and are not browser-accessible.
    """
    from pathlib import Path
    import re

    import lancedb
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from sentence_transformers import SentenceTransformer

    fastapi_app = FastAPI(title="Multimodal Lakehouse Search")

    def search_both_tables(query: str, k: int) -> dict:
        """Search text and image LanceDB tables and return separate rankings.

        Text and image rankings stay separate because MiniLM and CLIP distances
        are not calibrated against each other.
        """
        volume.reload()

        text_db = lancedb.connect(str(Path(DATA_DIR) / "lancedb_text"))
        image_db = lancedb.connect(str(Path(DATA_DIR) / "lancedb"))

        text_table = text_db.open_table("text_documents")
        image_table = image_db.open_table("image_documents")

        text_model = SentenceTransformer(TEXT_MODEL_NAME)
        text_query_vector = text_model.encode(
            [query],
            normalize_embeddings=True,
        )[0].astype("float32").tolist()

        text_results = (
            text_table.search(text_query_vector, vector_column_name="text_vector")
            .limit(k)
            .to_pandas()
        )

        image_model = SentenceTransformer(IMAGE_MODEL_NAME)
        image_query_vector = image_model.encode(
            [query],
            normalize_embeddings=True,
        )[0].astype("float32").tolist()

        image_results = (
            image_table.search(image_query_vector, vector_column_name="image_vector")
            .limit(k)
            .to_pandas()
        )

        text_matches = [
            {
                "id": row.get("id", ""),
                "text": row.get("text", "")[:700],
                "url": row.get("url", ""),
                "source": row.get("source", ""),
                "token_count": int(row.get("token_count", 0)),
                "distance": float(row.get("_distance", 0.0)),
            }
            for _, row in text_results.iterrows()
        ]

        image_matches = [
            {
                "image_id": row.get("image_id", ""),
                "cocoid": int(row.get("cocoid", -1)),
                "filename": row.get("filename", ""),
                "image_url": f"/image/{row.get('image_id', '')}",
                "caption": row.get("caption", ""),
                "source": row.get("source", ""),
                "split": row.get("split", ""),
                "distance": float(row.get("_distance", 0.0)),
            }
            for _, row in image_results.iterrows()
        ]

        return {
            "query": query,
            "k": k,
            "text_matches": text_matches,
            "image_matches": image_matches,
            "note": (
                "Text and image matches are ranked separately because they use "
                "different embedding models and distance scales."
            ),
        }

    @fastapi_app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Render the portfolio demo UI."""
        return HTML_PATH.read_text()

    @fastapi_app.post("/api/search_all")
    def api_search_all(item: dict) -> JSONResponse:
        """Search both modalities for the browser UI."""
        query = str(item.get("query", "")).strip()
        k = int(item.get("k", 5))

        if not query:
            return JSONResponse({"error": "Please provide a non-empty query."}, status_code=400)

        return JSONResponse(search_both_tables(query=query, k=k))

    @fastapi_app.get("/image/{image_id}")
    def serve_image(image_id: str) -> FileResponse:
        """Serve one image stored on the Modal Volume.

        The route validates `image_id` before building a file path so the UI can
        display persisted COCO images without exposing arbitrary filesystem
        access.
        """
        volume.reload()

        if not re.fullmatch(r"coco_[0-9]{12}", image_id):
            return JSONResponse({"error": "Invalid image_id"}, status_code=400)

        image_path = Path(DATA_DIR) / "coco_images" / f"{image_id}.jpg"

        if not image_path.exists():
            return JSONResponse({"error": "Image not found"}, status_code=404)

        return FileResponse(str(image_path), media_type="image/jpeg")

    return fastapi_app



@app.function(
    cpu=2,
    memory=12_288,
    timeout=300,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
@modal.fastapi_endpoint(method="POST")
def search_all(item: dict) -> dict:
    """Search both text and image tables from one query.

    This endpoint demonstrates the final multimodal serving shape. It searches
    the FineWeb-Edu text table with a text embedding model and the COCO image
    table with a CLIP text encoder.

    The result lists are intentionally kept separate because MiniLM text-vector
    distances and CLIP text-image distances are not calibrated against each
    other. Returning one merged ranking would be misleading.
    """
    from pathlib import Path

    import lancedb
    from sentence_transformers import SentenceTransformer

    volume.reload()

    query = str(item.get("query", "")).strip()
    k = int(item.get("k", 5))

    if not query:
        return {"error": "Please provide a non-empty query."}

    text_db_path = Path(DATA_DIR) / "lancedb_text"
    image_db_path = Path(DATA_DIR) / "lancedb"

    text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    image_model_name = "sentence-transformers/clip-ViT-B-32"

    text_db = lancedb.connect(str(text_db_path))
    image_db = lancedb.connect(str(image_db_path))

    text_table = text_db.open_table("text_documents")
    image_table = image_db.open_table("image_documents")

    # Text search uses the same MiniLM model used to embed FineWeb-Edu records.
    text_model = SentenceTransformer(text_model_name)
    text_query_vector = text_model.encode(
        [query],
        normalize_embeddings=True,
    )[0].astype("float32").tolist()

    text_results = (
        text_table.search(text_query_vector, vector_column_name="text_vector")
        .limit(k)
        .to_pandas()
    )

    text_matches = []

    for _, row in text_results.iterrows():
        text_matches.append(
            {
                "id": row.get("id", ""),
                "text": row.get("text", "")[:700],
                "url": row.get("url", ""),
                "source": row.get("source", ""),
                "token_count": int(row.get("token_count", 0)),
                "distance": float(row.get("_distance", 0.0)),
            }
        )

    # Image search uses CLIP's text encode, then compares the query vector 
    # to stored image vector
    image_model = SentenceTransformer(image_model_name)
    image_query_vector = image_model.encode(
        [query],
        normalize_embeddings=True,
    )[0].astype("float32").tolist()

    image_results = (
        image_table.search(image_query_vector, vector_column_name="image_vector")
        .limit(k)
        .to_pandas()
    )

    image_matches = []

    for _, row in image_results.iterrows():
        image_matches.append(
            {
                "image_id": row.get("image_id", ""),
                "cocoid": int(row.get("cocoid", -1)),
                "filename": row.get("filename", ""),
                "image_path": row.get("image_path", ""),
                "caption": row.get("caption", ""),
                "source": row.get("source", ""),
                "split": row.get("split", ""),
                "distance": float(row.get("_distance", 0.0)),
            }
        )

    return {
        "query": query,
        "k": k,
        "text_matches": text_matches,
        "image_matches": image_matches,
        "note": (
            "Text and image matches are ranked separately because they use "
            "different embedding models and distance scales."
        ),
    }



@app.local_entrypoint()
def main(
    text_limit: int = 500,
    image_limit: int = 100,
    text_batch_size: int = 128,
    image_batch_size: int = 32,
    build_text: bool = True,
    build_images: bool = True,
) -> None:
    """Run a remote Modal health checks from the local CLI.
    Run configurable Modal batch jobs from the CLI.

    Args:
        text_limit: Number of FineWeb-Edu records to stream for text embeddings.
        image_limit: Number of COCO image-caption records to stream for image embeddings.
        text_batch_size: Ray batch size for text embedding.
        image_batch_size: Ray batch size for image embedding.
        build_text: Whether to rebuild the text LanceDB table.
        build_images: Whether to rebuild the image LanceDB table.
    """
    import time 
    health = health_check.remote()
    print("Health check results:", health)

    gpu = gpu_smoke_test.remote()
    print("GPU smoke test results:", gpu)

    start = time.time()
    print(f"{start}")
    """Run the Modal GPU image batch job."""
    if build_images:
        image_result = build_image_table.remote(
            limit=image_limit,
            batch_size=image_batch_size,
        )
        print("Image table result:", image_result)

    image_end = time.time()

    if build_text:
        text_result = build_text_table.remote(
            limit=text_limit,
            batch_size=text_batch_size,
        )
        print("Text table result:", text_result)

    text_end = time.time()
    print(f"{start}, {image_end}, {text_end}")