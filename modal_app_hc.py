"""Modal app for the multimodal lakehouse search demo.

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

image = modal.Image.debian_slim(python_version="3.11").pip_install(
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
    gpu="L4",
    cpu=2,
    memory=8192,
    timeout=600,
    volumes={DATA_DIR: volume},
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

    vectors = model.encode(
        prompts, batch_size=3, normalize_embeddings=True, show_progress_bar=True
    )

    return {
        "cuda_available": cuda_available,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else "CPU",
        "prompt_count": len(prompts),
        "embedding_shape": list(vectors.shape),
    }


@app.local_entrypoint()
def main() -> None:
    """Run a remote Modal health checks from the local CLI."""

    health = health_check.remote()
    print("Health check results:", health)

    gpu = gpu_smoke_test.remote()
    print("GPU smoke test results:", gpu)
