# modal_pipeline.py
"""Modal orchestration for the 12-stage training-data pipeline.

Each stage runs sequentially — connectors must finish before preprocessing,
preprocessing before quality/dedup, etc.
"""

from pathlib import Path
import modal

APP_NAME = "multimodal-lakehouse-pipeline"
VOLUME_NAME = "multimodal-lakehouse-volume"
DATA_DIR = "/data"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "ray[data]>=2.40.0",
        "datasets>=2.20.0",
        "sentence-transformers>=3.0.0",
        "torch",
        "transformers",
        "lancedb>=0.17.0",
        "pyarrow>=15.0.0",
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "faiss-cpu",
        "torchcodec",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "Pillow>=10.0.0",
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME, image=image)


def _setup():
    """Shared setup for all pipeline stages."""
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    for d in [
        "cas",
        "manifests",
        "preprocessed",
        "filtered",
        "catalog",
        "dataset_versions",
        "shards",
        "metrics",
    ]:
        (Path(DATA_DIR) / d).mkdir(parents=True, exist_ok=True)


@app.function(
    cpu=4,
    memory=16_384,
    timeout=3600,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
def stage_01_connectors(
    text_limit: int = 500,
    image_limit: int = 100,
    video_limit: int = 50,
    audio_limit: int = 100,
) -> dict:
    """Stage 1-2: Source connectors + CAS ingestion."""
    _setup()
    from src.cas import ContentAddressedStore
    from src.connectors.text import FineWebEduConnector
    from src.connectors.image import COCOCaptionsConnector
    from src.connectors.video import FineVideoConnector
    from src.connectors.audio import LibriSpeechConnector
    import pandas as pd

    data = Path(DATA_DIR)
    cas = ContentAddressedStore(root=data / "cas")
    results = {}

    print("--- Text ---")
    text_manifest = FineWebEduConnector(output_dir=data / "manifests", cas=None).run(
        limit=text_limit
    )
    results["text"] = len(pd.read_parquet(text_manifest, columns=["id"]))

    print("--- Image ---")
    image_manifest = COCOCaptionsConnector(
        output_dir=data / "manifests",
        image_dir=data / "raw" / "images",
        cas=cas,
    ).run(limit=image_limit)
    results["image"] = len(pd.read_parquet(image_manifest, columns=["id"]))

    print("--- Video ---")
    video_manifest = FineVideoConnector(
        output_dir=data / "manifests",
        video_dir=data / "raw" / "videos",
        clip_dir=data / "raw" / "video_clips",
        keyframe_dir=data / "raw" / "keyframes",
        cas=cas,
    ).run(limit=video_limit)
    results["video"] = len(pd.read_parquet(video_manifest, columns=["id"]))

    print("--- Audio ---")
    audio_manifest = LibriSpeechConnector(
        output_dir=data / "manifests",
        audio_dir=data / "raw" / "audio",
        cas=cas,
    ).run(limit=audio_limit)
    results["audio"] = len(pd.read_parquet(audio_manifest, columns=["id"]))

    volume.commit()
    return {"stage": "connectors", "status": "ok", "results": results}


@app.function(
    gpu="L4:2",
    cpu=8,
    memory=49_152,
    timeout=3600,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
def stage_02_preprocessing() -> dict:
    """Stage 3: Ray Data preprocessing with stateful actors."""
    _setup()
    import time
    import pandas as pd
    import ray, ray.data
    from src.preprocessing.text import TextPreprocessor
    from src.preprocessing.image import ImagePreprocessor
    from src.preprocessing.video import VideoPreprocessor
    from src.preprocessing.audio import AudioPreprocessor

    data = Path(DATA_DIR)
    output_dir = data / "preprocessed"
    results = {}

    pipelines = [
        {
            "modality": "text",
            "manifest_name": "fineweb_edu",
            "preprocessor_class": TextPreprocessor,
            "batch_size": 128,
            "actor_count": 2,
            "num_gpus": 1.0,
            "memory": 4 * 1024**3,
        },
        {
            "modality": "image",
            "manifest_name": "coco_captions",
            "preprocessor_class": ImagePreprocessor,
            "batch_size": 64,
            "actor_count": 2,
            "num_gpus": 1.0,
            "memory": 6 * 1024**3,
        },
        {
            "modality": "video",
            "manifest_name": "finevideo",
            "preprocessor_class": VideoPreprocessor,
            "batch_size": 8,
            "actor_count": 2,
            "num_gpus": 1.0,
            "memory": 6 * 1024**3,
        },
        {
            "modality": "audio",
            "manifest_name": "librispeech",
            "preprocessor_class": AudioPreprocessor,
            "batch_size": 16,
            "actor_count": 2,
            "num_gpus": 1.0,
            "memory": 6 * 1024**3,
        },
    ]

    for config in pipelines:
        modality = config["modality"]
        manifest_name = config["manifest_name"]
        manifest = data / "manifests" / f"{manifest_name}_manifest.parquet"
        if not manifest.exists():
            print(f"Skipping {modality} — no manifest")
            continue
        row_count = len(pd.read_parquet(manifest, columns=["id"]))
        if row_count == 0:
            print(f"Skipping {modality} — manifest is empty")
            results[modality] = {"status": "skipped", "reason": "empty manifest"}
            continue

        print(f"\n--- {modality.upper()} ---")
        print(
            "Ray config:",
            {
                "batch_size": config["batch_size"],
                "actor_count": config["actor_count"],
                "num_gpus_per_actor": config["num_gpus"],
                "memory_per_actor_gib": round(config["memory"] / 1024**3, 2),
            },
        )
        ray.init(ignore_reinit_error=True)
        start = time.time()

        ds = ray.data.read_parquet(str(manifest))
        embedded = ds.map_batches(
            config["preprocessor_class"],
            batch_size=config["batch_size"],
            compute=ray.data.ActorPoolStrategy(size=config["actor_count"]),
            num_gpus=config["num_gpus"],
            memory=config["memory"],
        )
        embedded.write_parquet(str(output_dir / f"{modality}_embedded.parquet"))

        elapsed = time.time() - start
        results[modality] = {"seconds": round(elapsed, 2)}
        ray.shutdown()

    volume.commit()
    return {"stage": "preprocessing", "status": "ok", "results": results}


@app.function(
    cpu=4,
    memory=16_384,
    timeout=1800,
    volumes={DATA_DIR: volume},
)
def stage_03_quality_dedup() -> dict:
    """Stage 4: Quality gates + ANN dedup."""
    _setup()
    from src.quality import QualityGate
    from src.dedup import EmbeddingDeduplicator
    import pandas as pd

    data = Path(DATA_DIR)
    results = {}
    modality_names = {
        "fineweb_edu": "text",
        "coco_captions": "image",
        "finevideo": "video",
        "librispeech": "audio",
    }

    for manifest_name in ["coco_captions", "finevideo", "fineweb_edu", "librispeech"]:
        manifest = data / "manifests" / f"{manifest_name}_manifest.parquet"
        if not manifest.exists():
            continue
        manifest_name = manifest.stem.replace("_manifest", "")
        modality = modality_names.get(manifest_name, manifest_name)
        print(f"\n--- {manifest_name.upper()} ---")

        df = pd.read_parquet(manifest)
        gate = QualityGate()
        df = gate.run(df)
        report = gate.report(df)

        passed = df[df["quality_status"] == "pass"]
        near_dups = 0

        embedded_path = data / "preprocessed" / f"{modality}_embedded.parquet"
        if embedded_path.exists() and len(passed) > 0:
            embedded_df = pd.read_parquet(embedded_path)
            if "id" in embedded_df.columns and "embedding" in embedded_df.columns:
                embedded_df = embedded_df[embedded_df["id"].isin(passed["id"])]
                embeddings = embedded_df["embedding"].tolist()
                if embeddings:
                    first_embedding = embeddings[0]
                    if hasattr(first_embedding, "tolist"):
                        first_embedding = first_embedding.tolist()
                    deduper = EmbeddingDeduplicator(dim=len(first_embedding))
                    keep_ids = deduper.deduplicate(
                        ids=embedded_df["id"].tolist(),
                        embeddings=[
                            e.tolist() if hasattr(e, "tolist") else e
                            for e in embeddings
                        ],
                    )
                    embedded_ids = set(embedded_df["id"].tolist())
                    duplicate_ids = embedded_ids - set(keep_ids)
                    near_dups = len(duplicate_ids)
                    passed = passed[~passed["id"].isin(duplicate_ids)]

        out_dir = data / "filtered"
        out_dir.mkdir(parents=True, exist_ok=True)
        passed.to_parquet(out_dir / f"{manifest_name}_filtered.parquet", index=False)

        results[manifest_name] = {
            **report,
            "near_duplicates_removed": near_dups,
            "rows_final": len(passed),
        }

    volume.commit()
    return {"stage": "quality_dedup", "status": "ok", "results": results}


@app.function(
    cpu=2,
    memory=8192,
    timeout=900,
    volumes={DATA_DIR: volume},
)
def stage_04_catalog() -> dict:
    """Stage 5-6: Build unified LanceDB catalog with modality vectors."""
    _setup()
    from src.catalog import MetadataCatalog
    import shutil

    data = Path(DATA_DIR)
    current_manifest_names = {
        "fineweb_edu",
        "coco_captions",
        "finevideo",
        "librispeech",
    }
    manifest_paths = [
        path
        for path in sorted((data / "filtered").glob("*_filtered.parquet"))
        if path.stem.replace("_filtered", "") in current_manifest_names
    ]

    # LanceDB/Lance uses atomic renames while writing. Modal Volumes can reject
    # those renames, so build locally first and copy the completed catalog in.
    local_catalog_path = Path("/tmp/catalog_build")
    if local_catalog_path.exists():
        shutil.rmtree(local_catalog_path)
    local_catalog_path.mkdir(parents=True, exist_ok=True)

    catalog = MetadataCatalog(local_catalog_path)

    count = catalog.replace_from_manifests(
        manifest_paths=manifest_paths,
        preprocessed_dir=data / "preprocessed",
    )

    volume_catalog_path = data / "catalog"
    if volume_catalog_path.exists():
        shutil.rmtree(volume_catalog_path)
    shutil.copytree(local_catalog_path, volume_catalog_path)

    print(f"Rebuilt item_catalog with {count} quality-passed rows")

    volume.commit()
    return {
        "stage": "catalog",
        "status": "ok",
        "rows": count,
        "manifests": [path.name for path in manifest_paths],
        "vector_columns": {
            "text_vector": "text/audio embedding space",
            "clip_vector": "image/video embedding space",
        },
    }


@app.function(
    cpu=2,
    memory=8192,
    timeout=600,
    volumes={DATA_DIR: volume},
)
def stage_05_versioning(version_name: str = "multimodal-demo-v001") -> dict:
    """Stage 7: Create dataset version manifest."""
    _setup()
    from src.catalog import MetadataCatalog
    from src.versioning import DatasetVersion

    data = Path(DATA_DIR)
    catalog = MetadataCatalog(data / "catalog")
    versions = DatasetVersion(catalog, data / "dataset_versions")

    manifest = versions.create(
        version_name=version_name,
        modalities=["text", "image", "video", "audio"],
        limit_per_modality=5000,
    )

    volume.commit()
    return {"stage": "versioning", "status": "ok", "version": manifest}


@app.function(
    cpu=2,
    memory=8192,
    timeout=1200,
    volumes={DATA_DIR: volume},
)
def stage_06_sharding(version_name: str = "multimodal-demo-v001") -> dict:
    """Stage 8: Materialize WebDataset tar shards."""
    _setup()
    import pandas as pd
    from src.cas import ContentAddressedStore
    from src.sharding import ShardWriter
    from src.versioning import DatasetVersion
    from src.catalog import MetadataCatalog

    data = Path(DATA_DIR)
    catalog = MetadataCatalog(data / "catalog")
    cas = ContentAddressedStore(data / "cas")

    versions = DatasetVersion(catalog, data / "dataset_versions")
    version = versions.load(version_name)

    all_records = catalog.query(limit=100_000)
    records_df = all_records[all_records["id"].isin(version["item_ids"])]

    writer = ShardWriter(cas=cas, output_dir=data / "shards" / version_name)
    shards = writer.materialize(version, records_df=records_df)

    volume.commit()
    return {"stage": "sharding", "status": "ok", "shards": len(shards)}


@app.function(
    cpu=2,
    memory=8192,
    timeout=600,
    volumes={DATA_DIR: volume},
)
def stage_07_benchmark(version_name: str = "multimodal-demo-v001") -> dict:
    """Stage 9: Loader benchmark."""
    _setup()
    from src.loader_benchmark import ShardLoaderBenchmark

    data = Path(DATA_DIR)
    shard_dir = data / "shards" / version_name

    if not shard_dir.exists():
        return {"stage": "benchmark", "status": "skipped", "reason": "no shards"}

    benchmark = ShardLoaderBenchmark()
    metrics = benchmark.benchmark(shard_dir)

    volume.commit()
    return {"stage": "benchmark", "status": "ok", "metrics": metrics}


@app.local_entrypoint()
def main(
    text_limit: int = 500,
    image_limit: int = 100,
    video_limit: int = 50,
    audio_limit: int = 100,
    version_name: str = "multimodal-demo-v001",
) -> None:
    """Run all pipeline stages sequentially."""

    # Stage 1-2: Connectors + CAS
    result = stage_01_connectors.remote(
        text_limit=text_limit,
        image_limit=image_limit,
        video_limit=video_limit,
        audio_limit=audio_limit,
    )
    print("Stage 1-2:", result)

    # Stage 3: Preprocessing (GPU)
    result = stage_02_preprocessing.remote()
    print("Stage 3:", result)

    # Stage 4: Quality + dedup
    result = stage_03_quality_dedup.remote()
    print("Stage 4:", result)

    # Stage 5-6: Catalog
    result = stage_04_catalog.remote()
    print("Stage 5-6:", result)

    # Stage 7: Versioning
    result = stage_05_versioning.remote(version_name=version_name)
    print("Stage 7:", result)

    # Stage 8: Sharding
    result = stage_06_sharding.remote(version_name=version_name)
    print("Stage 8:", result)

    # Stage 9: Benchmark
    result = stage_07_benchmark.remote(version_name=version_name)
    print("Stage 9:", result)

    print("Pipeline complete.")
