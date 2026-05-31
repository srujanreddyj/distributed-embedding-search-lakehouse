import json
import time
from pathlib import Path

import ray
import ray.data

from src.preprocessing.text import TextPreprocessor
from src.preprocessing.image import ImagePreprocessor
from src.preprocessing.video import VideoPreprocessor
from src.preprocessing.audio import AudioPreprocessor

DATA_DIR = Path("data")
MANIFEST_DIR = DATA_DIR / "manifests"
OUTPUT_DIR = DATA_DIR / "preprocessed"


def run_ray_pipeline(
    manifest_path: Path,
    output_path: Path,
    preprocessor_class: type,
    batch_size: int = 16,
    num_gpus: float = 0,
    actor_count: int = 1,
) -> dict:
    """Run a Ray Data pipeline with a stateful preprocessor actor."""

    ray.init(ignore_reinit_error=True)

    start = time.time()

    ds = ray.data.read_parquet(str(manifest_path))

    embedded = ds.map_batches(
        preprocessor_class,
        batch_size=batch_size,
        compute=ray.data.ActorPoolStrategy(size=actor_count),
        num_gpus=num_gpus,
    )

    embedded.write_parquet(str(output_path))

    elapsed = time.time() - start
    row_count = embedded.count()

    ray.shutdown()

    metrics = {
        "manifest": str(manifest_path),
        "output": str(output_path),
        "rows": row_count,
        "seconds": round(elapsed, 2),
        "rows_per_second": round(row_count / elapsed, 2) if elapsed > 0 else 0,
        "batch_size": batch_size,
        "actor_count": actor_count,
        "num_gpus": num_gpus,
    }
    return metrics


def main(
    text: bool = True,
    image: bool = True,
    video: bool = True,
    audio: bool = True,
    num_gpus: float = 0,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = DATA_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pipelines = []

    if text:
        pipelines.append(("text", TextPreprocessor, 64))
    if image:
        pipelines.append(("image", ImagePreprocessor, 16))
    if video:
        pipelines.append(("video", VideoPreprocessor, 8))
    if audio:
        pipelines.append(("audio", AudioPreprocessor, 16))

    manifest_names = {
        "text": "fineweb_edu",
        "image": "coco_captions",
        "video": "finevideo",
        "audio": "librispeech",
    }

    for modality, preprocessor_class, batch_size in pipelines:
        manifest = MANIFEST_DIR / f"{manifest_names[modality]}_manifest.parquet"

        if not manifest.exists():
            print(f"Skipping {modality} — manifest not found at {manifest}")
            continue

        print(f"\n--- {modality.upper()} ---")
        metrics = run_ray_pipeline(
            manifest_path=manifest,
            output_path=OUTPUT_DIR / f"{modality}_embedded.parquet",
            preprocessor_class=preprocessor_class,
            batch_size=batch_size,
            num_gpus=num_gpus,
        )

        metrics_path = metrics_dir / f"metrics_preprocess_{modality}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
