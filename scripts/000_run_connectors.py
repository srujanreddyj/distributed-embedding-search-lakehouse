# scripts/00_run_connectors.py

from pathlib import Path

from src.cas import ContentAddressedStore
from src.connectors.text import FineWebEduConnector
from src.connectors.image import COCOCaptionsConnector
from src.connectors.video import FineVideoConnector
from src.connectors.audio import LibriSpeechConnector

DATA_DIR = Path("data")
cas = ContentAddressedStore(root=DATA_DIR / "cas")


def main(
    text_limit: int = 500,
    image_limit: int = 100,
    video_limit: int = 50,
    audio_limit: int = 100,
) -> None:

    print("--- Text ---")
    FineWebEduConnector(
        output_dir=DATA_DIR / "manifests",
        cas=None,  # text has no file asset
    ).run(limit=text_limit)

    print("--- Image ---")
    COCOCaptionsConnector(
        output_dir=DATA_DIR / "manifests",
        image_dir=DATA_DIR / "raw" / "images",
        cas=cas,
    ).run(limit=image_limit)

    print("--- Video ---")
    FineVideoConnector(
        output_dir=DATA_DIR / "manifests",
        video_dir=DATA_DIR / "raw" / "videos",
        clip_dir=DATA_DIR / "raw" / "video_clips",
        keyframe_dir=DATA_DIR / "raw" / "keyframes",
        cas=cas,
    ).run(limit=video_limit)

    print("--- Audio ---")
    LibriSpeechConnector(
        output_dir=DATA_DIR / "manifests",
        audio_dir=DATA_DIR / "raw" / "audio",
        cas=cas,
    ).run(limit=audio_limit)


if __name__ == "__main__":
    main()
