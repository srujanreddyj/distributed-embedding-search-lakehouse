# src/connectors/image.py

import json
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

from src.cas import ContentAddressedStore
from src.connectors.base import SourceConnector


class COCOCaptionsConnector(SourceConnector):

    def __init__(
        self,
        output_dir: Path,
        image_dir: Path,
        cas: ContentAddressedStore | None = None,
    ) -> None:
        super().__init__("coco_captions", output_dir, cas=cas)
        self.image_dir = image_dir
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def connect(self, **kwargs) -> Iterator:
        return load_dataset(
            "Multimodal-Fatima/COCO_captions_train",
            split="train",
            streaming=True,
        )

    def transform(self, raw_row: dict, idx: int) -> dict:
        captions = [
            str(c).strip() for c in raw_row.get("sentences_raw", []) if str(c).strip()
        ]
        if not captions:
            return None

        caption = captions[0]
        cocoid = int(raw_row.get("cocoid") or idx)
        image_id = f"coco_{cocoid:012d}"
        image_path = self.image_dir / f"{image_id}.jpg"

        raw_row["image"].convert("RGB").save(image_path, format="JPEG", quality=90)

        # content_hash combines image_id + caption — pure image hash would require reading pixel bytes which is expensive at this stage

        return {
            "id": image_id,
            "source": "Multimodal-Fatima/COCO_captions_train",
            "modality": "image",
            "content_hash": self.hash_content(image_id + caption),
            "payload": {
                "type": "image",
                "content": str(image_path),
                "caption": caption,
                "metadata": {
                    "cocoid": cocoid,
                    "filename": raw_row.get("filename", ""),
                    "split": raw_row.get("split", "train"),
                    "all_captions": json.dumps(captions),
                },
            },
        }
