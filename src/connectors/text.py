# src/connectors/text.py

from itertools import islice
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

from src.cas import ContentAddressedStore
from src.connectors.base import SourceConnector


class FineWebEduConnector(SourceConnector):

    def __init__(
        self,
        output_dir: Path,
        cas: ContentAddressedStore | None = None,
    ) -> None:
        super().__init__("fineweb_edu", output_dir, cas=cas)

    def connect(self, **kwargs) -> Iterator:
        return load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

    def transform(self, raw_row: dict, idx: int) -> dict:
        text = " ".join(str(raw_row.get("text", "")).split())[:2_000]
        if len(text.split()) < 30:
            return None

        return {
            "id": f"fineweb_{idx:08d}",
            "source": "HuggingFaceFW/fineweb-edu/sample-10BT",
            "modality": "text",
            "content_hash": self.hash_content(text),
            "payload": {
                "type": "text",
                "content": text,
                "caption": text[:200],
                "metadata": {
                    "url": raw_row.get("url", ""),
                    "token_count": int(raw_row.get("token_count") or 0),
                },
            },
        }
