"""
our version manifest says which items go into a training run.
But training frameworks can't efficiently read 50,000 individual files from a CAS store — random reads starve the GPU.
They need sequential shards: a small number of large files (e.g., 500 tar files) that can be streamed in order.

Why WebDataset:

Each shard is a .tar file — simple, inspectable with tar -tf
Each item is two files inside the tar: {id}.jpg + {id}.json with metadata + embedding
The training loader opens one tar and reads items sequentially — no random seeks
"""

import json
import tarfile
import io
from pathlib import Path
from typing import Any
from typing import Iterator
import pandas as pd

from src.cas import ContentAddressedStore


class ShardWriter:
    """Materialize a dataset version into WebDataset tar shards"""

    def __init__(
        self, cas: ContentAddressedStore, output_dir: Path, shard_size: int = 1000
    ) -> None:
        self.cas = cas
        self.output_dir = output_dir
        self.shard_size = shard_size
        # self.records_df = records_df

    def materialize(
        self, version_manifest: dict, records_df: pd.DataFrame
    ) -> list[dict]:
        """Write shards using a passed-in DataFrame of catalog records."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Index for fast lookup by id
        df_indexed = records_df.set_index("id")
        shard_manifests = []

        for shard_idx, item_ids in enumerate(
            self._batches(version_manifest["item_ids"], self.shard_size)
        ):
            shard_path = self.output_dir / f"shard-{shard_idx:06d}.tar"
            shard_items = []

            with tarfile.open(shard_path, "w") as tar:
                for item_id in item_ids:
                    # Query catalog for this item's details
                    # (simplified — in practice you'd pass loaded records)
                    # record = self._get_record(item_id)
                    try:
                        record = df_indexed.loc[item_id]
                    except KeyError:
                        continue

                    record = self._normalize_record(record)
                    if record is None:
                        continue

                    # Write binary asset files only for file-backed modalities.
                    # Text rows store content inline, so content_path is not a path.
                    asset_name = self._add_asset(tar, item_id, record)

                    # Write metadata + embedding as JSON
                    meta = {
                        "id": item_id,
                        "source": record["source"],
                        "modality": record["modality"],
                        "caption": record["caption"],
                        "quality_status": record.get("quality_status", "unknown"),
                        "license": record.get("license", "unknown"),
                        "metadata_json": record.get("metadata_json", "{}"),
                        "asset": asset_name,
                        "text": self._text_for_record(record),
                        "embedding": self._embedding_for_record(record),
                    }
                    meta_bytes = json.dumps(meta).encode()
                    info = tarfile.TarInfo(name=f"{item_id}.json")
                    info.size = len(meta_bytes)
                    tar.addfile(info, io.BytesIO(meta_bytes))

                    shard_items.append(item_id)

            manifest = {
                "shard": shard_path.name,
                "item_count": len(shard_items),
            }
            shard_manifests.append(manifest)

        return shard_manifests

    def _batches(self, items: list, size: int) -> Iterator[list]:
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _normalize_record(self, record: Any) -> dict | None:
        if record is None:
            return None
        if isinstance(record, pd.DataFrame):
            if record.empty:
                return None
            record = record.iloc[0]
        if isinstance(record, pd.Series):
            return record.to_dict()
        if isinstance(record, dict):
            return record
        return None

    def _embedding_for_record(self, record: dict) -> list[float]:
        if record.get("modality") in {"text", "audio"}:
            value = record.get("text_vector", [])
        else:
            value = record.get("clip_vector", [])
        if hasattr(value, "tolist"):
            return value.tolist()
        return value if isinstance(value, list) else []

    def _text_for_record(self, record: dict) -> str:
        if record.get("modality") != "text":
            return ""
        return str(record.get("content_path") or record.get("caption") or "")

    def _add_asset(self, tar: tarfile.TarFile, item_id: str, record: dict) -> str:
        if record.get("modality") == "text":
            return ""

        content_path = record.get("content_path")
        if not content_path:
            return ""

        try:
            asset_path = Path(str(content_path))
            if asset_path.exists() and asset_path.is_file():
                asset_name = f"{item_id}{asset_path.suffix}"
                tar.add(asset_path, arcname=asset_name)
                return asset_name
        except OSError:
            return ""

        return ""

    def _get_record(self, item_id: str) -> dict | None:
        """Stub — in practice queries the catalog."""
        return None
