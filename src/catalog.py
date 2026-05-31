"""
One query catalog for all 4 modalities

Right now our manifests and preprocessed data live in separate parquet files per modality.
A catalog unifies them into one LanceDB table with a common schema and modality-specific metadata columns.

Easy to query across modalities --> "find all english text + image pairs from COCO with quality = pass"
Semantic search queries one catalog instead of 4 separate tables
Schema evolution too
"""

from pathlib import Path
from typing import Any

import lancedb
import pandas as pd

TEXT_VECTOR_DIM = 384
CLIP_VECTOR_DIM = 512
SOURCE_LICENSES = {
    "fineweb-edu": "mit",
    "coco": "cc-by-4.0",
    "finevideo": "cc-by-4.0",
    "huggingfacefv": "cc-by-4.0",
    "msr-vtt": "research-only",
    "msrvtt": "research-only",
    "librispeech": "cc-by-4.0",
    "openslr": "cc-by-4.0",
}


def _zero_vector(dim: int) -> list[float]:
    return [0.0] * dim


def _payload_get(payload: Any, key: str, default: Any = "") -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return default


def _normalize_vector(value: Any, dim: int) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == dim:
        return [float(v) for v in value]
    return _zero_vector(dim)


def _has_vector(value: Any) -> bool:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return isinstance(value, list) and len(value) > 0


def _license_for_source(source: str) -> str:
    normalized = str(source).lower()
    for source_key, license_name in SOURCE_LICENSES.items():
        if source_key in normalized:
            return license_name
    return "unknown"


class MetadataCatalog:
    """Unified queryable catalog for all modalities.

    Stores: id, source, modality, content_hash, quality_status,
    embedding pointers, and modality-specific metadata as JSON.
    """

    def __init__(self, db_path: Path) -> None:
        self.db = lancedb.connect(str(db_path))
        self.table_name = "item_catalog"

    def build_records(
        self,
        manifest_path: Path,
        embedded_path: Path | None = None,
    ) -> list[dict]:
        """Flatten a manifest and attach precomputed embeddings when available."""
        df = pd.read_parquet(manifest_path)

        embedded_by_id = {}
        if embedded_path and embedded_path.exists():
            embedded_df = pd.read_parquet(embedded_path)
            if "id" in embedded_df.columns and "embedding" in embedded_df.columns:
                embedded_by_id = embedded_df.set_index("id")["embedding"].to_dict()

        records = []
        for _, row in df.iterrows():
            payload = row.get("payload", {})
            metadata = _payload_get(payload, "metadata", {})
            modality = row["modality"]
            embedding = embedded_by_id.get(row["id"], [])

            text_vector = _zero_vector(TEXT_VECTOR_DIM)
            clip_vector = _zero_vector(CLIP_VECTOR_DIM)

            if modality in {"text", "audio"}:
                text_vector = _normalize_vector(embedding, TEXT_VECTOR_DIM)
            elif modality in {"image", "video"}:
                clip_vector = _normalize_vector(embedding, CLIP_VECTOR_DIM)

            records.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "modality": modality,
                    "content_hash": row["content_hash"],
                    "quality_status": row.get("quality_status", "unknown"),
                    "license": row.get("license", _license_for_source(row["source"])),
                    "caption": _payload_get(payload, "caption", ""),
                    "content_path": _payload_get(payload, "content", ""),
                    "metadata_json": str(metadata) if metadata else "{}",
                    "text_vector": text_vector,
                    "clip_vector": clip_vector,
                    "has_embedding": _has_vector(embedding),
                }
            )

        return records

    def replace_from_manifests(
        self,
        manifest_paths: list[Path],
        preprocessed_dir: Path,
    ) -> int:
        """Rebuild the unified catalog from manifests plus preprocessed vectors."""
        all_records = []
        embedded_names = {
            "fineweb_edu": "text",
            "coco_captions": "image",
            "finevideo": "video",
            "msrvtt": "video",
            "librispeech": "audio",
        }

        for manifest_path in manifest_paths:
            manifest_name = manifest_path.stem.replace("_manifest", "").replace(
                "_filtered", ""
            )
            modality = embedded_names.get(manifest_name, manifest_name)
            embedded_path = preprocessed_dir / f"{modality}_embedded.parquet"
            all_records.extend(self.build_records(manifest_path, embedded_path))

        if not all_records:
            try:
                self.db.drop_table(self.table_name)
            except Exception:
                pass
            return 0

        self.db.create_table(self.table_name, all_records, mode="overwrite")
        return len(all_records)

    def ingest_manifest(self, manifest_path: Path) -> int:
        """Add all items from a manifest parquet into the catalog."""
        ingest_df = pd.DataFrame(self.build_records(manifest_path))

        try:
            table = self.db.open_table(self.table_name)
            table.add(ingest_df.to_dict("records"))
        except Exception:
            table = self.db.create_table(self.table_name, ingest_df.to_dict("records"))

        return len(ingest_df)

    def query(
        self, modality: str | None = None, source: str | None = None, limit: int = 100
    ) -> pd.DataFrame:
        """Search catalog by modality and/or source."""
        table = self.db.open_table(self.table_name)
        filters = []
        if modality:
            filters.append(f"modality = '{modality}'")
        if source:
            filters.append(f"source = '{source}'")
        where_clause = " AND ".join(filters) if filters else None
        query = table.search()
        if where_clause:
            query = query.where(where_clause)
        return query.limit(limit).to_pandas()
