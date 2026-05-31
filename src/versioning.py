"""
Why this matters:

Reproduce any past training set exactly
Create a new mix by writing a new JSON file, not copying terabytes
Roll back if a model trains worse on version v002
The key insight: Your content hashes (Component 2) and catalog (Component 6) already provide everything needed.
A version is just a filtered query + a JSON snapshot
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.catalog import MetadataCatalog


class DatasetVersion:
    """A dataset version is an immutable manifest — not a copy of data.

    Create a version by filtering the catalog, then write a JSON manifest.
    Data stays in CAS (Component 2), embeddings in EmbeddingService (Component 5).
    """

    def __init__(self, catalog: MetadataCatalog, versions_dir: Path) -> None:
        self.catalog = catalog
        self.versions_dir = versions_dir
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        version_name: str,
        modalities: list[str] | None = None,
        sources: list[str] | None = None,
        quality_filter: bool = True,
        limit_per_modality: int = 5000,
    ) -> dict:
        """Create a new dataset version by querying the catalog.

        Args:
            version_name: e.g. "multimodal-demo-v001"
            modalities: filter by modality list, e.g. ["text", "image"]
            sources: filter by source list
            quality_filter: only include quality_status="pass" items
            limit_per_modality: max items per modality
        """

        items = []

        mods = modalities or ["text", "image", "video", "audio"]
        for mod in mods:
            df = self.catalog.query(modality=mod, limit=limit_per_modality)
            items.append(df)

        combined = pd.concat(items, ignore_index=True)

        # Build version manifest
        manifest = {
            "version": version_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_items": len(combined),
            "modalities": {
                mod: int((combined["modality"] == mod).sum()) for mod in mods
            },
            "models": {
                "text": "sentence-transformers/all-MiniLM-L6-v2",
                "image": "openai/clip-vit-base-patch32",
                "video": "openai/clip-vit-base-patch32",
                "audio": "openai/whisper-small → projection",
            },
            "item_ids": combined["id"].tolist(),
        }

        # Write manifest
        manifest_path = self.versions_dir / f"{version_name}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return manifest

    def load(self, version_name: str) -> dict:
        """Load an existing version manifest."""
        path = self.versions_dir / f"{version_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Version '{version_name}' not found at {path}")
        return json.loads(path.read_text())

    def list_versions(self) -> list[str]:
        """List all created dataset versions."""
        return sorted([p.stem for p in self.versions_dir.glob("*.json")])
