# src/connectors/base.py

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.cas import ContentAddressedStore


class SourceConnector(ABC):
    """Base class for all source connectors.

    Each connector handles one data source. The only method subclasses
    must implement is `transform()` — everything else is shared.
    """

    def __init__(
        self,
        source_name: str,
        output_dir: Path,
        cas: ContentAddressedStore | None = None,
    ) -> None:
        self.source_name = source_name
        self.output_dir = output_dir
        self.cas = cas
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def connect(self, **kwargs) -> Iterator:
        """Stream raw rows from the source."""
        ...

    @abstractmethod
    def transform(self, raw_row: dict, idx: int) -> dict | list[dict] | None:
        """Convert one raw source row into the uniform schema."""
        ...

    def hash_content(self, content: str) -> str:
        """SHA256 hash of content string — seeds the CAS store later."""
        return hashlib.sha256(content.encode()).hexdigest()

    def write_manifest(self, records: list[dict], filename: str) -> Path:
        """Write records to parquet manifest."""
        df = pd.DataFrame(records)
        path = self.output_dir / filename
        df.to_parquet(path, index=False)
        print(f"Wrote {len(df)} records to {path}")
        return path

    def run(self, limit: int = 1000, max_scan: int | None = None, **kwargs) -> Path:
        """Connect, transform, and write manifest. Main entry point."""
        if limit <= 0:
            return self.write_manifest([], f"{self.source_name}_manifest.parquet")

        max_scan = max_scan or max(limit * 20, limit)
        records = []
        for idx, raw_row in enumerate(self.connect(**kwargs)):
            if idx >= max_scan or len(records) >= limit:
                break
            transformed = self.transform(raw_row, idx)
            if not transformed:
                continue
            new_records = (
                transformed if isinstance(transformed, list) else [transformed]
            )
            for record in new_records:
                if len(records) >= limit:
                    break
                # move raw asset into CAS if store is provided
                if self.cas and record["payload"]["content"]:
                    src = Path(record["payload"]["content"])
                    if src.exists():
                        cas_path = self.cas.put(src, record["content_hash"])
                        record["payload"]["content"] = str(cas_path)
                records.append(record)
        if len(records) < limit:
            print(
                f"Warning: requested {limit} records from {self.source_name}, "
                f"accepted {len(records)} after scanning up to {max_scan} rows"
            )
        return self.write_manifest(records, f"{self.source_name}_manifest.parquet")
